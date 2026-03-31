import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math

class RelationGuideFusion(nn.Module):

    def __init__(self, dim, num_ent):
        super().__init__()
        self.dim = dim
        self.num_ent = num_ent

        
        self.rel_proj = nn.ModuleList([
            nn.Linear(dim, dim),  # structural
            nn.Linear(dim, dim),  # visual
            nn.Linear(dim, dim),  # textual
            nn.Linear(dim, dim),  # multimodal
        ])

        
        self.modality_anchors = nn.Parameter(
            torch.randn(4, dim) * 0.01
        )

        self.base_weights = nn.Parameter(torch.ones(4) * 0.25)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
    def compute_confidence(self, rel_core, modality_idx):
        """
        rel_core: [batch, dim] 
        """
        
        rel_m = self.rel_proj[modality_idx](rel_core)  # [batch, dim]

        anchor = self.modality_anchors[modality_idx]   # [dim]

        similarity = (rel_m * anchor).sum(dim=-1, keepdim=True)

        confidence = torch.sigmoid(
            similarity / (self.temperature.abs() + 1e-8)
        )

        return confidence

    
    def forward(self, scores, rel_core):
        """
        scores: List[Tensor], 4 × [batch, num_ent]
        rel_core: Tensor [batch, dim]
        """

        confidences = []
        for i in range(4):
            conf = self.compute_confidence(rel_core, i)
            confidences.append(conf)

        confidences = torch.cat(confidences, dim=-1)  # [batch, 4]

        base_w = F.softmax(self.base_weights, dim=0)
        dynamic_weights = confidences * base_w.unsqueeze(0)

        weights = dynamic_weights / (
            dynamic_weights.sum(dim=-1, keepdim=True) + 1e-8
        )

        fused_score = torch.zeros_like(scores[0])
        for i, s in enumerate(scores):
            fused_score += weights[:, i:i+1] * s

        return fused_score

class ConvELayer(nn.Module):
    def __init__(self, dim, out_channels, kernel_size, k_h, k_w):
        super(ConvELayer, self).__init__()

        self.input_drop = nn.Dropout(0.2)
        self.conv_drop = nn.Dropout2d(0.2)
        self.hidden_drop = nn.Dropout(0.2)
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm1d(dim)

        self.conv = torch.nn.Conv2d(1, out_channels=out_channels, kernel_size=(kernel_size, kernel_size),
                                    stride=1, padding=0, bias=True)
        assert k_h * k_w == dim
        flat_sz_h = int(2 * k_w) - kernel_size + 1
        flat_sz_w = k_h - kernel_size + 1
        self.flat_sz = flat_sz_h * flat_sz_w * out_channels
        self.fc = nn.Linear(self.flat_sz, dim, bias=True)

    def forward(self, conv_input):
        x = self.bn0(conv_input)
        x = self.input_drop(x)
        x = self.conv(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x


class ConvKBLayer(nn.Module):
    def __init__(self, input_dim, input_seq_len, in_channels, out_channels, drop_prob, alpha_leaky):
        super(ConvKBLayer, self).__init__()

        self.conv_layer = nn.Conv2d(in_channels, out_channels, (1, input_seq_len))
        self.dropout = nn.Dropout(drop_prob)
        self.non_linearity = nn.ReLU()
        self.fc_layer = nn.Linear(input_dim * out_channels, 256)

        nn.init.xavier_uniform_(self.fc_layer.weight, gain=1.414)
        nn.init.xavier_uniform_(self.conv_layer.weight, gain=1.414)

    def forward(self, conv_input):
        batch_size, length, dim = conv_input.size()
        conv_input = conv_input.transpose(1, 2)
        conv_input = conv_input.unsqueeze(1)
        out_conv = self.dropout(self.non_linearity(self.conv_layer(conv_input)))
        input_fc = out_conv.squeeze(-1).view(batch_size, -1)
        output = self.fc_layer(input_fc)
        return output


class InterpretableModalityGate(nn.Module):

    def __init__(self, feature_dim, lambda_err=1.0, eps=1e-6):
        super().__init__()
        self.feature_dim = feature_dim
        self.lambda_err = lambda_err
        self.eps = eps

        
        self.quality_scale = nn.Parameter(torch.tensor(1.0))
        self.quality_shift = nn.Parameter(torch.tensor(0.0))

        
        self.uniqueness_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, feat, rec_err):
        """
        feat: [B, feature_dim]
        rec_err: [B, 1]
        """
        
        feat_mean = feat.mean(dim=1, keepdim=True)  # [B, 1]
        feat_std = feat.std(dim=1, keepdim=True) + self.eps  

        
        snr = (feat_mean ** 2) / feat_std  # [B, 1]
        quality = torch.sigmoid(self.quality_scale * snr + self.quality_shift)  

        # 2. 
        prob = F.softmax(feat, dim=1)
        entropy = -torch.sum(prob * torch.log(prob + self.eps), dim=1, keepdim=True)  # [B, 1]
        info_score = torch.sigmoid(entropy / math.log(self.feature_dim))  

        # 3. 
        err_score = torch.exp(-self.lambda_err * rec_err)  # [B, 1]

        
        reliability = quality * err_score
        uniqueness = quality * (1 - err_score)

        score = reliability + self.uniqueness_weight * uniqueness * info_score

        return score


class AttentionCrossModalReconstruction(nn.Module):

    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.attn_f1 = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.attn_f2 = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

       
        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, f1, f2):


        # sequence dim: [B, 1, dim]
        f1 = f1.unsqueeze(1)
        f2 = f2.unsqueeze(1)

        
        # key, value = f2, query = f1
        out1, _ = self.attn_f1(query=f1, key=f2, value=f2)  # [B, 1, dim]
        out2, _ = self.attn_f2(query=f2, key=f1, value=f1)  # [B, 1, dim]

        
        fusion_input = torch.cat([out1, out2], dim=-1).squeeze(1)  # [B, dim * 2]

       
        reconstructed = self.fusion(fusion_input)  # [B, dim]

        return reconstructed

class CrossModalReconstructionFusion(nn.Module):
    def __init__(self, in_dim, out_dim):
        
        super(CrossModalReconstructionFusion, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        
        self.modal1_proj = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU()
        )
        self.modal2_proj = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU()
        )
        self.modal3_proj = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU()
        )

        
        self.fuse1 = nn.Sequential(
            nn.Linear(out_dim , out_dim),
            nn.ReLU()
        )
        self.fuse2 = nn.Sequential(
            nn.Linear(out_dim , out_dim),
            nn.ReLU()
        )
        self.fuse3 = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU()
        )

        
        self.rec1 = AttentionCrossModalReconstruction(out_dim)
        self.rec2 = AttentionCrossModalReconstruction(out_dim)
        self.rec3 = AttentionCrossModalReconstruction(out_dim)

        
        self.temperature = nn.Parameter(torch.tensor(1.0))

        
        self.gate1 = InterpretableModalityGate(feature_dim=256)
        self.gate2 = InterpretableModalityGate(feature_dim=256)
        self.gate3 = InterpretableModalityGate(feature_dim=256)

        
        self.fusion_fc = nn.Linear(out_dim, out_dim)
        self.fusion_norm = nn.LayerNorm(out_dim)

    def forward(self, modal1_emb, modal2_emb, modal3_emb):
        """
       
          modal1_emb, modal2_emb, modal3_emb: [B, in_dim]
        
          fused: [B, out_dim]
          weights: [B, 3]
          rec_errors: [B, 3]
          gate_scores: [B, 3]
        """
        
        f1 = self.modal1_proj(modal1_emb)  # [B, out_dim]
        f2 = self.modal2_proj(modal2_emb)  # [B, out_dim]
        f3 = self.modal3_proj(modal3_emb)  # [B, out_dim]


        r1 = self.rec1(f2, f3)
        r2 = self.rec2(f1, f3)
        r3 = self.rec3(f1, f2)


        err1 = torch.norm(f1 - r1, p=2, dim=1, keepdim=True)
        err2 = torch.norm(f2 - r2, p=2, dim=1, keepdim=True)
        err3 = torch.norm(f3 - r3, p=2, dim=1, keepdim=True)
        rec_errors = torch.cat([err1, err2, err3], dim=1)


        score1 = self.gate1(f1, err1)
        score2 = self.gate2(f2, err2)
        score3 = self.gate3(f3, err3)
        gate_scores = torch.cat([score1, score2, score3], dim=1)


        weights = F.softmax(gate_scores / self.temperature, dim=1)


        fused_weighted = weights[:, 0:1] * f1 + weights[:, 1:2] * f2 + weights[:, 2:3] * f3


        fused = self.fusion_fc(fused_weighted) + fused_weighted
        fused = self.fusion_norm(fused)


        return fused

class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, A_hat):
        # x: [B, T, in_features]
        # A_hat: [B, T, T] 
        out = torch.bmm(A_hat, x)  # [B, T, in_features]
        out = torch.matmul(out, self.weight)  # [B, T, out_features]
        return F.relu(out)


class GraphConvolutionalTokenEnhancer(nn.Module):

    def __init__(self, input_dim=256, output_dim=256, hidden_dim=512):
        super(GraphConvolutionalTokenEnhancer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.feature_dim = 4 * input_dim  

        
        self.mlp = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.layer_norm = nn.LayerNorm(output_dim)

        
        self.gcn = GraphConvolution(input_dim, output_dim)

        
        self.alpha = nn.Parameter(torch.tensor(0.5))  

        
        self.ent_attn = nn.Linear(output_dim, 1, bias=False)

    def forward(self, x, mask):
        B, T, D = x.shape
        valid_mask = (~mask).float()  # [B, T]
        x_valid = x * valid_mask.unsqueeze(-1)  # [B, T, D]

        
        zero_mask = (x_valid.abs().sum(dim=-1, keepdim=True) == 0)
        x_valid = x_valid + zero_mask * torch.randn_like(x_valid) * 1e-3

        
        sum_all = torch.sum(x_valid, dim=1, keepdim=True)
        count = valid_mask.sum(dim=1, keepdim=True)
        denom = (count - 1).clamp(min=1).unsqueeze(-1)
        context = (sum_all - x_valid) / denom
        inter_features = torch.cat([x_valid, context, x_valid * context, x_valid - context], dim=-1)
        enhanced_interaction = self.mlp(inter_features)
        enhanced_interaction = self.layer_norm(enhanced_interaction)
        enhanced_interaction = enhanced_interaction.masked_fill(mask.unsqueeze(-1), 0.0)

        
        norm = torch.norm(x_valid, p=2, dim=-1, keepdim=True) + 1e-6
        x_normed = x_valid / norm
        A = torch.bmm(x_normed, x_normed.transpose(1, 2))  # [B, T, T]
        valid_mask_expanded = valid_mask.unsqueeze(1) * valid_mask.unsqueeze(2)
        A = A * valid_mask_expanded
        I = torch.eye(T, device=x.device).unsqueeze(0).expand(B, T, T)
        A_with_I = A + I

        
        D_mat = torch.sum(A_with_I, dim=-1)
        D_mat = torch.clamp(D_mat, min=1e-3)
        D_inv_sqrt = torch.pow(D_mat, -0.5)
        D_inv_sqrt = torch.diag_embed(D_inv_sqrt)
        A_hat = torch.bmm(torch.bmm(D_inv_sqrt, A_with_I), D_inv_sqrt)

        gcn_output = self.gcn(x_valid, A_hat)
        gcn_output = gcn_output.masked_fill(mask.unsqueeze(-1), 0.0)

        
        fused_tokens = self.alpha * enhanced_interaction + (1 - self.alpha) * gcn_output

        
        u = torch.tanh(fused_tokens)
        scores = self.ent_attn(u).squeeze(-1)  # [B, T]
        scores = scores.masked_fill(mask, -1e4)  

        all_masked = mask.all(dim=-1, keepdim=True)
        scores = torch.where(all_masked, torch.zeros_like(scores), scores)

        attn_weights = torch.softmax(scores, dim=-1)
        default_weights = torch.ones_like(attn_weights) / attn_weights.shape[-1]
        condition = all_masked.expand_as(attn_weights)
        attn_weights = torch.where(condition, default_weights, attn_weights)

        entity_embedding = torch.sum(attn_weights.unsqueeze(-1) * fused_tokens, dim=1)  # [B, output_dim]

        return entity_embedding
