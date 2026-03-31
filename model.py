import torch
import torch.nn as nn

from module import *


class STIRF(nn.Module):
    def __init__(
            self,
            num_ent,
            num_rel,
            ent_vis_mask,
            ent_txt_mask,
            dim_str,
            num_head,
            dim_hid,
            num_layer_enc_ent,
            num_layer_enc_rel,
            num_layer_dec,
            dropout=0.1,
            emb_dropout=0.6,
            vis_dropout=0.1,
            txt_dropout=0.1,
            visual_token_index=None,
            text_token_index=None,
            neighbor_token_index=None,
    ):
        super(STIRF, self).__init__()
        self.dim_str = dim_str
        self.num_head = num_head
        self.dim_hid = dim_hid
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.num_vis = ent_vis_mask.shape[1]
        self.num_txt = ent_txt_mask.shape[1]
        self.num_neighbor = neighbor_token_index.shape[1]

        visual_tokens = torch.load("tokens/visual.pth")
        textual_tokens = torch.load("tokens/textual.pth")
        self.visual_token_index = visual_token_index
        self.visual_token_embedding = nn.Embedding.from_pretrained(visual_tokens).requires_grad_(False)
        self.text_token_index = text_token_index
        self.text_token_embedding = nn.Embedding.from_pretrained(textual_tokens).requires_grad_(False)
        self.neighbor_token_index = neighbor_token_index

        self.visual_token_embedding.requires_grad_(False)
        self.text_token_embedding.requires_grad_(False)

        self.select_v_mask = ent_vis_mask
        self.select_t_mask = ent_txt_mask

        self.s_ent_token = nn.Parameter(torch.Tensor(1, 1, dim_str))

        self.ent_embeddings = nn.Parameter(torch.Tensor(num_ent, 1, dim_str))
        self.rel_embeddings_s = nn.Parameter(torch.Tensor(num_rel, dim_str))

        self.lp_token = nn.Parameter(torch.Tensor(1, dim_str))

        self.str_ent_ln = nn.LayerNorm(dim_str)
        self.str_rel_ln = nn.LayerNorm(dim_str)
        self.img_rel_ln = nn.LayerNorm(dim_str)
        self.txt_rel_ln = nn.LayerNorm(dim_str)
        self.vis_ln = nn.LayerNorm(dim_str)
        self.txt_ln = nn.LayerNorm(dim_str)

        self.embdr = nn.Dropout(p=emb_dropout)
        self.visdr = nn.Dropout(p=vis_dropout)
        self.txtdr = nn.Dropout(p=txt_dropout)

        self.pos_vis_ent = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.pos_txt_ent = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.pos_str_ent = nn.Parameter(torch.Tensor(1, 1, dim_str))

        self.s_cls = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.v_cls = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.t_cls = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.mm_cls = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.pos_head_s = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.pos_head_v = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.pos_head_t = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.pos_head_m = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.pos_rel_s = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.pos_rel_v = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.pos_rel_t = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.pos_rel_m = nn.Parameter(torch.Tensor(1, 1, dim_str))

        self.proj_ent_vis = nn.Linear(32, dim_str)
        self.proj_ent_txt = nn.Linear(768, dim_str)
        self.select_v_encoder = GraphConvolutionalTokenEnhancer(dim_str, dim_str)
        self.select_t_encoder = GraphConvolutionalTokenEnhancer(dim_str, dim_str)

        s_ent_encoder_layer = nn.TransformerEncoderLayer(dim_str, num_head, dim_hid, dropout, batch_first=True)
        self.s_ent_encoder = nn.TransformerEncoder(s_ent_encoder_layer, num_layer_enc_ent)

        self.fusion_model = CrossModalReconstructionFusion(
            in_dim=dim_str,
            out_dim=dim_str,
        )
        self.fusion_rel = CrossModalReconstructionFusion(
            in_dim=dim_str,
            out_dim=dim_str,
        )
        self.relationguidefusion = RelationGuideFusion(dim_str, self.num_ent)

        decoder_layer_s = nn.TransformerEncoderLayer(dim_str, num_head, dim_hid, dropout, batch_first=True)
        self.decoder_s = nn.TransformerEncoder(decoder_layer_s, num_layer_dec)
        decoder_layer_i = nn.TransformerEncoderLayer(dim_str, num_head, dim_hid, dropout, batch_first=True)
        self.decoder_i = nn.TransformerEncoder(decoder_layer_i, num_layer_dec)
        decoder_layer_t = nn.TransformerEncoderLayer(dim_str, num_head, dim_hid, dropout, batch_first=True)
        self.decoder_t = nn.TransformerEncoder(decoder_layer_t, num_layer_dec)
        decoder_layer_mm = nn.TransformerEncoderLayer(dim_str, num_head, dim_hid, dropout, batch_first=True)
        self.decoder_mm = nn.TransformerEncoder(decoder_layer_mm, num_layer_dec)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.ent_embeddings)
        nn.init.xavier_uniform_(self.rel_embeddings_s)
        nn.init.xavier_uniform_(self.proj_ent_vis.weight)
        nn.init.xavier_uniform_(self.proj_ent_txt.weight)
        nn.init.xavier_uniform_(self.s_ent_token)
        nn.init.xavier_uniform_(self.lp_token)
        nn.init.xavier_uniform_(self.pos_vis_ent)
        nn.init.xavier_uniform_(self.pos_txt_ent)
        nn.init.xavier_uniform_(self.pos_str_ent)
        nn.init.xavier_uniform_(self.pos_head_s)
        nn.init.xavier_uniform_(self.pos_head_v)
        nn.init.xavier_uniform_(self.pos_head_t)
        nn.init.xavier_uniform_(self.pos_head_m)
        nn.init.xavier_uniform_(self.pos_rel_s)
        nn.init.xavier_uniform_(self.pos_rel_v)
        nn.init.xavier_uniform_(self.pos_rel_t)
        nn.init.xavier_uniform_(self.pos_rel_m)
        nn.init.xavier_uniform_(self.s_cls)
        nn.init.xavier_uniform_(self.v_cls)
        nn.init.xavier_uniform_(self.t_cls)
        nn.init.xavier_uniform_(self.mm_cls)


    def forward(self):
        s_ent_tkn = self.s_ent_token.tile(self.num_ent, 1, 1)
        rep_ent_str = self.embdr(self.str_ent_ln(self.ent_embeddings)) 
        ori_str_tokens = torch.cat([rep_ent_str, rep_ent_str, rep_ent_str, rep_ent_str, rep_ent_str], dim=1) + self.pos_str_ent
        s_ent_seq = torch.cat([s_ent_tkn, ori_str_tokens], dim=1)
        str_embeddings = self.s_ent_encoder(s_ent_seq)[:, 0]

        
        entity_visual_tokens = self.visual_token_embedding(self.visual_token_index)
        ori_visual_tokens = self.visdr(self.vis_ln(self.proj_ent_vis(entity_visual_tokens))) + self.pos_vis_ent
        select_v_emb = self.select_v_encoder(ori_visual_tokens, self.select_v_mask)
        visual_embeddings = select_v_emb  

       
        entity_text_tokens = self.text_token_embedding(self.text_token_index)
        ori_text_tokens = self.txtdr(self.txt_ln(self.proj_ent_txt(entity_text_tokens))) + self.pos_txt_ent
        select_t_emb = self.select_t_encoder(ori_text_tokens, self.select_t_mask)
        text_embeddings = select_t_emb  

        
        ent_s = torch.cat([str_embeddings, self.lp_token], dim=0)
        ent_v = torch.cat([visual_embeddings, self.lp_token], dim=0)
        ent_t = torch.cat([text_embeddings, self.lp_token], dim=0)
        

        rep_rel_s = self.embdr(self.str_rel_ln(self.rel_embeddings_s))
        rep_rel_i = self.visdr(self.img_rel_ln(self.rel_embeddings_s))
        rep_rel_t = self.txtdr(self.txt_rel_ln(self.rel_embeddings_s))
        rep_rel_mm = self.fusion_rel(rep_rel_s, rep_rel_i, rep_rel_t)
        return [ent_s, ent_v, ent_t], [rep_rel_s, rep_rel_i, rep_rel_t, rep_rel_mm]



    def score(self, emb_ent, emb_rel, triplets, labels):

        indexs = triplets != self.num_ent + self.num_rel
        indexs[:, 1] = False
        ids = triplets[indexs] - self.num_rel

        emb_ent_mm = self.fusion_model(emb_ent[0][ids], emb_ent[1][ids], emb_ent[2][ids])
        all_mm = self.fusion_model(emb_ent[0][:-1], emb_ent[1][:-1], emb_ent[2][:-1])


        cls_seq_s = self.s_cls.tile(triplets.size(0), 1, 1)
        h_seq_s = emb_ent[0][ids].unsqueeze(1) + self.pos_head_s
        r_seq_s = emb_rel[0][triplets[:, 1] - self.num_ent].unsqueeze(dim=1) + self.pos_rel_s

        dec_seq_s = torch.cat([cls_seq_s, h_seq_s, r_seq_s], dim=1)
        
        cls_seq_i = self.v_cls.tile(triplets.size(0), 1, 1)
        h_seq_i = emb_ent[1][ids].unsqueeze(1) + self.pos_head_v
        r_seq_i = emb_rel[1][triplets[:, 1] - self.num_ent].unsqueeze(dim=1) + self.pos_rel_v
        
        dec_seq_i = torch.cat([cls_seq_i, h_seq_i, r_seq_i], dim=1)
        
        cls_seq_t = self.t_cls.tile(triplets.size(0), 1, 1)
        h_seq_t = emb_ent[2][ids].unsqueeze(1) + self.pos_head_t
        r_seq_t = emb_rel[2][triplets[:, 1] - self.num_ent].unsqueeze(dim=1) + self.pos_rel_t
        
        dec_seq_t = torch.cat([cls_seq_t, h_seq_t, r_seq_t], dim=1)
        
        cls_seq_mm = self.mm_cls.tile(triplets.size(0), 1, 1)
        h_seq_mm = emb_ent_mm.unsqueeze(1) + self.pos_head_m
        r_seq_mm = emb_rel[3][triplets[:, 1] - self.num_ent].unsqueeze(dim=1) + self.pos_rel_m
        
        dec_seq_mm = torch.cat([cls_seq_mm, h_seq_mm, r_seq_mm], dim=1)

        output_dec_s = self.decoder_s(dec_seq_s)
        output_dec_i = self.decoder_i(dec_seq_i)
        output_dec_t = self.decoder_t(dec_seq_t)
        output_dec_mm = self.decoder_mm(dec_seq_mm)
        
        rel_emb_s = output_dec_s[:, 2, :]  
        ctx_emb_s = output_dec_s[:, 0, :]  
        rel_emb_i = output_dec_i[:, 2, :]  
        ctx_emb_i = output_dec_i[:, 0, :]  
        rel_emb_t = output_dec_t[:, 2, :]  
        ctx_emb_t = output_dec_t[:, 0, :]  
        rel_emb_mm = output_dec_mm[:, 2, :] 
        ctx_emb_mm = output_dec_mm[:, 0, :]  

        score_s = torch.mm(ctx_emb_s, emb_ent[0][:-1].transpose(1, 0))
        score_i = torch.mm(ctx_emb_i, emb_ent[1][:-1].transpose(1, 0))
        score_t = torch.mm(ctx_emb_t, emb_ent[2][:-1].transpose(1, 0))
        score_mm = torch.mm(ctx_emb_mm, all_mm.transpose(1, 0))
        scores = [score_s, score_i, score_t, score_mm]
        rel_embs = self.rel_embeddings_s[triplets[:, 1] - self.num_ent] # [rel_emb_s, rel_emb_i, rel_emb_t, rel_emb_mm]
           
        out_d = self.relationguidefusion(scores, rel_embs)

        if self.training:
            return [score_s, score_i, score_t, score_mm]
        else:
            return out_d

