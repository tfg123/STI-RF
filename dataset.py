import torch
from torch.utils.data import Dataset
import random
import os
from tqdm import tqdm


class STKG(Dataset):
    def __init__(self, data, logger, max_vis_len=-1):
        self.data = data
        self.logger = logger
        self.dir = f"data/{data}/"
        self.ent2id = {}
        self.id2ent = []
        self.rel2id = {}
        self.id2rel = []
        with open(self.dir + "entities.txt", encoding='utf-8') as f:
            for idx, line in enumerate(f.readlines()):
                self.ent2id[line.strip()] = idx
                self.id2ent.append(line.strip())
        self.num_ent = len(self.ent2id)

        with open(self.dir + "relations.txt", encoding='utf-8') as f:
            for idx, line in enumerate(f.readlines()):
                self.rel2id[line.strip()] = idx
                self.id2rel.append(line.strip())
        self.num_rel = len(self.rel2id)

        self.train = []
        with open(self.dir + "train.txt", encoding='utf-8') as f:
            for line in f.readlines():
                h, r, t = line.strip().split("\t")
                self.train.append((self.ent2id[h], self.rel2id[r], self.ent2id[t]))

        self.valid = []
        with open(self.dir + "valid.txt", encoding='utf-8') as f:
            for line in f.readlines():
                h, r, t = line.strip().split("\t")
                self.valid.append((self.ent2id[h], self.rel2id[r], self.ent2id[t]))

        self.test = []
        with open(self.dir + "test.txt", encoding='utf-8') as f:
            for line in f.readlines():
                h, r, t = line.strip().split("\t")
                self.test.append((self.ent2id[h], self.rel2id[r], self.ent2id[t]))

        self.filter_dict = {}

        for data_split in [self.train, self.valid, self.test]:
            for triplet in data_split:
                h, r, t = triplet
                if (-1, r, t) not in self.filter_dict:
                    self.filter_dict[(-1, r, t)] = []
                self.filter_dict[(-1, r, t)].append(h)
                if (h, r, -1) not in self.filter_dict:
                    self.filter_dict[(h, r, -1)] = []
                self.filter_dict[(h, r, -1)].append(t)

        self.max_vis_len_ent = max_vis_len
        self.max_vis_len_rel = max_vis_len

    def __len__(self):
        return len(self.train)

    def __getitem__(self, idx):
        h, r, t = self.train[idx]
        if random.random() < 0.5:
            masked_triplet = [self.num_ent + self.num_rel, r + self.num_ent, t + self.num_rel]
            label = h  
        else:
            masked_triplet = [h + self.num_rel, r + self.num_ent, self.num_ent + self.num_rel]
            label = t  

        return torch.tensor(masked_triplet), torch.tensor(label)
