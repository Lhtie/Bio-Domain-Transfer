import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
import pickle

from .base import BaseDataConfig

# Configuration
eps = 1e-10

class BiomedicalBaseDataConfig(BaseDataConfig):
    def __init__(self, ds_name, tokenizer_name, granularity="para", cache_dir=".cache/", overwrite=False):
        super().__init__(ds_name, tokenizer_name, granularity, cache_dir, overwrite)

        cache_file = os.path.join(cache_dir, f"{ds_name}_{tokenizer_name}_raw.pt")
        os.makedirs(cache_dir, exist_ok=True)

        if not overwrite and os.path.exists(cache_file):
            with open(cache_file, "rb") as file:
                self.ett_rel_set, self.rel_deg, self.annotations, self.texts = pickle.load(file)
        else:
            self.ett_rel_set, self.rel_deg = {}, {}
            self.load_raw_data()
            with open(cache_file, "wb") as file:
                pickle.dump((self.ett_rel_set, self.rel_deg, self.annotations, self.texts), file)
                
        self.rels = list(self.rel_deg.keys())
        self.etts = list(self.ett_rel_set.keys())
        self.init_sim_weight()

    @abstractmethod
    def load_raw_data(self):
        raise NotImplementedError()

    def init_sim_weight(self):
        self.rel_contrib = 1. / torch.tensor(list(self.rel_deg.values()), dtype=torch.float)
        self.rel_contrib = torch.sqrt(F.softmax(self.rel_contrib, dim=0))

        self.ett_rel_mtx = torch.zeros(len(self.etts), len(self.rels))
        for ett_id, ett in enumerate(self.etts):
            for rel in self.ett_rel_set[ett]:
                rel_id = self.rels.index(rel)
                self.ett_rel_mtx[ett_id, rel_id] = self.rel_contrib[rel_id]

        self.sim_weight = torch.matmul(self.ett_rel_mtx, torch.t(self.ett_rel_mtx))
        self.sim_weight += torch.diag(torch.tensor([eps] * len(self.sim_weight)))
        self.sim_weight /= torch.max(self.sim_weight, dim=1).values[:, None]

    def add_relation(self, ett, rel):
        if not ett in self.ett_rel_set:
            self.ett_rel_set[ett] = set()
        if rel is not None:
            if not rel in self.rel_deg:
                self.rel_deg[rel] = 0
            if not rel in self.ett_rel_set[ett]:
                self.rel_deg[rel] += 1
                self.ett_rel_set[ett].add(rel)
