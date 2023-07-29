import torch
from torch import nn
import torch.nn.functional as F
import os
import datasets
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict

from .base import BaseDataConfig
from .pc import pc
from .cg import cg
from .id import id
from .bc5cdr import bc5cdr
from .drugprot import drugprot

datasets = {
    "pc": pc,
    "cg": cg,
    "id": id,
    # "bc5cdr": bc5cdr,
    # "drugprot": drugprot
}

class biomedical(BaseDataConfig):
    def __init__(self, cfg, tokenizer_name, granularity="para", cache_dir=".cache/", overwrite=False):
        self.dataset_names, self.datasets = [], []
        for key, val in datasets.items():
            if hasattr(cfg.DATA, "BIOMEDICAL") and key not in cfg.DATA.BIOMEDICAL:
                continue
            self.dataset_names.append(key)
            self.datasets.append(val(tokenizer_name, granularity, cache_dir, overwrite))
        if hasattr(cfg.DATA, "BIOMEDICAL"):
            overwrite = True

        super().__init__(
            "BIOMEDICAL_" + "_".join(self.dataset_names), 
            tokenizer_name, 
            granularity, 
            cache_dir, 
            overwrite
        )

        self.labels = ["O", "B-Chemical", "I-Chemical"]
        for dataset in self.datasets:
            for label in dataset.labels[3:]:
                if not label in self.labels:
                    self.labels.append(label)
        for dataset in self.datasets:
            for key, val in dataset.label_ids.items():
                if val > 2:
                    dataset.label_ids[key] = self.labels.index(dataset.labels[dataset.label_ids[key]])
        self.id2label = {id_: label for id_, label in enumerate(self.labels)}
        self.label2id = {label: id_ for id_, label in enumerate(self.labels)}
        
        self.etts, self.rels = [], []
        for dataset in self.datasets:
            self.etts += dataset.etts
            self.rels += dataset.rels
        for dataset in self.datasets:
            dataset.etts = self.etts
        self.sim_weight = torch.block_diag(*[dataset.sim_weight for dataset in self.datasets])

    def load_dataset(self, tokenizer=None):
        dataset = DatasetDict()
        for split in ['training', 'development', 'evaluation']:
            agg_dataset = {'tokens': [], 'ner_tags': [], 'token_id': []}
            for d in self.datasets:
                sing_dataset = d.process(split, tokenizer)
                for key in agg_dataset:
                    agg_dataset[key] += sing_dataset[key]
            dataset[split] = Dataset.from_dict(agg_dataset)
        print(dataset)
        return dataset