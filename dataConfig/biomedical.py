import torch
from torch import nn
import torch.nn.functional as F
import os
import pickle
import datasets
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict

from .base import BaseDataConfig
from .bioNLP.biomedical_base import BiomedicalBaseDataConfig
from .bioNLP import pc, cg, id

datasets = {
    "pc": pc,
    "cg": cg,
    "id": id
}

class biomedical(BaseDataConfig):
    def __init__(self, cfg, tokenizer_name, granularity="para", cache_dir=".cache/", overwrite=False, retain_chem=False, auto=False):
        self.dataset_names = list(datasets.keys())
        self.datasets = []
        self.sim_method = None

        if hasattr(cfg.DATA, "BIOMEDICAL"):
            if hasattr(cfg.DATA.BIOMEDICAL, "DATASETS") and cfg.DATA.BIOMEDICAL.DATASETS != "None":
                self.dataset_names = cfg.DATA.BIOMEDICAL.DATASETS
            if hasattr(cfg.DATA.BIOMEDICAL, "SIM_METHOD"):
                self.sim_method = cfg.DATA.BIOMEDICAL.SIM_METHOD
                if self.sim_method == "None":
                    self.sim_method = None

        for key, val in datasets.items():
            if key in self.dataset_names:
                self.datasets.append(val(tokenizer_name, granularity, cache_dir, overwrite, retain_chem, self.sim_method, auto))

        if retain_chem:
            cache_dir = os.path.join(cache_dir, "retain_chem")
        super().__init__(
            "BIOMEDICAL_" + "_".join(self.dataset_names) + ("_auto" if auto else ""), 
            tokenizer_name, 
            granularity, 
            cache_dir, 
            overwrite
        )
        cache_file = os.path.join(cache_dir, f"{self.ds_name}_{tokenizer_name}_{self.sim_method}_raw.pt")
        os.makedirs(cache_dir, exist_ok=True)

        if retain_chem:
            self.labels = ["O", "B-Chemical", "I-Chemical"]
        else:
            self.labels = ["O"]
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
        
        if not overwrite and os.path.exists(cache_file):
            with open(cache_file, "rb") as file:
                self.ett_rel_set, self.sim_weight, self.K, self.clusters = pickle.load(file)
            self.etts = list(self.ett_rel_set.keys())
        else:
            self.ett_rel_set = {}
            for dataset in self.datasets:
                for entity, events in dataset.ett_rel_set.items():
                    if entity not in self.ett_rel_set:
                        self.ett_rel_set[entity] = list(events)
                    else:
                        self.ett_rel_set[entity] += list(events)
            self.etts = list(self.ett_rel_set.keys())
            if self.sim_method is not None:
                self.sim_weight = BiomedicalBaseDataConfig.calc_sim_weight(
                                self.etts, self.ett_rel_set, self.sim_method.split('-')[1])
                self.K, self.clusters = BiomedicalBaseDataConfig.init_clusters(self.ett_rel_set)
            else:
                self.sim_weight = torch.zeros(len(self.etts), len(self.etts))
                self.K = 0
                self.clusters = {key: 0 for key in range(len(self.etts))}
            with open(cache_file, "wb") as file:
                pickle.dump((self.ett_rel_set, self.sim_weight, self.K, self.clusters), file)
            
        for dataset in self.datasets:
            dataset.etts = self.etts

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