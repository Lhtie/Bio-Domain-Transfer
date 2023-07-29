import torch
from torch import nn
import torch.nn.functional as F
import os
import datasets
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict
from .biomedical_base import BiomedicalBaseDataConfig

data_dir = "/mnt/data/oss_beijing/liuhongyi/datasets/drugprot-gs-training-development"
split_dir = {
    "training": "training/drugprot_training", 
    "development": "development/drugprot_development", 
    "evaluation": "test-background/test_background"
}

class drugprot(BiomedicalBaseDataConfig):
    def __init__(self, tokenizer_name, granularity="para", cache_dir=".cache/", overwrite=False):
        super().__init__("DRUGPROT", tokenizer_name, granularity, cache_dir, overwrite)

        self.labels = [ 
            "O", "B-CHEMICAL", "I-CHEMICAL", "B-Gene_or_gene_product", "I-Gene_or_gene_product"
        ]
        self.id2label = {id_: label for id_, label in enumerate(self.labels)}
        self.label2id = {label: id_ for id_, label in enumerate(self.labels)}
        self.label_ids = {
            "NO CLASS": 0,
            "CHEMICAL": 1,
            "GENE-Y": 3,
            "GENE-N": 3,
            "GENE": 3
        }
    
    def load_raw_data(self):
        for split, s_dir in split_dir.items():
            self.read_from_file(os.path.join(data_dir, s_dir), split)

    def read_from_file(self, file, split):
        self.texts[split], self.annotations[split] = {}, {}
        with open(file + "_abstracts.tsv", "r") as f:
            for line in f.readlines()[:500]:
                pmid, title, text = line.strip().split('\t')
                self.texts[split][pmid] = title + '\n' + text
        with open(file + "_entities.tsv", "r") as f:
            for line in f.readlines():
                pmid, id, label, start, end, entity = line.strip().split('\t')[:6]
                if not pmid in self.texts[split]:
                    continue
                if not pmid in self.annotations[split]:
                    self.annotations[split][pmid] = []
                self.annotations[split][pmid].append((
                    int(start), int(end), entity, label, pmid + id
                ))
                if split == "training":
                    self.add_relation(pmid + id, None)
        if split == "training":
            with open(file + "_relations.tsv", "r") as f:
                for line in f.readlines():
                    pmid, rel_type, arg1, arg2 = line.strip().split('\t')[:4]
                    if not pmid in self.texts[split]:
                        continue
                    self.add_relation(pmid + arg1.split(':')[1], f"{rel_type}-Arg1")
                    self.add_relation(pmid + arg2.split(':')[1], f"{rel_type}-Arg2")

    def load_dataset(self, tokenizer=None):
        dataset = DatasetDict()
        for split in ['training', 'development', 'evaluation']:
            split_dataset = self.process(split, tokenizer)
            dataset[split] = Dataset.from_dict(split_dataset)
        print(dataset)
        return dataset