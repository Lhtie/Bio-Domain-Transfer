import torch
from torch import nn
import torch.nn.functional as F
import os
import datasets
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict
from .base import BaseDataConfig

class chemdner(BaseDataConfig):
    def __init__(self, data_dir, granularity="para", cache_dir=".cache/", overwrite=False):
        dataset_name = "CHEMDNER"
        super().__init__(dataset_name, None, granularity, cache_dir, overwrite)
        self.data_dir = data_dir

        self.labels = [ 
            "O", "B-ABBREVIATION", "I-ABBREVIATION", "B-IDENTIFIER", "I-IDENTIFIER", "B-FORMULA", "I-FORMULA", "B-SYSTEMATIC", "I-SYSTEMATIC", "B-MULTIPLE", "I-MULTIPLE", "B-TRIVIAL", "I-TRIVIAL", "B-FAMILY", "I-FAMILY"
        ]
        self.id2label = {id_: label for id_, label in enumerate(self.labels)}
        self.label2id = {label: id_ for id_, label in enumerate(self.labels)}
        self.label_ids = {
            "NO CLASS": 0,
            "ABBREVIATION": 1,
            "IDENTIFIER": 3,
            "FORMULA": 5,
            "SYSTEMATIC": 7,
            "MULTIPLE": 9,
            "TRIVIAL": 11,
            "FAMILY": 13
        }
        for split in ['evaluation']:
            self.read_from_file(split)

    def read_from_file(self, split):
        self.texts[split] = {}
        offsets = {}
        with open(os.path.join(self.data_dir, f"{split}.abstracts.txt"), "r") as f:
            for line in f.readlines():
                pmid, title, abstract = line.strip().split('\t')[:3]
                self.texts[split][pmid] = title + '\n' + abstract
                offsets[pmid] = len(title) + 1

        self.annotations[split] = {}
        with open(os.path.join(self.data_dir, f"{split}.annotations.txt"), "r") as f:
            for line in f.readlines():
                pmid, type, start, end, entity, label = line.strip().split('\t')[:6]
                if type == "A":
                    start = int(start) + offsets[pmid]
                    end = int(end) + offsets[pmid]
                if pmid not in self.annotations[split]:
                    self.annotations[split][pmid] = []
                self.annotations[split][pmid].append((
                    int(start), int(end), entity, label, entity.lower()
                ))
                if not entity.lower in self.etts:
                    self.etts.append(entity.lower())

    def load_dataset(self):
        dataset = DatasetDict()
        for split in ['evaluation']:
            split_dataset = self.process(split)
            dataset[split] = Dataset.from_dict(split_dataset)
        print(dataset)  
        return dataset