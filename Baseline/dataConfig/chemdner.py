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

data_dir = "/mnt/data/oss_beijing/liuhongyi/datasets/chemdner_corpus"
letter_number = 'abcdefghijklmnopqrstuvwxyz0123456789'

class chemdner(BaseDataConfig):
    def __init__(self, tokenizer_name, granularity="para", cache_dir=".cache/", overwrite=False, oracle=False):
        dataset_name = "CHEMDNER"
        if oracle:
            dataset_name += "_oracle"
        super().__init__(dataset_name, tokenizer_name, granularity, cache_dir, overwrite)
        self.oracle = oracle

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

        self.raw_cache_file = os.path.join(cache_dir, f"{dataset_name}_{tokenizer_name}_raw.pt")
        os.makedirs(cache_dir, exist_ok=True)

        if not overwrite and os.path.exists(self.raw_cache_file):
            with open(self.raw_cache_file, "rb") as file:
                self.annotations, self.texts, self.etts = pickle.load(file)
        else:
            for split in ['training', 'development', 'evaluation']:
                self.read_from_file(split)
            with open(self.raw_cache_file, "wb") as file:
                pickle.dump((self.annotations, self.texts, self.etts), file)

    def read_from_file(self, split):
        self.texts[split] = {}
        offsets = {}
        with open(os.path.join(data_dir, f"{split}.abstracts.txt"), "r") as f:
            for line in f.readlines():
                pmid, title, abstract = line.strip().split('\t')[:3]
                self.texts[split][pmid] = title + '\n' + abstract
                offsets[pmid] = len(title) + 1

        self.annotations[split] = {}
        with open(os.path.join(data_dir, f"{split}.annotations.txt"), "r") as f:
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

    def load_dataset(self, tokenizer=None):
        dataset = DatasetDict()
        for split in ['training', 'development', 'evaluation']:
            split_dataset = self.process(split, tokenizer)
            dataset[split] = Dataset.from_dict(split_dataset)
        if not self.oracle:
            for key in ['training', 'development']:
                num = len(dataset[key])
                indices = set()
                for label in range(len(self.labels)):
                    count = 0
                    for idx, tags in enumerate(dataset[key]['ner_tags']):
                        if label in tags:
                            indices.add(idx)
                            count += 1
                            if count >= num * 0.05 / len(self.labels):
                                break
                dataset[key] = Dataset.from_dict(dataset[key][indices])

                label_cnt = {}
                for labels in dataset[key]['ner_tags']:
                    for label in labels:
                        if not label in label_cnt:
                            label_cnt[label] = 0
                        label_cnt[label] += 1
                print(label_cnt)
        print(dataset)  
        return dataset