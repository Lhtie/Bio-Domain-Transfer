import torch
import os
import json
import datasets
import pickle
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict

from ..base import BaseDataConfig

data_dir = "/mnt/data/oss_beijing/liuhongyi/datasets/CrossNER/ner_data/"
split_dir = {
    "training": "train.txt", 
    "development": "dev.txt", 
    "evaluation": "test.txt"
}

class CrossNERBaseDataConfig(BaseDataConfig):
    def __init__(self, ds_name, tokenizer_name, cache_dir=".cache/", overwrite=False, sim_method=None):
        super().__init__(ds_name, tokenizer_name, "sent", cache_dir, overwrite)

        cache_file = os.path.join(cache_dir, f"{ds_name}_{tokenizer_name}_raw.pt")
        os.makedirs(cache_dir, exist_ok=True)

        if not overwrite and os.path.exists(cache_file):
            with open(cache_file, "rb") as file:
                self.dataset, self.etts = pickle.load(file)
        else:
            self.dataset, self.etts = {}, []
            self.load_raw_data()
            with open(cache_file, "wb") as file:
                pickle.dump((self.dataset, self.etts), file)

    def read_from_file(self, file, split):
        self.dataset[split] = {'tokens': [], 'ner_tags': [], 'token_id': []}
        with open(file, "r") as f:
            lines = f.readlines()

            ett_list, entity = [], []
            def push2etts():
                if len(entity) > 0:
                    cur_ett = ' '.join(entity)
                    ett_list.append(cur_ett)
                    if split == "training" and not cur_ett in self.etts:
                        self.etts.append(cur_ett)
            for line in lines:
                if line != "\n":
                    token, ner_tag = line.strip().split('\t')
                    if ner_tag.startswith("I-"):
                        entity.append(token)
                    elif ner_tag.startswith("B-"):
                        push2etts()
                        entity = [token]
            push2etts()
                        
            ett_idx = -1
            tokens, ner_tags, token_id = [], [], []
            def push2dataset():
                if len(tokens) > 0 and len(ner_tags) > 0 and len(token_id) > 0:
                    self.dataset[split]['tokens'].append(tokens)
                    self.dataset[split]['ner_tags'].append(ner_tags)
                    self.dataset[split]['token_id'].append(token_id)
            for line in lines:
                if line == "\n":
                    push2dataset()
                    tokens, ner_tags, token_id = [], [], []
                else:
                    token, ner_tag = line.strip().split('\t')
                    tokens.append(token)
                    ner_tags.append(self.label2id[ner_tag])
                    if ner_tag.startswith("B-"):
                        ett_idx += 1
                    if not ner_tag.startswith("O"):
                        entity = ett_list[ett_idx]
                        token_id.append(self.etts.index(entity) if entity in self.etts else -1)
                    else:
                        token_id.append(0)
            push2dataset()
    
    def load_raw_data(self):
        for split, s_dir in split_dir.items():
            self.read_from_file(os.path.join(data_dir, self.ds_name.lower(), s_dir), split)

    def load_dataset(self, tokenizer=None):
        dataset = DatasetDict()
        for split in ['training', 'development', 'evaluation']:
            dataset[split] = Dataset.from_dict(self.dataset[split])
        print(dataset)
        return dataset