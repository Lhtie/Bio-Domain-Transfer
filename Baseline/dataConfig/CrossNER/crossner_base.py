import torch
import os
import json
import datasets
import pickle
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict

from ..base import BaseDataConfig
from utils.entity_encoder import EntityEncoder
from utils.kmeans import *

bert_path = "/mnt/data/oss_beijing/liuhongyi/models/bert-base-uncased"
adapter_path = {
    "politics": "adapter/DAPT_Politics",
    "science": "adapter/DAPT_Science"
}
data_dir = "/mnt/data/oss_beijing/liuhongyi/datasets/CrossNER/ner_data/"
split_dir = {
    "training": "train.txt", 
    "development": "dev.txt", 
    "evaluation": "test.txt"
}

class CrossNERBaseDataConfig(BaseDataConfig):
    def __init__(self, cfg, ds_name, tokenizer_name, cache_dir=".cache/", overwrite=False, oracle=False, sim_method=None):
        if oracle:
            ds_name += "_oracle"
        super().__init__(ds_name, tokenizer_name, "sent", cache_dir, overwrite)

        if ds_name.split("_")[0].lower() == cfg.DATA.SRC_DATASET and hasattr(cfg.DATA, "CROSSNER"):
            if hasattr(cfg.DATA.CROSSNER, "SIM_METHOD"):
                sim_method = cfg.DATA.CROSSNER.SIM_METHOD
                if sim_method == "None":
                    sim_method = None
        self.sim_method = sim_method
        self.emb_method, self.agg_method = None, None
        if self.sim_method is not None:
            self.emb_method, self.agg_method = self.sim_method.split('-')
        self.oracle = oracle

        cache_file = os.path.join(cache_dir, f"{ds_name}_{tokenizer_name}_{sim_method}_raw.pt")
        os.makedirs(cache_dir, exist_ok=True)

        if not overwrite and os.path.exists(cache_file):
            with open(cache_file, "rb") as file:
                self.dataset, self.etts, self.sim_weight, self.K, self.clusters = pickle.load(file)
        else:
            self.dataset, self.etts = {}, []
            self.load_raw_data()
            self.init_sim_weight()
            with open(cache_file, "wb") as file:
                pickle.dump((self.dataset, self.etts, self.sim_weight, self.K, self.clusters), file)

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
            self.read_from_file(os.path.join(data_dir, self.ds_name.split('_')[0].lower(), s_dir), split)

    def load_dataset(self, tokenizer=None):
        dataset = DatasetDict()
        for split in ['training', 'development', 'evaluation']:
            dataset[split] = Dataset.from_dict(self.dataset[split])
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
                            if count >= num * 0.25 / len(self.labels):
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

    def init_embeddings(self):
        if self.emb_method == "entityEnc":
            model = EntityEncoder(bert_path, cache_dir=self.cache_dir, 
                                    adapter_path=adapter_path[self.ds_name.split("_")[0].lower()])
            print(f"Totally {len(self.etts)} entities to embed")
            embs = model.get_embedding(self.etts)
            self.ett_rel_set = {}
            for entity in self.etts:
                self.ett_rel_set[entity] = embs[entity]
        else:
            raise NotImplementedError()
    
    def calc_sim_weight(self, etts, ett_rel_set, agg_method):
        if agg_method == "clus":
            return torch.zeros(len(etts), len(etts))
        else:
            stacked_emb = torch.tensor(np.stack(ett_rel_set.values()), dtype=torch.float)
            stacked_emb = stacked_emb / stacked_emb.norm(dim=1)[:, None]
            stacked_emb = torch.matmul(stacked_emb, torch.t(stacked_emb))
            return stacked_emb

    def init_clusters(self, ett_rel_set):
        central_emb = np.stack(ett_rel_set.values())
        k_range = range(2, 20)
        best_k, best_labels, results = chooseBestKforKMeansParallel(central_emb, k_range)
        print(results)
        print(f"Best K: {best_k}")
        clusters = torch.tensor([-1] * len(ett_rel_set.keys()))
        for id, label in enumerate(best_labels):
            clusters[id] = label
        return best_k, clusters

    def init_sim_weight(self):
        if self.sim_method is not None:
            self.init_embeddings()
            self.sim_weight = self.calc_sim_weight(self.etts, self.ett_rel_set, self.agg_method)
            self.K, self.clusters = self.init_clusters(self.ett_rel_set)
        else:
            self.sim_weight = torch.zeros(len(self.etts), len(self.etts))
            self.K = 0
            self.clusters = {key: 0 for key in range(len(self.etts))}