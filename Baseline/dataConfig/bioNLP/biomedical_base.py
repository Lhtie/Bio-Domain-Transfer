import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from tqdm.auto import tqdm
import numpy as np
import pickle

from utils.entity_encoder import EntityEncoder
from utils.kmeans import *
from ..base import BaseDataConfig

# Configuration
eps = 1e-10
sapbert_path = "/mnt/data/oss_beijing/liuhongyi/models/SapBERT-from-PubMedBERT-fulltext"
sentbert_path = "/mnt/data/oss_beijing/liuhongyi/models/S-PubMedBert-MS-MARCO-SCIFACT"
bert_path = "/mnt/data/oss_beijing/liuhongyi/models/bert-base-uncased"

from sentence_transformers import SentenceTransformer, util
sent_model = SentenceTransformer('all-MiniLM-L6-v2')
# sent_model = SentenceTransformer(sentbert_path)

class Event:
    arguments = ["Theme", "Cause", "Product", "Site"]
    rand_pt = 0
    rand_emb_pool = []

    def __init__(self, Identifier, pad_tp="Pad: %s", Theme=None, Cause=None, Product=None, Site=None, **fn_kwargs):
        self.identifier = Identifier
        self.arg_vals = [Theme, Cause, Product, Site]
        self.full_args_map = fn_kwargs
        for i, (arg_key, arg_val) in enumerate(zip(Event.arguments, self.arg_vals)):
            if arg_val is None:
                self.arg_vals[i] = [pad_tp % arg_key]
            else:
                self.full_args_map[arg_key] = arg_val

        self.self = None

    def __str__(self):
        return f'<{self.identifier}: ' + ' '.join(
            [f'({key})[{val}]' for key, val in self.full_args_map.items()]
        ) + '>'
    
    def set_self(self, x):
        self.self = x
        return self
    
    def get_entities(self):
        entities = [self.identifier]
        for args in self.arg_vals:
            for arg in args:
                if arg is not None and not arg.startswith("Nested_Event-"):
                    entities.append(arg)
        return entities
    
    @staticmethod
    def get_rand_emb(mean, cov, n=1):
        if Event.rand_pt + n > len(Event.rand_emb_pool):
            Event.rand_emb_pool = np.random.multivariate_normal(mean, cov, 30000)
            Event.rand_pt = 0
        if n == 1:
            ret = Event.rand_emb_pool[Event.rand_pt]
        else:
            ret = Event.rand_emb_pool[Event.rand_pt:Event.rand_pt+n]
        Event.rand_pt += n
        return ret

    def downsize(self, emb, stride):
        compressed = []
        for i in range(0, len(emb), stride):
            compressed.append(np.mean(emb[i:i+stride]))
        return np.array(compressed)

    def get_embedding_concat(self, id2event, embs_map, mean, cov, self_emb):
        arr = [embs_map[self.identifier]]
        for key, vals in zip(self.arguments, self.arg_vals):
            cur_embs = []
            for id, val in enumerate(vals):
                if self.self == (key, id):
                    cur_embs.append(self_emb)
                elif val is not None:
                    if val.startswith("Nested_Event-"):
                        nested_emb = id2event[val[13:]].get_embedding_concat(id2event, embs_map, mean, cov, self_emb)
                        cur_embs.append(self.downsize(nested_emb, len(self.arguments) + 1))
                    else:
                        cur_embs.append(embs_map[val])
                else:
                    cur_embs.append(self.get_rand_emb(mean, cov))
            arr.append(np.mean(cur_embs, axis=0))
        return np.concatenate(arr)

    def get_embedding_sentEnc(self, id2event, tp):
        event_type, trigger = self.identifier.split(": ")
        self.full_args_map["Trigger"] = [trigger]
        event_tp = tp[event_type.split(" ")[0]]

        sent = event_tp["template"]
        for key in event_tp["arguments"]:
            replace = ""
            if key in self.full_args_map:
                e = []
                for id, val in enumerate(self.full_args_map[key]):
                    if self.self == (key, id):
                        e.append(f"{val} (self)")
                    elif val.startswith("Nested_Event-"):
                        event = id2event[val[13:]]
                        e.append("where " + event.get_embedding_sentEnc(id2event, tp).lower()[:-1])
                    else:
                        e.append(val)
                replace = ", ".join(e)
            sent = sent.replace(f"<{key}>", replace)
        return sent

class BiomedicalBaseDataConfig(BaseDataConfig):
    def __init__(self, ds_name, tokenizer_name, granularity="para", cache_dir=".cache/", overwrite=False, sim_method=None):
        super().__init__(ds_name, tokenizer_name, granularity, cache_dir, overwrite)
        self.sim_method = sim_method
        self.emb_method, self.agg_method = None, None
        if self.sim_method is not None:
            self.emb_method, self.agg_method = self.sim_method.split('-')

        cache_file = os.path.join(cache_dir, f"{ds_name}_{tokenizer_name}_{self.emb_method}_raw.pt")
        os.makedirs(cache_dir, exist_ok=True)

        if not overwrite and os.path.exists(cache_file):
            with open(cache_file, "rb") as file:
                self.ett_rel_set, self.annotations, self.texts = pickle.load(file)
            self.etts = list(self.ett_rel_set.keys())
        else:
            self.ett_rel_set, self.id2event = {}, {}
            self.load_raw_data()
            if self.emb_method is not None:
                self.init_embeddings()
            self.etts = list(self.ett_rel_set.keys())
            with open(cache_file, "wb") as file:
                pickle.dump((self.ett_rel_set, self.annotations, self.texts), file)

    @abstractmethod
    def load_raw_data(self):
        raise NotImplementedError()

    def init_embeddings(self):
        if self.emb_method == "concat":
            entities = set()
            for _, events in self.ett_rel_set.items():
                for event in events:
                    entities.update(event.get_entities())
            for _, event in self.id2event.items():
                    entities.update(event.get_entities())
            print(f"Totally {len(entities)} entities to embed")
            model = EntityEncoder(sapbert_path, cache_dir=self.cache_dir)
            embs = model.get_embedding(entities)
            embs_stacked = np.stack(embs.values())
            mean = embs_stacked.mean(axis=0)
            cov = np.cov(embs_stacked, rowvar=False)

            self_emb = Event.get_rand_emb(mean, cov)
            for entity, events in self.ett_rel_set.items():
                self.ett_rel_set[entity] = [e.get_embedding_concat(self.id2event, embs, mean, cov, self_emb) for e in events]
        elif self.emb_method == "sentEnc":
            assert hasattr(self, "emb_tp")
            sents = set()
            for entity, events in self.ett_rel_set.items():
                self.ett_rel_set[entity] = [e.get_embedding_sentEnc(self.id2event, self.emb_tp) for e in events]
                sents.update(self.ett_rel_set[entity])

            sents = list(sents)
            embs = sent_model.encode(sents)
            for entity, events in self.ett_rel_set.items():
                self.ett_rel_set[entity] = [embs[sents.index(e)] for e in events]
        elif self.emb_method == "entityEnc":
            model = EntityEncoder(sapbert_path, cache_dir=self.cache_dir)
            entities = list(self.ett_rel_set.keys())
            print(f"Totally {len(entities)} entities to embed")
            embs = model.get_embedding(entities)
            for entity, _ in self.ett_rel_set.items():
                self.ett_rel_set[entity] = [embs[entity]]
        else:
            raise NotImplementedError()
    
    @staticmethod
    def calc_sim_weight(etts, ett_rel_set, agg_method):
        if agg_method == "clus":
            return torch.zeros(len(etts), len(etts))
        if agg_method == "centraldist":
            ids = []
            central = {}
            for id, (entity, events) in enumerate(ett_rel_set.items()):
                if len(events) > 0:
                    ids.append(id)
                    central[entity] = np.mean(events, axis=0)
            stacked_emb = torch.tensor(np.stack(central.values()), dtype=torch.float)
            stacked_emb = stacked_emb / stacked_emb.norm(dim=1)[:, None]
            stacked_emb = torch.matmul(stacked_emb, torch.t(stacked_emb))

            sim_weight = torch.zeros(len(etts), len(etts))
            rows = torch.zeros(len(ids), len(etts))
            rows[:, ids] = stacked_emb
            sim_weight[ids] = rows
            return sim_weight

        stacked_emb = []
        ett_rel_id_set = {}
        for entity, events in ett_rel_set.items():
            ids = []
            for emb in events:
                ids.append(len(stacked_emb))
                stacked_emb.append(emb)
            ett_rel_id_set[entity] = ids
        stacked_emb = torch.tensor(np.stack(stacked_emb))
        stacked_emb = stacked_emb / stacked_emb.norm(dim=1)[:, None]
        sim_tensor = torch.matmul(stacked_emb, torch.t(stacked_emb))

        sim_weight = np.zeros([len(etts), len(etts)])
        for i, e1 in enumerate(tqdm(etts)):
            for j, e2 in enumerate(etts):
                similarities = []
                for emb1 in ett_rel_id_set[e1]:
                    for emb2 in ett_rel_id_set[e2]:
                        similarities.append(sim_tensor[emb1, emb2].item())
                if len(similarities) != 0:
                    if agg_method == "max":
                        sim_weight[i, j] = np.max(similarities)
                    elif agg_method == "min":
                        sim_weight[i, j] = np.min(similarities)
                    elif agg_method == "mean":
                        sim_weight[i, j] = np.mean(similarities)
                    else:
                        raise NotImplementedError()
        return torch.tensor(sim_weight)

    @staticmethod
    def init_clusters(ett_rel_set):
        ids, central_emb = [], []
        for id, (entity, events) in enumerate(ett_rel_set.items()):
            if len(events) > 0:
                ids.append(id)
                central_emb.append(np.mean(events, axis=0))
            
        central_emb = np.stack(central_emb)
        k_range = range(2, 20)
        best_k, best_labels, results = chooseBestKforKMeansParallel(central_emb, k_range)
        print(results)
        print(f"Best K: {best_k}")
        clusters = torch.tensor([-1] * len(ett_rel_set.keys()))
        for id, label in zip(ids, best_labels):
            clusters[id] = label
        return best_k, clusters
        
    def add_relation(self, ett, rel):
        if not ett in self.ett_rel_set:
            self.ett_rel_set[ett] = set()
        if rel is not None:
            self.ett_rel_set[ett].add(rel)
