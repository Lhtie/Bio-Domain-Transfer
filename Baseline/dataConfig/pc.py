import torch
from torch import nn
import torch.nn.functional as F
import os
import json
import datasets
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict

from .biomedical_base import BiomedicalBaseDataConfig, Event

data_dir = "/mnt/data/oss_beijing/liuhongyi/datasets/BioNLP-ST_2013_PC"
emb_tp_path = "dataConfig/embedding_templates/pc_tp.json"
split_dir = {
    "training": "BioNLP-ST_2013_PC_training_data", 
    "development": "BioNLP-ST_2013_PC_development_data", 
    "evaluation": "BioNLP-ST_2013_PC_test_data"
}

class pc(BiomedicalBaseDataConfig):
    def __init__(self, tokenizer_name, granularity="para", cache_dir=".cache/", overwrite=False, sim_method=None):
        super().__init__("PathwayCuration", tokenizer_name, granularity, cache_dir, overwrite, sim_method)

        self.labels = [ 
            "O", "B-Simple_chemical", "I-Simple_chemical", "B-Gene_or_gene_product", "I-Gene_or_gene_product", "B-Complex", "I-Complex", "B-Cellular_component", "I-Cellular_component"
        ]
        self.id2label = {id_: label for id_, label in enumerate(self.labels)}
        self.label2id = {label: id_ for id_, label in enumerate(self.labels)}
        self.label_ids = {
            "NO CLASS": 0,
            "Simple_chemical": 1,
            "Gene_or_gene_product": 3,
            "Complex": 5,
            "Cellular_component": 7
        }

    def convert(self, key):
        if key == "AtLoc" or key == "ToLoc" or key == "FromLoc":
            return "Site"
        elif key == "Participant":
            return "Theme"
        else:
            return key

    def pre_parse(self, lines, pmid, id2entity):
        triggers, msgs = {}, {}
        for line in lines:
            if line.startswith("T"):
                idx, middle, entity = line.strip().split('\t')
                triggers[idx] = entity.lower()
                id2entity[pmid+idx] = entity.lower()
            if line.startswith("M"):
                _, msg = line.strip().split('\t')
                type, idx = msg.strip().split(' ')
                if idx not in msgs:
                    msgs[idx] = set()
                msgs[idx].add(type)
        return triggers, msgs

    def parse(self, lines, pmid, id2entity, triggers, msgs):
        for line in lines:
            if line.startswith("*"):
                t1, t2 = line.strip().split(' ')[-2:]
                self.add_relation(id2entity[pmid+t1], Event(f"Equivalent-{pmid}_{t1}_{t2}"))
                self.add_relation(id2entity[pmid+t2], Event(f"Equivalent-{pmid}_{t1}_{t2}"))
            if line.startswith("E"):
                idx, ctx = line.strip().split('\t')
                ctx = ctx.split(' ')

                event_id = ctx[0].split(':')[0]
                if idx in msgs:
                    event_id += f" ({', '.join(list(msgs[idx]))})"
                if ctx[0].split(':')[1] in triggers:
                    event_id += f": {triggers[ctx[0].split(':')[1]]}"

                args = {}
                for arg in ctx[1:]:
                    role, ett = arg.split(':')
                    role = role.rstrip("0123456789")
                    # Regulate argument type for concat method
                    if self.emb_method == "concat" and role not in Event.arguments:
                        role = self.convert(role)
                    if ett.startswith("T"):
                        ett = id2entity[pmid+ett]
                    elif ett.startswith("E"):
                        ett = "Nested_Event-" + pmid + ett
                    else:
                        raise NotImplementedError()

                    if role not in args:
                        args[role] = []
                    args[role].append(ett)

                self.id2event[pmid+idx] = Event(event_id, **args)
                for key, vals in args.items():
                    for id, val in enumerate(vals):
                        if not val.startswith("Nested_Event"):
                            pad_tp = f"Pad: {val}, {event_id}, %s"
                            self.add_relation(val, Event(event_id, pad_tp=pad_tp, **args).set_self((key, id)))

    def read_from_file(self, dir, split):
        file_path = os.listdir(dir)
        self.texts[split], self.annotations[split] = {}, {}
        id2entity = {}
        for file in tqdm(file_path):
            if file.endswith(".txt"):
                with open(os.path.join(dir, file), "r") as f:
                    text = f.read()
                    self.texts[split][file[:-4]] = text
            
            if file.endswith(".a1"):
                pmid = file[:-3]
                self.annotations[split][file[:-3]] = []
                with open(os.path.join(dir, file), "r") as f:
                    for line in f.readlines():
                        idx, middle, entity = line.strip().split('\t')
                        label, start, end = middle.split(' ')
                        self.annotations[split][file[:-3]].append((
                            int(start), int(end), entity, label, entity.lower()
                        ))
                        if split == "training":
                            self.add_relation(entity.lower(), None)
                            id2entity[pmid+idx] = entity.lower()
        # Construct mutual similarity
        if split == "training":
            for file in tqdm(file_path):
                if file.endswith(".a2"):
                    pmid = file[:-3]
                    with open(os.path.join(dir, file), "r") as f:
                        lines = f.readlines()
                        triggers, msgs = self.pre_parse(lines, pmid, id2entity)
                        self.parse(lines, pmid, id2entity, triggers, msgs)

    def load_raw_data(self):
        with open(emb_tp_path, "r") as f:
            self.emb_tp = json.load(f)
        for split, s_dir in split_dir.items():
            self.read_from_file(os.path.join(data_dir, s_dir), split)

    def load_dataset(self, tokenizer=None):
        dataset = DatasetDict()
        for split in ['training', 'development', 'evaluation']:
            split_dataset = self.process(split, tokenizer)
            dataset[split] = Dataset.from_dict(split_dataset)
        print(dataset)
        return dataset