import torch
from torch import nn
import torch.nn.functional as F
import os
import datasets
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict
from .biomedical_base import BiomedicalBaseDataConfig

data_dir = "/mnt/data/oss_beijing/liuhongyi/datasets/BioCreative-V-CDR-Corpus/CDR_Data/CDR_Data/CDR.Corpus.v010516"
split_dir = {
    "training": "CDR_TrainingSet.PubTator.txt", 
    "development": "CDR_DevelopmentSet.PubTator.txt", 
    "evaluation": "CDR_TestSet.PubTator.txt"
}

class bc5cdr(BiomedicalBaseDataConfig):
    def __init__(self, tokenizer_name, granularity="para", cache_dir=".cache/", overwrite=False):
        super().__init__("BC5CDR", tokenizer_name, granularity, cache_dir, overwrite)

        self.labels = [ 
            "O", "B-Chemical", "I-Chemical", "B-Disease", "I-Disease"
        ]
        self.id2label = {id_: label for id_, label in enumerate(self.labels)}
        self.label2id = {label: id_ for id_, label in enumerate(self.labels)}
        self.label_ids = {
            "NO CLASS": 0,
            "Chemical": 1,
            "Disease": 3
        }
    
    def load_raw_data(self):
        for split, s_dir in split_dir.items():
            self.read_from_file(os.path.join(data_dir, s_dir), split)

    def read_from_file(self, file, split):
        self.texts[split], self.annotations[split] = {}, {}
        with open(file, "r") as f:
            for line in f.readlines():
                if line == "\n":
                    continue
                if not '\t' in line:
                    pmid, type, text = line.strip().split('|')
                    if type == 't':
                        self.texts[split][pmid] = text
                    elif type == 'a':
                        self.texts[split][pmid] += '\n' + text
                    else:
                        raise NotImplementedError()
                elif line.split('\t')[1] == "CID":
                    pmid, _, id0, id1 = line.strip().split('\t')
                    if split == "training":
                        self.add_relation(pmid + id0, id1)
                else:
                    pmid, start, end, entity, label, id = line.strip().split('\t')[:6]
                    if not pmid in self.annotations[split]:
                        self.annotations[split][pmid] = []
                    self.annotations[split][pmid].append((
                        int(start), int(end), entity, label, pmid + id
                    ))
                    if label == "Chemical" and split == "training":
                        self.add_relation(pmid + id, None)

    def load_dataset(self, tokenizer=None):
        dataset = DatasetDict()
        for split in ['training', 'development', 'evaluation']:
            split_dataset = self.process(split, tokenizer)
            dataset[split] = Dataset.from_dict(split_dataset)
        print(dataset)
        return dataset