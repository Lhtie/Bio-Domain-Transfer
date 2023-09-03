import torch
from torch import nn
import torch.nn.functional as F
import os
import pickle
import json
import datasets
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoAdapterModel

from .base import BaseDataConfig
from dataConfig.biomedical import biomedical
from dataConfig.CrossNER import *

data_dir = "/root/autodl-tmp/datasets/BioCreative-V-CDR-Corpus/CDR_Data/CDR_Data/CDR.Corpus.v010516"
split_dir = {
    "training": "CDR_TrainingSet.PubTator.txt", 
    "development": "CDR_DevelopmentSet.PubTator.txt", 
    "evaluation": "CDR_TestSet.PubTator.txt"
}
device = "cuda" if torch.cuda.is_available() else "cpu"

class bc5cdr(BaseDataConfig):
    def __init__(self, tokenizer_name, granularity="para", cache_dir=".cache/", overwrite=False, oracle=False):
        dataset_name = "BC5CDR"
        if oracle:
            dataset_name += "_oracle"
        super().__init__(dataset_name, tokenizer_name, granularity, cache_dir, overwrite)
        self.oracle = oracle

        self.labels = [ 
            "O", "B-Chemical", "I-Chemical"
        ]
        self.id2label = {id_: label for id_, label in enumerate(self.labels)}
        self.label2id = {label: id_ for id_, label in enumerate(self.labels)}
        self.label_ids = {
            "NO CLASS": 0,
            "Chemical": 1
        }

        self.raw_cache_file = os.path.join(cache_dir, f"{dataset_name}_{tokenizer_name}_raw.pt")
        os.makedirs(cache_dir, exist_ok=True)

        if not overwrite and os.path.exists(self.raw_cache_file):
            with open(self.raw_cache_file, "rb") as file:
                self.annotations, self.texts, self.etts = pickle.load(file)
        else:
            self.load_raw_data()
            with open(self.raw_cache_file, "wb") as file:
                pickle.dump((self.annotations, self.texts, self.etts), file)
    
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
                    pass
                else:
                    pmid, start, end, entity, label, id = line.strip().split('\t')[:6]
                    if label != "Chemical":
                        continue
                    if not pmid in self.annotations[split]:
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
                            if count >= num * 0.2 / (len(self.labels) - 1):
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

class bc5cdr_pse(bc5cdr):
    def __init__(self, cfg, tokenizer_name, granularity="para", cache_dir=".cache/", overwrite=False, oracle=False):
        super().__init__(tokenizer_name, granularity, cache_dir, overwrite, oracle)
        self.cfg = cfg
        self.cache_file += ".pse"
        self.raw_cache_file += ".pse"
        self.pseudo_label_file = os.path.join(cfg.OUTPUT.RESULT_SAVE_DIR, f"{self.ds_name}_pseudo_label.json")
        os.makedirs(cfg.OUTPUT.RESULT_SAVE_DIR, exist_ok=True)

        if overwrite or not os.path.exists(self.cache_file) or not os.path.exists(self.raw_cache_file):
            self.init_pseudo_label()
        else:
            with open(self.raw_cache_file, "rb") as file:
                self.annotations, self.texts, self.etts = pickle.load(file)
        self.pseudo_labels = ["O", "B-OOD", "I-OOD"]

    def add_pseudo_tag(self, batch):
        tokens = []
        pse_tags = [0] * len(batch['tokens'])
        for col, (token, token_id, tag, pse) in enumerate(zip(batch['tokens'], batch['token_id'], batch['ner_tags'], batch['pse_labels'])):
            if tag == 0:
                if pse.startswith("B-"):
                    if len(tokens) > 0:
                        self.etts.append(' '.join(tokens))
                        tokens = []
                if pse != "O":
                    tokens.append(token)
                    batch['token_id'][col] = len(self.etts)
                    pse_tags[col] = 1 + (pse.startswith("I-"))
        if len(tokens) > 0:
            self.etts.append(' '.join(tokens))
        batch['pse_tags'] = pse_tags
        return batch

    def init_pseudo_label(self):
        model_name = self.cfg.MODEL.PATH
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # predict
        if os.path.exists(self.pseudo_label_file):
            with open(self.pseudo_label_file, "r") as f:
                predictions = json.load(f)
        else:
            if self.cfg.DATA.TGT_DATASET == "biomedical":
                src_data = biomedical(self.cfg, self.cfg.MODEL.BACKBONE, granularity=self.cfg.DATA.GRANULARITY)
            elif self.cfg.DATA.TGT_DATASET in ["politics", "science", "music", "literature", "ai"]:
                src_data = globals()[self.cfg.DATA.TGT_DATASET](self.cfg, self.cfg.MODEL.BACKBONE)
            else:
                raise NotImplementedError(f"dataset {self.cfg.DATA.TGT_DATASET} is not supported")
            dataset = self.load(tokenizer)
            dataloader = torch.utils.data.DataLoader(dataset["training"], batch_size=self.cfg.EVAL.BATCH_SIZE)

            adapter_name = self.cfg.DATA.TGT_DATASET + "_ner_" + self.cfg.MODEL.BACKBONE
            head_name = self.cfg.DATA.TGT_DATASET + "_ner_" + self.cfg.MODEL.BACKBONE + "_head"
            model = AutoAdapterModel.from_pretrained(model_name)
            model.load_adapter(os.path.join(self.cfg.OUTPUT.ADAPTER_SAVE_DIR, adapter_name + "_inter"))
            model.set_active_adapters([adapter_name])
            model.load_head(os.path.join(self.cfg.OUTPUT.HEAD_SAVE_DIR, head_name + "_inter"))

            model.to(device).eval()
            predictions, references = [], []
            for batch in tqdm(dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}

                with torch.no_grad():
                    preds = model(batch["input_ids"]).logits
                    preds = preds.detach().cpu().numpy()
                    preds = np.argmax(preds, axis=2)
                
                for label_mask, pred, _ in zip(batch["label_mask"], preds, batch["labels"]):
                    predictions.append([src_data.id2label[id.item()] for mask, id in zip(label_mask, pred) if mask == 1])

            with open(self.pseudo_label_file, "w") as f:
                json.dump(predictions, f)

        # tokenize
        dataset = self.load_dataset(tokenizer)
        dataset['training'] = dataset['training'].add_column("pse_labels", predictions)
        dataset['training'] = dataset['training'].map(self.add_pseudo_tag, batched=False)
        dataset = dataset.map(self.encode_labels, fn_kwargs={'tokenizer': tokenizer})
        dataset = dataset.map(self.encode_data, batched=True, batch_size=32, fn_kwargs={'tokenizer': tokenizer})
        dataset.set_format(type='torch', columns=[
            'input_ids', 'token_type_ids', 'attention_mask', 'labels', 'label_mask', 'ner_tags', 'pse_tags', 'token_id'
        ])
        with open(self.cache_file, "wb") as file:
            pickle.dump(dataset, file)
        with open(self.raw_cache_file, "wb") as file:
            pickle.dump((self.annotations, self.texts, self.etts), file)
    
    def encode_data(self, data, tokenizer):
        encoded = tokenizer([" ".join(doc) for doc in data["tokens"]], pad_to_max_length=True, padding="max_length",
                            max_length=512, truncation=True, add_special_tokens=True)
        for col in ['ner_tags', 'pse_tags', 'token_id']:
            if col in data:
                encoded[col] = [vec[:512] + [0] * max(0, 512 - len(vec)) for vec in data[col]]
            else:
                encoded[col] = [[0] * 512 for vec in data['tokens']]
        return encoded