import torch
from torch import nn
import torch.nn.functional as F
import os
import json
import pickle
import datasets
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoAdapterModel

from dataConfig.biomedical import biomedical
from dataConfig.CrossNER import *
from .chemdner import chemdner

data_dir = "/mnt/data/oss_beijing/liuhongyi/datasets/chemdner_corpus"
letter_number = 'abcdefghijklmnopqrstuvwxyz0123456789'
device = "cuda" if torch.cuda.is_available() else "cpu"

class chemdner_pse(chemdner):
    def __init__(self, cfg, tokenizer_name, granularity="para", cache_dir=".cache/", overwrite=False, oracle=False):
        super().__init__(tokenizer_name, granularity, cache_dir, overwrite, oracle)
        self.cfg = cfg
        self.cache_file += ".pse"
        self.raw_cache_file += ".pse"
        self.pseudo_label_file = os.path.join(cfg.OUTPUT.RESULT_SAVE_DIR, "pseudo_label.json")
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
            if tag == 0 and pse not in ["B-Chemical", "I-Chemical"]:
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
            if self.cfg.DATA.SRC_DATASET == "biomedical":
                src_data = biomedical(self.cfg, self.cfg.MODEL.BACKBONE, granularity=self.cfg.DATA.GRANULARITY)
            elif self.cfg.DATA.SRC_DATASET in ["politics", "science", "music", "literature", "ai"]:
                src_data = globals()[type](self.cfg, self.cfg.MODEL.BACKBONE)
            else:
                raise NotImplementedError(f"dataset {cfg.DATA.SRC_DATASET} is not supported")
            dataset = self.load(tokenizer)
            dataloader = torch.utils.data.DataLoader(dataset["training"], batch_size=self.cfg.EVAL.BATCH_SIZE)

            adapter_name = self.cfg.DATA.SRC_DATASET + "_ner_" + self.cfg.MODEL.BACKBONE
            head_name = self.cfg.DATA.SRC_DATASET + "_ner_" + self.cfg.MODEL.BACKBONE + "_head"
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
    
    def report_bio_chem_portion(self):
        with open(self.pseudo_label_file, "r") as f:
            predictions = json.load(f)
        dataset = self.load_dataset()
        dataset['training'] = dataset['training'].add_column("pse_labels", predictions)
        bio_in_chem, chem = 0, 0
        for batch in dataset['training']:
            for col, (token, token_id, tag, pse) in enumerate(zip(batch['tokens'], batch['token_id'], batch['ner_tags'], batch['pse_labels'])):
                if tag > 0 and pse != "O":
                    bio_in_chem += 1
                if tag > 0:
                    chem += 1
        print(f"Pseudo Biomedical percentage in Chemicals: {bio_in_chem / chem}")