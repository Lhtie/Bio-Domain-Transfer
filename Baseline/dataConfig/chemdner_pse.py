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

from utils.config import get_src_dataset, get_tgt_dataset
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

        if overwrite or not os.path.exists(self.cache_file) or not os.path.exists(self.raw_cache_file):
            self.init_pseudo_label()
        else:
            with open(self.raw_cache_file, "rb") as file:
                self.annotations, self.texts, self.etts = pickle.load(file)

    def add_pseudo_label(self, batch):
        tokens = []
        for col, (token, token_id, tag, pse) in enumerate(zip(batch['tokens'], batch['token_id'], batch['ner_tags'], batch['pse_labels'])):
            if tag == 0 and pse not in ["B-Chemical", "I-Chemical"]:
                if pse.startswith("B-"):
                    if len(tokens) > 0:
                        self.etts.append(' '.join(tokens))
                        tokens = []
                if pse != "O":
                    tokens.append(token)
                    batch['token_id'][col] = len(self.etts)
                    batch['ner_tags'][col] = len(self.labels) + (pse.startswith("I-"))
        if len(tokens) > 0:
            self.etts.append(' '.join(tokens))
        return batch

    def init_pseudo_label(self):
        model_name = self.cfg.MODEL.PATH
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # predict
        if os.path.exists(self.pseudo_label_file):
            with open(self.pseudo_label_file, "r") as f:
                predictions = json.load(f)
        else:
            src_data = get_src_dataset(self.cfg)
            dataset = self.load(tokenizer)
            dataloader = torch.utils.data.DataLoader(dataset["training"], batch_size=self.cfg.EVAL.BATCH_SIZE)

            adapter_name = self.cfg.ADAPTER.EVAL
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
        dataset['training'] = dataset['training'].map(self.add_pseudo_label, batched=False)
        dataset = dataset.map(self.encode_labels, fn_kwargs={'tokenizer': tokenizer})
        dataset = dataset.map(self.encode_data, batched=True, batch_size=32, fn_kwargs={'tokenizer': tokenizer})
        dataset.set_format(type='torch', columns=[
            'input_ids', 'token_type_ids', 'attention_mask', 'labels', 'label_mask', 'ner_tags', 'token_id'
        ])
        with open(self.cache_file, "wb") as file:
            pickle.dump(dataset, file)
        with open(self.raw_cache_file, "wb") as file:
            pickle.dump((self.annotations, self.texts, self.etts), file)