import os
import pickle
import torch
from torch import nn
import torch.nn.functional as F
import datasets
from abc import ABC, abstractmethod

import nltk
from nltk.tokenize import wordpunct_tokenize, word_tokenize, sent_tokenize

class BaseDataConfig(ABC):
    def __init__(self, ds_name, tokenizer_name, granularity="para", cache_dir=".cache/", overwrite=False):
        self.ds_name = ds_name
        self.tokenizer_name = tokenizer_name
        self.granularity = granularity
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, f"{ds_name}_{tokenizer_name}_{granularity}_tensorized.pt")
        self.overwrite = overwrite
        os.makedirs(cache_dir, exist_ok=True)

        self.texts, self.annotations = {}, {}
        self.etts = []

    def encode_data(self, data, tokenizer):
        encoded = tokenizer([" ".join(doc) for doc in data["tokens"]], pad_to_max_length=True, padding="max_length",
                            max_length=512, truncation=True, add_special_tokens=True)
        for col in ['ner_tags', 'token_id']:
            if col in data:
                encoded[col] = [vec[:512] + [0] * max(0, 512 - len(vec)) for vec in data[col]]
            else:
                encoded[col] = [[0] * 512 for vec in data['tokens']]
        return encoded

    def encode_labels(self, example, tokenizer):
        r_tags = []
        token2word = []
        label_mask = []
        for index, token in enumerate(example["tokens"]):
            for idx_inner, token_inner in enumerate(tokenizer.tokenize(token)):
                if idx_inner == 0:
                    r_tags.append(example["ner_tags"][index])
                    label_mask.append(1)
                else:
                    if r_tags[-1] % 2 == 1:
                        r_tags.append(r_tags[-1] + 1)
                    else:
                        r_tags.append(r_tags[-1])
                    label_mask.append(0)
                token2word.append(index)

        def __push(vec, col):
            vec = torch.tensor(vec)
            # Pad token to maximum length for using batches
            example[col] = F.pad(vec, pad=(1, 511 - vec.shape[0]), mode='constant', value=0)
            # Truncate if the document is too long
            example[col] = example[col][:512]

        __push(token2word, "token2word")
        __push(r_tags, "labels")
        __push(label_mask, "label_mask")
        return example

    @abstractmethod
    def load_dataset(self, tokenizer=None):
        return NotImplementedError()

    def load(self, tokenizer):
        if not self.overwrite and os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as file:
                dataset = pickle.load(file)
        else:
            dataset = self.load_dataset(tokenizer)
            dataset = dataset.map(self.encode_labels, fn_kwargs={'tokenizer': tokenizer})
            dataset = dataset.map(self.encode_data, batched=True, batch_size=32, fn_kwargs={'tokenizer': tokenizer})

            dataset.set_format(type='torch', columns=[
                'input_ids', 'token_type_ids', 'attention_mask', 'labels', 'label_mask', 'ner_tags', 'token_id'
            ])
            with open(self.cache_file, "wb") as file:
                pickle.dump(dataset, file)

        return dataset

    def process(self, split, tokenizer):
        dataset = {'tokens': [], 'ner_tags': [], 'token_id': []}
        lens = []
        for pmid, txt in self.texts[split].items():
            if pmid in self.annotations[split] and len(self.annotations[split][pmid]) > 0:
                tokens = word_tokenize(txt)
                ner_tags = [0] * len(tokens)
                token_id = [0] * len(tokens)
                lens.append(len(tokens))

                pos = 0
                start_pos = []
                for token in tokens:
                    while pos + len(token) <= len(txt) and token != txt[pos:pos+len(token)]:
                        pos += 1
                    start_pos.append(pos)
                    pos += len(token)
                for s, t, e, label, id in self.annotations[split][pmid]:
                    label_id = self.label_ids[label]
                    coverage = []
                    start_idx = None
                    for idx, pos in enumerate(start_pos):
                        if pos >= s:
                            if pos == s:
                                start_idx = idx
                            if pos + len(tokens[idx]) <= t:
                                coverage.append(idx)
                        elif pos > t:
                            break
                    if start_idx is None:
                        # annotation not found
                        continue
                    if len(coverage) == 0:
                        # Split the token
                        left_part = tokens[start_idx][:t-s]
                        right_part = tokens[start_idx][t-s:]
                        tokens[start_idx] = left_part
                        tokens.insert(start_idx + 1, right_part)

                        ner_tags[start_idx] = label_id
                        ner_tags.insert(start_idx + 1, 0)

                        token_id[start_idx] = self.etts.index(id) if id in self.etts else -1
                        token_id.insert(start_idx + 1, 0)

                        start_pos.insert(start_idx + 1, start_pos[start_idx] + len(left_part))
                    else:
                        ner_tags[coverage[0]] = label_id
                        if len(coverage) > 1:
                            for idx in coverage[1:]:
                                ner_tags[idx] = label_id + 1
                        for idx in coverage:
                            token_id[idx] = self.etts.index(id) if id in self.etts else -1

                if self.granularity == "para":          
                    dataset['tokens'].append(tokens)
                    dataset['ner_tags'].append(ner_tags)
                    dataset['token_id'].append(token_id)
                else:
                    tokens_sent, ner_tags_sent, token_id_sent = [], [], []
                    for token, ner_tag, id in zip(tokens, ner_tags, token_id):
                        tokens_sent.append(token)
                        ner_tags_sent.append(ner_tag)
                        token_id_sent.append(id)
                        if token.endswith('.'):
                            dataset['tokens'].append(tokens_sent)
                            dataset['ner_tags'].append(ner_tags_sent)
                            dataset['token_id'].append(token_id_sent)
                            tokens_sent, ner_tags_sent, token_id_sent = [], [], []
                    if len(tokens_sent) > 0:
                        dataset['tokens'].append(tokens_sent)
                        dataset['ner_tags'].append(ner_tags_sent)
                        dataset['token_id'].append(token_id_sent)
        return dataset