import os
import pickle
import torch
from torch import nn
import torch.nn.functional as F
import datasets
from abc import ABC, abstractmethod

class BaseDataConfig(ABC):
    def __init__(self, ds_name, tokenizer_name, label_offset=0, cache_dir=".cache/", overwrite=False):
        self.ds_name = ds_name
        self.tokenizer_name = tokenizer_name
        self.label_offset = label_offset
        self.cache_file = os.path.join(cache_dir, f"{ds_name}_{tokenizer_name}.pt")
        self.overwrite = overwrite
        os.makedirs(cache_dir, exist_ok=True)

    def encode_data(self, data, tokenizer):
        encoded = tokenizer([" ".join(doc) for doc in data["tokens"]], pad_to_max_length=True, padding="max_length",
                            max_length=512, truncation=True, add_special_tokens=True)
        return (encoded)

    def encode_labels(self, example, tokenizer):
        r_tags = []
        label_mask = []
        token2word = []
        for index, token in enumerate(example["tokens"]):
            for idx_inner, token_inner in enumerate(tokenizer.tokenize(token)):
                if idx_inner == 0:
                    r_tags.append(example["ner_tags"][index] + self.label_offset)
                    label_mask.append(1)
                else:
                    r_tags.append(r_tags[-1] + self.label_offset)
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
    def load_dataset(self):
        return NotImplemented

    def load(self, tokenizer):
        if not self.overwrite and os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as file:
                dataset = pickle.load(file)
        else:
            dataset = self.load_dataset()
            dataset = dataset.map(self.encode_data, batched=True, batch_size=32, fn_kwargs={'tokenizer': tokenizer})
            dataset = dataset.map(self.encode_labels, fn_kwargs={'tokenizer': tokenizer})

            dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels', 'label_mask'])
            with open(self.cache_file, "wb") as file:
                pickle.dump(dataset, file)
        return dataset