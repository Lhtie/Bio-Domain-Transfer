import torch
from torch import nn
import torch.nn.functional as F
import datasets
from datasets import load_dataset, Dataset, DatasetDict
from .base import BaseDataConfig

class few_nerd(BaseDataConfig):
    def __init__(self, tokenizer_name, label_offset=0):
        super().__init__("few-nerd", tokenizer_name, label_offset)

        self.labels = [ 
            "O", "B-art", "I-art", "B-building", "I-building", "B-event", "I-event", "B-location", "I-location", "B-organization", "I-organization", "B-other", "I-other", "B-person", "I-person", "B-product", "I-product"
        ]
        self.id2label = {id_: label for id_, label in enumerate(self.labels)}
        self.label2id = {label: id_ for id_, label in enumerate(self.labels)}

    def rescale_labels(self, examples):
        for i, (tokens, tags) in enumerate(zip(examples["tokens"], examples["ner_tags"])):
            for j, (token, tag) in enumerate(zip(tokens, tags)):
                if tag == 0:
                    continue
                if j == 0 or token != tokens[j-1]:
                    tags[j] = tag * 2 - 1
                else:
                    tags[j] = tag * 2
            examples["ner_tags"][i] = tags
        return examples

    def load_dataset(self):
        raw = datasets.load_dataset("DFKI-SLT/few-nerd", "supervised")
        dataset = DatasetDict()
        for key in raw.keys():
            num = len(raw[key])
            dataset[key] = Dataset.from_dict(raw[key][:int(num * 0.2)])
            dataset[key] = dataset[key].map(self.rescale_labels, batched=True, batch_size=32)
        return dataset