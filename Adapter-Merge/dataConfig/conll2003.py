import torch
from torch import nn
import torch.nn.functional as F
import datasets
from datasets import load_dataset
from .base import BaseDataConfig

class conll2003(BaseDataConfig):
    def __init__(self, tokenizer_name, label_offset=0):
        super().__init__("conll2003", tokenizer_name, label_offset)

        self.labels = [ 
            "O", "B-PER", "I-PER", "B-ORG", "B-LOC", "I-LOC", "I-ORG", "B-MISC", "I-MISC"
        ]
        self.id2label = {id_: label for id_, label in enumerate(self.labels)}
        self.label2id = {label: id_ for id_, label in enumerate(self.labels)}

    def load_dataset(self):
        dataset = load_dataset("conll2003")
        return dataset