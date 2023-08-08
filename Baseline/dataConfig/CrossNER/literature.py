import torch
import os
import json
import datasets

from ..base import BaseDataConfig
from .crossner_base import CrossNERBaseDataConfig

class literature(CrossNERBaseDataConfig):
    def __init__(self, tokenizer_name, cache_dir=".cache/", overwrite=False, sim_method=None):
        self.sim_method = sim_method
        self.labels = [ 
            "O", "B-book", "I-book", "B-writer", "I-writer", "B-award", "I-award", "B-poem", "I-poem", "B-event", "I-event", "B-magazine", "I-magazine", "B-literarygenre", "I-literarygenre", 'B-country', 'I-country', "B-person", "I-person", "B-location", "I-location", 'B-organisation', 'I-organisation', 'B-misc', 'I-misc'
        ]
        self.id2label = {id_: label for id_, label in enumerate(self.labels)}
        self.label2id = {label: id_ for id_, label in enumerate(self.labels)}

        super().__init__("Literature", tokenizer_name, cache_dir, overwrite, sim_method)