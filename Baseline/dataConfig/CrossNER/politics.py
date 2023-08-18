import torch
import os
import json
import datasets

from ..base import BaseDataConfig
from .crossner_base import CrossNERBaseDataConfig

class politics(CrossNERBaseDataConfig):
    def __init__(self, cfg, tokenizer_name, cache_dir=".cache/", overwrite=False, oracle=True, sim_method=None):
        self.sim_method = sim_method
        self.labels = [ 
            'O', 'B-country', 'B-politician', 'I-politician', 'B-election', 'I-election', 'B-person', 'I-person', 'B-organisation', 'I-organisation', 'B-location', 'B-misc', 'I-location', 'I-country', 'I-misc', 'B-politicalparty', 'I-politicalparty', 'B-event', 'I-event'
        ]
        self.id2label = {id_: label for id_, label in enumerate(self.labels)}
        self.label2id = {label: id_ for id_, label in enumerate(self.labels)}

        super().__init__(cfg, "Politics", tokenizer_name, cache_dir, overwrite, oracle, sim_method)