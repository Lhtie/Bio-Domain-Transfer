import torch
import os
import json
import datasets

from ..base import BaseDataConfig
from .crossner_base import CrossNERBaseDataConfig

class music(CrossNERBaseDataConfig):
    def __init__(self, cfg, tokenizer_name, cache_dir=".cache/", overwrite=False, oracle=False, sim_method=None):
        self.sim_method = sim_method
        self.labels = [ 
            'O', 'B-musicgenre', 'I-musicgenre', 'B-song', 'I-song', 'B-band', 'I-band', 'B-album', 'I-album', 'B-musicalartist', 'I-musicalartist', 'B-musicalinstrument', 'I-musicalinstrument', 'B-award', 'I-award', 'B-event', 'I-event', 'B-country', 'I-country', 'B-location', 'I-location', 'B-organisation', 'I-organisation', 'B-person', 'I-person', 'B-misc', 'I-misc'
        ]
        self.id2label = {id_: label for id_, label in enumerate(self.labels)}
        self.label2id = {label: id_ for id_, label in enumerate(self.labels)}
        
        super().__init__(cfg, "Music", tokenizer_name, cache_dir, overwrite, oracle, sim_method)