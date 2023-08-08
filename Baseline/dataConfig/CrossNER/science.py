import torch
import os
import json
import datasets

from ..base import BaseDataConfig
from .crossner_base import CrossNERBaseDataConfig

class science(CrossNERBaseDataConfig):
    def __init__(self, tokenizer_name, cache_dir=".cache/", overwrite=False, sim_method=None):
        self.sim_method = sim_method
        self.labels = [ 
            'O', 'B-scientist', 'I-scientist', 'B-person', 'I-person', 'B-university', 'I-university', 'B-organisation', 'I-organisation', 'B-country', 'I-country', 'B-location', 'I-location', 'B-discipline', 'I-discipline', 'B-enzyme', 'I-enzyme', 'B-protein', 'I-protein', 'B-chemicalelement', 'I-chemicalelement', 'B-chemicalcompound', 'I-chemicalcompound', 'B-astronomicalobject', 'I-astronomicalobject', 'B-academicjournal', 'I-academicjournal', 'B-event', 'I-event', 'B-theory', 'I-theory', 'B-award', 'I-award', 'B-misc', 'I-misc'
        ]
        self.id2label = {id_: label for id_, label in enumerate(self.labels)}
        self.label2id = {label: id_ for id_, label in enumerate(self.labels)}
        
        super().__init__("Science", tokenizer_name, cache_dir, overwrite, sim_method)