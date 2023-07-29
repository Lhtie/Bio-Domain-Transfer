import torch
from torch import nn
import torch.nn.functional as F
import os
import datasets
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict
from .biomedical_base import BiomedicalBaseDataConfig

data_dir = "/mnt/data/oss_beijing/liuhongyi/datasets/bionlp-st-2013-cg/original-data"
split_dir = {
    "training": "train", 
    "development": "devel",
    "evaluation": "test"
}

class cg(BiomedicalBaseDataConfig):
    def __init__(self, tokenizer_name, granularity="para", cache_dir=".cache/", overwrite=False):
        super().__init__("CancerGenetics", tokenizer_name, granularity, cache_dir, overwrite)

        self.labels = [ 
            'O', 'B-Simple_chemical', 'I-Simple_chemical', 'B-Organism', 'I-Organism', 'B-Organism_subdivision', 'I-Organism_subdivision', 'B-Anatomical_system', 'I-Anatomical_system', 'B-Organ', 'I-Organ', 'B-Multi-tissue_structure', 'I-Multi-tissue_structure', 'B-Tissue', 'I-Tissue', 'B-Developing_anatomical_structure', 'I-Developing_anatomical_structure', 'B-Cell', 'I-Cell', 'B-Cellular component', 'I-Cellular component', 'B-Organism_substance', 'I-Organism_substance', 'B-Immaterial_anatomical_entity', 'I-Immaterial_anatomical_entity', 'B-Gene_or_gene_product', 'I-Gene_or_gene_product', 'B-Protein_domain_or_region', 'I-Protein_domain_or_region', 'B-Amino_acid', 'I-Amino_acid', 'B-DNA_domain_or_region', 'I-DNA_domain_or_region', 'B-Pathological_formation', 'I-Pathological_formation', 'B-Cancer', 'I-Cancer'
        ]
        self.id2label = {id_: label for id_, label in enumerate(self.labels)}
        self.label2id = {label: id_ for id_, label in enumerate(self.labels)}
        self.label_ids = {
            "NO CLASS": 0,
            "Simple_chemical": 1,
            "Organism": 3,
            "Organism_subdivision": 5,
            "Anatomical_system": 7,
            "Organ": 9,
            "Multi-tissue_structure": 11,
            "Tissue": 13,
            "Developing_anatomical_structure": 15,
            "Cell": 17,
            "Cellular_component": 19,
            "Organism_substance": 21,
            "Immaterial_anatomical_entity": 23,
            "Gene_or_gene_product": 25,
            "Protein_domain_or_region": 27,
            "Amino_acid": 29,
            "DNA_domain_or_region": 31,
            "Pathological_formation": 33,
            "Cancer": 35
        }
        
    def load_raw_data(self):
        for split, s_dir in split_dir.items():
            self.read_from_file(os.path.join(data_dir, s_dir), split)

    def read_from_file(self, dir, split):
        file_path = os.listdir(dir)
        self.texts[split], self.annotations[split] = {}, {}
        for file in tqdm(file_path):
            if file.endswith(".txt"):
                with open(os.path.join(dir, file), "r") as f:
                    text = f.read()
                    self.texts[split][file[:-4]] = text
            
            if file.endswith(".a1"):
                pmid = file[:-3]
                self.annotations[split][file[:-3]] = []
                with open(os.path.join(dir, file), "r") as f:
                    for line in f.readlines():
                        idx, middle, entity = line.strip().split('\t')
                        label, start, end = middle.split(' ')
                        self.annotations[split][file[:-3]].append((
                            int(start), int(end), entity, label, pmid + idx
                        ))
                        if split == "training":
                            self.add_relation(pmid + idx, None)
        # Construct mutual similarity
        if split == "training":
            equiv_id = 0
            id2rel_type = {}
            for file in tqdm(file_path):
                if file.endswith(".a2"):
                    pmid = file[:-3]
                    with open(os.path.join(dir, file), "r") as f:
                        lines = f.readlines()
                        for line in lines:
                            if line [0] == "E":
                                id, ctx = line.strip().split('\t')
                                rel_type = ctx.split(' ')[0].split(':')[0]
                                id2rel_type[pmid+id] = rel_type
                        for line in lines:
                            if line[0] == "*":
                                t1, t2 = line.strip().split(' ')[-2:]
                                self.add_relation(pmid + t1, f"Equivalent-{equiv_id}")
                                self.add_relation(pmid + t2, f"Equivalent-{equiv_id}")
                                equiv_id += 1
                            elif line[0] == "E":
                                id, ctx = line.strip().split('\t')
                                ctx = ctx.split(' ')
                                rel_type = ctx[0].split(':')[0]
                                for arg in ctx[1:]:
                                    role, ett = arg.split(':')
                                    if ett.startswith('E'):
                                        rel_type += f"_{role}:{id2rel_type[pmid+ett]}"
                                for arg in ctx[1:]:
                                    role, ett = arg.split(':')
                                    role = role.rstrip("0123456789")
                                    if ett.startswith('T'):
                                        self.add_relation(pmid + ett, f"{rel_type}-{role}")

    def load_dataset(self, tokenizer=None):
        dataset = DatasetDict()
        for split in ['training', 'development', 'evaluation']:
            split_dataset = self.process(split, tokenizer)
            dataset[split] = Dataset.from_dict(split_dataset)
        print(dataset)
        return dataset
