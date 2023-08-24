import yaml
import json
import random
import numpy as np
import torch

from dataConfig.chemdner import chemdner
from dataConfig.chemdner_pse import chemdner_pse
from dataConfig.biomedical import biomedical
from dataConfig.CrossNER import *

class Config:
    def __init__(self, entries):
        self.__dict__.update(entries)

def read_config(file):
    with open(file, "r") as f:
        entries = yaml.load(f, Loader=yaml.Loader)
    return json.loads(json.dumps(entries), object_hook=Config)

def get_dataset(cfg, type):
    if type == "chemdner":
        data = chemdner(cfg.MODEL.BACKBONE, granularity=cfg.DATA.GRANULARITY, oracle=cfg.TRAIN.ORACLE)
    elif type == "chemdner_pse":
        data = chemdner_pse(read_config("configs/para/transfer_learning.yaml"), cfg.MODEL.BACKBONE, 
                            granularity=cfg.DATA.GRANULARITY, oracle=cfg.TRAIN.ORACLE)
    elif type == "biomedical":
        data = biomedical(cfg, cfg.MODEL.BACKBONE, granularity=cfg.DATA.GRANULARITY)
    elif type in ["politics", "science", "music", "literature", "ai"]:
        data = globals()[type](cfg, cfg.MODEL.BACKBONE)
    else:
        raise NotImplementedError(f"dataset {cfg.DATA.SRC_DATASET} is not supported")
    return data

def get_src_dataset(cfg):
    return get_dataset(cfg, cfg.DATA.SRC_DATASET)

def get_tgt_dataset(cfg):
    return get_dataset(cfg, cfg.DATA.TGT_DATASET)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)