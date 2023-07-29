import yaml
import json
import random
import numpy as np
import torch

from dataConfig.chemdner import chemdner
from dataConfig.biomedical import biomedical

class Config:
    def __init__(self, entries):
        self.__dict__.update(entries)

def read_config(file):
    with open(file, "r") as f:
        entries = yaml.load(f, Loader=yaml.Loader)
    return json.loads(json.dumps(entries), object_hook=Config)

def get_src_dataset(cfg):
    if cfg.DATA.SRC_DATASET == "chemdner":
        data = chemdner(cfg.MODEL.BACKBONE, granularity=cfg.DATA.GRANULARITY, oracle=cfg.TRAIN.ORACLE)
        if cfg.TRAIN.ORACLE:
            cfg.DATA.SRC_DATASET += "_oracle"
    elif cfg.DATA.SRC_DATASET == "biomedical":
        data = biomedical(cfg, cfg.MODEL.BACKBONE, granularity=cfg.DATA.GRANULARITY)
    else:
        raise NotImplementedError(f"dataset {cfg.DATA.SRC_DATASET} is not supported")

    cfg.data = data
    return data

def get_tgt_dataset(cfg):
    if cfg.DATA.TGT_DATASET == "chemdner":
        data = chemdner(cfg.MODEL.BACKBONE, granularity=cfg.DATA.GRANULARITY, oracle=cfg.TRAIN.ORACLE)
        if cfg.TRAIN.ORACLE:
            cfg.DATA.TGT_DATASET += "_oracle"
    elif cfg.DATA.TGT_DATASET == "biomedical":
        data = biomedical(cfg, cfg.MODEL.BACKBONE, granularity=cfg.DATA.GRANULARITY)
    else:
        raise NotImplementedError(f"dataset {cfg.DATA.TGT_DATASET} is not supported")

    cfg.data = data
    return data

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)