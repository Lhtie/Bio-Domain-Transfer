import yaml
import json
import random
import numpy as np
import torch

from dataConfig.chemdner import chemdner, chemdner_pse, chemdner_cls
from dataConfig.bc5cdr import bc5cdr, bc5cdr_pse, bc5cdr_cls
from dataConfig.drugprot import drugprot, drugprot_pse, drugprot_cls
from dataConfig.biomedical import biomedical

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
        data = chemdner_pse(cfg, cfg.MODEL.BACKBONE, granularity=cfg.DATA.GRANULARITY, oracle=cfg.TRAIN.ORACLE)
    elif type == "chemdner_pse_no_ms":
        data = chemdner_pse(cfg, cfg.MODEL.BACKBONE, granularity=cfg.DATA.GRANULARITY, oracle=cfg.TRAIN.ORACLE, use_ms=False)
    elif type == "chemdner_cls":
        data = chemdner_cls(cfg, cfg.MODEL.BACKBONE, granularity=cfg.DATA.GRANULARITY, oracle=cfg.TRAIN.ORACLE)
    elif type == "bc5cdr":
        data = bc5cdr(cfg.MODEL.BACKBONE, granularity=cfg.DATA.GRANULARITY, oracle=cfg.TRAIN.ORACLE)
    elif type == "bc5cdr_pse":
        data = bc5cdr_pse(cfg, cfg.MODEL.BACKBONE, granularity=cfg.DATA.GRANULARITY, oracle=cfg.TRAIN.ORACLE)
    elif type == "bc5cdr_pse_no_ms":
        data = bc5cdr_pse(cfg, cfg.MODEL.BACKBONE, granularity=cfg.DATA.GRANULARITY, oracle=cfg.TRAIN.ORACLE, use_ms=False)
    elif type == "bc5cdr_cls":
        data = bc5cdr_cls(cfg, cfg.MODEL.BACKBONE, granularity=cfg.DATA.GRANULARITY, oracle=cfg.TRAIN.ORACLE)
    elif type == "drugprot":
        data = drugprot(cfg.MODEL.BACKBONE, granularity=cfg.DATA.GRANULARITY, oracle=cfg.TRAIN.ORACLE)
    elif type == "drugprot_pse":
        data = drugprot_pse(cfg, cfg.MODEL.BACKBONE, granularity=cfg.DATA.GRANULARITY, oracle=cfg.TRAIN.ORACLE)
    elif type == "drugprot_pse_no_ms":
        data = drugprot_pse(cfg, cfg.MODEL.BACKBONE, granularity=cfg.DATA.GRANULARITY, oracle=cfg.TRAIN.ORACLE, use_ms=False)
    elif type == "drugprot_cls":
        data = drugprot_cls(cfg, cfg.MODEL.BACKBONE, granularity=cfg.DATA.GRANULARITY, oracle=cfg.TRAIN.ORACLE)
    elif type == "biomedical":
        data = biomedical(cfg, cfg.MODEL.BACKBONE, granularity=cfg.DATA.GRANULARITY)
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