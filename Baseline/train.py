from transformers import AutoAdapterModel, AutoTokenizer, AutoConfig
from datasets import load_dataset
from torch.utils.data import Dataset
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import os
import copy
import argparse
import numpy as np
import logging

from utils.trainer import get, train
from utils.config import read_config, get_tgt_dataset, get_src_dataset, set_seed

adapter_dir = "adapter/"

def get_dataloaders(cfg, tokenizer, is_src):
    if cfg.local_rank not in [-1, 0]:
        torch.distributed.barrier()
    if is_src:
        data = get_src_dataset(cfg)
        cfg.data = data
    else:
        data = get_tgt_dataset(cfg)
        cfg.data = data
    dataset = data.load(tokenizer)
    if cfg.local_rank == 0:
        print(dataset)
        torch.distributed.barrier()

    if cfg.local_rank == -1:
        train_sampler = None
    else:
        train_sampler = DistributedSampler(dataset['training'], shuffle=True, seed=cfg.TRAIN.SEED)
    batch_size = cfg.TRAIN.SRC_BATCH_SIZE if is_src else cfg.TRAIN.BATCH_SIZE
    train_dataloader = torch.utils.data.DataLoader(dataset['training'], sampler=train_sampler, batch_size=batch_size // cfg.world_size)
    dev_dataloader = torch.utils.data.DataLoader(dataset['development'], batch_size=batch_size // cfg.world_size)

    return train_dataloader, dev_dataloader, data

def train_single(cfg, model, tokenizer):
    # load data
    train_dataloader, dev_dataloader, data = get_dataloaders(cfg, tokenizer, False)

    # add adapter
    adapter_name = cfg.DATA.TGT_DATASET + "_ner_" + cfg.MODEL.BACKBONE
    head_name = adapter_name + "_head"
    model.add_adapter(adapter_name)
    model.add_tagging_head(head_name, num_labels=len(data.labels), id2label=data.id2label)
    model.train_adapter([adapter_name])
    model.to(cfg.device)
    if cfg.local_rank != -1:
        model = torch.nn.parallel.distributed.DistributedDataParallel(model, device_ids=[cfg.local_rank], output_device=cfg.local_rank)

    # train
    cfg.TRAIN.EPOCHS = cfg.TRAIN.TGT_EPOCHS
    model, best_f1 = train(cfg, model, tokenizer, train_dataloader, dev_dataloader, adapter_name, head_name)
    return model, adapter_name, head_name, best_f1

def train_two_stage(cfg, model, tokenizer):
    # load data
    train_dataloader, dev_dataloader, data = get_dataloaders(cfg, tokenizer, True)

    # add adapter
    adapter_name = cfg.DATA.SRC_DATASET + "_ner_" + cfg.MODEL.BACKBONE + "_2stage"
    head_name = cfg.DATA.SRC_DATASET + "_ner_" + cfg.MODEL.BACKBONE + "_head"
    model.add_adapter(adapter_name)
    model.add_tagging_head(head_name, num_labels=len(data.labels), id2label=data.id2label)
    model.train_adapter([adapter_name])
    model.to(cfg.device)
    if cfg.local_rank != -1:
        model = torch.nn.parallel.distributed.DistributedDataParallel(model, device_ids=[cfg.local_rank], output_device=cfg.local_rank)

    # train 4 src
    cfg.TRAIN.EPOCHS = cfg.TRAIN.SRC_EPOCHS
    if cfg.LOSSES.NAME == "CE_MS":
        model, _ = train(cfg, model, tokenizer, train_dataloader, dev_dataloader, adapter_name, head_name, use_ms=True, pretrain=True)
    elif cfg.LOSSES.NAME == "CrossEntropy":
        model, _ = train(cfg, model, tokenizer, train_dataloader, dev_dataloader, adapter_name, head_name, pretrain=True)
    else:
        raise NotImplemented
    if cfg.local_rank in [-1, 0]:
        os.makedirs(os.path.join(cfg.OUTPUT.ADAPTER_SAVE_DIR, adapter_name + "_inter"), exist_ok=True)
        os.makedirs(os.path.join(cfg.OUTPUT.HEAD_SAVE_DIR, head_name + "_inter"), exist_ok=True)
        get(cfg, model).save_adapter(os.path.join(cfg.OUTPUT.ADAPTER_SAVE_DIR, adapter_name + "_inter"), adapter_name)
        get(cfg, model).save_head(os.path.join(cfg.OUTPUT.HEAD_SAVE_DIR, head_name + "_inter"), head_name)
        cfg.logger.info("Best model for the 1st stage saved")

    # prepare for target
    train_dataloader, dev_dataloader, data = get_dataloaders(cfg, tokenizer, False)

    model = AutoAdapterModel.from_pretrained(cfg.MODEL.PATH)
    model.load_adapter(os.path.join(cfg.OUTPUT.ADAPTER_SAVE_DIR, adapter_name + "_inter"))
    head_name = cfg.DATA.TGT_DATASET + "_ner_" + cfg.MODEL.BACKBONE + "_head_2stage"
    model.add_tagging_head(head_name, num_labels=len(data.labels), id2label=data.id2label, overwrite_ok=True)
    model.train_adapter([adapter_name])
    model.to(cfg.device)
    if cfg.local_rank != -1:
        model = torch.nn.parallel.distributed.DistributedDataParallel(model, device_ids=[cfg.local_rank], output_device=cfg.local_rank)

    # train 4 tgt
    cfg.TRAIN.EPOCHS = cfg.TRAIN.TGT_EPOCHS
    model, best_f1 = train(cfg, model, tokenizer, train_dataloader, dev_dataloader, adapter_name, head_name)
    return model, adapter_name, head_name, best_f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", type=str, default="configs/sample.yaml")
    args = parser.parse_args()

    cfg = read_config(args.cfg_file)
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        local_rank = -1
        world_size = 1
    cfg.local_rank = local_rank
    cfg.world_size = world_size
    if local_rank == -1:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    cfg.device = device
    if not hasattr(cfg.TRAIN, "SEED"):
        cfg.TRAIN.SEED = 42
    set_seed(cfg.TRAIN.SEED)

    # initialize distributed process group
    if local_rank !=- 1:
        torch.distributed.init_process_group(backend='nccl')

    handlers = [logging.StreamHandler()]
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=handlers)
    logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
    logger = logging.getLogger(__name__)
    cfg.logger = logger
    cfg.logger.info(args)

    # load model
    model_name = cfg.MODEL.PATH
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoAdapterModel.from_pretrained(model_name)

    if not cfg.TRAIN.TWO_STAGE:
        model, adapter_name, head_name, best_f1 = train_single(cfg, model, tokenizer)
    else:
        model, adapter_name, head_name, best_f1 = train_two_stage(cfg, model, tokenizer)

    if cfg.local_rank in [-1, 0]:
        os.makedirs(os.path.join(cfg.OUTPUT.ADAPTER_SAVE_DIR, adapter_name), exist_ok=True)
        os.makedirs(os.path.join(cfg.OUTPUT.HEAD_SAVE_DIR, head_name), exist_ok=True)
        get(cfg, model).save_adapter(os.path.join(cfg.OUTPUT.ADAPTER_SAVE_DIR, adapter_name), adapter_name)
        get(cfg, model).save_head(os.path.join(cfg.OUTPUT.HEAD_SAVE_DIR, head_name), head_name)
        cfg.logger.info("Best model saved")