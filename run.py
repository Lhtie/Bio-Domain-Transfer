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
import json
import numpy as np
import logging
import evaluate
import matplotlib.pyplot as plt

from train import train_two_stage, train_single
from utils.trainer import get
from utils.config import read_config, get_tgt_dataset, get_src_dataset, set_seed

def modify_configs(cfg, args):
    if args.model_path is not None:
        cfg.MODEL.PATH = args.model_path
        cfg.MODEL.BACKBONE = args.model_path.split("/")[-1]
    if args.seed is not None:
        cfg.TRAIN.SEED = args.seed
    if args.method is not None:
        cfg.DATA.BIOMEDICAL.SIM_METHOD = args.method
    if args.datasets is not None:
        cfg.DATA.BIOMEDICAL.DATASETS = args.datasets.split('-')
    if args.src_dataset is not None:
        cfg.DATA.SRC_DATASET = args.src_dataset
    if args.tgt_dataset is not None:
        cfg.DATA.TGT_DATASET = args.tgt_dataset
    if args.src_lambda is not None:
        cfg.SRC_LOSS.LAMBDA = args.src_lambda
    if args.tgt_lambda is not None:
        cfg.TGT_LOSS.LAMBDA = args.tgt_lambda
    if args.src_vanilla:
        cfg.SRC_LOSS.MULTI_SIMILARITY_LOSS.VANILLA = True
        
    suffix = ""
    if hasattr(cfg.DATA, "BIOMEDICAL") and hasattr(cfg.DATA.BIOMEDICAL, "SIM_METHOD"):
        if cfg.DATA.BIOMEDICAL.SIM_METHOD != "None":
            suffix += f"/{cfg.DATA.BIOMEDICAL.SIM_METHOD}"
    if hasattr(cfg, "SRC_LOSS") and hasattr(cfg.SRC_LOSS, "LAMBDA"):
        suffix += f"/src-lambda={cfg.SRC_LOSS.LAMBDA}"
    if hasattr(cfg.DATA, "SRC_DATASET"):
        suffix += f"/{cfg.DATA.SRC_DATASET}"
        if hasattr(cfg.DATA, "BIOMEDICAL") and hasattr(cfg.DATA.BIOMEDICAL, "DATASETS"):
            if cfg.DATA.BIOMEDICAL.DATASETS != "None":
                ds = '_'.join(cfg.DATA.BIOMEDICAL.DATASETS)
                suffix += "_" + ds
    suffix += f"/{cfg.DATA.TGT_DATASET}"
    
    cfg.OUTPUT.ADAPTER_SAVE_DIR += suffix
    cfg.OUTPUT.HEAD_SAVE_DIR += suffix
    cfg.OUTPUT.RESULT_SAVE_DIR += suffix
    return cfg

def run(cfg):
    # reset random seed
    set_seed(cfg.TRAIN.SEED)

    # load model
    model_name = cfg.MODEL.PATH
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoAdapterModel.from_pretrained(model_name)
    if not cfg.TRAIN.TWO_STAGE:
        model, adapter_name, head_name, best_f1, valid_f1s = train_single(cfg, model, tokenizer)
    else:
        model, adapter_name, head_name, best_f1, valid_f1s = train_two_stage(cfg, model, tokenizer)
    if cfg.local_rank in [-1, 0]:
        os.makedirs(os.path.join(cfg.OUTPUT.ADAPTER_SAVE_DIR, adapter_name), exist_ok=True)
        os.makedirs(os.path.join(cfg.OUTPUT.HEAD_SAVE_DIR, head_name), exist_ok=True)
        model.save_adapter(os.path.join(cfg.OUTPUT.ADAPTER_SAVE_DIR, adapter_name), adapter_name)
        model.save_head(os.path.join(cfg.OUTPUT.HEAD_SAVE_DIR, head_name), head_name)
        cfg.logger.info("Best model saved")

    if cfg.local_rank in [-1, 0]:
        data = get_tgt_dataset(cfg)
        dataset = data.load(tokenizer)
        print(dataset)
        dataloader = torch.utils.data.DataLoader(dataset["evaluation"], batch_size=cfg.EVAL.BATCH_SIZE)

        model = AutoAdapterModel.from_pretrained(model_name)
        model.load_adapter(os.path.join(cfg.OUTPUT.ADAPTER_SAVE_DIR, adapter_name))
        model.set_active_adapters([adapter_name])
        model.load_head(os.path.join(cfg.OUTPUT.HEAD_SAVE_DIR, head_name))
        
        # predict
        model.to(cfg.device).eval()
        predictions, references = [], []
        for batch in tqdm(dataloader):
            batch = {k: v.to(cfg.device) for k, v in batch.items()}

            with torch.no_grad():
                preds = model(batch["input_ids"]).logits
                preds = preds.detach().cpu().numpy()
                preds = np.argmax(preds, axis=2)
            
            for label_mask, pred, ref in zip(batch["label_mask"], preds, batch["labels"]):
                predictions.append([data.id2label[id.item()] for mask, id in zip(label_mask, pred) if mask == 1])
                references.append([data.id2label[id.item()] for mask, id in zip(label_mask, ref) if mask == 1])
                
        # clear OOD
        for idx, pred in enumerate(predictions):
            predictions[idx] = ["O" if p.endswith("OOD") else p for p in pred]

        print(cfg.OUTPUT.RESULT_SAVE_DIR)
        print(f"Best f1 on validation: {best_f1}")
        seqeval = evaluate.load('evaluate-metric/seqeval')
        # seqeval = evaluate.load('../seqeval')
        results = seqeval.compute(predictions=predictions, references=references)
        print(results)
        
        os.makedirs(os.path.join(cfg.OUTPUT.RESULT_SAVE_DIR, "validation_f1s_rec"), exist_ok=True)
        tgt_lambda = 0
        if hasattr(cfg, "TGT_LOSS") and hasattr(cfg.TGT_LOSS, "LAMBDA"):
            tgt_lambda = cfg.TGT_LOSS.LAMBDA
        with open(os.path.join(cfg.OUTPUT.RESULT_SAVE_DIR, "validation_f1s_rec", 
                               f"tgt-lambda={tgt_lambda}_seed={cfg.TRAIN.SEED}.json"), "w") as f:
            json.dump(valid_f1s, f, indent=4)
        return model, adapter_name, head_name, best_f1, results['overall_f1'], results['overall_precision'], results['overall_recall']
    else:
        return None, None, None, 0, 0, 0, 0
    
def ensemble(cfg):
    # reset random seed
    set_seed(cfg.TRAIN.SEED)

    # load model
    model_name = cfg.MODEL.PATH
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = AutoAdapterModel.from_pretrained(model_name)
    cfg_m = copy.deepcopy(cfg)
    cfg_m.DATA.TGT_DATASET = cfg.DATA.TGT_DATASET.split('_')[0]
    model, _, _, best_f1, _ = train_single(cfg_m, model, tokenizer)
    
    # reset random seed
    set_seed(cfg.TRAIN.SEED)
    
    classifier = AutoAdapterModel.from_pretrained(model_name)
    classifier, _, _, _, _ = train_single(cfg, classifier, tokenizer)

    if cfg.local_rank in [-1, 0]:
        data = get_tgt_dataset(cfg_m)
        dataset = data.load(tokenizer)
        print(dataset)
        dataloader = torch.utils.data.DataLoader(dataset["evaluation"], batch_size=cfg.EVAL.BATCH_SIZE)
        
        # predict
        model.to(cfg.device).eval()
        classifier.to(cfg.device).eval()
        predictions, references = [], []
        for batch in tqdm(dataloader):
            batch = {k: v.to(cfg.device) for k, v in batch.items()}

            with torch.no_grad():
                preds = model(batch["input_ids"]).logits
                preds = preds.detach().cpu().numpy()
                preds = np.argmax(preds, axis=2)
                
                clses = classifier(batch["input_ids"]).logits
                clses = clses.detach().cpu().numpy()
                clses = np.argmax(clses, axis=2)
            
            for label_mask, pred, ref, cls in zip(batch["label_mask"], preds, batch["labels"], clses):
                predictions.append([data.id2label[id.item()] if c < 3 else "O" for mask, id, c in zip(label_mask, pred, cls) if mask == 1])
                references.append([data.id2label[id.item()] for mask, id in zip(label_mask, ref) if mask == 1])
                
        seqeval = evaluate.load('evaluate-metric/seqeval')
        results = seqeval.compute(predictions=predictions, references=references)
        print(results)
        
        return best_f1, results['overall_f1'], results['overall_precision'], results['overall_recall']
    else:
        return 0, 0, 0, 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", type=str, default="configs/para/transfer_learning.yaml")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--src_lambda", type=float, default=None)
    parser.add_argument("--tgt_lambda", type=float, default=None)
    parser.add_argument("--src_vanilla", default=False, action="store_true")
    parser.add_argument("--src_dataset", type=str, default=None)
    parser.add_argument("--datasets", type=str, default=None)
    parser.add_argument("--tgt_dataset", type=str, default=None)
    parser.add_argument("--tune_src", default=False, action="store_true")
    parser.add_argument("--tune_tgt", default=False, action="store_true")
    parser.add_argument("--ensemble", default=False, action="store_true")
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

    if args.tune_tgt:
        res = {}
        arange = np.arange(0.10, 0.35, 0.05)
        for lambda_disc in arange:
            args.tgt_lambda = lambda_disc
            cfg_m = modify_configs(copy.deepcopy(cfg), args)
            cfg_m.ADAPTER.TRAIN = os.path.join(
                os.path.dirname(cfg_m.OUTPUT.ADAPTER_SAVE_DIR),
                cfg_m.DATA.SRC_DATASET + "_ner_" + cfg_m.MODEL.BACKBONE + "_inter"
            )
            valid_f1s, test_f1s, test_precs, test_recs = [], [], [], []
            for seed in [13, 42, 87]:
                cfg_m.TRAIN.SEED = seed
                model, adapter_name, head_name, valid_f1, test_f1, test_prec, test_rec = run(cfg_m)
                valid_f1s.append(valid_f1)
                test_f1s.append(test_f1)
                test_precs.append(test_prec)
                test_recs.append(test_rec)
            if cfg_m.local_rank in [-1, 0]:
                res[lambda_disc] = {
                    "validation": [
                        np.mean(valid_f1s),
                        valid_f1s
                    ],
                    "test": {
                        "prec": [np.mean(test_precs), np.std(test_precs), test_precs],
                        "rec": [np.mean(test_recs), np.std(test_recs), test_recs],
                        "f1": [np.mean(test_f1s), np.std(test_f1s), test_f1s]
                    }
                }
                
                os.makedirs(cfg_m.OUTPUT.RESULT_SAVE_DIR, exist_ok=True)
                with open(os.path.join(cfg_m.OUTPUT.RESULT_SAVE_DIR, "tgt_disc.json"), "w") as f:
                    json.dump(res, f, indent=4)

                plt.clf()
                plt.plot([float(x) for x in res.keys()], [float(x["validation"][0]) for x in res.values()], label="Validation")
                plt.plot([float(x) for x in res.keys()], [float(x["test"]["f1"][0]) for x in res.values()], label="Test")
                plt.axhline(float(list(res.values())[0]["test"]["f1"][0]), linestyle='--', label="Direct Transfer")
                plt.legend()
                plt.xlabel(r"$\lambda$")
                plt.ylabel(r"F1 Score")

                plt.savefig(os.path.join(cfg_m.OUTPUT.RESULT_SAVE_DIR, "tgt_disc.png"), dpi=300)

    elif args.tune_src:
        res = {}
        for lambda_eg in [0.6, 0.8, 1.0, 1.2, 1.4]:
            args.src_lambda = lambda_eg
            cfg_m = modify_configs(copy.deepcopy(cfg), args)
            cfg_m.ADAPTER.TRAIN = os.path.join(
                os.path.dirname(cfg_m.OUTPUT.ADAPTER_SAVE_DIR),
                cfg_m.DATA.SRC_DATASET + "_ner_" + cfg_m.MODEL.BACKBONE + "_inter"
            )
            
            valid_f1s, test_f1s, test_precs, test_recs = [], [], [], []
            if not os.path.exists(cfg_m.ADAPTER.TRAIN):
                cfg_m.TRAIN.SEED = 42
                model, adapter_name, head_name, valid_f1, test_f1, test_prec, test_rec = run(cfg_m)
                if cfg_m.local_rank in [-1, 0]:
                    valid_f1s.append(valid_f1)
                    test_f1s.append(test_f1)
                    test_precs.append(test_prec)
                    test_recs.append(test_rec)
                seeds = [13, 87]
            else:
                seeds = [42, 13, 87]

            cfg_m.TRAIN.TWO_STAGE = False
            for seed in seeds:
                cfg_m.TRAIN.SEED = seed
                model, adapter_name, head_name, valid_f1, test_f1, test_prec, test_rec = run(cfg_m)
                if cfg_m.local_rank in [-1, 0]:
                    valid_f1s.append(valid_f1)
                    test_f1s.append(test_f1)
                    test_precs.append(test_prec)
                    test_recs.append(test_rec)
            if cfg_m.local_rank in [-1, 0]:
                res[lambda_eg] = {
                    "validation": [
                        np.mean(valid_f1s),
                        valid_f1s
                    ],
                    "test": {
                        "prec": [np.mean(test_precs), np.std(test_precs), test_precs],
                        "rec": [np.mean(test_recs), np.std(test_recs), test_recs],
                        "f1": [np.mean(test_f1s), np.std(test_f1s), test_f1s]
                    }
                }

                os.makedirs(cfg_m.OUTPUT.RESULT_SAVE_DIR, exist_ok=True)
                with open(os.path.join(cfg_m.OUTPUT.RESULT_SAVE_DIR, "src_lambda.json"), "w") as f:
                    json.dump(res, f, indent=4)

                plt.clf()
                plt.plot([float(x) for x in res.keys()], [float(x["validation"][0]) for x in res.values()], label="Validation")
                plt.plot([float(x) for x in res.keys()], [float(x["test"]["f1"][0]) for x in res.values()], label="Test")
                plt.axhline(float(list(res.values())[0]["test"]["f1"][0]), linestyle='--', label="Direct Transfer")
                plt.legend()
                plt.xlabel(r"$\lambda$")
                plt.ylabel(r"F1 Score")
                
                plt.savefig(os.path.join(cfg_m.OUTPUT.RESULT_SAVE_DIR, "src_lambda.png"), dpi=300)
    
    else:
        valid_f1s, test_f1s, test_precs, test_recs = [], [], [], []
        if args.seed is not None:
            seeds = [args.seed]
        else:
            seeds = [42, 13, 87]
        for seed in seeds:
            args.seed = seed
            cfg_m = modify_configs(copy.deepcopy(cfg), args)
            if hasattr(cfg_m.DATA, "SRC_DATASET"):
                cfg_m.ADAPTER.TRAIN = os.path.join(
                    os.path.dirname(cfg_m.OUTPUT.ADAPTER_SAVE_DIR),
                    cfg_m.DATA.SRC_DATASET + "_ner_" + cfg_m.MODEL.BACKBONE + "_inter"
                )
            if args.ensemble:
                valid_f1, test_f1, test_prec, test_rec = ensemble(cfg_m)
            else:
                model, adapter_name, head_name, valid_f1, test_f1, test_prec, test_rec = run(cfg_m)
            if cfg_m.local_rank in [-1, 0]:
                valid_f1s.append(valid_f1)
                test_f1s.append(test_f1)
                test_precs.append(test_prec)
                test_recs.append(test_rec)
                
        if cfg_m.local_rank in [-1, 0]:
            os.makedirs(cfg_m.OUTPUT.RESULT_SAVE_DIR, exist_ok=True)
            res = {
                "validation": [
                    np.mean(valid_f1s),
                    valid_f1s
                ],
                "test": {
                    "prec": [np.mean(test_precs), np.std(test_precs), test_precs],
                    "rec": [np.mean(test_recs), np.std(test_recs), test_recs],
                    "f1": [np.mean(test_f1s), np.std(test_f1s), test_f1s]
                }
            }
            with open(os.path.join(cfg_m.OUTPUT.RESULT_SAVE_DIR, "avg.json"), "w") as f:
                json.dump(res, f, indent=4)