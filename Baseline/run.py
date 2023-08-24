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

adapter_dir = "adapter/"

def run(cfg):
    # reset random seed
    set_seed(cfg.TRAIN.SEED)

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

        print(cfg.OUTPUT.RESULT_SAVE_DIR)
        # if cfg.LOSSES.NAME == "CE_MS":
        #     print("Lambda:", cfg.LOSSES.LAMBDA_DISC, cfg.LOSSES.LAMBDA_CLUS)
        # if args.scale_pos_w is not None or args.scale_neg_w is not None:
        #     print("MS Scale Weight:", args.scale_pos_w, args.scale_neg_w)
        print(f"Best f1 on validation: {best_f1}")
        seqeval = evaluate.load('evaluate-metric/seqeval')
        results = seqeval.compute(predictions=predictions, references=references)
        print(results)
        return model, adapter_name, head_name, best_f1, results['overall_f1']
    else:
        return None, None, None, 0, 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", type=str, default="configs/para/transfer_learning_ms.yaml")
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--datasets", type=str, default=None)
    parser.add_argument("--vanilla", default=False, action="store_true")
    parser.add_argument("--src_lambda", type=float, default=None)
    parser.add_argument("--tgt_lambda", type=float, default=None)
    parser.add_argument("--src_dataset", type=str, default=None)
    parser.add_argument("--tgt_dataset", type=str, default=None)
    parser.add_argument("--scale_pos_w", type=float, default=None)
    parser.add_argument("--scale_neg_w", type=float, default=None)
    parser.add_argument("--two_stage_train", default=False, action="store_true")
    parser.add_argument("--tune_tgt_ms", default=False, action="store_true")
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

    suffix = ""
    if args.vanilla:
        cfg.LOSSES.MULTI_SIMILARITY_LOSS.VANILLA = True
        suffix += "/vanilla"
    if args.method is not None:
        if hasattr(cfg.DATA, "BIOMEDICAL"):
            cfg.DATA.BIOMEDICAL.SIM_METHOD = args.method
        if hasattr(cfg.DATA, "CROSSNER"):
            cfg.DATA.CROSSNER.SIM_METHOD = args.method
        suffix += f"/{args.method}"
    if args.datasets is not None:
        cfg.DATA.BIOMEDICAL.DATASETS = args.datasets.split('-')
    if args.src_dataset is not None:
        cfg.DATA.SRC_DATASET = args.src_dataset
    if args.tgt_dataset is not None:
        cfg.DATA.TGT_DATASET = args.tgt_dataset

    if args.src_lambda is not None:
        cfg.SRC_LOSS.LAMBDA = args.src_lambda
        suffix += f"/src-lambda={args.src_lambda}"
    if args.tgt_lambda is not None:
        cfg.TGT_LOSS.LAMBDA = args.tgt_lambda
        suffix += f"/tgt_lambda={args.tgt_lambda}"
    # if args.scale_pos_w is not None:
    #     cfg.LOSSES.MULTI_SIMILARITY_LOSS.SCALE_POS_WEIGHT = args.scale_pos_w
    # if args.scale_neg_w is not None:
    #     cfg.LOSSES.MULTI_SIMILARITY_LOSS.SCALE_NEG_WEIGHT = args.scale_neg_w

    suffix += f"/{cfg.DATA.SRC_DATASET}_{args.datasets}/{cfg.DATA.TGT_DATASET}"
    cfg.OUTPUT.ADAPTER_SAVE_DIR = '/'.join(cfg.OUTPUT.ADAPTER_SAVE_DIR.split('/')[:-2]) + suffix
    cfg.OUTPUT.HEAD_SAVE_DIR = '/'.join(cfg.OUTPUT.HEAD_SAVE_DIR.split('/')[:-2]) + suffix
    cfg.OUTPUT.RESULT_SAVE_DIR = '/'.join(cfg.OUTPUT.RESULT_SAVE_DIR.split('/')[:-2]) + suffix

    if args.tune_tgt_ms:
        res = {}
        best_valid_f1 = 0
        best_model, best_adapter_name, best_head_name = None, None, None
        for lambda_disc in np.arange(0, 0.45, 0.05):
            cfg.TGT_LOSS.LAMBDA = lambda_disc
            valid_f1s, test_f1s = [], []
            for seed in [13, 26, 42, 87, 100]:
                cfg.TRAIN.SEED = seed
                model, adapter_name, head_name, valid_f1, test_f1 = run(cfg)
                valid_f1s.append(valid_f1)
                test_f1s.append(test_f1)
            if cfg.local_rank in [-1, 0]:
                res[lambda_disc] = [np.mean(valid_f1s), np.mean(test_f1s), valid_f1s, test_f1s]
                if best_model is None or best_valid_f1 < np.mean(valid_f1s):
                    best_valid_f1 = np.mean(valid_f1s)
                    best_model = model
                    best_adapter_name = adapter_name
                    best_head_name = head_name

        if cfg.local_rank in [-1, 0]:
            os.makedirs(os.path.join(cfg.OUTPUT.ADAPTER_SAVE_DIR, best_adapter_name), exist_ok=True)
            os.makedirs(os.path.join(cfg.OUTPUT.HEAD_SAVE_DIR, best_head_name), exist_ok=True)
            best_model.save_adapter(os.path.join(cfg.OUTPUT.ADAPTER_SAVE_DIR, best_adapter_name), best_adapter_name)
            best_model.save_head(os.path.join(cfg.OUTPUT.HEAD_SAVE_DIR, best_head_name), best_head_name)
            cfg.logger.info("Best model saved")
            with open("results/para/eg/tgt_disc.json", "w") as f:
                json.dump(res, f)

            plt.clf()
            plt.plot([float(x) for x in res.keys()], [float(x[0]) for x in res.values()], label="Validation")
            plt.plot([float(x) for x in res.keys()], [float(x[1]) for x in res.values()], label="Test")
            plt.axhline(float(list(res.values())[0][1]), linestyle='--', label="Direct Transfer")
            plt.legend()
            plt.xlabel(r"$\lambda$")
            plt.ylabel(r"F1 Score")

            best = np.argmax([float(x[0]) for x in res.values()])
            print(list(res.keys())[best], list(res.values())[best][0], list(res.values())[best][1])

            plt.savefig("results/para/eg/disc_result.png", dpi=300)

    elif args.two_stage_train:
        valid_f1s, test_f1s = [], []
        model, adapter_name, head_name, valid_f1, test_f1 = run(cfg)
        if cfg.local_rank in [-1, 0]:   # seend 42
            valid_f1s.append(valid_f1)
            test_f1s.append(test_f1)

        cfg.TRAIN.TWO_STAGE = False
        cfg.ADAPTER.TRAIN = cfg.OUTPUT.ADAPTER_SAVE_DIR + "/biomedical_ner_bert-base-uncased_inter"
        for seed in [13, 26, 87, 100]:
            cfg.TRAIN.SEED = seed
            model, adapter_name, head_name, valid_f1, test_f1 = run(cfg)
            if cfg.local_rank in [-1, 0]:
                valid_f1s.append(valid_f1)
                test_f1s.append(test_f1)
        if cfg.local_rank in [-1, 0]:
            print(f'{np.mean(valid_f1s)}, {np.mean(test_f1s)},')
            print(f'{valid_f1s}, {test_f1s}')
    
    else:
        run(cfg)