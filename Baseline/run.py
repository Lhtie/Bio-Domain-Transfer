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
import evaluate

from train import train_two_stage
from utils.trainer import get
from utils.config import read_config, get_tgt_dataset, get_src_dataset, set_seed

adapter_dir = "adapter/"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", type=str, default="configs/para/transfer_learning_ms.yaml")
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--datasets", type=str, default=None)
    parser.add_argument("--vanilla", default=False, action="store_true")
    parser.add_argument("--lambda_disc", type=float, default=None)
    parser.add_argument("--lambda_clus", type=float, default=None)
    parser.add_argument("--src_dataset", type=str, default=None)
    parser.add_argument("--tgt_dataset", type=str, default=None)
    # parser.add_argument("--scale_pos", type=int)
    # parser.add_argument("--scale_neg", type=int)
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

    # cfg.LOSSES.MULTI_SIMILARITY_LOSS.SCALE_POS_WEIGHT = args.scale_pos
    # cfg.LOSSES.MULTI_SIMILARITY_LOSS.SCALE_NEG_WEIGHT = args.scale_neg
    # cfg.OUTPUT.ADAPTER_SAVE_DIR += f"/{args.scale_pos}_{args.scale_neg}"
    # cfg.OUTPUT.HEAD_SAVE_DIR += f"/{args.scale_pos}_{args.scale_neg}"
    # cfg.OUTPUT.RESULT_SAVE_DIR += f"/{args.scale_pos}_{args.scale_neg}"
    suffix = ""
    if args.vanilla:
        cfg.LOSSES.MULTI_SIMILARITY_LOSS.VANILLA = True
        suffix += "/vanilla"
    if args.method is not None:
        cfg.DATA.BIOMEDICAL.SIM_METHOD = args.method
        suffix += f"/{args.method}"
    if args.datasets is not None:
        cfg.DATA.BIOMEDICAL.DATASETS = args.datasets.split('-')
        suffix += f"/{args.datasets}"
    if args.lambda_disc is not None:
        cfg.LOSSES.LAMBDA_DISC = args.lambda_disc
    if args.lambda_clus is not None:
        cfg.LOSSES.LAMBDA_CLUS = args.lambda_clus
    if args.src_dataset is not None:
        cfg.DATA.SRC_DATASET = args.src_dataset
        suffix += f"/{args.src_dataset}"
    if args.tgt_dataset is not None:
        cfg.DATA.TGT_DATASET = args.tgt_dataset
        suffix += f"/{args.tgt_dataset}"
    cfg.OUTPUT.ADAPTER_SAVE_DIR += suffix
    cfg.OUTPUT.HEAD_SAVE_DIR += suffix
    cfg.OUTPUT.RESULT_SAVE_DIR += suffix

    # load model
    model_name = cfg.MODEL.PATH
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoAdapterModel.from_pretrained(model_name)
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

        print(cfg.OUTPUT.RESULT_SAVE_DIR, args.lambda_disc, args.lambda_clus)
        print(f"Best f1 on validation: {best_f1}")
        seqeval = evaluate.load('evaluate-metric/seqeval')
        results = seqeval.compute(predictions=predictions, references=references)
        print(results)