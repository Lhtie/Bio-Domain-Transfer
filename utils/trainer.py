import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import os
import copy
import pickle

from tqdm import tqdm
from .multi_similarity_loss import MultiSimilarityLoss
from .multi_similarity_loss import extract_feat
from .metric import *
from .config import get_src_dataset, get_tgt_dataset

def train(cfg, model, tokenizer, train_dataloader, dev_dataloader, adapter_name, head_name, 
            use_ms=False, pretrain=False, early_stop=True):
    ce_loss_fn = nn.CrossEntropyLoss(reduction='none')
    if use_ms:
        model.active_head = None
        ms_loss_fn = MultiSimilarityLoss(cfg, "src" if pretrain else "tgt")
        # determine whether use clusters
        use_clus = pretrain and cfg.data.sim_method is not None and cfg.data.sim_method.split('-')[1] == "clus"
        # prepare sim weight matrix
        if pretrain and hasattr(cfg.SRC_LOSS.MULTI_SIMILARITY_LOSS, "VANILLA") and \
            not cfg.SRC_LOSS.MULTI_SIMILARITY_LOSS.VANILLA:
            sim_weight = cfg.data.sim_weight.to(cfg.device)
        else:
            sim_weight = None
        if use_clus:
            cfg.SRC_LOSS.MULTI_SIMILARITY_LOSS.VANILLA = True
        
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                        "weight_decay": 1e-5,
                    },
                    {
                        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                        "weight_decay": 0.0,
                    },
                ]
    optimizer = torch.optim.AdamW(params=optimizer_grouped_parameters, 
                                  lr=cfg.TRAIN.LR if not pretrain else cfg.TRAIN.SRC_LR)

    if cfg.local_rank != -1:
        model = torch.nn.parallel.distributed.DistributedDataParallel(model, device_ids=[cfg.local_rank], output_device=cfg.local_rank)

    f1, best_f1, best_epoch, best_model = -1, -1, -1, None
    best_epoch = torch.tensor(best_epoch).to(cfg.device)
    epoch = 0
    pos_pair_thresh, neg_pair_thresh, pos_pair_w, neg_pair_w = [], [], [], []
    valid_f1s = []
    while True:
        if cfg.local_rank in [-1, 0]:
            cfg.logger.info(f"Epoch: {epoch}")
        # train
        model.train()
        if use_ms:
            ms_loss_fn.reset()

        losses, ce_losses, ms_losses = [], [], []
        for i, batch in enumerate(train_dataloader):
            batch = {k: v.to(cfg.device) for k, v in batch.items()}

            outputs = model(batch["input_ids"])
            if use_ms:
                feat_obj = extract_feat(
                    outputs[0],
                    batched_label_mask=batch["label_mask"],
                    batched_token_id=batch['token_id'],
                    batched_token_label=batch['ner_tags'],
                    pseudo_labels=batch['pse_tags'] 
                        if cfg.data.__class__.__name__.endswith("_pse") else None,
                    cluster_map=cfg.data.clusters if pretrain and use_clus else None
                )
                ids, feats, labels = feat_obj.tensorize()
                ms_loss = ms_loss_fn(
                    feats, 
                    labels,
                    sim_weight=sim_weight[ids][:, ids] \
                        if sim_weight is not None else None
                )
                outputs = get(cfg, model).forward_head(
                    (outputs[0],) + outputs[2:],
                    head_name=head_name
                )

            # we need to reformat the tensors for the loss function
            # they need to have the shape (N, C) and (N,) where N is the number
            # of tokens and C the number of classes
            predictions = torch.flatten(outputs[0], 0, 1)
            expected = torch.flatten(batch["labels"].long(), 0, 1)
            label_mask = torch.flatten(batch["label_mask"].long(), 0, 1)
            ce_loss = (ce_loss_fn(predictions, expected) * label_mask).mean()
        
            if use_ms:
                if pretrain:
                    loss = ce_loss + cfg.SRC_LOSS.LAMBDA * ms_loss
                else:
                    loss = ce_loss + cfg.TGT_LOSS.LAMBDA * ms_loss
                ms_losses.append(ms_loss.item())
            else:
                loss = ce_loss
            ce_losses.append(ce_loss.item())
            losses.append(loss.item())
                
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if cfg.local_rank in [-1, 0]:
            if use_ms:
                cfg.logger.info(f"loss: {np.mean(losses)}")
                cfg.logger.info(f"ce: {np.mean(ce_losses)}; ms: {np.mean(ms_losses)}")
            else:
                cfg.logger.info(f"loss: {np.mean(losses)}")

        if epoch % 10 == 0 and use_ms and pretrain:
            pos_pair_thresh.append(torch.concat(ms_loss_fn.pos_pair_thresh).cpu().detach())
            neg_pair_thresh.append(torch.concat(ms_loss_fn.neg_pair_thresh).cpu().detach())
            if not ms_loss_fn.vanilla:
                pos_pair_w.append(torch.concat(ms_loss_fn.pos_pair_w).cpu().detach())
                neg_pair_w.append(torch.concat(ms_loss_fn.neg_pair_w).cpu().detach())
            # dist.barrier()
            os.makedirs("results/quantize", exist_ok=True)
            with open(f"results/quantize/{cfg.local_rank}.pt", "wb") as f:
                pickle.dump((
                    pos_pair_thresh,
                    neg_pair_thresh,
                    pos_pair_w,
                    neg_pair_w
                ), f)
        
        # eval on dev
        model.eval()
        predictions, references = [], []
        for i, batch in enumerate(dev_dataloader):
            batch = {k: v.to(cfg.device) for k, v in batch.items()}

            with torch.no_grad():
                if use_ms:
                    outputs = model(batch["input_ids"])
                    preds = get(cfg, model).forward_head(
                        (outputs[0],) + outputs[2:],
                        head_name=head_name
                    )[0]
                else:
                    preds = model(batch["input_ids"]).logits
                preds = preds.detach().cpu().numpy()
                preds = np.argmax(preds, axis=2)
            for label_mask, pred, ref in zip(batch["label_mask"], preds, batch["labels"]):
                predictions.append([cfg.data.id2label[id.item()] for mask, id in zip(label_mask, pred) if mask == 1])
                references.append([cfg.data.id2label[id.item()] for mask, id in zip(label_mask, ref) if mask == 1])

        pred_sum, tp_sum, true_sum = extract_tp_actual_correct(references, predictions)
        pred_sum = torch.tensor(pred_sum).to(cfg.device)
        tp_sum = torch.tensor(tp_sum).to(cfg.device)
        true_sum = torch.tensor(true_sum).to(cfg.device)
        if cfg.local_rank != -1:
            dist.reduce(pred_sum, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(tp_sum, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(true_sum, dst=0, op=dist.ReduceOp.SUM)
        if cfg.local_rank in [-1, 0]:
            if pred_sum.sum() == 0 or tp_sum.sum() == 0 or true_sum.sum() == 0:
                precision, recall, f1 = 0, 0, 0
            else:
                precision = tp_sum.sum() / pred_sum.sum()
                recall = tp_sum.sum() / true_sum.sum()
                f1 = 2. * precision * recall / (precision + recall)
                f1 = f1.item()
            cfg.logger.info(f"Eval f1: {f1}")
            valid_f1s.append(f1)

            if best_f1 == -1 or best_f1 < f1:
                best_f1 = f1
                best_epoch = torch.tensor(epoch).to(cfg.device)
                best_model = copy.deepcopy(model)
                cfg.logger.info(f"Best epoch: {best_epoch}, Best f1: {best_f1}")
        if cfg.local_rank != -1:
            dist.broadcast(best_epoch, src=0)
        # stop condition
        if epoch >= cfg.TRAIN.EPOCHS and epoch - best_epoch >= 20:
            if cfg.local_rank in [-1, 0]:
                if early_stop:
                    cfg.logger.info(f"Best checkpoint at {best_epoch} epoch and stopped")
                else:
                    cfg.logger.info(f"Best checkpoint at {epoch} epoch and stopped")
            break
        
        epoch += 1
    if early_stop:
        return get(cfg, best_model), best_f1, valid_f1s
    else:
        return get(cfg, model), f1, valid_f1s

def get(cfg, model):
    if model is None:
        return None
    if cfg.local_rank == -1:
        return model
    else:
        return model.module

def collator_padding(batchs):
    for batch in batchs:
        for key in batch.keys():
            batch[key] = F.pad(torch.tensor(batch[key]), pad=(0, 512 - len(batch[key])), mode='constant', value=0)
    return batchs