import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import copy
import pickle

from tqdm import tqdm
from .multi_similarity_loss import MultiSimilarityLoss
from .multi_similarity_loss import extract_feat
from .metric import *

def train(cfg, model, tokenizer, train_dataloader, dev_dataloader, adapter_name, head_name, use_ms=False, pretrain=False):
    ce_loss_fn = nn.CrossEntropyLoss(reduction='none')
    if use_ms:
        get(cfg, model).active_head = None
        cfg.data.sim_weight = cfg.data.sim_weight.to(cfg.device)
        ms_loss_fn = [
            MultiSimilarityLoss(cfg),
            MultiSimilarityLoss(cfg)
        ]
        
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

    best_f1, best_epoch, best_model = -1, -1, None
    best_epoch = torch.tensor(best_epoch).to(cfg.device)
    epoch = 0
    pos_pair_thresh, neg_pair_thresh, pos_pair_w, neg_pair_w = [], [], [], []
    while True:
        if cfg.local_rank in [-1, 0]:
            cfg.logger.info(f"Epoch: {epoch}")
        # train
        model.train()
        ce_losses, ms_disc_losses, ms_clus_losses, losses = [], [], [], []

        if use_ms:
            ms_loss_fn[0].reset()
            ms_loss_fn[1].reset()

        for i, batch in enumerate(train_dataloader):
            batch = {k: v.to(cfg.device) for k, v in batch.items()}

            outputs = model(batch["input_ids"])
            if use_ms:
                feat_disc, feat_clus = extract_feat(
                    outputs[0], 
                    batched_label_mask=batch["label_mask"],
                    batched_token_id=batch['token_id'],
                    batched_token_label=batch['ner_tags'],
                    K=cfg.data.K,
                    clusters=cfg.data.clusters
                )
                ids, feats, labels = feat_disc.tensorize()
                ms_disc_loss = ms_loss_fn[0](
                    feats,
                    labels,
                    sim_weight=cfg.data.sim_weight[ids][:, ids]
                )
                ids, feats, labels = feat_clus.tensorize()
                ms_clus_loss = ms_loss_fn[1](
                    feats,
                    labels,
                    sim_weight=cfg.data.sim_weight[ids][:, ids]
                )
                outputs = get(cfg, model).forward_head(
                    (outputs[0],) + outputs[2:],
                    head_name=head_name,
                    pooled_output=outputs[1]
                )

            # we need to reformat the tensors for the loss function
            # they need to have the shape (N, C) and (N,) where N is the number
            # of tokens and C the number of classes
            predictions = torch.flatten(outputs[0], 0, 1)
            expected = torch.flatten(batch["labels"].long(), 0, 1)
            label_mask = torch.flatten(batch["label_mask"].long(), 0, 1)
            ce_loss = (ce_loss_fn(predictions, expected) * label_mask).mean()
            
            if use_ms:
                loss = ce_loss + cfg.LOSSES.LAMBDA_DISC * ms_disc_loss + cfg.LOSSES.LAMBDA_CLUS * ms_clus_loss
                ms_disc_losses.append(ms_disc_loss.item())
                ms_clus_losses.append(ms_clus_loss.item())
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
                cfg.logger.info(f"ce:{np.mean(ce_losses)}")
                cfg.logger.info(f"ms_disc:{np.mean(ms_disc_losses)}")
                cfg.logger.info(f"ms_clus:{np.mean(ms_clus_losses)}")
            else:
                cfg.logger.info(f"loss: {np.mean(losses)}")

        if epoch % 10 == 0 and use_ms:
            pos_pair_thresh.append(torch.concat(ms_loss_fn[0].pos_pair_thresh).cpu().detach())
            neg_pair_thresh.append(torch.concat(ms_loss_fn[0].neg_pair_thresh).cpu().detach())
            pos_pair_w.append(torch.concat(ms_loss_fn[0].pos_pair_w).cpu().detach())
            neg_pair_w.append(torch.concat(ms_loss_fn[0].neg_pair_w).cpu().detach())
            # dist.barrier()
            with open(f"results/quantize-{cfg.local_rank}.pt", "wb") as f:
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
                        head_name=head_name,
                        pooled_output=outputs[1]
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
            cfg.logger.info(f"Eval f1: {f1}")

            if best_f1 == -1 or best_f1 < f1:
                best_f1 = f1
                best_epoch = torch.tensor(epoch).to(cfg.device)
                best_model = copy.deepcopy(model)
                cfg.logger.info(f"Best epoch: {best_epoch}, Best f1: {best_f1}")
        if cfg.local_rank != -1:
            dist.broadcast(best_epoch, src=0)
        # stop condition
        if epoch >= cfg.TRAIN.EPOCHS and epoch - best_epoch >= 10:
            if cfg.local_rank in [-1, 0]:
                cfg.logger.info(f"Best checkpoint at {best_epoch} epoch and stopped")
            break
        
        epoch += 1
    return best_model

def get(cfg, model):
    if cfg.local_rank == -1:
        return model
    else:
        return model.module

def collator_padding(batchs):
    for batch in batchs:
        for key in batch.keys():
            batch[key] = F.pad(torch.tensor(batch[key]), pad=(0, 512 - len(batch[key])), mode='constant', value=0)
    return batchs