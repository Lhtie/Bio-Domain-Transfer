import torch
from torch import nn

# Adapted from https://github.com/msight-tech/research-ms-loss/blob/master/ret_benchmark/losses/multi_similarity_loss.py
class MultiSimilarityLoss(nn.Module):
    def __init__(self, cfg):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.margin = 0.1

        self.scale_pos = cfg.LOSSES.MULTI_SIMILARITY_LOSS.SCALE_POS
        self.scale_neg = cfg.LOSSES.MULTI_SIMILARITY_LOSS.SCALE_NEG
        self.scale_pos_w = cfg.LOSSES.MULTI_SIMILARITY_LOSS.SCALE_POS_WEIGHT
        self.scale_neg_w = cfg.LOSSES.MULTI_SIMILARITY_LOSS.SCALE_NEG_WEIGHT
        self.vanila = False
        
        if hasattr(cfg.LOSSES.MULTI_SIMILARITY_LOSS, "VANILA"):
            self.vanila = cfg.LOSSES.MULTI_SIMILARITY_LOSS.VANILA

    def forward(self, feats, labels, sim_weight):
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        batch_size = feats.size(0)
        feats_norm = feats / feats.norm(dim=1)[:, None]
        sim_mat = torch.matmul(feats_norm, torch.t(feats_norm))

        epsilon = 1e-5
        loss = list()

        for i in range(batch_size):
            pos_pair_w = sim_weight[i][labels == labels[i]]
            pos_pair_ = sim_mat[i][labels == labels[i]]
            # pos_pair_w = pos_pair_w[pos_pair_ < 1 - epsilon]
            # pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]

            neg_pair_w = sim_weight[i][labels != labels[i]]
            neg_pair_ = sim_mat[i][labels != labels[i]]

            if len(pos_pair_) > 0:
                neg_pair_w = neg_pair_w[neg_pair_ + self.margin > min(pos_pair_)]
                neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_)]
            else:
                neg_pair = neg_pair_
            if len(neg_pair_) > 0:
                pos_pair_w = pos_pair_w[pos_pair_ - self.margin < max(neg_pair_)]
                pos_pair = pos_pair_[pos_pair_ - self.margin < max(neg_pair_)]
            else:
                pos_pair = pos_pair_

            if len(neg_pair) < 1 and len(pos_pair) < 1:
                continue

            # weighting step
            if self.vanila:
                pos_loss = 1.0 / self.scale_pos * torch.log(
                    1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh))))
                neg_loss = 1.0 / self.scale_neg * torch.log(
                    1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh))))
            else:
                pos_loss = 1.0 / self.scale_pos * torch.log(
                    1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh) + self.scale_pos_w * pos_pair_w)))
                neg_loss = 1.0 / self.scale_neg * torch.log(
                    1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh) - self.scale_neg_w * neg_pair_w)))
            loss.append(pos_loss + neg_loss)

        if len(loss) == 0:
            return torch.zeros([], requires_grad=True)

        loss = sum(loss) / batch_size
        return loss

def extract_feat(batched_feat, batched_label_mask, batched_token_id, batched_token_label):
    ids, feats, labels = [], [], []
    for feat, label_mask, token_id, token_label in zip(batched_feat, batched_label_mask, batched_token_id, batched_token_label):
        token_feat = []
        # use feature of first wordpiece to represent the token
        for feat_t, mask in zip(feat, label_mask):
            if mask == 1:
                token_feat.append(feat_t)
        token_feat = torch.stack(token_feat)

        # mean over token to represent entity
        ett_feat = {}
        token_num = len(token_feat)
        for feat, id, label in zip(token_feat, token_id[:token_num], token_label[:token_num]):
            if label > 0 and id != -1: # part of entity
                if not id in ett_feat:
                    ett_feat[id] = (label == 1 or label == 2, []) # chemmical entity
                ett_feat[id][1].append(feat)
        for token_id, (token_label, token_feats) in ett_feat.items():
            ids.append(token_id)
            feats.append(torch.mean(torch.stack(token_feats), dim=0))
            labels.append(token_label)

    if len(ids) == 0 or len(feats) == 0 or len(labels) == 0:
        return torch.tensor([], dtype=torch.int), torch.tensor([], dtype=torch.int), torch.tensor([], dtype=torch.int)
    else:
        return torch.stack(ids), torch.stack(feats), torch.stack(labels)