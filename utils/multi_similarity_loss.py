import torch
from torch import nn

# Adapted from https://github.com/msight-tech/research-ms-loss/blob/master/ret_benchmark/losses/multi_similarity_loss.py
class MultiSimilarityLoss(nn.Module):
    def __init__(self, cfg, stage="tgt"):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.margin = 0.1

        if stage == "src":
            loss_config = cfg.SRC_LOSS
        elif stage == "tgt":
            loss_config = cfg.TGT_LOSS
        else:
            raise NotImplementedError()
        self.scale_pos = loss_config.MULTI_SIMILARITY_LOSS.SCALE_POS
        self.scale_neg = loss_config.MULTI_SIMILARITY_LOSS.SCALE_NEG
        self.vanilla = True
        if hasattr(loss_config.MULTI_SIMILARITY_LOSS, "VANILLA"):
            self.vanilla = loss_config.MULTI_SIMILARITY_LOSS.VANILLA
        if not self.vanilla:
            self.scale_pos_w = loss_config.MULTI_SIMILARITY_LOSS.SCALE_POS_WEIGHT
            self.scale_neg_w = loss_config.MULTI_SIMILARITY_LOSS.SCALE_NEG_WEIGHT

    def reset(self):
        self.pos_pair_thresh = []
        self.neg_pair_thresh = []
        self.pos_pair_w = []
        self.neg_pair_w = []

    def forward(self, feats, labels, sim_weight=None):
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        batch_size = feats.size(0)
        feats_norm = feats / feats.norm(dim=1)[:, None]
        sim_mat = torch.matmul(feats_norm, torch.t(feats_norm))

        epsilon = 1e-5
        loss = list()

        for i in range(batch_size):
            if not self.vanilla:
                pos_pair_w = sim_weight[i][labels == labels[i]]
            pos_pair_ = sim_mat[i][labels == labels[i]]
            # pos_pair_w = pos_pair_w[pos_pair_ < 1 - epsilon]
            # pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]

            if not self.vanilla:
                neg_pair_w = sim_weight[i][labels != labels[i]]
            neg_pair_ = sim_mat[i][labels != labels[i]]

            if len(pos_pair_) > 0:
                if not self.vanilla:
                    neg_pair_w = neg_pair_w[neg_pair_ + self.margin > min(pos_pair_)]
                neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_)]
            else:
                neg_pair = neg_pair_
            if len(neg_pair_) > 0:
                if not self.vanilla:
                    pos_pair_w = pos_pair_w[pos_pair_ - self.margin < max(neg_pair_)]
                pos_pair = pos_pair_[pos_pair_ - self.margin < max(neg_pair_)]
            else:
                pos_pair = pos_pair_

            if len(neg_pair) < 1 and len(pos_pair) < 1:
                continue

            # weighting step
            if self.vanilla:
                pos_loss = 1.0 / self.scale_pos * torch.log(
                    1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh))))
                neg_loss = 1.0 / self.scale_neg * torch.log(
                    1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh))))
            else:
                pos_loss = 1.0 / self.scale_pos * torch.log(
                    1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh) + self.scale_pos_w * pos_pair_w)))
                neg_loss = 1.0 / self.scale_neg * torch.log(
                    1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh) - self.scale_neg_w * neg_pair_w)))

            self.pos_pair_thresh.append(pos_pair - self.thresh)
            self.neg_pair_thresh.append(neg_pair - self.thresh)
            if not self.vanilla:
                self.pos_pair_w.append(pos_pair_w)
                self.neg_pair_w.append(neg_pair_w)
            loss.append(pos_loss + neg_loss)

        if len(loss) == 0:
            return torch.zeros([], requires_grad=True)

        loss = sum(loss) / batch_size
        return loss

class Feats:
    def __init__(self):
        self.ids, self.feats, self.labels = [], [], []

    def push(self, ett_feat):
        # mean over token to represent entity
        if ett_feat[0] != -1:
            self.ids.append(ett_feat[0])
            self.feats.append(torch.mean(torch.stack(ett_feat[1]), dim=0))
            self.labels.append(ett_feat[2])
    
    def tensorize(self):
        if len(self.ids) == 0 or len(self.feats) == 0 or len(self.labels) == 0:
            return torch.tensor([], dtype=torch.int), torch.tensor([], dtype=torch.int), torch.tensor([], dtype=torch.int)
        else:
            return torch.stack(self.ids), torch.stack(self.feats), torch.stack(self.labels)

def extract_feat(batched_feat, batched_label_mask, batched_token_id, batched_token_label, pseudo_labels=None, cluster_map=None):
    feat_obj = Feats()
    if pseudo_labels is None:
        pseudo_labels = torch.zeros_like(batched_token_label)
        no_pse = True
    else:
        no_pse = False
    for feat, label_mask, token_id, token_label, pse_label in zip(batched_feat, batched_label_mask, batched_token_id, batched_token_label, pseudo_labels):
        token_feat = []
        # use feature of first wordpiece to represent the token
        for feat_t, mask in zip(feat, label_mask):
            if mask == 1:
                token_feat.append(feat_t)
        token_feat = torch.stack(token_feat)
        
        ett_feat = (-1, [], 0)
        token_num = len(token_feat)
        for feat, id, label, p_label in zip(token_feat, token_id[:token_num], token_label[:token_num], pse_label[:token_num]):
            if (label > 0 or p_label > 0) and id != -1: # part of entity
                if label % 2 == 1 or p_label % 2 == 1: # start of an entity
                    if ett_feat[2] >= 0:
                        feat_obj.push(ett_feat)

                    if cluster_map is not None:             # use cluster id as ms_label
                        ms_label = cluster_map[id.item()]
                    elif not no_pse:                        # use is_pse as ms_label
                        ms_label = torch.tensor(p_label > 0)
                    else:
                        ms_label = label
                    ett_feat = (id, [feat], ms_label)
                else:
                    ett_feat[1].append(feat)
        if ett_feat[2] >= 0:
            feat_obj.push(ett_feat)
    return feat_obj   