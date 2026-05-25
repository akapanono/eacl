from config import *
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
@dataclass
class HybridLossOutput:
    ce_loss:torch.Tensor = None
    cl_loss:torch.Tensor = None
    pull_loss:torch.Tensor = None
    inter_loss:torch.Tensor = None
    intra_div_loss:torch.Tensor = None
    sentiment_representations:torch.Tensor = None
    sentiment_labels:torch.Tensor = None
    sentiment_anchortypes:torch.Tensor = None
    anchortype_labels:torch.Tensor = None
    max_cosine:torch.Tensor = None

def loss_function(log_prob, reps, raw_reps, label, mask, model):
    class_weights = getattr(model, "ce_class_weights", None)
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-1, weight=class_weights).to(reps.device)
    scl_loss_fn = SupConLoss(model.args)
    cl_loss = scl_loss_fn(reps, label, model, return_representations=not model.training)
    ce_loss = ce_loss_fn(log_prob[mask], label[mask])
    warmup_epochs = getattr(model.args, "warmup_anchor_update_epochs", 0)
    current_epoch = getattr(model, "current_epoch", 0)
    if model.training and current_epoch >= warmup_epochs and not getattr(model.args, "use_cluster_anchors", False):
        model.update_anchors(raw_reps, label)
    return HybridLossOutput(
        ce_loss=ce_loss,
        cl_loss=cl_loss.loss,
        pull_loss=reps.new_tensor(0.0),
        inter_loss=reps.new_tensor(0.0),
        intra_div_loss=reps.new_tensor(0.0),
        sentiment_representations=cl_loss.sentiment_representations,
        sentiment_labels=cl_loss.sentiment_labels,
        sentiment_anchortypes=cl_loss.sentiment_anchortypes,
        anchortype_labels=cl_loss.anchortype_labels,
        max_cosine = cl_loss.max_cosine
    ) 

def anchor_pull_loss(reps, labels, anchors):
    valid_mask = labels >= 0
    if valid_mask.sum().item() == 0:
        return reps.new_tensor(0.0)
    reps = F.normalize(reps[valid_mask], dim=-1)
    labels = labels[valid_mask].long()
    anchors = F.normalize(anchors, dim=-1)
    cur_anchors = anchors[labels]
    sim = torch.einsum("bd,bkd->bk", reps, cur_anchors)
    best_k = sim.argmax(dim=1)
    batch_idx = torch.arange(reps.size(0), device=reps.device)
    pos_anchor = cur_anchors[batch_idx, best_k]
    return (1.0 - F.cosine_similarity(reps, pos_anchor, dim=-1)).mean()

def hyperspherical_inter_anchor_loss(anchors):
    C, K, D = anchors.shape
    anchors = F.normalize(anchors, dim=-1)
    flat = anchors.reshape(C * K, D)
    sim = torch.matmul(flat, flat.t())
    labels = torch.arange(C, device=anchors.device).repeat_interleave(K)
    diff_mask = labels[:, None] != labels[None, :]
    hardest_diff_sim = sim.masked_fill(~diff_mask, -1e4).max(dim=1).values
    return hardest_diff_sim.mean()

def intra_anchor_diversity_loss(anchors, same_upper=0.85):
    C, K, D = anchors.shape
    if K <= 1:
        return anchors.new_tensor(0.0)
    anchors = F.normalize(anchors, dim=-1)
    flat = anchors.reshape(C * K, D)
    sim = torch.matmul(flat, flat.t())
    labels = torch.arange(C, device=anchors.device).repeat_interleave(K)
    same_mask = labels[:, None] == labels[None, :]
    same_mask = same_mask & (~torch.eye(C * K, dtype=torch.bool, device=anchors.device))
    same_sim = sim[same_mask]
    if same_sim.numel() == 0:
        return anchors.new_tensor(0.0)
    return F.relu(same_sim - same_upper).mean()

def anchor_similarity_stats(anchors):
    C, K, D = anchors.shape
    if C * K <= 1:
        zero = anchors.new_tensor(0.0)
        return {
            "avg_same_class_anchor_cos": zero,
            "max_same_class_anchor_cos": zero,
            "avg_diff_class_anchor_cos": zero,
            "max_diff_class_anchor_cos": zero,
        }
    anchors = F.normalize(anchors, dim=-1)
    flat = anchors.reshape(C * K, D)
    sim = torch.matmul(flat, flat.t())
    labels = torch.arange(C, device=anchors.device).repeat_interleave(K)
    eye = torch.eye(C * K, dtype=torch.bool, device=anchors.device)
    same_mask = (labels[:, None] == labels[None, :]) & (~eye)
    diff_mask = labels[:, None] != labels[None, :]
    same_sim = sim[same_mask]
    diff_sim = sim[diff_mask]
    return {
        "avg_same_class_anchor_cos": same_sim.mean() if same_sim.numel() else anchors.new_tensor(0.0),
        "max_same_class_anchor_cos": same_sim.max() if same_sim.numel() else anchors.new_tensor(0.0),
        "avg_diff_class_anchor_cos": diff_sim.mean() if diff_sim.numel() else anchors.new_tensor(0.0),
        "max_diff_class_anchor_cos": diff_sim.max() if diff_sim.numel() else anchors.new_tensor(0.0),
    }

def compute_subanchor_assignment_counts(reps, labels, anchors):
    valid_mask = labels >= 0
    counts = torch.zeros(anchors.shape[0], anchors.shape[1], dtype=torch.long, device=anchors.device)
    if valid_mask.sum().item() == 0:
        return counts
    reps = F.normalize(reps[valid_mask], dim=-1)
    labels = labels[valid_mask].long()
    anchors = F.normalize(anchors, dim=-1)
    cur_anchors = anchors[labels]
    sim = torch.einsum("bd,bkd->bk", reps, cur_anchors)
    best_k = sim.argmax(dim=1)
    for class_id, subanchor_id in zip(labels.tolist(), best_k.tolist()):
        counts[class_id, subanchor_id] += 1
    return counts

def AngleLoss(means):
    g_mean = means.mean(dim=0)
    centered_mean = means - g_mean
    means_ = F.normalize(centered_mean, p=2, dim=1)
    cosine = torch.matmul(means_, means_.t())
    cosine = cosine - 2. * torch.diag(torch.diag(cosine))
    max_cosine = cosine.max().clamp(-0.99999, 0.99999)
    loss = -torch.acos(cosine.max(dim=1)[0].clamp(-0.99999, 0.99999)).mean()

    return loss, max_cosine

@dataclass
class SupConOutput:
    loss:torch.Tensor = None
    sentiment_representations:torch.Tensor = None
    sentiment_labels:torch.Tensor = None
    sentiment_anchortypes:torch.Tensor = None
    anchortype_labels:torch.Tensor = None
    max_cosine:torch.Tensor = None


class SupConLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.temperature = args.temp
        self.eps = 1e-8
        self.args = args

    def score_func(self, x, y):
        return (1 + F.cosine_similarity(x, y, dim=-1))/2 + self.eps
    
    def forward(self, reps, labels, model, return_representations=False):
        batch_size = reps.shape[0]
        if getattr(self.args, "use_cluster_anchors", False):
            emo_anchor = model.get_active_mapped_anchors()
        elif self.args.prototype_pooling == "domain_gated":
            emo_anchor = model.get_domain_mapped_anchors()
        else:
            emo_anchor = model.get_mapped_anchors()
        flat_anchor = emo_anchor.view(-1, emo_anchor.shape[-1])
        anchor_labels = model.emo_label.to(reps.device)
        class_anchor = emo_anchor.mean(dim=1)
        if return_representations:
            sentiment_labels = labels
            sentiment_representations = reps.detach()
            sentiment_anchortypes = flat_anchor.detach()
        else:
            sentiment_labels = None
            sentiment_representations = None
            sentiment_anchortypes = None
        if self.args.disable_emo_anchor:
            concated_reps = reps
            concated_labels = labels
            concated_bsz = batch_size
        else:
            concated_reps = torch.cat([reps, flat_anchor], dim=0)
            concated_labels = torch.cat([labels, anchor_labels], dim=0)
            concated_bsz = batch_size + flat_anchor.shape[0]
        mask1 = concated_labels.unsqueeze(0).expand(concated_labels.shape[0], concated_labels.shape[0])
        mask2 = concated_labels.unsqueeze(1).expand(concated_labels.shape[0], concated_labels.shape[0])
        mask = 1 - torch.eye(concated_bsz).to(reps.device)
        pos_mask = (mask1 == mask2).long()
        rep1 = concated_reps.unsqueeze(0).expand(concated_bsz, concated_bsz, concated_reps.shape[-1])
        rep2 = concated_reps.unsqueeze(1).expand(concated_bsz, concated_bsz, concated_reps.shape[-1])
        scores = self.score_func(rep1, rep2)
        scores *= 1 - torch.eye(concated_bsz).to(scores.device)
        
        scores /= self.temperature
        scores = scores[:concated_bsz]
        pos_mask = pos_mask[:concated_bsz]
        mask = mask[:concated_bsz]
        
        scores -= torch.max(scores).item()

        angleloss, max_cosine = AngleLoss(class_anchor)
        # print(max_cosine)

        scores = torch.exp(scores)
        pos_scores = scores * (pos_mask * mask)
        neg_scores = scores * (1 - pos_mask)
        probs = pos_scores.sum(-1)/(pos_scores.sum(-1) + neg_scores.sum(-1))
        probs /= (pos_mask * mask).sum(-1) + self.eps
        loss = - torch.log(probs + self.eps)
        loss_mask = (loss > 0.0).long()
        loss = (loss * loss_mask).sum() / (loss_mask.sum().item() + self.eps)

        loss += self.args.angle_loss_weight * angleloss
        return SupConOutput(
            loss=loss,
            sentiment_representations=sentiment_representations,
            sentiment_labels=sentiment_labels,
            sentiment_anchortypes=sentiment_anchortypes,
            anchortype_labels=anchor_labels,
            max_cosine = max_cosine
        )
    
