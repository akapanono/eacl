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
    transfer_loss:torch.Tensor = None
    sentiment_representations:torch.Tensor = None
    sentiment_labels:torch.Tensor = None
    sentiment_anchortypes:torch.Tensor = None
    anchortype_labels:torch.Tensor = None
    max_cosine:torch.Tensor = None

def loss_function(log_prob, reps, raw_reps, semantic_reps, anchor_weights, class_anchors, label, mask, model):
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-1).to(reps.device)
    scl_loss_fn = SupConLoss(model.args)
    cl_loss = scl_loss_fn(reps, label, model, return_representations=not model.training)
    ce_loss = ce_loss_fn(log_prob[mask], label[mask])
    transfer_loss = prototype_transfer_loss(reps, semantic_reps, class_anchors, anchor_weights, label, model.args)
    if model.training:
        model.update_anchors(raw_reps, label)
    return HybridLossOutput(
        ce_loss=ce_loss,
        cl_loss=cl_loss.loss,
        transfer_loss=transfer_loss,
        sentiment_representations=cl_loss.sentiment_representations,
        sentiment_labels=cl_loss.sentiment_labels,
        sentiment_anchortypes=cl_loss.sentiment_anchortypes,
        anchortype_labels=cl_loss.anchortype_labels,
        max_cosine = cl_loss.max_cosine
    ) 

def prototype_transfer_loss(reps, semantic_reps, class_anchors, anchor_weights, labels, args):
    valid_mask = labels >= 0
    if valid_mask.sum().item() == 0:
        return reps.new_tensor(0.0)

    reps = reps[valid_mask]
    semantic_reps = semantic_reps[valid_mask]
    labels = labels[valid_mask]
    anchor_weights = anchor_weights[valid_mask]
    pos_anchors = class_anchors[labels]

    pos_align = 1 - F.cosine_similarity(reps, pos_anchors, dim=-1)
    semantic_consistency = 1 - F.cosine_similarity(reps, semantic_reps, dim=-1)
    class_indices = torch.arange(class_anchors.shape[0], device=labels.device).unsqueeze(0)
    neg_mask = class_indices != labels.unsqueeze(1)
    neg_scores = torch.matmul(F.normalize(reps, dim=-1), F.normalize(class_anchors, dim=-1).transpose(0, 1))
    hardest_negative = neg_scores.masked_fill(~neg_mask, -1e4).max(dim=-1)[0]
    margin = F.relu(hardest_negative - (1 - pos_align) + args.transfer_margin)
    entropy = -(anchor_weights * torch.log(anchor_weights + 1e-8)).sum(dim=-1)
    return (
        args.prototype_align_weight * pos_align.mean()
        + args.semantic_consistency_weight * semantic_consistency.mean()
        + args.transfer_margin_weight * margin.mean()
        + args.anchor_entropy_weight * entropy.mean()
    )

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
    
