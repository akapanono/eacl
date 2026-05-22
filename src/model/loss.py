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
    total_loss:torch.Tensor = None
    neutral_loss:torch.Tensor = None
    supcon_loss:torch.Tensor = None
    angle_loss:torch.Tensor = None
    sas_loss:torch.Tensor = None
    hard_loss:torch.Tensor = None
    sentiment_representations:torch.Tensor = None
    sentiment_labels:torch.Tensor = None
    sentiment_anchortypes:torch.Tensor = None
    anchortype_labels:torch.Tensor = None
    max_cosine:torch.Tensor = None

def zero_like_reps(reps):
    return reps.new_tensor(0.0)

def parse_similar_pairs(args):
    pairs = getattr(args, "similar_emotion_pairs", "happy:excited,sad:frustrated,angry:frustrated")
    if isinstance(pairs, (list, tuple)):
        return [tuple(pair) for pair in pairs if len(pair) == 2]
    parsed = []
    for item in str(pairs).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            left, right = item.split(":", 1)
        elif "-" in item:
            left, right = item.split("-", 1)
        else:
            continue
        parsed.append((left.strip().lower(), right.strip().lower()))
    return parsed

def get_similar_pair_ids(model):
    label_to_id = {name.lower(): idx for idx, name in enumerate(model.label_names)}
    pair_ids = []
    for left, right in parse_similar_pairs(model.args):
        if left not in label_to_id or right not in label_to_id:
            continue
        left_id = label_to_id[left]
        right_id = label_to_id[right]
        if model.use_neutral_decoupling:
            left_id = model.original_to_non_neutral[left_id].item()
            right_id = model.original_to_non_neutral[right_id].item()
            if left_id < 0 or right_id < 0:
                continue
        pair_ids.append((left_id, right_id))
    return pair_ids

def loss_function(log_prob, reps, raw_reps, label, mask, model):
    class_weights = getattr(model, "ce_class_weights", None)
    if getattr(model, "use_neutral_decoupling", False):
        ce_loss_fn = nn.NLLLoss(ignore_index=-1, weight=class_weights).to(reps.device)
    else:
        ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-1, weight=class_weights).to(reps.device)
    scl_loss_fn = SupConLoss(model.args)
    cl_loss = scl_loss_fn(reps, label, model, return_representations=not model.training)
    ce_loss = ce_loss_fn(log_prob[mask], label[mask])
    neutral_loss = zero_like_reps(reps)
    sas_loss = zero_like_reps(reps)
    hard_loss = zero_like_reps(reps)
    if getattr(model, "use_neutral_decoupling", False):
        neutral_logit = model.last_forward_output["neutral_logit"]
        neutral_target = (label == model.neutral_id).float()
        neutral_loss = F.binary_cross_entropy_with_logits(neutral_logit[mask], neutral_target[mask])
    if getattr(model.args, "use_similar_anchor_separation", False):
        anchor_embeddings = model.get_domain_mapped_anchors() if model.args.prototype_pooling == "domain_gated" else model.get_mapped_anchors()
        sas_loss = similar_anchor_separation_loss(
            anchor_embeddings,
            get_similar_pair_ids(model),
            margin=getattr(model.args, "sas_margin", 0.30),
        )
    if getattr(model.args, "use_hard_anchor_negative", False):
        hard_loss = hard_anchor_negative_loss(
            reps,
            label,
            model,
            get_similar_pair_ids(model),
            temperature=getattr(model.args, "hard_negative_temperature", 0.07),
            rho=getattr(model.args, "hard_negative_rho", 2.0),
        )
    if any([
        getattr(model.args, "use_neutral_decoupling", False),
        getattr(model.args, "use_speaker_state", False),
        getattr(model.args, "use_similar_anchor_separation", False),
        getattr(model.args, "use_hard_anchor_negative", False),
    ]):
        total_loss = (
            ce_loss
            + getattr(model.args, "lambda_neu", 0.5) * neutral_loss
            + getattr(model.args, "lambda_supcon", 1.0) * cl_loss.supcon_loss
            + getattr(model.args, "lambda_angle", getattr(model.args, "angle_loss_weight", 1.0)) * cl_loss.angle_loss
            + getattr(model.args, "lambda_sas", 0.02) * sas_loss
            + getattr(model.args, "lambda_hard", 0.05) * hard_loss
        )
        combined_cl = cl_loss.supcon_loss + getattr(model.args, "lambda_angle", getattr(model.args, "angle_loss_weight", 1.0)) * cl_loss.angle_loss
    else:
        combined_cl = cl_loss.loss
        total_loss = ce_loss * model.args.ce_loss_weight + (1 - model.args.ce_loss_weight) * combined_cl
    if model.training:
        model.update_anchors(raw_reps, label)
    return HybridLossOutput(
        ce_loss=ce_loss,
        cl_loss=combined_cl,
        total_loss=total_loss,
        neutral_loss=neutral_loss,
        supcon_loss=cl_loss.supcon_loss,
        angle_loss=cl_loss.angle_loss,
        sas_loss=sas_loss,
        hard_loss=hard_loss,
        sentiment_representations=cl_loss.sentiment_representations,
        sentiment_labels=cl_loss.sentiment_labels,
        sentiment_anchortypes=cl_loss.sentiment_anchortypes,
        anchortype_labels=cl_loss.anchortype_labels,
        max_cosine = cl_loss.max_cosine
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
    supcon_loss:torch.Tensor = None
    angle_loss:torch.Tensor = None
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
        if getattr(model, "use_neutral_decoupling", False):
            valid_sample_mask = (labels >= 0) & (labels != model.neutral_id)
            reps = reps[valid_sample_mask]
            labels = model.original_to_non_neutral[labels[valid_sample_mask]]
        batch_size = reps.shape[0]
        if self.args.prototype_pooling == "domain_gated":
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
        if concated_bsz == 0:
            angleloss, max_cosine = AngleLoss(class_anchor)
            loss = reps.new_tensor(0.0)
            return SupConOutput(
                loss=loss + self.args.angle_loss_weight * angleloss,
                supcon_loss=loss,
                angle_loss=angleloss,
                sentiment_representations=sentiment_representations,
                sentiment_labels=sentiment_labels,
                sentiment_anchortypes=sentiment_anchortypes,
                anchortype_labels=anchor_labels,
                max_cosine=max_cosine,
            )
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

        supcon_loss = loss
        loss = supcon_loss + self.args.angle_loss_weight * angleloss
        return SupConOutput(
            loss=loss,
            supcon_loss=supcon_loss,
            angle_loss=angleloss,
            sentiment_representations=sentiment_representations,
            sentiment_labels=sentiment_labels,
            sentiment_anchortypes=sentiment_anchortypes,
            anchortype_labels=anchor_labels,
            max_cosine = max_cosine
        )
    
def similar_anchor_separation_loss(anchor_embeddings, similar_pair_ids, margin=0.30):
    if not similar_pair_ids:
        return anchor_embeddings.new_tensor(0.0)
    anchors = F.normalize(anchor_embeddings, dim=-1)
    losses = []
    for left_id, right_id in similar_pair_ids:
        if left_id >= anchors.shape[0] or right_id >= anchors.shape[0]:
            continue
        cosine = torch.sum(anchors[left_id] * anchors[right_id], dim=-1)
        losses.append(F.relu(cosine - margin).pow(2).mean())
    if not losses:
        return anchor_embeddings.new_tensor(0.0)
    return torch.stack(losses).mean()

def hard_anchor_negative_loss(sample_repr, labels, model, similar_pair_ids, temperature=0.07, rho=2.0):
    if not similar_pair_ids:
        return sample_repr.new_tensor(0.0)
    valid_mask = labels >= 0
    if getattr(model, "use_neutral_decoupling", False):
        valid_mask = valid_mask & (labels != model.neutral_id)
    if valid_mask.sum().item() == 0:
        return sample_repr.new_tensor(0.0)
    sample_repr = sample_repr[valid_mask]
    labels = labels[valid_mask]
    if getattr(model, "use_neutral_decoupling", False):
        labels = model.original_to_non_neutral[labels]
    anchors = model.get_domain_mapped_anchors() if model.args.prototype_pooling == "domain_gated" else model.get_mapped_anchors()
    class_anchors = anchors.mean(dim=1)
    sample_repr = F.normalize(sample_repr, dim=-1)
    class_anchors = F.normalize(class_anchors, dim=-1)
    logits = torch.matmul(sample_repr, class_anchors.t()) / temperature
    exp_logits = torch.exp(logits - logits.max(dim=-1, keepdim=True)[0])
    similar_pair_set = set(similar_pair_ids) | {(right, left) for left, right in similar_pair_ids}
    losses = []
    for i in range(sample_repr.shape[0]):
        y = labels[i].item()
        if y < 0 or y >= class_anchors.shape[0]:
            continue
        pos = exp_logits[i, y]
        denom = pos.clone()
        for class_id in range(class_anchors.shape[0]):
            if class_id == y:
                continue
            weight = 1.0 + rho if (y, class_id) in similar_pair_set else 1.0
            denom = denom + weight * exp_logits[i, class_id]
        losses.append(-torch.log(pos / (denom + 1e-8) + 1e-8))
    if not losses:
        return sample_repr.new_tensor(0.0)
    return torch.stack(losses).mean()
