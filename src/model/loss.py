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
    task_loss:torch.Tensor = None
    emo_loss:torch.Tensor = None
    neutral_loss:torch.Tensor = None
    supcon_loss:torch.Tensor = None
    angle_loss:torch.Tensor = None
    sas_loss:torch.Tensor = None
    hard_loss:torch.Tensor = None
    gate_entropy:torch.Tensor = None
    sentiment_representations:torch.Tensor = None
    sentiment_labels:torch.Tensor = None
    sentiment_anchortypes:torch.Tensor = None
    anchortype_labels:torch.Tensor = None
    max_cosine:torch.Tensor = None

def zero_like_reps(reps):
    return reps.new_tensor(0.0)

def check_finite_loss(loss_dict):
    for name, value in loss_dict.items():
        if value is None or not isinstance(value, torch.Tensor):
            continue
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} is NaN or Inf: {value.detach().cpu()}")

def neutral_decoupling_loss_stable(label, mask, model):
    output = model.last_forward_output
    neutral_logit = output["neutral_logit"]
    non_neutral_logits = output["non_neutral_logits"]
    neutral_target = (label == model.neutral_id).float()
    neutral_loss = F.binary_cross_entropy_with_logits(neutral_logit[mask], neutral_target[mask])

    non_neutral_mask = mask & (label >= 0) & (label != model.neutral_id)
    if non_neutral_mask.sum().item() == 0:
        emo_loss = neutral_logit.new_tensor(0.0)
    else:
        mapped_labels = model.original_to_non_neutral[label[non_neutral_mask]]
        class_weights = getattr(model, "ce_class_weights", None)
        non_neutral_weights = None
        if class_weights is not None:
            non_neutral_weights = class_weights[model.non_neutral_to_original]
        emo_loss = F.cross_entropy(
            non_neutral_logits[non_neutral_mask],
            mapped_labels,
            weight=non_neutral_weights,
        )

    check_finite_loss({
        "neutral_loss": neutral_loss,
        "emo_loss": emo_loss,
    })
    return emo_loss, neutral_loss

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
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-1, weight=class_weights).to(reps.device)
    scl_loss_fn = SupConLoss(model.args)
    cl_loss = scl_loss_fn(reps, label, model, return_representations=not model.training)
    neutral_loss = zero_like_reps(reps)
    sas_loss = zero_like_reps(reps)
    hard_loss = zero_like_reps(reps)
    gate_entropy = getattr(model, "last_gate_entropy", None)
    if gate_entropy is None:
        gate_entropy = zero_like_reps(reps)
    if getattr(model, "use_neutral_decoupling", False):
        ce_loss, neutral_loss = neutral_decoupling_loss_stable(label, mask, model)
    else:
        if mask.sum().item() == 0:
            ce_loss = zero_like_reps(reps)
        else:
            ce_loss = ce_loss_fn(log_prob[mask], label[mask])
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
        task_loss = ce_loss + getattr(model.args, "lambda_neu", 0.5) * neutral_loss
        total_loss = (
            task_loss
            + getattr(model.args, "lambda_supcon", 1.0) * cl_loss.supcon_loss
            + getattr(model.args, "lambda_angle", getattr(model.args, "angle_loss_weight", 1.0)) * cl_loss.angle_loss
            + getattr(model.args, "lambda_sas", 0.02) * sas_loss
            + getattr(model.args, "lambda_hard", 0.05) * hard_loss
            - getattr(model.args, "lambda_gate_entropy", 0.0) * gate_entropy
        )
        combined_cl = cl_loss.supcon_loss + getattr(model.args, "lambda_angle", getattr(model.args, "angle_loss_weight", 1.0)) * cl_loss.angle_loss
    else:
        combined_cl = cl_loss.loss
        task_loss = ce_loss
        total_loss = ce_loss * model.args.ce_loss_weight + (1 - model.args.ce_loss_weight) * combined_cl
    check_finite_loss({
        "total_loss": total_loss,
        "task_loss": task_loss,
        "ce_loss": ce_loss,
        "neutral_loss": neutral_loss,
        "supcon_loss": cl_loss.supcon_loss,
        "angle_loss": cl_loss.angle_loss,
        "sas_loss": sas_loss,
        "hard_loss": hard_loss,
        "gate_entropy": gate_entropy,
    })
    if model.training:
        model.update_anchors(raw_reps, label)
    return HybridLossOutput(
        ce_loss=ce_loss,
        cl_loss=combined_cl,
        total_loss=total_loss,
        task_loss=task_loss,
        emo_loss=ce_loss,
        neutral_loss=neutral_loss,
        supcon_loss=cl_loss.supcon_loss,
        angle_loss=cl_loss.angle_loss,
        sas_loss=sas_loss,
        hard_loss=hard_loss,
        gate_entropy=gate_entropy,
        sentiment_representations=cl_loss.sentiment_representations,
        sentiment_labels=cl_loss.sentiment_labels,
        sentiment_anchortypes=cl_loss.sentiment_anchortypes,
        anchortype_labels=cl_loss.anchortype_labels,
        max_cosine = cl_loss.max_cosine
    ) 

def AngleLoss(means):
    if means.shape[0] <= 1:
        return means.new_tensor(0.0), means.new_tensor(0.0)
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
        angleloss, max_cosine = AngleLoss(class_anchor)
        if return_representations:
            sentiment_labels = labels
            sentiment_representations = reps.detach()
            sentiment_anchortypes = flat_anchor.detach()
        else:
            sentiment_labels = None
            sentiment_representations = None
            sentiment_anchortypes = None
        if batch_size == 0:
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

        scores = torch.exp(scores)
        pos_scores = scores * (pos_mask * mask)
        neg_scores = scores * (1 - pos_mask)
        positive_count = (pos_mask * mask).sum(-1)
        valid_rows = positive_count > 0
        if valid_rows.sum().item() == 0:
            loss = reps.new_tensor(0.0)
        else:
            probs = pos_scores.sum(-1) / (pos_scores.sum(-1) + neg_scores.sum(-1) + self.eps)
            probs = probs / positive_count.clamp_min(1).float()
            row_loss = -torch.log(probs.clamp_min(self.eps))
            row_loss = row_loss[valid_rows]
            loss_mask = torch.isfinite(row_loss) & (row_loss > 0.0)
            if loss_mask.sum().item() == 0:
                loss = reps.new_tensor(0.0)
            else:
                loss = row_loss[loss_mask].mean()

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
        cosine = torch.sum(anchors[left_id] * anchors[right_id], dim=-1).clamp(-1.0, 1.0)
        losses.append(F.relu(cosine - margin).pow(2).mean())
    if not losses:
        return anchor_embeddings.new_tensor(0.0)
    loss = torch.stack(losses).mean()
    check_finite_loss({"sas_loss": loss})
    return loss

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
    temperature = max(float(temperature), 1e-6)
    logits = torch.matmul(sample_repr, class_anchors.t()) / temperature
    logits = logits.clamp(min=-50.0, max=50.0)
    weight = torch.ones_like(logits)
    similar_pair_set = set(similar_pair_ids) | {(right, left) for left, right in similar_pair_ids}
    for i in range(sample_repr.shape[0]):
        y = labels[i].item()
        if y < 0 or y >= class_anchors.shape[0]:
            continue
        for class_id in range(class_anchors.shape[0]):
            if class_id == y:
                continue
            if (y, class_id) in similar_pair_set:
                weight[i, class_id] = 1.0 + rho
    valid_label_mask = (labels >= 0) & (labels < class_anchors.shape[0])
    if valid_label_mask.sum().item() == 0:
        return sample_repr.new_tensor(0.0)
    logits = logits[valid_label_mask]
    labels = labels[valid_label_mask]
    weight = weight[valid_label_mask]
    weighted_logits = logits + torch.log(weight.clamp_min(1e-8))
    log_den = torch.logsumexp(weighted_logits, dim=-1)
    index = torch.arange(labels.shape[0], device=labels.device)
    log_pos = logits[index, labels]
    loss = (log_den - log_pos).mean()
    check_finite_loss({"hard_loss": loss})
    return loss
