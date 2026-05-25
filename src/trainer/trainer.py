import numpy as np
import torch
import torch.distributed as dist
# from dataloader import IEMOCAPDataset
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
from model.loss import (
    anchor_pull_loss,
    anchor_similarity_stats,
    compute_subanchor_assignment_counts,
    hyperspherical_inter_anchor_loss,
    intra_anchor_diversity_loss,
)

def train_or_eval_model(model, loss_function, dataloader, epoch, device, args, optimizer=None, lr_scheduler=None, train=False):
    losses, preds, labels = [], [], []
    sentiment_representations, sentiment_labels = [], []
    stat_sums, stat_count = {}, 0
    assignment_counts = None

    assert not train or optimizer != None
    if train:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()
    model.current_epoch = epoch
    if args.disable_training_progress_bar:
        pbar = dataloader
    else:
        pbar = tqdm(dataloader)
    
    for batch_id, batch in enumerate(pbar):
        
        input_ids, label = batch
       
        input_orig = input_ids
        input_aug = None

        if args.fp16:
            with torch.autocast(device_type="cuda" if args.cuda else "cpu"):
                loss, loss_output, log_prob, label, mask, anchor_scores = _forward(model, loss_function, input_orig, input_aug, label, device)
        else:
            loss, loss_output, log_prob, label, mask, anchor_scores = _forward(model, loss_function, input_orig, input_aug, label, device)

        if args.use_nearest_neighbour:
            pred = torch.argmax(anchor_scores[mask], dim=-1)
        else:
            pred = torch.argmax(log_prob[mask], dim = -1)

        preds.append(pred)
        labels.append(label)
        losses.append(loss.item())
        batch_stats = {
            "ce": loss_output.ce_loss,
            "cl": loss_output.cl_loss,
            "pull": loss_output.pull_loss,
            "inter": loss_output.inter_loss,
            "intra_div": loss_output.intra_div_loss,
        }
        if getattr(args, "use_cluster_anchors", False):
            for key, value in anchor_similarity_stats(model.get_active_mapped_anchors()).items():
                batch_stats[key] = value
            batch_counts = compute_subanchor_assignment_counts(
                getattr(loss_output, "active_reps", None),
                getattr(loss_output, "active_labels", None),
                model.get_active_mapped_anchors(),
            )
            assignment_counts = batch_counts if assignment_counts is None else assignment_counts + batch_counts
        for key, value in batch_stats.items():
            if value is None:
                continue
            stat_sums[key] = stat_sums.get(key, 0.0) + float(value.detach().cpu())
        stat_count += 1

        if train:
            (loss / max(1, args.accumulation_step)).backward()
            should_step = (batch_id + 1) % max(1, args.accumulation_step) == 0 or (batch_id + 1) == len(dataloader)
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, norm_type=2)
                optimizer.step()
                if lr_scheduler is not None and getattr(args, "lr_scheduler", "step") != "step":
                    lr_scheduler.step()
                optimizer.zero_grad()
        else:
            sentiment_representations.append(loss_output.sentiment_representations)
            sentiment_labels.append(loss_output.sentiment_labels)
    if len(preds) != 0:
        new_preds = []
        new_labels = []
        for i,label in enumerate(labels):
            for j,l in enumerate(label):
                if l != -1:
                    new_labels.append(l.cpu().item())
                    new_preds.append(preds[i][j].cpu().item())
    else:
        return float('nan'), float('nan'), [], [], float('nan'), [], [], [], [], []

    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = round(accuracy_score(new_labels, new_preds) * 100, 2)

    max_cosine = loss_output.max_cosine
    model.last_epoch_loss_stats = {
        key: round(value / max(1, stat_count), 6)
        for key, value in stat_sums.items()
    }
    if assignment_counts is not None:
        model.last_epoch_assignment_counts = assignment_counts.detach().cpu().tolist()
    else:
        model.last_epoch_assignment_counts = None

    avg_fscore = round(f1_score(new_labels, new_preds, average='weighted') * 100, 2)

    f1_scores = []

    new_labels = np.array(new_labels)
    new_preds = np.array(new_preds)

    if args.dataset_name in ['IEMOCAP']:
        n = 6
    else:
        n = 7

    for class_id in range(n):
        true_label = []
        pred_label = []
        for i in range(len(new_labels)):
            if new_labels[i] == class_id:
                true_label.append(1)
                if new_preds[i] == class_id:
                    pred_label.append(1)
                else:
                    pred_label.append(0)
            elif new_preds[i] == class_id:
                pred_label.append(1)
                if new_labels[i] == class_id:
                    true_label.append(1)
                else:
                    true_label.append(0)
        f1 = round(f1_score(true_label, pred_label) * 100, 2)
        f1_scores.append(f1)

    return avg_loss, avg_accuracy, labels, preds, avg_fscore, f1_scores, max_cosine


def _forward(model, loss_function, input_orig, input_aug, label, device):

    input_ids = input_orig.to(device)
    label = label.to(device)
    mask = torch.ones(len(input_orig)).to(device)
    mask = mask > 0.5
    if model.training:
        log_prob, masked_mapped_output, masked_output, anchor_scores = model(input_ids, return_mask_output=True) 
        loss_output = loss_function(log_prob, masked_mapped_output, masked_output, label, mask, model)
    else:
        with torch.no_grad():
            log_prob, masked_mapped_output, masked_output, anchor_scores = model(input_ids, return_mask_output=True) 
            loss_output = loss_function(log_prob, masked_mapped_output, masked_output, label, mask, model)
    loss = loss_output.ce_loss * model.args.ce_loss_weight + (1 - model.args.ce_loss_weight) * loss_output.cl_loss
    if getattr(model.args, "use_cluster_anchors", False):
        anchors = model.get_active_mapped_anchors()
        active_reps = masked_mapped_output[mask]
        active_labels = label[mask]
        loss_output.active_reps = active_reps.detach()
        loss_output.active_labels = active_labels.detach()
        if model.args.anchor_pull_weight > 0:
            loss_output.pull_loss = anchor_pull_loss(active_reps, active_labels, anchors)
            loss = loss + model.args.anchor_pull_weight * loss_output.pull_loss
        if model.args.hyp_inter_weight > 0:
            loss_output.inter_loss = hyperspherical_inter_anchor_loss(anchors)
            loss = loss + model.args.hyp_inter_weight * loss_output.inter_loss
        if model.args.intra_div_weight > 0:
            loss_output.intra_div_loss = intra_anchor_diversity_loss(
                anchors,
                same_upper=model.args.intra_same_upper,
            )
            loss = loss + model.args.intra_div_weight * loss_output.intra_div_loss

    return loss, loss_output, log_prob, label[mask], mask, anchor_scores

def retrain(model, loss_function, dataloader, epoch, device, args, optimizer=None, lr_scheduler=None, train=False):
    losses, ce_losses, preds, labels = [], [], [], []
    
    for batch in dataloader:
        data, label = batch
        data = data.to(device)
        label = label.to(device)
        if args.fp16:
            with torch.autocast(device_type="cuda" if args.cuda else "cpu"):
                log_prob = model(data) 
        else:
            log_prob = model(data)
        
        loss = loss_function(log_prob, label)
        losses.append(loss.item())
        ce_losses.append(loss.item())
        pred = torch.argmax(log_prob, dim = -1)
        preds.append(pred)
        labels.append(label)
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, norm_type=2)
            optimizer.step()
            optimizer.zero_grad()
    if len(preds) != 0:
        new_preds = []
        new_labels = []
        for i,label in enumerate(labels):
            for j,l in enumerate(label):
                if l != -1:
                    new_labels.append(l.cpu().item())
                    new_preds.append(preds[i][j].cpu().item())
    else:
        return float('nan'), float('nan'), [], [], float('nan'), [], [], [], [], []
        # plot_representations(sentiment_representations, sentiment_labels, sentiment_anchortypes, anchortype_labels)
    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_ce_loss = round(np.sum(ce_losses) / len(ce_losses), 4)
    avg_accuracy = round(accuracy_score(new_labels, new_preds) * 100, 2)
    f1_scores = []

    avg_fscore = round(f1_score(new_labels, new_preds, average='weighted') * 100, 2)

    new_labels = np.array(new_labels)
    new_preds = np.array(new_preds)

    if args.dataset_name in ['IEMOCAP']:
        n = 6
    else:
        n = 7

    for class_id in range(n):
        true_label = []
        pred_label = []
        for i in range(len(new_labels)):
            if new_labels[i] == class_id:
                true_label.append(1)
                if new_preds[i] == class_id:
                    pred_label.append(1)
                else:
                    pred_label.append(0)
            elif new_preds[i] == class_id:
                pred_label.append(1)
                if new_labels[i] == class_id:
                    true_label.append(1)
                else:
                    true_label.append(0)
        f1 = round(f1_score(true_label, pred_label) * 100, 2)
        f1_scores.append(f1)
    # list(precision_recall_fscore_support(y_true=new_labels, y_pred=new_preds)[2])

    return avg_loss, avg_ce_loss, avg_accuracy, labels, preds, avg_fscore, f1_scores
