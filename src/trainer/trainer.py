import numpy as np
import torch
import torch.distributed as dist
# from dataloader import IEMOCAPDataset
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

def unpack_batch(batch):
    if len(batch) >= 4:
        input_ids, label, state_input_ids, state_attention_mask = batch[:4]
        return input_ids, label, state_input_ids, state_attention_mask
    input_ids, label = batch
    return input_ids, label, None, None

def train_or_eval_model(model, loss_function, dataloader, epoch, device, args, optimizer=None, lr_scheduler=None, train=False):
    losses, preds, labels = [], [], []
    sentiment_representations, sentiment_labels = [], []
    component_losses = {
        "ce": [], "cl": [], "neutral": [], "supcon": [], "angle": [], "sas": [], "hard": []
    }

    assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()
    if args.disable_training_progress_bar:
        pbar = dataloader
    else:
        pbar = tqdm(dataloader)
    
    for batch_id, batch in enumerate(pbar):
        
        input_ids, label, state_input_ids, state_attention_mask = unpack_batch(batch)
       
        input_orig = input_ids
        input_aug = None

        if args.fp16:
            with torch.autocast(device_type="cuda" if args.cuda else "cpu"):
                loss, loss_output, log_prob, label, mask, anchor_scores = _forward(
                    model, loss_function, input_orig, input_aug, label, device,
                    state_input_ids=state_input_ids,
                    state_attention_mask=state_attention_mask,
                )
        else:
            loss, loss_output, log_prob, label, mask, anchor_scores = _forward(
                model, loss_function, input_orig, input_aug, label, device,
                state_input_ids=state_input_ids,
                state_attention_mask=state_attention_mask,
            )

        if args.use_nearest_neighbour:
            pred = torch.argmax(anchor_scores[mask], dim=-1)
        else:
            pred = torch.argmax(log_prob[mask], dim = -1)

        preds.append(pred)
        labels.append(label)
        losses.append(loss.item())
        component_losses["ce"].append(loss_output.ce_loss.item())
        component_losses["cl"].append(loss_output.cl_loss.item())
        component_losses["neutral"].append(loss_output.neutral_loss.item())
        component_losses["supcon"].append(loss_output.supcon_loss.item())
        component_losses["angle"].append(loss_output.angle_loss.item())
        component_losses["sas"].append(loss_output.sas_loss.item())
        component_losses["hard"].append(loss_output.hard_loss.item())

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, norm_type=2)
            if batch_id % args.accumulation_step == 0:
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
        return float('nan'), float('nan'), [], [], float('nan'), [], [], {}

    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = round(accuracy_score(new_labels, new_preds) * 100, 2)

    max_cosine = loss_output.max_cosine

    avg_fscore = round(f1_score(new_labels, new_preds, average='weighted') * 100, 2)
    stats = {
        f"loss_{name}": round(float(np.mean(values)), 4) if values else 0.0
        for name, values in component_losses.items()
    }

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
        f1 = round(f1_score(true_label, pred_label, zero_division=0) * 100, 2)
        f1_scores.append(f1)

    if hasattr(model, "neutral_id") and model.neutral_id is not None:
        neutral_true = (new_labels == model.neutral_id).astype(int)
        neutral_pred = (new_preds == model.neutral_id).astype(int)
        stats["neutral_f1"] = round(f1_score(neutral_true, neutral_pred, zero_division=0) * 100, 2)
        non_neutral_ids = [idx for idx in range(n) if idx != model.neutral_id]
        stats["non_neutral_macro_f1"] = round(
            f1_score(new_labels, new_preds, labels=non_neutral_ids, average="macro", zero_division=0) * 100,
            2,
        )
    if hasattr(model, "last_domain_weights"):
        stats["avg_domain_weight"] = [
            round(item, 4) for item in model.last_domain_weights.float().mean(dim=0).detach().cpu().tolist()
        ]

    return avg_loss, avg_accuracy, labels, preds, avg_fscore, f1_scores, max_cosine, stats


def _forward(model, loss_function, input_orig, input_aug, label, device, state_input_ids=None, state_attention_mask=None):

    input_ids = input_orig.to(device)
    label = label.to(device)
    if state_input_ids is not None:
        state_input_ids = state_input_ids.to(device)
    if state_attention_mask is not None:
        state_attention_mask = state_attention_mask.to(device)
    mask = torch.ones(len(input_orig)).to(device)
    mask = mask > 0.5
    if model.training:
        log_prob, masked_mapped_output, masked_output, anchor_scores = model(
            input_ids,
            state_input_ids=state_input_ids,
            state_attention_mask=state_attention_mask,
            return_mask_output=True,
        ) 
        loss_output = loss_function(log_prob, masked_mapped_output, masked_output, label, mask, model)
    else:
        with torch.no_grad():
            log_prob, masked_mapped_output, masked_output, anchor_scores = model(
                input_ids,
                state_input_ids=state_input_ids,
                state_attention_mask=state_attention_mask,
                return_mask_output=True,
            ) 
            loss_output = loss_function(log_prob, masked_mapped_output, masked_output, label, mask, model)
    loss = loss_output.total_loss

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
        f1 = round(f1_score(true_label, pred_label, zero_division=0) * 100, 2)
        f1_scores.append(f1)
    # list(precision_recall_fscore_support(y_true=new_labels, y_pred=new_preds)[2])

    return avg_loss, avg_ce_loss, avg_accuracy, labels, preds, avg_fscore, f1_scores
