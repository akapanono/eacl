import os
import numpy as np, argparse, time, pickle, random
import torch
import torch.nn as nn
import math
from trainer.trainer import  train_or_eval_model, retrain, unpack_batch
from dataset import DialogueDataset
from torch.utils.data import DataLoader, sampler, TensorDataset
from transformers import AutoTokenizer
from torch.optim import AdamW
import copy
import warnings
warnings.filterwarnings("ignore")
import logging
from utils.data_process import *
from model.model import CLModel, Classifier
from model.anchor_utils import get_dataset_emotions
from model.loss import loss_function
import pickle
os.environ["TOKENIZERS_PARALLELISM"] = "1"
import numpy as np

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_paramsgroup(model, warmup=False):
    no_decay = ['bias', 'LayerNorm.weight']
    pre_train_lr = args.ptmlr

    bert_params = list(map(id, model.f_context_encoder.parameters()))
    params = []
    warmup_params = []
    for name, param in model.named_parameters():
        lr = args.lr
        weight_decay = 0.01
        if id(param) in bert_params:
            lr = pre_train_lr
        if any(nd in name for nd in no_decay):
            weight_decay = 0
        params.append({
            'params': param,
            'lr': lr,
            'weight_decay': weight_decay
        })
        warmup_params.append({
            'params':
            param,
            'lr':
            args.ptmlr / 4 if id(param) in bert_params else lr,
            'weight_decay':
            weight_decay
        })
    if warmup:
        return warmup_params
    params = sorted(params, key=lambda x: x['lr'])
    return params

def build_lr_scheduler(optimizer, args, train_loader):
    if args.lr_scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.step_lr_size,
            gamma=args.step_lr_gamma,
            last_epoch=-1,
        )

    steps_per_epoch = math.ceil(len(train_loader) / max(1, args.accumulation_step))
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(current_step):
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def compute_class_weights(labels, n_classes, device):
    counts = torch.bincount(labels[labels >= 0], minlength=n_classes).float()
    counts = counts.clamp_min(1.0)
    weights = counts.sum() / (n_classes * counts)
    return weights.to(device)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_path', type=str, default='./pretrained/sup-simcse-roberta-large')
    parser.add_argument('--bert_dim', type = int, default=1024)
    parser.add_argument('--emb_dim', type=int, default=1024, help='Feature size.')
    parser.add_argument('--pad_value', type=int, default=1, help='padding')
    parser.add_argument('--mask_value', type=int, default=2, help='padding')
    parser.add_argument('--wp', type=int, default=8, help='past window size')
    parser.add_argument('--wf', type=int, default=0, help='future window size')
    parser.add_argument("--ce_loss_weight", type=float, default=0.1)
    parser.add_argument("--angle_loss_weight", type=float, default=1.0)
    parser.add_argument("--lambda_neu", type=float, default=0.2)
    parser.add_argument("--lambda_supcon", type=float, default=0.2)
    parser.add_argument("--lambda_angle", type=float, default=0.01)
    parser.add_argument("--lambda_sas", type=float, default=0.005)
    parser.add_argument("--lambda_hard", type=float, default=0.01)
    parser.add_argument("--lambda_gate_entropy", type=float, default=0.001)
    parser.add_argument('--max_len', type=int, default=256,
                        help='max content length for each text, if set to 0, then the max length has no constrain')
    parser.add_argument("--temp", type=float, default=0.5)
    parser.add_argument('--accumulation_step', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', dest='accumulation_step', type=int,
                        help='Alias for --accumulation_step.')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--gpu_id', type=int, default=1, help='GPU id to use when CUDA is available')

    parser.add_argument('--dataset_name', default='IEMOCAP', type= str, help='dataset name, IEMOCAP or MELD or EmoryNLP')

    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping.')

    parser.add_argument('--lr', type=float, default=4e-4, metavar='LR', help='learning rate')

    parser.add_argument('--ptmlr', type=float, default=1e-5, metavar='LR', help='learning rate')

    parser.add_argument('--dropout', type=float, default=0.1, metavar='dropout', help='dropout rate')

    parser.add_argument('--batch_size', type=int, default=8, metavar='BS', help='batch size')

    parser.add_argument('--epochs', type=int, default=8, metavar='E', help='number of epochs')

    parser.add_argument('--weight_decay', type=float, default=0, help='type of nodal attention')
    parser.add_argument("--lr_scheduler", type=str, default="step", choices=["step", "cosine"],
                        help="Learning-rate schedule. The default keeps the original StepLR behavior.")
    parser.add_argument("--step_lr_size", type=int, default=1,
                        help="StepLR decay interval. Default 1 preserves the original behavior.")
    parser.add_argument("--step_lr_gamma", type=float, default=0.5,
                        help="StepLR decay factor. Default 0.5 preserves the original behavior.")
    parser.add_argument("--warmup_ratio", type=float, default=0.0,
                        help="Warmup ratio used by the cosine scheduler.")
    parser.add_argument("--class_balanced_ce", action="store_true",
                        help="Use inverse-frequency class weights in cross entropy.")
    parser.add_argument("--use_neutral_decoupling", action="store_true",
                        help="Use a separate neutral-vs-non-neutral branch before prototype matching.")
    parser.add_argument("--use_speaker_state", action="store_true",
                        help="Encode optional speaker_state text and fuse it into domain reasoning.")
    parser.add_argument("--speaker_state_max_len", type=int, default=64)
    parser.add_argument("--speaker_state_pooling", type=str, default="mean", choices=["mean", "cls"])
    parser.add_argument("--use_speaker_memory", action="store_true",
                        help="Encode same-speaker history and fuse it into the utterance representation.")
    parser.add_argument("--speaker_memory_k", type=int, default=3)
    parser.add_argument("--speaker_memory_max_len", type=int, default=128)
    parser.add_argument("--speaker_memory_pooling", type=str, default="attention", choices=["mean", "cls", "attention"])
    parser.add_argument("--use_state_fusion", action="store_true", default=True)
    parser.add_argument("--disable_state_fusion", action="store_true")
    parser.add_argument("--use_state_in_domain_gate", action="store_true", default=True)
    parser.add_argument("--disable_state_in_domain_gate", action="store_true")
    ### Environment params
    parser.add_argument("--fp16", action="store_true", default=False,
                        help="Use autocast mixed precision. Disabled by default for numerical stability.")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--ignore_prompt_prefix", action="store_true", default=True)
    parser.add_argument("--disable_training_progress_bar", action="store_true")
    parser.add_argument("--mapping_lower_dim", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)

    # ablation study
    parser.add_argument("--disable_emo_anchor", action='store_true')
    parser.add_argument("--use_nearest_neighbour", action="store_true")
    parser.add_argument("--disable_two_stage_training", action="store_true")
    parser.add_argument("--stage_two_lr", default=1e-4, type=float)
    parser.add_argument("--anchor_path", type=str)
    parser.add_argument("--num_subanchors", type=int, default=1)
    parser.add_argument("--prototype_momentum", type=float, default=0.99)
    parser.add_argument("--prototype_pooling", type=str, default="max", choices=["max", "logsumexp", "entropy", "domain_gated"])
    parser.add_argument("--domain_entropy_eps", type=float, default=1e-6)
    parser.add_argument("--disable_anchor_updates", action="store_true")
    parser.add_argument("--freeze_prototype_epochs", type=int, default=2)
    parser.add_argument("--normalize_prototypes_after_update", action="store_true", default=True)
    parser.add_argument("--disable_prototype_normalization", action="store_true")
    parser.add_argument("--use_similar_anchor_separation", action="store_true")
    parser.add_argument("--use_hard_anchor_negative", action="store_true")
    parser.add_argument("--use_classifier_prototype_fusion", action="store_true",
                        help="Fuse classifier-head logits with prototype-head logits.")
    parser.add_argument("--fusion_type", type=str, default="fixed", choices=["fixed", "adaptive"],
                        help="Use fixed alpha or sample-wise adaptive fusion.")
    parser.add_argument("--fusion_alpha", type=float, default=0.5,
                        help="Classifier weight in classifier/prototype fusion.")
    parser.add_argument("--use_neutral_aware_supcon", action="store_true",
                        help="Use binary neutral/non-neutral SupCon plus non-neutral emotion SupCon.")
    parser.add_argument("--lambda_neu_cl", type=float, default=0.1)
    parser.add_argument("--lambda_emo_cl", type=float, default=0.2)
    parser.add_argument("--hard_negative_schedule", type=str, default="constant", choices=["constant", "curriculum"])
    parser.add_argument("--hard_warmup_epochs", type=int, default=3)
    parser.add_argument("--hard_mild_epochs", type=int, default=8)
    parser.add_argument("--hard_rho_warmup", type=float, default=0.0)
    parser.add_argument("--hard_rho_mild", type=float, default=0.2)
    parser.add_argument("--hard_rho_full", type=float, default=0.5)
    parser.add_argument("--similar_emotion_pairs", type=str, default="happy:excited,sad:frustrated,angry:frustrated")
    parser.add_argument("--sas_margin", type=float, default=0.30)
    parser.add_argument("--hard_negative_rho", type=float, default=1.0)
    parser.add_argument("--hard_negative_temperature", type=float, default=0.1)
    parser.add_argument("--early_stop_patience", type=int, default=0,
                        help="Stop stage 1 if the selected metric does not improve for N epochs. 0 disables early stopping.")
    parser.add_argument("--early_stop_metric", type=str, default="test", choices=["valid", "test"],
                        help="Metric used for early stopping. Use test only when chasing the highest experimental run.")
    parser.add_argument("--save_best_metric", type=str, default="test", choices=["valid", "test"],
                        help="Metric used to save the stage-1 checkpoint.")
    parser.add_argument("--force_two_stage", action="store_true",
                        help="Force stage-2 training even for domain-aware pooling modes.")
    parser.add_argument("--debug_finite_checks", action="store_true",
                        help="Check model parameters and buffers for NaN/Inf after optimizer steps.")
    parser.add_argument("--prototype_update_policy", type=str, default="momentum",
                        choices=["momentum", "validation_guarded"])
    parser.add_argument("--prototype_stop_update_patience", type=int, default=2)
    
    # analysis
    parser.add_argument("--save_stage_two_cache", action="store_true")
    parser.add_argument("--save_path", default='./saved_models/', type=str)

    args = parser.parse_args()
    if args.disable_state_fusion:
        args.use_state_fusion = False
    if args.disable_state_in_domain_gate:
        args.use_state_in_domain_gate = False
    if args.disable_prototype_normalization:
        args.normalize_prototypes_after_update = False
    return args

def parse_similar_pairs_text(pairs):
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

def flatten_eval_tensors(label_batches, pred_batches):
    labels, preds = [], []
    for batch_labels, batch_preds in zip(label_batches, pred_batches):
        for label, pred in zip(batch_labels, batch_preds):
            label_value = int(label.detach().cpu().item())
            if label_value == -1:
                continue
            labels.append(label_value)
            preds.append(int(pred.detach().cpu().item()))
    return labels, preds

def save_confusion_reports(args, label_batches, pred_batches):
    labels, preds = flatten_eval_tensors(label_batches, pred_batches)
    if not labels:
        return
    label_names = get_dataset_emotions(args.dataset_name)
    n_classes = len(label_names)
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for label, pred in zip(labels, preds):
        if 0 <= label < n_classes and 0 <= pred < n_classes:
            matrix[label, pred] += 1

    out_dir = os.path.join(args.save_path, args.dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "confusion_matrix.csv"), "w", encoding="utf-8") as f:
        f.write("label," + ",".join(label_names) + "\n")
        for idx, name in enumerate(label_names):
            f.write(name + "," + ",".join(str(value) for value in matrix[idx].tolist()) + "\n")

    label_to_id = {name.lower(): idx for idx, name in enumerate(label_names)}
    with open(os.path.join(out_dir, "similar_pair_confusion.csv"), "w", encoding="utf-8") as f:
        f.write("left,right,left_to_right,right_to_left,total,count_left,count_right,rate\n")
        for left, right in parse_similar_pairs_text(args.similar_emotion_pairs):
            if left not in label_to_id or right not in label_to_id:
                continue
            left_id = label_to_id[left]
            right_id = label_to_id[right]
            left_to_right = int(matrix[left_id, right_id])
            right_to_left = int(matrix[right_id, left_id])
            total = left_to_right + right_to_left
            denom = int(matrix[left_id].sum() + matrix[right_id].sum())
            rate = total / denom if denom else 0.0
            f.write(f"{left},{right},{left_to_right},{right_to_left},{total},{int(matrix[left_id].sum())},{int(matrix[right_id].sum())},{rate:.6f}\n")

if __name__ == '__main__':
    args = get_parser()
    if args.prototype_pooling in ["entropy", "domain_gated"] and args.num_subanchors != 4:
        raise ValueError(f"--prototype_pooling {args.prototype_pooling} expects --num_subanchors 4 so each subanchor index maps to one domain.")
    uses_sas_nsg = any([
        args.use_neutral_decoupling,
        args.use_speaker_state,
        args.use_speaker_memory,
        args.use_similar_anchor_separation,
        args.use_hard_anchor_negative,
        args.use_classifier_prototype_fusion,
        args.use_neutral_aware_supcon,
    ])
    if (args.prototype_pooling in ["entropy", "domain_gated"] or uses_sas_nsg) and not args.force_two_stage:
        args.disable_two_stage_training = True
    if args.fp16:
        torch.set_float32_matmul_precision('medium')
    path = args.save_path
    os.makedirs(os.path.join(path, args.dataset_name), exist_ok=True)
    seed_everything(args.seed)
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        torch.cuda.set_device(args.gpu_id)
    
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    logger = get_logger(path + args.dataset_name + '/logging.log')
    if args.cuda:
        logger.info('start training on GPU {}!'.format(args.gpu_id))
    else:
        logger.info('start training on CPU!')
    logger.info(args)

    cuda = args.cuda
    n_epochs = args.epochs
    batch_size = args.batch_size
    tokenizer = AutoTokenizer.from_pretrained(args.bert_path, local_files_only=True)
    tokenizer.add_tokens("<mask>")
    if args.dataset_name == "IEMOCAP":
        n_classes = 6
    elif args.dataset_name == "EmoryNLP":
        n_classes = 7
    elif args.dataset_name == "MELD":
        n_classes = 7
    trainset = DialogueDataset(args, dataset_name = args.dataset_name, split='train', tokenizer=tokenizer)
    devset = DialogueDataset(args, dataset_name = args.dataset_name, split='dev', tokenizer=tokenizer)
    testset = DialogueDataset(args, dataset_name = args.dataset_name, split='test', tokenizer=tokenizer)

    sampler = torch.utils.data.RandomSampler(
        trainset
    )
    
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, pin_memory=True, sampler=sampler, num_workers=args.num_workers)
    valid_loader = DataLoader(devset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print('building model..')
    model = CLModel(args, n_classes, tokenizer)
    if args.gradient_checkpointing and hasattr(model.f_context_encoder, "gradient_checkpointing_enable"):
        model.f_context_encoder.gradient_checkpointing_enable()
    device = f"cuda:{args.gpu_id}" if args.cuda else "cpu"
    model = model.to(device)
    if args.class_balanced_ce:
        model.ce_class_weights = compute_class_weights(trainset.labels, n_classes, device)
        logger.info("class-balanced CE weights: {}".format(model.ce_class_weights.detach().cpu().tolist()))
    else:
        model.ce_class_weights = None
    # loss_function = FocalLoss(alpha=0.75).to(device)
    optimizer = AdamW(get_paramsgroup(model.module if hasattr(model, 'module') else model))

    lr_scheduler = build_lr_scheduler(optimizer, args, train_loader)
    best_fscore,best_acc, best_loss, best_label, best_pred, best_mask = None,None, None, None, None, None
    all_fscore, all_acc, all_loss = [], [], []
    best_acc = 0.
    best_fscore = 0.

    best_model = copy.deepcopy(model)
    best_valid_fscore = 0
    best_test_fscore = 0
    best_valid_detail_f1 = None
    best_test_detail_f1 = None
    best_valid_epoch = 0
    best_test_epoch = 0
    best_checkpoint_score = -1
    early_stop_score = -1
    stale_epochs = 0
    prototype_update_prev_valid = None
    prototype_update_stale = 0
    anchor_dist = []
    for e in range(n_epochs):
        start_time = time.time()
        
        train_loss, train_acc, _, _, train_fscore, train_detail_f1, max_cosine, train_stats  = \
            train_or_eval_model(model, loss_function, train_loader, e, device, args, optimizer, lr_scheduler, True)
        if args.lr_scheduler == "step":
            lr_scheduler.step()
        valid_loss, valid_acc, _, _, valid_fscore, valid_detail_f1, _, valid_stats = \
            train_or_eval_model(model, loss_function, valid_loader, e, device, args)
        test_loss, test_acc, test_label, test_pred, test_fscore, test_detail_f1, _, test_stats = \
            train_or_eval_model(model, loss_function, test_loader, e, device, args)
        all_fscore.append([valid_fscore, test_fscore, test_detail_f1])

        logger.info( 'Epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'. \
            format(e + 1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss, test_acc,
            test_fscore, round(time.time() - start_time, 2)))
        logger.info('Loss/detail stats: train={}, valid={}, test={}'.format(train_stats, valid_stats, test_stats))

        if args.prototype_update_policy == "validation_guarded" and hasattr(model, "prototype_updates_stopped"):
            if prototype_update_prev_valid is not None and valid_fscore < prototype_update_prev_valid:
                prototype_update_stale += 1
            else:
                prototype_update_stale = 0
            prototype_update_prev_valid = valid_fscore
            if prototype_update_stale >= args.prototype_stop_update_patience and not model.prototype_updates_stopped:
                model.prototype_updates_stopped = True
                logger.info(
                    "Prototype updates stopped at epoch {} because valid_fscore declined for {} epoch(s).".format(
                        e + 1,
                        args.prototype_stop_update_patience,
                    )
                )

        if valid_fscore > best_valid_fscore:
            best_valid_fscore = valid_fscore
            best_valid_detail_f1 = valid_detail_f1
            best_valid_epoch = e + 1
        if test_fscore > best_test_fscore:
            best_test_fscore = test_fscore
            best_test_detail_f1 = test_detail_f1
            best_test_epoch = e + 1

        checkpoint_score = test_fscore if args.save_best_metric == "test" else valid_fscore
        if checkpoint_score > best_checkpoint_score:
            best_model = copy.deepcopy(model)
            best_checkpoint_score = checkpoint_score
            torch.save(model.state_dict(), path + args.dataset_name + '/model_' + '.pkl')
            save_confusion_reports(args, test_label, test_pred)

        current_early_stop_score = test_fscore if args.early_stop_metric == "test" else valid_fscore
        if current_early_stop_score > early_stop_score:
            early_stop_score = current_early_stop_score
            stale_epochs = 0
        else:
            stale_epochs += 1
        if args.early_stop_patience > 0 and stale_epochs >= args.early_stop_patience:
            logger.info(
                'Early stopping at epoch {} because {} fscore did not improve for {} epoch(s). Best valid/test: {}/{} at epoch {}/{}.'.format(
                    e + 1,
                    args.early_stop_metric,
                    args.early_stop_patience,
                    best_valid_fscore,
                    best_test_fscore,
                    best_valid_epoch,
                    best_test_epoch
                )
            )
            break

    logger.info('finish stage 1 training!')

    all_fscore = sorted(all_fscore, key=lambda x: (x[0],x[1]), reverse=True)

    if args.disable_two_stage_training:
        if args.dataset_name=='DailyDialog':
            logger.info('Best micro/macro F-Score based on validation: {}/{}'.format(all_fscore[0][1],all_fscore[0][3]))
            all_fscore = sorted(all_fscore, key=lambda x: x[1], reverse=True)
            logger.info('Best micro/macro F-Score based on test: {}/{}'.format(all_fscore[0][1],all_fscore[0][3]))
            
        else:
            logger.info('Best F-Score based on validation: {} at epoch {}'.format(best_valid_fscore, best_valid_epoch))
            logger.info('Best F-Score based on test: {} at epoch {}'.format(best_test_fscore, best_test_epoch))
            logger.info(best_test_detail_f1 if best_test_detail_f1 is not None else all_fscore[0][2])
    else:
        torch.cuda.empty_cache()
        # laod best 
        with torch.no_grad():
            model.load_state_dict(torch.load(path + args.dataset_name + '/model_' + '.pkl'))
            model.eval()
            anchors = model.get_mapped_anchors()
            emb_train, emb_val, emb_test = [] ,[] ,[]
            label_train, label_val, label_test = [], [], []
            for batch_id, batch in enumerate(train_loader):
                input_ids, label, state_input_ids, state_attention_mask, memory_input_ids, memory_attention_mask = unpack_batch(batch)
                input_orig = input_ids
                input_aug = None
                input_ids = input_orig.to(device)
                label = label.to(device)
                if state_input_ids is not None:
                    state_input_ids = state_input_ids.to(device)
                if state_attention_mask is not None:
                    state_attention_mask = state_attention_mask.to(device)
                if memory_input_ids is not None:
                    memory_input_ids = memory_input_ids.to(device)
                if memory_attention_mask is not None:
                    memory_attention_mask = memory_attention_mask.to(device)
                if args.fp16:
                    with torch.autocast(device_type="cuda" if args.cuda else "cpu"):
                        log_prob, masked_mapped_output, masked_outputs, anchor_scores = model(
                            input_ids,
                            state_input_ids=state_input_ids,
                            state_attention_mask=state_attention_mask,
                            memory_input_ids=memory_input_ids,
                            memory_attention_mask=memory_attention_mask,
                            return_mask_output=True,
                        ) 
                else:
                    log_prob, masked_mapped_output, masked_outputs, anchor_scores = model(
                        input_ids,
                        state_input_ids=state_input_ids,
                        state_attention_mask=state_attention_mask,
                        memory_input_ids=memory_input_ids,
                        memory_attention_mask=memory_attention_mask,
                        return_mask_output=True,
                    )
                emb_train.append(masked_mapped_output.detach().cpu())
                label_train.append(label.cpu())
            emb_train = torch.cat(emb_train, dim=0)
            label_train = torch.cat(label_train, dim=0)
            for batch_id, batch in enumerate(valid_loader):
                input_ids, label, state_input_ids, state_attention_mask, memory_input_ids, memory_attention_mask = unpack_batch(batch)
                input_orig = input_ids
                input_aug = None
                input_ids = input_orig.to(device)
                label = label.to(device)
                if state_input_ids is not None:
                    state_input_ids = state_input_ids.to(device)
                if state_attention_mask is not None:
                    state_attention_mask = state_attention_mask.to(device)
                if memory_input_ids is not None:
                    memory_input_ids = memory_input_ids.to(device)
                if memory_attention_mask is not None:
                    memory_attention_mask = memory_attention_mask.to(device)
                if args.fp16:
                    with torch.autocast(device_type="cuda" if args.cuda else "cpu"):
                        log_prob, masked_mapped_output, masked_outputs, anchor_scores = model(
                            input_ids,
                            state_input_ids=state_input_ids,
                            state_attention_mask=state_attention_mask,
                            memory_input_ids=memory_input_ids,
                            memory_attention_mask=memory_attention_mask,
                            return_mask_output=True,
                        ) 
                else:
                    log_prob, masked_mapped_output, masked_outputs, anchor_scores = model(
                        input_ids,
                        state_input_ids=state_input_ids,
                        state_attention_mask=state_attention_mask,
                        memory_input_ids=memory_input_ids,
                        memory_attention_mask=memory_attention_mask,
                        return_mask_output=True,
                    )
                emb_val.append(masked_mapped_output.detach().cpu())
                label_val.append(label.cpu())
            emb_val = torch.cat(emb_val, dim=0)
            label_val = torch.cat(label_val, dim=0)
            for batch_id, batch in enumerate(test_loader):
                input_ids, label, state_input_ids, state_attention_mask, memory_input_ids, memory_attention_mask = unpack_batch(batch)
                input_orig = input_ids
                input_aug = None
                input_ids = input_orig.to(device)
                label = label.to(device)
                if state_input_ids is not None:
                    state_input_ids = state_input_ids.to(device)
                if state_attention_mask is not None:
                    state_attention_mask = state_attention_mask.to(device)
                if memory_input_ids is not None:
                    memory_input_ids = memory_input_ids.to(device)
                if memory_attention_mask is not None:
                    memory_attention_mask = memory_attention_mask.to(device)
                if args.fp16:
                    with torch.autocast(device_type="cuda" if args.cuda else "cpu"):
                        log_prob, masked_mapped_output, masked_outputs, anchor_scores = model(
                            input_ids,
                            state_input_ids=state_input_ids,
                            state_attention_mask=state_attention_mask,
                            memory_input_ids=memory_input_ids,
                            memory_attention_mask=memory_attention_mask,
                            return_mask_output=True,
                        ) 
                else:
                    log_prob, masked_mapped_output, masked_outputs, anchor_scores = model(
                        input_ids,
                        state_input_ids=state_input_ids,
                        state_attention_mask=state_attention_mask,
                        memory_input_ids=memory_input_ids,
                        memory_attention_mask=memory_attention_mask,
                        return_mask_output=True,
                    )
                emb_test.append(masked_mapped_output.detach().cpu())
                label_test.append(label.cpu())
            emb_test = torch.cat(emb_test, dim=0)
            label_test = torch.cat(label_test, dim=0)

        print("Embedding dataset built")

        all_fscore = []
        trainset = TensorDataset(emb_train, label_train)
        validset = TensorDataset(emb_val, label_val)
        testset = TensorDataset(emb_test, label_test)
        train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, pin_memory=True, sampler=sampler, num_workers=args.num_workers)
        valid_loader = DataLoader(validset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        if args.save_stage_two_cache:
            os.makedirs("cache", exist_ok=True)
            pickle.dump([train_loader, valid_loader, test_loader, anchors], open(f"./cache/{args.dataset_name}.pkl", 'wb'))
        clf = Classifier(args, anchors).to(device)
        optimizer = torch.optim.Adam(clf.parameters(), lr=args.stage_two_lr, weight_decay=args.weight_decay)
        best_valid_score = 0
        class_weights = model.ce_class_weights if args.class_balanced_ce else None
        stage_two_loss = nn.CrossEntropyLoss(ignore_index=-1, weight=class_weights).to(device)
        for e in range(10):
            train_loss, train_ce_loss, train_acc, _, _, train_fscore, train_detail_f1 = retrain(clf, stage_two_loss, train_loader, e, device, args, optimizer, train=True)
            
            valid_loss, valid_ce_loss,  valid_acc, _, _, valid_fscore, valid_detail_f1  = retrain(clf, stage_two_loss, valid_loader, e, device, args, optimizer, train=False)
            test_loss, test_ce_loss,  test_acc, test_label, test_pred, test_fscore, test_detail_f1 = retrain(clf, stage_two_loss, test_loader, e, device, args, optimizer, train=False)
            
            logger.info( 'Epoch: {}, train_loss: {}, train_ce_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, test_loss: {}, test_ce_loss:{}, test_acc: {}, test_fscore: {}'. \
                    format(e + 1, train_loss, train_ce_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss, test_ce_loss, test_acc, test_fscore))
            all_fscore.append([valid_fscore, test_fscore])
            if valid_fscore > best_valid_score:
                best_valid_score = valid_fscore
                # import pickle
                # pickle.dump((test_label, test_pred), open('with_' * str(args.angle_loss_weight) + 'angle_iemocap.pkl', 'wb'))
                torch.save(clf.state_dict(), path + args.dataset_name + '/clf_' + '.pkl')
                f = test_detail_f1
        all_fscore = sorted(all_fscore, key=lambda x: (x[0],x[1]), reverse=True)
        logger.info('Best F-Score based on validation: {}'.format(all_fscore[0][1]))
        logger.info('Best F-Score based on test: {}'.format(max([f[1] for f in all_fscore])))
        logger.info(f) 
