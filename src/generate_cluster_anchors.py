import argparse
import os
import random
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import DialogueDataset
from model.model import CLModel

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "1"


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_path", type=str, default="./pretrained/sup-simcse-roberta-large")
    parser.add_argument("--dataset_name", type=str, default="IEMOCAP", choices=["IEMOCAP", "MELD", "EmoryNLP"])
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--anchor_path", type=str, required=True)
    parser.add_argument("--num_subanchors", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="./cluster_anchors/sup-simcse-roberta-large")
    parser.add_argument("--mapping_lower_dim", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--pad_value", type=int, default=1)
    parser.add_argument("--mask_value", type=int, default=2)
    parser.add_argument("--wp", type=int, default=8)
    parser.add_argument("--wf", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temp", type=float, default=0.5)
    parser.add_argument("--prototype_pooling", type=str, default="max", choices=["max", "logsumexp", "entropy", "domain_gated"])
    parser.add_argument("--prototype_momentum", type=float, default=0.9)
    parser.add_argument("--domain_entropy_eps", type=float, default=1e-6)
    parser.add_argument("--disable_anchor_updates", action="store_true")
    parser.add_argument("--disable_emo_anchor", action="store_true")
    parser.add_argument("--use_nearest_neighbour", action="store_true")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kmeans_iters", type=int, default=30)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_num_classes(dataset_name):
    if dataset_name == "IEMOCAP":
        return 6
    if dataset_name in ["MELD", "EmoryNLP"]:
        return 7
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def load_matching_state_dict(model, checkpoint_path, device):
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    current = model.state_dict()
    filtered = {}
    skipped = []
    for key, value in state.items():
        clean_key = key[7:] if key.startswith("module.") else key
        if clean_key in current and current[clean_key].shape == value.shape:
            filtered[clean_key] = value
        else:
            skipped.append(clean_key)
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"Loaded {len(filtered)} checkpoint tensors from {checkpoint_path}")
    if skipped:
        print(f"Skipped {len(skipped)} mismatched checkpoint tensors")
    if missing:
        print(f"Missing tensors after partial load: {len(missing)}")
    if unexpected:
        print(f"Unexpected tensors after partial load: {len(unexpected)}")


@torch.no_grad()
def extract_mapped_embeddings(model, train_loader, device):
    model.eval()
    all_reps, all_labels = [], []
    for input_ids, labels in train_loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        _, mask_mapped_outputs, _, _ = model(input_ids, return_mask_output=True)
        all_reps.append(F.normalize(mask_mapped_outputs, dim=-1).cpu())
        all_labels.append(labels.cpu())
    return torch.cat(all_reps, dim=0), torch.cat(all_labels, dim=0)


def spherical_kmeans(samples, num_clusters, iters=30):
    samples = F.normalize(samples.float(), dim=-1)
    n_samples = samples.shape[0]
    k = min(num_clusters, n_samples)
    init_idx = torch.linspace(0, n_samples - 1, steps=k).round().long()
    centers = samples[init_idx].clone()
    assignments = torch.zeros(n_samples, dtype=torch.long)

    for _ in range(iters):
        sim = torch.matmul(samples, centers.t())
        assignments = sim.argmax(dim=1)
        new_centers = []
        for cluster_id in range(k):
            members = samples[assignments == cluster_id]
            if members.numel() == 0:
                new_centers.append(centers[cluster_id])
            else:
                new_centers.append(F.normalize(members.mean(dim=0), dim=0))
        new_centers = torch.stack(new_centers, dim=0)
        if torch.allclose(new_centers, centers, atol=1e-5):
            centers = new_centers
            break
        centers = new_centers

    counts = torch.zeros(num_clusters, dtype=torch.long)
    for cluster_id in range(k):
        counts[cluster_id] = int((assignments == cluster_id).sum())
    if k < num_clusters:
        pad = centers[:1].repeat(num_clusters - k, 1)
        centers = torch.cat([centers, pad], dim=0)
    return F.normalize(centers, dim=-1), counts


def build_cluster_anchors(all_reps, all_labels, num_classes, num_subanchors, iters):
    anchors, counts = [], []
    for class_id in range(num_classes):
        cls_reps = all_reps[all_labels == class_id]
        if cls_reps.shape[0] == 0:
            raise ValueError(f"No samples found for class {class_id}")
        centers, cls_counts = spherical_kmeans(cls_reps, num_subanchors, iters=iters)
        anchors.append(centers)
        counts.append(cls_counts)
    return torch.stack(anchors, dim=0), torch.stack(counts, dim=0)


def main():
    args = get_parser()
    args.disable_anchor_updates = True
    seed_everything(args.seed)
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    device = f"cuda:{args.gpu_id}" if args.cuda else "cpu"
    if args.cuda:
        torch.cuda.set_device(args.gpu_id)

    args.use_cluster_anchors = False
    args.cluster_anchor_path = None
    args.freeze_cluster_anchors = False

    tokenizer = AutoTokenizer.from_pretrained(args.bert_path, local_files_only=True)
    tokenizer.add_tokens("<mask>")
    trainset = DialogueDataset(args, dataset_name=args.dataset_name, split="train", tokenizer=tokenizer)
    train_loader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = CLModel(args, get_num_classes(args.dataset_name), tokenizer).to(device)
    load_matching_state_dict(model, args.checkpoint_path, device)

    reps, labels = extract_mapped_embeddings(model, train_loader, device)
    anchors, counts = build_cluster_anchors(
        reps,
        labels,
        get_num_classes(args.dataset_name),
        args.num_subanchors,
        args.kmeans_iters,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.dataset_name}_cluster_{args.num_subanchors}.pt"
    torch.save(
        {
            "anchors": anchors,
            "counts": counts,
            "dataset_name": args.dataset_name,
            "num_subanchors": args.num_subanchors,
            "space": "mapped",
            "normalize": True,
            "checkpoint_path": args.checkpoint_path,
        },
        output_path,
    )
    print(f"Saved cluster anchors to {output_path}")
    print(f"anchors shape: {tuple(anchors.shape)}")
    print(f"counts: {counts.tolist()}")


if __name__ == "__main__":
    main()
