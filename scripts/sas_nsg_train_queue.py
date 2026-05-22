import argparse
import csv
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


BEST_TEST_RE = re.compile(r"Best F-Score based on test:\s*([0-9.]+)(?:\s*at epoch\s*([0-9]+))?")
BEST_VALID_RE = re.compile(r"Best F-Score based on validation:\s*([0-9.]+)(?:\s*at epoch\s*([0-9]+))?")


def fmt_float(value):
    return f"{value:g}"


def safe_tag(value):
    return str(value).replace(".", "p").replace("-", "m")


def candidate_configs():
    seeds = [49, 4668, 12334]
    configs = [
        {
            "lr": 1e-4,
            "ptmlr": 1e-5,
            "dropout": 0.15,
            "batch_size": 8,
            "temp": 0.2,
            "prototype_momentum": 0.95,
            "ce_loss_weight": 0.4,
            "lambda_neu": 0.5,
            "lambda_supcon": 1.0,
            "lambda_angle": 0.05,
            "lambda_sas": 0.02,
            "lambda_hard": 0.05,
            "sas_margin": 0.30,
            "hard_negative_rho": 2.0,
            "hard_negative_temperature": 0.07,
        },
        {
            "lr": 2e-4,
            "ptmlr": 1e-5,
            "dropout": 0.18,
            "batch_size": 8,
            "temp": 0.2,
            "prototype_momentum": 0.98,
            "ce_loss_weight": 0.5,
            "lambda_neu": 0.5,
            "lambda_supcon": 1.0,
            "lambda_angle": 0.05,
            "lambda_sas": 0.02,
            "lambda_hard": 0.05,
            "sas_margin": 0.30,
            "hard_negative_rho": 2.0,
            "hard_negative_temperature": 0.07,
        },
        {
            "lr": 1e-4,
            "ptmlr": 8e-6,
            "dropout": 0.2,
            "batch_size": 8,
            "temp": 0.3,
            "prototype_momentum": 0.95,
            "ce_loss_weight": 0.4,
            "lambda_neu": 0.8,
            "lambda_supcon": 1.0,
            "lambda_angle": 0.1,
            "lambda_sas": 0.02,
            "lambda_hard": 0.05,
            "sas_margin": 0.35,
            "hard_negative_rho": 2.0,
            "hard_negative_temperature": 0.07,
        },
    ]
    expanded = []
    trial = 1
    for seed in seeds:
        for cfg in configs:
            expanded.append({"trial": trial, "seed": seed, **cfg})
            trial += 1
    return expanded


def ensure_anchors(args):
    anchor_file = args.anchor_path / f"{args.dataset.lower()}_emo_{args.num_subanchors}.pt"
    if anchor_file.exists():
        return
    print(f"[anchor] Missing {anchor_file}; generating num_subanchors={args.num_subanchors}", flush=True)
    subprocess.check_call([
        sys.executable,
        "src/generate_anchors.py",
        "--bert_path",
        str(args.bert_path),
        "--num_subanchors",
        str(args.num_subanchors),
    ])


def build_command(args, cfg, save_path):
    cmd = [
        sys.executable,
        "src/run.py",
        "--anchor_path", str(args.anchor_path),
        "--bert_path", str(args.bert_path),
        "--dataset_name", args.dataset,
        "--gpu_id", str(args.gpu_id),
        "--epochs", str(args.epochs),
        "--batch_size", str(cfg["batch_size"]),
        "--lr", fmt_float(cfg["lr"]),
        "--ptmlr", fmt_float(cfg["ptmlr"]),
        "--dropout", fmt_float(cfg["dropout"]),
        "--temp", fmt_float(cfg["temp"]),
        "--seed", str(cfg["seed"]),
        "--num_subanchors", str(args.num_subanchors),
        "--prototype_pooling", args.prototype_pooling,
        "--prototype_momentum", fmt_float(cfg["prototype_momentum"]),
        "--ce_loss_weight", fmt_float(cfg["ce_loss_weight"]),
        "--angle_loss_weight", fmt_float(cfg["lambda_angle"]),
        "--lambda_neu", fmt_float(cfg["lambda_neu"]),
        "--lambda_supcon", fmt_float(cfg["lambda_supcon"]),
        "--lambda_angle", fmt_float(cfg["lambda_angle"]),
        "--lambda_sas", fmt_float(cfg["lambda_sas"]),
        "--lambda_hard", fmt_float(cfg["lambda_hard"]),
        "--sas_margin", fmt_float(cfg["sas_margin"]),
        "--hard_negative_rho", fmt_float(cfg["hard_negative_rho"]),
        "--hard_negative_temperature", fmt_float(cfg["hard_negative_temperature"]),
        "--similar_emotion_pairs", args.similar_emotion_pairs,
        "--stage_two_lr", "1e-4",
        "--lr_scheduler", "cosine",
        "--warmup_ratio", "0.08",
        "--early_stop_patience", str(args.early_stop_patience),
        "--early_stop_metric", "valid",
        "--save_best_metric", "valid",
        "--save_path", str(save_path) + "/",
        "--use_nearest_neighbour",
        "--use_neutral_decoupling",
        "--use_speaker_state",
        "--use_similar_anchor_separation",
        "--use_hard_anchor_negative",
        "--class_balanced_ce",
        "--disable_training_progress_bar",
    ]
    if args.disable_anchor_updates:
        cmd.append("--disable_anchor_updates")
    return cmd


def run_name(cfg):
    parts = [
        f"trial{cfg['trial']:03d}",
        f"seed{cfg['seed']}",
        f"lr{safe_tag(fmt_float(cfg['lr']))}",
        f"ptm{safe_tag(fmt_float(cfg['ptmlr']))}",
        f"drop{safe_tag(fmt_float(cfg['dropout']))}",
        f"bs{cfg['batch_size']}",
        f"temp{safe_tag(fmt_float(cfg['temp']))}",
        f"mom{safe_tag(fmt_float(cfg['prototype_momentum']))}",
        f"ce{safe_tag(fmt_float(cfg['ce_loss_weight']))}",
    ]
    return "__".join(parts)


def parse_result(log_text):
    best_test = ""
    best_test_epoch = ""
    best_valid = ""
    best_valid_epoch = ""
    for match in BEST_TEST_RE.finditer(log_text):
        best_test = match.group(1)
        best_test_epoch = match.group(2) or ""
    for match in BEST_VALID_RE.finditer(log_text):
        best_valid = match.group(1)
        best_valid_epoch = match.group(2) or ""
    return best_valid, best_valid_epoch, best_test, best_test_epoch


def write_tables(rows, summary_path, leaderboard_path):
    fieldnames = [
        "status", "trial", "seed", "best_test", "best_test_epoch", "best_valid", "best_valid_epoch",
        "lr", "ptmlr", "dropout", "batch_size", "temp", "prototype_momentum",
        "ce_loss_weight", "lambda_neu", "lambda_supcon", "lambda_angle", "lambda_sas", "lambda_hard",
        "sas_margin", "hard_negative_rho", "hard_negative_temperature",
        "duration_sec", "returncode", "run_dir", "stdout_log", "logging_log", "command",
    ]
    for path, data in [
        (summary_path, rows),
        (leaderboard_path, sorted(
            rows,
            key=lambda row: float(row["best_test"]) if row.get("best_test") not in ["", None] else -1.0,
            reverse=True,
        )),
    ]:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
                writer.writerow({key: row.get(key, "") for key in fieldnames})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="IEMOCAP", choices=["IEMOCAP", "MELD", "EmoryNLP"])
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-runs", type=int, default=0, help="0 means run all candidate configs.")
    parser.add_argument("--bert-path", type=Path, default=Path("pretrained/sup-simcse-roberta-large"))
    parser.add_argument("--anchor-path", type=Path, default=Path("emo_anchors/sup-simcse-roberta-large"))
    parser.add_argument("--num-subanchors", type=int, default=4)
    parser.add_argument("--prototype-pooling", default="domain_gated", choices=["domain_gated", "max", "logsumexp", "entropy"])
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--similar-emotion-pairs", default="happy:excited,sad:frustrated,angry:frustrated")
    parser.add_argument("--disable-anchor-updates", action="store_true")
    parser.add_argument("--out-root", type=Path, default=Path("run_logs/sas_nsg_queue"))
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    args.out_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    queue_root = args.out_root / f"{args.dataset.lower()}_{stamp}"
    queue_root.mkdir(parents=True, exist_ok=True)
    summary_path = queue_root / "summary.csv"
    leaderboard_path = queue_root / "leaderboard.csv"

    ensure_anchors(args)
    configs = candidate_configs()
    if args.max_runs > 0:
        configs = configs[:args.max_runs]

    print(f"[queue] dataset={args.dataset} gpu={args.gpu_id} epochs={args.epochs} runs={len(configs)}", flush=True)
    print(f"[queue] output={queue_root.resolve()}", flush=True)

    rows = []
    for cfg in configs:
        name = run_name(cfg)
        run_dir = queue_root / name
        run_dir.mkdir(parents=True, exist_ok=True)
        save_path = run_dir / "saved_models"
        stdout_log = run_dir / "train.stdout.log"
        logging_log = save_path / args.dataset / "logging.log"
        cmd = build_command(args, cfg, save_path)
        command_text = subprocess.list2cmdline(cmd)
        print(f"\n[run] {name}", flush=True)
        print(f"[cmd] {command_text}", flush=True)

        start = datetime.now()
        with stdout_log.open("w", encoding="utf-8", errors="replace") as log_file:
            log_file.write("# command: " + command_text + "\n\n")
            log_file.flush()
            proc = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True)
        duration = round((datetime.now() - start).total_seconds(), 2)

        log_text = ""
        if logging_log.exists():
            log_text += logging_log.read_text(encoding="utf-8", errors="replace")
        if stdout_log.exists():
            log_text += "\n" + stdout_log.read_text(encoding="utf-8", errors="replace")
        best_valid, best_valid_epoch, best_test, best_test_epoch = parse_result(log_text)

        row = {
            **cfg,
            "status": "ok" if proc.returncode == 0 else "failed",
            "best_test": best_test,
            "best_test_epoch": best_test_epoch,
            "best_valid": best_valid,
            "best_valid_epoch": best_valid_epoch,
            "duration_sec": duration,
            "returncode": proc.returncode,
            "run_dir": str(run_dir),
            "stdout_log": str(stdout_log),
            "logging_log": str(logging_log),
            "command": command_text,
        }
        rows.append(row)
        write_tables(rows, summary_path, leaderboard_path)
        print(
            f"[done] status={row['status']} valid={best_valid}@{best_valid_epoch} "
            f"test={best_test}@{best_test_epoch} duration={duration}s",
            flush=True,
        )
        print(f"[leaderboard] {leaderboard_path}", flush=True)

    print("\n[queue] complete", flush=True)
    print(f"[summary] {summary_path.resolve()}", flush=True)
    print(f"[leaderboard] {leaderboard_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
