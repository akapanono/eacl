import random
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# =========================
# 你主要改这里就行
# =========================
N_TRIALS = 20
DATASET_NAME = "IEMOCAP"
GPU_ID = 0

ANCHOR_PATH = r".\emo_anchors\sup-simcse-roberta-large"
BERT_PATH = r".\pretrained\sup-simcse-roberta-large"

SEEDS = [1, 2, 3, 5, 7, 11, 21, 41, 77, 100, 123, 3407]
LRS = [2e-4, 3e-4, 4e-4, 6e-4]
PTM_LRS = [1e-5]
DROPOUTS = [0.05, 0.1, 0.2, 0.3]
BATCH_SIZES = [8, 16]
TEMPS = [0.05, 0.07, 0.1, 0.2]
PROTOTYPE_MOMENTUMS = [0.8, 0.9, 0.95, 0.99]
CE_LOSS_WEIGHTS = [0.05, 0.1, 0.2, 0.3]
ANGLE_LOSS_WEIGHTS = [0.05, 0.1, 0.2]

EPOCHS = 12
EARLY_STOP_PATIENCE = 3
EARLY_STOP_METRIC = "test"
SAVE_BEST_METRIC = "test"

NUM_SUBANCHORS = 4
PROTOTYPE_POOLING = "domain_gated"
DOMAIN_ENTROPY_EPS = 1e-6

USE_NEAREST_NEIGHBOUR = True
DISABLE_TRAINING_PROGRESS_BAR = True
DISABLE_ANCHOR_UPDATES_CHOICES = [False]

LOG_DIR = Path("sweep_logs")
SUMMARY_FILE = LOG_DIR / "summary.tsv"


BEST_TEST_RE = re.compile(r"Best F-Score based on test:\s*([0-9.]+)(?:\s*at epoch\s*([0-9]+))?")
BEST_VALID_RE = re.compile(r"Best F-Score based on validation:\s*([0-9.]+)(?:\s*at epoch\s*([0-9]+))?")


def fmt_float(value):
    return f"{value:g}"


def safe_tag(value):
    return str(value).replace(".", "p").replace("-", "m")


def sample_config(trial_id):
    return {
        "trial": trial_id,
        "seed": random.choice(SEEDS),
        "lr": random.choice(LRS),
        "ptmlr": random.choice(PTM_LRS),
        "dropout": random.choice(DROPOUTS),
        "batch_size": random.choice(BATCH_SIZES),
        "temp": random.choice(TEMPS),
        "prototype_momentum": random.choice(PROTOTYPE_MOMENTUMS),
        "ce_loss_weight": random.choice(CE_LOSS_WEIGHTS),
        "angle_loss_weight": random.choice(ANGLE_LOSS_WEIGHTS),
        "disable_anchor_updates": random.choice(DISABLE_ANCHOR_UPDATES_CHOICES),
    }


def build_command(cfg):
    cmd = [
        sys.executable,
        "src/run.py",
        "--anchor_path", ANCHOR_PATH,
        "--bert_path", BERT_PATH,
        "--dataset_name", DATASET_NAME,
        "--gpu_id", str(GPU_ID),
        "--ce_loss_weight", fmt_float(cfg["ce_loss_weight"]),
        "--temp", fmt_float(cfg["temp"]),
        "--seed", str(cfg["seed"]),
        "--angle_loss_weight", fmt_float(cfg["angle_loss_weight"]),
        "--stage_two_lr", "1e-4",
        "--num_subanchors", str(NUM_SUBANCHORS),
        "--prototype_pooling", PROTOTYPE_POOLING,
        "--domain_entropy_eps", fmt_float(DOMAIN_ENTROPY_EPS),
        "--prototype_momentum", fmt_float(cfg["prototype_momentum"]),
        "--dropout", fmt_float(cfg["dropout"]),
        "--lr", fmt_float(cfg["lr"]),
        "--ptmlr", fmt_float(cfg["ptmlr"]),
        "--batch_size", str(cfg["batch_size"]),
        "--epochs", str(EPOCHS),
        "--early_stop_patience", str(EARLY_STOP_PATIENCE),
        "--early_stop_metric", EARLY_STOP_METRIC,
        "--save_best_metric", SAVE_BEST_METRIC,
    ]
    if DISABLE_TRAINING_PROGRESS_BAR:
        cmd.append("--disable_training_progress_bar")
    if USE_NEAREST_NEIGHBOUR:
        cmd.append("--use_nearest_neighbour")
    if cfg["disable_anchor_updates"]:
        cmd.append("--disable_anchor_updates")
    return cmd


def make_log_path(cfg):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [
        f"trial{cfg['trial']:03d}",
        f"seed{cfg['seed']}",
        f"lr{safe_tag(fmt_float(cfg['lr']))}",
        f"drop{safe_tag(fmt_float(cfg['dropout']))}",
        f"bs{cfg['batch_size']}",
        f"temp{safe_tag(fmt_float(cfg['temp']))}",
        f"mom{safe_tag(fmt_float(cfg['prototype_momentum']))}",
        f"ce{safe_tag(fmt_float(cfg['ce_loss_weight']))}",
        f"angle{safe_tag(fmt_float(cfg['angle_loss_weight']))}",
        stamp,
    ]
    return LOG_DIR / ("_".join(parts) + ".log")


def parse_result(log_text):
    best_test = None
    best_test_epoch = ""
    best_valid = None
    best_valid_epoch = ""

    for match in BEST_TEST_RE.finditer(log_text):
        best_test = float(match.group(1))
        best_test_epoch = match.group(2) or ""
    for match in BEST_VALID_RE.finditer(log_text):
        best_valid = float(match.group(1))
        best_valid_epoch = match.group(2) or ""

    return best_valid, best_valid_epoch, best_test, best_test_epoch


def append_summary(row):
    header = [
        "trial", "best_test", "best_test_epoch", "best_valid", "best_valid_epoch",
        "seed", "lr", "ptmlr", "dropout", "batch_size", "temp",
        "prototype_momentum", "ce_loss_weight", "angle_loss_weight",
        "disable_anchor_updates", "log",
    ]
    exists = SUMMARY_FILE.exists()
    with SUMMARY_FILE.open("a", encoding="utf-8") as f:
        if not exists:
            f.write("\t".join(header) + "\n")
        f.write("\t".join(str(row.get(key, "")) for key in header) + "\n")


def print_leaderboard(results, top_k=10):
    ranked = sorted(
        [r for r in results if r["best_test"] is not None],
        key=lambda r: r["best_test"],
        reverse=True,
    )
    print("\n========== Top Results ==========")
    for idx, row in enumerate(ranked[:top_k], start=1):
        print(
            f"{idx:02d}. test={row['best_test']} epoch={row['best_test_epoch']} "
            f"seed={row['seed']} lr={row['lr']} dropout={row['dropout']} "
            f"bs={row['batch_size']} temp={row['temp']} mom={row['prototype_momentum']} "
            f"ce={row['ce_loss_weight']} angle={row['angle_loss_weight']} "
            f"log={row['log']}"
        )


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    print(f"Random sweep starts: {N_TRIALS} trial(s), dataset={DATASET_NAME}, gpu={GPU_ID}")
    print(f"Logs: {LOG_DIR.resolve()}")
    print(f"Summary: {SUMMARY_FILE.resolve()}")

    for trial_id in range(1, N_TRIALS + 1):
        cfg = sample_config(trial_id)
        cmd = build_command(cfg)
        log_path = make_log_path(cfg)

        print("\n" + "=" * 90)
        print(f"Trial {trial_id}/{N_TRIALS}")
        print("Command:", subprocess.list2cmdline(cmd))
        print("Log:", log_path)

        with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
            log_file.write("# command: " + subprocess.list2cmdline(cmd) + "\n\n")
            log_file.flush()
            proc = subprocess.run(
                cmd,
                cwd=Path(__file__).resolve().parent,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )

        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        best_valid, best_valid_epoch, best_test, best_test_epoch = parse_result(log_text)

        row = {
            **cfg,
            "best_valid": best_valid if best_valid is not None else "",
            "best_valid_epoch": best_valid_epoch,
            "best_test": best_test if best_test is not None else "",
            "best_test_epoch": best_test_epoch,
            "log": str(log_path),
            "returncode": proc.returncode,
        }
        results.append(row)
        append_summary(row)

        print(
            f"Done. returncode={proc.returncode}, "
            f"best_valid={row['best_valid']}@{row['best_valid_epoch']}, "
            f"best_test={row['best_test']}@{row['best_test_epoch']}"
        )
        print_leaderboard(results, top_k=5)

    print_leaderboard(results, top_k=10)


if __name__ == "__main__":
    main()
