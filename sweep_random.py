import csv
import random
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# cd G:\xhy\eacl
# conda activate eacl
# powershell -ExecutionPolicy Bypass -File .\launch_sweep_background.ps1 -PythonExe "C:\Users\Administrator\miniconda3\envs\eacl\python.exe"

N_TRIALS = 100
DATASET_NAME = "MELD"
GPU_IDS = [0]
MAX_PARALLEL_JOBS = 1
AUTO_SCHEDULE_BY_GPU = True
GPU_UTIL_THRESHOLD = 90
MIN_FREE_MEMORY_MB = 5000
POLL_SECONDS = 15
STOP_ON_ERROR = False
USE_IMPROVED_TRAINING = True

ANCHOR_PATH = str(Path("emo_anchors") / "sup-simcse-roberta-large")
BERT_PATH = str(Path("pretrained") / "sup-simcse-roberta-large")

SEEDS = [49,4668,12334,4998,5684]
LRS = [3e-5, 5e-5, 8e-5, 1e-4]
PTM_LRS = [3e-6, 5e-6, 8e-6, 1e-5]
DROPOUTS = [0.2, 0.25, 0.3]
BATCH_SIZES = [8, 16]
TEMPS = [0.2, 0.3, 0.5]
PROTOTYPE_MOMENTUMS = [0.99, 0.995]
MAX_GRAD_NORMS = [0.5, 1.0]
FREEZE_PROTOTYPE_EPOCHS = [2, 3]
CE_LOSS_WEIGHTS = [0.3, 0.4, 0.5, 0.6]
ANGLE_LOSS_WEIGHTS = [0.005, 0.01, 0.02]
LAMBDA_NEUS = [0.1, 0.2, 0.3]
LAMBDA_SUPCONS = [0.1, 0.2, 0.5]
LAMBDA_ANGLES = [0.005, 0.01, 0.02]
LAMBDA_SASES = [0.002, 0.005]
LAMBDA_HARDS = [0.0, 0.005, 0.01]
LAMBDA_GATE_ENTROPIES = [0.0, 0.001, 0.002]
LR_SCHEDULER = "cosine"
WARMUP_RATIO = 0.08
STEP_LR_SIZE = 5
STEP_LR_GAMMA = 0.8
CLASS_BALANCED_CE = True

EPOCHS = 30
EARLY_STOP_PATIENCE = 5
EARLY_STOP_METRIC = "valid"
SAVE_BEST_METRIC = "valid"

NUM_SUBANCHORS = 4
PROTOTYPE_POOLINGS = ["domain_gated", "logsumexp", "max"]
DOMAIN_ENTROPY_EPS = 1e-6
DOMAIN_ANCHOR_VARIANTS = 1
DOMAIN_VARIANT_POOLINGS = ["logsumexp"]
DOMAIN_VARIANT_TEMPS = [0.2]

USE_NEAREST_NEIGHBOUR = True
DISABLE_TRAINING_PROGRESS_BAR = True
DISABLE_ANCHOR_UPDATES_CHOICES = [False, True]
USE_NEUTRAL_DECOUPLING_CHOICES = [False, True]
USE_SPEAKER_STATE_CHOICES = [False, True]
USE_SIMILAR_ANCHOR_SEPARATION_CHOICES = [False, True]
USE_HARD_ANCHOR_NEGATIVE_CHOICES = [False, True]
SIMILAR_EMOTION_PAIRS = "happy:excited,sad:frustrated,angry:frustrated"
SAS_MARGINS = [0.25, 0.3, 0.35]
HARD_NEGATIVE_RHOS = [0.5, 1.0]
HARD_NEGATIVE_TEMPS = [0.1, 0.2]

if not USE_IMPROVED_TRAINING:
    LRS = [1e-4, 2e-4, 3e-4, 4e-4, 1e-5]
    PTM_LRS = [1e-5]
    CE_LOSS_WEIGHTS = [0.1, 0.2, 0.3]
    LR_SCHEDULER = "step"
    WARMUP_RATIO = 0.0
    STEP_LR_SIZE = 1
    STEP_LR_GAMMA = 0.5
    CLASS_BALANCED_CE = False
    PROTOTYPE_POOLINGS = ["domain_gated"]
    DISABLE_ANCHOR_UPDATES_CHOICES = [False]
    USE_NEUTRAL_DECOUPLING_CHOICES = [False]
    USE_SPEAKER_STATE_CHOICES = [False]
    USE_SIMILAR_ANCHOR_SEPARATION_CHOICES = [False]
    USE_HARD_ANCHOR_NEGATIVE_CHOICES = [False]

LOG_DIR = Path("sweep_logs")
SUMMARY_FILE = LOG_DIR / "summary.tsv"
SUMMARY_CSV_FILE = LOG_DIR / "summary.csv"


BEST_TEST_RE = re.compile(r"Best F-Score based on test:\s*([0-9.]+)(?:\s*at epoch\s*([0-9]+))?")
BEST_VALID_RE = re.compile(r"Best F-Score based on validation:\s*([0-9.]+)(?:\s*at epoch\s*([0-9]+))?")


def ensure_anchor_file():
    anchor_file = Path(ANCHOR_PATH) / f"{DATASET_NAME.lower()}_emo_{NUM_SUBANCHORS}.pt"
    if anchor_file.exists():
        return
    print(f"Missing anchor file: {anchor_file}")
    print(f"Generating anchors with num_subanchors={NUM_SUBANCHORS} ...")
    subprocess.check_call([
        sys.executable,
        "src/generate_anchors.py",
        "--bert_path",
        BERT_PATH,
        "--num_subanchors",
        str(NUM_SUBANCHORS),
    ])


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
        "max_grad_norm": random.choice(MAX_GRAD_NORMS),
        "freeze_prototype_epochs": random.choice(FREEZE_PROTOTYPE_EPOCHS),
        "ce_loss_weight": random.choice(CE_LOSS_WEIGHTS),
        "angle_loss_weight": random.choice(ANGLE_LOSS_WEIGHTS),
        "lambda_neu": random.choice(LAMBDA_NEUS),
        "lambda_supcon": random.choice(LAMBDA_SUPCONS),
        "lambda_angle": random.choice(LAMBDA_ANGLES),
        "lambda_sas": random.choice(LAMBDA_SASES),
        "lambda_hard": random.choice(LAMBDA_HARDS),
        "lambda_gate_entropy": random.choice(LAMBDA_GATE_ENTROPIES),
        "prototype_pooling": random.choice(PROTOTYPE_POOLINGS),
        "disable_anchor_updates": random.choice(DISABLE_ANCHOR_UPDATES_CHOICES),
        "use_neutral_decoupling": random.choice(USE_NEUTRAL_DECOUPLING_CHOICES),
        "use_speaker_state": random.choice(USE_SPEAKER_STATE_CHOICES),
        "use_similar_anchor_separation": random.choice(USE_SIMILAR_ANCHOR_SEPARATION_CHOICES),
        "use_hard_anchor_negative": random.choice(USE_HARD_ANCHOR_NEGATIVE_CHOICES),
        "sas_margin": random.choice(SAS_MARGINS),
        "hard_negative_rho": random.choice(HARD_NEGATIVE_RHOS),
        "hard_negative_temperature": random.choice(HARD_NEGATIVE_TEMPS),
    }


def build_command(cfg):
    cmd = [
        sys.executable,
        "src/run.py",
        "--anchor_path", ANCHOR_PATH,
        "--bert_path", BERT_PATH,
        "--dataset_name", DATASET_NAME,
        "--gpu_id", str(cfg["gpu_id"]),
        "--ce_loss_weight", fmt_float(cfg["ce_loss_weight"]),
        "--temp", fmt_float(cfg["temp"]),
        "--seed", str(cfg["seed"]),
        "--angle_loss_weight", fmt_float(cfg["angle_loss_weight"]),
        "--lambda_neu", fmt_float(cfg["lambda_neu"]),
        "--lambda_supcon", fmt_float(cfg["lambda_supcon"]),
        "--lambda_angle", fmt_float(cfg["lambda_angle"]),
        "--lambda_sas", fmt_float(cfg["lambda_sas"]),
        "--lambda_hard", fmt_float(cfg["lambda_hard"]),
        "--stage_two_lr", "1e-4",
        "--num_subanchors", str(NUM_SUBANCHORS),
        "--prototype_pooling", cfg["prototype_pooling"],
        "--domain_entropy_eps", fmt_float(DOMAIN_ENTROPY_EPS),
        "--prototype_momentum", fmt_float(cfg["prototype_momentum"]),
        "--max_grad_norm", fmt_float(cfg["max_grad_norm"]),
        "--freeze_prototype_epochs", str(cfg["freeze_prototype_epochs"]),
        "--similar_emotion_pairs", SIMILAR_EMOTION_PAIRS,
        "--sas_margin", fmt_float(cfg["sas_margin"]),
        "--hard_negative_rho", fmt_float(cfg["hard_negative_rho"]),
        "--hard_negative_temperature", fmt_float(cfg["hard_negative_temperature"]),
        "--lambda_gate_entropy", fmt_float(cfg["lambda_gate_entropy"]),
        "--dropout", fmt_float(cfg["dropout"]),
        "--lr", fmt_float(cfg["lr"]),
        "--ptmlr", fmt_float(cfg["ptmlr"]),
        "--batch_size", str(cfg["batch_size"]),
        "--epochs", str(EPOCHS),
        "--lr_scheduler", LR_SCHEDULER,
        "--step_lr_size", str(STEP_LR_SIZE),
        "--step_lr_gamma", fmt_float(STEP_LR_GAMMA),
        "--warmup_ratio", fmt_float(WARMUP_RATIO),
        "--early_stop_patience", str(EARLY_STOP_PATIENCE),
        "--early_stop_metric", EARLY_STOP_METRIC,
        "--save_best_metric", SAVE_BEST_METRIC,
    ]
    if DISABLE_TRAINING_PROGRESS_BAR:
        cmd.append("--disable_training_progress_bar")
    if USE_NEAREST_NEIGHBOUR:
        cmd.append("--use_nearest_neighbour")
    if CLASS_BALANCED_CE:
        cmd.append("--class_balanced_ce")
    if cfg["disable_anchor_updates"]:
        cmd.append("--disable_anchor_updates")
    if cfg["use_neutral_decoupling"]:
        cmd.append("--use_neutral_decoupling")
    if cfg["use_speaker_state"]:
        cmd.append("--use_speaker_state")
    if cfg["use_similar_anchor_separation"]:
        cmd.append("--use_similar_anchor_separation")
    if cfg["use_hard_anchor_negative"]:
        cmd.append("--use_hard_anchor_negative")
    cmd.append("--normalize_prototypes_after_update")
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
        f"pool{cfg['prototype_pooling']}",
        f"mom{safe_tag(fmt_float(cfg['prototype_momentum']))}",
        f"ce{safe_tag(fmt_float(cfg['ce_loss_weight']))}",
        f"angle{safe_tag(fmt_float(cfg['angle_loss_weight']))}",
        f"nd{int(cfg['use_neutral_decoupling'])}",
        f"sg{int(cfg['use_speaker_state'])}",
        f"sas{int(cfg['use_similar_anchor_separation'])}",
        f"hard{int(cfg['use_hard_anchor_negative'])}",
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
        "status", "start_time", "end_time", "duration_sec",
        "trial", "best_test", "best_test_epoch", "best_valid", "best_valid_epoch",
        "seed", "lr", "ptmlr", "dropout", "batch_size", "temp",
        "prototype_pooling", "prototype_momentum", "max_grad_norm", "freeze_prototype_epochs",
        "ce_loss_weight", "angle_loss_weight", "lambda_neu", "lambda_supcon", "lambda_angle",
        "lambda_sas", "lambda_hard", "lambda_gate_entropy",
        "lr_scheduler", "warmup_ratio", "class_balanced_ce",
        "use_neutral_decoupling", "use_speaker_state", "use_similar_anchor_separation",
        "use_hard_anchor_negative", "sas_margin", "hard_negative_rho", "hard_negative_temperature",
        "disable_anchor_updates", "gpu_id", "returncode", "command", "log",
    ]
    exists = SUMMARY_FILE.exists()
    with SUMMARY_FILE.open("a", encoding="utf-8") as f:
        if not exists:
            f.write("\t".join(header) + "\n")
        f.write("\t".join(str(row.get(key, "")) for key in header) + "\n")
    csv_exists = SUMMARY_CSV_FILE.exists()
    with SUMMARY_CSV_FILE.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not csv_exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in header})


def print_leaderboard(results, top_k=10):
    ranked = sorted(
        [r for r in results if r["best_test"] not in [None, ""] and r["returncode"] == 0],
        key=lambda r: r["best_test"],
        reverse=True,
    )
    print("\n========== Top Results ==========")
    if not ranked:
        print("No successful runs yet.")
        return
    for idx, row in enumerate(ranked[:top_k], start=1):
        print(
            f"{idx:02d}. test={row['best_test']} epoch={row['best_test_epoch']} "
            f"seed={row['seed']} lr={row['lr']} dropout={row['dropout']} "
            f"bs={row['batch_size']} temp={row['temp']} pool={row['prototype_pooling']} "
            f"mom={row['prototype_momentum']} "
            f"ce={row['ce_loss_weight']} angle={row['angle_loss_weight']} "
            f"nd={row['use_neutral_decoupling']} sg={row['use_speaker_state']} "
            f"sas={row['use_similar_anchor_separation']} hard={row['use_hard_anchor_negative']} "
            f"gpu={row['gpu_id']} "
            f"log={row['log']}"
        )


def query_gpu_status():
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                f"--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception:
        return {}

    status = {}
    for line in output.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        idx, util, mem_used, mem_total = map(int, parts)
        status[idx] = {
            "util": util,
            "mem_used": mem_used,
            "mem_total": mem_total,
            "mem_free": mem_total - mem_used,
        }
    return status


def can_launch_on_gpu(gpu_id, running_jobs):
    if sum(1 for job in running_jobs if job["gpu_id"] == gpu_id) >= MAX_PARALLEL_JOBS:
        return False
    if not AUTO_SCHEDULE_BY_GPU:
        return True
    status = query_gpu_status().get(gpu_id)
    if status is None:
        return True
    return status["util"] < GPU_UTIL_THRESHOLD and status["mem_free"] >= MIN_FREE_MEMORY_MB


def start_job(cfg):
    cmd = build_command(cfg)
    log_path = make_log_path(cfg)
    command_text = subprocess.list2cmdline(cmd)
    start_time = datetime.now()
    print("\n" + "=" * 90)
    print(f"Trial {cfg['trial']}/{N_TRIALS} -> GPU {cfg['gpu_id']}")
    print("Command:", command_text)
    print("Log:", log_path)

    log_file = log_path.open("w", encoding="utf-8", errors="replace")
    log_file.write("# start_time: " + start_time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    log_file.write("# command: " + command_text + "\n\n")
    log_file.flush()
    proc = subprocess.Popen(
        cmd,
        cwd=Path(__file__).resolve().parent,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return {
        "cfg": cfg,
        "proc": proc,
        "log_path": log_path,
        "log_file": log_file,
        "gpu_id": cfg["gpu_id"],
        "command": command_text,
        "start_time": start_time,
    }


def finish_job(job):
    proc = job["proc"]
    proc.wait()
    end_time = datetime.now()
    job["log_file"].flush()
    job["log_file"].close()
    log_text = job["log_path"].read_text(encoding="utf-8", errors="replace")
    best_valid, best_valid_epoch, best_test, best_test_epoch = parse_result(log_text)
    duration = round((end_time - job["start_time"]).total_seconds(), 2)
    row = {
        **job["cfg"],
        "status": "ok" if proc.returncode == 0 else "failed",
        "start_time": job["start_time"].strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_sec": duration,
        "best_valid": best_valid if best_valid is not None else "",
        "best_valid_epoch": best_valid_epoch,
        "best_test": best_test if best_test is not None else "",
        "best_test_epoch": best_test_epoch,
        "lr_scheduler": LR_SCHEDULER,
        "warmup_ratio": WARMUP_RATIO,
        "class_balanced_ce": CLASS_BALANCED_CE,
        "log": str(job["log_path"]),
        "returncode": proc.returncode,
        "command": job["command"],
    }
    return row


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ensure_anchor_file()
    results = []
    pending = []
    running = []

    print(f"Random sweep starts: {N_TRIALS} trial(s), dataset={DATASET_NAME}, gpus={GPU_IDS}")
    print(f"Logs: {LOG_DIR.resolve()}")
    print(f"Summary: {SUMMARY_FILE.resolve()}")
    print(
        f"Parallel config: max_jobs_per_gpu={MAX_PARALLEL_JOBS}, "
        f"auto_schedule={AUTO_SCHEDULE_BY_GPU}, util<th{GPU_UTIL_THRESHOLD}, free_mem>{MIN_FREE_MEMORY_MB}MB"
    )
    print(
        f"Training config: improved={USE_IMPROVED_TRAINING}, scheduler={LR_SCHEDULER}, "
        f"warmup={WARMUP_RATIO}, class_balanced_ce={CLASS_BALANCED_CE}, "
        f"poolings={PROTOTYPE_POOLINGS}"
    )

    for trial_id in range(1, N_TRIALS + 1):
        cfg = sample_config(trial_id)
        cfg["gpu_id"] = GPU_IDS[(trial_id - 1) % len(GPU_IDS)]
        pending.append(cfg)

    while pending or running:
        launched = False
        for gpu_id in GPU_IDS:
            while pending and can_launch_on_gpu(gpu_id, running):
                next_idx = next((idx for idx, cfg in enumerate(pending) if cfg["gpu_id"] == gpu_id), None)
                if next_idx is None:
                    break
                cfg = pending.pop(next_idx)
                running.append(start_job(cfg))
                launched = True
                if sum(1 for job in running if job["gpu_id"] == gpu_id) >= MAX_PARALLEL_JOBS:
                    break

        finished = []
        for job in running:
            if job["proc"].poll() is not None:
                finished.append(job)

        for job in finished:
            running.remove(job)
            row = finish_job(job)
            results.append(row)
            append_summary(row)
            print(
                f"Done. returncode={row['returncode']}, "
                f"best_valid={row['best_valid']}@{row['best_valid_epoch']}, "
                f"best_test={row['best_test']}@{row['best_test_epoch']}"
            )
            print_leaderboard(results, top_k=5)
            if STOP_ON_ERROR and row["returncode"] != 0:
                print("Stopping sweep because STOP_ON_ERROR=True and a run failed.")
                pending.clear()
                for still_running in running:
                    still_running["proc"].terminate()
                break

        if pending or running:
            if not launched and not finished:
                time.sleep(POLL_SECONDS)

    print_leaderboard(results, top_k=10)


if __name__ == "__main__":
    main()
