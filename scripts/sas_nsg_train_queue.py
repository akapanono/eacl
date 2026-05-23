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


BASE_CONFIG = {
    "lr": 5e-5,
    "ptmlr": 5e-6,
    "dropout": 0.25,
    "batch_size": 8,
    "temp": 0.3,
    "prototype_pooling": "domain_gated",
    "prototype_momentum": 0.995,
    "max_grad_norm": 0.5,
    "freeze_prototype_epochs": 3,
    "ce_loss_weight": 0.4,
    "lambda_neu": 0.2,
    "lambda_supcon": 0.2,
    "lambda_angle": 0.01,
    "lambda_sas": 0.002,
    "lambda_hard": 0.005,
    "lambda_gate_entropy": 0.001,
    "sas_margin": 0.30,
    "hard_negative_rho": 0.5,
    "hard_negative_temperature": 0.2,
    "use_nearest_neighbour": True,
    "use_neutral_decoupling": True,
    "use_speaker_state": True,
    "use_speaker_memory": False,
    "speaker_memory_k": 3,
    "speaker_memory_pooling": "attention",
    "use_similar_anchor_separation": True,
    "use_hard_anchor_negative": True,
    "hard_negative_schedule": "constant",
    "class_balanced_ce": True,
    "disable_anchor_updates": False,
    "use_classifier_prototype_fusion": False,
    "fusion_type": "fixed",
    "fusion_alpha": 0.5,
    "use_neutral_aware_supcon": False,
    "prototype_update_policy": "momentum",
    "accumulation_step": 1,
}


def make_config(name, group, **overrides):
    cfg = {"name": name, "group": group, **BASE_CONFIG}
    cfg.update(overrides)
    return cfg


def candidate_configs():
    configs = [
        make_config("B0_baseline_trial004", "baseline"),
        make_config("A1_no_hard", "ablation", use_hard_anchor_negative=False, lambda_hard=0.0),
        make_config("A2_no_sas_no_hard", "ablation", use_similar_anchor_separation=False, use_hard_anchor_negative=False, lambda_sas=0.0, lambda_hard=0.0),
        make_config("A3_no_speaker_state", "ablation", use_speaker_state=False),
        make_config("A4_no_neutral_decoupling", "ablation", use_neutral_decoupling=False),
        make_config("A5_no_class_balanced_ce", "ablation", class_balanced_ce=False),
        make_config("A6_logsumexp_pooling", "ablation", prototype_pooling="logsumexp"),
        make_config("A7_entropy_pooling", "ablation", prototype_pooling="entropy"),
        make_config("A8_freeze8", "ablation", freeze_prototype_epochs=8),
        make_config("R1_stronger_ce", "targeted", ce_loss_weight=0.6, lambda_supcon=0.1, lambda_neu=0.1, lambda_sas=0.001, lambda_hard=0.002),
        make_config("R2_ce_heavy_no_hard", "targeted", ce_loss_weight=0.7, lambda_supcon=0.05, lambda_neu=0.1, lambda_sas=0.001, lambda_hard=0.0, use_hard_anchor_negative=False),
        make_config("R3_no_hard", "targeted", use_hard_anchor_negative=False, lambda_hard=0.0),
        make_config("R4_no_sas_no_hard", "targeted", use_similar_anchor_separation=False, use_hard_anchor_negative=False, lambda_sas=0.0, lambda_hard=0.0),
        make_config("R5_dropout020", "targeted", dropout=0.20),
        make_config("R6_dropout015", "targeted", dropout=0.15),
        make_config("R7_batch16_temp03", "targeted", batch_size=16, temp=0.3),
        make_config("R8_batch16_temp02", "targeted", batch_size=16, temp=0.2),
        make_config("R9_logsumexp_pooling", "targeted", prototype_pooling="logsumexp"),
        make_config("R10_freeze8", "targeted", freeze_prototype_epochs=8, prototype_momentum=0.995),
        make_config("F1_fusion_alpha03", "fusion", use_classifier_prototype_fusion=True, fusion_alpha=0.3),
        make_config("F2_fusion_alpha05", "fusion", use_classifier_prototype_fusion=True, fusion_alpha=0.5),
        make_config("F3_fusion_alpha07", "fusion", use_classifier_prototype_fusion=True, fusion_alpha=0.7),
        make_config("S1_speaker_memory_mean", "smf", use_speaker_state=False, use_speaker_memory=True, speaker_memory_pooling="mean"),
        make_config("S2_speaker_memory_attention", "smf", use_speaker_state=False, use_speaker_memory=True, speaker_memory_pooling="attention"),
        make_config("S3_adaptive_fusion", "smf", use_classifier_prototype_fusion=True, fusion_type="adaptive"),
        make_config("S4_memory_adaptive_fusion", "smf", use_speaker_state=False, use_speaker_memory=True, use_classifier_prototype_fusion=True, fusion_type="adaptive"),
        make_config("S5_neutral_aware_supcon", "smf", use_neutral_aware_supcon=True),
        make_config("S6_curriculum_hard", "smf", hard_negative_schedule="curriculum"),
        make_config("S7_validation_guarded_proto", "smf", prototype_update_policy="validation_guarded"),
        make_config(
            "S8_smf_full",
            "smf",
            use_speaker_state=False,
            use_speaker_memory=True,
            use_classifier_prototype_fusion=True,
            fusion_type="adaptive",
            use_neutral_aware_supcon=True,
            hard_negative_schedule="curriculum",
            prototype_update_policy="validation_guarded",
        ),
    ]
    return [{"trial": idx, "seed": 4668, **cfg} for idx, cfg in enumerate(configs, start=1)]


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


def filter_configs(configs, experiment_set):
    if experiment_set == "all":
        return configs
    if experiment_set == "ablation":
        return [cfg for cfg in configs if cfg["group"] in ["baseline", "ablation"]]
    if experiment_set == "targeted":
        return [cfg for cfg in configs if cfg["group"] in ["baseline", "targeted"]]
    if experiment_set == "fusion":
        return [cfg for cfg in configs if cfg["group"] in ["baseline", "fusion"]]
    if experiment_set == "smf":
        return [cfg for cfg in configs if cfg["group"] in ["baseline", "smf"]]
    return configs


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
        "--prototype_pooling", cfg["prototype_pooling"],
        "--prototype_momentum", fmt_float(cfg["prototype_momentum"]),
        "--max_grad_norm", fmt_float(cfg["max_grad_norm"]),
        "--freeze_prototype_epochs", str(cfg["freeze_prototype_epochs"]),
        "--ce_loss_weight", fmt_float(cfg["ce_loss_weight"]),
        "--angle_loss_weight", fmt_float(cfg["lambda_angle"]),
        "--lambda_neu", fmt_float(cfg["lambda_neu"]),
        "--lambda_supcon", fmt_float(cfg["lambda_supcon"]),
        "--lambda_angle", fmt_float(cfg["lambda_angle"]),
        "--lambda_sas", fmt_float(cfg["lambda_sas"]),
        "--lambda_hard", fmt_float(cfg["lambda_hard"]),
        "--lambda_gate_entropy", fmt_float(cfg["lambda_gate_entropy"]),
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
        "--accumulation_step", str(cfg["accumulation_step"]),
        "--speaker_memory_k", str(cfg["speaker_memory_k"]),
        "--speaker_memory_pooling", cfg["speaker_memory_pooling"],
        "--fusion_type", cfg["fusion_type"],
        "--hard_negative_schedule", cfg["hard_negative_schedule"],
        "--prototype_update_policy", cfg["prototype_update_policy"],
        "--normalize_prototypes_after_update",
        "--disable_training_progress_bar",
    ]
    if cfg["use_nearest_neighbour"]:
        cmd.append("--use_nearest_neighbour")
    if cfg["use_neutral_decoupling"]:
        cmd.append("--use_neutral_decoupling")
    if cfg["use_speaker_state"]:
        cmd.append("--use_speaker_state")
    if cfg["use_speaker_memory"]:
        cmd.append("--use_speaker_memory")
    if cfg["use_similar_anchor_separation"]:
        cmd.append("--use_similar_anchor_separation")
    if cfg["use_hard_anchor_negative"]:
        cmd.append("--use_hard_anchor_negative")
    if cfg["class_balanced_ce"]:
        cmd.append("--class_balanced_ce")
    if cfg["disable_anchor_updates"] or args.disable_anchor_updates:
        cmd.append("--disable_anchor_updates")
    if cfg["use_classifier_prototype_fusion"]:
        cmd.extend(["--use_classifier_prototype_fusion", "--fusion_alpha", fmt_float(cfg["fusion_alpha"])])
    if cfg["use_neutral_aware_supcon"]:
        cmd.append("--use_neutral_aware_supcon")
    return cmd


def run_name(cfg):
    parts = [
        f"trial{cfg['trial']:03d}",
        cfg["name"],
        f"seed{cfg['seed']}",
        f"lr{safe_tag(fmt_float(cfg['lr']))}",
        f"ptm{safe_tag(fmt_float(cfg['ptmlr']))}",
        f"drop{safe_tag(fmt_float(cfg['dropout']))}",
        f"bs{cfg['batch_size']}",
        f"temp{safe_tag(fmt_float(cfg['temp']))}",
        f"pool{cfg['prototype_pooling']}",
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
        "status", "trial", "name", "group", "seed", "best_test", "best_test_epoch", "best_valid", "best_valid_epoch",
        "lr", "ptmlr", "dropout", "batch_size", "temp", "prototype_momentum",
        "max_grad_norm", "freeze_prototype_epochs", "ce_loss_weight", "lambda_neu", "lambda_supcon",
        "lambda_angle", "lambda_sas", "lambda_hard", "lambda_gate_entropy",
        "sas_margin", "hard_negative_rho", "hard_negative_temperature", "prototype_pooling",
        "use_nearest_neighbour", "use_neutral_decoupling", "use_speaker_state",
        "use_speaker_memory", "speaker_memory_k", "speaker_memory_pooling",
        "use_similar_anchor_separation", "use_hard_anchor_negative", "class_balanced_ce",
        "hard_negative_schedule", "disable_anchor_updates", "use_classifier_prototype_fusion",
        "fusion_type", "fusion_alpha", "use_neutral_aware_supcon", "prototype_update_policy",
        "accumulation_step", "confusion_matrix", "similar_pair_confusion",
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
    parser.add_argument("--experiment-set", default="all", choices=["all", "ablation", "targeted", "fusion", "smf"],
                        help="Which predefined next-round experiments to run.")
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
    configs = filter_configs(candidate_configs(), args.experiment_set)
    if args.max_runs > 0:
        configs = configs[:args.max_runs]

    print(f"[queue] dataset={args.dataset} gpu={args.gpu_id} epochs={args.epochs} set={args.experiment_set} runs={len(configs)}", flush=True)
    print(f"[queue] output={queue_root.resolve()}", flush=True)

    rows = []
    for cfg in configs:
        name = run_name(cfg)
        run_dir = queue_root / name
        run_dir.mkdir(parents=True, exist_ok=True)
        save_path = run_dir / "saved_models"
        stdout_log = run_dir / "train.stdout.log"
        logging_log = save_path / args.dataset / "logging.log"
        confusion_matrix = save_path / args.dataset / "confusion_matrix.csv"
        similar_pair_confusion = save_path / args.dataset / "similar_pair_confusion.csv"
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
            "confusion_matrix": str(confusion_matrix) if confusion_matrix.exists() else "",
            "similar_pair_confusion": str(similar_pair_confusion) if similar_pair_confusion.exists() else "",
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
