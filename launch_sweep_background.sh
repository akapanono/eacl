#!/usr/bin/env bash
set -euo pipefail

PYTHON_EXE="${PYTHON_EXE:-python}"
SCRIPT_PATH="${SCRIPT_PATH:-sweep_random.py}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

mkdir -p sweep_logs
STAMP="$(date +%Y%m%d_%H%M%S)"
STDOUT_LOG="sweep_logs/launcher_${STAMP}.log"
STDERR_LOG="sweep_logs/launcher_${STAMP}.err.log"
META_FILE="sweep_logs/last_background_sweep.txt"

nohup "$PYTHON_EXE" "$SCRIPT_PATH" > "$STDOUT_LOG" 2> "$STDERR_LOG" &
PID="$!"

{
  echo "started_at=$(date '+%Y-%m-%d %H:%M:%S')"
  echo "pid=$PID"
  echo "python=$PYTHON_EXE"
  echo "script=$SCRIPT_PATH"
  echo "dataset=${DATASET_NAME:-${DATASET:-MELD}}"
  echo "gpu_ids=${GPU_IDS:-0}"
  echo "n_trials=${N_TRIALS:-100}"
  echo "epochs=${EPOCHS:-30}"
  echo "num_subanchors=${NUM_SUBANCHORS:-5}"
  echo "stdout_log=$STDOUT_LOG"
  echo "stderr_log=$STDERR_LOG"
  echo "summary_tsv=sweep_logs/summary.tsv"
  echo "summary_csv=sweep_logs/summary.csv"
} > "$META_FILE"

echo "Background sweep started."
echo "PID: $PID"
echo "Stdout log: $STDOUT_LOG"
echo "Stderr log: $STDERR_LOG"
echo "Meta file: $META_FILE"
