#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-IEMOCAP}"
GPU_ID="${GPU_ID:-0}"
EPOCHS="${EPOCHS:-30}"
MAX_RUNS="${MAX_RUNS:-0}"
OUT_ROOT="${OUT_ROOT:-run_logs/sas_nsg_queue}"
BERT_PATH="${BERT_PATH:-pretrained/sup-simcse-roberta-large}"
ANCHOR_PATH="${ANCHOR_PATH:-emo_anchors/sup-simcse-roberta-large}"

mkdir -p "${OUT_ROOT}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LAUNCH_LOG="${OUT_ROOT}/launcher_${DATASET}_${STAMP}.log"
PID_FILE="${OUT_ROOT}/last_${DATASET}.pid"

nohup python scripts/sas_nsg_train_queue.py \
  --dataset "${DATASET}" \
  --gpu-id "${GPU_ID}" \
  --epochs "${EPOCHS}" \
  --max-runs "${MAX_RUNS}" \
  --bert-path "${BERT_PATH}" \
  --anchor-path "${ANCHOR_PATH}" \
  --out-root "${OUT_ROOT}" \
  > "${LAUNCH_LOG}" 2>&1 &

PID="$!"
echo "${PID}" > "${PID_FILE}"

echo "SAS-NSG-EACL background queue started."
echo "PID: ${PID}"
echo "Launcher log: ${LAUNCH_LOG}"
echo "PID file: ${PID_FILE}"
echo
echo "Watch launcher:"
echo "  tail -f ${LAUNCH_LOG}"
echo
echo "Find leaderboard:"
echo "  find ${OUT_ROOT} -name leaderboard.csv -print"
