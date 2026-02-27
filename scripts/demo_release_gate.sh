#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
cd "${ROOT_DIR}"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RETRIEVAL_INPUT="${RETRIEVAL_INPUT:-data/eval_set.json}"
ANSWER_INPUT="${ANSWER_INPUT:-data/eval_answer_demo_v1.json}"
OUTPUT_DIR="${OUTPUT_DIR:-data/eval_reports}"
BASELINE_REPORT="${BASELINE_REPORT:-data/eval_baseline.json}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
ANSWER_TIMEOUT="${ANSWER_TIMEOUT:-420}"
ANSWER_RETRIES="${ANSWER_RETRIES:-1}"
ANSWER_RESCUE_TIMEOUT="${ANSWER_RESCUE_TIMEOUT:-900}"
ANSWER_GATE_THRESHOLD="${ANSWER_GATE_THRESHOLD:-0.85}"

RETRIEVAL_STATUS=0
ANSWER_STATUS=0

mkdir -p "${OUTPUT_DIR}"

echo "[gate] run_id=${RUN_ID}"
echo "[gate] retrieval eval"
"${PYTHON_BIN}" scripts/run_eval.py \
  --input "${RETRIEVAL_INPUT}" \
  --output-dir "${OUTPUT_DIR}" \
  --run-id "${RUN_ID}" \
  --include-nofilter || RETRIEVAL_STATUS=$?

if [[ ${RETRIEVAL_STATUS} -eq 0 ]]; then
  "${PYTHON_BIN}" scripts/check_eval_regression.py \
    --baseline "${BASELINE_REPORT}" \
    --report "${OUTPUT_DIR}/${RUN_ID}.json" \
    --stages hybrid,rerank \
    --k 5,10 || RETRIEVAL_STATUS=$?
fi

if [[ ${RETRIEVAL_STATUS} -ne 0 ]]; then
  echo "[gate] retrieval gate failed (status=${RETRIEVAL_STATUS})"
else
  echo "[gate] retrieval gate passed"
fi

echo "[gate] answer eval"
"${PYTHON_BIN}" scripts/run_answer_eval.py \
  --input "${ANSWER_INPUT}" \
  --output-dir "${OUTPUT_DIR}" \
  --run-id "${RUN_ID}" \
  --base-url "${BASE_URL}" \
  --timeout "${ANSWER_TIMEOUT}" \
  --retries "${ANSWER_RETRIES}" \
  --rescue-timeout "${ANSWER_RESCUE_TIMEOUT}" || ANSWER_STATUS=$?

if [[ ${ANSWER_STATUS} -eq 0 ]]; then
  "${PYTHON_BIN}" scripts/check_answer_eval_gate.py \
    --report "${OUTPUT_DIR}/answer_${RUN_ID}.json" \
    --threshold "${ANSWER_GATE_THRESHOLD}" || ANSWER_STATUS=$?
fi

if [[ ${ANSWER_STATUS} -ne 0 ]]; then
  echo "[gate] answer gate failed (status=${ANSWER_STATUS})"
else
  echo "[gate] answer gate passed"
fi

echo "[gate] summary"
echo "- run_id: ${RUN_ID}"
echo "- retrieval_status: ${RETRIEVAL_STATUS}"
echo "- answer_status: ${ANSWER_STATUS}"

if [[ ${RETRIEVAL_STATUS} -ne 0 || ${ANSWER_STATUS} -ne 0 ]]; then
  exit 1
fi

echo "[gate] all gates passed"
exit 0
