#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs/demo"

CHAT_LLM_PID_FILE="${LOG_DIR}/chat_llm.pid"
EMBED_LLM_PID_FILE="${LOG_DIR}/embed_llm.pid"
API_PID_FILE="${LOG_DIR}/api.pid"
CHAINLIT_PID_FILE="${LOG_DIR}/chainlit.pid"

CHAT_LLM_LOG_FILE="${LOG_DIR}/chat_llm.log"
EMBED_LLM_LOG_FILE="${LOG_DIR}/embed_llm.log"
API_LOG_FILE="${LOG_DIR}/api.log"
CHAINLIT_LOG_FILE="${LOG_DIR}/chainlit.log"

# llama-server (chat) — reuses an existing instance when one is already
# listening, so other projects sharing the same model are not disturbed.
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-${HOME}/llama.cpp/build/bin/llama-server}"
CHAT_LLM_HOST="${CHAT_LLM_HOST:-127.0.0.1}"
CHAT_LLM_PORT="${CHAT_LLM_PORT:-8000}"
CHAT_LLM_MODEL="${CHAT_LLM_MODEL:-${HOME}/models/qwen3.6-27b/Qwen3.6-27B-UD-Q4_K_XL.gguf}"
CHAT_LLM_ALIAS="${CHAT_LLM_ALIAS:-qwen3.6-27b}"
CHAT_LLM_CTX="${CHAT_LLM_CTX:-131072}"
CHAT_LLM_NGL="${CHAT_LLM_NGL:-999}"

# llama-server (embeddings) — always managed by this script.
EMBED_LLM_HOST="${EMBED_LLM_HOST:-127.0.0.1}"
EMBED_LLM_PORT="${EMBED_LLM_PORT:-8001}"
EMBED_LLM_MODEL="${EMBED_LLM_MODEL:-${HOME}/models/bge-m3/bge-m3-f16.gguf}"
EMBED_LLM_ALIAS="${EMBED_LLM_ALIAS:-bge-m3}"
EMBED_LLM_CTX="${EMBED_LLM_CTX:-8192}"
EMBED_LLM_NGL="${EMBED_LLM_NGL:-99}"
# Pin the embed server to a specific GPU index (chat server typically owns GPU 0).
EMBED_LLM_CUDA_DEVICES="${EMBED_LLM_CUDA_DEVICES:-1}"

API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8088}"
CHAINLIT_HOST="${CHAINLIT_HOST:-0.0.0.0}"
CHAINLIT_PORT="${CHAINLIT_PORT:-8501}"
BASE_URL="${BASE_URL:-http://127.0.0.1:${API_PORT}}"
CHAINLIT_URL="${CHAINLIT_URL:-http://127.0.0.1:${CHAINLIT_PORT}}"

CHAT_LLM_URL="http://${CHAT_LLM_HOST}:${CHAT_LLM_PORT}"
EMBED_LLM_URL="http://${EMBED_LLM_HOST}:${EMBED_LLM_PORT}"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
UVICORN_BIN="${UVICORN_BIN:-${ROOT_DIR}/.venv/bin/uvicorn}"
CHAINLIT_BIN="${CHAINLIT_BIN:-${ROOT_DIR}/.venv/bin/chainlit}"

mkdir -p "${LOG_DIR}"

is_running_pid_file() {
  local pid_file="$1"
  [[ -f "${pid_file}" ]] || return 1
  local pid
  pid="$(cat "${pid_file}" 2>/dev/null || true)"
  [[ -n "${pid}" ]] || return 1
  kill -0 "${pid}" 2>/dev/null
}

probe_url_ok() {
  local url="$1"
  curl -fsS -m 5 -o /dev/null "${url}" >/dev/null 2>&1
}

wait_for_health() {
  local url="$1"
  local name="$2"
  local attempts="${3:-60}"
  for _ in $(seq 1 "${attempts}"); do
    if probe_url_ok "${url}"; then
      return 0
    fi
    sleep 1
  done
  echo "${name} did not become healthy at ${url} after ${attempts}s"
  return 1
}

start_chat_llm() {
  if is_running_pid_file "${CHAT_LLM_PID_FILE}"; then
    echo "Chat llama-server already managed (pid=$(cat "${CHAT_LLM_PID_FILE}"))"
    return 0
  fi

  if probe_url_ok "${CHAT_LLM_URL}/health"; then
    echo "Chat llama-server already running externally on ${CHAT_LLM_URL} (not managed by demo_ctl)"
    return 0
  fi

  if [[ ! -x "${LLAMA_SERVER_BIN}" ]]; then
    echo "llama-server binary not found at ${LLAMA_SERVER_BIN}"
    return 1
  fi
  if [[ ! -f "${CHAT_LLM_MODEL}" ]]; then
    echo "Chat model GGUF not found at ${CHAT_LLM_MODEL}"
    return 1
  fi

  echo "Starting chat llama-server on ${CHAT_LLM_HOST}:${CHAT_LLM_PORT}"
  nohup "${LLAMA_SERVER_BIN}" \
    --model "${CHAT_LLM_MODEL}" \
    --alias "${CHAT_LLM_ALIAS}" \
    --host "${CHAT_LLM_HOST}" --port "${CHAT_LLM_PORT}" \
    --n-gpu-layers "${CHAT_LLM_NGL}" \
    --ctx-size "${CHAT_LLM_CTX}" \
    --parallel 1 \
    --cache-type-k q4_0 --cache-type-v q4_0 \
    --flash-attn on \
    --cache-reuse 256 \
    --reasoning-format auto \
    -ub 1024 -b 2048 \
    --metrics --jinja \
    >"${CHAT_LLM_LOG_FILE}" 2>&1 &
  local pid=$!
  echo "${pid}" >"${CHAT_LLM_PID_FILE}"
  if ! wait_for_health "${CHAT_LLM_URL}/health" "Chat llama-server" 90; then
    tail -n 40 "${CHAT_LLM_LOG_FILE}" || true
    return 1
  fi
  echo "Chat llama-server started (pid=${pid})"
}

start_embed_llm() {
  if is_running_pid_file "${EMBED_LLM_PID_FILE}"; then
    echo "Embed llama-server already managed (pid=$(cat "${EMBED_LLM_PID_FILE}"))"
    return 0
  fi

  if probe_url_ok "${EMBED_LLM_URL}/health"; then
    echo "Embed llama-server already running externally on ${EMBED_LLM_URL} (not managed by demo_ctl)"
    return 0
  fi

  if [[ ! -x "${LLAMA_SERVER_BIN}" ]]; then
    echo "llama-server binary not found at ${LLAMA_SERVER_BIN}"
    return 1
  fi
  if [[ ! -f "${EMBED_LLM_MODEL}" ]]; then
    echo "Embed model GGUF not found at ${EMBED_LLM_MODEL}"
    return 1
  fi

  echo "Starting embed llama-server on ${EMBED_LLM_HOST}:${EMBED_LLM_PORT} (CUDA_VISIBLE_DEVICES=${EMBED_LLM_CUDA_DEVICES})"
  nohup env CUDA_VISIBLE_DEVICES="${EMBED_LLM_CUDA_DEVICES}" "${LLAMA_SERVER_BIN}" \
    --model "${EMBED_LLM_MODEL}" \
    --alias "${EMBED_LLM_ALIAS}" \
    --host "${EMBED_LLM_HOST}" --port "${EMBED_LLM_PORT}" \
    --embeddings \
    --pooling cls \
    --ctx-size "${EMBED_LLM_CTX}" -b "${EMBED_LLM_CTX}" -ub "${EMBED_LLM_CTX}" \
    --n-gpu-layers "${EMBED_LLM_NGL}" \
    --embd-normalize 2 \
    >"${EMBED_LLM_LOG_FILE}" 2>&1 &
  local pid=$!
  echo "${pid}" >"${EMBED_LLM_PID_FILE}"
  if ! wait_for_health "${EMBED_LLM_URL}/health" "Embed llama-server" 60; then
    tail -n 40 "${EMBED_LLM_LOG_FILE}" || true
    return 1
  fi
  echo "Embed llama-server started (pid=${pid})"
}

start_api() {
  if is_running_pid_file "${API_PID_FILE}"; then
    echo "API already running (pid=$(cat "${API_PID_FILE}"))"
    return 0
  fi

  echo "Starting API on ${API_HOST}:${API_PORT}"
  nohup "${UVICORN_BIN}" api.main:app --host "${API_HOST}" --port "${API_PORT}" >"${API_LOG_FILE}" 2>&1 &
  local pid=$!
  echo "${pid}" >"${API_PID_FILE}"
  sleep 1
  if ! kill -0 "${pid}" 2>/dev/null; then
    echo "API failed to stay running. Check ${API_LOG_FILE}"
    rm -f "${API_PID_FILE}"
    tail -n 40 "${API_LOG_FILE}" || true
    return 1
  fi
  echo "API started (pid=${pid})"
}

start_chainlit() {
  if is_running_pid_file "${CHAINLIT_PID_FILE}"; then
    echo "Chainlit already running (pid=$(cat "${CHAINLIT_PID_FILE}"))"
    return 0
  fi

  echo "Starting Chainlit on ${CHAINLIT_HOST}:${CHAINLIT_PORT}"
  # Chainlit maps DEBUG env var to its --debug flag and expects a boolean.
  # Force DEBUG to a valid value and disable auto-browser open in remote sessions.
  nohup env DEBUG=0 BROWSER= CHAINLIT_BACKEND_URL="${BASE_URL}" \
    "${CHAINLIT_BIN}" run ui/chainlit_app.py --headless --host "${CHAINLIT_HOST}" --port "${CHAINLIT_PORT}" \
    >"${CHAINLIT_LOG_FILE}" 2>&1 &
  local pid=$!
  echo "${pid}" >"${CHAINLIT_PID_FILE}"
  sleep 1
  if ! kill -0 "${pid}" 2>/dev/null; then
    echo "Chainlit failed to stay running. Check ${CHAINLIT_LOG_FILE}"
    rm -f "${CHAINLIT_PID_FILE}"
    tail -n 40 "${CHAINLIT_LOG_FILE}" || true
    return 1
  fi
  echo "Chainlit started (pid=${pid})"
}

stop_service() {
  local name="$1"
  local pid_file="$2"

  if ! [[ -f "${pid_file}" ]]; then
    echo "${name} pid file not found"
    return 0
  fi

  local pid
  pid="$(cat "${pid_file}" 2>/dev/null || true)"
  if [[ -z "${pid}" ]]; then
    rm -f "${pid_file}"
    echo "${name} pid file was empty; cleaned"
    return 0
  fi

  if kill -0 "${pid}" 2>/dev/null; then
    echo "Stopping ${name} (pid=${pid})"
    kill "${pid}" 2>/dev/null || true
    for _ in $(seq 1 30); do
      if ! kill -0 "${pid}" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if kill -0 "${pid}" 2>/dev/null; then
      echo "${name} did not stop gracefully, killing"
      kill -9 "${pid}" 2>/dev/null || true
    fi
  else
    echo "${name} pid ${pid} not running; cleaning stale pid file"
  fi

  rm -f "${pid_file}"
}

show_status() {
  echo "=== Processes ==="
  if is_running_pid_file "${CHAT_LLM_PID_FILE}"; then
    echo "Chat llama-server: running (pid=$(cat "${CHAT_LLM_PID_FILE}"))"
  elif probe_url_ok "${CHAT_LLM_URL}/health"; then
    echo "Chat llama-server: externally running at ${CHAT_LLM_URL} (not managed)"
  else
    echo "Chat llama-server: not running"
  fi

  if is_running_pid_file "${EMBED_LLM_PID_FILE}"; then
    echo "Embed llama-server: running (pid=$(cat "${EMBED_LLM_PID_FILE}"))"
  elif probe_url_ok "${EMBED_LLM_URL}/health"; then
    echo "Embed llama-server: externally running at ${EMBED_LLM_URL} (not managed)"
  else
    echo "Embed llama-server: not running"
  fi

  if is_running_pid_file "${API_PID_FILE}"; then
    echo "API: running (pid=$(cat "${API_PID_FILE}"))"
  elif probe_url_ok "${BASE_URL}/health"; then
    echo "API: externally running at ${BASE_URL} (not managed)"
  else
    echo "API: not running"
  fi

  if is_running_pid_file "${CHAINLIT_PID_FILE}"; then
    echo "Chainlit: running (pid=$(cat "${CHAINLIT_PID_FILE}"))"
  elif probe_url_ok "${CHAINLIT_URL}/"; then
    echo "Chainlit: externally running at ${CHAINLIT_URL} (not managed)"
  else
    echo "Chainlit: not running"
  fi

  if command -v curl >/dev/null 2>&1; then
    echo
    echo "=== Probes ==="
    echo "Chat ${CHAT_LLM_URL}/health: $(curl -sS -m 5 "${CHAT_LLM_URL}/health" || echo '(failed)')"
    echo "Embed ${EMBED_LLM_URL}/health: $(curl -sS -m 5 "${EMBED_LLM_URL}/health" || echo '(failed)')"
    echo "API ${BASE_URL}/health: $(curl -sS -m 5 "${BASE_URL}/health" || echo '(failed)')"
    echo "API ${BASE_URL}/ready: $(curl -sS -m 5 "${BASE_URL}/ready" || echo '(failed)')"
  fi
}

show_logs() {
  local follow="${1:-}"
  touch "${CHAT_LLM_LOG_FILE}" "${EMBED_LLM_LOG_FILE}" "${API_LOG_FILE}" "${CHAINLIT_LOG_FILE}"
  if [[ "${follow}" == "--follow" ]]; then
    tail -n 80 -f "${CHAT_LLM_LOG_FILE}" "${EMBED_LLM_LOG_FILE}" "${API_LOG_FILE}" "${CHAINLIT_LOG_FILE}"
  else
    tail -n 80 "${CHAT_LLM_LOG_FILE}" "${EMBED_LLM_LOG_FILE}" "${API_LOG_FILE}" "${CHAINLIT_LOG_FILE}"
  fi
}

run_smoke() {
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/demo_smoke.py" --base-url "${BASE_URL}"
}

usage() {
  cat <<EOF
Usage: scripts/demo_ctl.sh <command>
Commands:
  start    Start chat + embed llama-servers (chat only if not already up),
           API on :${API_PORT}, and Chainlit on :${CHAINLIT_PORT}
  stop     Stop everything managed via pid files (external chat server left running)
  status   Show process state and health probes for all four services
  logs     Show last 80 lines of llama-server, API, and Chainlit logs
  logs --follow  Tail logs continuously
  smoke    Run scripts/demo_smoke.py against BASE_URL (${BASE_URL})
EOF
}

main() {
  cd "${ROOT_DIR}"

  local cmd="${1:-}"
  case "${cmd}" in
    start)
      start_chat_llm
      start_embed_llm
      start_api
      start_chainlit
      ;;
    stop)
      stop_service "Chainlit" "${CHAINLIT_PID_FILE}"
      stop_service "API" "${API_PID_FILE}"
      stop_service "Embed llama-server" "${EMBED_LLM_PID_FILE}"
      stop_service "Chat llama-server" "${CHAT_LLM_PID_FILE}"
      ;;
    status)
      show_status
      ;;
    logs)
      show_logs "${2:-}"
      ;;
    smoke)
      run_smoke
      ;;
    *)
      usage
      exit 1
      ;;
  esac
}

main "$@"
