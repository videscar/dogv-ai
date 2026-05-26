#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs/demo"
API_PID_FILE="${LOG_DIR}/api.pid"
CHAINLIT_PID_FILE="${LOG_DIR}/chainlit.pid"
API_LOG_FILE="${LOG_DIR}/api.log"
CHAINLIT_LOG_FILE="${LOG_DIR}/chainlit.log"

API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8000}"
CHAINLIT_HOST="${CHAINLIT_HOST:-0.0.0.0}"
CHAINLIT_PORT="${CHAINLIT_PORT:-8501}"
BASE_URL="${BASE_URL:-http://127.0.0.1:${API_PORT}}"
CHAINLIT_URL="${CHAINLIT_URL:-http://127.0.0.1:${CHAINLIT_PORT}}"

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
  nohup env DEBUG=0 BROWSER= "${CHAINLIT_BIN}" run ui/chainlit_app.py --headless --host "${CHAINLIT_HOST}" --port "${CHAINLIT_PORT}" >"${CHAINLIT_LOG_FILE}" 2>&1 &
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
  local api_probe_ok=1
  local chainlit_probe_ok=1
  if probe_url_ok "${BASE_URL}/health"; then
    api_probe_ok=0
  fi
  if probe_url_ok "${CHAINLIT_URL}/"; then
    chainlit_probe_ok=0
  fi

  if is_running_pid_file "${API_PID_FILE}"; then
    echo "API: running (pid=$(cat "${API_PID_FILE}"))"
  elif [[ ${api_probe_ok} -eq 0 ]]; then
    echo "API: externally running/reachable (no managed pid file)"
  else
    echo "API: not running"
  fi

  if is_running_pid_file "${CHAINLIT_PID_FILE}"; then
    echo "Chainlit: running (pid=$(cat "${CHAINLIT_PID_FILE}"))"
  elif [[ ${chainlit_probe_ok} -eq 0 ]]; then
    echo "Chainlit: externally running/reachable (no managed pid file)"
  else
    echo "Chainlit: not running"
  fi

  if command -v curl >/dev/null 2>&1; then
    echo "Health probe (${BASE_URL}/health):"
    curl -sS -m 5 "${BASE_URL}/health" || echo "(health probe failed)"
    echo
    echo "Ready probe (${BASE_URL}/ready):"
    curl -sS -m 5 "${BASE_URL}/ready" || echo "(ready probe failed)"
    echo
  fi
}

show_logs() {
  local follow="${1:-}"
  touch "${API_LOG_FILE}" "${CHAINLIT_LOG_FILE}"
  if [[ "${follow}" == "--follow" ]]; then
    tail -n 80 -f "${API_LOG_FILE}" "${CHAINLIT_LOG_FILE}"
  else
    tail -n 80 "${API_LOG_FILE}" "${CHAINLIT_LOG_FILE}"
  fi
}

run_smoke() {
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/demo_smoke.py" --base-url "${BASE_URL}"
}

usage() {
  cat <<EOF
Usage: scripts/demo_ctl.sh <command>
Commands:
  start    Start API and Chainlit using nohup + pid files
  stop     Stop API and Chainlit and clean pid files
  status   Show process state and /health + /ready probes
  logs     Show last 80 lines of API and Chainlit logs
  logs --follow  Tail logs continuously
  smoke    Run scripts/demo_smoke.py against BASE_URL
EOF
}

main() {
  cd "${ROOT_DIR}"

  local cmd="${1:-}"
  case "${cmd}" in
    start)
      start_api
      start_chainlit
      ;;
    stop)
      stop_service "Chainlit" "${CHAINLIT_PID_FILE}"
      stop_service "API" "${API_PID_FILE}"
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
