#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID=2
HOST=127.0.0.1
PORT=8000
MODEL="Qwen/Qwen2.5-14B-Instruct"
TIMEOUT=600
PROMPT="Reply with exactly one short sentence saying the server is healthy."
MAX_MODEL_LEN=2048
GPU_MEMORY_UTILIZATION=0.9
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU_ID="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --prompt) PROMPT="$2"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --gpu-memory-utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "${CONDA_PREFIX:-}" ]] || { echo "No active conda environment" >&2; exit 1; }
LOG_FILE="$(mktemp -t vllm-test.XXXXXX.log)"
SERVER_PID=""
cleanup() {
  [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null && kill "${SERVER_PID}" 2>/dev/null || true
  [[ -n "${SERVER_PID}" ]] && wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

cd "${ROOT}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID}}" \
HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}" \
VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}" \
python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --trust-remote-code \
  "${EXTRA_ARGS[@]}" >"${LOG_FILE}" 2>&1 &
SERVER_PID=$!

deadline=$((SECONDS + TIMEOUT))
until curl -fsS "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; do
  kill -0 "${SERVER_PID}" 2>/dev/null || { tail -n 120 "${LOG_FILE}" >&2; exit 1; }
  (( SECONDS < deadline )) || { tail -n 120 "${LOG_FILE}" >&2; exit 1; }
  sleep 2
done

python - "${HOST}" "${PORT}" "${MODEL}" "${PROMPT}" <<'PY'
import json, sys, urllib.request
host, port, model, prompt = sys.argv[1:5]
payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 64, "temperature": 0}
req = urllib.request.Request(
    f"http://{host}:{port}/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=120) as resp:
    data = json.loads(resp.read().decode())
print(json.dumps(data, ensure_ascii=False, indent=2))
assert data["choices"][0]["message"]["content"].strip()
PY
