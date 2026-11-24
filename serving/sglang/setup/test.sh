#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID=2
HOST=127.0.0.1
PORT=30000
MODEL="Qwen/Qwen2.5-14B-Instruct"
TASK="${TASK:-auto}"
TIMEOUT=600
PROMPT="Reply with the single word READY."
MAX_MODEL_LEN=2048
MEM_FRACTION=0.85
IMAGE_SIZE=1024x1024
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU_ID="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --prompt) PROMPT="$2"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --mem-fraction-static) MEM_FRACTION="$2"; shift 2 ;;
    --size) IMAGE_SIZE="$2"; shift 2 ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "${CONDA_PREFIX:-}" ]] || { echo "No active conda environment" >&2; exit 1; }
LOG_FILE="$(mktemp -t sglang-test.XXXXXX.log)"
SERVER_PID=""
cleanup() {
  [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null && kill "${SERVER_PID}" 2>/dev/null || true
  [[ -n "${SERVER_PID}" ]] && wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

cd "${ROOT}"
if [[ "${TASK}" == "auto" ]]; then
  case "${MODEL,,}" in
    *image*|*diffusers*|*flux*|*wan*)
      TASK="image"
      ;;
    *)
      TASK="text"
      ;;
  esac
fi

if [[ "${TASK}" == "image" ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID}}" \
  HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}" \
  sglang serve \
    --model-path "${MODEL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    "${EXTRA_ARGS[@]}" >"${LOG_FILE}" 2>&1 &
else
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID}}" \
  HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}" \
  python -m sglang.launch_server \
    --model-path "${MODEL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --context-length "${MAX_MODEL_LEN}" \
    --mem-fraction-static "${MEM_FRACTION}" \
    --trust-remote-code \
    --disable-piecewise-cuda-graph \
    "${EXTRA_ARGS[@]}" >"${LOG_FILE}" 2>&1 &
fi
SERVER_PID=$!

deadline=$((SECONDS + TIMEOUT))
READY_PATH="/v1/models"
[[ "${TASK}" == "image" ]] && READY_PATH="/models"
until curl -fsS "http://${HOST}:${PORT}${READY_PATH}" >/dev/null 2>&1; do
  kill -0 "${SERVER_PID}" 2>/dev/null || { tail -n 120 "${LOG_FILE}" >&2; exit 1; }
  (( SECONDS < deadline )) || { tail -n 120 "${LOG_FILE}" >&2; exit 1; }
  sleep 2
done

python - "${HOST}" "${PORT}" "${MODEL}" "${PROMPT}" "${TASK}" "${IMAGE_SIZE}" <<'PY'
import json, sys, urllib.request
host, port, model, prompt, task, image_size = sys.argv[1:7]
if task == "image":
    payload = {
        "prompt": prompt,
        "size": image_size,
        "n": 1,
        "response_format": "b64_json",
    }
    req = urllib.request.Request(
        f"http://{host}:{port}/v1/images/generations",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
else:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 32,
        "temperature": 0,
    }
    req = urllib.request.Request(
        f"http://{host}:{port}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
with urllib.request.urlopen(req, timeout=240) as resp:
    data = json.loads(resp.read().decode())
print(json.dumps(data, ensure_ascii=False, indent=2))
if task == "image":
    assert data["data"][0]["b64_json"].strip()
else:
    assert data["choices"][0]["message"]["content"].strip()
PY
