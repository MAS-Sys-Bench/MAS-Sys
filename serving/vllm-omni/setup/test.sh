#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID=2
HOST=127.0.0.1
PORT=8091
MODEL="Qwen/Qwen-Image"
TASK="${TASK:-auto}"
TIMEOUT=600
PROMPT="A small red cube on a white table."
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
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "${CONDA_PREFIX:-}" ]] || { echo "No active conda environment" >&2; exit 1; }
LOG_FILE="$(mktemp -t vllm-omni-test.XXXXXX.log)"
SERVER_PID=""
cleanup() {
  [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null && kill "${SERVER_PID}" 2>/dev/null || true
  [[ -n "${SERVER_PID}" ]] && wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

cd "${ROOT}"
if [[ "${TASK}" == "auto" ]]; then
  case "${MODEL,,}" in
    *image*|*flux*|*diffusion*|*video*)
      TASK="image"
      ;;
    *)
      TASK="text"
      ;;
  esac
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID}}" \
HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}" \
VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}" \
python -m vllm_omni.entrypoints.cli.main serve "${MODEL}" \
  --omni \
  --host "${HOST}" \
  --port "${PORT}" \
  --num-gpus 1 \
  --trust-remote-code \
  "${EXTRA_ARGS[@]}" >"${LOG_FILE}" 2>&1 &
SERVER_PID=$!

deadline=$((SECONDS + TIMEOUT))
until curl -fsS "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; do
  kill -0 "${SERVER_PID}" 2>/dev/null || { tail -n 120 "${LOG_FILE}" >&2; exit 1; }
  (( SECONDS < deadline )) || { tail -n 120 "${LOG_FILE}" >&2; exit 1; }
  sleep 2
done

python - "${HOST}" "${PORT}" "${MODEL}" "${PROMPT}" "${TASK}" <<'PY'
import json, sys, urllib.request
host, port, model, prompt, task = sys.argv[1:6]
if task == "image":
    payload = {
        "model": model,
        "prompt": prompt,
        "size": "1024x1024",
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
        "max_tokens": 16,
        "temperature": 0,
    }
    req = urllib.request.Request(
        f"http://{host}:{port}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
with urllib.request.urlopen(req, timeout=120) as resp:
    data = json.loads(resp.read().decode())
print(json.dumps(data, ensure_ascii=False, indent=2))
if task == "image":
    assert data["data"][0]["b64_json"].strip()
else:
    assert data["choices"][0]["message"]["content"].strip()
PY
