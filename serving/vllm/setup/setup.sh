#!/usr/bin/env bash
set -euo pipefail

SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_DIR="$(cd "${SETUP_DIR}/.." && pwd)"
ENV_NAME="${ENV_NAME:-mas-vllm}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.11.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.26.0}"
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-5.5.4}"
VLLM_VERSION_OVERRIDE="${VLLM_VERSION_OVERRIDE:-0.19.2.dev0+gf150107.local}"
VLLM_PRECOMPILED_WHEEL_COMMIT="${VLLM_PRECOMPILED_WHEEL_COMMIT:-f150107efd15a873617a65b09189c945b27d61f5}"
VLLM_PRECOMPILED_WHEEL_VARIANT="${VLLM_PRECOMPILED_WHEEL_VARIANT:-cu129}"

[[ -f "${VLLM_DIR}/pyproject.toml" ]] || { echo "Cannot find vLLM source at ${VLLM_DIR}" >&2; exit 1; }
command -v conda >/dev/null 2>&1 || { echo "conda not found" >&2; exit 1; }
eval "$(conda shell.bash hook)"
conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}" || conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
conda activate "${ENV_NAME}"
python - <<'PY'
import sys
raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)
PY

cd "${VLLM_DIR}"
pip install -U pip certifi setuptools setuptools-scm wheel numpy
export VLLM_USE_PRECOMPILED=1
export VLLM_PRECOMPILED_WHEEL_VARIANT
export VLLM_VERSION_OVERRIDE
export VLLM_PRECOMPILED_WHEEL_COMMIT
pip install -e .
pip install "torch==${TORCH_VERSION}" "torchaudio==${TORCHAUDIO_VERSION}" "torchvision==${TORCHVISION_VERSION}"
pip install "transformers==${TRANSFORMERS_VERSION}" hf_transfer
python -c "import vllm, torch; print(vllm.__version__); print(torch.__version__, torch.cuda.is_available())"

echo "vLLM ready: conda activate ${ENV_NAME}"
