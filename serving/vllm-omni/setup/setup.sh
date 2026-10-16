#!/usr/bin/env bash
set -euo pipefail

SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OMNI_DIR="$(cd "${SETUP_DIR}/.." && pwd)"
ENV_NAME="${ENV_NAME:-mas-vllm-omni}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
VLLM_VERSION="${VLLM_VERSION:-0.18.0}"
VLLM_OMNI_VERSION_OVERRIDE="${VLLM_OMNI_VERSION_OVERRIDE:-0.16.0+local}"

[[ -f "${OMNI_DIR}/pyproject.toml" ]] || { echo "Cannot find vLLM-Omni source at ${OMNI_DIR}" >&2; exit 1; }
command -v conda >/dev/null 2>&1 || { echo "conda not found" >&2; exit 1; }
eval "$(conda shell.bash hook)"
conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}" || conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
conda activate "${ENV_NAME}"
python - <<'PY'
import sys
raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)
PY

cd "${OMNI_DIR}"
pip install -U pip certifi setuptools setuptools-scm wheel numpy
pip install "vllm==${VLLM_VERSION}"
export VLLM_OMNI_VERSION_OVERRIDE
pip install -e .
pip install hf_transfer
python -c "import vllm_omni, vllm, torch; print(getattr(vllm_omni, '__version__', 'unknown')); print(vllm.__version__); print(torch.__version__, torch.cuda.is_available())"

echo "vLLM-Omni ready: conda activate ${ENV_NAME}"
