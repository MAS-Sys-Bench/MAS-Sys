#!/usr/bin/env bash
set -euo pipefail

SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_DIR="$(cd "${SETUP_DIR}/.." && pwd)"
PYTHON_DIR="${SGLANG_DIR}/python"
ENV_NAME="${ENV_NAME:-mas-sglang}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-5.5.4}"
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG="${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG:-0.0.0.dev0+local}"

[[ -f "${PYTHON_DIR}/pyproject.toml" ]] || { echo "Cannot find SGLang Python source at ${PYTHON_DIR}" >&2; exit 1; }
command -v conda >/dev/null 2>&1 || { echo "conda not found" >&2; exit 1; }
eval "$(conda shell.bash hook)"
conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}" || conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
conda activate "${ENV_NAME}"
python - <<'PY'
import sys
raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)
PY

cd "${PYTHON_DIR}"
pip install -U pip certifi setuptools setuptools-scm wheel numpy
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG
pip install -e ".[diffusion]"
pip install "transformers==${TRANSFORMERS_VERSION}" hf_transfer ninja wheel
cd "${SGLANG_DIR}"
python -c "import sglang, transformers, torch; print(getattr(sglang, '__version__', 'unknown')); print(transformers.__version__); print(torch.__version__, torch.cuda.is_available())"

echo "SGLang ready: conda activate ${ENV_NAME}"
