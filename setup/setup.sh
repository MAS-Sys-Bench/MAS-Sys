#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-mas-sys}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

command -v conda >/dev/null 2>&1 || {
  echo "conda not found" >&2
  exit 1
}

eval "$(conda shell.bash hook)"
conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}" || conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
conda activate "${ENV_NAME}"

python - <<'PY'
import sys
raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)
PY

cd "${ROOT_DIR}"
pip install -U pip
pip install -r requirements.txt
python -m playwright install chromium
python -c "import pydantic, openai, yaml; print('MAS-Sys import ok')"

echo "MAS-Sys ready: conda activate ${ENV_NAME}"
