#!/usr/bin/env bash
set -euo pipefail

# Create a reproducible SCAPE-EasyOPD runtime under /opt only.
# Usage:
#   bash scripts/setup_scape_easyopd_env.sh /opt/scape-easyopd
# Optional env vars:
#   PYTHON_BIN=/opt/scape-h1003-hf-scorer/bin/python
#   REQUIRE_PYTEST=1

TARGET="${1:-/opt/scape-easyopd}"
PYTHON_BIN="${PYTHON_BIN:-/opt/scape-h1003-hf-scorer/bin/python}"
REQUIRE_PYTEST="${REQUIRE_PYTEST:-1}"
EASYOPD_ROOT="${EASYOPD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SCAPE_ROOT="${SCAPE_ROOT:-/mnt/songzijun/Capability_Evolution/SCAPE}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: python not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ -e "${TARGET}" && ! -d "${TARGET}" ]]; then
  echo "ERROR: target exists and is not a directory: ${TARGET}" >&2
  exit 1
fi

mkdir -p "${TARGET}"
"${PYTHON_BIN}" -m venv "${TARGET}"
source "${TARGET}/bin/activate"
python -m pip install --upgrade pip
python -m pip install torch pyyaml omegaconf pytest transformers peft accelerate 'ray[default]==2.47.1'
if [[ "${REQUIRE_PYTEST}" != "0" ]]; then
  python -m pytest --version
fi
export PYTHONPATH="${EASYOPD_ROOT}:${SCAPE_ROOT}:${PYTHONPATH:-}"
python - <<'PY2'
import sys
mods = ['torch', 'yaml', 'pytest', 'transformers', 'peft', 'accelerate']
print(sys.executable)
for m in mods:
    mod = __import__(m)
    print(m, getattr(mod, '__version__', 'n/a'))
PY2

echo "SCAPE-EasyOPD env ready at ${TARGET}"
