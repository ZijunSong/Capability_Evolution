#!/usr/bin/env bash
# Create and configure the BiSHOP conda environment.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BASE="${CONDA_BASE:-/data/ppnm/miniconda3}"
ENV_NAME="${BISHOP_CONDA_ENV:-bishop}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Conda env '${ENV_NAME}' already exists; updating packages..."
  conda activate "${ENV_NAME}"
else
  echo "Creating conda env '${ENV_NAME}' (Python 3.11)..."
  conda create -n "${ENV_NAME}" python=3.11 pip -y
  conda activate "${ENV_NAME}"
fi

cd "${REPO_ROOT}"

echo "Upgrading pip..."
python -m pip install --upgrade pip wheel setuptools

echo "Installing tinker-cookbook (editable)..."
python -m pip install -e ./tinker-cookbook

echo "Installing BiSHOP / harness-1 (editable)..."
python -m pip install -e .

echo "Installing optional rollout (vLLM) and dev tools..."
python -m pip install \
  'vllm>=0.13.0' \
  rank-bm25>=0.2.2 \
  datasketch>=1.6.5 \
  pytest>=8.3.3 \
  ruff>=0.14.3 \
  mypy>=1.18.2

if [[ ! -f "${REPO_ROOT}/.env" && -f "${REPO_ROOT}/.env.example" ]]; then
  cp "${REPO_ROOT}/.env.example" "${REPO_ROOT}/.env"
  echo "Created .env from .env.example — fill in your API keys before running eval."
fi

echo ""
echo "Running smoke tests..."
cd "${REPO_ROOT}"
PYTHONPATH=. python tests/smoke_imports.py
PYTHONPATH=. python -m pytest tests/test_module_config.py tests/test_module_fallbacks.py \
  tests/test_teacher_student_views.py tests/test_opd_alignment.py tests/test_lifecycle_decision.py -q

cat <<EOF

BiSHOP conda environment is ready.

  conda activate ${ENV_NAME}
  cd ${REPO_ROOT}
  export PYTHONPATH=.

Before full BrowseComp+ eval, edit ${REPO_ROOT}/.env with your API keys.

Quick checks:
  PYTHONPATH=. python inference/evaluate_modules.py --dry-run
  bash scripts/smoke_opd_vllm_hf_4gpu.sh   # vLLM rollout + HF train smoke (needs 4 GPUs)

EOF
