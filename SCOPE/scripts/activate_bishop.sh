#!/usr/bin/env bash
# Quick activation helper for the BiSHOP conda environment.
CONDA_BASE="${CONDA_BASE:-/data/ppnm/miniconda3}"
ENV_NAME="${BISHOP_CONDA_ENV:-bishop}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
cd "${REPO_ROOT}"
export PYTHONPATH=.

echo "Active: $(which python) ($(python --version))"
echo "Repo:   ${REPO_ROOT}"
echo "PYTHONPATH=."
