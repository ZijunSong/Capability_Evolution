#!/usr/bin/env bash
# Launch true SCAPE plumbing smoke on 8×H20:
#   Group A = GPU0-3  (P0→P3)
#   Group B = GPU4-7  (Q0→Q3)
set -euo pipefail

SCAPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-${SCAPE_ROOT}/outputs/true_scape_pipeline_smoke}"
MODEL_PATH="${MODEL_PATH:-/data/ppnm/models/Qwen2.5-7B-Instruct}"
PYTHON_BIN="${PYTHON_BIN:-/data/ppnm/miniconda3/envs/bishop/bin/python}"
LOG_DIR="${SCAPE_ROOT}/logs/true_scape_smoke"
EPOCHS="${EPOCHS:-1}"

mkdir -p "${OUT_ROOT}/group_a" "${OUT_ROOT}/group_b" "${LOG_DIR}" "${OUT_ROOT}/pids"

source /data/ppnm/miniconda3/etc/profile.d/conda.sh
conda activate bishop

export PYTHONPATH="${SCAPE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

echo "[preflight] running scripts/preflight_scape.py"
"${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/preflight_scape.py" \
  --model-path "${MODEL_PATH}" \
  --json-out "${OUT_ROOT}/PREFLIGHT.json"

# Kill stale smoke sessions if any
for s in scape_smoke_A scape_smoke_B scape_smoke_agg scape_smoke_mon; do
  screen -S "$s" -X quit 2>/dev/null || true
done

# Group A: GPU 0-3
screen -dmS scape_smoke_A bash -c "
  source /data/ppnm/miniconda3/etc/profile.d/conda.sh
  conda activate bishop
  export PYTHONPATH='${SCAPE_ROOT}'
  export CUDA_VISIBLE_DEVICES=0,1,2,3
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export TOKENIZERS_PARALLELISM=false
  cd '${SCAPE_ROOT}'
  echo \$\$ > '${OUT_ROOT}/pids/group_a.pid'
  '${PYTHON_BIN}' scripts/run_true_scape_pipeline_smoke.py \
    --group A \
    --out '${OUT_ROOT}/group_a' \
    --model-path '${MODEL_PATH}' \
    --component-id evidence_graph \
    --epochs '${EPOCHS}' \
    2>&1 | tee '${LOG_DIR}/group_a.log'
  echo EXIT:\${PIPESTATUS[0]} | tee -a '${LOG_DIR}/group_a.log'
"

# Group B: GPU 4-7 (parallel)
screen -dmS scape_smoke_B bash -c "
  source /data/ppnm/miniconda3/etc/profile.d/conda.sh
  conda activate bishop
  export PYTHONPATH='${SCAPE_ROOT}'
  export CUDA_VISIBLE_DEVICES=4,5,6,7
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export TOKENIZERS_PARALLELISM=false
  cd '${SCAPE_ROOT}'
  echo \$\$ > '${OUT_ROOT}/pids/group_b.pid'
  '${PYTHON_BIN}' scripts/run_true_scape_pipeline_smoke.py \
    --group B \
    --out '${OUT_ROOT}/group_b' \
    --model-path '${MODEL_PATH}' \
    --component-id evidence_graph \
    --epochs '${EPOCHS}' \
    2>&1 | tee '${LOG_DIR}/group_b.log'
  echo EXIT:\${PIPESTATUS[0]} | tee -a '${LOG_DIR}/group_b.log'
"

# Monitor + aggregate when both DONE
screen -dmS scape_smoke_mon bash -c "
  source /data/ppnm/miniconda3/etc/profile.d/conda.sh
  conda activate bishop
  export PYTHONPATH='${SCAPE_ROOT}'
  cd '${SCAPE_ROOT}'
  '${SCAPE_ROOT}/scripts/monitor_true_scape_smoke.sh' \
    '${OUT_ROOT}' '${LOG_DIR}' '${PYTHON_BIN}'
"

echo "[launch] screens:"
screen -ls | grep scape_smoke || true
echo "[launch] OUT_ROOT=${OUT_ROOT}"
echo "[launch] logs: ${LOG_DIR}/group_{a,b}.log"
