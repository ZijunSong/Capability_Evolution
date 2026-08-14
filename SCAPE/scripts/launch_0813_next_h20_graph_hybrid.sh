#!/usr/bin/env bash
# SCAPE-0813-Next-H20 Phase B — Graph-Hybrid micro 512/2K on 8×H20.
set -euo pipefail

SCAPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-${SCAPE_ROOT}/outputs/0813_next_h20}"
GH_ROOT="${OUT_ROOT}/graph_hybrid"
DATA_DIR="${GH_ROOT}/data"
MICRO="${GH_ROOT}/micro"
MODEL_PATH="${MODEL_PATH:-/data/ppnm/models/harness-1}"
PYTHON_BIN="${PYTHON_BIN:-/data/ppnm/miniconda3/envs/bishop/bin/python}"
LOG_DIR="${OUT_ROOT}/logs"
PID_DIR="${OUT_ROOT}/pids"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"

mkdir -p "${OUT_ROOT}" "${GH_ROOT}" "${DATA_DIR}" "${MICRO}" "${LOG_DIR}" "${PID_DIR}"

source /data/ppnm/miniconda3/etc/profile.d/conda.sh
conda activate bishop
export PYTHONPATH="${SCAPE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

if [[ ! -f "${DATA_DIR}/GH_TRAIN_8K.jsonl" ]]; then
  echo "[data] build graph-hybrid splits"
  "${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/build_graph_hybrid_splits.py" --out-dir "${DATA_DIR}"
fi

TRAIN="${DATA_DIR}/GH_TRAIN_8K.jsonl"
VALID="${DATA_DIR}/GH_VALID_1K.jsonl"
TEST="${DATA_DIR}/GH_TEST_1K.jsonl"

run_cell() {
  local gpu="$1" tag="$2" n="$3" seed="$4" loss="$5"
  local out="${MICRO}/gpu${gpu}/${tag}"
  if [[ -f "${out}/DONE" ]]; then
    echo "[skip] gpu${gpu} ${tag}"
    return 0
  fi
  mkdir -p "${out}"
  echo "[launch] gpu${gpu} ${tag} n=${n} seed=${seed} loss=${loss}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    "${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/run_true_scape_stage_l_cell.py" \
    --out "${out}" \
    --model-path "${MODEL_PATH}" \
    --train-jsonl "${TRAIN}" \
    --valid-jsonl "${VALID}" \
    --test-jsonl "${TEST}" \
    --component-id evidence_graph_hybrid \
    --n-samples "${n}" \
    --seed "${seed}" \
    --loss-path "${loss}" \
    --gpu 0 \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    >"${LOG_DIR}/gh_gpu${gpu}_${tag}.log" 2>&1
  touch "${out}/DONE"
}

run_gpu_queue() {
  local gpu="$1"
  local log="${LOG_DIR}/gh_gpu${gpu}_queue.log"
  {
    echo "[$(date -Iseconds)] gpu${gpu} Graph-Hybrid micro start"
    case "${gpu}" in
      0) run_cell 0 name_only_s42_L512 512 42 tool_name_only_kl; run_cell 0 name_only_s42_L2K 2000 42 tool_name_only_kl ;;
      1) run_cell 1 name_only_s43_L512 512 43 tool_name_only_kl; run_cell 1 name_only_s43_L2K 2000 43 tool_name_only_kl ;;
      2) run_cell 2 uniform_s42_L512 512 42 tool_token_kl; run_cell 2 uniform_s42_L2K 2000 42 tool_token_kl ;;
      3) run_cell 3 uniform_s43_L512 512 43 tool_token_kl; run_cell 3 uniform_s43_L2K 2000 43 tool_token_kl ;;
      4) run_cell 4 action_ce_s42_L2K 2000 42 action_ce ;;
      5) run_cell 5 full_kl_s42_L2K 2000 42 full_response_kl ;;
      6) run_cell 6 offpolicy_s42_L2K 2000 42 offpolicy_matched ;;
      7)
        echo "[gpu7] baseline profiler — V2/V3 token counts from data audit"
        "${PYTHON_BIN}" - <<'PY' "${DATA_DIR}/DATA_AUDIT.json" "${GH_ROOT}/BASELINE_RUNTIME.md"
import json, sys
from pathlib import Path
meta = json.loads(Path(sys.argv[1]).read_text())
lines = ["# Graph-Hybrid baseline runtime", "", f"splits: {list(meta.get('splits', {}).keys())}"]
Path(sys.argv[2]).write_text("\n".join(lines) + "\n")
PY
        ;;
    esac
    echo "[$(date -Iseconds)] gpu${gpu} Graph-Hybrid ALL_DONE"
    touch "${MICRO}/gpu${gpu}/ALL_DONE"
  } >>"${log}" 2>&1
}

for g in 0 1 2 3 4 5 6 7; do
  mkdir -p "${MICRO}/gpu${g}"
done

if [[ -n "${GPU_ONLY:-}" ]]; then
  run_gpu_queue "${GPU_ONLY}"
else
  for g in 0 1 2 3 4 5 6 7; do
    run_gpu_queue "${g}" &
    echo $! >"${PID_DIR}/gh_gpu${g}.pid"
  done
  echo "[launch] Graph-Hybrid micro 8-GPU under ${GH_ROOT}"
fi
