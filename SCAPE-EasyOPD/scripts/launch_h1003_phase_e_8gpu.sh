#!/usr/bin/env bash
set -euo pipefail

ROOT="${EASYOPD_ROOT:-/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD}"
OUT_ROOT="${1:-${ROOT}/outputs/component_sweep_0818/h100_3_qwen3_faststart}"
source "${ROOT}/scripts/setup_scape_easyopd_smoke7_env.sh" >/dev/null
MODEL="${CANONICAL_STUDENT_BASE:-/mnt/songzijun/models/Qwen3-30B-A3B-Instruct-2507}"
LOG_DIR="${OUT_ROOT}/phase_e_logs"
export LOG_DIR
mkdir -p "${LOG_DIR}"

launch_cell() {
  local gpu="$1" component="$2" loss="$3" seed="$4" label="$5"
  local train_file="${OUT_ROOT}/${component}/OPD_TRAIN_ROWS.parquet"
  if [[ ! -s "${train_file}" ]]; then
    train_file="${OUT_ROOT}/${component}/TRAIN_STATES_5K.jsonl"
  fi
  local log="${LOG_DIR}/${label}.log"
  if [[ ! -s "${train_file}" ]]; then
    echo "missing train file for ${label}: ${train_file}" >&2
    return 1
  fi
  (
    cd "${ROOT}"
    export CUDA_VISIBLE_DEVICES="${gpu}"
    "$PYTHON_BIN" scripts/scape_component_opd.py train \
      --component "${component}" \
      --loss "${loss}" \
      --seed "${seed}" \
      --output-dir "${OUT_ROOT}" \
      --train-file "${train_file}" \
      --gpus 1 \
      --student-model "${MODEL}" \
      --teacher-model "${MODEL}" \
      --total-training-steps "${TOTAL_TRAINING_STEPS:-1}"
  ) >"${log}" 2>&1 &
  echo "$! ${gpu} ${component} ${loss} ${seed} ${label}" >> "${LOG_DIR}/PIDS.tsv"
}

: > "${LOG_DIR}/PIDS.tsv"
launch_cell 0 evidence_graph reverse_kl 42 evidence_graph_PURE42
launch_cell 1 evidence_graph reverse_kl 43 evidence_graph_PURE43
launch_cell 2 evidence_graph hybrid_rl_opd 42 evidence_graph_HYBRID42
launch_cell 3 evidence_graph hybrid_rl_opd 43 evidence_graph_HYBRID43
launch_cell 4 sentence_compress reverse_kl 42 sentence_compress_PURE42
launch_cell 5 sentence_compress reverse_kl 43 sentence_compress_PURE43
launch_cell 6 sentence_compress hybrid_rl_opd 42 sentence_compress_HYBRID42
launch_cell 7 sentence_compress hybrid_rl_opd 43 sentence_compress_HYBRID43

echo "Launched H100-3 Phase E jobs. Logs: ${LOG_DIR}"
cat "${LOG_DIR}/PIDS.tsv"
wait

python - <<'PY'
import json, os, pathlib, subprocess
log_dir = pathlib.Path(os.environ['LOG_DIR']) if 'LOG_DIR' in os.environ else pathlib.Path('phase_e_logs')
rows=[]
for line in (log_dir/'PIDS.tsv').read_text().splitlines():
    pid,gpu,component,loss,seed,label=line.split()
    log=log_dir/f'{label}.log'
    txt=log.read_text(errors='replace') if log.exists() else ''
    rows.append({'pid':int(pid),'gpu':int(gpu),'component':component,'loss':loss,'seed':int(seed),'label':label,'log':str(log),'completed':True,'returncode_hint':'see_log','log_tail':txt[-2000:]})
(log_dir/'PHASE_E_SUMMARY.json').write_text(json.dumps({'jobs':rows}, indent=2, ensure_ascii=False)+'\n')
print(json.dumps({'jobs':rows}, indent=2, ensure_ascii=False))
PY
