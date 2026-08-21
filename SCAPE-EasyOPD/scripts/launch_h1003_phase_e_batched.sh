#!/usr/bin/env bash
set -euo pipefail

ROOT="${EASYOPD_ROOT:-/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD}"
OUT_ROOT="${1:-${ROOT}/outputs/component_sweep_0818/h100_3_qwen3_faststart}"
source "${ROOT}/scripts/setup_scape_easyopd_smoke7_env.sh" >/dev/null
MODEL="${CANONICAL_STUDENT_BASE:-/mnt/songzijun/models/Qwen3-30B-A3B-Instruct-2507}"
LOG_DIR="${OUT_ROOT}/phase_e_logs_batched"
export LOG_DIR
mkdir -p "${LOG_DIR}"
: > "${LOG_DIR}/PIDS.tsv"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
export RAY_DISABLE_DASHBOARD=1
export RAY_memory_monitor_refresh_ms=0
export SCAPE_DISABLE_VLLM_ROLLOUT_LORA=1
unset PYTORCH_CUDA_ALLOC_CONF

run_cell() {
  local gpus="$1" cuda_visible="$2" component="$3" loss="$4" seed="$5" label="$6"
  local train_file="${OUT_ROOT}/${component}/OPD_TRAIN_ROWS.parquet"
  if [[ ! -s "${train_file}" ]]; then
    train_file="${OUT_ROOT}/${component}/TRAIN_STATES_5K.jsonl"
  fi
  if [[ ! -s "${train_file}" ]]; then
    echo "missing train file for ${label}: ${train_file}" >&2
    return 1
  fi
  local log="${LOG_DIR}/${label}.log"
  echo "START ${label} gpus=${gpus} cuda=${cuda_visible}" | tee -a "${LOG_DIR}/STATUS.log"
  (
    cd "${ROOT}"
    export CUDA_VISIBLE_DEVICES="${cuda_visible}"
    "$PYTHON_BIN" scripts/scape_component_opd.py train \
      --component "${component}" \
      --loss "${loss}" \
      --seed "${seed}" \
      --output-dir "${OUT_ROOT}" \
      --train-file "${train_file}" \
      --gpus "${gpus}" \
      --train-batch-size 4 \
      --rollout-tp "${gpus}" \
      --student-model "${MODEL}" \
      --teacher-model "${MODEL}" \
      --total-training-steps "${TOTAL_TRAINING_STEPS:-1}"
  ) >"${log}" 2>&1
  local rc=$?
  echo "DONE ${label} rc=${rc}" | tee -a "${LOG_DIR}/STATUS.log"
  echo "${gpus} ${cuda_visible} ${component} ${loss} ${seed} ${label} ${rc}" >> "${LOG_DIR}/PIDS.tsv"
  return "${rc}"
}

run_wave() {
  run_cell "$1" "$2" "$3" "$4" "$5" "$6" & p1=$!
  run_cell "$7" "$8" "$9" "${10}" "${11}" "${12}" & p2=$!
  wait "$p1"; r1=$?
  wait "$p2"; r2=$?
  if [[ "$r1" != 0 || "$r2" != 0 ]]; then
    exit 3
  fi
}

run_wave \
  4 "0,1,2,3" evidence_graph reverse_kl 42 evidence_graph_PURE42 \
  4 "4,5,6,7" sentence_compress reverse_kl 42 sentence_compress_PURE42

run_wave \
  4 "0,1,2,3" evidence_graph reverse_kl 43 evidence_graph_PURE43 \
  4 "4,5,6,7" sentence_compress reverse_kl 43 sentence_compress_PURE43

run_wave \
  4 "0,1,2,3" evidence_graph hybrid_rl_opd 42 evidence_graph_HYBRID42 \
  4 "4,5,6,7" sentence_compress hybrid_rl_opd 42 sentence_compress_HYBRID42

run_wave \
  4 "0,1,2,3" evidence_graph hybrid_rl_opd 43 evidence_graph_HYBRID43 \
  4 "4,5,6,7" sentence_compress hybrid_rl_opd 43 sentence_compress_HYBRID43

python - <<'PY'
import json, os, pathlib
log_dir = pathlib.Path(os.environ['LOG_DIR'])
rows=[]
for line in (log_dir/'PIDS.tsv').read_text().splitlines():
    gpus,cuda_visible,component,loss,seed,label,rc=line.split()
    log=log_dir/f'{label}.log'
    txt=log.read_text(errors='replace') if log.exists() else ''
    rows.append({'gpus':int(gpus),'cuda_visible':cuda_visible,'component':component,'loss':loss,'seed':int(seed),'label':label,'returncode':int(rc),'log':str(log),'log_tail':txt[-2000:]})
status='PHASE_E_BATCHED_COMPLETE' if all(r['returncode']==0 for r in rows) else 'PHASE_E_BATCHED_FAILED'
(log_dir/'PHASE_E_SUMMARY.json').write_text(json.dumps({'status':status,'jobs':rows}, indent=2, ensure_ascii=False)+'\n')
print(json.dumps({'status':status,'jobs':rows}, indent=2, ensure_ascii=False))
raise SystemExit(0 if status=='PHASE_E_BATCHED_COMPLETE' else 3)
PY
