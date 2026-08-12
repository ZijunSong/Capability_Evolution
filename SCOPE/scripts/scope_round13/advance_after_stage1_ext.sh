#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
r13_setup

r13_log "EXT advance_after_stage1: rebuild test SDI + gates + stage2 launch on free GPUs"
python training/scope_round13/build_operation_sdi.py --with-test \
  >> "${LOG_DIR}/build_operation_sdi_test.log" 2>&1 || true
python training/scope_round13/stage1_gates.py \
  >> "${LOG_DIR}/stage1_gates.log" 2>&1 || true

# Launch remaining Stage2 seeds on free GPUs (5,6,7) if data gate passes
GATE="${OUT}/stage2_targeted/DATASET_GATE.json"
if [[ -f "${GATE}" ]]; then
  pass=$(python -c "import json;print(json.load(open('${GATE}')).get('NONDEGENERATE_STAGE2_DATA_PASS', False))")
  if [[ "${pass}" == "True" ]]; then
    declare -A S2=( [5]=r13_ckpt_pointer_seed42 [6]=r13_ckpt_pointer_seed43 [7]=r13_ckpt_pointer_seed44 )
    for gpu in 5 6 7; do
      v="${S2[$gpu]}"
      if [[ -f "${OUT}/stage2_targeted/training/${v}/DONE" ]]; then
        continue
      fi
      # skip if already running
      if pgrep -f "run_stage2_gpu.sh ${gpu} " >/dev/null 2>&1; then
        continue
      fi
      nohup bash "$(dirname "$0")/run_stage2_gpu.sh" "${gpu}" "${v}" \
        >> "${LOG_DIR}/stage2_${v}_supervisor.log" 2>&1 &
      echo $! > "${PID_DIR}/stage2_gpu${gpu}.pid"
      r13_log "started stage2 ${v} on GPU${gpu} pid=$!"
      sleep 5
    done
  fi
fi

python training/scope_round13/write_round13_report.py \
  >> "${LOG_DIR}/write_report.log" 2>&1 || true
touch "${MARKER_DIR}/post_stage1_advanced"
