#!/usr/bin/env bash
# Restart failed Round12 GPU jobs with correct physical --gpu binding.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
r12_setup

r12_log "restart_failed_gpus: cuda check"
python -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count()>=8, (torch.cuda.is_available(), torch.cuda.device_count())'
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tee -a "${LOG_DIR}/restart_failed_gpus.log"

# Keep GPU0 if still running; restart 1-7
for gpu in 1 2 3 4 5 6 7; do
  r12_stop_recorded "vllm_port_$(r12_port_for_gpu "${gpu}")" || true
  if [[ "${gpu}" -le 5 ]]; then
    r12_stop_recorded "cross_gpu${gpu}" || true
    pkill -f "run_cross_view_gpu.sh ${gpu} " 2>/dev/null || true
  else
    r12_stop_recorded "ckpt_gpu${gpu}" || true
    pkill -f "run_ckpt_selector_gpu.sh ${gpu} " 2>/dev/null || true
  fi
done
sleep 3

for job in M0_V1 M1_V0 M1_V1 M2_V0 M2_V1; do
  rm -f "${OUT}/phase_b_operation_boundary/cross_view_replays/${job}/DONE"
  rm -f "${OUT}/phase_b_operation_boundary/cross_view_replays/${job}/eval_offline_valid/canonical_vllm_replay.jsonl"
  rm -f "${OUT}/phase_b_operation_boundary/cross_view_replays/${job}/eval_holdout/canonical_vllm_replay.jsonl"
done
rm -f "${OUT}/phase_a_ckpt_provenance/per_selector_scores/C11L_DONE" \
      "${OUT}/phase_a_ckpt_provenance/per_selector_scores/C11P_DONE" \
      "${OUT}/phase_a_ckpt_provenance/per_selector_scores/C11L_oracle_replay.jsonl" \
      "${OUT}/phase_a_ckpt_provenance/per_selector_scores/C11P_oracle_replay.jsonl"

JOBS=(M0_V0 M0_V1 M1_V0 M1_V1 M2_V0 M2_V1)
for gpu in 1 2 3 4 5; do
  job="${JOBS[$gpu]}"
  nohup bash "$(dirname "$0")/run_cross_view_gpu.sh" "${gpu}" "${job}" \
    >> "${LOG_DIR}/supervisor_cross_${job}.log" 2>&1 &
  echo $! > "${PID_DIR}/cross_gpu${gpu}.pid"
  r12_log "RESTART GPU${gpu} ${job} pid=$!"
  sleep 40
done

nohup bash "$(dirname "$0")/run_ckpt_selector_gpu.sh" 6 C11L \
  >> "${LOG_DIR}/supervisor_ckpt_C11L.log" 2>&1 &
echo $! > "${PID_DIR}/ckpt_gpu6.pid"
r12_log "RESTART GPU6 C11L pid=$!"
sleep 40

nohup bash "$(dirname "$0")/run_ckpt_selector_gpu.sh" 7 C11P \
  >> "${LOG_DIR}/supervisor_ckpt_C11P.log" 2>&1 &
echo $! > "${PID_DIR}/ckpt_gpu7.pid"
r12_log "RESTART GPU7 C11P pid=$!"

sleep 60
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader \
  | tee -a "${LOG_DIR}/restart_failed_gpus.log"
r12_log "restart_failed_gpus complete"
