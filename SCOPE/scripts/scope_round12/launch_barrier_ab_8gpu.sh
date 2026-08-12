#!/usr/bin/env bash
# Launch Round12 Barrier A/B parallel jobs across 8×H20.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
r12_setup

r12_log "=== Launch Barrier A/B 8gpu ==="

# GPU0-5: cross-view matrix
JOBS=(M0_V0 M0_V1 M1_V0 M1_V1 M2_V0 M2_V1)
for gpu in 0 1 2 3 4 5; do
  job="${JOBS[$gpu]}"
  r12_stop_recorded "cross_gpu${gpu}" || true
  r12_stop_recorded "vllm_port_$(r12_port_for_gpu "${gpu}")" || true
  nohup bash "$(dirname "$0")/run_cross_view_gpu.sh" "${gpu}" "${job}" \
    >> "${LOG_DIR}/supervisor_cross_${job}.log" 2>&1 &
  echo $! > "${PID_DIR}/cross_gpu${gpu}.pid"
  r12_log "started GPU${gpu} ${job} pid=$!"
  sleep 20  # stagger vLLM startups
done

# GPU6/7: C11L / C11P oracle Stage2 re-verify
r12_stop_recorded "ckpt_gpu6" || true
r12_stop_recorded "vllm_port_$(r12_port_for_gpu 6)" || true
nohup bash "$(dirname "$0")/run_ckpt_selector_gpu.sh" 6 C11L \
  >> "${LOG_DIR}/supervisor_ckpt_C11L.log" 2>&1 &
echo $! > "${PID_DIR}/ckpt_gpu6.pid"
r12_log "started GPU6 C11L pid=$!"
sleep 20

r12_stop_recorded "ckpt_gpu7" || true
r12_stop_recorded "vllm_port_$(r12_port_for_gpu 7)" || true
nohup bash "$(dirname "$0")/run_ckpt_selector_gpu.sh" 7 C11P \
  >> "${LOG_DIR}/supervisor_ckpt_C11P.log" 2>&1 &
echo $! > "${PID_DIR}/ckpt_gpu7.pid"
r12_log "started GPU7 C11P pid=$!"

# Monitor
nohup bash "$(dirname "$0")/monitor_loop.sh" ab \
  >> "${LOG_DIR}/monitor_loop.log" 2>&1 &
echo $! > "${PID_DIR}/monitor_loop.pid"
r12_log "monitor_loop pid=$!"
r12_log "=== Launch complete ==="
