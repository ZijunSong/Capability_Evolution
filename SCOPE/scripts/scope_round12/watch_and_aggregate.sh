#!/usr/bin/env bash
# Poll Round12 A/B jobs; when complete, aggregate and decide Phase C.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
r12_setup

r12_log "watch_and_aggregate start"

while true; do
  done_n=0
  for job in M0_V0 M0_V1 M1_V0 M1_V1 M2_V0 M2_V1; do
    if [[ -f "${OUT}/phase_b_operation_boundary/cross_view_replays/${job}/DONE" ]]; then
      done_n=$((done_n + 1))
    fi
  done
  ckpt_done=0
  for sel in C11L C11P; do
    if [[ -f "${OUT}/phase_a_ckpt_provenance/per_selector_scores/${sel}_DONE" ]]; then
      ckpt_done=$((ckpt_done + 1))
    fi
  done

  # progress lines
  prog=""
  for job in M0_V0 M0_V1 M1_V0 M1_V1 M2_V0 M2_V1; do
    off=0; hol=0
    f1="${OUT}/phase_b_operation_boundary/cross_view_replays/${job}/eval_offline_valid/canonical_vllm_replay.jsonl"
    f2="${OUT}/phase_b_operation_boundary/cross_view_replays/${job}/eval_holdout/canonical_vllm_replay.jsonl"
    [[ -f "${f1}" ]] && off=$(wc -l < "${f1}" | tr -d ' ')
    [[ -f "${f2}" ]] && hol=$(wc -l < "${f2}" | tr -d ' ')
    prog+="${job}:${off}/${hol} "
  done
  for sel in C11L C11P; do
    n=0
    f="${OUT}/phase_a_ckpt_provenance/per_selector_scores/${sel}_oracle_replay.jsonl"
    [[ -f "${f}" ]] && n=$(wc -l < "${f}" | tr -d ' ')
    prog+="${sel}:${n} "
  done
  r12_log "watch cross=${done_n}/6 ckpt=${ckpt_done}/2 | ${prog}"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader \
    >> "${LOG_DIR}/gpu_watch.csv" 2>/dev/null || true

  if [[ "${done_n}" -ge 6 && "${ckpt_done}" -ge 2 ]]; then
    r12_log "All A/B GPU jobs DONE — aggregating"
    python training/scope_round12/build_canonical_ckpt_events.py >> "${LOG_DIR}/agg_a_events.log" 2>&1 || true
    python training/scope_round12/eval_selector_provenance.py >> "${LOG_DIR}/agg_a_prov.log" 2>&1 || true
    python training/scope_round12/ckpt_observability.py >> "${LOG_DIR}/agg_a_obs.log" 2>&1 || true
    python training/scope_round12/calibrate_boundary.py >> "${LOG_DIR}/agg_b_cal.log" 2>&1 || true
    python training/scope_round12/aggregate_round12.py >> "${LOG_DIR}/agg_round12.log" 2>&1 || true
    allow=$(python -c "import json; print(json.load(open('${OUT}/phase_b_operation_boundary/BARRIER_B_DECISION.json')).get('allow_phase_c_mainline', False))")
    r12_log "Barrier B allow_phase_c_mainline=${allow}"
    if [[ "${allow}" == "True" ]]; then
      bash "$(dirname "$0")/launch_phase_c_8gpu.sh" >> "${LOG_DIR}/phase_c_launch.log" 2>&1 || true
    else
      r12_log "STOP after operation boundary — no Phase C / closed-loop"
    fi
    # refresh report
    python training/scope_round12/write_report.py >> "${LOG_DIR}/write_report.log" 2>&1 || true
    r12_log "watch_and_aggregate DONE"
    exit 0
  fi

  # Stale recovery (45 min)
  now=$(date +%s)
  for gpu in 0 1 2 3 4 5; do
    job=$(echo M0_V0 M0_V1 M1_V0 M1_V1 M2_V0 M2_V1 | awk -v i=$gpu '{print $(i+1)}')
    root="${OUT}/phase_b_operation_boundary/cross_view_replays/${job}"
    [[ -f "${root}/DONE" ]] && continue
    [[ -f "${root}/HEARTBEAT" ]] || continue
    hb_ts=$(date -d "$(cat "${root}/HEARTBEAT")" +%s 2>/dev/null || echo 0)
    age=$((now - hb_ts))
    if [[ "${age}" -gt 2700 ]]; then
      r12_log "STALE ${job} age=${age}s — restart GPU${gpu}"
      r12_stop_recorded "vllm_port_$(r12_port_for_gpu "${gpu}")" || true
      pkill -f "run_cross_view_gpu.sh ${gpu} ${job}" 2>/dev/null || true
      sleep 2
      nohup bash "$(dirname "$0")/run_cross_view_gpu.sh" "${gpu}" "${job}" \
        >> "${LOG_DIR}/supervisor_cross_${job}.log" 2>&1 &
      echo $! > "${PID_DIR}/cross_gpu${gpu}.pid"
    fi
  done
  for pair in "6:C11L" "7:C11P"; do
    gpu="${pair%%:*}"; sel="${pair##*:}"
    [[ -f "${OUT}/phase_a_ckpt_provenance/per_selector_scores/${sel}_DONE" ]] && continue
    hb="${OUT}/phase_a_ckpt_provenance/per_selector_scores/${sel}_HEARTBEAT"
    [[ -f "${hb}" ]] || continue
    hb_ts=$(date -d "$(cat "${hb}")" +%s 2>/dev/null || echo 0)
    age=$((now - hb_ts))
    if [[ "${age}" -gt 2700 ]]; then
      r12_log "STALE ${sel} age=${age}s — restart GPU${gpu}"
      r12_stop_recorded "vllm_port_$(r12_port_for_gpu "${gpu}")" || true
      pkill -f "run_ckpt_selector_gpu.sh ${gpu} ${sel}" 2>/dev/null || true
      sleep 2
      nohup bash "$(dirname "$0")/run_ckpt_selector_gpu.sh" "${gpu}" "${sel}" \
        >> "${LOG_DIR}/supervisor_ckpt_${sel}.log" 2>&1 &
      echo $! > "${PID_DIR}/ckpt_gpu${gpu}.pid"
    fi
  done

  sleep 180
done
