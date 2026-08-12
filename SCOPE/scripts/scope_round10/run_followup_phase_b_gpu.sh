#!/usr/bin/env bash
# Followup Phase B: one GPU slot (0808-todo1.md §3).
# Requires CANONICAL_BACKEND_GATE.pass=true. Writes under outputs/scope_round10_followup/.
set -euo pipefail
source "$(dirname "$0")/_common_followup.sh"
followup_setup

GPU="${1:?gpu 0-7}"
VARIANT="${PHASE_B_VARIANTS[$GPU]}"
PORT="$(followup_port_for_gpu "${GPU}")"
VDIR="${OUT}/phase_b/${VARIANT}"
MARKER="${VDIR}/DONE"
GATE_A="${OUT}/CANONICAL_BACKEND_GATE.json"
STALE_SEC="${FOLLOWUP_STALE_SEC:-7200}"

if [[ ! -f "${GATE_A}" ]]; then
  followup_log "ERROR: missing CANONICAL_BACKEND_GATE.json — refuse Phase B"
  exit 2
fi
PASS=$(python -c "import json; print(json.load(open('${GATE_A}')).get('pass', False))")
if [[ "${PASS}" != "True" ]]; then
  followup_log "CANONICAL_BACKEND_GATE.pass=false — STOP_AFTER_PHASE_A; skip ${VARIANT}"
  exit 3
fi

if [[ -f "${MARKER}" ]]; then
  followup_log "Skip Phase B ${VARIANT} (DONE)"
  # GPU7 still aggregates even if threshold-only already done
  if [[ "${GPU}" == "7" ]]; then
    python training/scope_round10/aggregate_followup_phase_b_gate.py \
      >> "${LOG_DIR}/phase_b_aggregate.log" 2>&1 || true
  fi
  exit 0
fi

mkdir -p "${VDIR}/eval_offline_valid" "${VDIR}/eval_holdout" "${VDIR}/reports" "${VDIR}/canonical"

# Heartbeat for stall detection
heartbeat() {
  date -Is > "${VDIR}/HEARTBEAT"
}
heartbeat
HB_PID=""
start_hb_loop() {
  (
    while true; do
      heartbeat
      sleep 60
    done
  ) &
  HB_PID=$!
}
stop_hb_loop() {
  if [[ -n "${HB_PID}" ]] && kill -0 "${HB_PID}" 2>/dev/null; then
    kill "${HB_PID}" 2>/dev/null || true
  fi
  HB_PID=""
}
trap 'stop_hb_loop' EXIT

run_eval_split() {
  local split="$1" inp="$2"
  local canonical_out="${VDIR}/eval_${split}/canonical_vllm_replay.jsonl"
  local n_expected
  n_expected=$(wc -l < "${inp}" | tr -d ' ')
  if [[ ! -f "${canonical_out}" ]] || [[ "$(wc -l < "${canonical_out}" | tr -d ' ')" -lt "${n_expected}" ]]; then
    rm -f "${canonical_out}"
    followup_log "${VARIANT} canonical-vLLM ${split}"
    start_hb_loop
    SCOPE_VLLM_OUT_ROOT="${OUT}" CUDA_VISIBLE_DEVICES="${GPU}" \
      python training/scope_round9/run_vllm_replay_split.py \
      --model-path "${MODEL}" --input "${inp}" --output "${canonical_out}" \
      --port "${PORT}" --gpu "${GPU}" \
      >> "${LOG_DIR}/phase_b_${VARIANT}_${split}_canonical.log" 2>&1
    stop_hb_loop
    # symlink for aggregate helpers that look for vllm_replay.jsonl
    ln -sfn "$(basename "${canonical_out}")" "${VDIR}/eval_${split}/vllm_replay.jsonl"
    heartbeat
  fi
}

if [[ "${VARIANT}" == "r10_threshold_only_p0_seed42" ]]; then
  followup_log "${VARIANT}: threshold sweep on P0 seed42 (no train)"
  MODEL="$(followup_p0_merged 42)"
  heartbeat
  CUDA_VISIBLE_DEVICES="${GPU}" python training/scope_round10/calibrate_threshold_p0.py \
    --model-path "${MODEL}" \
    --offline "${OFFLINE_VALID}" \
    --holdout "${BASE_LIVE}" \
    --out-dir "${VDIR}" \
    --hf-offline-replay "${R10_OUT}/phase_a/seed42/offline_valid/hf_float32_replay.jsonl" \
    --hf-holdout-replay "${R10_OUT}/phase_a/seed42/base_live/hf_float32_replay.jsonl" \
    >> "${LOG_DIR}/phase_b_${VARIANT}.log" 2>&1
  heartbeat
  # Wait for other GPUs' DONE (with stall watchdog), then aggregate
  followup_log "GPU7 waiting for other Phase B variants..."
  for _ in $(seq 1 720); do
    heartbeat  # keep own HEARTBEAT fresh so global watchdog does not restart waiters
    done_n=0
    for v in "${PHASE_B_VARIANTS[@]}"; do
      [[ "${v}" == "r10_threshold_only_p0_seed42" ]] && continue
      [[ -f "${OUT}/phase_b/${v}/DONE" ]] && done_n=$((done_n + 1))
    done
    if [[ "${done_n}" -ge 7 ]]; then
      break
    fi
    # stall detection: kill/log if any HEARTBEAT too old while no DONE
    now=$(date +%s)
    for v in "${PHASE_B_VARIANTS[@]}"; do
      [[ "${v}" == "r10_threshold_only_p0_seed42" ]] && continue
      d="${OUT}/phase_b/${v}"
      [[ -f "${d}/DONE" ]] && continue
      hb="${d}/HEARTBEAT"
      if [[ -f "${hb}" ]]; then
        hb_ts=$(date -d "$(cat "${hb}")" +%s 2>/dev/null || echo 0)
        age=$((now - hb_ts))
        if [[ "${age}" -gt "${STALE_SEC}" ]]; then
          followup_log "WARN: ${v} stale heartbeat age=${age}s > ${STALE_SEC}s"
        fi
      fi
    done
    sleep 60
  done
  python training/scope_round10/aggregate_followup_phase_b_gate.py \
    >> "${LOG_DIR}/phase_b_aggregate.log" 2>&1
  touch "${MARKER}"
  followup_log "${VARIANT} DONE + aggregate"
  exit 0
fi

followup_log "${VARIANT} train"
start_hb_loop
# Quote --gpu=cuda:0 so bash never treats `cuda:0` as a label/command on resume.
CUDA_VISIBLE_DEVICES="${GPU}" python training/scope_round10/run_phase_b_train.py \
  --variant "${VARIANT}" --gpu="cuda:0" --out-root "${OUT}/phase_b" \
  >> "${LOG_DIR}/phase_b_${VARIANT}_train.log" 2>&1
stop_hb_loop
heartbeat

MODEL="${VDIR}/merged"
if [[ ! -f "${MODEL}/config.json" ]]; then
  followup_log "ERROR: missing merged model ${MODEL}"
  exit 1
fi

run_eval_split offline_valid "${OFFLINE_VALID}"
run_eval_split holdout "${BASE_LIVE}"

# Metrics from canonical replay (gold vs pred); also write parity self-check
python training/scope_round10/score_followup_variant.py \
  --variant-dir "${VDIR}" --variant "${VARIANT}" \
  --output "${VDIR}/TRAIN_AND_EVAL_REPORT.json" \
  >> "${LOG_DIR}/phase_b_${VARIANT}_score.log" 2>&1

followup_stop_recorded "vllm_port_${PORT}" || true
heartbeat
touch "${MARKER}"
followup_log "Phase B ${VARIANT} DONE"
