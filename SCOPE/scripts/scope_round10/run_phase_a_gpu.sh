#!/usr/bin/env bash
# Phase A per-GPU queue (0807 §4). GPU0-2 HF float32; GPU3-5 vLLM fixed; GPU6 audits; GPU7 report/tests.
set -euo pipefail
source "$(dirname "$0")/_common_r10.sh"
scope10_setup

GPU="${1:?gpu}"
PORT="$(scope10_port_for_gpu "${GPU}")"
MARKER="${MARKER_DIR}/phase_a_gpu${GPU}.DONE"
[[ -f "${MARKER}" ]] && { scope10_log "Skip Phase A GPU${GPU}"; exit 0; }

P0() { echo "${P0_ROOT}/rollback_hier_o7_seed${1}/merged"; }

run_hf() {
  local seed="$1" split="$2" inp="$3"
  local dtype="${4:-bfloat16}"
  local tag="hf_${dtype}_replay"
  # Keep legacy float32 filename for offline_valid when dtype=float32
  if [[ "${dtype}" == "float32" ]]; then
    tag="hf_float32_replay"
  elif [[ "${dtype}" == "bfloat16" ]]; then
    tag="hf_bf16_replay"
  fi
  # Also accept either name as complete for skip logic on offline
  local out="${OUT}/phase_a/seed${seed}/${split}/${tag}.jsonl"
  local n_exp; n_exp=$(wc -l < "${inp}" | tr -d ' ')
  if [[ -f "${out}" ]] && [[ "$(wc -l < "${out}" | tr -d ' ')" -ge "${n_exp}" ]]; then
    scope10_log "GPU${GPU} skip HF seed${seed} ${split} (${dtype})"
    return 0
  fi
  # Compatibility: if float32 offline already done, don't redo as bf16 for offline
  if [[ "${split}" == "offline_valid" && "${dtype}" == "bfloat16" ]]; then
    local alt="${OUT}/phase_a/seed${seed}/${split}/hf_float32_replay.jsonl"
    if [[ -f "${alt}" ]] && [[ "$(wc -l < "${alt}" | tr -d ' ')" -ge "${n_exp}" ]]; then
      scope10_log "GPU${GPU} skip HF seed${seed} ${split} (float32 already present)"
      return 0
    fi
  fi
  rm -f "${out}"
  scope10_log "GPU${GPU} HF ${dtype} seed${seed} ${split}"
  CUDA_VISIBLE_DEVICES="${GPU}" python training/scope_round9/replay_frozen_hf.py \
    --model-path "$(P0 "${seed}")" --input "${inp}" --output "${out}" \
    --device cuda:0 --dtype "${dtype}" \
    >> "${LOG_DIR}/phase_a_gpu${GPU}_hf_seed${seed}_${split}.log" 2>&1
}

run_vllm() {
  local seed="$1" split="$2" inp="$3"
  local out="${OUT}/phase_a/seed${seed}/${split}/vllm_fixed_replay.jsonl"
  local n_exp; n_exp=$(wc -l < "${inp}" | tr -d ' ')
  if [[ -f "${out}" ]] && [[ "$(wc -l < "${out}" | tr -d ' ')" -ge "${n_exp}" ]]; then
    scope10_log "GPU${GPU} skip vLLM seed${seed} ${split}"
    return 0
  fi
  rm -f "${out}"
  scope10_log "GPU${GPU} vLLM fixed seed${seed} ${split}"
  CUDA_VISIBLE_DEVICES="${GPU}" python training/scope_round9/run_vllm_replay_split.py \
    --model-path "$(P0 "${seed}")" --input "${inp}" --output "${out}" \
    --port "${PORT}" --gpu "${GPU}" \
    >> "${LOG_DIR}/phase_a_gpu${GPU}_vllm_seed${seed}_${split}.log" 2>&1
}

case "${GPU}" in
  0)
    # offline: float32 for numerical audit; holdout: bf16 for throughput (residual float32 on GPU6)
    run_hf 42 offline_valid "${OFFLINE_VALID}" float32
    run_hf 42 base_live "${BASE_LIVE}" bfloat16
    CUDA_VISIBLE_DEVICES=0 python training/scope_round10/audit_adapter_merged.py \
      --seed 42 --n 80 --device cuda:0 \
      >> "${LOG_DIR}/phase_a_gpu0_adapter.log" 2>&1 || true
    ;;
  1)
    run_hf 43 offline_valid "${OFFLINE_VALID}" float32
    run_hf 43 base_live "${BASE_LIVE}" bfloat16
    ;;
  2)
    run_hf 44 offline_valid "${OFFLINE_VALID}" float32
    run_hf 44 base_live "${BASE_LIVE}" bfloat16
    ;;
  3)
    run_vllm 42 offline_valid "${OFFLINE_VALID}"
    run_vllm 42 base_live "${BASE_LIVE}"
    ;;
  4)
    run_vllm 43 offline_valid "${OFFLINE_VALID}"
    run_vllm 43 base_live "${BASE_LIVE}"
    ;;
  5)
    run_vllm 44 offline_valid "${OFFLINE_VALID}"
    run_vllm 44 base_live "${BASE_LIVE}"
    ;;
  6)
    # dtype / near-boundary audit using residual ledgers + optional float32 rescore
    for seed in 42 43 44; do
      for split in offline_valid base_live; do
        residual="${OUT}/phase_a/seed${seed}/${split}/residual_mismatch.jsonl"
        [[ -s "${residual}" ]] || continue
        CUDA_VISIBLE_DEVICES=6 python training/scope_round10/rescore_mismatch_hf.py \
          --model-path "$(P0 "${seed}")" --input "${residual}" \
          --output "${OUT}/phase_a/seed${seed}/${split}/float32_rescore.jsonl" \
          --device cuda:0 --dtype float32 \
          >> "${LOG_DIR}/phase_a_gpu6_float32_seed${seed}_${split}.log" 2>&1 || true
      done
    done
    CUDA_VISIBLE_DEVICES=6 python training/scope_round10/audit_numerical_boundary.py \
      >> "${LOG_DIR}/phase_a_gpu6_boundary.log" 2>&1 || true
    ;;
  7)
    # CPU/unit tests + wait-and-aggregate helper
    pytest tests/scope_round9/test_barrier_a_neartie.py \
      tests/scope_round9/test_vllm_token_id_scoring.py \
      tests/scope/test_decide_disable_replan_parity.py -q --tb=line \
      >> "${LOG_DIR}/phase_a_gpu7_tests.log" 2>&1 || true
    python training/scope_round10/phase_a_wait_aggregate.py \
      >> "${LOG_DIR}/phase_a_gpu7_aggregate.log" 2>&1
    ;;
  *)
    scope10_log "bad gpu ${GPU}"; exit 2 ;;
esac

touch "${MARKER}"
scope10_log "Phase A GPU${GPU} DONE"
