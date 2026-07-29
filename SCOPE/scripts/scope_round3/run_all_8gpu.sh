#!/usr/bin/env bash
# SCOPE Round 3 — shared environment
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_BASE="${CONDA_BASE:-/data/ppnm/miniconda3}"
ENV_NAME="${BISHOP_CONDA_ENV:-bishop}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
export PYTHONPATH="${REPO_ROOT}"
cd "${REPO_ROOT}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-dummy}"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export JAVA_HOME="${JAVA_HOME:-$CONDA_PREFIX/lib/jvm}"
export PATH="${JAVA_HOME}/bin:${PATH}"
export JVM_PATH="${JVM_PATH:-$JAVA_HOME/lib/server/libjvm.so}"
export BROWSECOMPPLUS_ANSWERS_PATH="${BROWSECOMPPLUS_ANSWERS_PATH:-$REPO_ROOT/external/BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl}"
export BROWSECOMPPLUS_QUERIES_PATH="${BROWSECOMPPLUS_QUERIES_PATH:-$REPO_ROOT/external/BrowseComp-Plus/topics-qrels/queries.tsv}"
export BROWSECOMP_BM25_INDEX_PATH="${BROWSECOMP_BM25_INDEX_PATH:-$REPO_ROOT/external/BrowseComp-Plus/indexes/bm25}"
if [[ -f "${REPO_ROOT}/.env" ]]; then set -a; source "${REPO_ROOT}/.env"; set +a; fi

ROOT="${REPO_ROOT}/outputs/scope_round3"
MANIFEST="${REPO_ROOT}/artifacts/datasets/round2_audit_100q/query_manifest.json"
BASE_MODEL="/data/ppnm/models/Qwen2.5-7B-Instruct"
ROUND1_MODEL="${REPO_ROOT}/outputs/dup_sdi_round1/merged_hf"
ROUND2_MAIN="${REPO_ROOT}/outputs/scope_round2/training/round2_main"
ROUND2_LEGACY="${REPO_ROOT}/outputs/scope_round2/training/round2_legacy_token_ce"
HARNESS_V2="${REPO_ROOT}/harness/configs/modules_minimal_v2.yaml"
DATASET_DIR="${REPO_ROOT}/artifacts/datasets/dup_sdi_round3"

_log_dir() {
  local gpu="$1" task="$2"
  local d="${ROOT}/logs/gpu${gpu}/${task}"
  mkdir -p "${d}"
  echo "${d}"
}

_task_status() {
  local gpu="$1" task="$2" code="$3"
  local d
  d="$(_log_dir "${gpu}" "${task}")"
  echo "${code}" > "${d}/status"
  date -Iseconds > "${d}/end_time"
}

_mark_done() {
  local gpu="$1" task="$2"
  _task_status "${gpu}" "${task}" "DONE"
}

_mark_failed() {
  local gpu="$1" task="$2"
  _task_status "${gpu}" "${task}" "FAILED"
}

_check_done() {
  local gpu="$1" task="$2" expected="$3"
  local d="${ROOT}/logs/gpu${gpu}/${task}"
  [[ -f "${d}/status" && "$(cat "${d}/status")" == "DONE" ]] && return 0
  return 1
}

barrier_a() {
  echo "=== Barrier A: tests ==="
  pytest tests/scope/ -q --tb=short
  echo "Barrier A PASS"
}

wave4_shard() {
  local gpu="$1" variant="$2" shard="$3" model="$4" port="$5"
  local out="${ROOT}/wave4_diagnostic/${variant}/${shard}"
  local task="wave4_${variant}_${shard}"
  local logd
  logd="$(_log_dir "${gpu}" "${task}")"
  if [[ -f "${out}/summary.json" ]]; then
    _mark_done "${gpu}" "${task}"
    return 0
  fi
  echo "$$" > "${logd}/pid"
  date -Iseconds > "${logd}/start_time"
  set +e
  CUDA_VISIBLE_DEVICES="${gpu}" python training/scope_round3/hmin_v2_dup_rollout.py \
    --output-dir "${out}" \
    --manifest "${MANIFEST}" \
    --shard "${shard}" \
    --n-shards 8 \
    --model-path "${model}" \
    --harness-config "${HARNESS_V2}" \
    --vllm-port "${port}" \
    --dup-operation \
    --parallel 1 \
    2>&1 | tee "${logd}/stdout.log"
  local rc=$?
  set -e
  if [[ ${rc} -eq 0 && -f "${out}/summary.json" ]]; then
    _mark_done "${gpu}" "${task}"
  else
    _mark_failed "${gpu}" "${task}"
    return "${rc}"
  fi
}

bilateral_shard() {
  local gpu="$1" shard="$2" port="$3"
  local rollout_out="${ROOT}/bilateral_rollout/${shard}"
  local label_out="${ROOT}/bilateral_labeling/${shard}"
  local task="bilateral_${shard}"
  local logd
  logd="$(_log_dir "${gpu}" "${task}")"
  if [[ -f "${label_out}/stats.json" ]]; then
    _mark_done "${gpu}" "${task}"
    return 0
  fi
  echo "$$" > "${logd}/pid"
  date -Iseconds > "${logd}/start_time"
  set +e
  CUDA_VISIBLE_DEVICES="${gpu}" python training/scope_round3/hmin_v2_dup_rollout.py \
    --output-dir "${rollout_out}" \
    --manifest "${MANIFEST}" \
    --shard "${shard}" \
    --n-shards 8 \
    --model-path "${BASE_MODEL}" \
    --harness-config "${HARNESS_V2}" \
    --vllm-port "${port}" \
    --collect-states-only \
    --parallel 1 \
    2>&1 | tee "${logd}/rollout.log"
  local rc=$?
  if [[ ${rc} -ne 0 ]]; then _mark_failed "${gpu}" "${task}"; return "${rc}"; fi
  CUDA_VISIBLE_DEVICES="${gpu}" python training/scope_round3/bilateral_label.py \
    --states "${rollout_out}/decision_states.jsonl" \
    --output-dir "${label_out}" \
    2>&1 | tee "${logd}/label.log"
  rc=$?
  set -e
  if [[ ${rc} -eq 0 && -f "${label_out}/stats.json" ]]; then
    _mark_done "${gpu}" "${task}"
  else
    _mark_failed "${gpu}" "${task}"
    return "${rc}"
  fi
}

train_variant() {
  local gpu="$1" name="$2" extra="${3:-}"
  local out="${ROOT}/training/${name}"
  local task="train_${name}"
  local logd
  logd="$(_log_dir "${gpu}" "${task}")"
  if [[ -f "${out}/train_summary.json" ]]; then
    _mark_done "${gpu}" "${task}"
    return 0
  fi
  echo "$$" > "${logd}/pid"
  date -Iseconds > "${logd}/start_time"
  set +e
  CUDA_VISIBLE_DEVICES="${gpu}" python training/train_sdi_dup.py \
    --config configs/scope/sdi_dup_round3_main.yaml \
    --train "${DATASET_DIR}/train.jsonl" \
    --valid "${DATASET_DIR}/valid.jsonl" \
    --output-dir "${out}" \
    ${extra} \
    2>&1 | tee "${logd}/stdout.log"
  local rc=$?
  set -e
  if [[ ${rc} -eq 0 && -f "${out}/train_summary.json" ]]; then
    _mark_done "${gpu}" "${task}"
  else
    _mark_failed "${gpu}" "${task}"
    return "${rc}"
  fi
}

merge_lora() {
  local gpu="$1" name="$2"
  local adapter="${ROOT}/training/${name}"
  local merged="${ROOT}/merged/${name}"
  local task="merge_${name}"
  if [[ -f "${merged}/config.json" ]]; then
    _mark_done "${gpu}" "${task}"
    return 0
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" python training/merge_lora_hf.py \
    --base "${BASE_MODEL}" \
    --adapter "${adapter}" \
    --output "${merged}" \
    && _mark_done "${gpu}" "${task}" || _mark_failed "${gpu}" "${task}"
}

offline_eval() {
  local gpu="$1" name="$2"
  local adapter="${ROOT}/training/${name}"
  local out="${ROOT}/eval/${name}_capability.json"
  local task="offline_${name}"
  if [[ -f "${out}" ]]; then _mark_done "${gpu}" "${task}"; return 0; fi
  CUDA_VISIBLE_DEVICES="${gpu}" python training/scope/eval_dup_capability.py \
    --valid "${DATASET_DIR}/valid.jsonl" \
    --model-path "${BASE_MODEL}" \
    --adapter-path "${adapter}" \
    --output "${out}" \
    && _mark_done "${gpu}" "${task}" || _mark_failed "${gpu}" "${task}"
}

closed_loop_100q() {
  local gpu="$1" name="$2" port="$3"
  local merged="${ROOT}/merged/${name}"
  local out="${ROOT}/closed_loop/${name}"
  local task="cl_${name}"
  local logd
  logd="$(_log_dir "${gpu}" "${task}")"
  if [[ -f "${out}/summary.json" ]]; then _mark_done "${gpu}" "${task}"; return 0; fi
  set +e
  CUDA_VISIBLE_DEVICES="${gpu}" python training/scope_round3/hmin_v2_dup_rollout.py \
    --output-dir "${out}" \
    --manifest "${MANIFEST}" \
    --shard shard0 --n-shards 1 \
    --model-path "${merged}" \
    --harness-config "${HARNESS_V2}" \
    --vllm-port "${port}" \
    --dup-operation --no-manage-vllm \
    2>&1 | tee "${logd}/stdout.log" || true
  # Full 100q: run all 8 shards sequentially on same GPU
  for s in shard0 shard1 shard2 shard3 shard4 shard5 shard6 shard7; do
    CUDA_VISIBLE_DEVICES="${gpu}" python training/scope_round3/hmin_v2_dup_rollout.py \
      --output-dir "${out}/${s}" \
      --manifest "${MANIFEST}" --shard "${s}" --n-shards 8 \
      --model-path "${merged}" \
      --harness-config "${HARNESS_V2}" \
      --vllm-port "${port}" \
      --dup-operation --parallel 1 \
      2>&1 | tee -a "${logd}/stdout.log"
  done
  python training/scope_round3/wave4_compare.py --root "${out}" 2>/dev/null || true
  if [[ -f "${out}/shard0/summary.json" ]]; then
    _mark_done "${gpu}" "${task}"
  else
    _mark_failed "${gpu}" "${task}"
  fi
}

gpu_pipeline() {
  local gpu="$1"
  case "${gpu}" in
    0)
      wave4_shard 0 base shard0 "${BASE_MODEL}" 8900
      bilateral_shard 0 shard0 8900
      if [[ "$(cat "${DATASET_DIR}/ROUND3_DATA_GO" 2>/dev/null)" == "true" ]]; then
        train_variant 0 round3_op_main_seed42 "--loss-mode operation_ce --class-balancing --route-balancing --compact-target --seed 42"
        merge_lora 0 round3_op_main_seed42
        offline_eval 0 round3_op_main_seed42
        closed_loop_100q 0 round3_op_main_seed42 8900
      fi
      ;;
    1)
      wave4_shard 1 base shard1 "${BASE_MODEL}" 8901
      bilateral_shard 1 shard1 8901
      if [[ "$(cat "${DATASET_DIR}/ROUND3_DATA_GO" 2>/dev/null)" == "true" ]]; then
        train_variant 1 round3_op_main_seed43 "--loss-mode operation_ce --class-balancing --route-balancing --compact-target --seed 43"
        merge_lora 1 round3_op_main_seed43
        offline_eval 1 round3_op_main_seed43
        closed_loop_100q 1 round3_op_main_seed43 8901
      fi
      ;;
    2)
      wave4_shard 2 round1 shard0 "${ROUND1_MODEL}" 8902
      bilateral_shard 2 shard2 8902
      if [[ "$(cat "${DATASET_DIR}/ROUND3_DATA_GO" 2>/dev/null)" == "true" ]]; then
        train_variant 2 round3_op_main_seed44 "--loss-mode operation_ce --class-balancing --route-balancing --compact-target --seed 44"
        merge_lora 2 round3_op_main_seed44
        offline_eval 2 round3_op_main_seed44
        closed_loop_100q 2 round3_op_main_seed44 8902
      fi
      ;;
    3)
      wave4_shard 3 round1 shard1 "${ROUND1_MODEL}" 8903
      bilateral_shard 3 shard3 8903
      if [[ "$(cat "${DATASET_DIR}/ROUND3_DATA_GO" 2>/dev/null)" == "true" ]]; then
        train_variant 3 round3_compact_json_sample_norm "--loss-mode sample_normalized_action_ce --compact-target --class-balancing"
        merge_lora 3 round3_compact_json_sample_norm
        offline_eval 3 round3_compact_json_sample_norm
        closed_loop_100q 3 round3_compact_json_sample_norm 8903
      fi
      ;;
    4)
      wave4_shard 4 round2_main shard0 "${BASE_MODEL}" 8904
      bilateral_shard 4 shard4 8904
      if [[ "$(cat "${DATASET_DIR}/ROUND3_DATA_GO" 2>/dev/null)" == "true" ]]; then
        train_variant 4 round3_legacy_full_action_token_ce "--loss-mode legacy_token_ce"
        merge_lora 4 round3_legacy_full_action_token_ce
        offline_eval 4 round3_legacy_full_action_token_ce
        closed_loop_100q 4 round3_legacy_full_action_token_ce 8904
      fi
      ;;
    5)
      wave4_shard 5 round2_main shard1 "${BASE_MODEL}" 8905
      bilateral_shard 5 shard5 8905
      if [[ "$(cat "${DATASET_DIR}/ROUND3_DATA_GO" 2>/dev/null)" == "true" ]]; then
        train_variant 5 round3_correct_only_op "--loss-mode operation_ce --route-filter CORRECT --compact-target"
        merge_lora 5 round3_correct_only_op
        offline_eval 5 round3_correct_only_op
        closed_loop_100q 5 round3_correct_only_op 8905
      fi
      ;;
    6)
      wave4_shard 6 round2_legacy shard0 "${BASE_MODEL}" 8906
      bilateral_shard 6 shard6 8906
      if [[ "$(cat "${DATASET_DIR}/ROUND3_DATA_GO" 2>/dev/null)" == "true" ]]; then
        train_variant 6 round3_endorse_only_op "--loss-mode operation_ce --route-filter ENDORSE --compact-target"
        merge_lora 6 round3_endorse_only_op
        offline_eval 6 round3_endorse_only_op
        closed_loop_100q 6 round3_endorse_only_op 8906
      fi
      ;;
    7)
      wave4_shard 7 round2_legacy shard1 "${BASE_MODEL}" 8907
      bilateral_shard 7 shard7 8907
      if [[ "$(cat "${DATASET_DIR}/ROUND3_DATA_GO" 2>/dev/null)" == "true" ]]; then
        train_variant 7 round3_op_no_balance "--loss-mode operation_ce --compact-target"
        merge_lora 7 round3_op_no_balance
        offline_eval 7 round3_op_no_balance
        closed_loop_100q 7 round3_op_no_balance 8907
      fi
      ;;
  esac
}

barrier_b() {
  python training/scope_round3/build_dataset.py \
    --shard-dirs \
      "${ROOT}/bilateral_labeling/shard0" \
      "${ROOT}/bilateral_labeling/shard1" \
      "${ROOT}/bilateral_labeling/shard2" \
      "${ROOT}/bilateral_labeling/shard3" \
      "${ROOT}/bilateral_labeling/shard4" \
      "${ROOT}/bilateral_labeling/shard5" \
      "${ROOT}/bilateral_labeling/shard6" \
      "${ROOT}/bilateral_labeling/shard7" \
    --output-dir "${DATASET_DIR}"
  python training/scope_round3/baselines.py \
    --valid "${DATASET_DIR}/valid.jsonl" \
    --train "${DATASET_DIR}/train.jsonl" \
    --model-path "${BASE_MODEL}" \
    --round2-adapter "${ROUND2_MAIN}" \
    --output "${ROOT}/eval/baselines.json" || true
}

wave4_merge() {
  python training/scope_round3/wave4_compare.py --root "${ROOT}/wave4_diagnostic"
}

PHASE="${PHASE:-all}"
case "${PHASE}" in
  barrier_a) barrier_a ;;
  wave4)
    for g in 0 1 2 3 4 5 6 7; do gpu_pipeline "${g}" & done
    wait
    wave4_merge
    ;;
  bilateral)
    for g in 0 1 2 3 4 5 6 7; do bilateral_shard "${g}" "shard${g}" $((8900+g)) & done
    wait
    barrier_b
    ;;
  train)
    for g in 0 1 2 3 4 5 6 7; do gpu_pipeline "${g}" & done
    wait
    ;;
  report) python training/scope_round3/final_report.py --root "${ROOT}" ;;
  all)
    barrier_a
    for g in 0 1 2 3 4 5 6 7; do gpu_pipeline "${g}" & done
    wait
    wave4_merge
    barrier_b
    for g in 0 1 2 3 4 5 6 7; do gpu_pipeline "${g}" & done
    wait
    python training/scope_round3/final_report.py --root "${ROOT}"
    ;;
  *) echo "Unknown PHASE=${PHASE}"; exit 1 ;;
esac
