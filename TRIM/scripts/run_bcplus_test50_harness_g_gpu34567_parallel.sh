#!/usr/bin/env bash
# Harness-G bcplus_test_50 — parallel four-group eval on GPU 3,4,5,6,7 (+0 when free).
#
# Strategy:
#   1. GPT zero/all on GPU pool 3,4,5,7 (TP=4) while an in-flight Qwen zero finishes on 0,6.
#   2. After GPT zero: run GPT-all (3,4,5) and Qwen-all (0,6,7) in parallel (TP=3 each).
#   3. Merge Qwen zero from a prior RUN_ID if WAIT_QWEN_ZERO_RUN is set.
#
# Usage:
#   bash TRIM/scripts/run_bcplus_test50_harness_g_gpu34567_parallel.sh
#
# Env:
#   RUN_ID, PY, GPU_UTIL, MAX_TURNS, WAIT_QWEN_ZERO_RUN (optional prior run id),
#   SKIP_GPT_ZERO=1 if already done
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRIM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${TRIM_ROOT}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_hg_test50_gpu34567_par}"
PY="${PY:-/data/ppnm/miniconda3/envs/bishop/bin/python}"
PKG="/data/ppnm/trim_bcplus_test50_eval_harness_g"
REL_OUT="outputs"
REL_LOGS="${REL_OUT}/logs/${RUN_ID}"
REL_SCAPE_EASYOPD="../SCAPE-EasyOPD"

GPU_UTIL="${GPU_UTIL:-0.75}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_TURNS="${MAX_TURNS:-40}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-1.0}"
REASONING_EFFORT="${REASONING_EFFORT:-low}"
SEARCH_K="${SEARCH_K:-10}"
EVAL_STAGGER="${EVAL_STAGGER:-8}"
DEFAULT_GRAPH_INDEX_PATH="${TRIM_ROOT}/../SCOPE/external/BrowseComp-Plus/indexes/harness_g_corpus_graph.pkl"
GRAPH_INDEX_PATH="${GRAPH_INDEX_PATH:-${DEFAULT_GRAPH_INDEX_PATH}}"
: "${GRAPH_INDEX_PATH:?ERROR: GRAPH_INDEX_PATH must be set for official Harness-G evaluation}"
if [[ ! -f "${GRAPH_INDEX_PATH}" && ! -d "${GRAPH_INDEX_PATH}" ]]; then
  echo "ERROR: graph index does not exist: ${GRAPH_INDEX_PATH}" >&2
  exit 2
fi

GPT_MODEL="/data/ppnm/models/gpt-oss-20b"
QWEN_MODEL="/data/ppnm/models/Qwen3-4B-Instruct-2507"
GPT_SLUG="gpt-oss-20b"
QWEN_SLUG="Qwen3-4B-Instruct-2507"

export PYTHONPATH=".:${REL_SCAPE_EASYOPD}${PYTHONPATH:+:${PYTHONPATH}}"
export TRIM_GPU_KEEPALIVE=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
unset CUDA_VISIBLE_DEVICES || true

mkdir -p "${REL_LOGS}" "${PKG}/results/${RUN_ID}" "${PKG}/logs/${RUN_ID}"
MASTER_LOG="${REL_LOGS}/master.log"

log() {
  echo "[$(date -Is)] $*" | tee -a "${MASTER_LOG}" >&2
}

run_one() {
  local model_path="$1" slug="$2" component="$3" gpus="$4" tp="$5" tag="$6"
  local gpu_tag="gpu${gpus//,/}"
  local out="${REL_OUT}/eval_hg_bcplus_test50_${slug}_${component}_${gpu_tag}_${RUN_ID}"
  local pkg_out="${PKG}/results/${RUN_ID}/${slug}/${component}"
  local log_path="${REL_LOGS}/eval_${tag}.log"

  log "START tag=${tag} model=${slug} component=${component} tp=${tp} gpus=${gpus} out=${out}"
  mkdir -p "${pkg_out}"
  local rc=0
  "${PY}" scripts/run_eval.py \
    --harness Harness-G \
    --benchmark bcplus_test_50 \
    --model_name "${model_path}" \
    --evaluation-path legacy_local \
    --eval-mode harness \
    --component "${component}" \
    --tp "${tp}" \
    --eval-gpus "${gpus}" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-turns "${MAX_TURNS}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --reasoning-effort "${REASONING_EFFORT}" \
    --search-k "${SEARCH_K}" \
    --graph-index-path "${GRAPH_INDEX_PATH}" \
    --eval-stagger-s "${EVAL_STAGGER}" \
    --rollout-backend vllm \
    --out "${out}" \
    2>&1 | tee -a "${log_path}" || rc=$?
  if [[ -d "${out}" ]]; then
    rsync -a "${out}/" "${pkg_out}/"
  fi
  cp -a "${log_path}" "${PKG}/logs/${RUN_ID}/eval_${tag}.log" 2>/dev/null || true
  if [[ "${rc}" -ne 0 ]]; then
    log "FAIL tag=${tag} rc=${rc}"
    return "${rc}"
  fi
  log "DONE tag=${tag}"
}

sync_qwen_zero_prior() {
  local prior="${WAIT_QWEN_ZERO_RUN:-}"
  [[ -n "${prior}" ]] || return 0
  local src="${PKG}/results/${prior}/${QWEN_SLUG}/zero"
  local dst="${PKG}/results/${RUN_ID}/${QWEN_SLUG}/zero"
  local qwen_out="${REL_OUT}/eval_hg_bcplus_test50_${QWEN_SLUG}_zero_gpu04567_${prior}"
  mkdir -p "${dst}"
  if [[ -d "${qwen_out}" ]]; then
    rsync -a "${qwen_out}/" "${dst}/"
  elif [[ -d "${src}" ]]; then
    rsync -a "${src}/" "${dst}/"
  else
    log "WARN prior Qwen zero output missing at ${qwen_out}"
    return 1
  fi
  return 0
}

wait_qwen_zero_prior() {
  local prior="${WAIT_QWEN_ZERO_RUN:-}"
  [[ -n "${prior}" ]] || return 0
  local qwen_out="${REL_OUT}/eval_hg_bcplus_test50_${QWEN_SLUG}_zero_gpu04567_${prior}"
  if [[ -f "${qwen_out}/FOUR_CELL_OFFICIAL_SUMMARY.json" ]]; then
    log "prior Qwen zero already complete; syncing results"
    sync_qwen_zero_prior
    return 0
  fi
  log "waiting for prior Qwen zero run_id=${prior}"
  while pgrep -f "eval_hg_bcplus_test50_${QWEN_SLUG}_zero_gpu04567_${prior}" >/dev/null 2>&1; do
    sleep 30
  done
  log "prior Qwen zero finished; syncing results"
  sync_qwen_zero_prior
}

launch_phase2_parallel() {
  log "PHASE2 launching GPT-all (gpus=3,4,5 tp=3) and Qwen-all (gpus=0,6,7 tp=3) in parallel"
  local rc_a=0 rc_b=0
  run_one "${GPT_MODEL}" "${GPT_SLUG}" all "3,4,5" 3 "gpt_all" &
  local pid_gpt=$!
  run_one "${QWEN_MODEL}" "${QWEN_SLUG}" all "0,6,7" 3 "qwen_all" &
  local pid_qwen=$!
  log "PHASE2 pids gpt_all=${pid_gpt} qwen_all=${pid_qwen}"
  wait "${pid_gpt}" || rc_a=$?
  wait "${pid_qwen}" || rc_b=$?
  local nfail=0
  [[ "${rc_a}" -eq 0 ]] || nfail=$((nfail + 1))
  [[ "${rc_b}" -eq 0 ]] || nfail=$((nfail + 1))
  return "${nfail}"
}

capture_git_launch_record() {
  GIT_COMMIT="$(git -C "${TRIM_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
  GIT_DIRTY="$(git -C "${TRIM_ROOT}" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
  {
    echo "run_id=${RUN_ID}"
    echo "started_at=$(date -Is)"
    echo "git_commit=${GIT_COMMIT}"
    echo "git_dirty_files=${GIT_DIRTY}"
    echo "parallel_profile=gpu34567_tp4_then_dual_tp3"
    echo "reasoning_effort=${REASONING_EFFORT}"
    echo "graph_index_path=${GRAPH_INDEX_PATH:-}"
  } > "${PKG}/results/${RUN_ID}/LAUNCH_RECORD.txt"
  cp "${PKG}/results/${RUN_ID}/LAUNCH_RECORD.txt" "${REL_LOGS}/LAUNCH_RECORD.txt"
  log "LAUNCH_RECORD git_commit=${GIT_COMMIT} dirty=${GIT_DIRTY}"
}

main() {
  [[ -x "${PY}" ]] || { log "missing python ${PY}"; exit 1; }
  capture_git_launch_record
  failed=0

  # Phase 1: GPT zero on idle GPUs (avoid 0,6 while prior Qwen zero may still use them).
  if [[ "${SKIP_GPT_ZERO:-0}" != "1" ]]; then
    run_one "${GPT_MODEL}" "${GPT_SLUG}" zero "3,4,5,7" 4 "gpt_zero" || failed=$((failed + 1))
  else
    log "SKIP_GPT_ZERO=1"
  fi

  # Wait for in-flight Qwen zero from a previous sequential launcher, then sync.
  wait_qwen_zero_prior

  # Phase 2: GPT-all and Qwen-all in parallel on disjoint GPU triples.
  local phase2_fail=0
  launch_phase2_parallel || phase2_fail=$?
  failed=$((failed + phase2_fail))

  cat > "${PKG}/results/${RUN_ID}/MANIFEST.json" <<EOF
{
  "run_id": "${RUN_ID}",
  "harness": "Harness-G",
  "benchmark": "bcplus_test_50",
  "parallel_profile": "gpu34567_tp4_then_dual_tp3",
  "wait_qwen_zero_run": "${WAIT_QWEN_ZERO_RUN:-}",
  "git_commit": "${GIT_COMMIT:-unknown}",
  "git_dirty_files": ${GIT_DIRTY:-0},
  "failed_runs": ${failed}
}
EOF
  log "finished failed=${failed} results=${PKG}/results/${RUN_ID}/"
  [[ "${failed}" -eq 0 ]]
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
