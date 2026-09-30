#!/usr/bin/env bash
# Harness-G bcplus_test_100 — sequential four-run eval on leftover GPU 0,3,6:
#   Qwen3-4B-Instruct-2507   × {zero, all}
#   gpt-oss-20b              × {zero, all}
#
# 100 queries are a frozen random subset of the official BC+ 166-test split
# (seed=20260921). --tp is data-parallel replica count: three independent
# vLLM workers, one GPU each, then merge.
#
# Usage:
#   bash TRIM/scripts/run_bcplus_test100_harness_g_gpu036_eval.sh
#
# Optional env:
#   RUN_ID, MODELS, COMPONENTS, SKIP_ZERO=1, TP, EVAL_GPUS,
#   GPU_UTIL, MAX_NUM_SEQS, MAX_MODEL_LEN, MAX_TURNS, TEMPERATURE,
#   EVAL_STAGGER, PY, BETWEEN_S, QUERY_IDS_FILE,
#   EVAL_CHUNK_SIZE, ENV_WORKERS, DOC_STORE_WORKERS,
#   FAIL_FAST=1 (default: stop after the first failed cell),
#   RESUME=1 (keep an existing RUN_ID directory and skip finished queries)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRIM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${TRIM_ROOT}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_hg_test100_gpu036}"
PY="${PY:-/data/ppnm/miniconda3/envs/bishop/bin/python}"

REL_OUT="outputs"
REL_LOGS="${REL_OUT}/logs/${RUN_ID}"
REL_SCAPE_EASYOPD="../SCAPE-EasyOPD"
PKG="/data/ppnm/trim_bcplus_test100_eval_harness_g"
QUERY_IDS_FILE="${QUERY_IDS_FILE:-${TRIM_ROOT}/manifests/browsecomp_plus_830/bcplus_test_100_random_20260921.ids}"

EVAL_GPUS="${EVAL_GPUS:-0,3,6}"
TP="${TP:-3}"
GPU_UTIL="${GPU_UTIL:-0.42}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_TURNS="${MAX_TURNS:-40}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-1.0}"
REASONING_EFFORT="${REASONING_EFFORT:-low}"
SEARCH_K="${SEARCH_K:-10}"
EVAL_STAGGER="${EVAL_STAGGER:-8}"
EVAL_CHUNK_SIZE="${EVAL_CHUNK_SIZE:-}"
ENV_WORKERS="${ENV_WORKERS:-}"
DOC_STORE_WORKERS="${DOC_STORE_WORKERS:-}"
FAIL_FAST="${FAIL_FAST:-1}"
RESUME="${RESUME:-0}"
DEFAULT_GRAPH_INDEX_PATH="${TRIM_ROOT}/../SCOPE/external/BrowseComp-Plus/indexes/harness_g_corpus_graph.pkl"
GRAPH_INDEX_PATH="${GRAPH_INDEX_PATH:-${DEFAULT_GRAPH_INDEX_PATH}}"
: "${GRAPH_INDEX_PATH:?ERROR: GRAPH_INDEX_PATH must be set for official Harness-G evaluation}"
if [[ ! -f "${GRAPH_INDEX_PATH}" && ! -d "${GRAPH_INDEX_PATH}" ]]; then
  echo "ERROR: graph index does not exist: ${GRAPH_INDEX_PATH}" >&2
  exit 2
fi
BETWEEN_S="${BETWEEN_S:-20}"

MODEL_PATHS=(
  "/data/ppnm/models/Qwen3-4B-Instruct-2507"
  "/data/ppnm/models/gpt-oss-20b"
)
MODEL_SLUGS=(
  "Qwen3-4B-Instruct-2507"
  "gpt-oss-20b"
)
COMPONENTS_DEFAULT=(zero all)

export PYTHONPATH=".:${REL_SCAPE_EASYOPD}${PYTHONPATH:+:${PYTHONPATH}}"
export TRIM_GPU_KEEPALIVE=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Parent must see all physical ids in --eval-gpus; children pin one GPU each.
unset CUDA_VISIBLE_DEVICES || true

mkdir -p "${REL_LOGS}" "${PKG}/results/${RUN_ID}" "${PKG}/logs/${RUN_ID}"
MASTER_LOG="${REL_LOGS}/master.log"

log() {
  echo "[$(date -Is)] $*" >> "${MASTER_LOG}"
  echo "[$(date -Is)] $*" >&2
}

csv_to_array() {
  local csv="$1"
  local -n _arr="$2"
  _arr=()
  local IFS=,
  read -ra _arr <<< "${csv}"
}

model_selected() {
  local slug="$1"
  [[ -z "${MODELS:-}" ]] && return 0
  local IFS=,
  local wanted
  for wanted in ${MODELS}; do
    [[ "${wanted}" == "${slug}" ]] && return 0
  done
  return 1
}

component_selected() {
  local c="$1"
  [[ -z "${COMPONENTS:-}" ]] && return 0
  local IFS=,
  local wanted
  for wanted in ${COMPONENTS}; do
    [[ "${wanted}" == "${c}" ]] && return 0
  done
  return 1
}

log_gpu() {
  local msg
  msg="$(nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable")"
  apps="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader 2>/dev/null || true)"
  log "gpu_status ${msg//$'\n'/ | }"
  if [[ -n "${apps}" ]]; then
    log "gpu_procs ${apps//$'\n'/ | }"
  fi
}

wait_between_runs() {
  local s="${1:-${BETWEEN_S}}"
  log "cooling ${s}s before next eval"
  sleep "${s}"
  log_gpu
}

load_query_ids() {
  [[ -f "${QUERY_IDS_FILE}" ]] || {
    log "missing QUERY_IDS_FILE ${QUERY_IDS_FILE}"
    exit 2
  }
  QUERY_IDS="$(awk 'NF{printf("%s%s", (n++?",":""), $1)} END{print ""}' "${QUERY_IDS_FILE}")"
  local n
  n="$(awk 'NF{c++} END{print c+0}' "${QUERY_IDS_FILE}")"
  (( n >= 1 )) || {
    log "QUERY_IDS_FILE has no ids: ${QUERY_IDS_FILE}"
    exit 2
  }
  N_QUERIES="${n}"
}

run_eval() {
  local model_path="$1" model_slug="$2" component="$3"
  local gpu_tag="gpu${EVAL_GPUS//,/}"
  local out="${REL_OUT}/eval_hg_bcplus_test100_${model_slug}_${component}_${gpu_tag}_${RUN_ID}"
  local pkg_out="${PKG}/results/${RUN_ID}/${model_slug}/${component}"
  local log_path="${REL_LOGS}/eval_${model_slug}_${component}.log"

  log "START harness=Harness-G model=${model_slug} component=${component} tp=${TP} gpus=${EVAL_GPUS} n_queries=${N_QUERIES} out=${out} resume=${RESUME} fail_fast=${FAIL_FAST}"
  if [[ -e "${out}" || -e "${pkg_out}" ]]; then
    if [[ "${RESUME}" != "1" ]]; then
      log "refuse to overwrite ${out}; set RESUME=1 or choose a new RUN_ID"
      return 2
    fi
    log "RESUME=1 keeping existing outputs under ${out}"
  fi
  mkdir -p "${pkg_out}"

  local -a extra_args=()
  if [[ -n "${EVAL_CHUNK_SIZE}" ]]; then
    extra_args+=(--eval-chunk-size "${EVAL_CHUNK_SIZE}")
  fi
  if [[ -n "${ENV_WORKERS}" ]]; then
    extra_args+=(--env-workers "${ENV_WORKERS}")
  fi
  if [[ -n "${DOC_STORE_WORKERS}" ]]; then
    extra_args+=(--doc-store-workers "${DOC_STORE_WORKERS}")
  fi

  local rc=0
  "${PY}" scripts/run_eval.py \
    --harness Harness-G \
    --benchmark bcplus_test_166 \
    --query-ids "${QUERY_IDS}" \
    --model_name "${model_path}" \
    --evaluation-path legacy_local \
    --eval-mode harness \
    --component "${component}" \
    --tp "${TP}" \
    --eval-gpus "${EVAL_GPUS}" \
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
    "${extra_args[@]}" \
    2>&1 | tee -a "${log_path}" >&2 || rc=$?

  if [[ -d "${out}" ]]; then
    rsync -a "${out}/" "${pkg_out}/"
  fi
  cp -a "${log_path}" "${PKG}/logs/${RUN_ID}/eval_${model_slug}_${component}.log" 2>/dev/null || true

  if [[ "${rc}" -ne 0 ]]; then
    log "WARN eval exited ${rc} model=${model_slug} component=${component}"
    return "${rc}"
  fi
  log "DONE model=${model_slug} component=${component}"
}

capture_git_launch_record() {
  GIT_COMMIT="$(git -C "${TRIM_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
  GIT_DIRTY="$(git -C "${TRIM_ROOT}" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
  GIT_DIRTY_FILES="$(git -C "${TRIM_ROOT}" status --porcelain 2>/dev/null || true)"
  {
    echo "run_id=${RUN_ID}"
    echo "started_at=$(date -Is)"
    echo "git_commit=${GIT_COMMIT}"
    echo "git_dirty_files=${GIT_DIRTY}"
    echo "eval_gpus=${EVAL_GPUS}"
    echo "eval_replicas=${TP}"
    echo "reasoning_effort=${REASONING_EFFORT}"
    echo "graph_index_path=${GRAPH_INDEX_PATH:-}"
    echo "max_turns=${MAX_TURNS}"
    echo "max_new_tokens=${MAX_NEW_TOKENS}"
    echo "temperature=${TEMPERATURE}"
    echo "gpu_util=${GPU_UTIL}"
    echo "benchmark=bcplus_test_166"
    echo "n_queries=100"
    echo "query_ids_file=${QUERY_IDS_FILE}"
    echo "query_ids_seed=20260921"
    echo "note=post_20260918_root_cause_fix_test100_random_from_official_166"
    if [[ -n "${GIT_DIRTY_FILES}" ]]; then
      echo "--- dirty files ---"
      echo "${GIT_DIRTY_FILES}"
    fi
  } > "${PKG}/results/${RUN_ID}/LAUNCH_RECORD.txt"
  cp "${PKG}/results/${RUN_ID}/LAUNCH_RECORD.txt" "${REL_LOGS}/LAUNCH_RECORD.txt"
  cp -a "${QUERY_IDS_FILE}" "${PKG}/results/${RUN_ID}/query_ids.txt"
  log "LAUNCH_RECORD git_commit=${GIT_COMMIT} git_dirty_files=${GIT_DIRTY}"
}

main() {
  csv_to_array "${EVAL_GPUS}" _gpus
  ((${#_gpus[@]} == TP)) || {
    log "EVAL_GPUS count (${#_gpus[@]}) must equal TP (${TP})"
    exit 1
  }
  [[ -x "${PY}" ]] || { log "missing python ${PY}"; exit 1; }
  load_query_ids
  capture_git_launch_record

  local comps=()
  if [[ -n "${COMPONENTS:-}" ]]; then
    csv_to_array "${COMPONENTS}" comps
  else
    comps=("${COMPONENTS_DEFAULT[@]}")
  fi

  log "RUN_ID=${RUN_ID} harness=Harness-G benchmark=bcplus_test_166 n_queries=${N_QUERIES} eval_gpus=${EVAL_GPUS} tp=${TP} gpu_util=${GPU_UTIL} max_num_seqs=${MAX_NUM_SEQS} max_model_len=${MAX_MODEL_LEN} max_turns=${MAX_TURNS} temperature=${TEMPERATURE} reasoning_effort=${REASONING_EFFORT} graph_index_path=${GRAPH_INDEX_PATH:-none} models=${MODELS:-all} components=${comps[*]} chunk=${EVAL_CHUNK_SIZE:-all} env_workers=${ENV_WORKERS:-coupled} fail_fast=${FAIL_FAST} resume=${RESUME} query_ids_file=${QUERY_IDS_FILE}"
  log_gpu

  local failed=0
  local stop=0
  local idx slug path comp first=1
  local -a ran_models=()
  local -a ran_pairs=()
  for idx in "${!MODEL_PATHS[@]}"; do
    path="${MODEL_PATHS[$idx]}"
    slug="${MODEL_SLUGS[$idx]}"
    model_selected "${slug}" || { log "skip model ${slug} (not in MODELS)"; continue; }
    [[ -d "${path}" ]] || { log "missing model dir ${path}"; exit 1; }
    ran_models+=("${slug}")

    for comp in "${comps[@]}"; do
      if [[ "${comp}" == "zero" && "${SKIP_ZERO:-0}" == "1" ]]; then
        log "SKIP_ZERO=1 — skipping ${slug} zero"
        continue
      fi
      component_selected "${comp}" || { log "skip component ${comp}"; continue; }
      if [[ "${first}" != "1" ]]; then
        wait_between_runs
      fi
      first=0
      ran_pairs+=("${slug}/${comp}")
      if ! run_eval "${path}" "${slug}" "${comp}"; then
        failed=$((failed + 1))
        if [[ "${FAIL_FAST}" == "1" ]]; then
          log "FAIL_FAST=1; not starting the next cell"
          stop=1
          break
        fi
      fi
    done
    if [[ "${stop}" == "1" ]]; then
      break
    fi
  done

  local models_json="["
  local comps_json="["
  local pairs_json="["
  local i
  for i in "${!ran_models[@]}"; do
    [[ "${i}" != "0" ]] && models_json+=", "
    models_json+="\"${ran_models[$i]}\""
  done
  models_json+="]"
  for i in "${!comps[@]}"; do
    [[ "${i}" != "0" ]] && comps_json+=", "
    comps_json+="\"${comps[$i]}\""
  done
  comps_json+="]"
  for i in "${!ran_pairs[@]}"; do
    [[ "${i}" != "0" ]] && pairs_json+=", "
    pairs_json+="\"${ran_pairs[$i]}\""
  done
  pairs_json+="]"

  cat > "${PKG}/results/${RUN_ID}/MANIFEST.json" <<EOF
{
  "run_id": "${RUN_ID}",
  "harness": "Harness-G",
  "benchmark": "bcplus_test_166",
  "n_queries": ${N_QUERIES},
  "query_ids_file": "${QUERY_IDS_FILE}",
  "query_ids_seed": 20260921,
  "evaluation_path": "legacy_local",
  "eval_profile": "harness_g_graph_runtime",
  "eval_gpus": "${EVAL_GPUS}",
  "eval_replicas": ${TP},
  "tensor_parallel_size": 1,
  "gpu_memory_utilization": ${GPU_UTIL},
  "max_num_seqs": ${MAX_NUM_SEQS},
  "max_model_len": ${MAX_MODEL_LEN},
  "max_turns": ${MAX_TURNS},
  "max_new_tokens": ${MAX_NEW_TOKENS},
  "temperature": ${TEMPERATURE},
  "reasoning_effort": "${REASONING_EFFORT}",
  "graph_index_path": "${GRAPH_INDEX_PATH:-}",
  "search_k": ${SEARCH_K},
  "eval_stagger_s": ${EVAL_STAGGER},
  "eval_chunk_size": "${EVAL_CHUNK_SIZE:-}",
  "env_workers": "${ENV_WORKERS:-}",
  "doc_store_workers": "${DOC_STORE_WORKERS:-}",
  "fail_fast": "${FAIL_FAST}",
  "resume": "${RESUME}",
  "models_filter": "${MODELS:-}",
  "models": ${models_json},
  "components": ${comps_json},
  "experiments": ${pairs_json},
  "git_commit": "${GIT_COMMIT}",
  "git_dirty_files": ${GIT_DIRTY:-0},
  "failed_runs": ${failed}
}
EOF
  cp "${PKG}/results/${RUN_ID}/MANIFEST.json" "${REL_LOGS}/MANIFEST.json"
  log "finished; failed=${failed}; results ${PKG}/results/${RUN_ID}/ manifest ${REL_LOGS}/MANIFEST.json"
  [[ "${failed}" -eq 0 ]]
}

stop_own_children() {
  # Direct children of this script only. Do not signal other users' GPU jobs.
  pkill -TERM -P "$$" 2>/dev/null || true
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  trap stop_own_children INT TERM
  main "$@"
fi
