#!/usr/bin/env bash
# Harness-1 upstream_api bcplus_full (830 queries) — six sequential runs:
#   gpt-oss-20b       × {zero, all}
#   harness-1         × {zero, all}
#   Qwen3-4B-Instruct × {zero, all}
#
# zero: single actor on GPU (default GPU=4), --tp shards share one vLLM.
# all:  4 GPUs — 3 actor vLLMs in parallel + 1 verifier (default actors 1,2,3 / verify 4).
#
# All repo/data paths are relative to TRIM root (parent of scripts/).
#   bash TRIM/scripts/run_bcplus_full_gpu4_eval.sh
#
# Optional env: GPU, TP, ALL_ACTOR_GPUS, ALL_VERIFY_GPU, ALL_ACTOR_PORTS, ALL_VERIFY_PORT, ALL_TP, ...
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRIM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${TRIM_ROOT}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_bcplus_full_gpu4}"
PY="${PY:-python}"
VLLM="${VLLM:-vllm}"

# --- relative paths (from TRIM root) ---
REL_MODELS="../../models"
REL_BCPLUS="../SCOPE/external/BrowseComp-Plus"
REL_OUT="outputs"
REL_LOGS="${REL_OUT}/logs/${RUN_ID}"
REL_SCAPE_EASYOPD="../SCAPE-EasyOPD"

BCPLUS_INDEX="${REL_BCPLUS}/indexes/bm25"
BCPLUS_CORPUS="${REL_BCPLUS}/data/browsecomp_plus_corpus_full.jsonl"
VERIFY_MODEL_PATH="${REL_MODELS}/harness-1"

MODEL_PATHS=(
  "${REL_MODELS}/gpt-oss-20b"
  "${REL_MODELS}/harness-1"
  "${REL_MODELS}/Qwen3-4B-Instruct-2507"
)
MODEL_SLUGS=(
  "gpt-oss-20b"
  "harness-1"
  "Qwen3-4B-Instruct-2507"
)
API_MODELS=(
  "gpt-oss-20b"
  "harness-1"
  "Qwen3-4B-Instruct-2507"
)
COMPONENTS=(zero all)

# zero: one GPU
GPU="${GPU:-4}"
TP="${TP:-3}"
ZERO_ACTOR_PORT="${ZERO_ACTOR_PORT:-8040}"

# all: 3 actor GPUs + 1 verifier GPU (4 cards total)
ALL_ACTOR_GPUS="${ALL_ACTOR_GPUS:-1,2,3}"
ALL_VERIFY_GPU="${ALL_VERIFY_GPU:-4}"
ALL_ACTOR_PORTS="${ALL_ACTOR_PORTS:-8040,8042,8044}"
ALL_VERIFY_PORT="${ALL_VERIFY_PORT:-8050}"
ALL_TP="${ALL_TP:-3}"

VERIFY_MODEL_NAME="${VERIFY_MODEL_NAME:-harness-1-verifier}"
GPU_UTIL="${GPU_UTIL:-0.75}"
WORKER_STAGGER="${WORKER_STAGGER:-30}"

export PYTHONPATH=".:${REL_SCAPE_EASYOPD}"
export TRIM_GPU_KEEPALIVE=0

mkdir -p "${REL_LOGS}"
ACTOR_PIDS=()
VERIFY_PID=""
MASTER_LOG="${REL_LOGS}/master.log"

log() {
  echo "[$(date -Is)] $*" | tee -a "${MASTER_LOG}"
}

stop_pid() {
  local pid="${1:-}"
  [[ -n "${pid}" ]] || return 0
  kill "${pid}" 2>/dev/null || true
  wait "${pid}" 2>/dev/null || true
}

stop_all_actors() {
  for pid in "${ACTOR_PIDS[@]:-}"; do
    stop_pid "${pid}"
  done
  ACTOR_PIDS=()
}

stop_verify() {
  stop_pid "${VERIFY_PID}"
  VERIFY_PID=""
}

cleanup() {
  stop_all_actors
  stop_verify
}
trap cleanup EXIT

wait_api() {
  local port="$1" timeout_s="${2:-900}"
  local deadline=$((SECONDS + timeout_s))
  while (( SECONDS < deadline )); do
    curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1 && return 0
    sleep 10
  done
  log "timeout waiting for API on port ${port}"
  return 1
}

wait_port_free() {
  local port="$1" timeout_s="${2:-120}"
  local deadline=$((SECONDS + timeout_s))
  while (( SECONDS < deadline )); do
    curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1 || return 0
    sleep 5
  done
  log "timeout waiting for port ${port} to go idle"
  return 1
}

join_urls() {
  local out=()
  for p in "$@"; do
    out+=("http://127.0.0.1:${p}/v1")
  done
  local IFS=,
  echo "${out[*]}"
}

csv_to_array() {
  local csv="$1"
  local -n _arr="$2"
  _arr=()
  local IFS=,
  read -ra _arr <<< "${csv}"
}

vllm_extra_for_model() {
  local slug="$1"
  case "${slug}" in
    gpt-oss-20b|harness-1)
      echo "--enable-auto-tool-choice --tool-call-parser openai --max-model-len 32768 --trust-remote-code --moe-backend triton"
      ;;
    Qwen3-4B-Instruct-2507)
      echo "--enable-auto-tool-choice --tool-call-parser openai --max-model-len 32768 --trust-remote-code"
      ;;
    *)
      echo "--enable-auto-tool-choice --tool-call-parser openai --max-model-len 32768"
      ;;
  esac
}

start_vllm_bg() {
  local gpu="$1" port="$2" model_path="$3" served_name="$4" extra="$5" log_path="$6"

  wait_port_free "${port}" 60 || true

  log "starting vLLM GPU${gpu} port ${port} model=${model_path} served=${served_name}"
  CUDA_VISIBLE_DEVICES="${gpu}" nohup "${VLLM}" serve "${model_path}" \
    --host 127.0.0.1 --port "${port}" \
    --served-model-name "${served_name}" \
    ${extra} \
    --gpu-memory-utilization "${GPU_UTIL}" --enforce-eager \
    > "${log_path}" 2>&1 &
  local pid=$!
  echo "${pid}" > "${log_path}.pid"
  wait_api "${port}"
  echo "${pid}"
}

start_zero_stack() {
  local model_path="$1" api_model="$2" model_slug="$3"
  stop_all_actors
  stop_verify

  local extra
  extra="$(vllm_extra_for_model "${model_slug}")"
  local pid
  pid="$(start_vllm_bg "${GPU}" "${ZERO_ACTOR_PORT}" "${model_path}" "${api_model}" \
    "${extra}" "${REL_LOGS}/actor_${model_slug}_zero.log")"
  ACTOR_PIDS=("${pid}")
}

start_all_stack() {
  local model_path="$1" api_model="$2" model_slug="$3"
  stop_all_actors
  stop_verify

  local actor_gpus=() actor_ports=()
  csv_to_array "${ALL_ACTOR_GPUS}" actor_gpus
  csv_to_array "${ALL_ACTOR_PORTS}" actor_ports

  if ((${#actor_gpus[@]} != ALL_TP)); then
    log "ALL_ACTOR_GPUS count (${#actor_gpus[@]}) must match ALL_TP (${ALL_TP})"
    exit 1
  fi
  if ((${#actor_ports[@]} != ALL_TP)); then
    log "ALL_ACTOR_PORTS count (${#actor_ports[@]}) must match ALL_TP (${ALL_TP})"
    exit 1
  fi

  local extra
  extra="$(vllm_extra_for_model "${model_slug}")"
  ACTOR_PIDS=()
  local i gpu port pid
  for i in "${!actor_gpus[@]}"; do
    gpu="${actor_gpus[$i]}"
    port="${actor_ports[$i]}"
    pid="$(start_vllm_bg "${gpu}" "${port}" "${model_path}" "${api_model}" \
      "${extra}" "${REL_LOGS}/actor_${model_slug}_all_gpu${gpu}.log")"
    ACTOR_PIDS+=("${pid}")
  done

  VERIFY_PID="$(start_vllm_bg "${ALL_VERIFY_GPU}" "${ALL_VERIFY_PORT}" "${VERIFY_MODEL_PATH}" \
    "${VERIFY_MODEL_NAME}" "--max-model-len 8192 --trust-remote-code --moe-backend triton" \
    "${REL_LOGS}/verify_${model_slug}.log")"
}

ensure_full_corpus() {
  if [[ -f "${BCPLUS_CORPUS}" ]]; then
    return 0
  fi
  log "building full corpus → ${BCPLUS_CORPUS}"
  "${PY}" scripts/build_browsecomp_corpus_from_index.py \
    --mode full \
    --out "${BCPLUS_CORPUS}" \
    >> "${REL_LOGS}/corpus_build.log" 2>&1
  local report="${BCPLUS_CORPUS%.jsonl}.BUILD_REPORT.json"
  [[ -f "${report}" ]] || { log "missing build report ${report}"; exit 1; }
  grep -q '"index_corpus_delta": 0' "${report}" || {
    log "corpus/index mismatch; see ${report}"
    exit 1
  }
}

run_eval() {
  local model_path="$1"
  local model_slug="$2"
  local api_model="$3"
  local component="$4"
  local actor_urls="$5"
  local tp="$6"
  local out_suffix="$7"
  local out="${REL_OUT}/eval_h1_bcplus_full_${model_slug}_${component}_${out_suffix}_${RUN_ID}"
  local log_path="${REL_LOGS}/eval_${model_slug}_${component}.log"
  local extra=()

  if [[ "${component}" == "all" ]]; then
    extra+=(--verify-base-url "http://127.0.0.1:${ALL_VERIFY_PORT}/v1")
    extra+=(--verify-model "${VERIFY_MODEL_NAME}")
  fi

  log "START model=${model_slug} component=${component} tp=${tp} urls=${actor_urls} out=${out}"
  "${PY}" scripts/run_eval.py \
    --harness Harness-1 \
    --benchmark bcplus_full \
    --model_name "${model_path}" \
    --evaluation-path upstream_api \
    --retrieval-backend local_bm25 \
    --api-base-url "${actor_urls}" \
    --api-model "${api_model}" \
    --index-path "${BCPLUS_INDEX}" \
    --corpus-path "${BCPLUS_CORPUS}" \
    --reranker none \
    --offline \
    --max-turns 40 \
    --max-new-tokens 2048 \
    --temperature 1.0 \
    --tp "${tp}" \
    --eval-stagger-s "${WORKER_STAGGER}" \
    --component "${component}" \
    --out "${out}" \
    "${extra[@]}" \
    2>&1 | tee "${log_path}"
  log "DONE model=${model_slug} component=${component}"
}

ensure_full_corpus

csv_to_array "${ALL_ACTOR_PORTS}" _ALL_PORTS_CHECK
ALL_ACTOR_URLS="$(join_urls "${_ALL_PORTS_CHECK[@]}")"
ZERO_ACTOR_URL="http://127.0.0.1:${ZERO_ACTOR_PORT}/v1"
ALL_VERIFY_URL="http://127.0.0.1:${ALL_VERIFY_PORT}/v1"

log "RUN_ID=${RUN_ID} zero_gpu=${GPU} zero_tp=${TP} all_actors=${ALL_ACTOR_GPUS} all_verify=${ALL_VERIFY_GPU} all_tp=${ALL_TP}"

for idx in "${!MODEL_PATHS[@]}"; do
  model_path="${MODEL_PATHS[$idx]}"
  model_slug="${MODEL_SLUGS[$idx]}"
  api_model="${API_MODELS[$idx]}"

  [[ -d "${model_path}" ]] || { log "missing model dir ${model_path}"; exit 1; }

  start_zero_stack "${model_path}" "${api_model}" "${model_slug}"
  run_eval "${model_path}" "${model_slug}" "${api_model}" "zero" \
    "${ZERO_ACTOR_URL}" "${TP}" "gpu${GPU}"

  start_all_stack "${model_path}" "${api_model}" "${model_slug}"
  run_eval "${model_path}" "${model_slug}" "${api_model}" "all" \
    "${ALL_ACTOR_URLS}" "${ALL_TP}" "gpu${ALL_ACTOR_GPUS//,/}v${ALL_VERIFY_GPU}"
done

stop_all_actors
stop_verify

GIT_COMMIT="$(git -C . rev-parse HEAD 2>/dev/null || echo unknown)"
cat > "${REL_LOGS}/MANIFEST.json" <<EOF
{
  "run_id": "${RUN_ID}",
  "benchmark": "bcplus_full",
  "evaluation_path": "upstream_api",
  "eval_profile": "upstream_core_local_bm25",
  "zero": {"gpu": "${GPU}", "tp": ${TP}, "actor_port": ${ZERO_ACTOR_PORT}},
  "all": {
    "actor_gpus": "${ALL_ACTOR_GPUS}",
    "actor_ports": "${ALL_ACTOR_PORTS}",
    "verify_gpu": "${ALL_VERIFY_GPU}",
    "verify_port": ${ALL_VERIFY_PORT},
    "tp": ${ALL_TP}
  },
  "experiments": [
    {"model": "${REL_MODELS}/gpt-oss-20b", "component": "zero", "layout": "1gpu"},
    {"model": "${REL_MODELS}/gpt-oss-20b", "component": "all", "layout": "3actor+1verify"},
    {"model": "${REL_MODELS}/harness-1", "component": "zero", "layout": "1gpu"},
    {"model": "${REL_MODELS}/harness-1", "component": "all", "layout": "3actor+1verify"},
    {"model": "${REL_MODELS}/Qwen3-4B-Instruct-2507", "component": "zero", "layout": "1gpu"},
    {"model": "${REL_MODELS}/Qwen3-4B-Instruct-2507", "component": "all", "layout": "3actor+1verify"}
  ],
  "zero_actor_url": "${ZERO_ACTOR_URL}",
  "all_actor_urls": "${ALL_ACTOR_URLS}",
  "all_verify_url": "${ALL_VERIFY_URL}",
  "corpus_path": "${BCPLUS_CORPUS}",
  "index_path": "${BCPLUS_INDEX}",
  "git_commit": "${GIT_COMMIT}"
}
EOF

log "finished six runs; outputs under ${REL_OUT}/eval_h1_bcplus_full_*_${RUN_ID}"
log "manifest ${REL_LOGS}/MANIFEST.json"
