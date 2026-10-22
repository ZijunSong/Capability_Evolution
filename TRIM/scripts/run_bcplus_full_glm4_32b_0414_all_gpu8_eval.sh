#!/usr/bin/env bash
# Harness-1 upstream_api bcplus_full (830 queries) — GLM-4-32B-0414 ALL (component=all),
# tuned for 8 × gm402-80g. Derived from run_bcplus_full_glm4_32b_0414_zero_gpu8_eval.sh.
#
# Why this layout (uses all 8 cards, no TP overhead):
#   * 7 single-GPU vLLM actors (TP=1, GPU 0..6) + 1 harness-1 verifier (GPU 7).
#   * GLM-4-32B-0414 is dense (~32.6B bf16 ~= 65GB/GPU); GQA 2 KV heads => tiny KV;
#     GPU_UTIL=0.90 (pure inference, no backward).
#   * No Chroma server. Retrieval is local_bm25 (Pyserini/Lucene) --reranker none --offline.
#   * GLM-4-32B-0414 chat_template has NO thinking markers -> non-thinking by default.
#   * tool-call-parser=glm45 (model_profiles routing). The vLLM glm45/glm47 parsers
#     cannot extract structured tool_calls from this model (it emits name\n{json} in
#     content); the api_adapter.extract_tool_calls_from_content() fallback recovers
#     them, so the eval proceeds. NO smoke_test_tool_call gate (would fail since
#     tool_calls is empty from vllm); the content fallback is the working path.
#
# Optional env (defaults shown):
#   ALL_ACTOR_GPUS=0,1,2,3,4,5,6  ALL_ACTOR_PORTS=8042,8044,...,8056
#   ALL_VERIFY_GPU=7  ALL_VERIFY_PORT=8058  ALL_TP=7  ALL_EVAL_TP=64
#   MAX_NUM_SEQS=8  MAX_MODEL_LEN=32768  GPU_UTIL=0.90  WORKER_STAGGER=5
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRIM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${TRIM_ROOT}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_bcplus_full_glm4_32b_0414_all_gpu8}"
PY="${PY:-python}"
VLLM="${VLLM:-vllm}"

# --- relative paths (from TRIM root) ---
REL_MODELS="../../models"
REL_BCPLUS="../SCOPE/external/BrowseComp-Plus"
REL_OUT="outputs"
REL_LOGS="${REL_OUT}/logs/${RUN_ID}"
REL_SCAPE_EASYOPD="../SCAPE-EasyOPD"

BCPLUS_INDEX="${BCPLUS_INDEX:-${REL_BCPLUS}/indexes/bm25}"
BCPLUS_CORPUS="${BCPLUS_CORPUS:-${REL_BCPLUS}/data/browsecomp_plus_corpus_full.jsonl}"
VERIFY_MODEL_PATH="${REL_MODELS}/harness-1"

# GLM-4-32B-0414 all. MODEL_PATH may be overridden by the CloudML boot script.
MODEL_PATH="${MODEL_PATH:-${REL_MODELS}/GLM-4-32B-0414}"
MODEL_SLUG="glm-4-32b-0414"
API_MODEL="glm-4-32b-0414"
COMPONENT="all"

# 7 single-GPU actors (TP=1) + 1 verifier on GPU 7.
ALL_ACTOR_GPUS="${ALL_ACTOR_GPUS:-0,1,2,3,4,5,6}"
ALL_ACTOR_PORTS="${ALL_ACTOR_PORTS:-8042,8044,8046,8048,8052,8054,8056}"
ALL_VERIFY_GPU="${ALL_VERIFY_GPU:-7}"
ALL_VERIFY_PORT="${ALL_VERIFY_PORT:-8058}"
ALL_TP="${ALL_TP:-7}"                 # 7 API-client workers (1 per actor GPU)
ALL_EVAL_TP="${ALL_EVAL_TP:-64}"      # eval shard/worker count
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
GPU_UTIL="${GPU_UTIL:-0.90}"
WORKER_STAGGER="${WORKER_STAGGER:-5}"
VERIFY_MODEL_NAME="${VERIFY_MODEL_NAME:-harness-1-verifier}"

export PYTHONPATH=".:${REL_SCAPE_EASYOPD}"
export TRIM_GPU_KEEPALIVE=0
export HARNESS1_FORBID_CHROMA=1

mkdir -p "${REL_LOGS}"
ACTOR_PIDS=()
VERIFY_PID=""
MASTER_LOG="${REL_LOGS}/master.log"

log() {
  echo "[$(date -Is)] $*" >> "${MASTER_LOG}"
  echo "[$(date -Is)] $*" >&2
}

normalize_pid() {
  local raw="${1:-}"
  raw="${raw//$'\r'/}"
  while [[ "${raw}" == *$'\n' ]]; do raw="${raw%$'\n'}"; done
  if [[ "${raw}" =~ ^([0-9]+)$ ]]; then echo "${BASH_REMATCH[1]}"; return 0; fi
  local last="${raw##*$'\n'}"
  [[ "${last}" =~ ^([0-9]+)$ ]] && { echo "${BASH_REMATCH[1]}"; return 0; }
  return 1
}

stop_pid() {
  local pid="${1:-}"
  pid="$(normalize_pid "${pid}" 2>/dev/null || true)"
  [[ -n "${pid}" ]] || return 0
  kill "${pid}" 2>/dev/null || true
  wait "${pid}" 2>/dev/null || true
}

pids_on_port() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then lsof -ti ":${port}" 2>/dev/null || true; return 0; fi
  if command -v ss >/dev/null 2>&1; then
    ss -lptn "sport = :${port}" 2>/dev/null | sed -n 's/.*pid=\([0-9]*\).*/\1/p' || true
    return 0
  fi
  return 0
}

stop_port() {
  local port="$1" pid_file="${2:-}" pid pids
  if [[ -n "${pid_file}" && -f "${pid_file}" ]]; then
    pid="$(normalize_pid "$(cat "${pid_file}")" 2>/dev/null || true)"
    stop_pid "${pid}"
    rm -f "${pid_file}"
  fi
  pids="$(pids_on_port "${port}")"
  if [[ -n "${pids}" ]]; then
    log "stopping stale listener(s) on port ${port}: ${pids//$'\n'/ }"
    # shellcheck disable=SC2086
    kill ${pids} 2>/dev/null || true
    sleep 2
    # shellcheck disable=SC2086
    kill -9 ${pids} 2>/dev/null || true
  fi
}

stop_all_actors() { for pid in "${ACTOR_PIDS[@]:-}"; do stop_pid "${pid}"; done; ACTOR_PIDS=(); }
stop_verify() { stop_pid "${VERIFY_PID}"; VERIFY_PID=""; }
cleanup() { stop_all_actors; stop_verify; }
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

verify_api_model() {
  local port="$1" expected="$2" body ids
  body="$(curl -sf "http://127.0.0.1:${port}/v1/models")" || {
    log "failed to read /v1/models on port ${port}"; return 1
  }
  ids="$("${PY}" -c "import json,sys; d=json.load(sys.stdin); print(','.join(m.get('id','') for m in d.get('data',[])))" <<< "${body}")"
  if [[ ",${ids}," != *",${expected},"* ]]; then
    log "port ${port} served-model mismatch: expected ${expected}, got [${ids}]"; return 1
  fi
  return 0
}

join_urls() {
  local out=() p
  for p in "$@"; do out+=("http://127.0.0.1:${p}/v1"); done
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
  # GLM-4-32B-0414: dense, model_type=glm4. vLLM glm45/glm47 cannot extract
  # structured tool_calls (model emits name\n{json} in content); the
  # api_adapter.extract_tool_calls_from_content() fallback recovers them.
  echo "--enable-auto-tool-choice --tool-call-parser glm45 --max-model-len ${MAX_MODEL_LEN} --trust-remote-code --max-num-seqs ${MAX_NUM_SEQS}"
}

start_vllm_bg() {
  local gpu="$1" port="$2" model_path="$3" served_name="$4" extra="$5" log_path="$6"
  stop_port "${port}" "${log_path}.pid"
  wait_port_free "${port}" 120
  log "starting vLLM GPU${gpu} port ${port} model=${model_path} served=${served_name}"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="${gpu}" nohup "${VLLM}" serve "${model_path}" \
    --host 127.0.0.1 --port "${port}" --served-model-name "${served_name}" \
    ${extra} --gpu-memory-utilization "${GPU_UTIL}" --enforce-eager \
    > "${log_path}" 2>&1 &
  local pid=$!
  echo "${pid}" > "${log_path}.pid"
  wait_api "${port}"
  verify_api_model "${port}" "${served_name}"
  echo "${pid}"
}

start_all_stack() {
  stop_all_actors; stop_verify
  local actor_gpus=() actor_ports=()
  csv_to_array "${ALL_ACTOR_GPUS}" actor_gpus
  csv_to_array "${ALL_ACTOR_PORTS}" actor_ports
  ((${#actor_gpus[@]} == ALL_TP)) || { log "ALL_ACTOR_GPUS count != ALL_TP"; exit 1; }
  ((${#actor_ports[@]} == ALL_TP)) || { log "ALL_ACTOR_PORTS count != ALL_TP"; exit 1; }
  local extra pid i gpu port
  extra="$(vllm_extra_for_model "${MODEL_SLUG}")"
  ACTOR_PIDS=()
  for i in "${!actor_gpus[@]}"; do
    gpu="${actor_gpus[$i]}"; port="${actor_ports[$i]}"
    pid="$(start_vllm_bg "${gpu}" "${port}" "${MODEL_PATH}" "${API_MODEL}" \
      "${extra}" "${REL_LOGS}/actor_${MODEL_SLUG}_all_gpu${gpu}.log")"
    ACTOR_PIDS+=("$(normalize_pid "${pid}")")
  done
  VERIFY_PID="$(start_vllm_bg "${ALL_VERIFY_GPU}" "${ALL_VERIFY_PORT}" "${VERIFY_MODEL_PATH}" \
    "${VERIFY_MODEL_NAME}" "--max-model-len 8192 --trust-remote-code --moe-backend triton" \
    "${REL_LOGS}/verify_${MODEL_SLUG}.log")"
  VERIFY_PID="$(normalize_pid "${VERIFY_PID}")"
}

ensure_full_corpus() {
  [[ -f "${BCPLUS_CORPUS}" ]] && return 0
  log "building full corpus -> ${BCPLUS_CORPUS}"
  "${PY}" scripts/build_browsecomp_corpus_from_index.py --mode full --out "${BCPLUS_CORPUS}" \
    >> "${REL_LOGS}/corpus_build.log" 2>&1
  local report="${BCPLUS_CORPUS%.jsonl}.BUILD_REPORT.json"
  [[ -f "${report}" ]] || { log "missing build report ${report}"; exit 1; }
  grep -q '"index_corpus_delta": 0' "${report}" || { log "corpus/index mismatch; see ${report}"; exit 1; }
}

run_eval() {
  local actor_urls="$1" tp="$2" out_suffix="$3"
  local out="${REL_OUT}/eval_h1_bcplus_full_${MODEL_SLUG}_${COMPONENT}_${out_suffix}_${RUN_ID}"
  local log_path="${REL_LOGS}/eval_${MODEL_SLUG}_${COMPONENT}.log"
  local extra=()
  [[ "${COMPONENT}" == "all" ]] && extra+=(
    --verify-base-url "http://127.0.0.1:${ALL_VERIFY_PORT}/v1"
    --verify-model "${VERIFY_MODEL_NAME}"
  )
  log "START model=${MODEL_SLUG} component=${COMPONENT} benchmark=bcplus_full tp=${tp} urls=${actor_urls} out=${out}"
  local rc=0
  "${PY}" scripts/run_eval.py \
    --harness Harness-1 \
    --benchmark bcplus_full \
    --model_name "${MODEL_PATH}" \
    --evaluation-path upstream_api \
    --retrieval-backend local_bm25 \
    --api-base-url "${actor_urls}" \
    --api-model "${API_MODEL}" \
    --index-path "${BCPLUS_INDEX}" \
    --corpus-path "${BCPLUS_CORPUS}" \
    --reranker none \
    --offline \
    --max-turns 40 \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --temperature 1.0 \
    --tp "${tp}" \
    --eval-stagger-s "${WORKER_STAGGER}" \
    --component "${COMPONENT}" \
    --out "${out}" \
    "${extra[@]}" \
    2>&1 | tee -a "${log_path}" >&2 || rc=$?
  [[ "${rc}" -eq 0 ]] || { log "WARN eval exited ${rc} model=${MODEL_SLUG} component=${COMPONENT}"; return "${rc}"; }
  log "DONE model=${MODEL_SLUG} component=${COMPONENT}"
}

main() {
  [[ -d "${MODEL_PATH}" ]] || { log "missing model dir ${MODEL_PATH}"; exit 1; }
  assert_local_retrieval_ready() {
    [[ -d "${BCPLUS_INDEX}" ]] || { log "missing Lucene index ${BCPLUS_INDEX}"; exit 1; }
    return 0
  }
  assert_local_retrieval_ready
  ensure_full_corpus

  csv_to_array "${ALL_ACTOR_PORTS}" _ports
  ALL_ACTOR_URLS="$(join_urls "${_ports[@]}")"

  log "RUN_ID=${RUN_ID} model=${MODEL_SLUG} component=all actors=${ALL_ACTOR_GPUS} verify=${ALL_VERIFY_GPU} all_eval_tp=${ALL_EVAL_TP} gpu_util=${GPU_UTIL} worker_stagger=${WORKER_STAGGER}"

  start_all_stack
  if ! run_eval "${ALL_ACTOR_URLS}" "${ALL_EVAL_TP}" "gpu${ALL_ACTOR_GPUS//,/}v${ALL_VERIFY_GPU}"; then
    log "continuing after all eval failure for ${MODEL_SLUG}"
  fi

  stop_all_actors; stop_verify

  GIT_COMMIT="$(git -C . rev-parse HEAD 2>/dev/null || echo unknown)"
  cat > "${REL_LOGS}/MANIFEST.json" <<EOF
{
  "run_id": "${RUN_ID}",
  "benchmark": "bcplus_full",
  "evaluation_path": "upstream_api",
  "eval_profile": "upstream_core_local_bm25",
  "model": "${MODEL_PATH}",
  "model_slug": "${MODEL_SLUG}",
  "component": "all",
  "actor_gpus": "${ALL_ACTOR_GPUS}",
  "actor_ports": "${ALL_ACTOR_PORTS}",
  "verify_gpu": "${ALL_VERIFY_GPU}",
  "verify_port": ${ALL_VERIFY_PORT},
  "actor_tp": ${ALL_TP},
  "eval_tp": ${ALL_EVAL_TP},
  "max_num_seqs": ${MAX_NUM_SEQS},
  "max_model_len": ${MAX_MODEL_LEN},
  "gpu_util": ${GPU_UTIL},
  "worker_stagger_s": ${WORKER_STAGGER},
  "retrieval": "local_bm25 (Pyserini/Lucene), no Chroma server",
  "tool_call_parser": "glm45",
  "reasoning_parser": null,
  "tool_call_recovery": "api_adapter.extract_tool_calls_from_content",
  "smoke_test": "skipped (tool_calls empty from vllm; content fallback is the path)",
  "all_actor_urls": "${ALL_ACTOR_URLS}",
  "all_verify_url": "http://127.0.0.1:${ALL_VERIFY_PORT}/v1",
  "corpus_path": "${BCPLUS_CORPUS}",
  "index_path": "${BCPLUS_INDEX}",
  "git_commit": "${GIT_COMMIT}"
}
EOF

  log "finished ${MODEL_SLUG} all run; outputs under ${REL_OUT}/eval_h1_bcplus_full_${MODEL_SLUG}_all_*_${RUN_ID}"
  log "manifest ${REL_LOGS}/MANIFEST.json"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
