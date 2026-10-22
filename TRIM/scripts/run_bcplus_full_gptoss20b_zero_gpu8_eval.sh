#!/usr/bin/env bash
# Harness-1 upstream_api bcplus_full (830 queries) — GPT-OSS-20B ZERO ONLY,
# tuned for 8 × gm402-80g. Derived from run_bcplus_full_gpu4_eval.sh.
#
# Why this layout (fastest path to 830 evaluated queries):
#   * The upstream_api eval path (run_eval.py --evaluation-path upstream_api)
#     does NOT launch vLLM itself; it sends chat/completions to external actor
#     servers started by this script. Each API-client worker
#     (trim.eval.harness1_api_worker) runs ONE query's multi-turn agent loop
#     sequentially (`for row in rows: await run_one_query_api`). So total
#     concurrency = #workers (= --tp / eval_replicas), and vLLM's per-actor
#     batching only helps when multiple in-flight workers hit the same actor.
#   * To use all 8 cards with no TP overhead, we start 8 single-GPU vLLM
#     actors (TP=1, one per GPU 0..7) and give run_eval 8 actor URLs. With
#     --tp 16 we launch 16 workers; identity_for_actor_rank maps worker rank r
#     to url[r % 8], so each GPU gets 2 workers → each actor batches ~2
#     concurrent sequences (vLLM --max-num-seqs 8 leaves headroom for bursts).
#     This is 2× the per-GPU concurrency of --tp 8 and avoids TP=8
#     all-reduce cost on a ~3.6B-active MoE that fits one gm402-80g easily.
#   * No Chroma server is used. Retrieval is local_bm25 (Pyserini/Lucene) with
#     --reranker none --offline; chromadb is imported by the harness chain but
#     never serves. Verified baseline: outputs/.../harness1_bcplus_full_gpt_oss_20b_zero_8gpu_20260908_memqa_v5
#     completed with every shard DONE.json ok=true, leak_count=0.
#
# Optional env (defaults shown):
#   ZERO_ACTOR_GPUS=0,1,2,3,4,5,6,7  ZERO_ACTOR_PORTS=8040,8042,...,8054
#   TP=16  MAX_NUM_SEQS=8  MAX_MODEL_LEN=32768  GPU_UTIL=0.85  WORKER_STAGGER=10
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRIM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${TRIM_ROOT}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_bcplus_full_gptoss20b_zero_gpu8}"
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

# gpt-oss-20b zero only.
MODEL_PATH="${REL_MODELS}/gpt-oss-20b"
MODEL_SLUG="gpt-oss-20b"
API_MODEL="gpt-oss-20b"
COMPONENT="zero"

# 8 single-GPU actors (TP=1 each). No tensor parallel → no all-reduce cost.
ZERO_ACTOR_GPUS="${ZERO_ACTOR_GPUS:-0,1,2,3,4,5,6,7}"
ZERO_ACTOR_PORTS="${ZERO_ACTOR_PORTS:-8040,8042,8044,8046,8048,8050,8052,8054}"
TP="${TP:-16}"                 # 16 API-client workers → 2 per GPU (raised from 3)
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"   # vLLM concurrent sequences per actor (raised)
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_UTIL="${GPU_UTIL:-0.85}"   # gpt-oss-20b is small; leave KV cache room
WORKER_STAGGER="${WORKER_STAGGER:-10}"

export PYTHONPATH=".:${REL_SCAPE_EASYOPD}"
export TRIM_GPU_KEEPALIVE=0

mkdir -p "${REL_LOGS}"
ACTOR_PIDS=()
MASTER_LOG="${REL_LOGS}/master.log"

# Log to file + stderr only. Never stdout (stdout is reserved for pid capture).
log() {
  echo "[$(date -Is)] $*" >> "${MASTER_LOG}"
  echo "[$(date -Is)] $*" >&2
}

normalize_pid() {
  local raw="${1:-}"
  raw="${raw//$'\r'/}"
  while [[ "${raw}" == *$'\n' ]]; do
    raw="${raw%$'\n'}"
  done
  if [[ "${raw}" =~ ^([0-9]+)$ ]]; then
    echo "${BASH_REMATCH[1]}"
    return 0
  fi
  local last="${raw##*$'\n'}"
  if [[ "${last}" =~ ^([0-9]+)$ ]]; then
    echo "${BASH_REMATCH[1]}"
    return 0
  fi
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
  if command -v lsof >/dev/null 2>&1; then
    lsof -ti ":${port}" 2>/dev/null || true
    return 0
  fi
  if command -v ss >/dev/null 2>&1; then
    ss -lptn "sport = :${port}" 2>/dev/null | sed -n 's/.*pid=\([0-9]*\).*/\1/p' || true
    return 0
  fi
  if command -v netstat >/dev/null 2>&1; then
    netstat -lptn 2>/dev/null | grep ":${port} " | sed -E 's|.*/([0-9]+)/.*|\1|' | sort -u || true
    return 0
  fi
  return 0
}

stop_port() {
  local port="$1" pid_file="${2:-}"
  local pid pids
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

stop_all_actors() {
  for pid in "${ACTOR_PIDS[@]:-}"; do
    stop_pid "${pid}"
  done
  ACTOR_PIDS=()
}
trap stop_all_actors EXIT

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
  local port="$1" expected="$2"
  local body ids
  body="$(curl -sf "http://127.0.0.1:${port}/v1/models")" || {
    log "failed to read /v1/models on port ${port}"
    return 1
  }
  ids="$("${PY}" -c "import json,sys; d=json.load(sys.stdin); print(','.join(m.get('id','') for m in d.get('data',[])))" <<< "${body}")"
  if [[ ",${ids}," != *",${expected},"* ]]; then
    log "port ${port} served-model mismatch: expected ${expected}, got [${ids}]"
    return 1
  fi
  return 0
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
  # gpt-oss-20b: openai tool-call parser (vLLM harmony). The upstream_api
  # chat/completions path lets vLLM's harmony chat template wrap prompts, so
  # this does NOT hit the Harness-G raw-token "not a gpt-oss Harmony prompt"
  # failure. (Verified by the completed memqa-v5 zero run.)
  echo "--enable-auto-tool-choice --tool-call-parser openai --max-model-len ${MAX_MODEL_LEN} --trust-remote-code --moe-backend triton --max-num-seqs ${MAX_NUM_SEQS}"
}

start_vllm_bg() {
  local gpu="$1" port="$2" model_path="$3" served_name="$4" extra="$5" log_path="$6"

  stop_port "${port}" "${log_path}.pid"
  wait_port_free "${port}" 120

  log "starting vLLM GPU${gpu} port ${port} model=${model_path} served=${served_name}"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="${gpu}" nohup "${VLLM}" serve "${model_path}" \
    --host 127.0.0.1 --port "${port}" \
    --served-model-name "${served_name}" \
    ${extra} \
    --gpu-memory-utilization "${GPU_UTIL}" --enforce-eager \
    > "${log_path}" 2>&1 &
  local pid=$!
  echo "${pid}" > "${log_path}.pid"
  wait_api "${port}"
  verify_api_model "${port}" "${served_name}"
  echo "${pid}"
}

start_zero_stack() {
  stop_all_actors

  local actor_gpus=() actor_ports=()
  csv_to_array "${ZERO_ACTOR_GPUS}" actor_gpus
  csv_to_array "${ZERO_ACTOR_PORTS}" actor_ports

  if ((${#actor_gpus[@]} != ${#actor_ports[@]})); then
    log "ZERO_ACTOR_GPUS count (${#actor_gpus[@]}) must match ZERO_ACTOR_PORTS count (${#actor_ports[@]})"
    exit 1
  fi
  local n_actors=${#actor_gpus[@]}
  log "starting ${n_actors} single-GPU vLLM actors (TP=1 each); TP=${TP} workers → ~$((TP / n_actors)) per GPU"

  local extra pid i gpu port
  extra="$(vllm_extra_for_model "${MODEL_SLUG}")"
  ACTOR_PIDS=()
  for i in "${!actor_gpus[@]}"; do
    gpu="${actor_gpus[$i]}"
    port="${actor_ports[$i]}"
    pid="$(start_vllm_bg "${gpu}" "${port}" "${MODEL_PATH}" "${API_MODEL}" \
      "${extra}" "${REL_LOGS}/actor_${MODEL_SLUG}_zero_gpu${gpu}.log")"
    pid="$(normalize_pid "${pid}")"
    ACTOR_PIDS+=("${pid}")
  done
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
  local actor_urls="$1" tp="$2" out_suffix="$3"
  local out="${REL_OUT}/eval_h1_bcplus_full_${MODEL_SLUG}_${COMPONENT}_${out_suffix}_${RUN_ID}"
  local log_path="${REL_LOGS}/eval_${MODEL_SLUG}_${COMPONENT}.log"

  log "START model=${MODEL_SLUG} component=${COMPONENT} tp=${tp} actor_urls=${actor_urls} out=${out}"
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
    --max-new-tokens 2048 \
    --temperature 1.0 \
    --tp "${tp}" \
    --eval-stagger-s "${WORKER_STAGGER}" \
    --component "${COMPONENT}" \
    --out "${out}" \
    2>&1 | tee -a "${log_path}" >&2 || rc=$?
  if [[ "${rc}" -ne 0 ]]; then
    log "WARN eval exited ${rc} model=${MODEL_SLUG} component=${COMPONENT} out=${out}"
    return "${rc}"
  fi
  log "DONE model=${MODEL_SLUG} component=${COMPONENT}"
  return 0
}

main() {
  ensure_full_corpus

  csv_to_array "${ZERO_ACTOR_PORTS}" _ZERO_PORTS_CHECK
  ZERO_ACTOR_URLS="$(join_urls "${_ZERO_PORTS_CHECK[@]}")"

  log "RUN_ID=${RUN_ID} actors=${ZERO_ACTOR_GPUS} ports=${ZERO_ACTOR_PORTS} tp=${TP} max_num_seqs=${MAX_NUM_SEQS} max_model_len=${MAX_MODEL_LEN} gpu_util=${GPU_UTIL} worker_stagger=${WORKER_STAGGER}"

  [[ -d "${MODEL_PATH}" ]] || { log "missing model dir ${MODEL_PATH}"; exit 1; }

  start_zero_stack
  if ! run_eval "${ZERO_ACTOR_URLS}" "${TP}" "gpu8_tp${TP}"; then
    log "continuing after zero eval failure for ${MODEL_SLUG}"
  fi

  stop_all_actors

  GIT_COMMIT="$(git -C . rev-parse HEAD 2>/dev/null || echo unknown)"
  cat > "${REL_LOGS}/MANIFEST.json" <<EOF
{
  "run_id": "${RUN_ID}",
  "benchmark": "bcplus_full",
  "evaluation_path": "upstream_api",
  "eval_profile": "upstream_core_local_bm25",
  "model": "${REL_MODELS}/gpt-oss-20b",
  "component": "zero",
  "actor_gpus": "${ZERO_ACTOR_GPUS}",
  "actor_ports": "${ZERO_ACTOR_PORTS}",
  "tp": ${TP},
  "max_num_seqs": ${MAX_NUM_SEQS},
  "max_model_len": ${MAX_MODEL_LEN},
  "gpu_util": ${GPU_UTIL},
  "worker_stagger_s": ${WORKER_STAGGER},
  "layout": "8 single-GPU vLLM actors (TP=1), ${TP} API-client workers",
  "retrieval": "local_bm25 (Pyserini/Lucene), no Chroma server",
  "zero_actor_urls": "${ZERO_ACTOR_URLS}",
  "corpus_path": "${BCPLUS_CORPUS}",
  "index_path": "${BCPLUS_INDEX}",
  "git_commit": "${GIT_COMMIT}"
}
EOF

  log "finished gpt-oss-20b zero run; outputs under ${REL_OUT}/eval_h1_bcplus_full_${MODEL_SLUG}_zero_*_${RUN_ID}"
  log "manifest ${REL_LOGS}/MANIFEST.json"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
