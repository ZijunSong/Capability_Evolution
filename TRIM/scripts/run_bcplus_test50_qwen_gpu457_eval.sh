#!/usr/bin/env bash
# Qwen3-4B-Instruct bcplus_test_50 — sequential zero → all, high-concurrency eval.
# GPUs 4,5,7: zero actor on 7; all actors on 5+7, harness-1 verifier on 4.
#
#   bash TRIM/scripts/run_bcplus_test50_qwen_gpu457_eval.sh
#
# Optional env: RUN_ID, GPU, TP, ALL_ACTOR_GPUS, ALL_VERIFY_GPU, ALL_EVAL_TP, ...
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRIM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${TRIM_ROOT}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_qwen_test50_gpu457}"
PY="${PY:-/data/ppnm/miniconda3/envs/bishop/bin/python}"
VLLM="${VLLM:-/data/ppnm/miniconda3/envs/bishop/bin/vllm}"

REL_MODELS="../../models"
REL_BCPLUS="../SCOPE/external/BrowseComp-Plus"
REL_OUT="outputs"
REL_LOGS="${REL_OUT}/logs/${RUN_ID}"
REL_SCAPE_EASYOPD="../SCAPE-EasyOPD"
PKG="/data/ppnm/trim_bcplus_test50_eval_upstream_api"

MODEL_PATH="${MODEL_PATH:-/data/ppnm/models/Qwen3-4B-Instruct-2507}"
MODEL_SLUG="Qwen3-4B-Instruct-2507"
API_MODEL="${API_MODEL:-Qwen3-4B-Instruct-2507}"

BCPLUS_INDEX="${REL_BCPLUS}/indexes/bm25"
BCPLUS_CORPUS="${REL_BCPLUS}/data/browsecomp_plus_corpus_full.jsonl"
VERIFY_MODEL_PATH="${REL_MODELS}/harness-1"

# zero: one actor on GPU 7
GPU="${GPU:-7}"
TP="${TP:-24}"
ZERO_ACTOR_PORT="${ZERO_ACTOR_PORT:-8040}"

# all: 2 Qwen actors (5,7) + harness-1 verifier (4)
ALL_ACTOR_GPUS="${ALL_ACTOR_GPUS:-5,7}"
ALL_VERIFY_GPU="${ALL_VERIFY_GPU:-4}"
ALL_ACTOR_PORTS="${ALL_ACTOR_PORTS:-8042,8044}"
ALL_VERIFY_PORT="${ALL_VERIFY_PORT:-8050}"
ALL_TP="${ALL_TP:-2}"
ALL_EVAL_TP="${ALL_EVAL_TP:-24}"

VERIFY_MODEL_NAME="${VERIFY_MODEL_NAME:-harness-1-verifier}"
GPU_UTIL="${GPU_UTIL:-0.75}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}"
WORKER_STAGGER="${WORKER_STAGGER:-2}"

export PYTHONPATH=".:${REL_SCAPE_EASYOPD}"
export TRIM_GPU_KEEPALIVE=0

mkdir -p "${REL_LOGS}" "${PKG}/results/${RUN_ID}" "${PKG}/logs/${RUN_ID}"

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
  return 1
}

verify_api_model() {
  local port="$1" expected="$2" body ids
  body="$(curl -sf "http://127.0.0.1:${port}/v1/models")" || return 1
  ids="$("${PY}" -c "import json,sys; d=json.load(sys.stdin); print(','.join(m.get('id','') for m in d.get('data',[])))" <<< "${body}")"
  [[ ",${ids}," == *",${expected},"* ]] || { log "port ${port} model mismatch: expected ${expected}, got [${ids}]"; return 1; }
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
  case "${1}" in
    Qwen3-4B-Instruct-2507)
      echo "--enable-auto-tool-choice --tool-call-parser hermes --max-model-len 32768 --trust-remote-code"
      ;;
    *)
      echo "--enable-auto-tool-choice --tool-call-parser openai --max-model-len 32768 --trust-remote-code"
      ;;
  esac
}

smoke_test_tool_call() {
  local port="$1" model="$2"
  log "smoke test tool call port=${port} model=${model}"
  "${PY}" - <<PY
import json, urllib.request
port, model = ${port}, ${model@Q}
payload = {
    "model": model,
    "messages": [{"role": "user", "content": "Search the corpus for smoke test query."}],
    "tools": [{"type": "function", "function": {
        "name": "search_corpus", "description": "Search corpus",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    }}],
    "tool_choice": "auto", "max_tokens": 256, "temperature": 0.0,
}
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/chat/completions",
    data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST",
)
with urllib.request.urlopen(req, timeout=120) as resp:
    body = json.loads(resp.read().decode())
tool_calls = (body.get("choices") or [{}])[0].get("message", {}).get("tool_calls") or []
if not tool_calls:
    raise SystemExit(f"smoke test missing tool_calls: {json.dumps(body)[:500]}")
print("smoke_ok", tool_calls[0]["function"]["name"])
PY
}

start_vllm_bg() {
  local gpu="$1" port="$2" model_path="$3" served_name="$4" extra="$5" log_path="$6"
  stop_port "${port}" "${log_path}.pid"
  wait_port_free "${port}" 120
  log "starting vLLM GPU${gpu} port ${port} model=${model_path} served=${served_name}"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="${gpu}" nohup "${VLLM}" serve "${model_path}" \
    --host 127.0.0.1 --port "${port}" --served-model-name "${served_name}" \
    ${extra} --gpu-memory-utilization "${GPU_UTIL}" --max-num-seqs "${VLLM_MAX_NUM_SEQS}" --enforce-eager \
    > "${log_path}" 2>&1 &
  local pid=$!
  echo "${pid}" > "${log_path}.pid"
  wait_api "${port}"
  verify_api_model "${port}" "${served_name}"
  echo "${pid}"
}

start_zero_stack() {
  stop_all_actors; stop_verify
  local extra pid
  extra="$(vllm_extra_for_model "${MODEL_SLUG}")"
  pid="$(start_vllm_bg "${GPU}" "${ZERO_ACTOR_PORT}" "${MODEL_PATH}" "${API_MODEL}" \
    "${extra}" "${REL_LOGS}/actor_zero.log")"
  ACTOR_PIDS=("$(normalize_pid "${pid}")")
  smoke_test_tool_call "${ZERO_ACTOR_PORT}" "${API_MODEL}"
}

start_all_stack() {
  stop_all_actors; stop_verify
  local actor_gpus=() actor_ports=()
  csv_to_array "${ALL_ACTOR_GPUS}" actor_gpus
  csv_to_array "${ALL_ACTOR_PORTS}" actor_ports
  ((${#actor_gpus[@]} == ALL_TP)) || { log "ALL_ACTOR_GPUS count != ALL_TP"; exit 1; }
  local extra pid i gpu port
  extra="$(vllm_extra_for_model "${MODEL_SLUG}")"
  ACTOR_PIDS=()
  for i in "${!actor_gpus[@]}"; do
    gpu="${actor_gpus[$i]}"; port="${actor_ports[$i]}"
    pid="$(start_vllm_bg "${gpu}" "${port}" "${MODEL_PATH}" "${API_MODEL}" \
      "${extra}" "${REL_LOGS}/actor_all_gpu${gpu}.log")"
    ACTOR_PIDS+=("$(normalize_pid "${pid}")")
  done
  smoke_test_tool_call "${actor_ports[0]}" "${API_MODEL}"
  VERIFY_PID="$(start_vllm_bg "${ALL_VERIFY_GPU}" "${ALL_VERIFY_PORT}" "${VERIFY_MODEL_PATH}" \
    "${VERIFY_MODEL_NAME}" "--max-model-len 8192 --trust-remote-code --moe-backend triton" \
    "${REL_LOGS}/verify.log")"
  VERIFY_PID="$(normalize_pid "${VERIFY_PID}")"
}

ensure_full_corpus() {
  [[ -f "${BCPLUS_CORPUS}" ]] && return 0
  log "building full corpus → ${BCPLUS_CORPUS}"
  "${PY}" scripts/build_browsecomp_corpus_from_index.py --mode full --out "${BCPLUS_CORPUS}" \
    >> "${REL_LOGS}/corpus_build.log" 2>&1
}

run_eval() {
  local component="$1" actor_urls="$2" tp="$3" out_suffix="$4"
  local out="${REL_OUT}/eval_h1_bcplus_test50_${MODEL_SLUG}_${component}_${out_suffix}_${RUN_ID}"
  local pkg_out="${PKG}/results/${RUN_ID}/${component}"
  local log_path="${REL_LOGS}/eval_${component}.log"
  local extra=()
  [[ "${component}" == "all" ]] && extra+=(
    --verify-base-url "http://127.0.0.1:${ALL_VERIFY_PORT}/v1"
    --verify-model "${VERIFY_MODEL_NAME}"
  )
  log "START component=${component} tp=${tp} urls=${actor_urls} out=${out}"
  rm -rf "${out}" "${pkg_out}"
  local rc=0
  "${PY}" scripts/run_eval.py \
    --harness Harness-1 --benchmark bcplus_test_50 --model_name "${MODEL_PATH}" \
    --evaluation-path upstream_api --retrieval-backend local_bm25 \
    --api-base-url "${actor_urls}" --api-model "${API_MODEL}" \
    --index-path "${BCPLUS_INDEX}" --corpus-path "${BCPLUS_CORPUS}" \
    --reranker none --offline \
    --max-turns 40 --max-new-tokens 2048 --temperature 1.0 \
    --tp "${tp}" --eval-stagger-s "${WORKER_STAGGER}" \
    --component "${component}" --out "${out}" \
    "${extra[@]}" 2>&1 | tee -a "${log_path}" >&2 || rc=$?
  rsync -a "${out}/" "${pkg_out}/"
  cp -a "${log_path}" "${PKG}/logs/${RUN_ID}/eval_${component}.log"
  [[ "${rc}" -eq 0 ]] || { log "WARN eval exited ${rc} component=${component}"; return "${rc}"; }
  log "DONE component=${component}"
}

main() {
  [[ -d "${MODEL_PATH}" ]] || { log "missing model ${MODEL_PATH}"; exit 1; }
  ensure_full_corpus

  csv_to_array "${ALL_ACTOR_PORTS}" _ports
  ALL_ACTOR_URLS="$(join_urls "${_ports[@]}")"
  ZERO_ACTOR_URL="http://127.0.0.1:${ZERO_ACTOR_PORT}/v1"

  log "RUN_ID=${RUN_ID} model=${MODEL_SLUG} zero_gpu=${GPU} tp=${TP} all_actors=${ALL_ACTOR_GPUS} verify=${ALL_VERIFY_GPU} all_eval_tp=${ALL_EVAL_TP}"

  start_zero_stack
  run_eval zero "${ZERO_ACTOR_URL}" "${TP}" "gpu${GPU}"

  start_all_stack
  run_eval all "${ALL_ACTOR_URLS}" "${ALL_EVAL_TP}" "gpu${ALL_ACTOR_GPUS//,/}v${ALL_VERIFY_GPU}"

  stop_all_actors; stop_verify

  GIT_COMMIT="$(git -C . rev-parse HEAD 2>/dev/null || echo unknown)"
  cat > "${PKG}/results/${RUN_ID}/MANIFEST.json" <<EOF
{
  "run_id": "${RUN_ID}",
  "model": "${MODEL_PATH}",
  "benchmark": "bcplus_test_50",
  "evaluation_path": "upstream_api",
  "eval_profile": "upstream_core_local_bm25",
  "corpus_path": "${BCPLUS_CORPUS}",
  "git_commit": "${GIT_COMMIT}",
  "zero": {"gpu": "${GPU}", "eval_tp": ${TP}, "actor_port": ${ZERO_ACTOR_PORT}},
  "all": {
    "actor_gpus": "${ALL_ACTOR_GPUS}",
    "actor_ports": "${ALL_ACTOR_PORTS}",
    "verify_gpu": "${ALL_VERIFY_GPU}",
    "verify_port": ${ALL_VERIFY_PORT},
    "actor_tp": ${ALL_TP},
    "eval_tp": ${ALL_EVAL_TP}
  },
  "vllm_max_num_seqs": ${VLLM_MAX_NUM_SEQS},
  "worker_stagger_s": ${WORKER_STAGGER},
  "tool_call_parser": "hermes"
}
EOF
  cp "${PKG}/results/${RUN_ID}/MANIFEST.json" "${REL_LOGS}/MANIFEST.json"
  log "finished zero+all; results ${PKG}/results/${RUN_ID}/"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
