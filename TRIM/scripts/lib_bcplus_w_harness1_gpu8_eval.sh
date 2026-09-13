# Shared helpers for bcplus_full Harness-1 upstream_api eval (8×GPU layout).
# Source from run_bcplus_w_harness1_{qwen,gptoss}_{zero,all}_gpu8_eval.sh — do not execute directly.

lib_bcplus_gpu8_init() {
  : "${SCRIPT_DIR:?SCRIPT_DIR must be set before sourcing lib}"
  TRIM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
  cd "${TRIM_ROOT}"

  PY="${PY:-python}"
  VLLM="${VLLM:-vllm}"
  BENCHMARK="${BENCHMARK:-bcplus_full}"

  REL_MODELS="${REL_MODELS:-../../models}"
  REL_BCPLUS="${REL_BCPLUS:-../SCOPE/external/BrowseComp-Plus}"
  REL_OUT="${REL_OUT:-outputs}"
  REL_SCAPE_EASYOPD="${REL_SCAPE_EASYOPD:-../SCAPE-EasyOPD}"

  BCPLUS_INDEX="${REL_BCPLUS}/indexes/bm25"
  BCPLUS_CORPUS="${REL_BCPLUS}/data/browsecomp_plus_corpus_full.jsonl"
  VERIFY_MODEL_PATH="${REL_MODELS}/harness-1"

  GPU="${GPU:-7}"
  TP="${TP:-64}"
  ZERO_ACTOR_PORT="${ZERO_ACTOR_PORT:-8040}"

  ALL_ACTOR_GPUS="${ALL_ACTOR_GPUS:-0,1,2,3,4,5}"
  ALL_ACTOR_PORTS="${ALL_ACTOR_PORTS:-8042,8044,8046,8048,8052,8054}"
  ALL_TP="${ALL_TP:-6}"
  ALL_VERIFY_GPU="${ALL_VERIFY_GPU:-6}"
  ALL_VERIFY_PORT="${ALL_VERIFY_PORT:-8050}"
  ALL_EVAL_TP="${ALL_EVAL_TP:-64}"

  VERIFY_MODEL_NAME="${VERIFY_MODEL_NAME:-harness-1-verifier}"
  GPU_UTIL="${GPU_UTIL:-0.70}"
  VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-128}"
  WORKER_STAGGER="${WORKER_STAGGER:-0}"

  export PYTHONPATH=".:${REL_SCAPE_EASYOPD}"
  export TRIM_GPU_KEEPALIVE=0
  export HARNESS1_FORBID_CHROMA=1
  export CURATE_NUDGE_POLICY="${CURATE_NUDGE_POLICY:-legacy}"

  : "${MODEL_PATH:?MODEL_PATH must be set}"
  : "${MODEL_SLUG:?MODEL_SLUG must be set}"
  : "${API_MODEL:?API_MODEL must be set}"
  : "${COMPONENT:?COMPONENT must be zero or all}"
  : "${RUN_ID:?RUN_ID must be set}"

  REL_LOGS="${REL_OUT}/logs/${RUN_ID}"
  mkdir -p "${REL_LOGS}"

  ACTOR_PIDS=()
  VERIFY_PID=""
  MASTER_LOG="${REL_LOGS}/master.log"
}

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
lib_bcplus_gpu8_cleanup() { stop_all_actors; stop_verify; }

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
    log "failed to read /v1/models on port ${port}"
    return 1
  }
  ids="$("${PY}" -c "import json,sys; d=json.load(sys.stdin); print(','.join(m.get('id','') for m in d.get('data',[])))" <<< "${body}")"
  if [[ ",${ids}," != *",${expected},"* ]]; then
    log "port ${port} served-model mismatch: expected ${expected}, got [${ids}]"
    return 1
  fi
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
  local slug="$1"
  case "${slug}" in
    gpt-oss-20b|harness-1)
      echo "--enable-auto-tool-choice --tool-call-parser openai --max-model-len 32768 --trust-remote-code --moe-backend triton"
      ;;
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
import json, urllib.error, urllib.request
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
try:
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode())
except urllib.error.HTTPError as exc:
    raise SystemExit(f"smoke test HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:500]}")
tool_calls = (body.get("choices") or [{}])[0].get("message", {}).get("tool_calls") or []
if not tool_calls:
    raise SystemExit(f"smoke test missing tool_calls: {json.dumps(body)[:500]}")
name = ((tool_calls[0].get("function") or {}).get("name") or "")
if name != "search_corpus":
    raise SystemExit(f"smoke test unexpected tool name: {name!r}")
print("smoke_ok", name)
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
  stop_all_actors
  stop_verify
  local extra pid
  extra="$(vllm_extra_for_model "${MODEL_SLUG}")"
  pid="$(start_vllm_bg "${GPU}" "${ZERO_ACTOR_PORT}" "${MODEL_PATH}" "${API_MODEL}" \
    "${extra}" "${REL_LOGS}/actor_${MODEL_SLUG}_zero.log")"
  ACTOR_PIDS=("$(normalize_pid "${pid}")")
  if [[ "${MODEL_SLUG}" == "Qwen3-4B-Instruct-2507" ]]; then
    smoke_test_tool_call "${ZERO_ACTOR_PORT}" "${API_MODEL}"
  fi
}

start_all_stack() {
  stop_all_actors
  stop_verify
  local actor_gpus=() actor_ports=()
  csv_to_array "${ALL_ACTOR_GPUS}" actor_gpus
  csv_to_array "${ALL_ACTOR_PORTS}" actor_ports
  ((${#actor_gpus[@]} == ALL_TP)) || { log "ALL_ACTOR_GPUS count != ALL_TP"; exit 1; }
  ((${#actor_ports[@]} == ALL_TP)) || { log "ALL_ACTOR_PORTS count != ALL_TP"; exit 1; }
  local extra pid i gpu port
  extra="$(vllm_extra_for_model "${MODEL_SLUG}")"
  ACTOR_PIDS=()
  for i in "${!actor_gpus[@]}"; do
    gpu="${actor_gpus[$i]}"
    port="${actor_ports[$i]}"
    pid="$(start_vllm_bg "${gpu}" "${port}" "${MODEL_PATH}" "${API_MODEL}" \
      "${extra}" "${REL_LOGS}/actor_${MODEL_SLUG}_all_gpu${gpu}.log")"
    ACTOR_PIDS+=("$(normalize_pid "${pid}")")
  done
  if [[ "${MODEL_SLUG}" == "Qwen3-4B-Instruct-2507" ]]; then
    smoke_test_tool_call "${actor_ports[0]}" "${API_MODEL}"
  fi
  VERIFY_PID="$(start_vllm_bg "${ALL_VERIFY_GPU}" "${ALL_VERIFY_PORT}" "${VERIFY_MODEL_PATH}" \
    "${VERIFY_MODEL_NAME}" "--max-model-len 8192 --trust-remote-code --moe-backend triton" \
    "${REL_LOGS}/verify_${MODEL_SLUG}.log")"
  VERIFY_PID="$(normalize_pid "${VERIFY_PID}")"
}

assert_local_retrieval_ready() {
  [[ -d "${BCPLUS_INDEX}" ]] || { log "missing Lucene index ${BCPLUS_INDEX} (local_bm25; no Chroma)"; exit 1; }
  if [[ ! -f "${BCPLUS_CORPUS}" ]]; then
    log "corpus missing ${BCPLUS_CORPUS}; will build from index"
    return 0
  fi
  log "local_bm25 ready index=${BCPLUS_INDEX} corpus=${BCPLUS_CORPUS}"
}

ensure_full_corpus() {
  [[ -f "${BCPLUS_CORPUS}" ]] && return 0
  log "building full corpus → ${BCPLUS_CORPUS}"
  "${PY}" scripts/build_browsecomp_corpus_from_index.py --mode full --out "${BCPLUS_CORPUS}" \
    >> "${REL_LOGS}/corpus_build.log" 2>&1
  local report="${BCPLUS_CORPUS%.jsonl}.BUILD_REPORT.json"
  [[ -f "${report}" ]] || { log "missing build report ${report}"; exit 1; }
  grep -q '"index_corpus_delta": 0' "${report}" || { log "corpus/index mismatch; see ${report}"; exit 1; }
}

run_eval_cell() {
  local actor_urls="$1" tp="$2" out_suffix="$3"
  local out="${REL_OUT}/eval_h1_${BENCHMARK}_${MODEL_SLUG}_${COMPONENT}_${out_suffix}_${RUN_ID}"
  local log_path="${REL_LOGS}/eval_${MODEL_SLUG}_${COMPONENT}.log"
  local extra=()
  [[ "${COMPONENT}" == "all" ]] && extra+=(
    --verify-base-url "http://127.0.0.1:${ALL_VERIFY_PORT}/v1"
    --verify-model "${VERIFY_MODEL_NAME}"
  )
  log "START model=${MODEL_SLUG} component=${COMPONENT} benchmark=${BENCHMARK} tp=${tp} urls=${actor_urls} out=${out}"
  local rc=0
  "${PY}" scripts/run_eval.py \
    --harness Harness-1 \
    --benchmark "${BENCHMARK}" \
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
    "${extra[@]}" \
    2>&1 | tee -a "${log_path}" >&2 || rc=$?
  [[ "${rc}" -eq 0 ]] || { log "WARN eval exited ${rc} model=${MODEL_SLUG} component=${COMPONENT}"; return "${rc}"; }
  log "DONE model=${MODEL_SLUG} component=${COMPONENT} out=${out}"
}

write_cell_manifest() {
  local actor_urls="$1" out_suffix="$2" eval_tp="$3"
  local tool_parser="openai"
  [[ "${MODEL_SLUG}" == "Qwen3-4B-Instruct-2507" ]] && tool_parser="hermes"
  GIT_COMMIT="$("${PY}" - <<'PY' 2>/dev/null || echo unknown
import subprocess
try:
    print(subprocess.check_output(["git", "-C", ".", "rev-parse", "HEAD"], text=True).strip())
except Exception:
    print("unknown")
PY
)"
  FIX_VERSION="$("${PY}" - <<'PY' 2>/dev/null || echo unknown
try:
    from trim.upstream_harness1.fix_manifest import PUBLIC_FIX_VERSION
    print(PUBLIC_FIX_VERSION)
except Exception:
    print("unknown")
PY
)"
  cat > "${REL_LOGS}/MANIFEST.json" <<EOF
{
  "run_id": "${RUN_ID}",
  "model": "${MODEL_PATH}",
  "model_slug": "${MODEL_SLUG}",
  "component": "${COMPONENT}",
  "benchmark": "${BENCHMARK}",
  "evaluation_path": "upstream_api",
  "eval_profile": "upstream_core_local_bm25",
  "retrieval_backend": "local_bm25",
  "chroma_required": false,
  "public_fix_version": "${FIX_VERSION}",
  "curate_nudge_policy": "${CURATE_NUDGE_POLICY}",
  "component_layout": "${COMPONENT}",
  "eval_tp": ${eval_tp},
  "out_suffix": "${out_suffix}",
  "actor_urls": "${actor_urls}",
  "verify_url": "http://127.0.0.1:${ALL_VERIFY_PORT}/v1",
  "corpus_path": "${BCPLUS_CORPUS}",
  "index_path": "${BCPLUS_INDEX}",
  "git_commit": "${GIT_COMMIT}",
  "tool_call_parser": "${tool_parser}",
  "vllm_max_num_seqs": ${VLLM_MAX_NUM_SEQS},
  "worker_stagger_s": ${WORKER_STAGGER},
  "gpu_util": ${GPU_UTIL}
}
EOF
}

lib_bcplus_gpu8_run_zero() {
  lib_bcplus_gpu8_init
  trap lib_bcplus_gpu8_cleanup EXIT

  [[ -d "${MODEL_PATH}" ]] || { log "missing model dir ${MODEL_PATH}"; exit 1; }
  assert_local_retrieval_ready
  ensure_full_corpus

  local actor_url="http://127.0.0.1:${ZERO_ACTOR_PORT}/v1"
  log "RUN_ID=${RUN_ID} model=${MODEL_SLUG} component=zero gpu=${GPU} tp=${TP} actor_port=${ZERO_ACTOR_PORT}"

  start_zero_stack
  run_eval_cell "${actor_url}" "${TP}" "gpu${GPU}"
  lib_bcplus_gpu8_cleanup
  trap - EXIT

  write_cell_manifest "${actor_url}" "gpu${GPU}" "${TP}"
  log "finished zero; output ${REL_OUT}/eval_h1_${BENCHMARK}_${MODEL_SLUG}_zero_gpu${GPU}_${RUN_ID}"
}

lib_bcplus_gpu8_run_all() {
  lib_bcplus_gpu8_init
  trap lib_bcplus_gpu8_cleanup EXIT

  [[ -d "${MODEL_PATH}" ]] || { log "missing model dir ${MODEL_PATH}"; exit 1; }
  assert_local_retrieval_ready
  ensure_full_corpus

  local actor_ports=()
  csv_to_array "${ALL_ACTOR_PORTS}" actor_ports
  local actor_urls
  actor_urls="$(join_urls "${actor_ports[@]}")"
  local out_suffix="gpu${ALL_ACTOR_GPUS//,/}v${ALL_VERIFY_GPU}"

  log "RUN_ID=${RUN_ID} model=${MODEL_SLUG} component=all actors=${ALL_ACTOR_GPUS} verify=${ALL_VERIFY_GPU} eval_tp=${ALL_EVAL_TP}"

  start_all_stack
  run_eval_cell "${actor_urls}" "${ALL_EVAL_TP}" "${out_suffix}"
  lib_bcplus_gpu8_cleanup
  trap - EXIT

  write_cell_manifest "${actor_urls}" "${out_suffix}" "${ALL_EVAL_TP}"
  log "finished all; output ${REL_OUT}/eval_h1_${BENCHMARK}_${MODEL_SLUG}_all_${out_suffix}_${RUN_ID}"
}
