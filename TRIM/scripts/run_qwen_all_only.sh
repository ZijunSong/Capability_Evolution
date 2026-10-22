#!/usr/bin/env bash
# Single-experiment wrapper derived from run_bcplus_full_gpu4_eval.sh.
#
# Runs ONLY the "Qwen3-4B-Instruct-2507 × all" experiment (component=all,
# 10/10 mask): 3 Qwen actor vLLMs in parallel + 1 harness-1 verifier vLLM,
# then run_eval.py upstream_api over the full 830-query bcplus_full split.
#
# The six-experiment parent loops gpt-oss / harness-1 first; this wrapper
# skips them and runs only Qwen-all, on this box's 4 GPUs (indices 0-3,
# since nvidia-smi exposes 0..3 not 1..4).
#
# Requires /opt/TRIM env:  source scripts/setup_bcplus_full_gpu4_env.sh
#   PY=/opt/TRIM/bin/python  VLLM=/opt/TRIM/bin/vllm
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRIM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${TRIM_ROOT}"

# ─── /opt/TRIM env (from setup_bcplus_full_gpu4_env.sh) ─────────────────────
TRIM_PY="${TRIM_PY:-/opt/TRIM/bin/python}"
TRIM_VLLM="${TRIM_VLLM:-/opt/TRIM/bin/vllm}"
[[ -x "${TRIM_PY}" ]] || { echo "[FATAL] /opt/TRIM/bin/python not found; source scripts/setup_bcplus_full_gpu4_env.sh" >&2; exit 1; }
export PY="${TRIM_PY}"
export VLLM="${TRIM_VLLM}"
[[ -x "${VLLM}" ]] || VLLM="${TRIM_PY} -m vllm"

# ─── JVM / Pyserini / tiktoken (same as setup script) ───────────────────────
export JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-21-openjdk-amd64}"
export JVM_PATH="${JVM_PATH:-${JAVA_HOME}/lib/server/libjvm.so}"
ANSERINI_JAR_DIR="${ANSERINI_CLASSPATH:-}"
if [[ -z "${ANSERINI_JAR_DIR}" ]] || ! ls "${ANSERINI_JAR_DIR}"/*.jar >/dev/null 2>&1; then
  ANSERINI_JAR_DIR="$("${TRIM_PY}" - <<'PY' 2>/dev/null || true
import os, glob, pyserini
root = os.path.dirname(pyserini.__file__)
cand = os.path.join(root, "resources", "jars")
print(cand if glob.glob(os.path.join(cand, "*.jar")) else "", end="")
PY
)"
fi
export ANSERINI_CLASSPATH="${ANSERINI_JAR_DIR}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-*******-***********}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:0/v1}"
TIKTOKEN_BUNDLE="${TIKTOKEN_BUNDLE:-/mnt/songzijun/tiktoken-bundle}"
export TIKTOKEN_ENCODINGS_BASE="${TIKTOKEN_ENCODINGS_BASE:-${TIKTOKEN_BUNDLE}/share/tiktoken}"
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-${TIKTOKEN_BUNDLE}/tiktoken_cache}"
export TIKTOKEN_RS_CACHE_DIR="${TIKTOKEN_RS_CACHE_DIR:-${TIKTOKEN_BUNDLE}/tiktoken_rs_cache}"
export BCPLUS_INDEX_FN="${BCPLUS_INDEX_FN:-${TRIM_ROOT}/../SCOPE/external/BrowseComp-Plus/indexes/bm25}"

# ─── run config ────────────────────────────────────────────────────────────
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_qwen_all_only}"
REL_MODELS="../../models"
REL_BCPLUS="../SCOPE/external/BrowseComp-Plus"
REL_OUT="outputs"
REL_LOGS="${REL_OUT}/logs/${RUN_ID}"
REL_SCAPE_EASYOPD="../SCAPE-EasyOPD"

MODEL_PATH="${REL_MODELS}/Qwen3-4B-Instruct-2507"
MODEL_SLUG="Qwen3-4B-Instruct-2507"
API_MODEL="Qwen3-4B-Instruct-2507"
VERIFY_MODEL_PATH="${REL_MODELS}/harness-1"
VERIFY_MODEL_NAME="${VERIFY_MODEL_NAME:-harness-1-verifier}"

BCPLUS_INDEX="${REL_BCPLUS}/indexes/bm25"
BCPLUS_CORPUS="${REL_BCPLUS}/data/browsecomp_plus_corpus_full.jsonl"

# 4 GPUs on this box are indices 0..3 (nvidia-smi). Use 0,1,2 actors + 3 verifier.
ALL_ACTOR_GPUS="${ALL_ACTOR_GPUS:-0,1,2}"
ALL_VERIFY_GPU="${ALL_VERIFY_GPU:-3}"
ALL_ACTOR_PORTS="${ALL_ACTOR_PORTS:-8040,8042,8044}"
ALL_VERIFY_PORT="${ALL_VERIFY_PORT:-8050}"
ALL_TP="${ALL_TP:-3}"               # number of actor vLLMs (= #GPUs for actors)
# Eval replicas (worker processes). MORE replicas than actor vLLMs lets vLLM
# continuous batching serve several concurrent queries per actor. With 3 actor
# URLs and EVAL_REPLICAS=9, each actor vLLM handles 3 concurrent workers
# (identity_for_actor_rank: url = urls[rank % len(urls)]).
EVAL_REPLICAS="${EVAL_REPLICAS:-9}"
GPU_UTIL="${GPU_UTIL:-0.75}"
WORKER_STAGGER="${WORKER_STAGGER:-5}"

export PYTHONPATH=".:${REL_SCAPE_EASYOPD}"
export TRIM_GPU_KEEPALIVE=0

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
  if [[ "${last}" =~ ^([0-9]+)$ ]]; then echo "${BASH_REMATCH[1]}"; return 0; fi
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
  if command -v ss >/dev/null 2>&1; then ss -lptn "sport = :${port}" 2>/dev/null | sed -n 's/.*pid=\([0-9]*\).*/\1/p' || true; return 0; fi
  return 0
}

stop_port() {
  local port="$1" pid_file="${2:-}" pid pids
  if [[ -n "${pid_file}" && -f "${pid_file}" ]]; then
    pid="$(normalize_pid "$(cat "${pid_file}")" 2>/dev/null || true)"
    stop_pid "${pid}"; rm -f "${pid_file}"
  fi
  pids="$(pids_on_port "${port}")"
  if [[ -n "${pids}" ]]; then
    log "stopping stale listener(s) on port ${port}: ${pids//$'\n'/ }"
    # shellcheck disable=SC2086
    kill ${pids} 2>/dev/null || true; sleep 2
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
  log "timeout waiting for API on port ${port}"; return 1
}

wait_port_free() {
  local port="$1" timeout_s="${2:-120}"
  local deadline=$((SECONDS + timeout_s))
  while (( SECONDS < deadline )); do
    curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1 || return 0
    sleep 5
  done
  log "timeout waiting for port ${port} to go idle"; return 1
}

verify_api_model() {
  local port="$1" expected="$2" body ids
  body="$(curl -sf "http://127.0.0.1:${port}/v1/models")" || { log "failed to read /v1/models on port ${port}"; return 1; }
  ids="$("${PY}" -c "import json,sys; d=json.load(sys.stdin); print(','.join(m.get('id','') for m in d.get('data',[])))" <<< "${body}")"
  if [[ ",${ids}," != *",${expected},"* ]]; then log "port ${port} served-model mismatch: expected ${expected}, got [${ids}]"; return 1; fi
  return 0
}

join_urls() {
  local out=(); for p in "$@"; do out+=("http://127.0.0.1:${p}/v1"); done
  local IFS=,; echo "${out[*]}"
}

csv_to_array() {
  local csv="$1"; local -n _arr="$2"; _arr=(); local IFS=,; read -ra _arr <<< "${csv}"
}

vllm_extra_for_model() {
  case "${1}" in
    Qwen3-4B-Instruct-2507) echo "--enable-auto-tool-choice --tool-call-parser hermes --max-model-len 32768 --trust-remote-code" ;;
    *) echo "--enable-auto-tool-choice --tool-call-parser openai --max-model-len 32768" ;;
  esac
}

smoke_test_tool_call() {
  local port="$1" model="$2"
  log "smoke test tool call port=${port} model=${model}"
  "${PY}" - <<PY
import json, sys, urllib.error, urllib.request
port = ${port}
model = ${model@Q}
payload = {
    "model": model,
    "messages": [{"role": "user", "content": "Search the corpus for the phrase smoke test query."}],
    "tools": [{"type": "function", "function": {"name": "search_corpus", "description": "Search corpus",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}],
    "tool_choice": "auto", "max_tokens": 256, "temperature": 0.0,
}
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
    data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
try:
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode("utf-8"))
except urllib.error.HTTPError as exc:
    raise SystemExit(f"smoke test HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:500]}")
message = (body.get("choices") or [{}])[0].get("message") or {}
tool_calls = message.get("tool_calls") or []
if not tool_calls:
    raise SystemExit(f"smoke test missing tool_calls: {json.dumps(message)[:500]}")
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

start_all_stack() {
  stop_all_actors; stop_verify
  local actor_gpus=() actor_ports=()
  csv_to_array "${ALL_ACTOR_GPUS}" actor_gpus
  csv_to_array "${ALL_ACTOR_PORTS}" actor_ports
  if ((${#actor_gpus[@]} != ALL_TP)); then log "ALL_ACTOR_GPUS count (${#actor_gpus[@]}) must match ALL_TP (${ALL_TP})"; exit 1; fi
  if ((${#actor_ports[@]} != ALL_TP)); then log "ALL_ACTOR_PORTS count (${#actor_ports[@]}) must match ALL_TP (${ALL_TP})"; exit 1; fi

  local extra pid i gpu port
  extra="$(vllm_extra_for_model "${MODEL_SLUG}")"
  ACTOR_PIDS=()
  for i in "${!actor_gpus[@]}"; do
    gpu="${actor_gpus[$i]}"; port="${actor_ports[$i]}"
    pid="$(start_vllm_bg "${gpu}" "${port}" "${MODEL_PATH}" "${API_MODEL}" \
      "${extra}" "${REL_LOGS}/actor_${MODEL_SLUG}_all_gpu${gpu}.log")"
    pid="$(normalize_pid "${pid}")"
    ACTOR_PIDS+=("${pid}")
  done

  smoke_test_tool_call "${actor_ports[0]}" "${API_MODEL}"

  VERIFY_PID="$(start_vllm_bg "${ALL_VERIFY_GPU}" "${ALL_VERIFY_PORT}" "${VERIFY_MODEL_PATH}" \
    "${VERIFY_MODEL_NAME}" "--max-model-len 8192 --trust-remote-code --moe-backend triton" \
    "${REL_LOGS}/verify_${MODEL_SLUG}.log")"
  VERIFY_PID="$(normalize_pid "${VERIFY_PID}")"
}

ensure_full_corpus() {
  [[ -f "${BCPLUS_CORPUS}" ]] && return 0
  log "building full corpus → ${BCPLUS_CORPUS}"
  "${PY}" scripts/build_browsecomp_corpus_from_index.py --mode full --out "${BCPLUS_CORPUS}" >> "${REL_LOGS}/corpus_build.log" 2>&1
  local report="${BCPLUS_CORPUS%.jsonl}.BUILD_REPORT.json"
  [[ -f "${report}" ]] || { log "missing build report ${report}"; exit 1; }
  grep -q '"index_corpus_delta": 0' "${report}" || { log "corpus/index mismatch; see ${report}"; exit 1; }
}

run_eval() {
  local actor_urls="$1" tp="$2" out_suffix="$3"
  local out="${REL_OUT}/eval_h1_bcplus_full_${MODEL_SLUG}_all_${out_suffix}_${RUN_ID}"
  local log_path="${REL_LOGS}/eval_${MODEL_SLUG}_all.log"
  log "START model=${MODEL_SLUG} component=all tp=${tp} urls=${actor_urls} out=${out}"
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
    --component all \
    --out "${out}" \
    --verify-base-url "http://127.0.0.1:${ALL_VERIFY_PORT}/v1" \
    --verify-model "${VERIFY_MODEL_NAME}" \
    2>&1 | tee -a "${log_path}" >&2
  log "DONE model=${MODEL_SLUG} component=all out=${out}"
  echo "${out}"
}

main() {
  ensure_full_corpus

  csv_to_array "${ALL_ACTOR_PORTS}" _ALL_PORTS_CHECK
  ALL_ACTOR_URLS="$(join_urls "${_ALL_PORTS_CHECK[@]}")"
  ALL_VERIFY_URL="http://127.0.0.1:${ALL_VERIFY_PORT}/v1"

  [[ -d "${MODEL_PATH}" ]] || { log "missing model dir ${MODEL_PATH}"; exit 1; }
  [[ -d "${VERIFY_MODEL_PATH}" ]] || { log "missing verifier dir ${VERIFY_MODEL_PATH}"; exit 1; }

  log "RUN_ID=${RUN_ID} qwen-all actors=${ALL_ACTOR_GPUS} verify=${ALL_VERIFY_GPU} actors=${ALL_TP} replicas=${EVAL_REPLICAS} gpu_util=${GPU_UTIL}"

  start_all_stack
  local out_dir
  out_dir="$(run_eval "${ALL_ACTOR_URLS}" "${EVAL_REPLICAS}" "gpu${ALL_ACTOR_GPUS//,/}v${ALL_VERIFY_GPU}")"

  stop_all_actors; stop_verify

  GIT_COMMIT="$(git -C . rev-parse HEAD 2>/dev/null || echo unknown)"
  cat > "${REL_LOGS}/MANIFEST.json" <<EOF
{
  "run_id": "${RUN_ID}",
  "experiment": "qwen_all_only",
  "benchmark": "bcplus_full",
  "evaluation_path": "upstream_api",
  "model": "${MODEL_PATH}",
  "api_model": "${API_MODEL}",
  "component": "all",
  "component_preset": "all",
  "tool_call_parser": "hermes",
  "all": {
    "actor_gpus": "${ALL_ACTOR_GPUS}",
    "actor_ports": "${ALL_ACTOR_PORTS}",
    "verify_gpu": "${ALL_VERIFY_GPU}",
    "verify_port": ${ALL_VERIFY_PORT},
    "verify_model_path": "${VERIFY_MODEL_PATH}",
    "verify_model_name": "${VERIFY_MODEL_NAME}",
    "tp": ${ALL_TP},
    "eval_replicas": ${EVAL_REPLICAS},
    "worker_stagger_s": ${WORKER_STAGGER}
  },
  "actor_urls": "${ALL_ACTOR_URLS}",
  "verify_url": "${ALL_VERIFY_URL}",
  "corpus_path": "${BCPLUS_CORPUS}",
  "index_path": "${BCPLUS_INDEX}",
  "out_dir": "${out_dir}",
  "git_commit": "${GIT_COMMIT}",
  "python": "${TRIM_PY}",
  "vllm": "${TRIM_VLLM}"
}
EOF

  log "finished qwen-all; output: ${out_dir}"
  log "manifest: ${REL_LOGS}/MANIFEST.json"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
