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
  # Tolerate legacy captures where log lines were mixed into stdout.
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
  return 0
}

stop_port() {
  local port="$1"
  local pid_file="${2:-}"
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
  local slug="$1"
  case "${slug}" in
    gpt-oss-20b|harness-1)
      echo "--enable-auto-tool-choice --tool-call-parser openai --max-model-len 32768 --trust-remote-code --moe-backend triton"
      ;;
    Qwen3-4B-Instruct-2507)
      echo "--enable-auto-tool-choice --tool-call-parser hermes --max-model-len 32768 --trust-remote-code"
      ;;
    *)
      echo "--enable-auto-tool-choice --tool-call-parser openai --max-model-len 32768"
      ;;
  esac
}

smoke_test_tool_call() {
  local port="$1" model="$2"
  log "smoke test tool call port=${port} model=${model}"
  "${PY}" - <<PY
import json
import sys
import urllib.error
import urllib.request

port = ${port}
model = ${model@Q}
payload = {
    "model": model,
    "messages": [{"role": "user", "content": "Search the corpus for the phrase smoke test query."}],
    "tools": [{
        "type": "function",
        "function": {
            "name": "search_corpus",
            "description": "Search corpus",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }],
    "tool_choice": "auto",
    "max_tokens": 256,
    "temperature": 0.0,
}
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
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

start_zero_stack() {
  local model_path="$1" api_model="$2" model_slug="$3"
  stop_all_actors
  stop_verify

  local extra pid
  extra="$(vllm_extra_for_model "${model_slug}")"
  pid="$(start_vllm_bg "${GPU}" "${ZERO_ACTOR_PORT}" "${model_path}" "${api_model}" \
    "${extra}" "${REL_LOGS}/actor_${model_slug}_zero.log")"
  pid="$(normalize_pid "${pid}")"
  ACTOR_PIDS=("${pid}")

  if [[ "${model_slug}" == "Qwen3-4B-Instruct-2507" ]]; then
    smoke_test_tool_call "${ZERO_ACTOR_PORT}" "${api_model}"
  fi
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
    pid="$(normalize_pid "${pid}")"
    ACTOR_PIDS+=("${pid}")
  done

  if [[ "${model_slug}" == "Qwen3-4B-Instruct-2507" ]]; then
    smoke_test_tool_call "${actor_ports[0]}" "${api_model}"
  fi

  VERIFY_PID="$(start_vllm_bg "${ALL_VERIFY_GPU}" "${ALL_VERIFY_PORT}" "${VERIFY_MODEL_PATH}" \
    "${VERIFY_MODEL_NAME}" "--max-model-len 8192 --trust-remote-code --moe-backend triton" \
    "${REL_LOGS}/verify_${model_slug}.log")"
  VERIFY_PID="$(normalize_pid "${VERIFY_PID}")"
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
  local rc=0
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
    2>&1 | tee -a "${log_path}" >&2 || rc=$?
  if [[ "${rc}" -ne 0 ]]; then
    log "WARN eval exited ${rc} model=${model_slug} component=${component} out=${out}"
    return "${rc}"
  fi
  log "DONE model=${model_slug} component=${component}"
  return 0
}

main() {
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
    if ! run_eval "${model_path}" "${model_slug}" "${api_model}" "zero" \
      "${ZERO_ACTOR_URL}" "${TP}" "gpu${GPU}"; then
      log "continuing after zero eval failure for ${model_slug}"
    fi

    start_all_stack "${model_path}" "${api_model}" "${model_slug}"
    if ! run_eval "${model_path}" "${model_slug}" "${api_model}" "all" \
      "${ALL_ACTOR_URLS}" "${ALL_TP}" "gpu${ALL_ACTOR_GPUS//,/}v${ALL_VERIFY_GPU}"; then
      log "continuing after all eval failure for ${model_slug}"
    fi
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
    {"model": "${REL_MODELS}/Qwen3-4B-Instruct-2507", "component": "zero", "layout": "1gpu", "tool_call_parser": "hermes"},
    {"model": "${REL_MODELS}/Qwen3-4B-Instruct-2507", "component": "all", "layout": "3actor+1verify", "tool_call_parser": "hermes"}
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
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
