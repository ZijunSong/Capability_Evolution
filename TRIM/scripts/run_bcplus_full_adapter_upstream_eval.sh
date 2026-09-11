#!/usr/bin/env bash
# =============================================================================
# BC+ full (830 queries) — high-concurrency upstream_api eval for trained adapters
#
# Official Harness-1 path: original SlidingWindowSearchEnv + chat/completions
# (message) API. Each eval worker talks to a local vLLM actor that serves
# base_model + LoRA adapter.
#
# Retrieval: local_bm25 (Lucene index + jsonl corpus) — no Chroma Cloud/local service.
# Verifier (COMPONENT=all only): always ../../models/harness-1 as harness-1-verifier,
# independent of BASE_MODEL / LoRA adapter. Do not swap per trained checkpoint.
#
# -----------------------------------------------------------------------------
# Quick start (8×GPU server, multi-actor layout — recommended)
# -----------------------------------------------------------------------------
#
#   ADAPTER=/path/to/train_out/adapters/scape_seed \
#   BASE_MODEL=/mnt/songzijun/models/openai/gpt-oss-20b \
#   bash TRIM/scripts/run_bcplus_full_adapter_upstream_eval.sh
#
# Or discover adapter from a training --out directory:
#
#   RUN_DIR=/path/to/train_out \
#   ADAPTER_CELL=scape_seed \
#   BASE_MODEL=/mnt/songzijun/models/openai/gpt-oss-20b \
#   bash TRIM/scripts/run_bcplus_full_adapter_upstream_eval.sh
#
# Equivalent to your legacy one-liner, but message/API based with high concurrency:
#
#   ADAPTER=/mnt/songzijun/Capability_Evolution/TRIM/outputs/.../adapters/scape_seed \
#   BASE_MODEL=/mnt/songzijun/models/openai/gpt-oss-20b \
#   COMPONENT=zero \
#   LAYOUT=multi \
#   ACTOR_GPUS=0,1,2,3,4,5,6,7 \
#   EVAL_TP=32 \
#   bash TRIM/scripts/run_bcplus_full_adapter_upstream_eval.sh
#
# -----------------------------------------------------------------------------
# Single-GPU / small machine (one vLLM actor, many eval workers)
# -----------------------------------------------------------------------------
#
#   ADAPTER=/path/to/adapter \
#   BASE_MODEL=/path/to/gpt-oss-20b \
#   LAYOUT=single \
#   ACTOR_GPU=0 \
#   EVAL_TP=24 \
#   bash TRIM/scripts/run_bcplus_full_adapter_upstream_eval.sh
#
# -----------------------------------------------------------------------------
# Component=all (needs harness-1 verifier on VERIFY_GPU)
# -----------------------------------------------------------------------------
#
#   ADAPTER=/path/to/adapter \
#   BASE_MODEL=/path/to/gpt-oss-20b \
#   COMPONENT=all \
#   ACTOR_GPUS=0,1,2,3,4,5,6 \
#   VERIFY_GPU=7 \
#   bash TRIM/scripts/run_bcplus_full_adapter_upstream_eval.sh
#
# -----------------------------------------------------------------------------
# Smoke test (6 queries)
# -----------------------------------------------------------------------------
#
#   ADAPTER=... BASE_MODEL=... SMOKE=1 bash TRIM/scripts/run_bcplus_full_adapter_upstream_eval.sh
#
# -----------------------------------------------------------------------------
# Key env vars
# -----------------------------------------------------------------------------
#   ADAPTER          LoRA directory (adapter_config.json + adapter_model.safetensors)
#   RUN_DIR          Training out dir; adapter auto-discovered when ADAPTER unset
#   ADAPTER_CELL     Cell name under RUN_DIR (default: scape_seed)
#   BASE_MODEL       HF base checkpoint path (default: read from adapter_config.json)
#   COMPONENT        zero | all  (default: zero)
#   LAYOUT           multi | single  (default: multi)
#   ACTOR_GPUS       Comma GPU ids for actor vLLMs in multi layout
#   ACTOR_GPU        Single actor GPU in single layout (default: 0)
#   ACTOR_PORTS      Comma ports, one per actor (default: 8040,8042,...)
#   BENCHMARK        bcplus_full | bcplus_test_50 (default: bcplus_full)
#   VERIFY_GPU       GPU for harness-1 verifier when COMPONENT=all (default: 7)
#   VERIFY_PORT      Verifier port (default: 8050; must not appear in ACTOR_PORTS)
#   VERIFY_MODEL_PATH  Verifier weights (default: ../../models/harness-1)
#   VERIFY_MODEL_NAME  vLLM served name (default: harness-1-verifier)
#   EVAL_TP          Parallel eval worker count (default: 32 multi, 24 single)
#   LORA_NAME        vLLM LoRA module id / --api-model (default: trained-policy)
#   MAX_LORA_RANK    vLLM --max-lora-rank (default: read from adapter_config.json)
#   VLLM_MAX_NUM_SEQS  vLLM concurrent sequences (default: 64)
#   WORKER_STAGGER   Seconds between eval worker launches (default: 2)
#   RUN_ID           Output suffix (default: timestamp)
#   PY, VLLM         Python / vLLM binaries
#   SMOKE=1          Run 6-query smoke instead of full 830
#
# Outputs: TRIM/outputs/eval_h1_<benchmark>_adapter_<component>_<layout>_<RUN_ID>/
# Logs:    TRIM/outputs/logs/<RUN_ID>/
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRIM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${TRIM_ROOT}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_bcplus_adapter_upstream}"
PY="${PY:-python3}"
VLLM="${VLLM:-vllm}"
BENCHMARK="${BENCHMARK:-bcplus_full}"

REL_BCPLUS="../SCOPE/external/BrowseComp-Plus"
REL_OUT="outputs"
REL_LOGS="${REL_OUT}/logs/${RUN_ID}"
REL_SCAPE_EASYOPD="../SCAPE-EasyOPD"
REL_MODELS="../../models"

BCPLUS_INDEX="${REL_BCPLUS}/indexes/bm25"
BCPLUS_CORPUS="${REL_BCPLUS}/data/browsecomp_plus_corpus_full.jsonl"
VERIFY_MODEL_PATH="${REL_MODELS}/harness-1"
VERIFY_MODEL_NAME="${VERIFY_MODEL_NAME:-harness-1-verifier}"

COMPONENT="${COMPONENT:-zero}"
LAYOUT="${LAYOUT:-multi}"
LORA_NAME="${LORA_NAME:-trained-policy}"
ADAPTER_CELL="${ADAPTER_CELL:-scape_seed}"
GPU_UTIL="${GPU_UTIL:-0.75}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}"
WORKER_STAGGER="${WORKER_STAGGER:-2}"
SMOKE="${SMOKE:-0}"

# multi: one actor vLLM per GPU; single: one actor, many eval workers
ACTOR_GPUS="${ACTOR_GPUS:-0,1,2,3,4,5,6,7}"
ACTOR_GPU="${ACTOR_GPU:-0}"
# 8050 is reserved for the harness-1 verifier (COMPONENT=all); do not assign to actors.
ACTOR_PORTS="${ACTOR_PORTS:-8040,8042,8044,8046,8048,8052,8054,8056}"
VERIFY_GPU="${VERIFY_GPU:-7}"
VERIFY_PORT="${VERIFY_PORT:-8050}"

export PYTHONPATH=".:${REL_SCAPE_EASYOPD}"
export TRIM_GPU_KEEPALIVE=0
# Hard-disable Chroma client construction in harness-1 (local_bm25 path only).
export HARNESS1_FORBID_CHROMA=1

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
  body="$(curl -sf "http://127.0.0.1:${port}/v1/models")" || { log "failed /v1/models port ${port}"; return 1; }
  ids="$("${PY}" -c "import json,sys; d=json.load(sys.stdin); print(','.join(m.get('id','') for m in d.get('data',[])))" <<< "${body}")"
  if [[ ",${ids}," != *",${expected},"* ]]; then
    log "port ${port} model mismatch: expected ${expected}, got [${ids}]"
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

resolve_paths() {
  if [[ -z "${ADAPTER:-}" && -n "${RUN_DIR:-}" ]]; then
    ADAPTER="$("${PY}" - <<PY
import sys
from pathlib import Path
sys.path.insert(0, ".")
from trim.cli.launch import discover_adapter_map
run_dir = Path(${RUN_DIR@Q})
cell = ${ADAPTER_CELL@Q}
found = discover_adapter_map(run_dir)
path = found.get(cell) or found.get("eval") or (next(iter(found.values())) if found else None)
if not path:
    raise SystemExit(f"no adapter under RUN_DIR={run_dir} (tried cell={cell!r}, keys={list(found)})")
print(path)
PY
)"
    log "discovered ADAPTER=${ADAPTER} from RUN_DIR=${RUN_DIR} cell=${ADAPTER_CELL}"
  fi

  if [[ -z "${ADAPTER:-}" ]]; then
    log "ADAPTER or RUN_DIR is required"
    exit 1
  fi
  ADAPTER="$(cd "$(dirname "${ADAPTER}")" && pwd)/$(basename "${ADAPTER}")"
  [[ -f "${ADAPTER}/adapter_config.json" ]] || { log "missing ${ADAPTER}/adapter_config.json"; exit 1; }
  [[ -f "${ADAPTER}/adapter_model.safetensors" ]] || { log "missing ${ADAPTER}/adapter_model.safetensors"; exit 1; }

  if [[ -z "${BASE_MODEL:-}" || -z "${MAX_LORA_RANK:-}" ]]; then
    read -r _cfg_base _cfg_rank <<< "$("${PY}" - <<PY
import json
from pathlib import Path
cfg = json.loads(Path(${ADAPTER@Q}).joinpath("adapter_config.json").read_text(encoding="utf-8"))
print(cfg.get("base_model_name_or_path") or "", int(cfg.get("r") or 64))
PY
)"
    if [[ -z "${BASE_MODEL:-}" ]]; then
      BASE_MODEL="${_cfg_base}"
      log "BASE_MODEL from adapter_config.json → ${BASE_MODEL}"
    fi
    if [[ -z "${MAX_LORA_RANK:-}" ]]; then
      MAX_LORA_RANK="${_cfg_rank}"
    fi
  fi
  [[ -d "${BASE_MODEL}" ]] || { log "missing BASE_MODEL dir ${BASE_MODEL}"; exit 1; }
}

vllm_extra_for_base() {
  local base="$1"
  case "${base}" in
    *Qwen3-4B*|*qwen3-4b*)
      echo "--enable-auto-tool-choice --tool-call-parser hermes --max-model-len 32768 --trust-remote-code"
      ;;
    *gpt-oss*|*harness-1*)
      echo "--enable-auto-tool-choice --tool-call-parser openai --max-model-len 32768 --trust-remote-code --moe-backend triton"
      ;;
    *)
      echo "--enable-auto-tool-choice --tool-call-parser openai --max-model-len 32768 --trust-remote-code"
      ;;
  esac
}

lora_vllm_flags() {
  echo "--enable-lora --lora-modules ${LORA_NAME}=${ADAPTER} --max-lora-rank ${MAX_LORA_RANK}"
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
print("smoke_ok", tool_calls[0]["function"]["name"])
PY
}

start_lora_vllm_bg() {
  local gpu="$1" port="$2" log_path="$3"
  local model_extra lora_extra
  model_extra="$(vllm_extra_for_base "${BASE_MODEL}")"
  lora_extra="$(lora_vllm_flags)"

  stop_port "${port}" "${log_path}.pid"
  wait_port_free "${port}" 120

  log "starting vLLM+LoRA GPU${gpu} port=${port} base=${BASE_MODEL} lora=${ADAPTER} served=${LORA_NAME}"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="${gpu}" nohup "${VLLM}" serve "${BASE_MODEL}" \
    --host 127.0.0.1 --port "${port}" \
    --served-model-name "${LORA_NAME}" \
    ${model_extra} ${lora_extra} \
    --gpu-memory-utilization "${GPU_UTIL}" --max-num-seqs "${VLLM_MAX_NUM_SEQS}" --enforce-eager \
    > "${log_path}" 2>&1 &
  local pid=$!
  echo "${pid}" > "${log_path}.pid"
  wait_api "${port}"
  verify_api_model "${port}" "${LORA_NAME}"
  echo "${pid}"
}

start_verify_vllm() {
  [[ -d "${VERIFY_MODEL_PATH}" ]] || {
    log "missing verifier weights ${VERIFY_MODEL_PATH} (local eval always uses harness-1)"
    exit 1
  }
  stop_verify
  log "starting verifier GPU${VERIFY_GPU} port=${VERIFY_PORT} model=${VERIFY_MODEL_PATH} served=${VERIFY_MODEL_NAME}"
  stop_port "${VERIFY_PORT}" "${REL_LOGS}/verify.log.pid"
  wait_port_free "${VERIFY_PORT}" 120
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="${VERIFY_GPU}" nohup "${VLLM}" serve "${VERIFY_MODEL_PATH}" \
    --host 127.0.0.1 --port "${VERIFY_PORT}" \
    --served-model-name "${VERIFY_MODEL_NAME}" \
    --max-model-len 8192 --trust-remote-code --moe-backend triton \
    --gpu-memory-utilization "${GPU_UTIL}" --enforce-eager \
    > "${REL_LOGS}/verify.log" 2>&1 &
  VERIFY_PID="$(normalize_pid "$!")"
  echo "${VERIFY_PID}" > "${REL_LOGS}/verify.log.pid"
  wait_api "${VERIFY_PORT}"
  verify_api_model "${VERIFY_PORT}" "${VERIFY_MODEL_NAME}"
}

start_actors() {
  stop_all_actors
  ACTOR_PIDS=()
  local actor_gpus=() actor_ports=()
  if [[ "${LAYOUT}" == "single" ]]; then
    actor_gpus=("${ACTOR_GPU}")
    actor_ports=("${ACTOR_PORTS%%,*}")
    if [[ -z "${actor_ports[0]}" ]]; then actor_ports=(8040); fi
  else
    csv_to_array "${ACTOR_GPUS}" actor_gpus
    csv_to_array "${ACTOR_PORTS}" actor_ports
    if ((${#actor_gpus[@]} != ${#actor_ports[@]})); then
      log "ACTOR_GPUS count (${#actor_gpus[@]}) must match ACTOR_PORTS (${#actor_ports[@]})"
      exit 1
    fi
  fi

  local i gpu port pid
  for i in "${!actor_gpus[@]}"; do
    gpu="${actor_gpus[$i]}"
    port="${actor_ports[$i]}"
    pid="$(start_lora_vllm_bg "${gpu}" "${port}" "${REL_LOGS}/actor_gpu${gpu}_p${port}.log")"
    ACTOR_PIDS+=("$(normalize_pid "${pid}")")
  done

  smoke_test_tool_call "${actor_ports[0]}" "${LORA_NAME}"

  if [[ "${COMPONENT}" == "all" ]]; then
    local gpu port
    for gpu in "${actor_gpus[@]}"; do
      if [[ "${gpu}" == "${VERIFY_GPU}" ]]; then
        log "ACTOR_GPUS includes VERIFY_GPU=${VERIFY_GPU}; use disjoint GPUs for COMPONENT=all"
        exit 1
      fi
    done
    for port in "${actor_ports[@]}"; do
      if [[ "${port}" == "${VERIFY_PORT}" ]]; then
        log "ACTOR_PORTS includes VERIFY_PORT=${VERIFY_PORT}; pick a different verifier port"
        exit 1
      fi
    done
    start_verify_vllm
  fi

  ACTOR_URLS="$(join_urls "${actor_ports[@]}")"
  N_ACTORS="${#actor_ports[@]}"
}

assert_local_retrieval_ready() {
  [[ -d "${BCPLUS_INDEX}" ]] || {
    log "missing Lucene index ${BCPLUS_INDEX} (local_bm25; no Chroma)"
    exit 1
  }
  if [[ ! -f "${BCPLUS_CORPUS}" ]]; then
    log "corpus missing ${BCPLUS_CORPUS}; will build from index"
    return 0
  fi
  log "local_bm25 ready index=${BCPLUS_INDEX} corpus=${BCPLUS_CORPUS}"
}

ensure_full_corpus() {
  [[ -f "${BCPLUS_CORPUS}" ]] && return 0
  log "building full corpus → ${BCPLUS_CORPUS}"
  "${PY}" scripts/build_browsecomp_corpus_from_index.py \
    --mode full --out "${BCPLUS_CORPUS}" >> "${REL_LOGS}/corpus_build.log" 2>&1
  local report="${BCPLUS_CORPUS%.jsonl}.BUILD_REPORT.json"
  [[ -f "${report}" ]] || { log "missing build report ${report}"; exit 1; }
  grep -q '"index_corpus_delta": 0' "${report}" || { log "corpus/index mismatch; see ${report}"; exit 1; }
}

run_eval() {
  local eval_tp="$1"
  local out="${REL_OUT}/eval_h1_${BENCHMARK}_adapter_${COMPONENT}_${LAYOUT}_${RUN_ID}"
  local log_path="${REL_LOGS}/eval_${COMPONENT}.log"
  local extra=()

  if [[ "${COMPONENT}" == "all" ]]; then
    extra+=(--verify-base-url "http://127.0.0.1:${VERIFY_PORT}/v1")
    extra+=(--verify-model "${VERIFY_MODEL_NAME}")
  fi
  if [[ "${SMOKE}" == "1" ]]; then
    extra+=(--smoke)
  fi

  log "START component=${COMPONENT} layout=${LAYOUT} eval_tp=${eval_tp} urls=${ACTOR_URLS} adapter=${ADAPTER} out=${out}"
  local rc=0
  "${PY}" scripts/run_eval.py \
    --harness Harness-1 \
    --benchmark "${BENCHMARK}" \
    --model_name "${BASE_MODEL}" \
    --evaluation-path upstream_api \
    --retrieval-backend local_bm25 \
    --api-base-url "${ACTOR_URLS}" \
    --api-model "${LORA_NAME}" \
    --adapter "${ADAPTER}" \
    --adapter-export lora \
    --eval-mode adapter \
    --index-path "${BCPLUS_INDEX}" \
    --corpus-path "${BCPLUS_CORPUS}" \
    --reranker none \
    --offline \
    --max-turns "${MAX_TURNS:-40}" \
    --max-new-tokens "${MAX_NEW_TOKENS:-2048}" \
    --temperature "${TEMPERATURE:-1.0}" \
    --tp "${eval_tp}" \
    --eval-stagger-s "${WORKER_STAGGER}" \
    --component "${COMPONENT}" \
    --out "${out}" \
    "${extra[@]}" \
    2>&1 | tee -a "${log_path}" >&2 || rc=$?

  if [[ "${rc}" -ne 0 ]]; then
    log "WARN eval exited ${rc} out=${out}"
    return "${rc}"
  fi
  log "DONE out=${out}"
  EVAL_OUT="${out}"
  return 0
}

main() {
  mkdir -p "${REL_LOGS}"
  resolve_paths
  assert_local_retrieval_ready
  ensure_full_corpus

  if [[ "${LAYOUT}" == "single" ]]; then
    EVAL_TP="${EVAL_TP:-24}"
  else
    EVAL_TP="${EVAL_TP:-32}"
  fi

  log "RUN_ID=${RUN_ID} benchmark=${BENCHMARK} component=${COMPONENT} layout=${LAYOUT} base=${BASE_MODEL} adapter=${ADAPTER} eval_tp=${EVAL_TP} lora_name=${LORA_NAME} max_lora_rank=${MAX_LORA_RANK} verify=${VERIFY_MODEL_PATH}:${VERIFY_MODEL_NAME}"

  start_actors
  run_eval "${EVAL_TP}" || log "eval returned non-zero; see ${REL_LOGS}"

  stop_all_actors
  stop_verify

  GIT_COMMIT="$(git -C . rev-parse HEAD 2>/dev/null || echo unknown)"
  cat > "${REL_LOGS}/MANIFEST.json" <<EOF
{
  "run_id": "${RUN_ID}",
  "benchmark": "${BENCHMARK}",
  "evaluation_path": "upstream_api",
  "eval_profile": "upstream_core_local_bm25",
  "retrieval_backend": "local_bm25",
  "chroma_required": false,
  "eval_mode": "adapter",
  "component": "${COMPONENT}",
  "layout": "${LAYOUT}",
  "base_model": "${BASE_MODEL}",
  "adapter": "${ADAPTER}",
  "lora_name": "${LORA_NAME}",
  "max_lora_rank": ${MAX_LORA_RANK},
  "eval_tp": ${EVAL_TP},
  "n_actors": ${N_ACTORS:-1},
  "actor_urls": "${ACTOR_URLS}",
  "verify_gpu": "${VERIFY_GPU}",
  "verify_port": ${VERIFY_PORT},
  "verify_model_path": "${VERIFY_MODEL_PATH}",
  "verify_model_name": "${VERIFY_MODEL_NAME}",
  "verify_url": "http://127.0.0.1:${VERIFY_PORT}/v1",
  "vllm_max_num_seqs": ${VLLM_MAX_NUM_SEQS},
  "worker_stagger_s": ${WORKER_STAGGER},
  "corpus_path": "${BCPLUS_CORPUS}",
  "index_path": "${BCPLUS_INDEX}",
  "git_commit": "${GIT_COMMIT}",
  "eval_out": "${EVAL_OUT:-}"
}
EOF
  log "finished; results ${EVAL_OUT:-unknown}; manifest ${REL_LOGS}/MANIFEST.json"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
