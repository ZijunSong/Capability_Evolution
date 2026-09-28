#!/usr/bin/env bash
# Transfer full eval: 3 models × 3 benchmarks × {zero, all} = 18 sequential jobs.
#
# Models:
#   gpt-oss-20b
#   harness-1
#   Qwen3-4B-Instruct-2507
# Benchmarks (local BM25, no Chroma):
#   longsealqa  (254)
#   frames      (824)
#   hotpotqa    (493)
#
# Order (actor stack reused across the three benchmarks of one model×component):
#   for model in gpt-oss-20b, harness-1, Qwen3-4B-Instruct-2507:
#     zero: start actor → longsealqa → frames → hotpotqa → stop
#     all : start actors+verifier → longsealqa → frames → hotpotqa → stop
#
# Default 8×GPU layout (same as BC+ Harness-1 GPU8):
#   zero : 1 actor on GPU 7, port 8040
#          override with ZERO_ACTOR_GPUS + ZERO_ACTOR_PORTS for multi-replica
#   all  : 6 actors on GPU 0–5 (8042…8054) + harness-1 verifier on GPU 6:8050
#
# Usage:
#   bash TRIM/scripts/run_transfer_w_harness1_gpu8_eval.sh
#
# Optional env:
#   RUN_ID, MODELS, COMPONENTS, BENCHMARKS, N_EVAL, DRY_RUN=1, FAIL_FAST=1
#   SKIP_EXISTING=1, SKIP_ZERO=1, SKIP_ALL=1, SKIP_JOBS=model/component/bench,...
#   REMAIN_JOB=model/component/bench REMAIN_QUERY_IDS_FILE=path
#   GPU, TP, ZERO_ACTOR_PORT, ZERO_ACTOR_GPUS, ZERO_ACTOR_PORTS,
#   ALL_ACTOR_GPUS, ALL_ACTOR_PORTS, ALL_VERIFY_GPU,
#   ALL_VERIFY_PORT, ALL_TP, ALL_EVAL_TP, PY, VLLM, GPU_UTIL, VLLM_MAX_NUM_SEQS,
#   MAX_TURNS, MAX_NEW_TOKENS, MAX_NEW_TOKENS_HARMONY, REASONING_EFFORT,
#   TEMPERATURE, WORKER_STAGGER, FORCE_UNSHARE_SHM=1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRIM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${TRIM_ROOT}"

if [[ -z "${PY:-}" ]]; then
  if [[ -x /data/ppnm/miniconda3/envs/bishop/bin/python ]]; then
    PY=/data/ppnm/miniconda3/envs/bishop/bin/python
  else
    PY=python
  fi
fi
if [[ -z "${VLLM:-}" ]]; then
  if [[ -x /data/ppnm/miniconda3/envs/bishop/bin/vllm ]]; then
    VLLM=/data/ppnm/miniconda3/envs/bishop/bin/vllm
  else
    VLLM=vllm
  fi
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_transfer_full_gpu8}"
REL_MODELS="${REL_MODELS:-../../models}"
REL_OUT="${REL_OUT:-outputs}"
REL_LOGS="${REL_OUT}/logs/${RUN_ID}"
REL_SCAPE_EASYOPD="${REL_SCAPE_EASYOPD:-../SCAPE-EasyOPD}"
TRANSFER_ROOT="${TRANSFER_ROOT:-manifests/transfer_local}"
VERIFY_MODEL_PATH="${VERIFY_MODEL_PATH:-${REL_MODELS}/harness-1}"
VERIFY_MODEL_NAME="${VERIFY_MODEL_NAME:-harness-1-verifier}"

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
ALL_BENCHMARKS=(longsealqa frames hotpotqa)
ALL_COMPONENTS=(zero all)

GPU="${GPU:-7}"
TP="${TP:-64}"
ZERO_ACTOR_PORT="${ZERO_ACTOR_PORT:-8040}"
ZERO_ACTOR_GPUS="${ZERO_ACTOR_GPUS:-${GPU}}"
ZERO_ACTOR_PORTS="${ZERO_ACTOR_PORTS:-${ZERO_ACTOR_PORT}}"

ALL_ACTOR_GPUS="${ALL_ACTOR_GPUS:-0,1,2,3,4,5}"
ALL_ACTOR_PORTS="${ALL_ACTOR_PORTS:-8042,8044,8046,8048,8052,8054}"
ALL_TP="${ALL_TP:-6}"
ALL_VERIFY_GPU="${ALL_VERIFY_GPU:-6}"
ALL_VERIFY_PORT="${ALL_VERIFY_PORT:-8050}"
ALL_EVAL_TP="${ALL_EVAL_TP:-64}"

GPU_UTIL="${GPU_UTIL:-0.70}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-128}"
WORKER_STAGGER="${WORKER_STAGGER:-0}"
MAX_TURNS="${MAX_TURNS:-40}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
MAX_NEW_TOKENS_HARMONY="${MAX_NEW_TOKENS_HARMONY:-4096}"
REASONING_EFFORT="${REASONING_EFFORT:-medium}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
DRY_RUN="${DRY_RUN:-0}"
FAIL_FAST="${FAIL_FAST:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
SKIP_ZERO="${SKIP_ZERO:-0}"
SKIP_ALL="${SKIP_ALL:-0}"
SKIP_JOBS="${SKIP_JOBS:-}"
REMAIN_JOB="${REMAIN_JOB:-}"
REMAIN_QUERY_IDS_FILE="${REMAIN_QUERY_IDS_FILE:-}"
FORCE_UNSHARE_SHM="${FORCE_UNSHARE_SHM:-0}"

export PYTHONPATH=".:${REL_SCAPE_EASYOPD}"
export TRIM_GPU_KEEPALIVE=0
export HARNESS1_FORBID_CHROMA=1
export CURATE_NUDGE_POLICY="${CURATE_NUDGE_POLICY:-legacy}"

mkdir -p "${REL_LOGS}"
ACTOR_PIDS=()
VERIFY_PID=""
MASTER_LOG="${REL_LOGS}/master.log"
FAILED_JOBS=()
DONE_JOBS=()
SKIPPED_JOBS=()

log() {
  echo "[$(date -Is)] $*" | tee -a "${MASTER_LOG}" >&2
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

in_csv() {
  local needle="${1,,}" haystack="${2:-}"
  [[ -z "${haystack}" ]] && return 0
  local IFS=, item
  for item in ${haystack}; do
    [[ "${item,,}" == "${needle}" ]] && return 0
  done
  return 1
}

# shellcheck source=scripts/vllm_model_extra.sh
source "${SCRIPT_DIR}/vllm_model_extra.sh"

is_harmony_slug() {
  local slug="$1"
  case "${slug}" in
    gpt-oss-20b|harness-1|gpt-oss*|harness1*) return 0 ;;
    *) return 1 ;;
  esac
}

need_unshare_shm() {
  [[ "${FORCE_UNSHARE_SHM}" == "1" ]] && return 0
  [[ -w /dev/shm ]] && return 1
  return 0
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
print("smoke_ok", (tool_calls[0].get("function") or {}).get("name"))
PY
}

start_vllm_bg() {
  local gpu="$1" port="$2" model_path="$3" served_name="$4" extra="$5" log_path="$6"
  stop_port "${port}" "${log_path}.pid"
  wait_port_free "${port}" 120
  log "starting vLLM GPU${gpu} port ${port} model=${model_path} served=${served_name}"
  if need_unshare_shm; then
    log "wrapping vLLM with unshare tmpfs /dev/shm (not writable or FORCE_UNSHARE_SHM=1)"
    # shellcheck disable=SC2086
    nohup unshare --user --map-root-user --mount bash -c "
      mount -t tmpfs -o size=16G tmpfs /dev/shm
      chmod 1777 /dev/shm
      export CUDA_VISIBLE_DEVICES='${gpu}'
      exec '${VLLM}' serve '${model_path}' \
        --host 127.0.0.1 --port '${port}' --served-model-name '${served_name}' \
        ${extra} --gpu-memory-utilization '${GPU_UTIL}' --max-num-seqs '${VLLM_MAX_NUM_SEQS}' --enforce-eager
    " > "${log_path}" 2>&1 &
  else
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES="${gpu}" nohup "${VLLM}" serve "${model_path}" \
      --host 127.0.0.1 --port "${port}" --served-model-name "${served_name}" \
      ${extra} --gpu-memory-utilization "${GPU_UTIL}" --max-num-seqs "${VLLM_MAX_NUM_SEQS}" --enforce-eager \
      > "${log_path}" 2>&1 &
  fi
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
  local actor_gpus=() actor_ports=()
  csv_to_array "${ZERO_ACTOR_GPUS}" actor_gpus
  csv_to_array "${ZERO_ACTOR_PORTS}" actor_ports
  ((${#actor_gpus[@]} > 0)) || { log "ZERO_ACTOR_GPUS is empty"; exit 1; }
  ((${#actor_gpus[@]} == ${#actor_ports[@]})) || { log "ZERO_ACTOR_GPUS count != ZERO_ACTOR_PORTS"; exit 1; }
  local extra pid i gpu port
  extra="$(vllm_extra_for_model "${model_slug}") ${VLLM_EXTRA:-}"
  ACTOR_PIDS=()
  for i in "${!actor_gpus[@]}"; do
    gpu="${actor_gpus[$i]}"
    port="${actor_ports[$i]}"
    pid="$(start_vllm_bg "${gpu}" "${port}" "${model_path}" "${api_model}" \
      "${extra}" "${REL_LOGS}/actor_${model_slug}_zero_gpu${gpu}.log")"
    ACTOR_PIDS+=("$(normalize_pid "${pid}")")
  done
  if model_needs_smoke_test "${model_slug}"; then
    smoke_test_tool_call "${actor_ports[0]}" "${api_model}"
  fi
}

start_all_stack() {
  local model_path="$1" api_model="$2" model_slug="$3"
  stop_all_actors
  stop_verify
  local actor_gpus=() actor_ports=()
  csv_to_array "${ALL_ACTOR_GPUS}" actor_gpus
  csv_to_array "${ALL_ACTOR_PORTS}" actor_ports
  ((${#actor_gpus[@]} == ALL_TP)) || { log "ALL_ACTOR_GPUS count != ALL_TP"; exit 1; }
  ((${#actor_ports[@]} == ALL_TP)) || { log "ALL_ACTOR_PORTS count != ALL_TP"; exit 1; }
  local extra pid i gpu port
  extra="$(vllm_extra_for_model "${model_slug}") ${VLLM_EXTRA:-}"
  ACTOR_PIDS=()
  for i in "${!actor_gpus[@]}"; do
    gpu="${actor_gpus[$i]}"
    port="${actor_ports[$i]}"
    pid="$(start_vllm_bg "${gpu}" "${port}" "${model_path}" "${api_model}" \
      "${extra}" "${REL_LOGS}/actor_${model_slug}_all_gpu${gpu}.log")"
    ACTOR_PIDS+=("$(normalize_pid "${pid}")")
  done
  if model_needs_smoke_test "${model_slug}"; then
    smoke_test_tool_call "${actor_ports[0]}" "${api_model}"
  fi
  VERIFY_PID="$(start_vllm_bg "${ALL_VERIFY_GPU}" "${ALL_VERIFY_PORT}" "${VERIFY_MODEL_PATH}" \
    "${VERIFY_MODEL_NAME}" "--max-model-len 8192 --trust-remote-code --moe-backend triton" \
    "${REL_LOGS}/verify_${model_slug}.log")"
  VERIFY_PID="$(normalize_pid "${VERIFY_PID}")"
}

assert_transfer_ready() {
  local bench root queries corpus index
  for bench in "${ALL_BENCHMARKS[@]}"; do
    in_csv "${bench}" "${BENCHMARKS:-}" || continue
    root="${TRANSFER_ROOT}/${bench}"
    queries="${root}/queries.jsonl"
    corpus="${root}/corpus.jsonl"
    index="${root}/indexes/bm25"
    [[ -f "${queries}" ]] || { log "missing ${queries}"; exit 1; }
    [[ -f "${corpus}" ]] || { log "missing ${corpus}"; exit 1; }
    [[ -d "${index}" ]] || { log "missing Lucene index ${index}"; exit 1; }
    log "local_bm25 ready benchmark=${bench} index=${index}"
  done
}

eval_out_dir() {
  local model_slug="$1" component="$2" bench="$3" out_suffix="$4"
  echo "${REL_OUT}/eval_h1_${bench}_${model_slug}_${component}_${out_suffix}_${RUN_ID}"
}

job_already_done() {
  local out="$1"
  [[ "${SKIP_EXISTING}" == "1" ]] || return 1
  local official="${out}/FOUR_CELL_OFFICIAL_SUMMARY.json"
  local summary="${out}/upstream_api/SUMMARY.json"
  local candidate=""
  if [[ -f "${official}" ]]; then
    candidate="${official}"
  elif [[ -f "${summary}" ]]; then
    candidate="${summary}"
  else
    return 1
  fi
  "${PY}" - <<PY
import json, sys
p = json.load(open(${candidate@Q}))
sys.exit(0 if p.get("formal_eligible") or p.get("coverage_complete") else 1)
PY
}

run_eval_job() {
  local model_path="$1" model_slug="$2" api_model="$3" component="$4"
  local bench="$5" actor_urls="$6" tp="$7" out_suffix="$8"
  local job="${model_slug}/${component}/${bench}"
  local out
  out="$(eval_out_dir "${model_slug}" "${component}" "${bench}" "${out_suffix}")"
  local log_path="${REL_LOGS}/eval_${model_slug}_${component}_${bench}.log"
  local extra=() max_new

  if [[ -n "${SKIP_JOBS}" ]] && in_csv "${job}" "${SKIP_JOBS}"; then
    log "SKIP_JOBS ${job}"
    SKIPPED_JOBS+=("${job}")
    echo "JOB skip model=${model_slug} component=${component} benchmark=${bench} out=${out}"
    return 0
  fi

  if [[ -n "${REMAIN_JOB}" && "${job}" == "${REMAIN_JOB}" ]]; then
    [[ -n "${REMAIN_QUERY_IDS_FILE}" && -f "${REMAIN_QUERY_IDS_FILE}" ]] || {
      log "REMAIN_JOB=${REMAIN_JOB} missing REMAIN_QUERY_IDS_FILE"
      exit 1
    }
    extra+=(--query-ids "$("${PY}" -c "from pathlib import Path; print(','.join(Path(r'${REMAIN_QUERY_IDS_FILE}').read_text().split()))")")
    out="${out}_remain"
    log "REMAIN ${job} n_ids=$(wc -l < "${REMAIN_QUERY_IDS_FILE}") out=${out}"
  fi

  if job_already_done "${out}"; then
    log "SKIP existing complete ${job} out=${out}"
    SKIPPED_JOBS+=("${job}")
    echo "JOB skip model=${model_slug} component=${component} benchmark=${bench} out=${out}"
    return 0
  fi

  max_new="${MAX_NEW_TOKENS}"
  if is_harmony_slug "${model_slug}"; then
    max_new="${MAX_NEW_TOKENS_HARMONY}"
    extra+=(--reasoning-effort "${REASONING_EFFORT}")
  fi
  [[ -n "${N_EVAL:-}" ]] && extra+=(--n-eval "${N_EVAL}")
  if [[ "${component}" == "all" ]]; then
    extra+=(
      --verify-base-url "http://127.0.0.1:${ALL_VERIFY_PORT}/v1"
      --verify-model "${VERIFY_MODEL_NAME}"
    )
  fi

  log "START ${job} tp=${tp} urls=${actor_urls} max_new=${max_new} out=${out}"
  echo "JOB start model=${model_slug} component=${component} benchmark=${bench} out=${out}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN ${PY} scripts/run_eval.py --harness Harness-1 --benchmark ${bench} --model_name ${model_path} --evaluation-path upstream_api --api-base-url ${actor_urls} --api-model ${api_model} --component ${component} --max-turns ${MAX_TURNS} --max-new-tokens ${max_new} --tp ${tp} --out ${out} ${extra[*]-}"
    DONE_JOBS+=("${job}")
    return 0
  fi

  local rc=0
  "${PY}" scripts/run_eval.py \
    --harness Harness-1 \
    --benchmark "${bench}" \
    --model_name "${model_path}" \
    --evaluation-path upstream_api \
    --retrieval-backend local_bm25 \
    --api-base-url "${actor_urls}" \
    --api-model "${api_model}" \
    --reranker none \
    --offline \
    --max-turns "${MAX_TURNS}" \
    --max-new-tokens "${max_new}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --temperature "${TEMPERATURE}" \
    --tp "${tp}" \
    --eval-stagger-s "${WORKER_STAGGER}" \
    --component "${component}" \
    --out "${out}" \
    "${extra[@]}" \
    > "${log_path}" 2>&1 || rc=$?
  if [[ "${rc}" -ne 0 ]]; then
    log "FAIL ${job} rc=${rc} see ${log_path}"
    FAILED_JOBS+=("${job}")
    tail -n 40 "${log_path}" | tee -a "${MASTER_LOG}" >&2 || true
    [[ "${FAIL_FAST}" == "1" ]] && return "${rc}"
    return 0
  fi
  log "DONE ${job} out=${out}"
  DONE_JOBS+=("${job}")
}

write_manifest() {
  local git_commit
  git_commit="$(git -C . rev-parse HEAD 2>/dev/null || echo unknown)"
  cat > "${REL_LOGS}/MANIFEST.json" <<EOF
{
  "run_id": "${RUN_ID}",
  "evaluation_path": "upstream_api",
  "eval_profile": "upstream_core_local_bm25",
  "retrieval_backend": "local_bm25",
  "chroma_required": false,
  "n_planned": 18,
  "benchmarks": ["longsealqa", "frames", "hotpotqa"],
  "components": ["zero", "all"],
  "models": [
    {"path": "${REL_MODELS}/gpt-oss-20b", "slug": "gpt-oss-20b"},
    {"path": "${REL_MODELS}/harness-1", "slug": "harness-1"},
    {"path": "${REL_MODELS}/Qwen3-4B-Instruct-2507", "slug": "Qwen3-4B-Instruct-2507"}
  ],
  "zero": {"gpu": "${GPU}", "actor_gpus": "${ZERO_ACTOR_GPUS}", "actor_ports": "${ZERO_ACTOR_PORTS}", "eval_tp": ${TP}, "actor_port": ${ZERO_ACTOR_PORT}},
  "all": {
    "actor_gpus": "${ALL_ACTOR_GPUS}",
    "actor_ports": "${ALL_ACTOR_PORTS}",
    "verify_gpu": "${ALL_VERIFY_GPU}",
    "verify_port": ${ALL_VERIFY_PORT},
    "actor_tp": ${ALL_TP},
    "eval_tp": ${ALL_EVAL_TP},
    "verify_model": "${VERIFY_MODEL_NAME}"
  },
  "max_turns": ${MAX_TURNS},
  "max_new_tokens": ${MAX_NEW_TOKENS},
  "max_new_tokens_harmony": ${MAX_NEW_TOKENS_HARMONY},
  "reasoning_effort": "${REASONING_EFFORT}",
  "n_eval": "${N_EVAL:-}",
  "dry_run": ${DRY_RUN},
  "git_commit": "${git_commit}",
  "done_jobs": $(printf '%s\n' "${DONE_JOBS[@]+"${DONE_JOBS[@]}"}" | "${PY}" -c 'import json,sys; print(json.dumps([x.strip() for x in sys.stdin if x.strip()]))'),
  "failed_jobs": $(printf '%s\n' "${FAILED_JOBS[@]+"${FAILED_JOBS[@]}"}" | "${PY}" -c 'import json,sys; print(json.dumps([x.strip() for x in sys.stdin if x.strip()]))'),
  "skipped_jobs": $(printf '%s\n' "${SKIPPED_JOBS[@]+"${SKIPPED_JOBS[@]}"}" | "${PY}" -c 'import json,sys; print(json.dumps([x.strip() for x in sys.stdin if x.strip()]))')
}
EOF
}

main() {
  assert_transfer_ready

  local actor_ports=() zero_urls all_urls
  csv_to_array "${ZERO_ACTOR_PORTS}" actor_ports
  zero_urls="$(join_urls "${actor_ports[@]}")"
  csv_to_array "${ALL_ACTOR_PORTS}" actor_ports
  all_urls="$(join_urls "${actor_ports[@]}")"
  local zero_suffix="gpu${ZERO_ACTOR_GPUS//,/}"
  local all_suffix="gpu${ALL_ACTOR_GPUS//,/}v${ALL_VERIFY_GPU}"

  log "RUN_ID=${RUN_ID} jobs=18 dry_run=${DRY_RUN} models=${MODELS:-all} components=${COMPONENTS:-zero,all} benchmarks=${BENCHMARKS:-longsealqa,frames,hotpotqa}"

  local idx model_path model_slug api_model component bench tp urls suffix
  for idx in "${!MODEL_PATHS[@]}"; do
    model_path="${MODEL_PATHS[$idx]}"
    model_slug="${MODEL_SLUGS[$idx]}"
    api_model="${API_MODELS[$idx]}"
    in_csv "${model_slug}" "${MODELS:-}" || { log "skip model ${model_slug} (not in MODELS)"; continue; }
    if [[ "${DRY_RUN}" != "1" && ! -d "${model_path}" ]]; then
      log "missing model dir ${model_path}"
      exit 1
    fi

    for component in "${ALL_COMPONENTS[@]}"; do
      in_csv "${component}" "${COMPONENTS:-}" || continue
      if [[ "${component}" == "zero" && "${SKIP_ZERO}" == "1" ]]; then
        log "SKIP_ZERO=1 — skipping ${model_slug}/zero"
        continue
      fi
      if [[ "${component}" == "all" && "${SKIP_ALL}" == "1" ]]; then
        log "SKIP_ALL=1 — skipping ${model_slug}/all"
        continue
      fi

      if [[ "${component}" == "zero" ]]; then
        urls="${zero_urls}"
        tp="${TP}"
        suffix="${zero_suffix}"
        [[ "${DRY_RUN}" == "1" ]] || start_zero_stack "${model_path}" "${api_model}" "${model_slug}"
      else
        urls="${all_urls}"
        tp="${ALL_EVAL_TP}"
        suffix="${all_suffix}"
        [[ "${DRY_RUN}" == "1" ]] || start_all_stack "${model_path}" "${api_model}" "${model_slug}"
      fi

      for bench in "${ALL_BENCHMARKS[@]}"; do
        in_csv "${bench}" "${BENCHMARKS:-}" || continue
        run_eval_job "${model_path}" "${model_slug}" "${api_model}" "${component}" \
          "${bench}" "${urls}" "${tp}" "${suffix}" || {
            local rc=$?
            write_manifest
            exit "${rc}"
          }
      done
    done
  done

  if [[ "${DRY_RUN}" != "1" ]]; then
    stop_all_actors
    stop_verify
    trap - EXIT
  fi
  write_manifest

  log "finished done=${#DONE_JOBS[@]} failed=${#FAILED_JOBS[@]} skipped=${#SKIPPED_JOBS[@]} manifest=${REL_LOGS}/MANIFEST.json"
  if ((${#FAILED_JOBS[@]} > 0)); then
    log "failed jobs: ${FAILED_JOBS[*]}"
    exit 1
  fi
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
