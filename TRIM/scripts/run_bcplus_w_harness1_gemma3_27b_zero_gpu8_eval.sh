#!/usr/bin/env bash
# Gemma-3-27B-it × zero — bcplus_full (830 queries), upstream_api + local_bm25.
# 单节点 8 × gm402-80g, FULL 8-CARD LOAD (8 single-GPU actors, no verifier).
#
# Forked from run_bcplus_full_qwen35_27b_zero_gpu8_eval.sh (Qwen3.5-27B zero,
# 成功 task t-20260914151008-fcanl: 8 single-GPU actors TP=1 GPU 0..7 + 16
# workers, 830/830 ok). Same 8-card full-load layout; only model / parser /
# thinking handling differ for Gemma-3.
#
# 被测模型 = /mnt/songzijun/models/gemma-3-27b-it
#   config.json 架构 Gemma3ForConditionalGeneration / model_type gemma3
#   / text_config 62 层 hidden 5376 / dense（无 MoE）/ 12 shards ~51.10 GiB bf16
#   / 多模态 VLM（siglip vision encoder），BC+ 走纯文本路径
#   / tokenizer: SentencePiece（tokenizer.model + tokenizer.json，非 harmony o200k）。
#
# vLLM 参数由 trim.eval.model_profiles vllm-extra 自动产出（GEMMA_PROFILE）：
#     --enable-auto-tool-choice --tool-call-parser pythonic
#     --max-model-len N --trust-remote-code
# （GEMMA_PROFILE 无 reasoning_parser、无 moe_backend；dense 模型。）
#
# Thinking：Gemma3 IT 无 thinking 模式，chat_template 无 enable_thinking 守卫、不输出
# 独立 reasoning 块。故无需改模板、无需 strip reasoning_parser（GEMMA_PROFILE 本就无
# reasoning_parser）。与 Qwen3.5-27B no-thinking 的差异：后者需在 no-thinking 下显式 STRIP
# --reasoning-parser qwen3；gemma 无此问题。不动 scorer/runner（CLAUDE.md：官方 runner 不 fork）。
#
# Why 8-card full load (充分利用 8 卡显存):
#   * zero component 无 verifier，故把 8 张卡全用来跑被测模型 actor（每卡 1 个 TP=1 actor）。
#   * 27B dense 权重 51.10 GiB；单卡 gm402-80g + GPU_UTIL 0.90 ~72 GiB，权重 51 GiB + KV ~16 GiB
#     = ~67 GiB/80 GiB，充分利用显存（同 Qwen3.5-27B zero 实测口径）。
#   * 8 actor (TP=1) + 16 worker → 每 GPU 2 worker；避免 TP=8 all-reduce 开销（27B 单卡放得下）。
#   * 对比 lib 的 lib_bcplus_gpu8_run_zero（单 actor GPU 7，7 卡空闲）：本脚本 8 卡全载，吞吐 8x。
#
# Optional env (defaults shown):
#   ZERO_ACTOR_GPUS=0,1,2,3,4,5,6,7  ZERO_ACTOR_PORTS=8040,8042,...,8054
#   TP=16  MAX_NUM_SEQS=8  MAX_MODEL_LEN=32768  GPU_UTIL=0.90  WORKER_STAGGER=10
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRIM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${TRIM_ROOT}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_bcplus_full_gemma3_27b_zero_gpu8}"
PY="${PY:-python}"
VLLM="${VLLM:-vllm}"

# --- relative paths (from TRIM root; resolve to /mnt/songzijun/... in CloudML) ---
REL_MODELS="../../models"
REL_BCPLUS="../SCOPE/external/BrowseComp-Plus"
REL_OUT="outputs"
REL_LOGS="${REL_OUT}/logs/${RUN_ID}"
REL_SCAPE_EASYOPD="../SCAPE-EasyOPD"

BCPLUS_INDEX="${BCPLUS_INDEX:-${REL_BCPLUS}/indexes/bm25}"
BCPLUS_CORPUS="${BCPLUS_CORPUS:-${REL_BCPLUS}/data/browsecomp_plus_corpus_full.jsonl}"

# Gemma-3-27B-it zero only. 绝对路径 = /mnt/songzijun/models/gemma-3-27b-it
MODEL_PATH="${REL_MODELS}/gemma-3-27b-it"
MODEL_SLUG="gemma-3-27b-it"
API_MODEL="gemma-3-27b-it"
COMPONENT="zero"

# 8 single-GPU actors (TP=1 each). No tensor parallel → no all-reduce cost. 8 卡全载。
ZERO_ACTOR_GPUS="${ZERO_ACTOR_GPUS:-0,1,2,3,4,5,6,7}"
ZERO_ACTOR_PORTS="${ZERO_ACTOR_PORTS:-8040,8042,8044,8046,8048,8050,8052,8054}"
TP="${TP:-16}"                 # 16 API-client workers → 2 per GPU
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"   # 27B 权重 51.10GiB；GPU_UTIL 0.90 单卡 ~72GiB，剩 ~20GiB KV。
                                   # 每 seq @32k KV ~2GiB，MAX_NUM_SEQS=8 → KV ~16GiB，安全余量足且
                                   # 与已验证 Qwen3.5-27B zero 布局一致；2 worker/GPU 实际并发，8 留 burst。
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
# 27B dense 权重 51.10GiB，单卡 80g 放得下；GPU_UTIL 0.90 充分利用显存、留 KV pool。
GPU_UTIL="${GPU_UTIL:-0.90}"
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
  [[ "${last}" =~ ^([0-9]+)$ ]] && {
    echo "${BASH_REMATCH[1]}"
    return 0
  }
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
  # Gemma-3-27B-it: 经 trim.eval.model_profiles vllm-extra 自动产出基础参数
  #   --enable-auto-tool-choice --tool-call-parser pythonic
  #   --max-model-len N --trust-remote-code
  # （dense 非 MoE，不加 --moe-backend；GEMMA_PROFILE 无 reasoning_parser，无需 strip。）
  local trim_root="${TRIM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
  PYTHONPATH="${trim_root}:${PYTHONPATH:-}" "${PY:-python}" \
    -m trim.eval.model_profiles vllm-extra "${MODEL_SLUG}"
}

start_vllm_bg() {
  local gpu="$1" port="$2" model_path="$3" served_name="$4" extra="$5" log_path="$6"

  stop_port "${port}" "${log_path}.pid"
  wait_port_free "${port}" 120

  log "starting vLLM GPU${gpu} port ${port} model=${model_path} served=${served_name} extra=${extra}"
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

smoke_test_tool_call() {
  local port="$1" model="$2"
  log "smoke test tool call port=${port} model=${model} (gemma pythonic parser)"
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
msg = (body.get("choices") or [{}])[0].get("message", {})
tool_calls = msg.get("tool_calls") or []
# 诊断打印：pythonic parser 是否正确解析出 tool_calls。
print(f"[smoke] content={repr(msg.get('content'))[:300]}")
print(f"[smoke] tool_calls_count={len(tool_calls)}")
if not tool_calls:
    raise SystemExit(f"smoke test missing tool_calls (content printed above); full body[:1500]={json.dumps(body)[:1500]}")
name = ((tool_calls[0].get("function") or {}).get("name") or "")
if name != "search_corpus":
    raise SystemExit(f"smoke test unexpected tool name: {name!r}")
print("smoke_ok", name)
PY
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
  log "starting ${n_actors} single-GPU vLLM actors (TP=1 each); TP=${TP} workers → ~$((TP / n_actors)) per GPU (8-card full load)"

  local extra pid i gpu port
  extra="$(vllm_extra_for_model "${MODEL_SLUG}")"
  log "vllm extra (resolved): ${extra}"
  ACTOR_PIDS=()
  for i in "${!actor_gpus[@]}"; do
    gpu="${actor_gpus[$i]}"
    port="${actor_ports[$i]}"
    pid="$(start_vllm_bg "${gpu}" "${port}" "${MODEL_PATH}" "${API_MODEL}" \
      "${extra}" "${REL_LOGS}/actor_${MODEL_SLUG}_zero_gpu${gpu}.log")"
    pid="$(normalize_pid "${pid}")"
    ACTOR_PIDS+=("${pid}")
  done

  # Gemma-3 用 pythonic parser；smoke-test tool_calls 能正常发出再交给 run_eval。
  smoke_test_tool_call "${actor_ports[0]}" "${API_MODEL}"
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

  log "RUN_ID=${RUN_ID} actors=${ZERO_ACTOR_GPUS} ports=${ZERO_ACTOR_PORTS} tp=${TP} max_num_seqs=${MAX_NUM_SEQS} max_model_len=${MAX_MODEL_LEN} gpu_util=${GPU_UTIL} worker_stagger=${WORKER_STAGGER} (8-card full load, no verifier)"

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
  "model": "${REL_MODELS}/gemma-3-27b-it",
  "model_arch": "Gemma3ForConditionalGeneration",
  "component": "zero",
  "actor_gpus": "${ZERO_ACTOR_GPUS}",
  "actor_ports": "${ZERO_ACTOR_PORTS}",
  "tp": ${TP},
  "max_num_seqs": ${MAX_NUM_SEQS},
  "max_model_len": ${MAX_MODEL_LEN},
  "gpu_util": ${GPU_UTIL},
  "worker_stagger_s": ${WORKER_STAGGER},
  "tool_call_parser": "pythonic",
  "reasoning_parser": "none",
  "thinking_disabled": true,
  "thinking_disable_mechanism": "gemma3_it_has_no_thinking_mode",
  "layout": "8 single-GPU vLLM actors (TP=1), ${TP} API-client workers, 8-card full load, no verifier",
  "retrieval": "local_bm25 (Pyserini/Lucene), no Chroma server",
  "zero_actor_urls": "${ZERO_ACTOR_URLS}",
  "corpus_path": "${BCPLUS_CORPUS}",
  "index_path": "${BCPLUS_INDEX}",
  "git_commit": "${GIT_COMMIT}"
}
EOF

  log "finished Gemma-3-27B-it zero run; outputs under ${REL_OUT}/eval_h1_bcplus_full_${MODEL_SLUG}_zero_*_${RUN_ID}"
  log "manifest ${REL_LOGS}/MANIFEST.json"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
