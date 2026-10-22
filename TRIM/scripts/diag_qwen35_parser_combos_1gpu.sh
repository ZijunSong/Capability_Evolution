#!/usr/bin/env bash
# 诊断脚本：Qwen3.5-27B 在 no-thinking 模式下 qwen3_coder tool parser 的正确 parser 组合。
# 1 卡 gm402，加载模型一次，用 4 种 vLLM serve 参数组合各发一次 tool-call smoke 请求，
# 打印每种的 tool_calls/content/reasoning 字段，确定哪种组合能正确解析出 tool_calls。
#
# 背景：recovery1（t-...acfl1）smoke test 失败 —— 模型生成了正确格式的 tool call
# （<function=search_corpus>...），但被 --reasoning-parser qwen3 当成 reasoning 吞进
# reasoning 字段，导致 tool_calls 为空、content 为 null。需确认 no-thinking 下正确组合。
#
# 用法：由 CloudML imageCommand 调用。1 卡 GUARANTEED_PUBLIC。
set -euo pipefail

JFS_ROOT="/mnt/songzijun"
TRIM_ROOT="${JFS_ROOT}/Capability_Evolution/TRIM"
MODEL_PATH="${JFS_ROOT}/models/Qwen3.5-27B"
PYPI_INDEX="https://pkgs.d.xiaomi.net/artifactory/api/pypi/pypi-virtual/simple"

echo "[diag] $(date -Iseconds) host=$(hostname)"
export PYTHONUNBUFFERED=1

# 0. cp 自身到 /tmp 再 exec（规避 JuiceFS heredoc 回读）
SELF="${BASH_SOURCE[0]:-$0}"
if [[ "${SELF}" == /mnt/songzijun/* ]]; then
  TMP_SELF="/tmp/$(basename "${SELF}")"
  cp -f "${SELF}" "${TMP_SELF}"
  exec bash "${TMP_SELF}" "$@"
fi

# 1. 装四件套（与正式任务一致，精简：只装跑 vllm serve 必需的）
echo "[diag] installing minimal locked runtime"
python3 -m pip install --no-cache-dir --ignore-installed --upgrade-strategy only-if-needed \
  --index-url "${PYPI_INDEX}" \
  "torch==2.10.0" "transformers==4.57.6" "vllm==0.19.1" "sentencepiece" \
  2>&1 | tail -5

python3 - <<'PY'
import torch, vllm, transformers
print(f"[diag] torch={torch.__version__} vllm={vllm.__version__} transformers={transformers.__version__}")
assert torch.cuda.device_count() >= 1
PY

# 2. 用 vLLM offline LLM 直接测 4 种 parser 组合（不起 server，用 LLM.generate + 手工解析太复杂；
#    改为起 4 次 server 太慢。最简方案：用 vllm LLM 的 chat + tool，让 vllm 内部走 parser。）
# 但 vllm LLM.generate 不走 tool_call_parser（那是 OpenAI server 层）。所以必须起 server。
# 折中：起 1 个 server（组合 A），测完 kill，起组合 B……共 4 次，每次加载~3min，总~15min。
# 优化：用 --enforce-eager + gpu-memory-utilization 0.90，模型加载后 KV cache 复用。

PORT=8140
LOGDIR="/tmp/diag_qwen35_parser"
mkdir -p "${LOGDIR}"

# 4 种组合：A=默认(qwen3_coder+qwen3 reasoning), B=去reasoning, C=hermes, D=hermes无reasoning
combos=(
  "A|--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3"
  "B|--enable-auto-tool-choice --tool-call-parser qwen3_coder"
  "C|--enable-auto-tool-choice --tool-call-parser hermes"
  "D|--enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3"
)

for combo in "${combos[@]}"; do
  label="${combo%%|*}"
  extra="${combo#*|}"
  echo "=================================================================="
  echo "[diag] combo ${label}: ${extra}"
  echo "=================================================================="
  LOG="${LOGDIR}/actor_${label}.log"
  # kill any stale
  pkill -f "vllm serve" 2>/dev/null || true
  sleep 3
  CUDA_VISIBLE_DEVICES=0 nohup vllm serve "${MODEL_PATH}" \
    --host 127.0.0.1 --port "${PORT}" --served-model-name Qwen3.5-27B \
    --max-model-len 32768 --trust-remote-code --gpu-memory-utilization 0.90 --enforce-eager \
    ${extra} > "${LOG}" 2>&1 &
  VPID=$!
  echo "[diag] vllm pid=${VPID}, waiting for API (up to 600s)..."
  ok=0
  for i in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then ok=1; break; fi
    sleep 10
  done
  if [ "${ok}" != "1" ]; then
    echo "[diag] combo ${label}: API not ready, tail log:"; tail -20 "${LOG}";
    kill "${VPID}" 2>/dev/null || true; sleep 3; continue
  fi
  echo "[diag] combo ${label}: API ready, sending smoke tool-call request..."
  python3 - <<PY
import json, urllib.request, urllib.error
port = ${PORT}
payload = {
    "model": "Qwen3.5-27B",
    "messages": [{"role": "user", "content": "Search the corpus for smoke test query."}],
    "tools": [{"type": "function", "function": {
        "name": "search_corpus", "description": "Search corpus",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    }}],
    "tool_choice": "auto", "max_tokens": 256, "temperature": 0.0,
}
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
    data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
try:
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode())
except urllib.error.HTTPError as exc:
    print(f"[diag] combo ${'${label}'}: HTTP {exc.code}: {exc.read().decode('utf-8','replace')[:400]}")
    raise SystemExit
msg = (body.get("choices") or [{}])[0].get("message", {})
tc = msg.get("tool_calls") or []
print(f"[diag] combo ${'${label}'}: tool_calls_count={len(tc)} content_type={type(msg.get('content')).__name__} reasoning_len={len(msg.get('reasoning') or '')}")
if tc:
    print(f"[diag] combo ${'${label}'}: FIRST tool name = {((tc[0].get('function') or {}).get('name') or '')!r}")
print(f"[diag] combo ${'${label}'}: content = {repr(msg.get('content'))[:200]}")
print(f"[diag] combo ${'${label}'}: reasoning = {repr(msg.get('reasoning'))[:200]}")
PY
  kill "${VPID}" 2>/dev/null || true
  sleep 5
done

echo "=================================================================="
echo "[diag] all combos done. pick the one with tool_calls_count >= 1."
echo "=================================================================="
