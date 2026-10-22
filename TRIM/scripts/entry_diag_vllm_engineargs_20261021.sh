#!/usr/bin/env bash
# Minimal diagnostic: does vLLM 0.19.1's LLM()/EngineArgs accept tool_call_parser /
# enable_auto_tool_choice / reasoning_parser? Inspects the dataclass fields (no model load).
# This determines whether we can fix legacy_local's 0% tool-call-parse bug by passing these
# to vllm_rollout_worker.py's LLM(**llm_kwargs).
set -euxo pipefail
export PYTHONUNBUFFERED=1

PYPI_INDEX="https://pkgs.d.xiaomi.net/artifactory/api/pypi/pypi-virtual/simple"
python3 -m pip install --no-cache-dir --quiet --index-url "${PYPI_INDEX}" \
  "vllm==0.19.1" "torch==2.10.0" 2>&1 | tail -3 || echo "[pip] some issue (non-fatal)"

python3 - <<'PY'
from vllm.engine.arg_utils import EngineArgs
import dataclasses
fields = {f.name for f in dataclasses.fields(EngineArgs)}
checks = ["tool_call_parser", "enable_auto_tool_choice", "reasoning_parser",
          "reasoning_parser_plugin", "chat_template", "tokenizer_mode"]
print("=== EngineArgs field check (vLLM 0.19.1) ===")
for c in checks:
    print(f"  {c}: {'YES' if c in fields else 'NO'}")
# Also try constructing EngineArgs with tool_call_parser to confirm it's accepted at construction
import vllm
print(f"vllm version: {vllm.__version__}")
try:
    ea = EngineArgs(model="dummy", tool_call_parser="qwen3_coder", enable_auto_tool_choice=True)
    print("EngineArgs(model, tool_call_parser='qwen3_coder', enable_auto_tool_choice=True): CONSTRUCTED OK")
    print(f"  ea.tool_call_parser={ea.tool_call_parser} ea.enable_auto_tool_choice={ea.enable_auto_tool_choice}")
except Exception as e:
    print(f"EngineArgs construction with tool_call_parser FAILED: {type(e).__name__}: {e}")
PY
echo "=== diagnostic done ==="