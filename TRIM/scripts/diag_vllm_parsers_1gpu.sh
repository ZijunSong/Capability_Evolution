#!/usr/bin/env bash
# 诊断脚本：枚举 vLLM 0.19.1 注册的 tool-call parser，并打印 GLM 相关 parser。
# 不加载模型，只验证 parser 注册表，快速（~3min）。
set -euo pipefail
JFS_ROOT="/mnt/songzijun"
TRIM_ROOT="${JFS_ROOT}/Capability_Evolution/TRIM"
PYPI_INDEX="https://pkgs.d.xiaomi.net/artifactory/api/pypi/pypi-virtual/simple"

echo "[diag] $(date -Iseconds) installing minimal vllm"
python3 -m pip install --no-cache-dir --ignore-installed --upgrade-strategy only-if-needed \
  --index-url "${PYPI_INDEX}" "vllm==0.19.1" 2>&1 | tail -2

echo "[diag] enumerating registered tool parsers"
python3 - <<'PY'
import importlib, pkgutil, inspect
import vllm
print("vllm:", vllm.__version__, vllm.__file__)
# Try several known registration entry points
found = {}
for modpath in [
    "vllm.entrypoints.openai.tool_parsers",
    "vllm.entrypoints.openai.tool_parsers.abstract_tool_parser",
    "vllm.v1.entrypoints.openai.tool_parsers",
    "vllm.entrypoints.tool_inputs",
    "vllm.tool_parsers",
    "vllm.tool_parsers.abstract_tool_parser",
]:
    try:
        m = importlib.import_module(modpath)
    except Exception as e:
        print(f"  import {modpath}: FAIL {e}")
        continue
    print(f"  import {modpath}: OK, attrs={[a for a in dir(m) if not a.startswith('_')][:40]}")
    for attr in ("ToolParserManager",):
        obj = getattr(m, attr, None)
        if obj is None:
            continue
        reg = getattr(obj, "_registry", None) or getattr(obj, "tool_parsers", None)
        if isinstance(reg, dict) and reg:
            found.update(reg)
            print(f"    {attr} registry keys: {sorted(reg.keys())}")
# Fallback: walk packages
try:
    for _, name, ispkg in pkgutil.walk_packages(vllm.__path__, prefix="vllm."):
        nl = name.lower()
        if ("tool_parser" in nl or "tool_call" in nl) and not ispkg:
            print(f"  found module: {name}")
            try:
                m = importlib.import_module(name)
                for a in dir(m):
                    if "Parser" in a or "registry" in a.lower():
                        obj = getattr(m, a)
                        reg = getattr(obj, "_registry", None) or getattr(obj, "tool_parsers", None)
                        if isinstance(reg, dict) and reg:
                            found.update(reg)
                            print(f"      {a} keys: {sorted(reg.keys())}")
            except Exception as e:
                print(f"      import err: {e}")
except Exception as e:
    print(f"  walk_packages failed: {e}")
print("=== ALL FOUND PARSERS ===")
for k in sorted(found.keys()):
    print(" -", k)
PY
echo "[diag] done"
