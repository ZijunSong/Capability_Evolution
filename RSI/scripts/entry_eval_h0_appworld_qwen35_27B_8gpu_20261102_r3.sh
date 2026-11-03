#!/usr/bin/env bash
# RSI H0 基线完整测评（完整 AppWorld benchmark）— Qwen3.5-27B，单节点 8xM402
# 布局：panel workers x8（测评无 proposer，8 卡全作 worker）
# 完整 benchmark：7 个 role 池全量 325 task 逐个评（harness minimal / official TGC）
#
# r3（2026-11-02，1nvje BrokenProcessPool 复盘）：27B r2(1nvje) 用的是旧 dep 块
# （未钉 torch/fla），装了 torch 2.15.0+cu132（节点 driver 是 cu128 族）+ 未钉版
# fla + causal-conv1d 编译失败 → 8 worker 各自首 generate 即被杀 → 每 split
# BrokenProcessPool，7 split 全灭 0 task 完成。对齐已验证的 dhlol(9B r2) dep 块：
# torch==2.10.0+cu128 钉死+CUDA 硬门、fla==0.6.0 钉死+is_flash_linear_attention
# 硬断言、causal-conv1d --no-build-isolation 回退。② campaign_id + _r3 → 全新 OUT。
set -euo pipefail
RSI_ROOT=/mnt/songzijun/Capability_Evolution/RSI
MODEL_SRC=/mnt/songzijun/models/Qwen3.5-27B
MODEL_LOCAL=/tmp/rsi_models/Qwen3.5-27B
CFG_SRC="$RSI_ROOT/configs/eval_h0_qwen35_27B_appworld_8gpu.yaml"
CFG_RUNTIME=/tmp/rsi_eval_h0_appworld_cfg_27B_r3.yaml
export PYTHONPATH="$RSI_ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
cd "$RSI_ROOT"

echo "== dep preflight =="
python - <<'PY'
import importlib, subprocess, sys

MIRROR = "https://pkgs.d.xiaomi.net/artifactory/api/pypi/pypi-virtual/simple"
TORCH_PIN = "torch==2.10.0"          # cu128 族，与 gm402 driver 匹配
TF_PIN = "transformers==5.17.0"      # ≥5.6 才有 qwen3_5 注册
MIN_GPUS = 8

def pip(*args, limit=900):
    """Bounded pip install against MIRROR; returns exit code (never raises)."""
    print("pip", " ".join(args), flush=True)
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-i", MIRROR, *args],
                       timeout=limit)
    if r.returncode != 0:
        print("pip failed:", args, flush=True)
    return r.returncode

# 1) 基础包：缺失才装，装则按钉死版本
need = []
for mod, pkg in (("yaml", "pyyaml"), ("numpy", "numpy"), ("torch", TORCH_PIN), ("transformers", TF_PIN)):
    try:
        importlib.import_module(mod)
    except ImportError:
        need.append(pkg)
if need:
    pip(*need)

# 2) 版本审计 + CUDA 硬门
def audit():
    import torch, transformers
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "devices", n)
    print("transformers", transformers.__version__)
    return torch.cuda.is_available() and n >= MIN_GPUS

if not audit():
    print(f"[WARN] CUDA unusable with current torch — forcing {TORCH_PIN} / {TF_PIN}", flush=True)
    pip("--force-reinstall", TORCH_PIN, TF_PIN)
    code = (
        "import torch, transformers, sys; "
        "n = torch.cuda.device_count() if torch.cuda.is_available() else 0; "
        "print('retry torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'devices', n); "
        "print('retry transformers', transformers.__version__); "
        "sys.exit(0 if torch.cuda.is_available() and n >= %d else 1)" % MIN_GPUS
    )
    r = subprocess.run([sys.executable, "-c", code])
    if r.returncode != 0:
        sys.exit("[FATAL] CUDA still unusable after pinning torch %s — node driver vs torch build mismatch" % TORCH_PIN)

# 3) qwen3_5 注册必须在（tf <5.6 无此模型类型）
try:
    from transformers.models import qwen3_5  # noqa: F401
    print("qwen3_5 registration: ok")
except Exception as e:  # noqa: BLE001
    sys.exit(f"[FATAL] transformers has no qwen3_5 model type ({e}); need transformers>=5.6")

# 4) Qwen3.5 的线性注意力/卷积核。缺了不是报错而是退回 PyTorch 参考实现
#    ——"correct but much slower"，对 8 卡 panel × 多轮 agentic rollout 是数倍
#    吞吐损失。flash-linear-attention 有纯 py wheel，必装；causal-conv1d 镜像上
#    只有 sdist（无 wheel，编译要 nvcc），尽力而为，失败就吃慢速回退。
print("kernel deps:", flush=True)
pip("flash-linear-attention==0.6.0", limit=600)
# causal-conv1d 镜像只有 sdist；pip 的 build isolation 会拉 latest torch 去编译，
# 于是去猜 GitHub 上 cu13/torch2.15 的预编译 wheel（外网不通）。先按常规装，
# 失败再 --no-build-isolation 用已装好的 torch 2.10.0 编译。
_r = pip("causal-conv1d", limit=600)
if _r != 0:
    _r = pip("--no-build-isolation", "causal-conv1d", limit=900)
print(("  causal-conv1d: installed" if _r == 0 else
       "  causal-conv1d: MISSING (slow reference fallback; sdist-only + no nvcc?)"), flush=True)
pip("openpyxl", "swebench", limit=600)
# ⚠️ 包名 flash-linear-attention 的导入名是 **fla**（top_level.txt=fla），真正的核在
# 依赖 fla-core==0.6.0 里。transformers 判定走 is_flash_linear_attention_available()
# （查 fla 版本 ≥0.2.2 + CUDA）。别 import flash_linear_attention（不存在，wtmif/r4 踩过）。
try:
    from transformers.utils import is_flash_linear_attention_available, is_causal_conv1d_available
    fla_ok = bool(is_flash_linear_attention_available())
    conv_ok = bool(is_causal_conv1d_available())
    print(f"  fla fused kernels: {'ok' if fla_ok else 'MISSING'} | causal_conv1d: {'ok' if conv_ok else 'MISSING'}", flush=True)
    if not fla_ok:
        sys.exit("[FATAL] flash-linear-attention(fla) not available — gated-delta-rule would fall back to slow PyTorch")
except ImportError as e:  # noqa: BLE001
    sys.exit(f"[FATAL] transformers kernel-availability API missing ({e}); transformers too old?")

# appworld: vendored repo + PINNED combo. Each pip is bounded in time and the
# resolver is kept away from big backtracking trees (stage-wise pins, --no-deps
# where safe) — an unbounded resolver pass stalled a previous run for 1h+.
try:
    import appworld
    print("appworld already importable")
except ImportError:
    pip("pydantic==2.12.0", "sqlalchemy==2.0.36", "sqlmodel==0.0.22", "sqlalchemy-utils==0.41.1", limit=300)
    pip("fastapi==0.115.6", "fastapi-login==1.10.3", "freezegun==1.5.1", limit=300)
    pip("--no-deps", "starlette==0.48.0", limit=300)
    pip("urllib3>=2.0.0", "requests>=2.28.0", limit=300)
    pip("uvicorn>=0.27", "uvloop>=0.21.0", "httpx2>=2.0.0", "pytest-xdist>=3.1.0", limit=300)
    pip("--no-deps", "/mnt/songzijun/datasets/appworld_src", limit=300)
    pip("pendulum>=3.0.0", "typer>=0.16.0", "text2num>=0.3.0", "orjson>=3.10.12",
        "pydantic-extra-types[pendulum]>=2.8.0", "inflection>=0.5.1", "email-validator>=2.0.0",
        "polyfactory>=2.15.0", "faker>=18.0.0", "xxhash>=3.0.0", "munch>=4.0.0",
        "rich>=13.0.0", "python-dotenv>=1.0.1", "python-multipart>=0.0.5",
        "httpx>=0.20.0", "libcst>=1.2.0", "jsonref>=1.1.0", "filelock>=3.1.0",
        "psutil>=5.9.0", "cryptography>=47.0.0", limit=600)
    import appworld as _aw  # verify importability now
    print("installed appworld from vendored repo (pinned combo)")
# unpack the encrypted apps bundle (bare installs ship it sealed)
import subprocess as _sp
_sp.run([sys.executable, "-m", "appworld.cli", "install"], check=False, timeout=300)
# apps 代码必须可导入。缺传递依赖就补装（限 5 轮）
for _round in range(5):
    try:
        import appworld.apps.admin  # noqa: F401  # hard assertion
        print("appworld.apps.admin import: ok")
        break
    except ModuleNotFoundError as e:
        _missing = (e.name or str(e)).split(".")[0]
        if _missing in ("appworld", ""):
            sys.exit("[FATAL] vendored appworld not importable; do NOT pip install appworld from PyPI")
        print(f"[WARN] missing module {_missing!r} — pip installing (round {_round + 1})", flush=True)
        if pip(_missing, limit=300) != 0:
            sys.exit(f"[FATAL] appworld import needs {_missing!r} but pip install failed")
else:
    sys.exit("[FATAL] appworld.apps.admin still unimportable after 5 repair rounds")
PY

echo "== model pre-copy =="
if [ ! -f "$MODEL_LOCAL/config.json" ]; then
  rm -rf "$MODEL_LOCAL"; mkdir -p "$MODEL_LOCAL"
  cp -r "$MODEL_SRC/." "$MODEL_LOCAL/"
fi
du -sh "$MODEL_LOCAL"

echo "== runtime config =="
python - <<PY
import yaml
cfg = yaml.safe_load(open("$CFG_SRC"))
cfg["policy"]["model_path"] = "$MODEL_LOCAL"
cfg["campaign_id"] = cfg["campaign_id"] + "_r3"
cfg["execution"]["runs_root"] = "$RSI_ROOT/runs"
yaml.safe_dump(cfg, open("$CFG_RUNTIME", "w"), sort_keys=False, allow_unicode=True)
print("model_path=", cfg["policy"]["model_path"], "campaign_id=", cfg["campaign_id"], "domains=", cfg["suite"]["domains"])
PY

echo "== data prepare / validate =="
python -m rsi.cli data prepare --config "$CFG_RUNTIME" | python -c "import json,sys; d=json.load(sys.stdin); assert d.get('ok'), d; print('  prepare:', d.get('split_manifest'))"
python -m rsi.cli data validate --config "$CFG_RUNTIME" | python -c "import json,sys; d=json.load(sys.stdin); assert d.get('ok'), d; print('  validate: ok')"

export RSI_PANEL_WORKERS=8
echo "== H0 complete-benchmark eval: 7 role pools =="
rc=0
for SPLIT in work_train harness_select weight_dev confirm_h confirm_w final_test transfer_test; do
  echo "---- eval split=$SPLIT harness=minimal ----"
  python -m rsi.cli eval --config "$CFG_RUNTIME" --harness minimal --split "$SPLIT" || rc=1
done
echo "== eval done rc=$rc =="
exit $rc
