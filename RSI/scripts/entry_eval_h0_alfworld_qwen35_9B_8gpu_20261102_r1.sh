#!/usr/bin/env bash
# RSI H0 基线完整测评（完整 ALFWorld benchmark）— Qwen3.5-9B，单节点 8xM402
# 布局：panel workers x8（测评无 proposer，8 卡全作 worker）
# 完整 benchmark：7 个 role 池全量 task 逐个评（harness minimal / official engine won）
#   final=valid_unseen(134) transfer=valid_seen(140) dev 五池=train(3553) hash 分组
#   → 共 5266 task。instruction=官方 turk_annotations.anns[0].task_desc。
#
# 基线（2026-11-02）：fork entry_eval_h0_appworld_qwen35_9B_8gpu_20261102_r2.sh
# （dhlol 已验证：torch==2.10.0+cu128 钉死+CUDA 硬门、fla==0.6.0 钉死+
# is_flash_linear_attention 硬断言、causal-conv1d --no-build-isolation 回退、
# parallel.py 错峰加载+心跳+ProcessPoolExecutor）。差异：
#   ① appworld 依赖块 → alfworld 依赖块（textworld/alfworld/jnius 不需 pyserini，
#     alfworld pip -e --no-deps 从 frozen 源码树装；ALFWORLD_DATA 环境变量钉路径）。
#   ② campaign_id → eval_h0_alfworld_qwen35_9b_bench_r1 → 全新 OUT。
set -euo pipefail
RSI_ROOT=/mnt/songzijun/Capability_Evolution/RSI
MODEL_SRC=/mnt/songzijun/models/Qwen3.5-9B
MODEL_LOCAL=/tmp/rsi_models/Qwen3.5-9B
CFG_SRC="$RSI_ROOT/configs/eval_h0_qwen35_9B_alfworld_8gpu.yaml"
CFG_RUNTIME=/tmp/rsi_eval_h0_alfworld_cfg_9B_r1.yaml
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

# 4) Qwen3.5 的线性注意力/卷积核（同 dhlol：缺核=数倍吞吐损失，fla 硬断言）
print("kernel deps:", flush=True)
pip("flash-linear-attention==0.6.0", limit=600)
_r = pip("causal-conv1d", limit=600)
if _r != 0:
    _r = pip("--no-build-isolation", "causal-conv1d", limit=900)
print(("  causal-conv1d: installed" if _r == 0 else
       "  causal-conv1d: MISSING (slow reference fallback; sdist-only + no nvcc?)"), flush=True)
# ⚠️ 包名 flash-linear-attention 的导入名是 fla（wtmif/r4 踩过 importlib 验错名）
try:
    from transformers.utils import is_flash_linear_attention_available, is_causal_conv1d_available
    fla_ok = bool(is_flash_linear_attention_available())
    conv_ok = bool(is_causal_conv1d_available())
    print(f"  fla fused kernels: {'ok' if fla_ok else 'MISSING'} | causal_conv1d: {'ok' if conv_ok else 'MISSING'}", flush=True)
    if not fla_ok:
        sys.exit("[FATAL] flash-linear-attention(fla) not available — gated-delta-rule would fall back to slow PyTorch")
except ImportError as e:  # noqa: BLE001
    sys.exit(f"[FATAL] transformers kernel-availability API missing ({e}); transformers too old?")

# 5) alfworld 依赖块（textworld 运行时 + frozen 源码树 pip -e --no-deps）。
#    textworld 依赖 room2text/nltk 等，pip 会从镜像解析；alfworld 包本体从
#    /mnt/songzijun/datasets/alfworld_src 装（prepare_alfworld_data.py 在 dev 已装过
#    dev 机 site-packages，但 CloudML 节点是全新容器，必须重装）。
pip("termcolor", "tqdm", "nltk", "networkx", limit=600)
if importlib.util.find_spec("textworld") is None:
    pip("textworld", limit=600)
_r = pip("--no-deps", "-e", "/mnt/songzijun/datasets/alfworld_src", limit=600)
if _r != 0:
    sys.exit("[FATAL] pip install -e alfworld_src failed — ALFWorld package required, no fallback")
try:
    import textworld  # noqa: F401
    print("textworld", textworld.__version__, "ok")
    import alfworld  # noqa: F401
    print("alfworld import: ok")
except Exception as e:  # noqa: BLE001
    sys.exit(f"[FATAL] alfworld/textworld not importable ({e})")
# tw-textworld 游戏解析还要 textworld 的额外引擎数据（不需要 th/jvm）。
#alfworld 引擎生成 game.tw-pddl 已物化（prepare 阶段完成），eval 只 replay。
PY

export ALFWORLD_DATA=/mnt/songzijun/datasets/alfworld
echo "ALFWORLD_DATA=$ALFWORLD_DATA"

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
cfg["campaign_id"] = cfg["campaign_id"] + "_r1"
cfg["execution"]["runs_root"] = "$RSI_ROOT/runs"
yaml.safe_dump(cfg, open("$CFG_RUNTIME", "w"), sort_keys=False, allow_unicode=True)
print("model_path=", cfg["policy"]["model_path"], "campaign_id=", cfg["campaign_id"], "domains=", cfg["suite"]["domains"])
PY

echo "== data prepare / validate =="
python -m rsi.cli data prepare --config "$CFG_RUNTIME" | python -c "import json,sys; d=json.load(sys.stdin); assert d.get('ok'), d; print('  prepare:', d.get('split_manifest'))"
python -m rsi.cli data validate --config "$CFG_RUNTIME" | python -c "import json,sys; d=json.load(sys.stdin); assert d.get('ok'), d; print('  validate: ok')"

echo "== instruction placeholder audit (must be 0 after schema fix) =="
python - <<'PY'
import sys
sys.path.insert(0, "src")
from rsi.environments.task_providers import AlfWorldTaskProvider
p = AlfWorldTaskProvider()
bad = []
n = 0
for role in ("final_test", "transfer_test", "confirm_h", "confirm_w"):
    for t in p.public_tasks(role):
        n += 1
        if t.instruction.startswith("(alfworld task") or not t.instruction.strip():
            bad.append(t.task_id)
print(f"panel tasks checked={n} placeholder={len(bad)}")
if bad:
    sys.exit(f"[FATAL] {len(bad)} placeholder instructions — extract_instruction schema regression: {bad[:3]}")
PY

export RSI_PANEL_WORKERS=8
echo "== H0 complete-benchmark eval: 7 role pools =="
rc=0
for SPLIT in work_train harness_select weight_dev confirm_h confirm_w final_test transfer_test; do
  echo "---- eval split=$SPLIT harness=minimal ----"
  python -m rsi.cli eval --config "$CFG_RUNTIME" --harness minimal --split "$SPLIT" || rc=1
done
echo "== eval done rc=$rc =="
exit $rc
