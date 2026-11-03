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
#   ② campaign_id → eval_h0_alfworld_qwen35_9b_bench_r2 → 全新 OUT。
#
# r4（2026-11-03，bwozi hang 复盘）：r3 dep 块全过、eval 起跑，但 8 个 spawn worker
# 各自冷 AlfWorldTaskProvider._load() 并发爬 JuiceFS 全树（5266 games + 5k traj 读）
# → 元数据互相放大，GPU 0%/CPU~0% 1h+ 零 task done。修复：① provider _load 加
# /tmp 进程间缓存（RSI_ALFWORLD_LOAD_CACHE，键=tree_digest，原子写）首个进程物化
# 其余直读；② entry 在 eval 前单进程预热缓存，worker spawn 后直读零爬盘。
# r3（2026-11-03，e21lh 复盘）：r2 前置硬门过了、pip -e 成功返回 0，但同一 python 进程内
# 立即 import alfworld 报 No module named 'alfworld'——PEP 660 editable finder 是
# __editable__*.pth，只在解释器启动时处理；同进程装完立即 import 必挂（textworld 是常规
# wheel 所以 ok）。修复：① import 验证改用全新子进程（.pth 生效）；② pip() 捕
# TimeoutExpired 返 124（r9mla/viz4o 同款教训）；③ -e limit 600→900s（实测 build
# isolation 拉 setuptools ~6.5min）；④ campaign_id _r3 → 全新 OUT。
# r2（2026-11-03，lmtw2 失败复盘）：lmtw2 排队 6h 后调度成功，但 pip install -e
# alfworld_src 在 metadata 阶段死于 FileNotFoundError: requirements-vis.txt——
# alfworld_src 工作树被污染（git index 488 文件 staged 删除，requirements-vis.txt
# 已从文件系统消失；dev 机 11-02 09:39 的拷贝型安装在文件丢失前完成故未暴露）。
# 修复：① git show HEAD:requirements-vis.txt 恢复文件（setup.py extras_require
# 无条件 open() 它）；② r2 entry 加 REQUIREMENTS_FILES 前置硬门——装包前逐个
# test -f setup.py 引用的三个 requirements*.txt，缺一 fail-fast（读得清失败原因，
# 不再烧到 pip traceback）；③ campaign_id + _r2 → 全新 OUT。
set -euo pipefail
RSI_ROOT=/mnt/songzijun/Capability_Evolution/RSI
MODEL_SRC=/mnt/songzijun/models/Qwen3.5-9B
MODEL_LOCAL=/tmp/rsi_models/Qwen3.5-9B
CFG_SRC="$RSI_ROOT/configs/eval_h0_qwen35_9B_alfworld_8gpu.yaml"
CFG_RUNTIME=/tmp/rsi_eval_h0_alfworld_cfg_9B_r4.yaml
export PYTHONPATH="$RSI_ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
cd "$RSI_ROOT"

echo "== dep preflight =="
python - <<'PY'
import importlib, os, subprocess, sys

MIRROR = "https://pkgs.d.xiaomi.net/artifactory/api/pypi/pypi-virtual/simple"
TORCH_PIN = "torch==2.10.0"          # cu128 族，与 gm402 driver 匹配
TF_PIN = "transformers==5.17.0"      # ≥5.6 才有 qwen3_5 注册
MIN_GPUS = 8

def pip(*args, limit=900):
    """Bounded pip install against MIRROR; returns exit code (never raises)."""
    print("pip", " ".join(args), flush=True)
    try:
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-i", MIRROR, *args],
                           timeout=limit)
    except subprocess.TimeoutExpired:
        print(f"pip TIMEOUT after {limit}s:", args, flush=True)
        return 124
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
#    前置硬门：setup.py extras_require 无条件 open() 三个 requirements*.txt，
#    lmtw2 死于 requirements-vis.txt 从工作树消失（FileNotFoundError in
#    get_requires_for_build_editable）。装包前先验文件在，缺一 fail-fast。
for _rf in ("requirements.txt", "requirements-full.txt", "requirements-vis.txt"):
    if not os.path.isfile(f"/mnt/songzijun/datasets/alfworld_src/{_rf}"):
        sys.exit(f"[FATAL] alfworld_src missing {_rf} — working tree polluted; restore via git show HEAD:{_rf}")
print("  requirements files: all present")
#    textworld 依赖 room2text/nltk 等，pip 会从镜像解析；alfworld 包本体从
#    /mnt/songzijun/datasets/alfworld_src 装（prepare_alfworld_data.py 在 dev 已装过
#    dev 机 site-packages，但 CloudML 节点是全新容器，必须重装）。
pip("termcolor", "tqdm", "nltk", "networkx", limit=600)
if importlib.util.find_spec("textworld") is None:
    pip("textworld", limit=600)
_r = pip("--no-deps", "-e", "/mnt/songzijun/datasets/alfworld_src", limit=900)
if _r != 0:
    sys.exit("[FATAL] pip install -e alfworld_src failed — ALFWorld package required, no fallback")
# ⚠️ e21lh 复盘：pip -e 的 PEP 660 finder 是 __editable__*.pth，只在**解释器启动时**处理——
# 同一个 python 进程内装完立即 import 必挂 "No module named alfworld"（textworld 是常规 wheel
# 不受影响，所以日志里 textworld ok + alfworld 失败）。验证必须用**全新子进程**（.pth 生效）。
_v = subprocess.run([sys.executable, "-c", "import textworld, alfworld; "
                     "print('textworld', textworld.__version__, '| alfworld', alfworld.__file__)"],
                    capture_output=True, text=True, timeout=120)
print(_v.stdout.strip() or _v.stderr.strip()[-400:], flush=True)
if _v.returncode != 0:
    sys.exit(f"[FATAL] alfworld/textworld not importable in fresh process ({_v.stderr.strip()[-300:]})")
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
cfg["campaign_id"] = cfg["campaign_id"] + "_r4"
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

# ---- provider load-cache prewarm（bwozi hang 修复）----
# 8 个 spawn worker 各自冷 _load() 会并发爬 JuiceFS 全树+读 5k traj → 元数据互相放大
# （观测：GPU 0% CPU~0% 1h+ 零 task）。这里先单进程物化 /tmp 缓存，worker spawn 后
# RSI_ALFWORLD_LOAD_CACHE 直读（键=tree_digest，脏缓存自动跳过）。
echo "== provider load-cache prewarm (single-process walk + traj read) =="
RSI_ALFWORLD_LOAD_CACHE=/tmp/rsi_alfworld_load_cache.json python - <<'PY'
import sys, time
sys.path.insert(0, "src")
from rsi.environments.task_providers import AlfWorldTaskProvider
t0 = time.time()
p = AlfWorldTaskProvider()
n = len(p.public_tasks("final_test"))
print(f"prewarm done: {time.time()-t0:.0f}s n_final_test={n}", flush=True)
assert n > 0
PY
ls -la /tmp/rsi_alfworld_load_cache.json
export RSI_ALFWORLD_LOAD_CACHE=/tmp/rsi_alfworld_load_cache.json

export RSI_PANEL_WORKERS=8
echo "== H0 complete-benchmark eval: 7 role pools =="
rc=0
for SPLIT in work_train harness_select weight_dev confirm_h confirm_w final_test transfer_test; do
  echo "---- eval split=$SPLIT harness=minimal ----"
  python -m rsi.cli eval --config "$CFG_RUNTIME" --harness minimal --split "$SPLIT" || rc=1
done
echo "== eval done rc=$rc =="
exit $rc
