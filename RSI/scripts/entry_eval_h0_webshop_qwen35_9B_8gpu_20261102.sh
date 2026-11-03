#!/usr/bin/env bash
# RSI H0 基线测评（WebShop，官方 500-goal human 面板 + 全 role 池）— Qwen3.5-9B，单节点 8xM402
# 布局：panel workers x8（测评无 proposer，8 卡全作 worker）
# 口径：官方 get_reward [0,1] + harsh success(reward==1)，frozen_goals 12087 冻结口径
#（frozen_goals[0:500] = final_test 官方面板，与 baseline_models/test.py range(500) 一致）。
#
# ⚠️ WebShop flavor=all 必须 224GB 大内存节点（官方 load_products 5.5GB json.load
#    峰值 >16GB，dev 容器 memcg 会 oom-kill；共享 SimServer 缓存已入 adapter——
#    每进程只构建一次，跨 task 会话按 goal index 隔离）。
# ⚠️ 先 smoke（work_train 前 8 task）再全量 7 pool：smoke 验证 vLLM→loop→adapter→
#    官方 scorer 全链路，smoke 不过则 fail-fast 不进全量。
set -euo pipefail
RSI_ROOT=/mnt/songzijun/Capability_Evolution/RSI
MODEL_SRC=/mnt/songzijun/models/Qwen3.5-9B
MODEL_LOCAL=/tmp/rsi_models/Qwen3.5-9B
CFG_SRC="$RSI_ROOT/configs/eval_h0_qwen35_9B_webshop_8gpu.yaml"
CFG_RUNTIME=/tmp/rsi_eval_h0_webshop_cfg_9B.yaml
export PYTHONPATH="$RSI_ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export RSI_WEBSHOP_FULL_ALLOW=1   # 224GB 节点放开 flavor=all 执行门
cd "$RSI_ROOT"

echo "== dep preflight =="
python - <<'PY'
import importlib, subprocess, sys

MIRROR = "https://pkgs.d.xiaomi.net/artifactory/api/pypi/pypi-virtual/simple"
TORCH_PIN = "torch==2.10.0"          # cu128 族，与 gm402 driver 匹配
TF_PIN = "transformers==5.17.0"      # ≥5.6 才有 qwen3_5 注册
MIN_GPUS = 8

def pip(*args, limit=900):
    """Bounded pip install against MIRROR; returns exit code (never raises).

    Timeout must be caught explicitly (r9mla: causal-conv1d sdist build blew
    the limit and subprocess.TimeoutExpired killed the whole entry — the
    returncode path alone does not cover it)."""
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

# 1) 基础包：缺失才装，装则按钉死版本（铁律：dep 块必须钉版本，防漂移到 cu132 毁 driver 匹配）
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
    return torch.__version__.startswith("2.10.") and torch.cuda.is_available() and n >= MIN_GPUS

if not audit():
    print(f"[WARN] torch/CUDA unusable or wrong version — forcing {TORCH_PIN} / {TF_PIN}", flush=True)
    pip("--force-reinstall", TORCH_PIN, TF_PIN)
    code = (
        "import torch, transformers, sys; "
        "n = torch.cuda.device_count() if torch.cuda.is_available() else 0; "
        "print('retry torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'devices', n); "
        "print('retry transformers', transformers.__version__); "
        "sys.exit(0 if torch.__version__.startswith('2.10.') and torch.cuda.is_available() and n >= %d else 1)" % MIN_GPUS
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

# 4) Qwen3.5 线性注意力核（fla 导入名是 fla，不是 flash_linear_attention——wtmif 踩过）
pip("flash-linear-attention==0.6.0", limit=600)
# causal-conv1d 是 best-effort（缺了走慢速参考实现，fla 才是硬断言）；
# r9mla：sdist 编译超时 TimeoutExpired 杀穿 entry —— 现已捕获，超时=走回退。
_r = pip("causal-conv1d", limit=600)
if _r != 0:
    _r = pip("--no-build-isolation", "causal-conv1d", limit=900)
try:
    from transformers.utils import is_flash_linear_attention_available, is_causal_conv1d_available
    fla_ok = bool(is_flash_linear_attention_available())
    conv_ok = bool(is_causal_conv1d_available())
    print(f"  fla fused kernels: {'ok' if fla_ok else 'MISSING'} | causal_conv1d: {'ok' if conv_ok else 'MISSING'}", flush=True)
    if not fla_ok:
        sys.exit("[FATAL] flash-linear-attention(fla) not available — gated-delta-rule would fall back to slow PyTorch")
except ImportError as e:  # noqa: BLE001
    sys.exit(f"[FATAL] transformers kernel-availability API missing ({e})")

# 5) WebShop 官方包依赖链（engine import 缺一不可；与 prepare entry 同款）
#    --ignore-installed: 镜像 distutils blinker 1.4 冲突会让整批 install 中止。
#    cleantext 不在内网 PyPI → _deps 站包 tgz 解到 site-packages。
pip("--ignore-installed", "beautifulsoup4", "thefuzz", "spacy", "flask",
    "pyserini==2.0.0", "rank_bm25==0.2.2", "rich", "tqdm", "gym==0.26.2",
    "emoji", "unidecode", "nltk", "selenium", limit=900)
SP_DIR = sysconfig_dir = None
import site, os
SP_DIR = site.getsitepackages()[0]
DEPS_TGZ = "/mnt/songzijun/datasets/webshop/_deps/cleantext-1.1.4-sitepkg.tgz"
if os.path.isfile(DEPS_TGZ):
    import tarfile
    with tarfile.open(DEPS_TGZ) as tf:
        tf.extractall(SP_DIR)
    print("cleantext unpacked from _deps")
else:
    sys.exit(f"[FATAL] cleantext site-packages tgz missing: {DEPS_TGZ}")
try:
    import bs4, thefuzz, spacy, cleantext, rank_bm25, rich, gym, emoji, unidecode, nltk, selenium  # noqa
    print("webshop deps ok")
except ImportError as e:
    sys.exit(f"[FATAL] webshop dep chain broken: {e!r}")
# spacy 模型（官方 get_reward 需要）。_deps 里是 site-packages tarball
#（-sitepkg.tgz，tar 解包式；集群出口不达 spacy 模型 CDN）——viz4o 踩坑：
# 误当 wheel 走 pip install 必败（neither setup.py nor pyproject.toml）。
# 与 prepare_webshop_data.py 同款 tar 解包。
try:
    import spacy
    spacy.load("en_core_web_sm")
    print("spacy model en_core_web_sm: ok")
except Exception:
    import glob
    cand = sorted(glob.glob("/mnt/songzijun/datasets/webshop/_deps/en_core_web_sm-*.tgz"))
    if not cand:
        sys.exit("[FATAL] spacy model en_core_web_sm unavailable and no _deps tgz staged")
    import tarfile
    with tarfile.open(cand[-1]) as tf:
        tf.extractall(SP_DIR)
    spacy.load("en_core_web_sm")
    print(f"spacy model installed from {cand[-1]}")

# 6) web_agent_site 必须已安装。editable 安装随 prepare job 容器消失（vytfr），
#    且 3wyxh 证明 pip -e 成功返回后**同进程 import 仍失败**——PEP 660 editable
#    finder 的 .pth 在本进程 site 处理时已过，同进程立即 import 不生效
#    （ALFWorld e21lh 同款坑）。修法：pip -e 后用【子进程】验证；子进程也失败
#    则放弃 editable，直接 tar 解包官方树进 site-packages（web_agent_site 是
#    纯 Python 包，等效且无 .pth 时序问题）。
try:
    import web_agent_site  # noqa
    print("web_agent_site import: ok")
except ImportError as e:
    print(f"web_agent_site not importable ({e!r}) — installing from prepared tree")
    WS_ROOT = "/mnt/songzijun/datasets/webshop"
    if not os.path.isfile(os.path.join(WS_ROOT, "pyproject.toml")):
        sys.exit(f"[FATAL] prepared webshop tree missing pyproject.toml at {WS_ROOT}")
    _r = pip("--no-deps", "--no-build-isolation", "-e", WS_ROOT, limit=600)
    # 子进程验证：fresh interpreter 走新 .pth
    _chk = subprocess.run([sys.executable, "-c", "import web_agent_site"],
                          capture_output=True, text=True)
    if _chk.returncode != 0:
        print(f"editable install ineffective in fresh interpreter ({_chk.stderr.strip()[-200:]}) "
              "— unpacking tree into site-packages")
        import tarfile
        _stamp = "/tmp/web_agent_site_unpacked.stamp"
        if not os.path.exists(_stamp):
            with tarfile.open("/tmp/web_agent_site_src.tar.gz", "w:gz") as tf:
                tf.add(os.path.join(WS_ROOT, "web_agent_site"), arcname="web_agent_site")
            with tarfile.open("/tmp/web_agent_site_src.tar.gz") as tf:
                tf.extractall(SP_DIR)
            open(_stamp, "w").write("ok")
        _chk = subprocess.run([sys.executable, "-c", "import web_agent_site"],
                              capture_output=True, text=True)
        if _chk.returncode != 0:
            sys.exit(f"[FATAL] web_agent_site still unimportable after unpack: {_chk.stderr.strip()[-300:]}")
    print("web_agent_site installed from prepared tree")

# 7) JVM（pyserini Lucene 搜索需要；pyserini 2.0 用 Java 11/21 均可）
find_so = subprocess.run(["bash", "-c",
    "find /usr/lib/jvm -name libjvm.so 2>/dev/null | head -1"], capture_output=True, text=True)
jvm = find_so.stdout.strip()
if jvm:
    os.environ.setdefault("JVM_PATH", jvm)
    print("JVM_PATH=", os.environ["JVM_PATH"])
else:
    print("[WARN] no libjvm.so found — Lucene search will fail loudly on first search()")

print("DEP_PREFLIGHT_OK")
PY

echo "== model pre-copy =="
if [ ! -f "$MODEL_LOCAL/config.json" ]; then
  rm -rf "$MODEL_LOCAL"; mkdir -p "$MODEL_LOCAL"
  cp -r "$MODEL_SRC/." "$MODEL_LOCAL/"
fi
du -sh "$MODEL_LOCAL"

echo "== runtime config =="
export CFG_RUNTIME=/tmp/rsi_eval_h0_webshop_cfg_9B.yaml
python - <<PY
import yaml
cfg = yaml.safe_load(open("$CFG_SRC"))
cfg["policy"]["model_path"] = "$MODEL_LOCAL"
cfg["campaign_id"] = cfg["campaign_id"]
cfg["execution"]["runs_root"] = "$RSI_ROOT/runs"
yaml.safe_dump(cfg, open("$CFG_RUNTIME", "w"), sort_keys=False, allow_unicode=True)
print("model_path=", cfg["policy"]["model_path"], "campaign_id=", cfg["campaign_id"], "domains=", cfg["suite"]["domains"])
PY

echo "== webshop capability probe (224GB flavor=all gate) =="
export RSI_WEBSHOP_FULL_ALLOW=1
python - <<'PY'
import sys
sys.path.insert(0, "/mnt/songzijun/Capability_Evolution/RSI/src")
from rsi.environments.registry import global_registry
reg = global_registry()
ok, why = reg.probe_executable("webshop")
print("probe:", ok, why)
if not ok:
    raise SystemExit(f"[FATAL] webshop not executable: {why}")
print("CAPABILITY_PROBE_OK")
PY

echo "== data prepare / validate =="
python -m rsi.cli data prepare --config "$CFG_RUNTIME" | python -c "import json,sys; d=json.load(sys.stdin); assert d.get('ok'), d; print('  prepare: ok')"
python -m rsi.cli data validate --config "$CFG_RUNTIME" | python -c "import json,sys; d=json.load(sys.stdin); assert d.get('ok'), d; print('  validate: ok')"

export RSI_PANEL_WORKERS=8
echo "== SMOKE: work_train first 8 tasks via max_per_domain gate =="
# ⚠️ smoke 不能用 `python - <<heredoc`：spawn 子进程回读 __main__ 时 '<stdin>' 必炸（5tesh）。
# ⚠️ 落盘文件还不够：spawn 会重新 import main module，顶层代码不守卫就递归再起进程
#    ——`_check_not_importing_main` RuntimeError（wzu2g 实证）。必须 `if __name__ == "__main__":`。
cat > /tmp/rsi_webshop_smoke.py <<'PY'
import os, sys, json
sys.path.insert(0, "/mnt/songzijun/Capability_Evolution/RSI/src")
from rsi.config import load_config
from rsi.evaluation.service import run_eval

def main() -> int:
    cfg = load_config(os.environ.get("CFG_RUNTIME", "/tmp/rsi_eval_h0_webshop_cfg_9B.yaml"))
    # smoke: 8 个 work_train task（run_eval 的 max_per_domain 走 service 层切片，不经 provider env）
    res = run_eval(cfg, harness="minimal", split="work_train", write=True, max_per_domain=8)
    pt = res["result"]["per_task"]
    vals = [m["value"] for m in pt.values()]
    reasons = [m["terminated_reason"] for m in pt.values()]
    print("SMOKE n_tasks:", len(pt), "values:", vals, "reasons:", reasons)
    errs = [m for m in pt.values() if m.get("synthetic")]
    if len(pt) != 8:
        print(f"[FATAL] smoke expected 8 tasks, got {len(pt)}")
        return 1
    if errs:
        print(f"[FATAL] smoke had synthetic/error records: {errs}")
        return 1
    print("WEBSHOP_EVAL_SMOKE_OK")
    return 0

if __name__ == "__main__":
    sys.exit(main())
PY
python /tmp/rsi_webshop_smoke.py

echo "== FULL: 7 role pools =="
rc=0
for SPLIT in work_train harness_select weight_dev confirm_h confirm_w final_test transfer_test; do
  echo "---- eval split=$SPLIT harness=minimal ----"
  python -m rsi.cli eval --config "$CFG_RUNTIME" --harness minimal --split "$SPLIT" || rc=1
done
echo "== eval done rc=$rc =="
exit $rc
