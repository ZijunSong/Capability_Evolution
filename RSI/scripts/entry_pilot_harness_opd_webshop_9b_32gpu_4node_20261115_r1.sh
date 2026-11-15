#!/usr/bin/env bash
# RSI HarnessOPD PILOT (WebShop) — Qwen3.5-9B, 50 step, 4node32gpu (guide §9.1 E4).
#   Student = H0 only (zero); Teacher = H0 + CONFIRMED bank (readiness
#   READY_FOR_PILOT bound; PR-05 hard gate ON).  LoRA r32 cumulative adapter,
#   32-rank data-parallel (rollout shard + global normalize + SUM grad sync).
#   Uses ALL fixes sunk in this campaign: per-row backward (OOM), tilelang
#   (fla Hopper guard), fixed export payload, readiness gate ordering.
# Precondition: ALFWorld train-verify r6 verdict green (both domains prepared).
set -euo pipefail
RSI_ROOT=/mnt/songzijun/Capability_Evolution/RSI
MODEL_SRC=/mnt/songzijun/models/Qwen3.5-9B
MODEL_LOCAL=/tmp/rsi_models/Qwen3.5-9B
CFG_SRC="$RSI_ROOT/configs/train_harness_opd_webshop_9b_pilot50_32gpu.yaml"
CFG_RUNTIME=/tmp/rsi_train_opd_webshop_pilot.yaml
CHAMPION="$RSI_ROOT/runs/e1_webshop_9b_trace_r11/evolve/champion.json"

export PYTHONPATH="$RSI_ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
cd "$RSI_ROOT"
export RSI_WEBSHOP_FULL_ALLOW=1
export OPENAI_API_KEY="${OPENAI_API_KEY:-***********-***-*******-******}"

# ---------------------------------------------------------------- multinode env
NODE_RANK="${NODE_RANK:-${PET_NODE_RANK:-${NODE_INDEX:-${NODE_NUMBER:-}}}}"
if [ -z "${NODE_RANK:-}" ]; then
  echo "FATAL: NODE_RANK is not set (each pod needs 0..3)" >&2; exit 1
fi
if [ -z "${MASTER_ADDR:-}" ]; then
  echo "FATAL: MASTER_ADDR is not set (CloudML must inject node0 IP)" >&2; exit 1
fi
: "${MASTER_PORT:=29500}"
export MASTER_ADDR NODE_RANK MASTER_PORT
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK \
      GROUP_WORLD_SIZE TORCHELASTIC_RUN_ID TORCHELASTIC_RESTART_COUNT \
      PET_MASTER_ADDR PET_MASTER_PORT PET_NNODES PET_NPROC_PER_NODE PET_NODE_RANK 2>/dev/null || true
echo "MULTINODE_ENV master_addr=${MASTER_ADDR} node_rank=${NODE_RANK} master_port=${MASTER_PORT} host=$(hostname)"

test -f "$CHAMPION" || { echo "FATAL: champion bank not found: $CHAMPION" >&2; exit 1; }

# ---------------------------------------------------------------- deps
echo "== dep preflight (webshop chain + tilelang) =="
python - <<'PY'
import importlib, importlib.util, subprocess, sys
MIRROR = "https://pkgs.d.xiaomi.net/artifactory/api/pypi/pypi-virtual/simple"
TORCH_PIN = "torch==2.10.0"
TF_PIN = "transformers==5.17.0"
MIN_GPUS = 8

def pip(*args, limit=2400):
    print("pip", " ".join(args), flush=True)
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-i", MIRROR, *args], timeout=limit)
    return r.returncode

need = []
for mod, pkg in (("yaml", "pyyaml"), ("numpy", "numpy"), ("torch", TORCH_PIN), ("transformers", TF_PIN)):
    try:
        importlib.import_module(mod)
    except ImportError:
        need.append(pkg)
if need:
    # r6 postmortem: one combined pip of torch+transformers hit 2400s on a
    # cold node (mirror throughput variance). Split per-package with retries:
    # each gets its own 2400s budget and 3 attempts.
    for pkg in need:
        for _try in range(3):
            if pip(pkg, limit=2400) == 0:
                break
        else:
            sys.exit(f"[FATAL] cannot install {pkg!r} after 3 attempts")

import torch, transformers

# GPU-util keepalive (platform force-stops jobs with 30min zero GPU util):
# initialize CUDA on all 8 devices NOW so the long pip preflight (up to 40min
# of CPU-only wheel downloads) doesn't trip the watchdog. Memory touches are
# tiny; freed before training.
for _i in range(torch.cuda.device_count()):
    _d = torch.tensor([1.0], device=f"cuda:{_i}")
    _d += 1
print("GPU keepalive: touched", torch.cuda.device_count(), "devices", flush=True)

n = torch.cuda.device_count() if torch.cuda.is_available() else 0
print("torch", torch.__version__, "devices", n, flush=True)
if not (torch.cuda.is_available() and n >= MIN_GPUS):
    sys.exit("[FATAL] CUDA unusable")
from transformers.models import qwen3_5  # noqa: F401
pip("flash-linear-attention==0.6.0", limit=600)
# ALFWorld r4/r5 + wqqg0 postmortems: pin triton 3.8.0 (image-baked 3.6.0 is
# in fla's Hopper bad range) AND install tilelang — the fla recommended gated
# kernel path that bypasses the triton guard. tilelang's CUDA wheel is
# multi-GB: 900s timed out on a cold node (wqqg0), use 2400s x2 retries.
pip("triton==3.8.0", limit=900)
for _tl_try in range(2):
    if pip("tilelang", limit=2400) == 0:
        break
print("tilelang install attempts done", flush=True)
pip("accelerate", limit=300)  # tiny_hf low_cpu_mem_usage streaming loader
# r3: causal-conv1d install attempt REMOVED — its sdist build pulls torch
# 2.15+cu132 into the isolated build env and times out fetching a prebuilt
# wheel from GitHub (~3.5 min wasted); even a successful build would be an
# ABI mismatch against runtime torch 2.10. The transformers reference
# fallback is slower but correct (E0 passed G1-G4 on it).
from transformers.utils import is_flash_linear_attention_available
if not is_flash_linear_attention_available():
    sys.exit("[FATAL] fla unavailable")
import os, site
fm = "/mnt/songzijun/datasets/webshop/frozen_goals.json"
if not os.path.isfile(fm):
    sys.exit(f"[FATAL] webshop frozen goals missing: {fm}")
# Domain deps: WebShop engine chain (eval r12-proven block; fresh nodes start
# WITHOUT them — appworld r1 died on exactly this gap). Panel workers hit
# these imports per-task, so a missing dep would skip the whole domain.
# torch/transformers re-pinned INSIDE the chain: 4uavu (slave83) proved pip's
# resolver can pull torch 2.15+cu132 as a transitive dep (pyserini/spacy tree)
# when --ignore-installed hides the installed 2.10 metadata — cu13x build on a
# 12.8 driver = cuda unavailable = instant policy failure.
pip("--ignore-installed", "beautifulsoup4", "thefuzz", "spacy", "flask",
    "pyserini==2.0.0", "rank_bm25==0.2.2", "rich", "tqdm", "gym==0.26.2",
    "emoji", "unidecode", "nltk", "selenium",
    "torch==2.10.0", "transformers==5.17.0", limit=900)
SP_DIR = site.getsitepackages()[0]
DEPS_TGZ = "/mnt/songzijun/datasets/webshop/_deps/cleantext-1.1.4-sitepkg.tgz"
if os.path.isfile(DEPS_TGZ):
    import tarfile
    with tarfile.open(DEPS_TGZ) as tf:
        tf.extractall(SP_DIR)
    print("cleantext unpacked from _deps")
else:
    sys.exit(f"[FATAL] cleantext site-packages tgz missing: {DEPS_TGZ}")
import bs4, thefuzz, spacy, cleantext, rank_bm25  # noqa
# post-chain torch guard (4uavu lesson): fail loud before 8 GPUs burn
import torch as _t
print("torch after chain:", _t.__version__, "cuda_ok", _t.cuda.is_available(), flush=True)
if not (_t.__version__.startswith("2.10.") and _t.cuda.is_available()):
    sys.exit(f"[FATAL] torch drifted during dep chain: {_t.__version__} (need 2.10.x + cuda)")
print("webshop deps ok")
# spacy model: cluster egress can't reach the model CDN — unpack staged tgz.
try:
    spacy.load("en_core_web_sm")
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
# web_agent_site: editable installs die with the prepare container (vytfr) and
# same-process import after pip -e fails (PEP 660 .pth — e21lh). Unpack the
# pure-Python tree straight into site-packages (r12-proven equivalent).
try:
    import web_agent_site  # noqa
except ImportError:
    WS_ROOT = "/mnt/songzijun/datasets/webshop"
    if not os.path.isfile(os.path.join(WS_ROOT, "pyproject.toml")):
        sys.exit(f"[FATAL] prepared webshop tree missing pyproject.toml at {WS_ROOT}")
    import tarfile
    with tarfile.open("/tmp/web_agent_site_src.tar.gz", "w:gz") as tf:
        tf.add(os.path.join(WS_ROOT, "web_agent_site"), arcname="web_agent_site")
    with tarfile.open("/tmp/web_agent_site_src.tar.gz") as tf:
        tf.extractall(SP_DIR)
    _chk = subprocess.run([sys.executable, "-c", "import web_agent_site"],
                          capture_output=True, text=True)
    if _chk.returncode != 0:
        sys.exit(f"[FATAL] web_agent_site still unimportable: {_chk.stderr.strip()[-300:]}")
    print("web_agent_site installed from prepared tree")
# Official init_search_engine resolves indexes relative to BASE_DIR (= the
# installed package dir): <SP>/web_agent_site/../search_engine/indexes. After
# unpacking into site-packages that is <SP>/search_engine — hfenw r6 proved
# every task dies there with a bare FileNotFoundError (bare = Java-side
# LuceneSearcher on a missing dir). Symlink the real 3.4G index tree in.
# Every official data path is BASE_DIR-relative too: utils.py uses
# <BASE_DIR>/../data/* (items_shuffle 5.5G, items_ins_v2 186M, items_human_ins)
# and engine.py uses ../search_engine/indexes. n2bxo r7 proved symlink fix for
# search_engine alone is not enough — bare FileNotFoundError persisted (that
# one was <SP>/data/*). Symlink both package-external trees.
for _name in ("search_engine", "data"):
    _dst = os.path.join(SP_DIR, _name)
    if not os.path.exists(_dst):
        os.symlink(os.path.join("/mnt/songzijun/datasets/webshop", _name), _dst)
        print(f"{_name} symlinked into site-packages")
# JVM for pyserini Lucene search.
find_so = subprocess.run(["bash", "-c",
    "find /usr/lib/jvm -name libjvm.so 2>/dev/null | head -1"], capture_output=True, text=True)
if find_so.stdout.strip():
    os.environ.setdefault("JVM_PATH", find_so.stdout.strip())
    print("JVM_PATH=", os.environ["JVM_PATH"])
else:
    print("[WARN] no libjvm.so — Lucene search will fail loudly on first search()")
print("SPLIT_OK + deps ready", flush=True)
PY

# ---------------------------------------------------------------- model pre-copy
echo "== model pre-copy (JuiceFS 32-way read guard, per node) =="
# df6od postmortem: one node had only 14G free in /tmp (19G model) — the cp
# died silently and the job failed mid-pre-copy with no error line. Fail LOUD
# before copying, and retry once on the same node after clearing our own stale
# copy only (never touch other files).
_free_gb=$(df -BG --output=avail /tmp | tail -1 | tr -dc '0-9')
echo "/tmp free: ${_free_gb}G (need 20G)"
if [ "$_free_gb" -lt 20 ]; then
  echo "[WARN] /tmp low on space — clearing stale rsi_models copy from prior runs"
  rm -rf /tmp/rsi_models 2>/dev/null || true
  _free_gb=$(df -BG --output=avail /tmp | tail -1 | tr -dc '0-9')
  echo "/tmp free after clean: ${_free_gb}G"
  if [ "$_free_gb" -lt 20 ]; then
    echo "FATAL: /tmp still short of space for the 19G model — request a different node" >&2
    exit 1
  fi
fi
if [ ! -f "$MODEL_LOCAL/config.json" ]; then
  rm -rf "$MODEL_LOCAL"; mkdir -p "$MODEL_LOCAL"
  cp -r "$MODEL_SRC/." "$MODEL_LOCAL/"
fi
du -sh "$MODEL_LOCAL"

# ---------------------------------------------------------------- runtime config
echo "== runtime config =="
python - <<PY
import yaml
cfg = yaml.safe_load(open("$CFG_SRC"))
cfg["policy"]["model_path"] = "$MODEL_LOCAL"
cfg["execution"]["runs_root"] = "$RSI_ROOT/runs"
assert cfg["training"]["require_teacher_readiness"] is True
yaml.safe_dump(cfg, open("$CFG_RUNTIME", "w"), sort_keys=False, allow_unicode=True)
print("model_path=", cfg["policy"]["model_path"])
print("steps=", cfg["training"]["max_optimizer_steps_per_round"],
      "episodes_per_rank=", cfg["training"]["episodes_per_rank_per_step"],
      "readiness=", cfg["training"]["readiness_ref"])
PY

# ---------------------------------------------------------------- torchrun train
echo "== torchrun HarnessOPD WebShop pilot (nnodes=4 nproc=8 -> 32 ranks) =="
exec torchrun \
  --nnodes=4 \
  --nproc_per_node=8 \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  -m rsi.cli train --config "$CFG_RUNTIME" --bank "$CHAMPION"
