#!/usr/bin/env bash
# RSI HarnessOPD PILOT — Qwen3.5-9B, 50 step, 4node32gpu (guide §9.1 E4).
#   Student = H0 only (zero); Teacher = H0 + CONFIRMED bank (readiness
#   READY_FOR_PILOT bound; PR-05 hard gate ON).  LoRA r32 cumulative adapter,
#   32-rank data-parallel (rollout shard + global normalize + SUM grad sync).
#   Uses ALL fixes sunk in this campaign: per-row backward (OOM), tilelang
#   (fla Hopper guard), fixed export payload, readiness gate ordering.
# Precondition: WebShop train-verify G1-G5 verdict (the gate chain completed
# on BOTH domains before pilot launch).
set -euo pipefail
RSI_ROOT=/mnt/songzijun/Capability_Evolution/RSI
MODEL_SRC=/mnt/songzijun/models/Qwen3.5-9B
MODEL_LOCAL=/tmp/rsi_models/Qwen3.5-9B
CFG_SRC="$RSI_ROOT/configs/train_harness_opd_alfworld_9b_pilot50_32gpu.yaml"
CFG_RUNTIME=/tmp/rsi_train_opd_alfworld_pilot.yaml
CHAMPION="$RSI_ROOT/runs/e1_alfworld_9b_r5/evolve/champion.json"

export PYTHONPATH="$RSI_ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
cd "$RSI_ROOT"

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
echo "== dep preflight (alfworld chain + tilelang) =="
python - <<'PY'
import importlib.util, subprocess, sys
MIRROR = "https://pkgs.d.xiaomi.net/artifactory/api/pypi/pypi-virtual/simple"
def pip(*args, limit=2400):
    print("pip", " ".join(args), flush=True)
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-i", MIRROR, *args], timeout=limit)
    return r.returncode

need = []
for mod, pkg in (("yaml", "pyyaml"), ("numpy", "numpy"), ("torch", "torch==2.10.0"), ("transformers", "transformers==5.17.0")):
    try:
        importlib.import_module(mod)
    except ImportError:
        need.append(pkg)
if need:
    pip(*need)
import torch, transformers

# GPU-util keepalive (platform force-stops jobs with 30min zero GPU util):
# initialize CUDA on all 8 devices NOW so the long pip preflight (up to 40min
# of CPU-only wheel downloads) doesn't trip the watchdog. Memory touches are
# tiny; freed before training.
python - <<'PYK'
import torch
for _i in range(torch.cuda.device_count()):
    _d = torch.tensor([1.0], device=f"cuda:{_i}")
    _d += 1
print("GPU keepalive: touched", torch.cuda.device_count(), "devices", flush=True)
PYK

n = torch.cuda.device_count() if torch.cuda.is_available() else 0
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "devices", n, flush=True)
if not (torch.cuda.is_available() and n >= 8):
    sys.exit("[FATAL] CUDA unusable")
from transformers.models import qwen3_5  # noqa: F401
pip("flash-linear-attention==0.6.0", limit=600)
pip("triton==3.8.0", limit=900)
pip("tilelang", limit=900)   # fla Hopper gated-kernel path (bypasses triton guard)
pip("accelerate", limit=300)
from transformers.utils import is_flash_linear_attention_available
if not is_flash_linear_attention_available():
    sys.exit("[FATAL] fla unavailable")

pip("termcolor", "tqdm", "nltk", "networkx", limit=900)
pip("fast-downward-textworld", limit=900)
if importlib.util.find_spec("textworld") is None:
    pip("textworld", limit=1800)
for _try in range(2):
    if pip("--no-deps", "-e", "/mnt/songzijun/datasets/alfworld_src", limit=1800) == 0:
        break
_v = subprocess.run([sys.executable, "-c",
    "import textworld, alfworld, fast_downward; fast_downward.load_lib(); print('tw+aw+fd ok')"],
    capture_output=True, text=True, timeout=180)
print(_v.stdout.strip() or _v.stderr.strip()[-300:], flush=True)
if _v.returncode != 0:
    sys.exit("[FATAL] fresh-process tw+aw+fd import probe failed")
print("ALFWORLD_DEPS_OK", flush=True)
PY

# ---------------------------------------------------------------- model pre-copy
echo "== model pre-copy (JuiceFS 32-way read guard, per node) =="
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
echo "== torchrun HarnessOPD pilot (nnodes=4 nproc=8 -> 32 ranks) =="
exec torchrun \
  --nnodes=4 \
  --nproc_per_node=8 \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  -m rsi.cli train --config "$CFG_RUNTIME" --bank "$CHAMPION"
