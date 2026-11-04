#!/usr/bin/env bash
# RSI E1 (guide §12): 9B ALFWorld evolution — 8 generations x 2 candidates on
# the frozen 12+12 official-gated manifest. Analysis runner feeds REAL
# action/observation steps into the packet (E1-pre steps persistence);
# dedup + PROPOSAL_STALLED prevent duplicate-proposal burn; baseline (H0)
# retained per PR-5 selection semantics.
# Precondition: E0 G1-G4 passed (hf4ax).
set -euo pipefail
RSI_ROOT=/mnt/songzijun/Capability_Evolution/RSI
MODEL_SRC=/mnt/songzijun/models/Qwen3.5-9B
MODEL_LOCAL=/tmp/rsi_models/Qwen3.5-9B
RUNTIME_CFG=$RSI_ROOT/configs/runtime_e1_alfworld_9b_r3.yaml
export PYTHONPATH="$RSI_ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
cd "$RSI_ROOT"

echo "== dep preflight (E0-proven block) =="
python - <<'PY'
import importlib, importlib.util, subprocess, sys
MIRROR = "https://pkgs.d.xiaomi.net/artifactory/api/pypi/pypi-virtual/simple"
TORCH_PIN = "torch==2.10.0"
TF_PIN = "transformers==5.17.0"
MIN_GPUS = 8

def pip(*args, limit=900):
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
    pip(*need)

import torch, transformers
n = torch.cuda.device_count() if torch.cuda.is_available() else 0
print("torch", torch.__version__, "devices", n, flush=True)
if not (torch.cuda.is_available() and n >= MIN_GPUS):
    sys.exit("[FATAL] CUDA unusable")
from transformers.models import qwen3_5  # noqa: F401
pip("flash-linear-attention==0.6.0", limit=600)
pip("accelerate", limit=300)  # tiny_hf low_cpu_mem_usage streaming loader
# r3: causal-conv1d install attempt REMOVED — its sdist build pulls torch
# 2.15+cu132 into the isolated build env and times out fetching a prebuilt
# wheel from GitHub (~3.5 min wasted); even a successful build would be an
# ABI mismatch against runtime torch 2.10. The transformers reference
# fallback is slower but correct (E0 passed G1-G4 on it).
from transformers.utils import is_flash_linear_attention_available
if not is_flash_linear_attention_available():
    sys.exit("[FATAL] fla unavailable")
import os
fm = "/mnt/songzijun/datasets/alfworld/frozen_manifest.json"
if not os.path.isfile(fm):
    sys.exit(f"[FATAL] alfworld frozen manifest missing: {fm}")
# Domain deps: textworld/alfworld runtime (eval r7-proven block; fresh nodes
# start WITHOUT them — appworld r1 died on exactly this). Panel workers hit
# these imports per-task, so a missing dep would skip the whole domain.
for _rf in ("requirements.txt", "requirements-full.txt", "requirements-vis.txt"):
    if not os.path.isfile(f"/mnt/songzijun/datasets/alfworld_src/{_rf}"):
        sys.exit(f"[FATAL] alfworld_src missing {_rf} — working tree polluted")
pip("termcolor", "tqdm", "nltk", "networkx", limit=600)
# fast-downward-textworld: PddlEnv hard dep (y9qbm/v9urg — without it every
# task errors "PddlEnv requires fast_downward" but the job exits 0).
pip("fast-downward-textworld", limit=600)
if importlib.util.find_spec("textworld") is None:
    pip("textworld", limit=600)
_r = pip("--no-deps", "-e", "/mnt/songzijun/datasets/alfworld_src", limit=900)
if _r != 0:
    sys.exit("[FATAL] pip install -e alfworld_src failed — no fallback")
# PEP 660 .pth only processes at interpreter start: verify in a FRESH process
# (e21lh — same-process import of a just-installed editable always fails).
_v = subprocess.run([sys.executable, "-c",
    "import textworld, alfworld, fast_downward; fast_downward.load_lib(); print('tw+aw+fd ok')"],
    capture_output=True, text=True, timeout=120)
print(_v.stdout.strip() or _v.stderr.strip()[-300:], flush=True)
if _v.returncode != 0:
    sys.exit(f"[FATAL] alfworld/textworld unusable ({_v.stderr.strip()[-200:]})")
os.environ.setdefault("ALFWORLD_DATA", "/mnt/songzijun/datasets/alfworld")
# post-chain torch guard (4uavu lesson): fail loud before 8 GPUs burn
import torch as _t
print("torch after chain:", _t.__version__, "cuda_ok", _t.cuda.is_available(), flush=True)
if not (_t.__version__.startswith("2.10.") and _t.cuda.is_available()):
    sys.exit(f"[FATAL] torch drifted during dep chain: {_t.__version__} (need 2.10.x + cuda)")
print("SPLIT_OK + deps ready", flush=True)
PY

echo "== model pre-copy =="
if [ ! -f "$MODEL_LOCAL/config.json" ]; then
  rm -rf "$MODEL_LOCAL"; mkdir -p "$MODEL_LOCAL"
  cp -r "$MODEL_SRC/." "$MODEL_LOCAL/"
fi
du -sh "$MODEL_LOCAL"

echo "== runtime config =="
export ALFWORLD_DATA=/mnt/songzijun/datasets/alfworld
python - <<PY
import yaml
cfg = yaml.safe_load(open("$RSI_ROOT/configs/rsi_alfworld_react_9b_v3.yaml"))
cfg["policy"]["model_path"] = "$MODEL_LOCAL"
cfg["campaign_id"] = "e1_alfworld_9b_r4"
cfg["execution"]["runs_root"] = "$RSI_ROOT/runs"
cfg["suite"]["split_manifest"] = "data/splits/e1_alfworld_9b.json"
yaml.safe_dump(cfg, open("$RUNTIME_CFG", "w"), sort_keys=False, allow_unicode=True)

print("cfg ready:", cfg["campaign_id"], "profile:", cfg["runtime"]["profile"])
PY

echo "== data prepare / validate =="
python -m rsi.cli data prepare --config $RUNTIME_CFG | python -c "import json,sys; d=json.load(sys.stdin); assert d.get('ok') and d.get('frozen'), d; print('  prepare frozen:', d['domains']['alfworld'])"
python -m rsi.cli data validate --config $RUNTIME_CFG | python -c "import json,sys; d=json.load(sys.stdin); assert d.get('ok'), d; print('  validate ok')"

# ---- provider load-cache prewarm (bwozi hang fix, r7-proven) ----
# 8 spawned panel workers each calling a cold _load() walk 4027 games + ~4k
# traj reads on JuiceFS concurrently → metadata amplification stalls the whole
# panel (observed: 0% CPU 1h+, zero tasks). Materialize /tmp cache in ONE
# process first; workers then read it back (keyed by tree_digest).
echo "== provider load-cache prewarm =="
RSI_ALFWORLD_LOAD_CACHE=/tmp/rsi_alfworld_load_cache.json python - <<'PY'
import sys, time
sys.path.insert(0, "src")
from rsi.environments.task_providers import AlfWorldTaskProvider
t0 = time.time()
p = AlfWorldTaskProvider()
n = len(p.public_tasks("harness_select"))
print(f"prewarm done: {time.time()-t0:.0f}s n_harness_select={n}", flush=True)
assert n > 0
PY
ls -la /tmp/rsi_alfworld_load_cache.json
export RSI_ALFWORLD_LOAD_CACHE=/tmp/rsi_alfworld_load_cache.json

echo "== E1 evolution: 8 gens x 2 candidates, real-trace analysis =="
export RSI_PANEL_WORKERS=7
export RSI_PROPOSER_DEVICE=cuda:7
python -m rsi.cli evolve --config $RUNTIME_CFG 2>&1 | tee /tmp/e1_evolve_stream.log | python -u -c "
import sys
for line in sys.stdin:
    if any(k in line for k in ('generation', 'champion', 'stopped', 'candidate', 'error', 'Error',
                               'panel_worker', 'alfworld_provider', 'prewarm')):
        sys.stdout.write(line)
        sys.stdout.flush()
"
echo "== E1 evolve done =="
ls -la "$RSI_ROOT/runs/e1_alfworld_9b_r4/evolve/" || true
exit 0
