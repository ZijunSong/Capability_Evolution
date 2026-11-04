#!/usr/bin/env bash
# RSI E1 (guide §12): 9B AppWorld evolution — 8 generations x 2 candidates on
# the frozen 12+12 official-gated manifest. Analysis runner feeds REAL
# action/observation steps into the packet (E1-pre steps persistence);
# dedup + PROPOSAL_STALLED prevent duplicate-proposal burn; baseline (H0)
# retained per PR-5 selection semantics.
# Precondition: E0 G1-G4 passed (hf4ax).
set -euo pipefail
RSI_ROOT=/mnt/songzijun/Capability_Evolution/RSI
MODEL_SRC=/mnt/songzijun/models/Qwen3.5-9B
MODEL_LOCAL=/tmp/rsi_models/Qwen3.5-9B
export PYTHONPATH="$RSI_ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
cd "$RSI_ROOT"

echo "== dep preflight (E0-proven block) =="
python - <<'PY'
import importlib, subprocess, sys
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
_r = pip("causal-conv1d", limit=600)
if _r != 0:
    # sdist build needs setuptools+nvcc; the reference fallback is slower but
    # correct (E0 passed G1-G4 on it) — log and continue.
    print("[WARN] causal-conv1d unavailable — slow reference fallback (correct)", flush=True)
from transformers.utils import is_flash_linear_attention_available
if not is_flash_linear_attention_available():
    sys.exit("[FATAL] fla unavailable")
# Fresh nodes start WITHOUT appworld: the E0 lesson was that image state does
# not persist across jobs — always run the full vendored-repo install block.
try:
    import appworld
    print("appworld importable", flush=True)
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
    import appworld as _aw
    print("appworld installed from vendored repo", flush=True)
import subprocess as _sp
_sp.run([sys.executable, "-m", "appworld.cli", "install"], check=False, timeout=300)
import appworld.apps.admin  # noqa: F401
import os
ds = "/mnt/songzijun/datasets/appworld/data/datasets"
for f in ("train.txt", "dev.txt", "test_normal.txt", "test_challenge.txt"):
    if not os.path.isfile(os.path.join(ds, f)):
        sys.exit(f"[FATAL] official dataset listing missing: {ds}/{f}")
print("OFFICIAL_SPLIT_OK + deps ready", flush=True)
PY

echo "== model pre-copy =="
if [ ! -f "$MODEL_LOCAL/config.json" ]; then
  rm -rf "$MODEL_LOCAL"; mkdir -p "$MODEL_LOCAL"
  cp -r "$MODEL_SRC/." "$MODEL_LOCAL/"
fi
du -sh "$MODEL_LOCAL"

echo "== runtime config =="
python - <<PY
import yaml, os, shutil
cfg = yaml.safe_load(open("$RSI_ROOT/configs/rsi_appworld_react_9b_v3.yaml"))
cfg["policy"]["model_path"] = "$MODEL_LOCAL"
cfg["campaign_id"] = "e1_appworld_9b_r1"
cfg["execution"]["runs_root"] = "$RSI_ROOT/runs"
cfg["suite"]["split_manifest"] = "data/splits/e1_appworld_9b.json"
yaml.safe_dump(cfg, open("/tmp/e1_appworld_cfg.yaml", "w"), sort_keys=False, allow_unicode=True)
shutil.copy("$RSI_ROOT/data/splits/e1_appworld_9b.json", "/tmp/data/splits/e1_appworld_9b.json")
print("cfg ready:", cfg["campaign_id"], "profile:", cfg["runtime"]["profile"])
PY

echo "== data prepare / validate =="
python -m rsi.cli data prepare --config /tmp/e1_appworld_cfg.yaml | python -c "import json,sys; d=json.load(sys.stdin); assert d.get('ok') and d.get('frozen'), d; print('  prepare frozen:', d['domains']['appworld'])"
python -m rsi.cli data validate --config /tmp/e1_appworld_cfg.yaml | python -c "import json,sys; d=json.load(sys.stdin); assert d.get('ok'), d; print('  validate ok')"

echo "== E1 evolution: 8 gens x 2 candidates, real-trace analysis =="
export RSI_PANEL_WORKERS=7
export RSI_PROPOSER_DEVICE=cuda:7
python -m rsi.cli evolve --config /tmp/e1_appworld_cfg.yaml 2>&1 | tee /tmp/e1_evolve_stream.log | python -u -c "
import sys
for line in sys.stdin:
    if any(k in line for k in ('generation', 'champion', 'stopped', 'candidate', 'error', 'Error')):
        sys.stdout.write(line)
        sys.stdout.flush()
"
echo "== E1 evolve done =="
ls -la "$RSI_ROOT/runs/e1_appworld_9b_r1/evolve/" || true
exit 0
