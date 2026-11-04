#!/usr/bin/env bash
# RSI E1 (guide §12): 9B Webshop evolution — 8 generations x 2 candidates on
# the frozen 12+12 official-gated manifest. Analysis runner feeds REAL
# action/observation steps into the packet (E1-pre steps persistence);
# dedup + PROPOSAL_STALLED prevent duplicate-proposal burn; baseline (H0)
# retained per PR-5 selection semantics.
# Precondition: E0 G1-G4 passed (hf4ax).
set -euo pipefail
RSI_ROOT=/mnt/songzijun/Capability_Evolution/RSI
MODEL_SRC=/mnt/songzijun/models/Qwen3.5-27B
MODEL_LOCAL=/tmp/rsi_models/Qwen3.5-27B
RUNTIME_CFG=$RSI_ROOT/configs/runtime_e1_webshop_27b_r3.yaml
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
import os, site
fm = "/mnt/songzijun/datasets/webshop/frozen_goals.json"
if not os.path.isfile(fm):
    sys.exit(f"[FATAL] webshop frozen goals missing: {fm}")
# Domain deps: WebShop engine chain (eval r12-proven block; fresh nodes start
# WITHOUT them — appworld r1 died on exactly this gap). Panel workers hit
# these imports per-task, so a missing dep would skip the whole domain.
pip("--ignore-installed", "beautifulsoup4", "thefuzz", "spacy", "flask",
    "pyserini==2.0.0", "rank_bm25==0.2.2", "rich", "tqdm", "gym==0.26.2",
    "emoji", "unidecode", "nltk", "selenium", limit=900)
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

export RSI_WEBSHOP_FULL_ALLOW=1   # flavor=all execution gate (r12-proven; missing = probe gate domain-skips whole panel -> all-0.0 artifact)

echo "== model pre-copy =="
if [ ! -f "$MODEL_LOCAL/config.json" ]; then
  rm -rf "$MODEL_LOCAL"; mkdir -p "$MODEL_LOCAL"
  cp -r "$MODEL_SRC/." "$MODEL_LOCAL/"
fi
du -sh "$MODEL_LOCAL"

echo "== runtime config =="
python - <<PY
import yaml
cfg = yaml.safe_load(open("$RSI_ROOT/configs/rsi_webshop_react_9b_v3.yaml"))
cfg["policy"]["model_path"] = "$MODEL_LOCAL"
cfg["campaign_id"] = "e1_webshop_27b_r3"
cfg["execution"]["runs_root"] = "$RSI_ROOT/runs"
cfg["suite"]["split_manifest"] = "data/splits/e1_webshop_9b.json"
yaml.safe_dump(cfg, open("$RUNTIME_CFG", "w"), sort_keys=False, allow_unicode=True)

print("cfg ready:", cfg["campaign_id"], "profile:", cfg["runtime"]["profile"])
PY

echo "== data prepare / validate =="
python -m rsi.cli data prepare --config $RUNTIME_CFG | python -c "import json,sys; d=json.load(sys.stdin); assert d.get('ok') and d.get('frozen'), d; print('  prepare frozen:', d['domains']['webshop'])"
python -m rsi.cli data validate --config $RUNTIME_CFG | python -c "import json,sys; d=json.load(sys.stdin); assert d.get('ok'), d; print('  validate ok')"

echo "== E1 evolution: 8 gens x 2 candidates, real-trace analysis =="
export RSI_PANEL_WORKERS=7
export RSI_PROPOSER_DEVICE=cuda:7
python -m rsi.cli evolve --config $RUNTIME_CFG 2>&1 | tee /tmp/e1_evolve_stream.log | python -u -c "
import sys
for line in sys.stdin:
    if any(k in line for k in ('generation', 'champion', 'stopped', 'candidate', 'error', 'Error')):
        sys.stdout.write(line)
        sys.stdout.flush()
"
echo "== E1 evolve done =="
ls -la "$RSI_ROOT/runs/e1_webshop_27b_r3/evolve/" || true
exit 0
