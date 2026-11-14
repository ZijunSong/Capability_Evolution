#!/usr/bin/env bash
# WebShop fresh confirm_h (guide §5.2/§5.3, 2026-11-14): the PRE-FROZEN PENDING candidate
# bd397b10 (bank a691cf12, trace-r11, pre-registered in runs/confirm_selection_freeze.json)
# vs H0 on the v2 manifest's confirm_h panel (2214 tasks; panel_per_domain=24
# => first 24 fresh groups, ~3 actor replicates budget per group start).
# Topology: SERIAL in-process panels (the only 5-hang-proof topology; propose
# JuiceFS scratch fix applies to appworld only — webshop runs the SimServer
# in process memory + local replay digests, no JuiceFS metadata churn).
# Gates for READY evidence (guide §5.3 #1): delta_h_strict > 0 with cost/
# regression info complete; UNRESOLVED_INTERVAL if CI crosses zero (24 paired
# groups may be too few — report honestly, do NOT resample to pass).
set -euo pipefail
RSI_ROOT=/mnt/songzijun/Capability_Evolution/RSI
MODEL_SRC=/mnt/songzijun/models/Qwen3.5-9B
MODEL_LOCAL=/tmp/rsi_models/Qwen3.5-9B
RUNTIME_CFG=$RSI_ROOT/configs/runtime_webshop_v2_confirm_r1.yaml
export PYTHONPATH="$RSI_ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
cd "$RSI_ROOT"
export RSI_WEBSHOP_FULL_ALLOW=1   # flavor=all execution gate (r12-proven; 224GB CloudML node) — r1 silently skipped the whole domain without it

echo "== dep preflight (webshop chain, eval-r12-proven) =="
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

echo "== model pre-copy =="
if [ ! -f "$MODEL_LOCAL/config.json" ]; then
  rm -rf "$MODEL_LOCAL"; mkdir -p "$MODEL_LOCAL"
  cp -r "$MODEL_SRC/." "$MODEL_LOCAL/"
fi
du -sh "$MODEL_LOCAL"

echo "== runtime config (v2 confirm manifest + frozen candidate) =="
python - <<PY
import yaml
cfg = yaml.safe_load(open("$RSI_ROOT/configs/runtime_webshop_v2_confirm.yaml"))
cfg["policy"]["model_path"] = "$MODEL_LOCAL"
cfg["campaign_id"] = "webshop_confirm_r1"
cfg["execution"]["runs_root"] = "$RSI_ROOT/runs"
assert cfg["suite"]["split_manifest"] == "data/splits/e1_webshop_v2_confirm.json"
assert cfg["runtime"]["profile"] == "react_zero_shot_v1"
yaml.safe_dump(cfg, open("$RUNTIME_CFG", "w"), sort_keys=False, allow_unicode=True)
print("cfg ready:", cfg["campaign_id"], "| manifest:", cfg["suite"]["split_manifest"])
PY

echo "== data prepare / validate =="
python -m rsi.cli data prepare --config "$RUNTIME_CFG" | python -c "import json,sys; d=json.load(sys.stdin); assert d.get('ok') and d.get('frozen'), d; print('  prepare frozen:', d['domains']['webshop'])"
python -m rsi.cli data validate --config "$RUNTIME_CFG" | python -c "import json,sys; d=json.load(sys.stdin); assert d.get('ok'), d; print('  validate ok')"

echo "== fresh confirm: armA H0 vs armB frozen bank bd397b10, serial panels =="
BANK=$RSI_ROOT/runs/e1_webshop_9b_trace_r11/evolve/champion.json
python3 - "$BANK" <<'PY'
import json, sys
raw = json.load(open(sys.argv[1]))
assert raw["bank_digest"].startswith("a691cf12"), raw["bank_digest"]
print("frozen candidate verified:", raw["candidate_id"][:12], raw["bank_digest"][:16])
PY
python -m rsi.cli confirm-harness --config "$RUNTIME_CFG" --bank "$BANK" --panel-per-domain 24 --workers 1 2>&1 | tail -80

echo "== verdict (guide §5.3 gates) =="
python - "$RSI_ROOT" <<'PY'
import glob, json, sys
root = sys.argv[1]
files = sorted(glob.glob(f"{root}/runs/webshop_confirm_r1/confirm_h/confirm_h_r0.json"))
assert files, "no confirm report"
r = json.load(open(files[0]))
out = {
    "delta_h_macro_value": r.get("delta_h_macro"),
    "delta_h_macro_strict": r.get("delta_h_macro_strict"),
    "wins_losses_ties": r.get("wins_losses_ties"),
    "strict_wins_losses_ties": r.get("strict_wins_losses_ties"),
    "n_paired": sum(d.get("n_paired", 0) for d in (r.get("domains") or {}).values()),
    "interpretation": r.get("interpretation", "")[:200],
    "honest_label": (
        "GAIN_CONFIRMED_ALIGNMENT_PENDING" if (r.get("delta_h_macro_strict") or 0) > 0
        else "NO_CONFIRMED_GAIN_ON_THIS_PANEL"
    ),
}
print(json.dumps(out, indent=2))
with open(f"{root}/runs/webshop_confirm_r1/confirm_verdict.json", "w") as f:
    json.dump(out, f, indent=2)
PY
echo "== WebShop fresh confirm done =="
