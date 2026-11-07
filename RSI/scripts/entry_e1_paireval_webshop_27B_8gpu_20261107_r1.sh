#!/usr/bin/env bash
# RSI E1 webshop-9B paired final eval: baseline (minimal/H0) vs evolved
# champion bank (accbce6a: goal_completion_guard + action_recovery_guard),
# same frozen harness_select 12-task split, same seed (17), 7 GPU panel
# workers. Two sequential arms in one job; per-task records + suite summary
# land under runs/e1_webshop_9b_r9/pair_eval/.
# Precondition: bojde evolve succeeded (champion promoted, PR-5 selection).
set -euo pipefail
RSI_ROOT=/mnt/songzijun/Capability_Evolution/RSI
MODEL_SRC=/mnt/songzijun/models/Qwen3.5-27B
MODEL_LOCAL=/tmp/rsi_models/Qwen3.5-27B
CHAMPION_BANK=$RSI_ROOT/runs/e1_webshop_27b_r12/candidates/1e6cec6576bf3d318f1f6695a79a3fbb4228959feaebc828bd5ddb1641ef08da/bank
RUNTIME_CFG=$RSI_ROOT/configs/runtime_e1_webshop_27b_r12_pair.yaml
export PYTHONPATH="$RSI_ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
cd "$RSI_ROOT"

echo "== dep preflight (bojde-proven block) =="
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
pip("accelerate", limit=300)
from transformers.utils import is_flash_linear_attention_available
if not is_flash_linear_attention_available():
    sys.exit("[FATAL] fla unavailable")
import os, site
fm = "/mnt/songzijun/datasets/webshop/frozen_goals.json"
if not os.path.isfile(fm):
    sys.exit(f"[FATAL] webshop frozen goals missing: {fm}")
# Domain deps (r12-proven chain) + torch re-pin inside the chain (4uavu) +
# both BASE_DIR-external symlinks (n2bxo: <SP>/data; hfenw: <SP>/search_engine).
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
import torch as _t
print("torch after chain:", _t.__version__, "cuda_ok", _t.cuda.is_available(), flush=True)
if not (_t.__version__.startswith("2.10.") and _t.cuda.is_available()):
    sys.exit(f"[FATAL] torch drifted during dep chain: {_t.__version__} (need 2.10.x + cuda)")
print("webshop deps ok")
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
# Official web_agent_site resolves ALL data/index paths relative to BASE_DIR
# (= installed package dir): ../data/* and ../search_engine/indexes. After a
# site-packages unpack both are missing — symlink the real trees in (bojde r8
# proven pair; both must print or the fix is not applied).
for _name in ("search_engine", "data"):
    _dst = os.path.join(SP_DIR, _name)
    if not os.path.exists(_dst):
        os.symlink(os.path.join("/mnt/songzijun/datasets/webshop", _name), _dst)
        print(f"{_name} symlinked into site-packages")
if not (os.path.isdir(os.path.join(SP_DIR, "data")) and os.path.isdir(os.path.join(SP_DIR, "search_engine"))):
    sys.exit("[FATAL] BASE_DIR-external symlinks missing after setup")
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

echo "== runtime config =="
python - <<PY
import yaml
cfg = yaml.safe_load(open("$RSI_ROOT/configs/rsi_webshop_react_27b_v3.yaml"))
cfg["policy"]["model_path"] = "$MODEL_LOCAL"
cfg["campaign_id"] = "e1_webshop_27b_r12_pair"
cfg["execution"]["runs_root"] = "$RSI_ROOT/runs"
cfg["suite"]["split_manifest"] = "data/splits/e1_webshop_9b.json"
yaml.safe_dump(cfg, open("$RUNTIME_CFG", "w"), sort_keys=False, allow_unicode=True)
print("cfg ready:", cfg["campaign_id"], "profile:", cfg["runtime"]["profile"])
PY

echo "== data prepare / validate =="
python -m rsi.cli data prepare --config $RUNTIME_CFG | python -c "import json,sys; d=json.load(sys.stdin); assert d.get('ok') and d.get('frozen'), d; print('  prepare frozen:', d['domains']['webshop'])"
python -m rsi.cli data validate --config $RUNTIME_CFG | python -c "import json,sys; d=json.load(sys.stdin); assert d.get('ok'), d; print('  validate ok')"

echo "== paired eval: arm A baseline (minimal/H0) then arm B champion (bank) =="
export RSI_WEBSHOP_FULL_ALLOW=1
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-rsi-offline-placeholder}"
export RSI_PANEL_WORKERS=7

python -m rsi.cli eval --config $RUNTIME_CFG --harness minimal --split harness_select \
  2>&1 | tee /tmp/paireval_armA.log | tail -5
# (arm A/B results are persisted by run_eval(write=True) under runs/<campaign>/eval/)

python -m rsi.cli eval --config $RUNTIME_CFG --harness bank --bank "$CHAMPION_BANK" --split harness_select \
  2>&1 | tee /tmp/paireval_armB.log | tail -5

echo "== paired eval done; collecting =="
python - <<PY
import json, glob, os
out = {}
import glob
out = {}
pats = {
    "baseline_minimal": "$RSI_ROOT/runs/e1_webshop_9b_r9_pair/eval/eval_harness_select_minimal_*.json",
    "champion_bank": "$RSI_ROOT/runs/e1_webshop_9b_r9_pair/eval/eval_harness_select_bank_*.json",
}
for arm, pat in pats.items():
    hits = sorted(glob.glob(pat))
    if not hits:
        out[arm] = {"error": f"no eval json matches {pat}"}
        continue
    obj = json.load(open(hits[-1]))
    # run_eval(write=True) nests under "result": {suite, per_task, ...}
    if "suite" not in obj and isinstance(obj.get("result"), dict):
        obj = obj["result"]
    out[arm] = obj
pair_dir = "$RSI_ROOT/runs/e1_webshop_9b_r9_pair/pair_eval"
os.makedirs(pair_dir, exist_ok=True)
with open(os.path.join(pair_dir, "paired_result.json"), "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
rows = {}
for arm, obj in out.items():
    s = (obj or {}).get("suite") or {}
    macro = s.get("macro_score")
    rows[arm] = macro
    doms = {d["domain"]: round(d["score"], 4) for d in s.get("domains", [])}
    n = {d["domain"]: d["n"] for d in s.get("domains", [])}
    print(f"{arm}: macro={macro} domains={doms} n={n}")
b, c = rows.get("baseline_minimal"), rows.get("champion_bank")
if isinstance(b, float) and isinstance(c, float):
    print(f"PAIRED DELTA (champion - baseline): {c - b:+.4f}")
    with open(os.path.join(pair_dir, "paired_summary.txt"), "w") as f:
        f.write(f"baseline_minimal={b}\nchampion_bank={c}\ndelta={c-b:+.6f}\n")
PY
exit 0
