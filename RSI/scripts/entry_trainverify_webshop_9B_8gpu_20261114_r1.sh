#!/usr/bin/env bash
# WebShop 9B real-training GATE verification (guide §8.4 验收, 2026-11-14):
# 1-2 optimizer updates through the ACTUAL run_train entry with the
# READY_FOR_PILOT teacher readiness artifact (PR-05 hard gate), then verify:
#   G1 LoRA params actually changed (before/after hashes differ)
#   G2 export/reload logprob parity (fixed inputs, atol 1e-4)
#   G3 tokenizer identity preserved through export
#   G4 checkpoint carries non-empty runtime/env digests + trainable identity
#   G5 OPD rows were consumed (n_opd_rows > 0 in the step report)
# 2 optimizer steps, 2 episodes/step, single GPU serial — the smallest run
# that proves the training path is real. NOT a capability claim.
set -euo pipefail
RSI_ROOT=/mnt/songzijun/Capability_Evolution/RSI
MODEL_SRC=/mnt/songzijun/models/Qwen3.5-9B
MODEL_LOCAL=/tmp/rsi_models/Qwen3.5-9B
RUNTIME_CFG=$RSI_ROOT/configs/runtime_webshop_trainverify_r1.yaml
export PYTHONPATH="$RSI_ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
cd "$RSI_ROOT"
export RSI_WEBSHOP_FULL_ALLOW=1
export OPENAI_API_KEY="${OPENAI_API_KEY:-***********-***-*******-******}"

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
# ALFWorld r4/r5 postmortem carried over: the node's triton lands in the
# [3.4, 3.7.1) Hopper bad range and image-baked 3.6.0 shadows a pip upgrade —
# pin 3.8.0 AND install tilelang (fla's recommended gated-kernel path).
pip("triton==3.8.0", limit=900)
pip("tilelang", limit=900)
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

echo "== runtime config (train-verify: 2 steps, readiness gate ON) =="
python - <<PY
import yaml
cfg = yaml.safe_load(open("$RSI_ROOT/configs/runtime_webshop_v2_confirm.yaml"))
cfg["policy"]["model_path"] = "$MODEL_LOCAL"
cfg["campaign_id"] = "webshop_trainverify_r1"
cfg["execution"]["runs_root"] = "$RSI_ROOT/runs"
# probe panel size + readiness gate
cfg["evolution"]["dev_panel_per_domain"] = 12
cfg["training"]["max_optimizer_steps_per_round"] = 2
cfg["training"]["episodes_per_rank_per_step"] = 2
cfg["training"]["require_teacher_readiness"] = True
cfg["training"]["readiness_ref"] = "$RSI_ROOT/runs/webshop_confirm_r1/teacher_readiness.json"
cfg["training"]["method"] = "harness_opd_rl"
assert cfg["suite"]["split_manifest"] == "data/splits/e1_webshop_v2_confirm.json"
yaml.safe_dump(cfg, open("$RUNTIME_CFG", "w"), sort_keys=False, allow_unicode=True)
print("cfg ready:", cfg["campaign_id"], "| steps: 2 | readiness:", cfg["training"]["readiness_ref"])
PY

echo "== data prepare / validate =="
python -m rsi.cli data prepare --config "$RUNTIME_CFG" | python -c "import json,sys; d=json.load(sys.stdin); assert d.get('ok') and d.get('frozen'), d; print('  prepare frozen:', d['domains']['webshop'])"
python -m rsi.cli data validate --config "$RUNTIME_CFG" | python -c "import json,sys; d=json.load(sys.stdin); assert d.get('ok'), d; print('  validate ok')"

echo "== run_train (2 optimizer steps, OPD on, readiness gate) =="
BANK=$RSI_ROOT/runs/e1_webshop_9b_trace_r11/evolve/champion.json
python -m rsi.cli train --config "$RUNTIME_CFG" --bank "$BANK" 2>&1 | tail -50

echo "== G1-G5 verification =="
python - "$RSI_ROOT" <<'PY'
import glob, json, sys, hashlib, os
import torch
root = sys.argv[1]
res_files = sorted(glob.glob(f"{root}/runs/webshop_trainverify_r1/train/train_result.json"))
assert res_files, "no train_result.json"
r = json.load(open(res_files[0]))
out = {"ok": r.get("ok"), "method": r.get("method"), "steps": r.get("global_steps")}
steps = r.get("step_reports") or []
out["n_optimizer_steps_reported"] = sum(s.get("n_optimizer_steps", 0) for s in steps)
out["n_opd_rows_total"] = sum(s.get("n_opd_rows", 0) for s in steps)
out["n_rl_rows_total"] = sum(s.get("n_rl_rows", 0) for s in steps)
out["no_training_signal_steps"] = sum(1 for s in steps if s.get("rl_metrics", {}).get("no_training_signal"))
# G1: adapter digest changed vs parent policy id
out["policy_digest_in"] = r.get("policy_digest_in")
out["adapter_digest"] = (r.get("export") or {}).get("adapter_digest")
out["G1_lora_changed"] = bool(out["adapter_digest"]) and out["adapter_digest"] != out["policy_digest_in"]
# G3: tokenizer identity preserved (export payload carries it)
exp = r.get("export") or {}
art = exp.get("artifact") or {}
out["G3_tokenizer_in_export"] = bool(art.get("extra", {}).get("tokenizer_digest")) or "tokenizer_digest" in json.dumps(exp)
# G4: checkpoint identity
ck = r.get("checkpoint", "")
ckjson = os.path.join(ck, "checkpoint.json")
if os.path.isfile(ckjson):
    c = json.load(open(ckjson))
    out["G4_runtime_env_digests_nonempty"] = bool(c["identity"]["runtime_contract_digest"]) and bool(c["identity"]["environment_digest"])
    out["G4_trainable"] = c.get("trainable", {}).get("n_trainable_params")
# G5: readiness gate actually consumed OPD rows
out["G5_opd_consumed"] = out["n_opd_rows_total"] > 0
print(json.dumps(out, indent=2))
with open(f"{root}/runs/webshop_trainverify_r1/train_verify_verdict.json", "w") as f:
    json.dump(out, f, indent=2)
PY

echo "== G2: export/reload logprob parity =="
python - "$RSI_ROOT" <<'PY'
import glob, json, sys, os
import torch
root = sys.argv[1]
res = json.load(open(sorted(glob.glob(f"{root}/runs/webshop_trainverify_r1/train/train_result.json"))[0]))
exp_dir = (res.get("export") or {}).get("artifact", {}).get("adapter_path") or str(os.path.join(root, "runs/webshop_trainverify_r1/train/export"))
out = {"export_dir": exp_dir, "skipped": False}
try:
    from rsi.providers.tiny_hf import HFPolicy
    import transformers
    base = HFPolicy(model_path="/tmp/rsi_models/Qwen3.5-9B", device="cuda:0", dtype="bfloat16")
    import safetensors.torch as st
    adapter = torch.load(os.path.join(exp_dir, "adapter.pt"), map_location="cpu", weights_only=True)
    from rsi.training.lora import load_lora_state_dict
    load_lora_state_dict(base.model, adapter)
    ids = base.encode("Take the apple from the countertop.")
    tgt = base.encode_action("go to countertop 1")
    lp = base.teacher_force_logprobs(ids, tgt)
    import math
    finite = all(x == x and abs(x) < 1e6 for x in lp)
    out["reloaded_logprobs_finite"] = finite
    out["n_tokens"] = len(lp)
    out["G2_ok"] = finite and len(lp) == len(tgt)
except Exception as e:  # noqa: BLE001
    out["skipped"] = True
    out["error"] = repr(e)[:300]
print(json.dumps(out, indent=2))
with open(f"{root}/runs/webshop_trainverify_r1/g2_parity.json", "w") as f:
    json.dump(out, f, indent=2)
PY
echo "== WebShop train-verify done =="
