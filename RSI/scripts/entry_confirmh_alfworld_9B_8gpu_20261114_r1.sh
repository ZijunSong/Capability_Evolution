#!/usr/bin/env bash
# ALFWorld fresh confirm_h (guide §5.2/§5.3, 2026-11-14): the FROZEN candidate
# 622625de (bank c31e0e88, pre-registered in runs/confirm_selection_freeze.json)
# vs H0 on the v2 manifest's confirm_h panel (732 tasks; panel_per_domain=24
# => first 24 fresh groups, ~3 actor replicates budget per group start).
# Topology: SERIAL in-process panels (the only 5-hang-proof topology; propose
# JuiceFS scratch fix applies to appworld only — alfworld runs TextWorld in
# process memory + local replay digests, no JuiceFS metadata churn).
# Gates for READY evidence (guide §5.3 #1): delta_h_strict > 0 with cost/
# regression info complete; UNRESOLVED_INTERVAL if CI crosses zero (24 paired
# groups may be too few — report honestly, do NOT resample to pass).
set -euo pipefail
RSI_ROOT=/mnt/songzijun/Capability_Evolution/RSI
MODEL_SRC=/mnt/songzijun/models/Qwen3.5-9B
MODEL_LOCAL=/tmp/rsi_models/Qwen3.5-9B
RUNTIME_CFG=$RSI_ROOT/configs/runtime_alfworld_v2_confirm_r1.yaml
export PYTHONPATH="$RSI_ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
cd "$RSI_ROOT"

echo "== dep preflight (alfworld chain) =="
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

for mod in ("textworld", "termcolor", "tqdm"):
    try:
        __import__(mod)
    except ImportError:
        pip(mod, limit=1800)  # textworld pulls numpy/pandas chain — 600s timed out on r1
try:
    import alfworld  # noqa: F401
    print("alfworld importable", flush=True)
except ImportError:
    # vendored editable install; its dependency resolution is slow on a cold
    # node — retry once after a generous per-attempt budget (r1 died at 600s)
    for _try in range(2):
        if pip("--no-deps", "/mnt/songzijun/datasets/alfworld_src", limit=1800) == 0:
            break
    import alfworld  # noqa: F401
print("ALFWORLD_DEPS_OK", flush=True)
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
cfg = yaml.safe_load(open("$RSI_ROOT/configs/runtime_alfworld_v2_confirm.yaml"))
cfg["policy"]["model_path"] = "$MODEL_LOCAL"
cfg["campaign_id"] = "alfworld_confirm_r1"
cfg["execution"]["runs_root"] = "$RSI_ROOT/runs"
assert cfg["suite"]["split_manifest"] == "data/splits/e1_alfworld_v2_confirm.json"
assert cfg["runtime"]["profile"] == "react_zero_shot_v1"
yaml.safe_dump(cfg, open("$RUNTIME_CFG", "w"), sort_keys=False, allow_unicode=True)
print("cfg ready:", cfg["campaign_id"], "| manifest:", cfg["suite"]["split_manifest"])
PY

echo "== data prepare / validate =="
python -m rsi.cli data prepare --config "$RUNTIME_CFG" | python -c "import json,sys; d=json.load(sys.stdin); assert d.get('ok') and d.get('frozen'), d; print('  prepare frozen:', d['domains']['alfworld'])"
python -m rsi.cli data validate --config "$RUNTIME_CFG" | python -c "import json,sys; d=json.load(sys.stdin); assert d.get('ok'), d; print('  validate ok')"

echo "== fresh confirm: armA H0 vs armB frozen bank 622625de, serial panels =="
BANK=$RSI_ROOT/runs/e1_alfworld_9b_r5/evolve/champion.json
python3 - "$BANK" <<'PY'
import json, sys
raw = json.load(open(sys.argv[1]))
assert raw["bank_digest"].startswith("c31e0e88"), raw["bank_digest"]
print("frozen candidate verified:", raw["candidate_id"][:12], raw["bank_digest"][:16])
PY
python -m rsi.cli confirm-harness --config "$RUNTIME_CFG" --bank "$BANK" --panel-per-domain 24 --workers 1 2>&1 | tail -80

echo "== verdict (guide §5.3 gates) =="
python - "$RSI_ROOT" <<'PY'
import glob, json, sys
root = sys.argv[1]
files = sorted(glob.glob(f"{root}/runs/alfworld_confirm_r1/confirm_h/confirm_h_r0.json"))
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
with open(f"{root}/runs/alfworld_confirm_r1/confirm_verdict.json", "w") as f:
    json.dump(out, f, indent=2)
PY
echo "== ALFWorld fresh confirm done =="
