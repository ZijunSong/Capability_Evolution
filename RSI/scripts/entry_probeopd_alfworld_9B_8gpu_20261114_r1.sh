#!/usr/bin/env bash
# ALFWorld probe-opd 试采 (guide §5.3 门 2/3, 2026-11-14): real distillability
# probe with the CONFIRMED teacher bank 622625de (c31e0e88) — engineering
# feasibility check, NOT statistics: real student H0 decision points, teacher
# forks at the decision snapshot with the exact public prefix, native anchors,
# legal targets, teacher-scored targets, OPD gradient contribution.
# Topology: SERIAL in-process (five-hang-proof). PR-06 guarantees the teacher
# branch starts from the decision's own snapshot + exact prefix.
set -euo pipefail
RSI_ROOT=/mnt/songzijun/Capability_Evolution/RSI
MODEL_SRC=/mnt/songzijun/models/Qwen3.5-9B
MODEL_LOCAL=/tmp/rsi_models/Qwen3.5-9B
RUNTIME_CFG=$RSI_ROOT/configs/runtime_alfworld_probe_r1.yaml
export PYTHONPATH="$RSI_ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
cd "$RSI_ROOT"

echo "== dep preflight (proven e1 chain) =="
python - <<'PY'
import importlib.util, subprocess, sys
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

echo "== model pre-copy =="
if [ ! -f "$MODEL_LOCAL/config.json" ]; then
  rm -rf "$MODEL_LOCAL"; mkdir -p "$MODEL_LOCAL"
  cp -r "$MODEL_SRC/." "$MODEL_LOCAL/"
fi
du -sh "$MODEL_LOCAL"

echo "== runtime config =="
python - <<PY
import yaml
cfg = yaml.safe_load(open("$RSI_ROOT/configs/runtime_alfworld_v2_confirm.yaml"))
cfg["policy"]["model_path"] = "$MODEL_LOCAL"
cfg["campaign_id"] = "alfworld_probe_r1"
cfg["execution"]["runs_root"] = "$RSI_ROOT/runs"
# r1 postmortem: the v2-confirm config carries dev_panel_per_domain=0 (an
# eval-era inheritance), which made the probe slice work_train[:0] — an EMPTY
# panel: total_decisions=0, INSUFFICIENT_OPD_SUPPORT with zero evidence. The
# probe panel size lives here (guide §5.3: 12-24 tasks).
cfg["evolution"]["dev_panel_per_domain"] = 12
assert cfg["suite"]["split_manifest"] == "data/splits/e1_alfworld_v2_confirm.json"
yaml.safe_dump(cfg, open("$RUNTIME_CFG", "w"), sort_keys=False, allow_unicode=True)
print("cfg ready:", cfg["campaign_id"])
PY

echo "== data prepare / validate =="
python -m rsi.cli data prepare --config "$RUNTIME_CFG" | python -c "import json,sys; d=json.load(sys.stdin); assert d.get('ok') and d.get('frozen'), d; print('  prepare frozen:', d['domains']['alfworld'])"
python -m rsi.cli data validate --config "$RUNTIME_CFG" | python -c "import json,sys; d=json.load(sys.stdin); assert d.get('ok'), d; print('  validate ok')"

echo "== probe-opd: confirmed teacher 622625de, 12 work_train tasks =="
BANK=$RSI_ROOT/runs/e1_alfworld_9b_r5/evolve/champion.json
python -m rsi.cli probe-opd --config "$RUNTIME_CFG" --bank "$BANK" 2>&1 | tail -60

echo "== probe verdict (guide §5.3 #2/#3) =="
python - "$RSI_ROOT" <<'PY'
import glob, json, sys
root = sys.argv[1]
files = sorted(glob.glob(f"{root}/runs/alfworld_probe_r1/opd/probe_report.json"))
assert files, "no probe report"
r = json.load(open(files[0]))
dom = r.get("domains", {}).get("alfworld", {})
stats = dom.get("stats", {})
out = {
    "admitted": dom.get("admitted"),
    "admission_reasons": dom.get("admission_reasons"),
    "total_decisions": stats.get("total_decisions"),
    "teacher_fork_success": stats.get("teacher_fork_success"),
    "native_anchors": stats.get("native_anchors"),
    "explicit_maps": stats.get("explicit_maps"),
    "legal_targets": stats.get("legal_targets"),
    "teacher_scored_targets": stats.get("teacher_scored_targets"),
    "effective_weight_tokens": stats.get("effective_weight_tokens"),
    "opd_grad_norm": dom.get("opd_grad_norm"),
    "coverage": stats.get("coverage"),
    "synthetic": dom.get("synthetic"),
}
print(json.dumps(out, indent=2))
with open(f"{root}/runs/alfworld_probe_r1/probe_verdict.json", "w") as f:
    json.dump(out, f, indent=2)
PY
echo "== ALFWorld probe-opd done =="
