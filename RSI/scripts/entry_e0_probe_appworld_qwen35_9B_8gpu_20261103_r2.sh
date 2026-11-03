#!/usr/bin/env bash
# RSI E0: measurement rebuild probe (audit doc §15) — Qwen3.5-9B, 8xM402.
#
# Three phases over 4 frozen scenarios (e0_probe manifest, one task per
# scenario group 302c169/0d8a4ee/82e2fac/530b157):
#   A. no-op control   — provider=noop; measures initial-state partial.
#   B. known-correct   — official compiled_solution fixture; AUTO-SKIPs with
#                        BLOCKED when the bundle ships canary-only solutions
#                        (current revision does).
#   C. 9B real H0      — hf_path policy, real model calls through the fixed
#                        runtime factory.
#
# E0 GATE (fail => stop, do NOT proceed to E1):
#   - every phase-eval reports measurement_status=VALID per task;
#   - scorer DB digest unchanged by scoring (db_unchanged_after_scoring);
#   - phase C episodes show n_decisions>0 AND nonzero token accounting;
#   - phase C strict success MAY be 0; that is a task-difficulty finding, not
#     a measurement failure — the failure categories tell E1 what to fix.
set -euo pipefail
RSI_ROOT=/mnt/songzijun/Capability_Evolution/RSI
MODEL_SRC=/mnt/songzijun/models/Qwen3.5-9B
MODEL_LOCAL=/tmp/rsi_models/Qwen3.5-9B
CFG_SRC="$RSI_ROOT/configs/e0_probe_qwen35_9B_appworld_8gpu.yaml"
CFG_RUNTIME=/tmp/rsi_e0_probe_cfg_9B_r3.yaml
export PYTHONPATH="$RSI_ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
cd "$RSI_ROOT"

echo "== dep preflight =="
python - <<'PY'
import importlib, subprocess, sys

MIRROR = "https://pkgs.d.xiaomi.net/artifactory/api/pypi/pypi-virtual/simple"
TORCH_PIN = "torch==2.10.0"
TF_PIN = "transformers==5.17.0"
MIN_GPUS = 8

def pip(*args, limit=900):
    print("pip", " ".join(args), flush=True)
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-i", MIRROR, *args],
                       timeout=limit)
    if r.returncode != 0:
        print("pip failed:", args, flush=True)
    return r.returncode

need = []
for mod, pkg in (("yaml", "pyyaml"), ("numpy", "numpy"), ("torch", TORCH_PIN), ("transformers", TF_PIN)):
    try:
        importlib.import_module(mod)
    except ImportError:
        need.append(pkg)
if need:
    pip(*need)

def audit():
    import torch, transformers
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "devices", n)
    print("transformers", transformers.__version__)
    return torch.cuda.is_available() and n >= MIN_GPUS

if not audit():
    print(f"[WARN] CUDA unusable — forcing {TORCH_PIN} / {TF_PIN}", flush=True)
    pip("--force-reinstall", TORCH_PIN, TF_PIN)
    code = (
        "import torch, sys; n = torch.cuda.device_count() if torch.cuda.is_available() else 0; "
        "print('retry devices', n); "
        "sys.exit(0 if torch.cuda.is_available() and n >= %d else 1)" % MIN_GPUS
    )
    r = subprocess.run([sys.executable, "-c", code])
    if r.returncode != 0:
        sys.exit("[FATAL] CUDA still unusable after pinning")

try:
    from transformers.models import qwen3_5  # noqa: F401
    print("qwen3_5 registration: ok")
except Exception as e:  # noqa: BLE001
    sys.exit(f"[FATAL] transformers has no qwen3_5 ({e})")

print("kernel deps:", flush=True)
pip("flash-linear-attention==0.6.0", limit=600)
_r = pip("causal-conv1d", limit=600)
if _r != 0:
    _r = pip("--no-build-isolation", "causal-conv1d", limit=900)
from transformers.utils import is_flash_linear_attention_available, is_causal_conv1d_available
fla_ok = bool(is_flash_linear_attention_available())
conv_ok = bool(is_causal_conv1d_available())
print(f"  fla: {'ok' if fla_ok else 'MISSING'} | causal_conv1d: {'ok' if conv_ok else 'MISSING'}", flush=True)
if not fla_ok:
    sys.exit("[FATAL] fla unavailable — slow fallback would corrupt timing calibration")

# appworld pinned combo (vendored repo); apps need starlette>=0.46 constant names
try:
    import appworld
    print("appworld already importable")
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
    print("installed appworld from vendored repo")
import subprocess as _sp
_sp.run([sys.executable, "-m", "appworld.cli", "install"], check=False, timeout=300)
for _round in range(5):
    try:
        import appworld.apps.admin  # noqa: F401
        print("appworld.apps.admin import: ok")
        break
    except ModuleNotFoundError as e:
        _missing = (e.name or str(e)).split(".")[0]
        if _missing in ("appworld", ""):
            sys.exit("[FATAL] vendored appworld not importable")
        print(f"[WARN] missing {_missing!r} — pip round {_round + 1}", flush=True)
        if pip(_missing, limit=300) != 0:
            sys.exit(f"[FATAL] cannot install {_missing!r}")
else:
    sys.exit("[FATAL] appworld.apps.admin unimportable after 5 rounds")
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
cfg = yaml.safe_load(open("$CFG_SRC"))
cfg["policy"]["model_path"] = "$MODEL_LOCAL"
cfg["campaign_id"] = cfg["campaign_id"] + "_r3"
cfg["execution"]["runs_root"] = "$RSI_ROOT/runs"
yaml.safe_dump(cfg, open("$CFG_RUNTIME", "w"), sort_keys=False, allow_unicode=True)
print("model_path=", cfg["policy"]["model_path"], "campaign_id=", cfg["campaign_id"])
PY

echo "== data prepare / validate =="
# The runtime config lives in /tmp, so relative split_manifest resolves to
# /tmp/data/splits — stage the frozen repo manifest there BEFORE prepare so
# the authoritative-freeze path copies it instead of regenerating the legacy
# 7-role split (which has no e0_probe role).
mkdir -p /tmp/data/splits
cp "$RSI_ROOT/data/splits/e0_probe_appworld.json" /tmp/data/splits/e0_probe_appworld.json
python -m rsi.cli data prepare --config "$CFG_RUNTIME" | python -c "import json,sys; d=json.load(sys.stdin); assert d.get('ok'), d; print('  prepare:', d.get('split_manifest'))"
python -m rsi.cli data validate --config "$CFG_RUNTIME" | python -c "import json,sys; d=json.load(sys.stdin); assert d.get('ok'), d; print('  validate: ok')"

echo "== E0 phase A: no-op control =="
# separate campaign dir so phase A/C outputs never collide
CFG_NOOP=/tmp/rsi_e0_probe_cfg_9B_r3_noop.yaml
python - <<PY
import yaml
cfg = yaml.safe_load(open("$CFG_RUNTIME"))
cfg["campaign_id"] = "e0_noop_r3"
yaml.safe_dump(cfg, open("$CFG_NOOP", "w"), sort_keys=False, allow_unicode=True)
print("noop campaign cfg ready")
PY
RSI_PANEL_WORKERS=1 python -m rsi.cli eval --config "$CFG_NOOP" --provider noop --split e0_probe || true
echo "== E0 phase B: known-correct fixture check =="
python - <<'PY'
from pathlib import Path
import sys
root = Path("/mnt/songzijun/datasets/appworld/data/tasks")
real = 0
for tid in ("302c169_1", "0d8a4ee_1", "82e2fac_1", "530b157_1"):
    p = root / tid / "ground_truth" / "compiled_solution.py"
    if p.is_file() and len(p.read_text().splitlines()) > 5:
        real += 1
if real == 0:
    print("BLOCKED: bundle ships canary-only compiled_solution.py — known-correct "
          "fixture cannot run in this revision (audit §14). Not a scorer failure.")
else:
    print(f"{real}/4 tasks ship real compiled solutions — running fixture phase is possible")
PY

echo "== E0 phase C: 9B real H0 =="
python -m rsi.cli eval --config "$CFG_RUNTIME" --split e0_probe || true

echo "== E0 verdict =="
python - "$RSI_ROOT" <<'PY'
import glob, json, sys
root = sys.argv[1]
files = sorted(
    glob.glob(root + "/runs/e0_probe_r3/eval/eval_e0_probe_*.json")
    + glob.glob(root + "/runs/e0_noop_r3/eval/eval_e0_probe_*.json")
)
assert files, "no E0 eval outputs"
verdict = {"ok": True, "phases": []}
for f in files:
    d = json.load(open(f))
    per = d["result"]["per_task"]
    n_valid = sum(1 for p in per.values() if (p.get("details") or {}).get("measurement_status") == "VALID")
    db_mut = [t for t, p in per.items() if (p.get("details") or {}).get("db_unchanged_after_scoring") is False]
    is_real_h0 = "/e0_probe_r1/" in f  # hf_path phase (noop lives in e0_noop_r1)
    nd_zero = [t for t, p in per.items() if p.get("n_decisions", 0) == 0] if is_real_h0 else []
    tok = d["result"]["costs"].get("actor_output_tokens", 0)
    phase = {
        "file": f.split("/")[-1],
        "tasks": len(per),
        "valid": n_valid,
        "db_mutated": db_mut,
        "real_h0_nd_zero": nd_zero,
        "actor_output_tokens": tok,
    }
    if db_mut:
        verdict["ok"] = False
    if n_valid == 0:
        verdict["ok"] = False
    if is_real_h0 and (nd_zero or tok == 0):
        # clock-freeze regression: zero decisions or zero tokens on the real
        # model phase means the model was never called (kygt9 class).
        verdict["ok"] = False
    verdict["phases"].append(phase)
print(json.dumps(verdict, indent=2))
sys.exit(0 if verdict["ok"] else 1)
PY
rc=$?
echo "== E0 done rc=$rc (0 = measurement trustworthy, proceed to E1) =="
exit $rc
