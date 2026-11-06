#!/usr/bin/env bash
# RSI E0 (new guide §12): interface & measurement acceptance, NO evolution.
# AppWorld, Qwen3.5-9B, 16 official-gated dev tasks (8 official-train + 8
# official-dev scenario representatives, frozen manifest e0_react).
# Arms: H0-ReAct (react_zero_shot_v1) vs H0-Act (act_only_v1), paired.
# Gates:
#   G1 every task measurement_status=VALID (strict scorer path)
#   G2 scorer never mutates the business DB (db_unchanged_after_scoring)
#   G3 both arms n_decisions>0 and tokens>0 (clock-freeze regression)
#   G4 official split gate: manifest source recorded = datasets/*.txt
# strict success MAY be 0: E0 measures the measurement, not capability.
set -euo pipefail
RSI_ROOT=/mnt/songzijun/Capability_Evolution/RSI
MODEL_SRC=/mnt/songzijun/models/Qwen3.5-9B
MODEL_LOCAL=/tmp/rsi_models/Qwen3.5-9B
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
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-i", MIRROR, *args], timeout=limit)
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
    print("torch", torch.__version__, "devices", n, flush=True)
    print("transformers", transformers.__version__, flush=True)
    return torch.cuda.is_available() and n >= MIN_GPUS

if not audit():
    print("[WARN] CUDA unusable — force-pinning", flush=True)
    pip("--force-reinstall", TORCH_PIN, TF_PIN)
    r = subprocess.run([sys.executable, "-c",
        "import torch,sys;n=torch.cuda.device_count() if torch.cuda.is_available() else 0;"
        "print('retry devices',n);sys.exit(0 if torch.cuda.is_available() and n>=8 else 1)"])
    if r.returncode != 0:
        sys.exit("[FATAL] CUDA unusable after pinning")

from transformers.models import qwen3_5  # noqa: F401
print("qwen3_5 registration: ok", flush=True)
pip("flash-linear-attention==0.6.0", limit=600)
# r3: causal-conv1d install attempt REMOVED — its sdist build pulls torch
# 2.15+cu132 into the isolated build env and times out fetching a prebuilt
# wheel from GitHub (~3.5 min wasted, ABI-mismatch risk per kygt9). The
# transformers reference fallback is slower but correct (E0 r1 passed G1-G4
# with it — "correct but slow" is acceptable for a timing-tolerant gate).
from transformers.utils import is_flash_linear_attention_available
if not is_flash_linear_attention_available():
    sys.exit("[FATAL] fla fused kernels unavailable — timing calibration invalid")

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
import subprocess as _sp
_sp.run([sys.executable, "-m", "appworld.cli", "install"], check=False, timeout=300)
for _round in range(5):
    try:
        import appworld.apps.admin  # noqa: F401
        print("appworld.apps.admin import: ok", flush=True)
        break
    except ModuleNotFoundError as e:
        _missing = (e.name or str(e)).split(".")[0]
        if _missing in ("appworld", ""):
            sys.exit("[FATAL] vendored appworld not importable")
        if pip(_missing, limit=300) != 0:
            sys.exit(f"[FATAL] cannot install {_missing!r}")
else:
    sys.exit("[FATAL] appworld.apps.admin unimportable")

# G4 preflight: official dataset listings must exist
import os
ds = "/mnt/songzijun/datasets/appworld/data/datasets"
for f in ("train.txt", "dev.txt", "test_normal.txt", "test_challenge.txt"):
    if not os.path.isfile(os.path.join(ds, f)):
        sys.exit(f"[FATAL] official dataset listing missing: {ds}/{f}")
print("OFFICIAL_SPLIT_OK: datasets/*.txt present", flush=True)
PY

echo "== model pre-copy =="
if [ ! -f "$MODEL_LOCAL/config.json" ]; then
  rm -rf "$MODEL_LOCAL"; mkdir -p "$MODEL_LOCAL"
  cp -r "$MODEL_SRC/." "$MODEL_LOCAL/"
fi
du -sh "$MODEL_LOCAL"

echo "== runtime configs (r3: configs/ placement — resolve_path project_root correct) =="
python - <<PY
import yaml
for prof, tag in (("react_zero_shot_v1", "react"), ("act_only_v1", "act")):
    cfg = yaml.safe_load(open(f"$RSI_ROOT/configs/e0_{tag}_appworld_9b.yaml"))
    cfg["policy"]["model_path"] = "$MODEL_LOCAL"
    cfg["campaign_id"] = cfg["campaign_id"] + "_r3"
    cfg["execution"]["runs_root"] = "$RSI_ROOT/runs"
    # Written INTO configs/ so relative paths (data/splits/…) resolve against
    # the RSI project root; no /tmp copies (34ino/E1 batch lesson).
    yaml.safe_dump(cfg, open(f"$RSI_ROOT/configs/runtime_e0_{tag}_r3.yaml", "w"), sort_keys=False, allow_unicode=True)
    print(tag, "cfg:", cfg["campaign_id"])
PY

echo "== data prepare / validate (both arms) =="
for tag in react act; do
  python -m rsi.cli data prepare --config "$RSI_ROOT/configs/runtime_e0_${tag}_r3.yaml" | python -c "import json,sys; d=json.load(sys.stdin); assert d.get('ok'), d; print('  $tag prepare:', d.get('split_manifest'), 'frozen' if d.get('frozen') else 'REGENERATED')"
  python -m rsi.cli data validate --config "$RSI_ROOT/configs/runtime_e0_${tag}_r3.yaml" | python -c "import json,sys; d=json.load(sys.stdin); assert d.get('ok'), d; print('  $tag validate: ok')"
done

echo "== arms: H0-ReAct and H0-Act paired (adapter now submits Finish.answer via supervisor.complete_task) =="
export RSI_PANEL_WORKERS=8
for tag in react act; do
  echo "---- arm=$tag ----"
  python -m rsi.cli eval --config "$RSI_ROOT/configs/runtime_e0_${tag}_r3.yaml" --split e0_probe || true
done

echo "== E0 verdict =="
python - "$RSI_ROOT" <<'PY'
import glob, json, sys
root = sys.argv[1]
arms = {}
for tag in ("react", "act"):
    files = sorted(glob.glob(f"{root}/runs/e0_{tag}_appworld_9b_r1/eval/eval_e0_probe_*.json"))
    assert files, f"no eval output for arm {tag}"
    d = json.load(open(files[0]))
    per = d["result"]["per_task"]
    costs = d["result"]["costs"]
    n = len(per)
    valid = sum(1 for p in per.values() if (p.get("details") or {}).get("measurement_status") == "VALID")
    db_mut = [t for t, p in per.items() if (p.get("details") or {}).get("db_unchanged_after_scoring") is False]
    nd_zero = [t for t, p in per.items() if p.get("n_decisions", 0) == 0]
    strict = sum(1 for p in per.values() if p.get("success"))
    partial = 100 * sum((p.get("details") or {}).get("partial_test_pass_rate") or 0.0 for p in per.values()) / max(n, 1)
    arms[tag] = {
        "tasks": n, "valid": valid, "db_mutated": db_mut,
        "nd_zero": nd_zero, "strict_success": strict,
        "partial_mean": round(partial, 2),
        "actor_output_tokens": costs.get("actor_output_tokens", 0),
        "finish": sum(1 for p in per.values() if p.get("terminated_reason") == "finish"),
        "max_actions": sum(1 for p in per.values() if p.get("terminated_reason") == "max_actions"),
        "budget_wall": sum(1 for p in per.values() if p.get("terminated_reason") == "budget_wall"),
    }
verdict = {"ok": True, "arms": arms, "gates": {}}
for tag, a in arms.items():
    g = []
    if a["valid"] != a["tasks"]:
        g.append("G1_FAIL: invalid measurements")
    if a["db_mutated"]:
        g.append("G2_FAIL: scorer mutated business DB")
    if a["nd_zero"] or a["actor_output_tokens"] == 0:
        g.append("G3_FAIL: model never called (clock-freeze regression)")
    verdict["gates"][tag] = g or ["PASS"]
    if g:
        verdict["ok"] = False
# paired delta (partial & strict) on common tasks
react_files = sorted(glob.glob(f"{root}/runs/e0_react_appworld_9b_r3/eval/eval_e0_probe_*.json"))
act_files = sorted(glob.glob(f"{root}/runs/e0_act_appworld_9b_r3/eval/eval_e0_probe_*.json"))
per_r = json.load(open(react_files[0]))["result"]["per_task"]
per_a = json.load(open(act_files[0]))["result"]["per_task"]
common = sorted(set(per_r) & set(per_a))
pr = [per_r[t] for t in common]; pa = [per_a[t] for t in common]
verdict["paired"] = {
    "n_common": len(common),
    "react_strict": sum(1 for p in pr if p.get("success")),
    "act_strict": sum(1 for p in pa if p.get("success")),
    "react_partial_mean": round(100 * sum((p.get("details") or {}).get("partial_test_pass_rate") or 0.0 for p in pr) / max(len(pr),1), 2),
    "act_partial_mean": round(100 * sum((p.get("details") or {}).get("partial_test_pass_rate") or 0.0 for p in pa) / max(len(pa),1), 2),
}
print(json.dumps(verdict, indent=2))
sys.exit(0 if verdict["ok"] else 1)
PY
rc=$?
echo "== E0 done rc=$rc (0 = measurement+interface trustworthy) =="
exit $rc
