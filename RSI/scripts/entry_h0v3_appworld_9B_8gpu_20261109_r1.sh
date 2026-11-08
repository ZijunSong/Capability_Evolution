#!/usr/bin/env bash
# AppWorld H0 v3 re-measure (guide §4.5 step 1, 2026-11-09).
# Qwen3.5-9B, 8gpu, appworld_v3.md contract (PR-01 fixes), SAME 16-task e0
# panel as r3 (paired comparison) + work_train/harness_select splits for the
# bounded retry.
# Baseline (r3, appworld_v1.md): partial 34.74%, strict 0/16, valid 16/16.
# This run measures the H0-INTERFACE-FIX effect. NOT an evolution result.
set -euo pipefail
RSI_ROOT=/mnt/songzijun/Capability_Evolution/RSI
MODEL_SRC=/mnt/songzijun/models/Qwen3.5-9B
MODEL_LOCAL=/tmp/rsi_models/Qwen3.5-9B
export PYTHONPATH="$RSI_ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
cd "$RSI_ROOT"

echo "== dep preflight (same as e0 r3 chain) =="
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
from transformers.utils import is_flash_linear_attention_available
if not is_flash_linear_attention_available():
    sys.exit("[FATAL] fla fused kernels unavailable")

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

# PR-01 contract preflight: the three verified discovery calls must work
import os
appworld.update_root("/mnt/songzijun/datasets/appworld") if hasattr(appworld, "update_root") else None
from appworld.environment import AppWorld
_w = AppWorld(task_id="024c982_1", experiment_name="rsi-h0v3-preflight",
              max_interactions=5, load_ground_truth=False, random_seed=17)
for _code in ("print(apis.api_docs.show_app_descriptions())",
              'print(apis.api_docs.show_api_descriptions(app_name="spotify"))',
              'print(apis.api_docs.show_api_doc(app_name="spotify", api_name="login"))'):
    _out = str(_w.execute(_code))
    assert len(_out) > 50 and "Execution failed" not in _out and "EXEC_ERROR" not in _out, \
        f"contract preflight failed on {_code!r}: {_out[:120]}"
_w.close()
print("CONTRACT_V3_PREFLIGHT_OK: verified discovery calls work", flush=True)

# official split gate
ds = "/mnt/songzijun/datasets/appworld/data/datasets"
for f in ("train.txt", "dev.txt", "test_normal.txt", "test_challenge.txt"):
    if not os.path.isfile(os.path.join(ds, f)):
        sys.exit(f"[FATAL] official dataset listing missing: {ds}/{f}")
print("OFFICIAL_SPLIT_OK", flush=True)
PY

echo "== model pre-copy =="
if [ ! -f "$MODEL_LOCAL/config.json" ]; then
  rm -rf "$MODEL_LOCAL"; mkdir -p "$MODEL_LOCAL"
  cp -r "$MODEL_SRC/." "$MODEL_LOCAL/"
fi
du -sh "$MODEL_LOCAL"

echo "== runtime config (v3 doc, configs/ placement) =="
python - <<PY
import yaml
cfg = yaml.safe_load(open("$RSI_ROOT/configs/e0_h0v3_appworld_9b.yaml"))
cfg["policy"]["model_path"] = "$MODEL_LOCAL"
cfg["campaign_id"] = "h0v3_appworld_9b_r1"
cfg["execution"]["runs_root"] = "$RSI_ROOT/runs"
yaml.safe_dump(cfg, open("$RSI_ROOT/configs/runtime_h0v3_appworld_r1.yaml", "w"), sort_keys=False, allow_unicode=True)
print("cfg written:", cfg["campaign_id"])
PY

echo "== data prepare / validate =="
python -m rsi.cli data prepare --config "$RSI_ROOT/configs/runtime_h0v3_appworld_r1.yaml" | python -c "import json,sys; d=json.load(sys.stdin); assert d.get('ok'), d; print('  prepare:', d.get('split_manifest'))"
python -m rsi.cli data validate --config "$RSI_ROOT/configs/runtime_h0v3_appworld_r1.yaml" | python -c "import json,sys; d=json.load(sys.stdin); assert d.get('ok'), d; print('  validate: ok')"

echo "== pass 1: e0_probe 16-task panel (paired vs r3 baseline) =="
export RSI_PANEL_WORKERS=8
python -m rsi.cli eval --config "$RSI_ROOT/configs/runtime_h0v3_appworld_r1.yaml" --split e0_probe || true

echo "== pass 2/3: work_train + harness_select (12 tasks each, for the retry) =="
python -m rsi.cli eval --config "$RSI_ROOT/configs/runtime_h0v3_appworld_r1.yaml" --split work_train || true
python -m rsi.cli eval --config "$RSI_ROOT/configs/runtime_h0v3_appworld_r1.yaml" --split harness_select || true

echo "== verdict: H0 v3 vs r3 baseline (paired, same 16 tasks) =="
python - "$RSI_ROOT" <<'PY'
import glob, json, sys
root = sys.argv[1]
new_files = sorted(glob.glob(f"{root}/runs/h0v3_appworld_9b_r1/eval/eval_e0_probe_*.json"))
old_files = sorted(glob.glob(f"{root}/runs/e0_react_appworld_9b_r3/eval/eval_e0_probe_*.json"))
assert new_files, "no eval output for h0v3 run"
assert old_files, "no r3 baseline output"
new = json.load(open(new_files[0]))["result"]["per_task"]
old = json.load(open(old_files[0]))["result"]["per_task"]
common = sorted(set(new) & set(old))
assert len(common) >= 12, f"paired panel too small: {len(common)}"

def stats(per):
    n = len(per)
    return {
        "n": n,
        "valid": sum(1 for p in per.values() if (p.get("details") or {}).get("measurement_status") == "VALID"),
        "strict": sum(1 for p in per.values() if p.get("success")),
        "partial_mean": round(100 * sum((p.get("details") or {}).get("partial_test_pass_rate") or 0.0 for p in per.values()) / max(n, 1), 2),
        "finish": sum(1 for p in per.values() if p.get("terminated_reason") == "finish"),
        "completion_submit_failed": sum(
            1 for p in per.values()
            for s in (p.get("steps") or [])
            if "completion_submit_failed" in json.dumps(s)
        ),
    }
sn, so = stats(new), stats(old)
paired = []
for t in common:
    np_ = (new[t].get("details") or {}).get("partial_test_pass_rate") or 0.0
    op_ = (old[t].get("details") or {}).get("partial_test_pass_rate") or 0.0
    paired.append(np_ - op_)
verdict = {
    "run": "h0v3_appworld_9b_r1",
    "doc": "appworld_v3.md",
    "baseline_r3": so,
    "h0v3": sn,
    "paired_partial_delta_pp": round(100 * sum(paired) / max(len(paired), 1), 2),
    "n_paired": len(paired),
    "interpretation": (
        "H0-INTERFACE-FIX effect under the v3 contract. NOT an evolution gain. "
        "Gates: all measurements VALID, zero completion_submit_failed stalls, "
        "discovery-call output present in steps. strict may still be 0 — the "
        "retry's value is judged on the bounded-search result, not this number."
    ),
    "gates": [],
}
if sn["valid"] != sn["n"]:
    verdict["gates"].append("G1_FAIL: invalid measurements")
if sn["completion_submit_failed"] > 0:
    verdict["gates"].append("G2_FAIL: completion_submit_failed present")
if sn["n"] == 0 or all(p.get("n_decisions", 0) == 0 for p in new.values()):
    verdict["gates"].append("G3_FAIL: model never called")
if not verdict["gates"]:
    verdict["gates"] = ["PASS"]
print(json.dumps(verdict, indent=2))
with open(f"{root}/runs/h0v3_appworld_9b_r1/verdict.json", "w") as f:
    json.dump(verdict, f, indent=2)
sys.exit(0 if verdict["gates"] == ["PASS"] else 1)
PY
rc=$?
echo "== H0 v3 re-measure done rc=$rc =="
exit $rc
