#!/usr/bin/env bash
# AppWorld bounded self-evolution RETRY (guide §4.5, 2026-11-09).
# Preconditions (verified before submission):
#   - H0 v3 re-measure (rsi-h0v3-appworld-9b-8gpu-20261109-r1) PASSED its
#     gates: the contract-v3 interface works end-to-end at 9B.
#   - Fixed budget (guide §4.5 #2, frozen BEFORE launch): 4 generations x
#     2 candidates, max 8 candidate evaluations, max 1 repair/candidate,
#     plateau patience 2.  NO extensions after launch.
# Expected end states (job success != method success):
#   BASELINE_RETAINED | NO_CONFIRMED_GAIN | UNRESOLVED_INTERVAL |
#   PROPOSAL_STALLED | RUNTIME_INVALID — or a frozen champion that still
#   must pass fresh confirm_h before any training.
# Mechanism coverage request to the proposer (guide §4.5 #3): API-doc-driven
# discovery correction; small-step execution with output checks; evidence-
# based subgoal/completion verification.  NOT three Finish guards.
set -euo pipefail
RSI_ROOT=/mnt/songzijun/Capability_Evolution/RSI
MODEL_SRC=/mnt/songzijun/models/Qwen3.5-9B
MODEL_LOCAL=/tmp/rsi_models/Qwen3.5-9B
RUNTIME_CFG=$RSI_ROOT/configs/runtime_appworld_retry_r1.yaml
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
pip("accelerate", limit=300)
from transformers.utils import is_flash_linear_attention_available
if not is_flash_linear_attention_available():
    sys.exit("[FATAL] fla unavailable")

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

# PR-01 contract preflight (same as the H0 v3 job): discovery calls work
appworld.update_root("/mnt/songzijun/datasets/appworld") if hasattr(appworld, "update_root") else None
from appworld.environment import AppWorld
_w = AppWorld(task_id="024c982_1", experiment_name="rsi-retry-preflight",
              max_interactions=5, load_ground_truth=False, random_seed=17)
for _code in ("print(apis.api_docs.show_app_descriptions())",
              'print(apis.api_docs.show_api_descriptions(app_name="spotify"))',
              'print(apis.api_docs.show_api_doc(app_name="spotify", api_name="login"))'):
    _out = str(_w.execute(_code))
    assert len(_out) > 50 and "Execution failed" not in _out and "EXEC_ERROR" not in _out, \
        f"contract preflight failed: {_out[:120]}"
_w.close()
print("CONTRACT_V3_PREFLIGHT_OK + deps ready", flush=True)
PY

echo "== model pre-copy =="
if [ ! -f "$MODEL_LOCAL/config.json" ]; then
  rm -rf "$MODEL_LOCAL"; mkdir -p "$MODEL_LOCAL"
  cp -r "$MODEL_SRC/." "$MODEL_LOCAL/"
fi
du -sh "$MODEL_LOCAL"

echo "== runtime config (appworld_retry_v1 + v3 doc + e1 split) =="
python - <<PY
import yaml
cfg = yaml.safe_load(open("$RSI_ROOT/configs/appworld_retry_v1.yaml"))
cfg["policy"]["model_path"] = "$MODEL_LOCAL"
cfg["campaign_id"] = "appworld_retry_r1"
cfg["execution"]["runs_root"] = "$RSI_ROOT/runs"
# e1 manifest (12 work_train + 12 harness_select, official-gated)
cfg["suite"]["split_manifest"] = "data/splits/e1_appworld_9b.json"
# contract v3 doc is already in the yaml; assert it survived the merge
assert cfg["runtime"]["environment_doc"] == "configs/env_docs/appworld_v3.md", cfg["runtime"]["environment_doc"]
assert cfg["evolution"]["max_candidate_evaluations"] == 8
assert cfg["evolution"]["max_candidate_repairs"] == 1
assert cfg["evolution"]["plateau_patience"] == 2
yaml.safe_dump(cfg, open("$RUNTIME_CFG", "w"), sort_keys=False, allow_unicode=True)
print("cfg ready:", cfg["campaign_id"], "| budget: 4gen x 2cand / 8eval / 1repair / patience2 | doc:", cfg["runtime"]["environment_doc"])
PY

echo "== data prepare / validate =="
python -m rsi.cli data prepare --config "$RUNTIME_CFG" | python -c "import json,sys; d=json.load(sys.stdin); assert d.get('ok') and d.get('frozen'), d; print('  prepare frozen:', d['domains']['appworld'])"
python -m rsi.cli data validate --config "$RUNTIME_CFG" | python -c "import json,sys; d=json.load(sys.stdin); assert d.get('ok'), d; print('  validate ok')"

echo "== bounded retry: 4 gens x 2 candidates, evidence-contract proposer =="
export RSI_PANEL_WORKERS=4  # irb8r-verified config (7-worker spawn hung after task 1 on 11-12)
export RSI_PROPOSER_DEVICE=cpu  # v2 postmortem: master-side CUDA context (any GPU) before spawn mp froze both runs (evbyz w/ cuda:7, advuk w/ cuda:0); CPU proposer breaks the CUDA coupling entirely
python -m rsi.cli evolve --config "$RUNTIME_CFG" 2>&1 | tee /tmp/retry_evolve_stream.log | python -u -c "
import sys
for line in sys.stdin:
    if any(k in line for k in ('generation', 'champion', 'stopped', 'candidate', 'evidence-rejected', 'error', 'Error')):
        sys.stdout.write(line)
        sys.stdout.flush()
"

echo "== end-state audit (honest labeling) =="
python - "$RSI_ROOT" <<'PY'
import glob, json, sys
root = sys.argv[1]
files = sorted(glob.glob(f"{root}/runs/appworld_retry_r1/evolve/evolve_result.json"))
assert files, "no evolve_result.json"
r = json.load(open(files[0]))
champ = r.get("champion")
out = {
    "stopped_reason": r.get("stopped_reason"),
    "evaluated_candidates": r.get("evaluated_candidates"),
    "champion": champ,
    "honest_label": (
        "CHAMPION_FROZEN_PENDING_CONFIRM" if champ
        else f"NO_TEACHER ({r.get('stopped_reason')}) — bounded retry ends here; "
             "no threshold lowering, no relaunch on the same confirm set"
    ),
    "generations": [
        {"generation": g.get("generation"), "selection_reason": (g.get("selection") or {}).get("reason"),
         "n_candidates": len(g.get("candidate_ids") or [])}
        for g in r.get("generations", [])
    ],
    "rejection_history_tail": [h for h in (r.get("history") or []) if ("error" in h or "rejected" in str(h))][-10:],
}
print(json.dumps(out, indent=2, default=str))
with open(f"{root}/runs/appworld_retry_r1/evolve/end_state.json", "w") as f:
    json.dump(out, f, indent=2, default=str)
sys.exit(0)
PY
echo "== bounded retry done =="
exit 0
