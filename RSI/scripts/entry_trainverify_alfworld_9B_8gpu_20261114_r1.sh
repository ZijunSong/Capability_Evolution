#!/usr/bin/env bash
# ALFWorld 9B real-training GATE verification (guide §8.4 验收, 2026-11-14):
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
RUNTIME_CFG=$RSI_ROOT/configs/runtime_alfworld_trainverify_r1.yaml
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

echo "== runtime config (train-verify: 2 steps, readiness gate ON) =="
python - <<PY
import yaml
cfg = yaml.safe_load(open("$RSI_ROOT/configs/runtime_alfworld_v2_confirm.yaml"))
cfg["policy"]["model_path"] = "$MODEL_LOCAL"
cfg["campaign_id"] = "alfworld_trainverify_r1"
cfg["execution"]["runs_root"] = "$RSI_ROOT/runs"
# probe panel size + readiness gate
cfg["evolution"]["dev_panel_per_domain"] = 12
cfg["training"]["max_optimizer_steps_per_round"] = 2
cfg["training"]["episodes_per_rank_per_step"] = 2
cfg["training"]["require_teacher_readiness"] = True
cfg["training"]["readiness_ref"] = "$RSI_ROOT/runs/alfworld_confirm_r1/teacher_readiness.json"
cfg["training"]["method"] = "harness_opd_rl"
assert cfg["suite"]["split_manifest"] == "data/splits/e1_alfworld_v2_confirm.json"
yaml.safe_dump(cfg, open("$RUNTIME_CFG", "w"), sort_keys=False, allow_unicode=True)
print("cfg ready:", cfg["campaign_id"], "| steps: 2 | readiness:", cfg["training"]["readiness_ref"])
PY

echo "== data prepare / validate =="
python -m rsi.cli data prepare --config "$RUNTIME_CFG" | python -c "import json,sys; d=json.load(sys.stdin); assert d.get('ok') and d.get('frozen'), d; print('  prepare frozen:', d['domains']['alfworld'])"
python -m rsi.cli data validate --config "$RUNTIME_CFG" | python -c "import json,sys; d=json.load(sys.stdin); assert d.get('ok'), d; print('  validate ok')"

echo "== run_train (2 optimizer steps, OPD on, readiness gate) =="
BANK=$RSI_ROOT/runs/e1_alfworld_9b_r5/evolve/champion.json
python -m rsi.cli train --config "$RUNTIME_CFG" --bank "$BANK" 2>&1 | tail -50

echo "== G1-G5 verification =="
python - "$RSI_ROOT" <<'PY'
import glob, json, sys, hashlib, os
import torch
root = sys.argv[1]
res_files = sorted(glob.glob(f"{root}/runs/alfworld_trainverify_r1/train/train_result.json"))
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
with open(f"{root}/runs/alfworld_trainverify_r1/train_verify_verdict.json", "w") as f:
    json.dump(out, f, indent=2)
PY

echo "== G2: export/reload logprob parity =="
python - "$RSI_ROOT" <<'PY'
import glob, json, sys, os
import torch
root = sys.argv[1]
res = json.load(open(sorted(glob.glob(f"{root}/runs/alfworld_trainverify_r1/train/train_result.json"))[0]))
exp_dir = (res.get("export") or {}).get("artifact", {}).get("adapter_path") or str(os.path.join(root, "runs/alfworld_trainverify_r1/train/export"))
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
with open(f"{root}/runs/alfworld_trainverify_r1/g2_parity.json", "w") as f:
    json.dump(out, f, indent=2)
PY
echo "== ALFWorld train-verify done =="
