#!/usr/bin/env bash
# RSI H0 基线完整测评（完整 AppWorld benchmark）— Qwen3.5-27B，单节点 8xM402
# 布局：panel workers x8（测评无 proposer，8 卡全作 worker）
# 完整 benchmark：7 个 role 池全量 325 task 逐个评（harness minimal / official TGC）
set -euo pipefail
RSI_ROOT=/mnt/songzijun/Capability_Evolution/RSI
MODEL_SRC=/mnt/songzijun/models/Qwen3.5-27B
MODEL_LOCAL=/tmp/rsi_models/Qwen3.5-27B
CFG_SRC="$RSI_ROOT/configs/eval_h0_qwen35_27B_appworld_8gpu.yaml"
CFG_RUNTIME=/tmp/rsi_eval_h0_appworld_cfg_27B.yaml
export PYTHONPATH="$RSI_ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
cd "$RSI_ROOT"

echo "== dep preflight =="
python - <<'PY'
import importlib, subprocess, sys
need = []
for mod, pkg in (("yaml", "pyyaml"), ("numpy", "numpy"), ("torch", "torch"), ("transformers", "transformers")):
    try:
        importlib.import_module(mod)
    except ImportError:
        need.append(pkg)
if need:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
        "-i", "https://pkgs.d.xiaomi.net/artifactory/api/pypi/pypi-virtual/simple", *need])
for pkg, mod in (("flash-linear-attention", "flash_linear_attention"), ("causal-conv1d", "causal_conv1d"), ("openpyxl", "openpyxl"), ("swebench", "swebench")):
    try:
        importlib.import_module(mod)
    except ImportError:
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
            "-i", "https://pkgs.d.xiaomi.net/artifactory/api/pypi/pypi-virtual/simple", pkg])
        print(("installed" if r.returncode == 0 else "MISSING:"), pkg)
# appworld: vendored repo + PINNED combo. Each pip is bounded in time and the
# resolver is kept away from big backtracking trees (stage-wise pins, --no-deps
# where safe) — an unbounded resolver pass stalled a previous run for 1h+.
try:
    import appworld
except ImportError:
    MIRROR = "https://pkgs.d.xiaomi.net/artifactory/api/pypi/pypi-virtual/simple"

    def pip(args, limit=300):
        print("pip", " ".join(args), flush=True)
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-i", MIRROR] + args,
                           timeout=limit)
        if r.returncode != 0:
            print("pip failed:", args, flush=True)

    pip(["pydantic==2.12.0", "sqlalchemy==2.0.36", "sqlmodel==0.0.22", "sqlalchemy-utils==0.41.1"])
    pip(["fastapi==0.115.6", "fastapi-login==1.10.3", "freezegun==1.5.1"])
    pip(["--no-deps", "starlette==0.48.0"])
    pip(["--no-deps", "/mnt/songzijun/datasets/appworld_src"])
    pip(["pendulum>=3.0.0", "typer>=0.16.0", "text2num>=0.3.0", "orjson>=3.10.12",
         "pydantic-extra-types[pendulum]>=2.8.0", "inflection>=0.5.1", "email-validator>=2.0.0",
         "polyfactory>=2.15.0", "faker>=18.0.0", "xxhash>=3.0.0", "munch>=4.0.0",
         "rich>=13.0.0", "python-dotenv>=1.0.1", "python-multipart>=0.0.5",
         "httpx>=0.20.0", "libcst>=1.2.0", "jsonref>=1.1.0", "filelock>=3.1.0",
         "psutil>=5.9.0", "cryptography>=47.0.0"], limit=600)
    import appworld as _aw  # verify importability now
    print("installed appworld from vendored repo (pinned combo)")
# unpack the encrypted apps bundle (bare installs ship it sealed)
import subprocess as _sp
_sp.run([sys.executable, "-m", "appworld.cli", "install"], check=False, timeout=300)
import appworld.apps.admin  # hard assertion: apps code must be importable
import torch, transformers
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "devices", torch.cuda.device_count() if torch.cuda.is_available() else 0)
print("transformers", transformers.__version__)
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
cfg["execution"]["runs_root"] = "$RSI_ROOT/runs"
yaml.safe_dump(cfg, open("$CFG_RUNTIME", "w"), sort_keys=False, allow_unicode=True)
print("model_path=", cfg["policy"]["model_path"], "domains=", cfg["suite"]["domains"])
PY

echo "== data prepare / validate =="
python -m rsi.cli data prepare --config "$CFG_RUNTIME" | python -c "import json,sys; d=json.load(sys.stdin); assert d.get('ok'), d; print('  prepare:', d.get('split_manifest'))"
python -m rsi.cli data validate --config "$CFG_RUNTIME" | python -c "import json,sys; d=json.load(sys.stdin); assert d.get('ok'), d; print('  validate: ok')"

export RSI_PANEL_WORKERS=8
echo "== H0 complete-benchmark eval: 7 role pools =="
rc=0
for SPLIT in work_train harness_select weight_dev confirm_h confirm_w final_test transfer_test; do
  echo "---- eval split=$SPLIT harness=minimal ----"
  python -m rsi.cli eval --config "$CFG_RUNTIME" --harness minimal --split "$SPLIT" || rc=1
done
echo "== eval done rc=$rc =="
exit $rc
