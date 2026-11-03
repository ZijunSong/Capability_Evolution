#!/usr/bin/env bash
# 训练链路 1 卡 preflight：在烧 32 卡排队前，用真实 9B + bf16 + LoRA 走一遍
# 任务的完整序列（generate/collect/joint step/export）。逐阶段 PASS/FAIL。
set -euo pipefail

RSI_ROOT=/mnt/songzijun/Capability_Evolution/RSI
MODEL_SRC=/mnt/songzijun/models/Qwen3.5-9B
MODEL_LOCAL=/tmp/rsi_models/Qwen3.5-9B
CFG_SRC="$RSI_ROOT/configs/train_harness_opd_qwen35_9b_200step_32gpu.yaml"
CFG_RUNTIME=/tmp/rsi_preflight_cfg.yaml
CHAMPION="$RSI_ROOT/runs/evolve_qwen35_9b_p1_r4/evolve/champion.json"

export PYTHONPATH="$RSI_ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
cd "$RSI_ROOT"

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
for pkg, mod in (("flash-linear-attention", "flash_linear_attention"), ("causal-conv1d", "causal_conv1d")):
    try:
        importlib.import_module(mod)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
            "-i", "https://pkgs.d.xiaomi.net/artifactory/api/pypi/pypi-virtual/simple", pkg])
import torch, transformers
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "devices", torch.cuda.device_count() if torch.cuda.is_available() else 0)
print("transformers", transformers.__version__)
PY

if [ ! -f "$MODEL_LOCAL/config.json" ]; then
  rm -rf "$MODEL_LOCAL"; mkdir -p "$MODEL_LOCAL"
  cp -r "$MODEL_SRC/." "$MODEL_LOCAL/"
fi
du -sh "$MODEL_LOCAL"

python - <<PY
import yaml
cfg = yaml.safe_load(open("$CFG_SRC"))
cfg["policy"]["model_path"] = "$MODEL_LOCAL"
cfg["execution"]["runs_root"] = "$RSI_ROOT/runs"
yaml.safe_dump(cfg, open("$CFG_RUNTIME", "w"), sort_keys=False, allow_unicode=True)
PY

exec python "$RSI_ROOT/scripts/preflight_train_step.py" "$CFG_RUNTIME" "$CHAMPION"
