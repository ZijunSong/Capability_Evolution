#!/usr/bin/env bash
# RSI 第一阶段 EVOLVE — Qwen3.5-9B（单节点 8xM402）
# 布局：panel workers x7 (GPU0-6) + self-proposer (GPU7)。
# 遵循：cp 模型到 /tmp（JuiceFS 8 路读）、entry 先拷 /tmp 再 bash（heredoc 回读）、
# runs 落共享盘、缺依赖走内部镜像。
set -euo pipefail

RSI_ROOT=/mnt/songzijun/Capability_Evolution/RSI
MODEL_SRC=/mnt/songzijun/models/Qwen3.5-9B
MODEL_LOCAL=/tmp/rsi_models/Qwen3.5-9B
CFG_SRC="$RSI_ROOT/configs/evolve_qwen35_9B_8gpu.yaml"
CFG_RUNTIME=/tmp/rsi_evolve_cfg_9b.yaml

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
    print("installing from internal mirror:", need)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
        "-i", "https://pkgs.d.xiaomi.net/artifactory/api/pypi/pypi-virtual/simple", *need])
# fast kernels for Qwen3.x hybrid linear attention (best-effort; without them
# transformers falls back to a reference PyTorch path ~30x slower and OOM-prone)
for pkg in ("flash-linear-attention", "causal-conv1d"):
    try:
        importlib.import_module(pkg.replace("-", "_").split("[")[0] if pkg != "flash-linear-attention" else "flash_linear_attention")
    except ImportError:
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
            "-i", "https://pkgs.d.xiaomi.net/artifactory/api/pypi/pypi-virtual/simple", pkg])
        print(("installed" if r.returncode == 0 else "MISSING (fallback kernel will be used):"), pkg)
import torch, transformers
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "devices", torch.cuda.device_count() if torch.cuda.is_available() else 0)
print("transformers", transformers.__version__)
PY

echo "== model pre-copy (JuiceFS 8-way read guard) =="
if [ ! -f "$MODEL_LOCAL/config.json" ]; then
  rm -rf "$MODEL_LOCAL"; mkdir -p "$MODEL_LOCAL"
  cp -r "$MODEL_SRC/." "$MODEL_LOCAL/"
fi
du -sh "$MODEL_LOCAL"

echo "== runtime config =="
python - <<PY
import yaml
cfg = yaml.safe_load(open("$CFG_SRC"))
cfg["campaign_id"] = "evolve_qwen35_9b_p1_r4"
cfg["policy"]["model_path"] = "$MODEL_LOCAL"
cfg["execution"]["runs_root"] = "$RSI_ROOT/runs"
yaml.safe_dump(cfg, open("$CFG_RUNTIME", "w"), sort_keys=False, allow_unicode=True)
print("model_path=", cfg["policy"]["model_path"], "runs_root=", cfg["execution"]["runs_root"])
PY

export RSI_PANEL_WORKERS=7
export RSI_PROPOSER_DEVICE=cuda:7

cp "$RSI_ROOT/scripts/entry_evolve_8gpu.sh" /tmp/entry_evolve_8gpu.sh
export RSI_PROJECT_ROOT="$RSI_ROOT"
bash /tmp/entry_evolve_8gpu.sh --config "$CFG_RUNTIME"
