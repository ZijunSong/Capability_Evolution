#!/usr/bin/env bash
# RSI HarnessOPD 内化 — Qwen3.5-9B all-teacher/zero-student，200-step，4node32gpu
#   Student = 仅 H0（zero）；Teacher = H0 + r4 champion bank（all，FinishGuard）
#   LoRA r32 累计 adapter（base 冻结）；32 rank 数据并行（rollout 分片+全局归一+SUM 梯度同步）
#   跳过 probe 准入（用户指定试跑）
#
# 多节点约定（沿用 TRIM 7nzzn/9ixnl 验证过的 CloudML 模式）：
#   CloudML 注入 MASTER_ADDR + NODE_RANK；RANK/WORLD_SIZE/PET_* 一律 unset 后
#   自启 torchrun（nnodes=4, nproc_per_node=8），world_size 以 RESOLVED 环境为准。
set -euo pipefail

RSI_ROOT=/mnt/songzijun/Capability_Evolution/RSI
MODEL_SRC=/mnt/songzijun/models/Qwen3.5-9B
MODEL_LOCAL=/tmp/rsi_models/Qwen3.5-9B
CFG_SRC="$RSI_ROOT/configs/train_harness_opd_qwen35_9b_200step_32gpu.yaml"
CFG_RUNTIME=/tmp/rsi_train_opd_9b.yaml
CHAMPION="$RSI_ROOT/runs/evolve_qwen35_9b_p1_r4/evolve/champion.json"

export PYTHONPATH="$RSI_ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
cd "$RSI_ROOT"

# ---------------------------------------------------------------- multinode env
NODE_RANK="${NODE_RANK:-${PET_NODE_RANK:-${NODE_INDEX:-${NODE_NUMBER:-}}}}"
if [ -z "${NODE_RANK:-}" ]; then
  echo "FATAL: NODE_RANK is not set (each pod needs 0..3)" >&2; exit 1
fi
if [ -z "${MASTER_ADDR:-}" ]; then
  echo "FATAL: MASTER_ADDR is not set (CloudML must inject node0 IP)" >&2; exit 1
fi
: "${MASTER_PORT:=29500}"
export MASTER_ADDR NODE_RANK MASTER_PORT
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK \
      GROUP_WORLD_SIZE TORCHELASTIC_RUN_ID TORCHELASTIC_RESTART_COUNT \
      PET_MASTER_ADDR PET_MASTER_PORT PET_NNODES PET_NPROC_PER_NODE PET_NODE_RANK 2>/dev/null || true
echo "MULTINODE_ENV master_addr=${MASTER_ADDR} node_rank=${NODE_RANK} master_port=${MASTER_PORT} host=$(hostname)"

test -f "$CHAMPION" || { echo "FATAL: champion bank not found: $CHAMPION" >&2; exit 1; }

# ---------------------------------------------------------------- deps
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
for pkg, mod in (("flash-linear-attention", "flash_linear_attention"), ("causal-conv1d", "causal_conv1d")):
    try:
        importlib.import_module(mod)
    except ImportError:
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
            "-i", "https://pkgs.d.xiaomi.net/artifactory/api/pypi/pypi-virtual/simple", pkg])
        print(("installed" if r.returncode == 0 else "MISSING (fallback kernel):"), pkg)
import torch, transformers
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "devices", torch.cuda.device_count() if torch.cuda.is_available() else 0)
print("transformers", transformers.__version__)
PY

# ---------------------------------------------------------------- model pre-copy
echo "== model pre-copy (JuiceFS 32-way read guard, per node) =="
if [ ! -f "$MODEL_LOCAL/config.json" ]; then
  rm -rf "$MODEL_LOCAL"; mkdir -p "$MODEL_LOCAL"
  cp -r "$MODEL_SRC/." "$MODEL_LOCAL/"
fi
du -sh "$MODEL_LOCAL"

# ---------------------------------------------------------------- runtime config
echo "== runtime config =="
python - <<PY
import yaml
cfg = yaml.safe_load(open("$CFG_SRC"))
cfg["policy"]["model_path"] = "$MODEL_LOCAL"
cfg["execution"]["runs_root"] = "$RSI_ROOT/runs"
yaml.safe_dump(cfg, open("$CFG_RUNTIME", "w"), sort_keys=False, allow_unicode=True)
print("model_path=", cfg["policy"]["model_path"])
print("steps=", cfg["training"]["max_optimizer_steps_per_round"],
      "episodes_per_rank=", cfg["training"]["episodes_per_rank_per_step"])
PY

# ---------------------------------------------------------------- torchrun train
echo "== torchrun HarnessOPD (nnodes=4 nproc=8 -> 32 ranks) =="
exec torchrun \
  --nnodes=4 \
  --nproc_per_node=8 \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  -m rsi.cli train --config "$CFG_RUNTIME" --bank "$CHAMPION"
