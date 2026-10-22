#!/usr/bin/env bash
# CloudML 启动脚本：TRIM Harness-1 BC+ full / Qwen3-4B-Instruct-2507 / ADAPTER eval
# 评测 9ixnl (t-20260925202402-9ixnl, 4node 32gpu torch_ddp_lora rl/zero 200step) 的 step100 LoRA。
# 单节点 8x gm402-80g。基座 Qwen3-4B + step100 LoRA adapter (r=8, alpha=16, q/k/o/v_proj)。
# 8 个单卡 vLLM actor (TP=1) 各加载 base+LoRA (enable_lora + lora-modules)，--tp 32 API-client workers。
# local_bm25 (Pyserini/Lucene)，无 Chroma，COMPONENT=zero 无 verifier。
# 镜像: micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto
#   (该镜像基础层缺 torch 等依赖，按 CLAUDE.md 经验在启动阶段显式锁定并安装四包配置)
# JuiceFS: volume=songzijun, JuiceFSCluster=bj9-cloudml (endpoint), mount=/mnt/songzijun
#   — JuiceFS 挂载由 CloudML 通过 CLI --juicefs_mount_configs 注入，不在本脚本处理。
# 依赖锁定对齐 TRIM Harness-1 BC+ 已验证 runtime：
#   torch==2.10.0 / transformers==4.57.6 / vllm==0.19.1 / pyserini==2.0.0 + structlog/chromadb/openai_harmony/chz/tinker/json_repair
#   (pyserini 必须固定 2.0.0；裸装会装到 2.4.0 致 preflight 断言失败 —— CLAUDE.md 经验)
# 由 ImageCommand 调用；保持幂等、可审计；preflight 全量打印后再进 eval。
# eval 脚本：TRIM/scripts/run_bcplus_w_harness1_adapter_gpu8_eval.sh
#   (8 个单卡 vLLM actor TP=1 + --enable-lora --lora-modules trained-policy=<adapter>,
#    --tp 32 workers，local_bm25 无 Chroma，COMPONENT=zero 无 verifier)
set -euo pipefail

JFS_ROOT="/mnt/songzijun"
TRIM_ROOT="${JFS_ROOT}/Capability_Evolution/TRIM"
EASYOPD_ROOT="${JFS_ROOT}/Capability_Evolution/SCAPE-EasyOPD"
MODEL_PATH="${JFS_ROOT}/models/Qwen3-4B-Instruct-2507"
# 9ixnl step200 adapter（torch_ddp_lora rl/zero 200step，step_000200 于 2026-09-26 22:35 落盘）。
# 用 checkpoints/rl/step_000200/adapter/（固定 step100），不是 adapters/rl/（那是最新 step）。
# 注意：adapter_config.base_model_name_or_path=/tmp/qwen3-4b（训练时 cp 到 /tmp 的本地路径），
# 但 eval 用 vllm serve $BASE_MODEL + --lora-modules 显式指定 base，不依赖 adapter_config 的 base 字段，
# 故 preflight 断言放宽（见下）。
ADAPTER_PATH="${ADAPTER_PATH:-${TRIM_ROOT}/outputs/cloudml_custom_train_harness1_bcplus_full_qwen3_4b_rl_ddp_200step_zero_32gpu_4node_20260925/checkpoints/rl/step_000200/adapter}"
BCPLUS_ROOT="${JFS_ROOT}/Capability_Evolution/SCOPE/external/BrowseComp-Plus"
BCPLUS_CORPUS="${BCPLUS_ROOT}/data/browsecomp_plus_corpus_full.jsonl"
BCPLUS_INDEX="${BCPLUS_ROOT}/indexes/bm25"
TIKTOKEN_ZIP="${JFS_ROOT}/tiktoken-bundle.zip"
TIKTOKEN_BUNDLE="/tmp/tiktoken-bundle"
PYPI_INDEX="https://pkgs.d.xiaomi.net/artifactory/api/pypi/pypi-virtual/simple"
EVAL_SCRIPT="${TRIM_ROOT}/scripts/run_bcplus_w_harness1_adapter_gpu8_eval.sh"

echo "=================================================================="
echo "[boot] TRIM Harness-1 BC+ full / Qwen3-4B / ADAPTER eval (9ixnl step200) / 8gpu  ($(date -Iseconds 2>/dev/null || echo unknown))"
echo "[boot] host=$(hostname) pwd=$(pwd) user=$(whoami) argv=$*"
echo "[boot] base=${MODEL_PATH} adapter=${ADAPTER_PATH}"
echo "=================================================================="

export PYTHONUNBUFFERED=1

# ---- 1. 依赖安装（独立诊断 attempt，按 CLAUDE.md 经验镜像连 torch 都缺）----
# 不要把 torch==2.6.0 与 vllm==0.19.1 一起装，后者要求 torch==2.10.0。
# 基础镜像 distutils blinker 1.4 阻断 pip 卸载，用内部 Xiaomi PyPI + --ignore-installed。
# 装包顺序：先四件套，再装 chz/tinker（必须 --no-deps 保护四件套，见 CLAUDE.md 经验）。
echo "[deps] installing locked runtime from ${PYPI_INDEX}"
python3 -m pip install --no-cache-dir --ignore-installed \
  --upgrade-strategy only-if-needed \
  --index-url "${PYPI_INDEX}" \
  "torch==2.10.0" \
  "transformers==4.57.6" \
  "accelerate" \
  "datasets" \
  "vllm==0.19.1" \
  "hydra-core" \
  "omegaconf" \
  "tensordict" \
  "codetiming" \
  "tqdm" \
  "python-dotenv" \
  "pandas" \
  "pyarrow" \
  "sentencepiece" \
  "pyserini==2.0.0" \
  "structlog" \
  "chromadb" \
  "peft" \
  "openai" \
  "tenacity" \
  "pydantic-settings"

# chz/tinker/json_repair 用 --no-deps，避免它们拉取会破坏四件套的依赖（eval 导入链 load_upstream_modules 需要）。
python3 -m pip install --no-cache-dir --no-deps \
  --index-url "${PYPI_INDEX}" \
  "chz" \
  "tinker" \
  "json_repair" || echo "[deps] WARN: chz/tinker/json_repair install had issues (may not block eval)"

echo "[deps] printing resolved versions:"
python3 - <<'PY'
import importlib.metadata
import torch, vllm, transformers
print(f"[deps] torch={torch.__version__} cuda={torch.version.cuda}")
print(f"[deps] vllm={vllm.__version__} transformers={transformers.__version__}")
try:
    print(f"[deps] pyserini={importlib.metadata.version('pyserini')}")
except Exception as e:
    print(f"[deps] pyserini import error: {e}")
PY

# ---- 2. preflight: GPU / runtime 版本（任一不符不得进入 eval）----
echo "[preflight][gpu] nvidia-smi summary:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv 2>&1 | head -20 || echo "[preflight][gpu] nvidia-smi unavailable"
python3 - <<'PY'
import os, sys, importlib.metadata
import torch, vllm, transformers
print(f"[preflight][rt] torch={torch.__version__} cuda={torch.version.cuda} "
      f"device_count={torch.cuda.device_count()} visible={os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"[preflight][rt] vllm={vllm.__version__} transformers={transformers.__version__}")
assert torch.cuda.device_count() == 8, f"expected 8 GPUs, got {torch.cuda.device_count()}"
pyserini_ver = importlib.metadata.version("pyserini")
print(f"[preflight][rt] pyserini={pyserini_ver}")
assert pyserini_ver == "2.0.0", f"pyserini must be 2.0.0, got {pyserini_ver}"
PY

# ---- 3. preflight: Java / PyJNI / Lucene（BC+ retrieval 走本地 BM25 Lucene）----
echo "[preflight][jvm] locating libjvm.so"
JVM_PATH="$(find /usr /opt -type f -name libjvm.so 2>/dev/null | head -n 1 || true)"
if [ -z "${JVM_PATH}" ]; then
  echo "[preflight][jvm] libjvm.so not found, installing openjdk-21-jre-headless"
  apt-get update
  apt-get install -y --no-install-recommends openjdk-21-jre-headless
  JVM_PATH="$(find /usr /opt -type f -name libjvm.so 2>/dev/null | head -n 1)"
fi
test -n "${JVM_PATH}" && test -f "${JVM_PATH}"
ldd "${JVM_PATH}" >/dev/null
export JVM_PATH JAVA_HOME
JAVA_HOME="$(dirname "$(dirname "$(dirname "${JVM_PATH}")")")"
echo "[preflight][jvm] JVM_PATH=${JVM_PATH} JAVA_HOME=${JAVA_HOME}"
export LD_LIBRARY_PATH="$(dirname "${JVM_PATH}"):${LD_LIBRARY_PATH:-}"
python3 -c 'from jnius import autoclass; print("[preflight][jvm] java_ok", autoclass("java.lang.String"))'

# ---- 4. preflight: 导入链（harness 关键依赖）+ ANSERINI_CLASSPATH ----
echo "[preflight][import] structlog/chromadb/pyserini/peft"
python3 - <<'PY'
import structlog, chromadb, pyserini, peft
print(f"[preflight][import] structlog={structlog.__version__} peft={peft.__version__}")
print("[preflight][import] chromadb/pyserini/peft OK")
PY
PYSERINI_ROOT=$(python3 -c 'import pathlib, pyserini; print(pathlib.Path(pyserini.__file__).resolve().parent)')
export ANSERINI_CLASSPATH="$(find "${PYSERINI_ROOT}/resources/jars" -type f -name '*.jar' -print | paste -sd: -)"
test -n "${ANSERINI_CLASSPATH}"
export CLASSPATH="${ANSERINI_CLASSPATH}"
echo "[preflight][import] ANSERINI_CLASSPATH=${ANSERINI_CLASSPATH}"

# ---- 5. preflight: JuiceFS 挂载与关键路径（含训练 adapter）----
echo "[preflight][fs] JuiceFS mount check:"
mount | grep -E "songzijun|JuiceFS" || echo "[preflight][fs] no songzijun mount line (may be CSI)"
for p in "${JFS_ROOT}" "${TRIM_ROOT}" "${EASYOPD_ROOT}" "${MODEL_PATH}" \
         "${ADAPTER_PATH}" "${BCPLUS_CORPUS}" "${BCPLUS_INDEX}" \
         "${TRIM_ROOT}/scripts/run_eval.py" "${EVAL_SCRIPT}"; do
  if [ -e "$p" ]; then
    echo "[preflight][fs] OK  $p"
  else
    echo "[preflight][fs] MISSING  $p" >&2
    exit 2
  fi
done
test -f "${MODEL_PATH}/config.json" \
  && echo "[preflight][fs] base model config.json OK" \
  || { echo "[preflight][fs] base model config.json MISSING" >&2; exit 2; }
test -f "${ADAPTER_PATH}/adapter_config.json" \
  && echo "[preflight][fs] adapter_config.json OK" \
  || { echo "[preflight][fs] adapter_config.json MISSING" >&2; exit 2; }
test -f "${ADAPTER_PATH}/adapter_model.safetensors" \
  && echo "[preflight][fs] adapter_model.safetensors OK" \
  || { echo "[preflight][fs] adapter_model.safetensors MISSING" >&2; exit 2; }
# adapter base_model_name_or_path 放宽校验：9ixnl 训练时 cp 模型到 /tmp/qwen3-4b，
# 故 adapter_config.base=/tmp/qwen3-4b（非 MODEL_PATH）。但 eval 用 vllm serve $BASE_MODEL
# + --lora-modules 显式指定 base，不依赖 adapter_config 的 base 字段，故只校验 r=8 + target_modules。
python3 - <<PY
import json
cfg = json.load(open("${ADAPTER_PATH}/adapter_config.json"))
base = cfg.get("base_model_name_or_path","")
r = int(cfg.get("r") or 64)
targets = cfg.get("target_modules") or []
print(f"[preflight][adapter] base_model_name_or_path={base} r={r} alpha={cfg.get('lora_alpha')} targets={targets}")
assert r == 8, f"expected r=8, got {r}"
assert set(targets) >= {"q_proj","k_proj","o_proj","v_proj"}, f"unexpected target_modules={targets}"
# base 放宽：允许 /tmp/qwen3-4b（训练本地路径）或 .../Qwen3-4B-Instruct-2507
assert base == "/tmp/qwen3-4b" or base.endswith("Qwen3-4B-Instruct-2507"), \
    f"adapter base {base} neither /tmp/qwen3-4b (train-local) nor Qwen3-4B-Instruct-2507"
print("[preflight][adapter] r=8 + q/k/o/v_proj OK, base=/tmp/qwen3-4b (train-local, eval uses --lora-modules explicit base)")
PY

# ---- 6. preflight: tiktoken 离线词表（harness 链 import openai_harmony 需 o200k_base）----
echo "[preflight][tiktoken] resolving bundle:"
if [ ! -d "${TIKTOKEN_BUNDLE}/share/tiktoken" ]; then
  echo "[preflight][tiktoken] extracting ${TIKTOKEN_ZIP} -> /tmp"
  python3 - <<PY
import zipfile, os, shutil
src = "${TIKTOKEN_ZIP}"; dst = "${TIKTOKEN_BUNDLE}"
if os.path.exists(dst): shutil.rmtree(dst, ignore_errors=True)
with zipfile.ZipFile(src) as z: z.extractall("/tmp")
print(f"[preflight][tiktoken] extracted -> {dst} exists={os.path.isdir(dst)}")
PY
fi
test -f "${TIKTOKEN_BUNDLE}/share/tiktoken/o200k_base.tiktoken" \
  && echo "[preflight][tiktoken] o200k_base.tiktoken OK" \
  || { echo "[preflight][tiktoken] o200k_base.tiktoken MISSING" >&2; exit 2; }
export TIKTOKEN_ENCODINGS_BASE="${TIKTOKEN_BUNDLE}/share/tiktoken"
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_BUNDLE}/tiktoken_cache"
export TIKTOKEN_RS_CACHE_DIR="${TIKTOKEN_BUNDLE}/tiktoken_rs_cache"
test -f "${TIKTOKEN_ENCODINGS_BASE}/o200k_base.tiktoken" \
  && echo "[preflight][tiktoken] TIKTOKEN_ENCODINGS_BASE=${TIKTOKEN_ENCODINGS_BASE} (o200k OK)" \
  || { echo "[preflight][tiktoken] o200k not under TIKTOKEN_ENCODINGS_BASE" >&2; exit 2; }

# ---- 7. 离线凭证占位（eval --offline；run_eval 可能仍读 OPENAI_API_KEY 占位）----
: "${OPENAI_API_KEY:=**-*****-*******}"
: "${CHROMA_API_KEY:=local-offline}"
: "${CHROMA_DATABASE:=local-offline}"
export OPENAI_API_KEY CHROMA_API_KEY CHROMA_DATABASE

# ---- 8. eval 环境变量 ----
# Qwen3-4B adapter eval: 8 单卡 vLLM actor (TP=1) 各加载 base+LoRA (enable_lora + lora-modules)。
# hermes tool-call parser（model_profiles 自动产出）。COMPONENT=zero 无 verifier。
export PYTHONPATH="${TRIM_ROOT}:${EASYOPD_ROOT}"
export BENCHMARK="${BENCHMARK:-longsealqa}"
export RUN_ID="${RUN_ID:-qwen3_4b_transfer_9ixnl_step200_8gpu_20260928_memqa_v5}"

# 充分利用 8 卡显存（/goal）：
#   GPU_UTIL=0.90 — 纯推理 eval 无 backward OOM 风险，提大 KV pool 利好短 query 批处理吞吐；
#                   4B 权重~8GB + LoRA，余 ~64GB KV/卡。
#   EVAL_TP=32    — 32 API-client workers，8 actor 各 4 workers，actor 批处理 ~4 并发 seq。
#   VLLM_MAX_NUM_SEQS=64 — vLLM 单 actor 并发序列上限，留 burst 余量。
#   MAX_TURNS=40  — Harness-1 Table-2 eval 标配。
export ADAPTER="${ADAPTER_PATH}"
export BASE_MODEL="${MODEL_PATH}"
export COMPONENT="zero"
export LAYOUT="multi"
export ACTOR_GPUS="0,1,2,3,4,5,6,7"
export EVAL_TP="${EVAL_TP:-32}"
export GPU_UTIL="${GPU_UTIL:-0.90}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}"
export MAX_TURNS="${MAX_TURNS:-40}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
export TEMPERATURE="${TEMPERATURE:-1.0}"
export WORKER_STAGGER="${WORKER_STAGGER:-2}"
# CloudML pytorch 框架注入 RANK/WORLD_SIZE=1 可能干扰，清掉（eval 不走 torchrun）。
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK NODE_RANK MASTER_ADDR MASTER_PORT PET_NPROC_PER_NODE 2>/dev/null || true

echo "[env] PYTHONPATH=${PYTHONPATH}"
echo "[env] ADAPTER=${ADAPTER}"
echo "[env] BASE_MODEL=${BASE_MODEL}"
echo "[env] COMPONENT=${COMPONENT} LAYOUT=${LAYOUT} ACTOR_GPUS=${ACTOR_GPUS}"
echo "[env] EVAL_TP=${EVAL_TP} GPU_UTIL=${GPU_UTIL} VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS}"
echo "[env] MAX_TURNS=${MAX_TURNS} MAX_NEW_TOKENS=${MAX_NEW_TOKENS} TEMPERATURE=${TEMPERATURE}"
echo "[env] TIKTOKEN_ENCODINGS_BASE=${TIKTOKEN_ENCODINGS_BASE}"
echo "[env] RUN_ID=${RUN_ID}"
echo "[env] OPENAI_API_KEY set (placeholder, offline)"

# ---- 9. 启动 adapter eval ----
echo "=================================================================="
echo "[eval] bash ${EVAL_SCRIPT}  (ADAPTER=${ADAPTER} BASE=${BASE_MODEL} COMPONENT=${COMPONENT})"
echo "=================================================================="
cd "${TRIM_ROOT}"
exec bash "${EVAL_SCRIPT}"
