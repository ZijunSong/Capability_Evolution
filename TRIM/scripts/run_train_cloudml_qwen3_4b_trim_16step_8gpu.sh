#!/usr/bin/env bash
# CloudML 启动脚本：TRIM Harness-1 BC+ full / Qwen3-4B-Instruct-2507 (原始 bf16，非反量化) / trim / 8x gm402-80g
# 目标：充分利用 8 卡显存（gpu_memory_utilization=0.90），16 步 trim 训练持续运行。
# 镜像: micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto
#   （该镜像基础层缺 torch 等依赖，按 CLAUDE.md 经验在启动阶段显式锁定并安装四包配置）
# JuiceFS: volume=songzijun, cluster=bj9-cloudml (endpoint), mount=/mnt/songzijun
#   endpoint 由 CLI --juicefs_mount_configs 显式传入（不在本脚本/yaml 里）。
# 依赖锁定对齐 TRIM Harness-1 BC+ 已验证的 runtime：
#   torch==2.10.0+cu128 / transformers==4.57.6 / vllm==0.19.1 / pyserini 2.0.0
#   + structlog/chromadb/openai_harmony/chz/tinker/json_repair
#
# 关键经验（CLAUDE.md / memory）：
#   - gm402 多卡 vLLM TP：custom all-reduce 在 engine 重启（step>=1 rollout）后死锁，改 NCCL（DISABLE_CUSTOM_ALL_REDUCE=1，worker 据此设 disable_custom_all_reduce=True）。
#   - --max-turns 默认 6，Python 不读 MAX_TURNS env，必须 CLI 显式传 --max-turns 40。
#   - CISPO backward 在 0.90 显存高水位下可能 OOM（GPT-OSS-20B recovery1 经验）；先按用户要求不带 CUDA_LAUNCH_BLOCKING=1，若 step0 出 CUDA launch race 再恢复时加。
#   - ImageCommand 直跑 JuiceFS bash 脚本时 heredoc 回读可能失败：先 cp 到 /tmp 再 exec。
# 由 ImageCommand: bash -lc 'cp 脚本 /tmp && bash /tmp/<script>' 调用，幂等可审计；preflight 全量打印后再进训练。
set -euo pipefail

JFS_ROOT="/mnt/songzijun"
TRIM_ROOT="${JFS_ROOT}/Capability_Evolution/TRIM"
EASYOPD_ROOT="${JFS_ROOT}/Capability_Evolution/SCAPE-EasyOPD"
MODEL_PATH="${JFS_ROOT}/models/Qwen3-4B-Instruct-2507"
RL_DATA="${TRIM_ROOT}/data/harness-1-rl-data"
SEC_CORPUS_ROOT="${TRIM_ROOT}/data/harness-1-sec-corpus"
TIKTOKEN_ZIP="${JFS_ROOT}/tiktoken-bundle.zip"
TIKTOKEN_BUNDLE="/tmp/tiktoken-bundle"
PYPI_INDEX="https://pkgs.d.xiaomi.net/artifactory/api/pypi/pypi-virtual/simple"
OUT="${TRIM_ROOT}/outputs/cloudml_trim_run_train_harness1_bcplus_full_qwen3_4b_trim_16step_8gpu_$(date +%Y%m%d_%H%M%S 2>/dev/null || echo unknown)"

echo "=================================================================="
echo "[boot] TRIM Harness-1 BC+ full / Qwen3-4B / trim / 8gpu ($(date -Iseconds 2>/dev/null || echo unknown))"
echo "[boot] host=$(hostname) pwd=$(pwd) user=$(whoami) argv=$*"
echo "[boot] OUT=${OUT}"
echo "=================================================================="

export PYTHONUNBUFFERED=1

# ---- 1. 依赖安装（独立诊断 attempt，按 CLAUDE.md 经验镜像连 torch 都缺）----
# 不要把 torch==2.6.0 与 vllm==0.19.1 一起装，后者要求 torch==2.10.0。
# 基础镜像 distutils blinker 1.4 阻断 pip 卸载，用内部 Xiaomi PyPI + --ignore-installed。
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
  "pyserini" \
  "structlog" \
  "chromadb" \
  "openai" \
  "tenacity" \
  "anthropic" \
  "peft" \
  "openai_harmony" \
  "chz" \
  "tinker" \
  "json_repair" \
  "pydantic-settings"

# ---- 2. preflight: GPU / runtime 版本（任一不符不得进入训练）----
echo "[preflight][gpu] nvidia-smi summary:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv 2>&1 | head -20 || echo "[preflight][gpu] nvidia-smi unavailable"
python3 - <<'PY'
import os, sys
import torch, vllm, transformers
print(f"[preflight][rt] torch={torch.__version__} cuda={torch.version.cuda} "
      f"device_count={torch.cuda.device_count()} visible={os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"[preflight][rt] vllm={vllm.__version__} transformers={transformers.__version__}")
assert torch.cuda.device_count() == 8, f"expected 8 GPUs, got {torch.cuda.device_count()}"
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

# ---- 4. preflight: 导入链（harness.ultra_core / harness.tools 关键依赖）----
echo "[preflight][import] structlog/chromadb/pyserini/peft/openai_harmony"
python3 - <<'PY'
import structlog, chromadb, pyserini, peft
print(f"[preflight][import] structlog={structlog.__version__} peft={peft.__version__}")
import openai_harmony
print("[preflight][import] openai_harmony OK")
PY

# ---- 5. preflight: JuiceFS 挂载与关键路径 ----
echo "[preflight][fs] JuiceFS mount check:"
mount | grep -E "songzijun|JuiceFS" || echo "[preflight][fs] no songzijun mount line (may be CSI)"
for p in "${JFS_ROOT}" "${TRIM_ROOT}" "${EASYOPD_ROOT}" "${MODEL_PATH}" "${RL_DATA}" \
         "${SEC_CORPUS_ROOT}/corpora/sec/train" "${SEC_CORPUS_ROOT}/indexes/bm25" \
         "${TRIM_ROOT}/scripts/run_train.py"; do
  if [ -e "$p" ]; then
    echo "[preflight][fs] OK  $p"
  else
    echo "[preflight][fs] MISSING  $p" >&2
    exit 2
  fi
done

# ---- 6. preflight: tiktoken 离线词表（o200k_harmony 用同一份 o200k_base.tiktoken）----
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
test -f "${TIKTOKEN_BUNDLE}/share/tiktoken/cl100k_base.tiktoken" \
  && echo "[preflight][tiktoken] cl100k_base.tiktoken OK" \
  || echo "[preflight][tiktoken] cl100k_base.tiktoken MISSING (warn)"
export TIKTOKEN_ENCODINGS_BASE="${TIKTOKEN_BUNDLE}/share/tiktoken"
export TIKTOKEN_RS_CACHE_DIR="${TIKTOKEN_BUNDLE}/tiktoken_rs_cache"
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_BUNDLE}/tiktoken_cache"

# ---- 7. 离线凭证占位（run_train.py 会再次调用 ensure_local_offline_credentials）----
: "${OPENAI_API_KEY:=**-*****-*******}"
: "${CHROMA_API_KEY:=local-offline}"
: "${CHROMA_DATABASE:=local-offline}"
export OPENAI_API_KEY CHROMA_API_KEY CHROMA_DATABASE

# ---- 8. 训练环境变量 ----
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MAX_TURNS=40
export PYTHONPATH="${TRIM_ROOT}:${EASYOPD_ROOT}"
export SCAPE_SEC_RL_DATA="${RL_DATA}"
export SCAPE_SEC_CORPUS_ROOT="${SEC_CORPUS_ROOT}"
export SCAPE_LOCAL_OPENAI_EMBEDDINGS=1
export SCAPE_FORCE_LOCAL_HARMONY=1
export SCAPE_CHROMA_PATH=/tmp/empty_chroma
mkdir -p "$SCAPE_CHROMA_PATH"
# gm402 多卡 vLLM TP：custom all-reduce 在 engine 重启（step>=1 rollout）后死锁，改用 NCCL all-reduce。
# vllm_rollout_worker 据此设 disable_custom_all_reduce=True（vllm_rollout_worker.py:141-149 读 DISABLE_CUSTOM_ALL_REDUCE）。
export DISABLE_CUSTOM_ALL_REDUCE=1
# 按 CLAUDE.md 经验 + 8 卡显存拉满需求，先把 KV/cache 显存打高（充分利用 8 卡显存）。
# 框架 run_train.py 默认 gpu_memory_utilization=0.90；下面用 --gpu-memory-utilization 0.90 显式固化。
# PYTORCH_CUDA_ALLOC_CONF 用 expandable_segments:True 可降低碎片（注意不要用与 vLLM memory pool 不兼容的配置）。
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# CISPO update CUDA launch race 修复（CLAUDE.md / memory trim-rl-step0-cispo-update-cuda-launch-failure）：
# Qwen3-4B 8 卡 step1 的 CISPO backward 在 accelerate device_map=auto 层间调度下出现异步 CUDA launch
# failure，表现为 backward 阶段主进程静默 hung（无 traceback，无新产物，无心跳）。加 CUDA_LAUNCH_BLOCKING=1
# 让 CUDA launch 同步执行，规避层间调度竞态；代价是变慢但能跑通（recovery 成功经验见 trim-rl-recov4step-cuda-launch-blocking-success）。
# 本次为 recovery1：前序 job t-20260914141132-3ojfs 在 optim_step_start step=1 后静默 ~31min 判定 hung 已 kill（保留 attempt）。
export CUDA_LAUNCH_BLOCKING=1

echo "[env] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[env] MAX_TURNS=${MAX_TURNS} (--max-turns 40 CLI 才生效；env 仅为记录)"
echo "[env] PYTHONPATH=${PYTHONPATH}"
echo "[env] SCAPE_SEC_RL_DATA=${SCAPE_SEC_RL_DATA}"
echo "[env] SCAPE_SEC_CORPUS_ROOT=${SCAPE_SEC_CORPUS_ROOT}"
echo "[env] DISABLE_CUSTOM_ALL_REDUCE=${DISABLE_CUSTOM_ALL_REDUCE}"
echo "[env] PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
echo "[env] CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING} (recovery1: 规避 CISPO backward 层间调度竞态)"
echo "[env] model_name=${MODEL_PATH}"

# ---- 9. py_compile 入口，确认无语法/补丁问题 ----
python3 -m py_compile "${TRIM_ROOT}/scripts/run_train.py"

# ---- 10. 启动训练 ----
echo "=================================================================="
echo "[train] cd ${JFS_ROOT}/Capability_Evolution && python TRIM/scripts/run_train.py ..."
echo "[train] train_method=trim / train-steps=16 / groups-per-step=32 / group-size=8"
echo "[train] train-micro-batch-size=4 / max-turns=40 / max-model-len=8192 / gpu-mem-util=0.90"
echo "[train] OUT=${OUT}"
echo "=================================================================="
cd "${JFS_ROOT}/Capability_Evolution"

exec python3 TRIM/scripts/run_train.py \
  --harness Harness-1 \
  --benchmark bcplus_full \
  --model_name "${MODEL_PATH}" \
  --train_method trim \
  --component all \
  --train-env local_legacy \
  --train-data sec \
  --rl-data "${RL_DATA}" \
  --sec-corpus-root "${SEC_CORPUS_ROOT}" \
  --train-steps 16 \
  --train-groups-per-step 32 \
  --group-size 8 \
  --train-micro-batch-size 4 \
  --train-heartbeat-every 8 \
  --max-turns 40 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --out "${OUT}"
