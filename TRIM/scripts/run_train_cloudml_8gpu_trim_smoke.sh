#!/usr/bin/env bash
# CloudML 启动脚本：TRIM Harness-1 BC+ full / GPT-OSS-20B / TRIM (smoke) / 单节点 8x gm402-80g
# 由 run_train_cloudml_8gpu.sh 派生：--train_method rl->trim；
# recovery2: --train-micro-batch-size 4->1 + --gpu-memory-utilization 0.60
#   （修 recovery1 step1 CISPO backward OOM：micro_batch=4 attention 峰值超 79GB）
# （默认 train-steps / group-size / max-model-len / 输出目录）。
# 镜像: micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto
#   （该镜像基础层缺 torch 等依赖，按 CLAUDE.md 经验在启动阶段显式锁定并安装四包配置）
# JuiceFS: volume=songzijun, cluster=bj9-cloudml (endpoint), mount=/mnt/songzijun
# 依赖锁定对齐 TRIM Harness-1 BC+ 已验证的 runtime：
#   torch==2.10.0 / transformers==4.57.6 / vllm==0.19.1 + structlog/chromadb/pyserini/peft/openai_harmony
# 由 ImageCommand 调用；保持幂等、可审计；preflight 全量打印后再进训练。
set -euo pipefail

JFS_ROOT="/mnt/songzijun"
TRIM_ROOT="${JFS_ROOT}/Capability_Evolution/TRIM"
EASYOPD_ROOT="${JFS_ROOT}/Capability_Evolution/SCAPE-EasyOPD"
MODEL_PATH="${JFS_ROOT}/models/openai/gpt-oss-20b"
RL_DATA="${TRIM_ROOT}/data/harness-1-rl-data"
SEC_CORPUS_ROOT="${TRIM_ROOT}/data/harness-1-sec-corpus"
TIKTOKEN_ZIP="${JFS_ROOT}/tiktoken-bundle.zip"
TIKTOKEN_BUNDLE="/tmp/tiktoken-bundle"
PYPI_INDEX="https://pkgs.d.xiaomi.net/artifactory/api/pypi/pypi-virtual/simple"

echo "=================================================================="
echo "[boot] TRIM Harness-1 BC+ full / GPT-OSS-20B / TRIM smoke / 8gpu  ($(date -Iseconds 2>/dev/null || echo unknown))"
echo "[boot] host=$(hostname) pwd=$(pwd) user=$(whoami) argv=$*"
echo "=================================================================="

export PYTHONUNBUFFERED=1

# ---- 1. 依赖安装（独立诊断 attempt，按 CLAUDE.md 经验镜像连 torch 都缺）----
# 注意：不要把 torch==2.6.0 与 vllm==0.19.1 一起装，后者要求 torch==2.10.0。
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
  "wandb" \
  "tensorboard" \
  "math-verify" \
  "pylatexenc" \
  "python-dotenv" \
  "pandas" \
  "pyarrow" \
  "sentencepiece" \
  "latex2sympy2" \
  "word2number" \
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

# ---- 1b. vLLM 0.19.1 LoRA shrink op 补丁：assert inputs.is_contiguous -> inputs.contiguous() ----
# recovery2 在 rollout 加载 LoRA adapter 时崩于 lora_shrink_op.py:177 assert inputs.is_contiguous()，
# 现成补丁 patch_vllm_lora_shrink_contiguous.py 将断言改为强制连续化。
echo "[patch] applying vLLM lora_shrink contiguous patch"
python3 "${TRIM_ROOT}/scripts/patch_vllm_lora_shrink_contiguous.py" \
  || { echo "[patch] FAILED vLLM lora_shrink patch" >&2; exit 3; }

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
  echo "[preflight][jvm] libjvm.so not found, installing openjdk-17-jre-headless"
  apt-get update
  apt-get install -y --no-install-recommends openjdk-17-jre-headless
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
# gm402 多卡 vLLM TP：custom all-reduce 在 engine 重启（step>=1 rollout）后死锁，
# 改用 NCCL all-reduce（CLAUDE.md 经验）。vllm_rollout_worker 据此设 disable_custom_all_reduce=True。
export DISABLE_CUSTOM_ALL_REDUCE=1
# recovery5: PyTorch caching allocator 用 expandable_segments 消除跨 step 的
# reserved-but-unallocated 碎片（step2 backward done=1 即 OOM，11.96GiB 碎片）。
# 安全性：SFT 路径 hf_sft.py:280,521 已用同一设置跑 GPT-OSS-20B+8GPU；vLLM 0.19.1 V1
# enforce_eager 下不与 CUDAGraph 交互，expandable_segments 只改段扩展策略不破坏正确性。
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[env] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[env] MAX_TURNS=${MAX_TURNS}"
echo "[env] PYTHONPATH=${PYTHONPATH}"
echo "[env] SCAPE_SEC_RL_DATA=${SCAPE_SEC_RL_DATA}"
echo "[env] SCAPE_SEC_CORPUS_ROOT=${SCAPE_SEC_CORPUS_ROOT}"
echo "[env] OPENAI_API_KEY set (placeholder, offline)"
echo "[env] model_name=${MODEL_PATH}"

# ---- 9. 启动训练 ----
echo "=================================================================="
echo "[train] cd ${JFS_ROOT}/Capability_Evolution && python TRIM/scripts/run_train.py ..."
echo "=================================================================="
cd "${JFS_ROOT}/Capability_Evolution"

exec python TRIM/scripts/run_train.py \
  --harness Harness-1 \
  --benchmark bcplus_full \
  --model_name "${MODEL_PATH}" \
  --train_method trim \
  --train-env local_legacy \
  --component all \
  --train-groups-per-step 32 \
  --train-micro-batch-size 1 \
  --gpu-memory-utilization 0.60 \
  --max-model-len 4096 \
  --train-heartbeat-every 8 \
  --out "${JFS_ROOT}/Capability_Evolution/TRIM/outputs/train_harness1_bcplus_full_gpt-oss-20b_trim_all_recovery5"
