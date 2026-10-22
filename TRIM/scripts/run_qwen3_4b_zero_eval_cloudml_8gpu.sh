#!/usr/bin/env bash
# CloudML 启动脚本：TRIM Harness-1 BC+ full / Qwen3-4B-Instruct-2507 / ZERO eval / 单节点 8x gm402-80g
# 镜像: micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto
#   （该镜像基础层缺 torch 等依赖，按 CLAUDE.md 经验在启动阶段显式锁定并安装四包配置）
# JuiceFS: volume=songzijun, JuiceFSCluster=bj9-cloudml (endpoint), mount=/mnt/songzijun
#   — JuiceFS 挂载由 CloudML 通过 CLI --juicefs_mount_configs 注入，不在本脚本处理。
# 依赖锁定对齐 TRIM Harness-1 BC+ 已验证的 runtime：
#   torch==2.10.0 / transformers==4.57.6 / vllm==0.19.1 / pyserini==2.0.0 + structlog/chromadb/openai_harmony
#   (pyserini 必须固定 2.0.0；裸装会装到 2.4.0 致 preflight 断言失败 —— CLAUDE.md 经验)
# 由 ImageCommand 调用；保持幂等、可审计；preflight 全量打印后再进 eval。
# eval 脚本：TRIM/scripts/run_bcplus_full_qwen3_4b_zero_gpu8_eval.sh
#   （8 个单卡 vLLM actor TP=1，--tp 16 workers，local_bm25 无 Chroma，无 verifier）
set -euo pipefail

JFS_ROOT="/mnt/songzijun"
TRIM_ROOT="${JFS_ROOT}/Capability_Evolution/TRIM"
EASYOPD_ROOT="${JFS_ROOT}/Capability_Evolution/SCAPE-EasyOPD"
MODEL_PATH="${JFS_ROOT}/models/Qwen3-4B-Instruct-2507"
BCPLUS_ROOT="${JFS_ROOT}/Capability_Evolution/SCOPE/external/BrowseComp-Plus"
BCPLUS_CORPUS="${BCPLUS_ROOT}/data/browsecomp_plus_corpus_full.jsonl"
BCPLUS_INDEX="${BCPLUS_ROOT}/indexes/bm25"
TIKTOKEN_ZIP="${JFS_ROOT}/tiktoken-bundle.zip"
TIKTOKEN_BUNDLE="/tmp/tiktoken-bundle"
PYPI_INDEX="https://pkgs.d.xiaomi.net/artifactory/api/pypi/pypi-virtual/simple"
EVAL_SCRIPT="${TRIM_ROOT}/scripts/run_bcplus_full_qwen3_4b_zero_gpu8_eval.sh"

echo "=================================================================="
echo "[boot] TRIM Harness-1 BC+ full / Qwen3-4B / ZERO eval / 8gpu  ($(date -Iseconds 2>/dev/null || echo unknown))"
echo "[boot] host=$(hostname) pwd=$(pwd) user=$(whoami) argv=$*"
echo "=================================================================="

export PYTHONUNBUFFERED=1

# ---- 1. 依赖安装（独立诊断 attempt，按 CLAUDE.md 经验镜像连 torch 都缺）----
# 注意：不要把 torch==2.6.0 与 vllm==0.19.1 一起装，后者要求 torch==2.10.0。
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
  "openai" \
  "tenacity" \
  "pydantic-settings"

# chz/tinker/json_repair 用 --no-deps，避免它们拉取会破坏四件套的依赖。
python3 -m pip install --no-cache-dir --no-deps \
  --index-url "${PYPI_INDEX}" \
  "chz" \
  "tinker" \
  "json_repair" || echo "[deps] WARN: chz/tinker/json_repair install had issues (eval zero may not need tinker)"

echo "[deps] printing resolved versions:"
python3 - <<'PY'
import torch, vllm, transformers
print(f"[deps] torch={torch.__version__} cuda={torch.version.cuda}")
print(f"[deps] vllm={vllm.__version__} transformers={transformers.__version__}")
try:
    import pyserini
    print(f"[deps] pyserini={getattr(pyserini,'__version__','?')}")
except Exception as e:
    print(f"[deps] pyserini import error: {e}")
PY

# ---- 2. preflight: GPU / runtime 版本（任一不符不得进入 eval）----
echo "[preflight][gpu] nvidia-smi summary:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv 2>&1 | head -20 || echo "[preflight][gpu] nvidia-smi unavailable"
python3 - <<'PY'
import os, sys
import torch, vllm, transformers
print(f"[preflight][rt] torch={torch.__version__} cuda={torch.version.cuda} "
      f"device_count={torch.cuda.device_count()} visible={os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"[preflight][rt] vllm={vllm.__version__} transformers={transformers.__version__}")
assert torch.cuda.device_count() == 8, f"expected 8 GPUs, got {torch.cuda.device_count()}"
import pyserini
print(f"[preflight][rt] pyserini={getattr(pyserini,'__version__','?')}")
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
# libjvm.so 在 <java>/lib/server/libjvm.so；JAVA_HOME 往上退三层到 jdk 根。
JAVA_HOME="$(dirname "$(dirname "$(dirname "${JVM_PATH}")")")"
echo "[preflight][jvm] JVM_PATH=${JVM_PATH} JAVA_HOME=${JAVA_HOME}"
export LD_LIBRARY_PATH="$(dirname "${JVM_PATH}"):${LD_LIBRARY_PATH:-}"
python3 -c 'from jnius import autoclass; print("[preflight][jvm] java_ok", autoclass("java.lang.String"))'

# ---- 4. preflight: 导入链（harness 关键依赖）----
echo "[preflight][import] structlog/chromadb/pyserini"
python3 - <<'PY'
import structlog, chromadb, pyserini
print(f"[preflight][import] structlog={structlog.__version__}")
print("[preflight][import] chromadb/pyserini OK")
PY

# ---- 5. preflight: JuiceFS 挂载与关键路径 ----
echo "[preflight][fs] JuiceFS mount check:"
mount | grep -E "songzijun|JuiceFS" || echo "[preflight][fs] no songzijun mount line (may be CSI)"
for p in "${JFS_ROOT}" "${TRIM_ROOT}" "${EASYOPD_ROOT}" "${MODEL_PATH}" \
         "${BCPLUS_CORPUS}" "${BCPLUS_INDEX}" \
         "${TRIM_ROOT}/scripts/run_eval.py" "${EVAL_SCRIPT}"; do
  if [ -e "$p" ]; then
    echo "[preflight][fs] OK  $p"
  else
    echo "[preflight][fs] MISSING  $p" >&2
    exit 2
  fi
done
test -f "${MODEL_PATH}/config.json" \
  && echo "[preflight][fs] model config.json OK" \
  || { echo "[preflight][fs] model config.json MISSING" >&2; exit 2; }

# ---- 6. preflight: tiktoken 离线词表（gpt-oss harmony 用 o200k_base；harness 链可能 import tiktoken）----
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
# 三个离线变量必须同时设置，对齐已 DONE 的 gpt-oss zero wrapper（recovery4 经验）：
# CloudML 无外网时若 harness 导入链 import openai_harmony，其 Rust 层读
# TIKTOKEN_ENCODINGS_BASE 指向的目录加载 o200k_base.tiktoken，缺该变量必崩 HarmonyError。
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
# Qwen zero eval 不做 tensor parallel（8 个单卡 actor TP=1），无需 disable_custom_all_reduce；
# 但保留 export 不影响（actor 各自单卡）。
export PYTHONPATH="${TRIM_ROOT}:${EASYOPD_ROOT}"
# RUN_ID 用 CloudML job 启动时间，便于 outputs 追溯。
export RUN_ID="${RUN_ID:-qwen3_4b_zero_8gpu_20260913_memqa_v5}"

# eval 并发与显存参数（在此显式 export 以保证到达 eval 脚本，不依赖 YAML EnvConfigs 是否被消费）。
# 充分利用 8 卡显存（/goal）：
#   GPU_UTIL=0.90 — 纯推理 eval 无 backward OOM 风险（GPT-OSS 0.90 OOM 仅见于训练 backward），
#                   提大 KV pool 利好短 query 批处理吞吐；4B 权重仅 ~8GB，余 ~64GB KV。
#   TP=32         — 4 workers/GPU（8 actor），低于已验证 "all" 配置 ~9 workers/GPU 的负载，安全；
#                   配合 max_num_seqs=8 让 actor 更忙，提升 830-query 吞吐。
#   MAX_NUM_SEQS=8 — 32k 上下文下单 seq KV ≈5.9GB，8 seq≈47GB，接近 80GB 显存上限，保留不升。
export TP="${TP:-32}"
export GPU_UTIL="${GPU_UTIL:-0.90}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
export WORKER_STAGGER="${WORKER_STAGGER:-10}"

echo "[env] PYTHONPATH=${PYTHONPATH}"
echo "[env] TP=${TP} GPU_UTIL=${GPU_UTIL} MAX_NUM_SEQS=${MAX_NUM_SEQS} MAX_MODEL_LEN=${MAX_MODEL_LEN} WORKER_STAGGER=${WORKER_STAGGER}"
echo "[env] TIKTOKEN_ENCODINGS_BASE=${TIKTOKEN_ENCODINGS_BASE}"
echo "[env] TIKTOKEN_CACHE_DIR=${TIKTOKEN_CACHE_DIR}"
echo "[env] TIKTOKEN_RS_CACHE_DIR=${TIKTOKEN_RS_CACHE_DIR}"
echo "[env] RUN_ID=${RUN_ID}"
echo "[env] OPENAI_API_KEY set (placeholder, offline)"
echo "[env] model=${MODEL_PATH}"

# ---- 9. 启动 eval ----
echo "=================================================================="
echo "[eval] bash ${EVAL_SCRIPT}"
echo "=================================================================="
cd "${TRIM_ROOT}"
exec bash "${EVAL_SCRIPT}"
