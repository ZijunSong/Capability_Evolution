#!/usr/bin/env bash
# CloudML 启动脚本：TRIM Harness-1 BC+ full / Qwen3.5-27B / ALL eval / 单节点 8x gm402-80g
# 镜像: micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto
#   （该镜像基础层缺 torch 等依赖，按 CLAUDE.md 经验在启动阶段显式锁定并安装四包配置）
# JuiceFS: volume=songzijun, JuiceFSCluster=bj9-cloudml (endpoint), mount=/mnt/songzijun
#   — JuiceFS 挂载由 CloudML 通过 CLI --juicefs_mount_configs 注入，不在本脚本处理。
# 依赖锁定对齐 TRIM Harness-1 BC+ 已验证的 runtime：
#   torch==2.10.0 / transformers==4.57.6 / vllm==0.19.1 / pyserini==2.0.0 + structlog/chromadb/openai_harmony
#   (pyserini 必须固定 2.0.0；裸装会装到 2.4.0 致 preflight 断言失败 —— CLAUDE.md 经验)
# 由 ImageCommand 调用（imageCommand 先 cp 本脚本到 /tmp 再 exec，规避 JuiceFS heredoc 回读抖动）；
# 保持幂等、可审计；preflight 全量打印后再进 eval。
#
# 模型：Qwen3.5-27B (Qwen3_5ForConditionalGeneration, dense 非 MoE, 51.75GiB bf16, 11 safetensors)。
#   走纯文本路径（BC+ 是纯文本 benchmark），tokenizer_class=Qwen2Tokenizer 自带 tokenizer.json。
#   profile=QWEN35: tool_call_parser=qwen3_coder, reasoning_parser=qwen3 (由 model_profiles 自动产出)。
# Thinking 关闭：wrapper 覆盖 vllm_extra_for_model 追加
#   --chat-template-kwargs {"enable_thinking":false}（无空格 JSON，经 ${extra} 分词为两个 argv token）。
# eval 脚本：TRIM/scripts/run_bcplus_w_harness1_qwen35_27b_all_gpu8_eval_nothinking.sh
#   （6 个 Qwen3.5-27B actor TP=1 on GPU 0-5 + harness-1 verifier on GPU 6，
#    run_eval.py --harness Harness-1 --evaluation-path upstream_api --component all --tp 64，
#    local_bm25 无 Chroma）
set -euo pipefail

JFS_ROOT="/mnt/songzijun"
TRIM_ROOT="${JFS_ROOT}/Capability_Evolution/TRIM"
EASYOPD_ROOT="${JFS_ROOT}/Capability_Evolution/SCAPE-EasyOPD"
MODEL_PATH="${JFS_ROOT}/models/Qwen3.5-27B"
VERIFY_MODEL_PATH="${JFS_ROOT}/models/harness-1"
BCPLUS_ROOT="${JFS_ROOT}/Capability_Evolution/SCOPE/external/BrowseComp-Plus"
BCPLUS_CORPUS="${BCPLUS_ROOT}/data/browsecomp_plus_corpus_full.jsonl"
BCPLUS_INDEX="${BCPLUS_ROOT}/indexes/bm25"
TIKTOKEN_ZIP="${JFS_ROOT}/tiktoken-bundle.zip"
TIKTOKEN_BUNDLE="/tmp/tiktoken-bundle"
PYPI_INDEX="https://pkgs.d.xiaomi.net/artifactory/api/pypi/pypi-virtual/simple"
# EVAL_SCRIPT_OVERRIDE lets the imageCommand copy the wrapper to /tmp (local FS,
# no JuiceFS cache lag) and point us at it. Falls back to the JuiceFS path.
EVAL_SCRIPT="${EVAL_SCRIPT_OVERRIDE:-${TRIM_ROOT}/scripts/run_bcplus_w_harness1_qwen35_27b_all_gpu8_eval_nothinking.sh}"

echo "=================================================================="
echo "[boot] TRIM Harness-1 BC+ full / Qwen3.5-27B / ALL (no-thinking) eval / 8gpu  ($(date -Iseconds 2>/dev/null || echo unknown))"
echo "[boot] host=$(hostname) pwd=$(pwd) user=$(whoami) argv=$*"
echo "=================================================================="

export PYTHONUNBUFFERED=1

# ---- 1. 依赖安装（按 CLAUDE.md 经验镜像连 torch 都缺）----
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
  "openai" \
  "tenacity" \
  "pydantic-settings"

# chz/tinker/json_repair 用 --no-deps，避免它们拉取会破坏四件套的依赖。
# upstream_api worker 路径（load_upstream_modules -> train_rl -> tinker_cookbook）需要它们。
python3 -m pip install --no-cache-dir --no-deps \
  --index-url "${PYPI_INDEX}" \
  "chz" \
  "tinker" \
  "json_repair" || echo "[deps] WARN: chz/tinker/json_repair install had issues"

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

# ---- 2b. preflight: Qwen3.5 架构在 vLLM 0.19.1 是否注册（关键风险）----
# config.json: architectures=Qwen3_5ForConditionalGeneration, model_type=qwen3_5。
# vLLM 0.19.1 是否支持该架构无法本地验证（本机 transformers 5.17.0、无 vllm）；
# 这里做注册检查，若 vLLM 不认该架构则干净退出，不浪费 GPU 跑半个任务。
python3 - <<'PY'
import sys
try:
    from vllm.model_executor.models.registry import ModelRegistry
    archs = ModelRegistry.get_supported_archs()
    qwen35 = [a for a in archs if "qwen3_5" in a.lower() or "qwen3.5" in a.lower()]
    print(f"[preflight][arch] vLLM qwen3_5-family archs = {qwen35}")
    print(f"[preflight][arch] vLLM all qwen-ish archs = {[a for a in archs if 'qwen' in a.lower()]}")
    if not qwen35:
        print("[preflight][arch] FATAL: vLLM 0.19.1 has no qwen3_5 architecture; loading will fail.")
        print("[preflight][arch] Fallback options: (A) upgrade vllm/transformers (breaks 4-pkg lock), "
              "(B) --trust-remote-code lets transformers load, but vLLM still needs arch registration.")
        sys.exit(5)
except Exception as e:
    print(f"[preflight][arch] vLLM arch probe failed (non-fatal, will let vllm serve decide): {e}")
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

# Pyserini ANSERINI_CLASSPATH（指向 resources/jars，CloudML 无外网不加载 prebuilt metadata）
PYSERINI_DIR="$(python3 -c 'import pyserini,os;print(os.path.dirname(pyserini.__file__))' 2>/dev/null || true)"
if [ -n "${PYSERINI_DIR}" ] && [ -d "${PYSERINI_DIR}/../resources/jars" ]; then
  export ANSERINI_CLASSPATH="${PYSERINI_DIR}/../resources/jars/*"
  echo "[preflight][jvm] ANSERINI_CLASSPATH=${ANSERINI_CLASSPATH}"
fi

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
for p in "${JFS_ROOT}" "${TRIM_ROOT}" "${EASYOPD_ROOT}" "${MODEL_PATH}" "${VERIFY_MODEL_PATH}" \
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

# ---- 6. preflight: tiktoken 离线词表（harness 链可能 import tiktoken/openai_harmony）----
# Qwen3.5 用 Qwen2Tokenizer 不需要 harmony 词表；但 verify 是 harness-1(gpt-oss, harmony)，
# 其链路 import openai_harmony 读 TIKTOKEN_ENCODINGS_BASE 加载 o200k_base.tiktoken，
# CloudML 无外网缺该变量必崩 HarmonyError。三个离线变量同时设置对齐已 DONE 的 gpt-oss wrapper。
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
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_BUNDLE}/tiktoken_cache"
export TIKTOKEN_RS_CACHE_DIR="${TIKTOKEN_BUNDLE}/tiktoken_rs_cache"

# ---- 7. 离线凭证占位（eval --offline；run_eval 可能仍读占位 KEY）----
: "${OPENAI_API_KEY:=**-*****-*******}"
: "${CHROMA_API_KEY:=local-offline}"
: "${CHROMA_DATABASE:=local-offline}"
export OPENAI_API_KEY CHROMA_API_KEY CHROMA_DATABASE

# ---- 8. eval 环境变量 ----
export PYTHONPATH="${TRIM_ROOT}:${EASYOPD_ROOT}"
# RUN_ID 用 CloudML job 启动时间，便于 outputs 追溯。
export RUN_ID="${RUN_ID:-qwen35_27b_all_nothink_8gpu_20260914_memqa_v5}"

# all 布局（lib 默认 ALL_TP=6 actors on GPU 0-5, ALL_VERIFY_GPU=6, ALL_EVAL_TP=64）。
# Qwen3.5-27B dense 权重 51.75GiB；单卡 gm402-80g 需 GPU_UTIL>=0.90（72GiB）才放得下权重+KV
# （CLAUDE.md 经验 + zero eval 筹备记忆同口径）。6 个 actor 各占一卡 + verify(GPU6) 跑 harness-1。
# KV 预算：72-51.75=20.25GiB headroom；16k ctx 单 seq KV≈4GiB。BC+ 实际上下文远小于 16k，
# vLLM 动态分配 KV，max-num-seqs=8 是按 GPT-OSS-20B zero eval 已验证值保守设；OOM 再降。
# max-model-len 由 wrapper 经 QWEN35_MAX_MODEL_LEN 降到 16384（profile 默认 32k 对 27B 必 OOM）。
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GPU_UTIL="${GPU_UTIL:-0.90}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-8}"
# max-model-len: leave unset to use the profile default 32768. An earlier 16384
# cap caused HTTP 400 on BC+ multi-turn queries (prompt 14337 + 2048 output).
# KV headroom ~20GiB handles 8 concurrent short queries (observed ~11% at 5 reqs);
# long outliers evict via paged KV instead of hard-failing. Set QWEN35_MAX_MODEL_LEN
# only to override downward deliberately.
export QWEN35_MAX_MODEL_LEN="${QWEN35_MAX_MODEL_LEN:-}"
export ALL_TP="${ALL_TP:-6}"
export ALL_EVAL_TP="${ALL_EVAL_TP:-64}"
export ALL_ACTOR_GPUS="${ALL_ACTOR_GPUS:-0,1,2,3,4,5}"
export ALL_VERIFY_GPU="${ALL_VERIFY_GPU:-6}"
export WORKER_STAGGER="${WORKER_STAGGER:-0}"
# 关闭 Qwen3.5 thinking（wrapper 据此追加 --chat-template-kwargs {"enable_thinking":false}）。
export QWEN35_NO_THINKING=1
export TRIM_GPU_KEEPALIVE=0
export HARNESS1_FORBID_CHROMA=1
export CURATE_NUDGE_POLICY=legacy
export MAX_TURNS="${MAX_TURNS:-40}"

echo "[env] PYTHONPATH=${PYTHONPATH}"
echo "[env] GPU_UTIL=${GPU_UTIL} VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS} QWEN35_MAX_MODEL_LEN=${QWEN35_MAX_MODEL_LEN} ALL_TP=${ALL_TP} ALL_EVAL_TP=${ALL_EVAL_TP}"
echo "[env] ALL_ACTOR_GPUS=${ALL_ACTOR_GPUS} ALL_VERIFY_GPU=${ALL_VERIFY_GPU}"
echo "[env] QWEN35_NO_THINKING=${QWEN35_NO_THINKING} (1=thinking off via vLLM serve chat-template-kwargs)"
echo "[env] TIKTOKEN_ENCODINGS_BASE=${TIKTOKEN_ENCODINGS_BASE}"
echo "[env] RUN_ID=${RUN_ID}"
echo "[env] model=${MODEL_PATH} verify=${VERIFY_MODEL_PATH}"

# ---- 9. 启动 eval ----
echo "=================================================================="
echo "[eval] bash ${EVAL_SCRIPT}"
echo "=================================================================="
cd "${TRIM_ROOT}"
exec bash "${EVAL_SCRIPT}"
