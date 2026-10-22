#!/usr/bin/env bash
# CloudML 启动脚本：TRIM Harness-G / (Qwen3-4B-Thinking-2507 | GPT-OSS-20B bf16-dequant) / HARNESS-G eval
# 单节点 8x gm402-80g。基座无 adapter，原始模型 baseline。
# COMPONENT 由 EVAL_MODE env 选 zero|all（Harness-G all = 8 graph-menu components，自包含，无 verifier）。
# 4 benchmark 共享此 entry：BENCHMARK 由 submit --env 注入（bcplus_full/longsealqa/frames/hotpotqa）。
# MODEL_KIND 由 submit --env 注入（qwen3_4b_thinking|gpt_oss_20b）选 MODEL_PATH。
#
# Harness-G 走 legacy_local graph runtime（非 Harness-1 upstream_api actor 模式）：
#   run_eval.py --harness Harness-G --tp 8 → 8 副本各 TP=1 数据并行分片 query
#   （不会撞 Harness-1 zero 单 actor 丢 query 坑；merge_traces loudly 报缺失）
#   --benchmark 自动路由 index/corpus（open_eval_retrieval），不需 lib transfer patch
#   --offline 自动强制；Harness-G all 不需 verifier 模型
#   4 benchmark is_formal_harness_g_eval=False → graph_required=False → 不需 --graph-index-path
#
# 最小依赖（不装 chromadb/chz/tinker，Harness-G legacy_local 不需要）：
#   structlog 必需（harness.ultra_core/tools 顶层硬 import）
#   tinker 不需（config.py/agent.py try/except 保护）
#   chromadb 不需（tools.py 函数体内 + SCAPE_LOCAL_OPENAI_EMBEDDINGS=1 跳过）
#   openai_harmony 必需（GPT-OSS Harmony encoding；Qwen3 走 HF chat template 不触发但装了无害）
set -euo pipefail

JFS_ROOT="/mnt/songzijun"
TRIM_ROOT="${JFS_ROOT}/Capability_Evolution/TRIM"
EASYOPD_ROOT="${JFS_ROOT}/Capability_Evolution/SCAPE-EasyOPD"
HARNESS1_EXTERNAL="${TRIM_ROOT}/external/harness-1"

# 模型选择（由 MODEL_KIND env 决定）
: "${MODEL_KIND:=qwen3_4b_thinking}"  # qwen3_4b_thinking | gpt_oss_20b
case "${MODEL_KIND}" in
  qwen3_4b_thinking)
    MODEL_PATH="${JFS_ROOT}/models/Qwen3-4B-Thinking-2507"
    MODEL_SLUG="Qwen3-4B-Thinking-2507"
    ;;
  gpt_oss_20b)
    # bf16-dequant（CLAUDE.md 铁律：eval 必须用 bf16-dequant 避 mxfp4+eval PTX；原版 mxfp4 禁用）
    MODEL_PATH="${JFS_ROOT}/models/openai/gpt-oss-20b-bf16-dequant"
    MODEL_SLUG="gpt-oss-20b"
    ;;
  *)
    echo "[boot] ERR unknown MODEL_KIND=${MODEL_KIND} (expect qwen3_4b_thinking|gpt_oss_20b)" >&2
    exit 2
    ;;
esac

TIKTOKEN_ZIP="${JFS_ROOT}/tiktoken-bundle.zip"
TIKTOKEN_BUNDLE="/tmp/tiktoken-bundle"
PYPI_INDEX="https://pkgs.d.xiaomi.net/artifactory/api/pypi/pypi-virtual/simple"

: "${EVAL_MODE:=zero}"          # zero | all
: "${BENCHMARK:=bcplus_full}"   # bcplus_full|longsealqa|frames|hotpotqa
: "${RUN_ID:=${MODEL_KIND}_harness_g_${EVAL_MODE}_${BENCHMARK}_8gpu_$(date +%Y%m%d_%H%M%S 2>/dev/null || echo unknown)_memqa_v5}"

echo "=================================================================="
echo "[boot] TRIM Harness-G / ${MODEL_SLUG} (no adapter) / HARNESS-G ${EVAL_MODE} eval / 8gpu"
echo "[boot] base=${MODEL_PATH} BENCHMARK=${BENCHMARK} RUN_ID=${RUN_ID}"
echo "[boot] layout: --tp 8 = 8 副本各 TP=1 数据并行（Harness-G legacy_local graph runtime）"
echo "=================================================================="

export PYTHONUNBUFFERED=1

echo "[deps] installing minimal Harness-G runtime from ${PYPI_INDEX}"
# 最小依赖：不装 chromadb/chz/tinker（Harness-G legacy_local 不需要）
# structlog 必需（harness 包顶层硬 import）；openai_harmony 必需（GPT-OSS Harmony encoding）
python3 -m pip install --no-cache-dir --ignore-installed --upgrade-strategy only-if-needed \
  --index-url "${PYPI_INDEX}" \
  "torch==2.10.0" "transformers==4.57.6" "accelerate" "datasets" "vllm==0.19.1" \
  "hydra-core" "omegaconf" "tensordict" "codetiming" "tqdm" "sentencepiece" \
  "math-verify" "pylatexenc" "python-dotenv" "pandas" "pyarrow" \
  "latex2sympy2" "word2number" "pyserini==2.0.0" "pyjnius" "structlog" "peft" \
  "openai" "tenacity" "pydantic-settings"

python3 -m pip install --no-cache-dir --no-deps --index-url "${PYPI_INDEX}" \
  "openai_harmony" || echo "[deps] WARN: openai_harmony --no-deps install issue"

echo "[deps] resolved:"
python3 - <<'PY'
import importlib.metadata, torch, vllm, transformers
print(f"[deps] torch={torch.__version__} cuda={torch.version.cuda} vllm={vllm.__version__} transformers={transformers.__version__} pyserini={importlib.metadata.version('pyserini')} structlog={importlib.metadata.version('structlog')}")
PY

echo "[preflight][gpu]"
nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>&1 | head -10
python3 - <<'PY'
import importlib.metadata, torch
assert torch.cuda.device_count() == 8, f"expected 8 GPUs, got {torch.cuda.device_count()}"
pv = importlib.metadata.version("pyserini"); assert pv == "2.0.0", f"pyserini={pv}"
print(f"[preflight][rt] 8 GPUs OK, pyserini={pv}")
PY

echo "[preflight][jvm] locating libjvm.so"
JVM_PATH="$(find /usr /opt -type f -name libjvm.so 2>/dev/null | head -n 1 || true)"
if [ -z "${JVM_PATH}" ]; then
  apt-get update; apt-get install -y --no-install-recommends openjdk-21-jre-headless
  JVM_PATH="$(find /usr /opt -type f -name libjvm.so 2>/dev/null | head -n 1)"
fi
test -n "${JVM_PATH}" && test -f "${JVM_PATH}"; ldd "${JVM_PATH}" >/dev/null
export JVM_PATH JAVA_HOME="$(dirname "$(dirname "$(dirname "${JVM_PATH}")")")"
export LD_LIBRARY_PATH="$(dirname "${JVM_PATH}"):${LD_LIBRARY_PATH:-}"
python3 -c 'from jnius import autoclass; print("[preflight][jvm] java_ok", autoclass("java.lang.String"))'

PYSERINI_ROOT=$(python3 -c 'import pathlib, pyserini; print(pathlib.Path(pyserini.__file__).resolve().parent)')
export ANSERINI_CLASSPATH="$(find "${PYSERINI_ROOT}/resources/jars" -type f -name '*.jar' -print | paste -sd: -)"
export CLASSPATH="${ANSERINI_CLASSPATH}"

# fs check
for p in "${JFS_ROOT}" "${TRIM_ROOT}" "${EASYOPD_ROOT}" "${HARNESS1_EXTERNAL}" "${MODEL_PATH}" \
         "${TRIM_ROOT}/scripts/run_eval.py"; do
  [ -e "$p" ] || { echo "[preflight][fs] MISSING  $p" >&2; exit 2; }
done
test -f "${MODEL_PATH}/config.json"

if [ ! -d "${TIKTOKEN_BUNDLE}/share/tiktoken" ]; then
  python3 - <<PY
import zipfile, os, shutil
src="${TIKTOKEN_ZIP}"; dst="${TIKTOKEN_BUNDLE}"
if os.path.exists(dst): shutil.rmtree(dst, ignore_errors=True)
with zipfile.ZipFile(src) as z: z.extractall("/tmp")
PY
fi
test -f "${TIKTOKEN_BUNDLE}/share/tiktoken/o200k_base.tiktoken"
export TIKTOKEN_ENCODINGS_BASE="${TIKTOKEN_BUNDLE}/share/tiktoken"
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_BUNDLE}/tiktoken_cache"
export TIKTOKEN_RS_CACHE_DIR="${TIKTOKEN_BUNDLE}/tiktoken_rs_cache"

# Harness-G: 不需 HARNESS1_FORBID_CHROMA（那是 harness-1 upstream_api 专用）；设 SCAPE_LOCAL_OPENAI_EMBEDDINGS=1 跳过 chromadb
: "${OPENAI_API_KEY:=**-*****-*******}"
export OPENAI_API_KEY CHROMA_API_KEY=local-offline CHROMA_DATABASE=local-offline
export SCAPE_LOCAL_OPENAI_EMBEDDINGS=1
export PYTHONPATH="${TRIM_ROOT}:${EASYOPD_ROOT}:${HARNESS1_EXTERNAL}"
export MODEL_PATH BENCHMARK RUN_ID MODEL_SLUG

# CloudML pytorch 注入 RANK/WORLD_SIZE=1 干扰（eval 不走 torchrun），清掉
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK NODE_RANK MASTER_ADDR MASTER_PORT PET_NPROC_PER_NODE 2>/dev/null || true

echo "[env] MODEL_KIND=${MODEL_KIND} MODEL_PATH=${MODEL_PATH} BENCHMARK=${BENCHMARK} EVAL_MODE=${EVAL_MODE} RUN_ID=${RUN_ID}"

# import 链 smoke 预检（第一次跑 Harness-G 的关键风险点）
# 若失败 → fallback 装 chromadb/chz/tinker 全依赖后重试 import
echo "[preflight][import] Harness-G import chain smoke"
if ! python3 - <<'PY'
import sys
sys.path.insert(0, "/mnt/songzijun/Capability_Evolution/TRIM")
sys.path.insert(0, "/mnt/songzijun/Capability_Evolution/TRIM/external/harness-1")
import structlog
from trim.eval.harness_g_contract import is_formal_harness_g_eval
from trim.eval.harness_g_graph import load_graph_index
from trim.eval.harness_g_runtime import build_prompt_ids  # noqa: F401
import harness.ultra_core  # noqa: F401
import harness.tools      # noqa: F401
print("[preflight][import] IMPORT_OK")
PY
then
  echo "[preflight][import] WARN: minimal deps import failed → fallback installing chromadb/chz/tinker"
  python3 -m pip install --no-cache-dir --ignore-installed --index-url "${PYPI_INDEX}" \
    "chromadb" "chz" "tinker" "json_repair" || echo "[deps] WARN: fallback install partial"
  python3 - <<'PY'
import sys
sys.path.insert(0, "/mnt/songzijun/Capability_Evolution/TRIM")
sys.path.insert(0, "/mnt/songzijun/Capability_Evolution/TRIM/external/harness-1")
import structlog
from trim.eval.harness_g_contract import is_formal_harness_g_eval
from trim.eval.harness_g_graph import load_graph_index
import harness.ultra_core, harness.tools
print("[preflight][import] IMPORT_OK (fallback deps)")
PY
fi

cd "${TRIM_ROOT}"
OUT_DIR="${TRIM_ROOT}/outputs/eval_hg_${BENCHMARK}_${MODEL_SLUG}_${EVAL_MODE}_${RUN_ID}"
echo "[eval] run_eval.py --harness Harness-G --benchmark ${BENCHMARK} --model_name ${MODEL_PATH} --component ${EVAL_MODE} --tp 8 --out ${OUT_DIR}"

exec python3 scripts/run_eval.py \
  --harness Harness-G \
  --benchmark "${BENCHMARK}" \
  --model_name "${MODEL_PATH}" \
  --component "${EVAL_MODE}" \
  --tp 8 \
  --max-turns 40 \
  --max-new-tokens 2048 \
  --temperature 1.0 \
  --out "${OUT_DIR}"
