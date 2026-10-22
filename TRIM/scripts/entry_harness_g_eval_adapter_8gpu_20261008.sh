#!/usr/bin/env bash
# CloudML 启动脚本：TRIM Harness-G eval（bcplus_full/longsealqa/frames/hotpotqa）
# 单节点 8x gm402-80g。
#
# === 2026-10-08 增强版（fork entry_harness_g_zero_all_bcplus_transfer_eval_8gpu.sh）===
# 在原版基础上增加：
#   1. MODEL_KIND=qwen3_4b_instruct（Qwen3-4B-Instruct-2507，dense bf16）支持
#   2. ADAPTER_PATH env：step50/100/150/200 训练 checkpoint 的 adapter，传 --adapter 给 run_eval.py
#      不设 ADAPTER_PATH = bare（step0，无 adapter，裸模型 baseline）
#   3. MODEL_PATH 自动指向 /tmp 本地拷贝（与训练 entry 一致，避 8-shard 并发 JuiceFS 读）
#      但 gpt-oss eval 仍用 bf16-dequant（CLAUDE.md 铁律：eval 必须用 bf16-dequant 避 mxfp4+eval PTX）
#
# COMPONENT 由 EVAL_MODE env 选 zero|all（Harness-G all = 8 graph-menu components，自包含，无 verifier）。
# 4 benchmark 共享此 entry：BENCHMARK 由 submit --env 注入。
#
# Harness-G 走 legacy_local graph runtime（非 Harness-1 upstream_api actor 模式）：
#   run_eval.py --harness Harness-G --tp 8 → 8 副本各 TP=1 数据并行分片 query
#   4 benchmark is_formal_harness_g_eval=False → graph_required=False → 不需 --graph-index-path（eval 侧）
#   注意：eval 不需 corpus graph（graph_required=False），但训练需要（训练 entry 单独校验）。
set -euo pipefail

JFS_ROOT="/mnt/songzijun"
TRIM_ROOT="${JFS_ROOT}/Capability_Evolution/TRIM"
EASYOPD_ROOT="${JFS_ROOT}/Capability_Evolution/SCAPE-EasyOPD"
HARNESS1_EXTERNAL="${TRIM_ROOT}/external/harness-1"

# 模型选择（由 MODEL_KIND env 决定）
: "${MODEL_KIND:=qwen3_4b_instruct}"  # qwen3_4b_instruct | qwen3_4b_thinking | gpt_oss_20b
case "${MODEL_KIND}" in
  qwen3_4b_instruct)
    MODEL_SRC="${JFS_ROOT}/models/Qwen3-4B-Instruct-2507"
    MODEL_SLUG="Qwen3-4B-Instruct-2507"
    ;;
  qwen3_4b_thinking)
    MODEL_SRC="${JFS_ROOT}/models/Qwen3-4B-Thinking-2507"
    MODEL_SLUG="Qwen3-4B-Thinking-2507"
    ;;
  gpt_oss_20b)
    # bf16-dequant（CLAUDE.md 铁律：eval 必须用 bf16-dequant 避 mxfp4+eval PTX；原版 mxfp4 禁用）
    MODEL_SRC="${JFS_ROOT}/models/openai/gpt-oss-20b-bf16-dequant"
    MODEL_SLUG="gpt-oss-20b"
    ;;
  *)
    echo "[boot] ERR unknown MODEL_KIND=${MODEL_KIND} (expect qwen3_4b_instruct|qwen3_4b_thinking|gpt_oss_20b)" >&2
    exit 2
    ;;
esac

TIKTOKEN_ZIP="${JFS_ROOT}/tiktoken-bundle.zip"
TIKTOKEN_BUNDLE="/tmp/tiktoken-bundle"
PYPI_INDEX="https://pkgs.d.xiaomi.net/artifactory/api/pypi/pypi-virtual/simple"

: "${EVAL_MODE:=zero}"          # zero | all
: "${BENCHMARK:=bcplus_full}"   # bcplus_full|longsealqa|frames|hotpotqa
: "${ADAPTER_PATH:=}"           # 空=bare（step0）；非空=step_N/adapter
if [ -n "${ADAPTER_PATH}" ]; then
  ADAPT_TAG="step-adapter"
else
  ADAPT_TAG="bare"
fi
: "${RUN_ID:=${MODEL_KIND}_harness_g_${EVAL_MODE}_${BENCHMARK}_${ADAPT_TAG}_8gpu_$(date +%Y%m%d_%H%M%S 2>/dev/null || echo unknown)_memqa_v5}"

echo "=================================================================="
echo "[boot] TRIM Harness-G / ${MODEL_SLUG} / ${ADAPT_TAG} / HARNESS-G ${EVAL_MODE} eval / 8gpu"
echo "[boot] src=${MODEL_SRC} adapter=${ADAPTER_PATH:-<none>} BENCHMARK=${BENCHMARK} RUN_ID=${RUN_ID}"
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
python3 -m pip install --no-cache-dir --no-deps --index-url "${PYPI_INDEX}" "openai_harmony" \
  || echo "[deps] WARN: openai_harmony --no-deps install issue"

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
for p in "${JFS_ROOT}" "${TRIM_ROOT}" "${EASYOPD_ROOT}" "${HARNESS1_EXTERNAL}" "${MODEL_SRC}" \
         "${TRIM_ROOT}/scripts/run_eval.py"; do
  [ -e "$p" ] || { echo "[preflight][fs] MISSING  $p" >&2; exit 2; }
done
test -f "${MODEL_SRC}/config.json"

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

# Harness-G: 不需 HARNESS1_FORBID_CHROMA（harness-1 upstream_api 专用）；设 SCAPE_LOCAL_OPENAI_EMBEDDINGS=1
: "${OPENAI_API_KEY:=**-*****-*******}"
export OPENAI_API_KEY CHROMA_API_KEY=local-offline CHROMA_DATABASE=local-offline
export SCAPE_LOCAL_OPENAI_EMBEDDINGS=1
export PYTHONPATH="${TRIM_ROOT}:${EASYOPD_ROOT}:${HARNESS1_EXTERNAL}"
export BENCHMARK RUN_ID MODEL_KIND MODEL_SRC MODEL_SLUG EVAL_MODE ADAPTER_PATH

# CloudML pytorch 注入 RANK/WORLD_SIZE=1 干扰（eval 不走 torchrun），清掉
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK NODE_RANK MASTER_ADDR MASTER_PORT PET_NPROC_PER_NODE 2>/dev/null || true

# --- Copy model to /tmp (avoid 8-shard concurrent JuiceFS read) ---
MODEL_LOCAL="/tmp/$(basename "${MODEL_SRC}")"
if [ ! -f "${MODEL_LOCAL}/config.json" ]; then
  rm -rf "${MODEL_LOCAL}"
  mkdir -p "${MODEL_LOCAL}"
  cp -r "${MODEL_SRC}/." "${MODEL_LOCAL}/"
fi
test -f "${MODEL_LOCAL}/config.json"
echo "[preflight][model] MODEL_LOCAL=${MODEL_LOCAL} ready"

# --- Adapter preflight (if ADAPTER_PATH set) ---
ADAPTER_ARG=""
if [ -n "${ADAPTER_PATH}" ]; then
  test -d "${ADAPTER_PATH}" || { echo "[preflight][adapter] ERR ADAPTER_PATH=${ADAPTER_PATH} not a dir" >&2; exit 2; }
  test -f "${ADAPTER_PATH}/adapter_config.json" || { echo "[preflight][adapter] ERR no adapter_config.json in ${ADAPTER_PATH}" >&2; exit 2; }
  test -f "${ADAPTER_PATH}/adapter_model.safetensors" || { echo "[preflight][adapter] ERR no adapter_model.safetensors in ${ADAPTER_PATH}" >&2; exit 2; }
  ADAPTER_ARG="--adapter ${ADAPTER_PATH}"
  python3 - <<PY
import json
cfg = json.load(open("${ADAPTER_PATH}/adapter_config.json"))
base = cfg.get("base_model_name_or_path","")
r = cfg.get("r")
alpha = cfg.get("lora_alpha")
print(f"[preflight][adapter] base={base} r={r} alpha={alpha}")
PY
fi

echo "[env] MODEL_KIND=${MODEL_KIND} MODEL_PATH=${MODEL_LOCAL} BENCHMARK=${BENCHMARK} EVAL_MODE=${EVAL_MODE} ADAPTER_PATH=${ADAPTER_PATH:-<none>} RUN_ID=${RUN_ID}"

# import 链 smoke 预检
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
echo "[eval] run_eval.py --harness Harness-G --benchmark ${BENCHMARK} --model_name ${MODEL_LOCAL} --component ${EVAL_MODE} --tp 8 ${ADAPTER_ARG} --out ${OUT_DIR}"

exec python3 scripts/run_eval.py \
  --harness Harness-G \
  --benchmark "${BENCHMARK}" \
  --model_name "${MODEL_LOCAL}" \
  --component "${EVAL_MODE}" \
  --tp 8 \
  --max-turns 40 \
  --max-new-tokens 2048 \
  --temperature 1.0 \
  ${ADAPTER_ARG} \
  --out "${OUT_DIR}"
