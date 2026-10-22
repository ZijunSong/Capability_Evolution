#!/usr/bin/env bash
# CloudML 启动脚本：TRIM pat-jj/harness-1 被测模型 / HARNESS eval（zero | all）
# 单节点 8x gm402-80g。被测 model = pat-jj_harness-1-full/harness-1（与旧 2026-09-14 eval 同口径，
# 旧表 all=31.38% / zero=27.11% on bcplus_full）。无 adapter，base model baseline。
# COMPONENT 由 EVAL_MODE env 选 zero|all（all 启完整 harness-1 组件 + verifier）。
# 4 benchmark 共享此 entry：BENCHMARK 由 submit --env 注入（bcplus_full/longsealqa/frames/hotpotqa）。
# Fork of entry_gpt_oss_20b_harness_zero_all_bcplus_transfer_eval_8gpu.sh，只改 MODEL_PATH/SLUG/API。
set -euo pipefail

JFS_ROOT="/mnt/songzijun"
TRIM_ROOT="${JFS_ROOT}/Capability_Evolution/TRIM"
EASYOPD_ROOT="${JFS_ROOT}/Capability_Evolution/SCAPE-EasyOPD"
HARNESS1_EXTERNAL="${TRIM_ROOT}/external/harness-1"
# 被测 model: pat-jj/harness-1（与旧 2026-09-14 eval 同口径）
MODEL_PATH="${JFS_ROOT}/models/pat-jj_harness-1-full/harness-1"
TIKTOKEN_ZIP="${JFS_ROOT}/tiktoken-bundle.zip"
TIKTOKEN_BUNDLE="/tmp/tiktoken-bundle"
PYPI_INDEX="https://pkgs.d.xiaomi.net/artifactory/api/pypi/pypi-virtual/simple"

: "${EVAL_MODE:=zero}"          # zero | all
: "${BENCHMARK:=bcplus_full}"   # bcplus_full|longsealqa|frames|hotpotqa
: "${RUN_ID:=gpt_oss_20b_harness_${EVAL_MODE}_${BENCHMARK}_$(date +%Y%m%d_%H%M%S 2>/dev/null || echo unknown)_memqa_v5}"

echo "=================================================================="
echo "[boot] TRIM Harness-1 / GPT-OSS-20B (no adapter) / HARNESS ${EVAL_MODE} eval / 8gpu"
echo "[boot] base=${MODEL_PATH} (pat-jj/harness-1) BENCHMARK=${BENCHMARK} RUN_ID=${RUN_ID}"
echo "=================================================================="

export PYTHONUNBUFFERED=1

echo "[deps] installing locked runtime from ${PYPI_INDEX}"
python3 -m pip install --no-cache-dir --ignore-installed --upgrade-strategy only-if-needed \
  --index-url "${PYPI_INDEX}" \
  "torch==2.10.0" "transformers==4.57.6" "accelerate" "datasets" "vllm==0.19.1" \
  "hydra-core" "omegaconf" "tensordict" "codetiming" "tqdm" "wandb" "tensorboard" \
  "math-verify" "pylatexenc" "python-dotenv" "pandas" "pyarrow" "sentencepiece" \
  "latex2sympy2" "word2number" "pyserini==2.0.0" "structlog" "chromadb" "peft" \
  "openai" "tenacity" "pydantic-settings"

python3 -m pip install --no-cache-dir --no-deps --index-url "${PYPI_INDEX}" \
  "chz" "tinker" "json_repair" "openai_harmony" || echo "[deps] WARN: --no-deps extras install issues"

echo "[deps] resolved:"
python3 - <<'PY'
import importlib.metadata, torch, vllm, transformers
print(f"[deps] torch={torch.__version__} cuda={torch.version.cuda} vllm={vllm.__version__} transformers={transformers.__version__} pyserini={importlib.metadata.version('pyserini')}")
PY

echo "[preflight][gpu]"
nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>&1 | head -10
python3 - <<'PY'
import os, importlib.metadata, torch, vllm, transformers
assert torch.cuda.device_count() == 8
pv = importlib.metadata.version("pyserini"); assert pv == "2.0.0"
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
for p in "${JFS_ROOT}" "${TRIM_ROOT}" "${EASYOPD_ROOT}" "${MODEL_PATH}" "${HARNESS1_EXTERNAL}" \
         "${TRIM_ROOT}/scripts/run_eval.py"; do
  [ -e "$p" ] || { echo "[preflight][fs] MISSING  $p" >&2; exit 2; }
done
test -f "${MODEL_PATH}/config.json"
# all 模式需 harness-1 verifier
if [ "${EVAL_MODE}" = "all" ]; then
  test -f "${JFS_ROOT}/models/harness-1/config.json" || { echo "MISSING harness-1 verifier" >&2; exit 2; }
fi

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

: "${OPENAI_API_KEY:=**-*****-*******}"
export OPENAI_API_KEY CHROMA_API_KEY=local-offline CHROMA_DATABASE=local-offline HARNESS1_FORBID_CHROMA=1
export PYTHONPATH="${TRIM_ROOT}:${EASYOPD_ROOT}:${HARNESS1_EXTERNAL}"
export MODEL_PATH BENCHMARK RUN_ID
# pat-jj/harness-1 被测：MODEL_SLUG/API_MODEL = harness-1（与旧 2026-09-14 eval 同）
export MODEL_SLUG="harness-1" API_MODEL="harness-1"
# FIX: zero 单 actor (GPU7) + 64 worker 并发太高致 query 丢失 (merged eval missing queries).
# 改 multi-8-actor layout（8 actor 各 4 worker = 32 worker），分散并发，避免 query 丢/超时。
# all 模式已是 multi（ALL_ACTOR_GPUS=0..5 + verifier），不动。
if [ "${EVAL_MODE}" = "zero" ]; then
  export ZERO_ACTOR_GPUS="0,1,2,3,4,5,6,7"
  export ZERO_ACTOR_PORTS="8040,8042,8044,8046,8048,8050,8052,8054"
  export TP="${TP:-32}"   # 8 actor x 4 worker = 32 (was 64 on single actor)
  export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}"
fi
# FIX: all 默认 ALL_EVAL_TP=64 打 6 actor 并发过高致 query 丢失 (merged eval missing).
# 降到 24 (6 actor x 4 worker)，verifier 仍在 GPU6。
if [ "${EVAL_MODE}" = "all" ]; then
  export ALL_EVAL_TP="${ALL_EVAL_TP:-24}"
  export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}"
fi
# CloudML pytorch 注入 RANK/WORLD_SIZE=1 干扰，清掉（eval 不走 torchrun）
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK NODE_RANK MASTER_ADDR MASTER_PORT PET_NPROC_PER_NODE 2>/dev/null || true

echo "[env] MODEL_PATH=${MODEL_PATH} BENCHMARK=${BENCHMARK} EVAL_MODE=${EVAL_MODE} RUN_ID=${RUN_ID}"

cd "${TRIM_ROOT}"
if [ "${EVAL_MODE}" = "all" ]; then
  echo "[eval] bash scripts/run_bcplus_w_harness1_patjj_harness1_all_gpu8_eval.sh"
  exec bash scripts/run_bcplus_w_harness1_patjj_harness1_all_gpu8_eval.sh
else
  echo "[eval] bash scripts/run_bcplus_w_harness1_patjj_harness1_zero_gpu8_eval.sh"
  exec bash scripts/run_bcplus_w_harness1_patjj_harness1_zero_gpu8_eval.sh
fi
