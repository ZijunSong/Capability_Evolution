#!/usr/bin/env bash
# CloudML imageCommand entry — Qwen3.5-9B harness self-evolution (run_harness_evolve.py).
# Fork of entry_qwen35_9b_harness_zero_bcplus_full_eval_8gpu_20261021.sh (the verified
# upstream_api base eval entry), but instead of running run_eval once, it:
#   1. starts 7 vLLM serve actors (Qwen3.5-9B, GPU 0-6, ports 8042-8054) + 1 verifier
#      serve (harness-1, GPU 7, port 8056) — the same all-mode stack the 27B base eval uses,
#      with --tool-call-parser qwen3_coder (strip --reasoning-parser qwen3 via the wrapper).
#   2. builds actor_urls + verify_url from the ports.
#   3. execs run_harness_evolve.py --evaluation-path upstream_api --api-base-url <actors>
#      --verify-base-url <verifier> --verify-model harness-1-verifier, which loops:
#      propose bank → score candidate (call run_eval --component all against the running
#      serve actors) → evolve. Frozen H0 weights; stops before TRIM.
#
# The H0 baseline (--zero-rollout-dir) is the upstream_api H0 we already completed
# (9B fa_recall=17.03%, 830q). run_harness_evolve.load_zero_baseline reads
# upstream_api/SUMMARY.json + PER_QUERY.jsonl (verified on dev box).
#
# Qwen3.5-9B: no-think chat_template (enable_thinking is not true guard, already edited),
# upstream_api + tool-call-parser (fixes legacy_local 0% — LLM() API can't parse tool calls,
# only vllm serve can). 9B dense 18GiB, GPU_UTIL 0.90, MAX_NUM_SEQS 32, MAX_MODEL_LEN 65536.
set -euo pipefail

JFS_ROOT="/mnt/songzijun"
TRIM_ROOT="${JFS_ROOT}/Capability_Evolution/TRIM"
EASYOPD_ROOT="${JFS_ROOT}/Capability_Evolution/SCAPE-EasyOPD"
MODEL_PATH="${JFS_ROOT}/models/Qwen3.5-9B"
VERIFY_MODEL_PATH="${JFS_ROOT}/models/harness-1"
BCPLUS_ROOT="${JFS_ROOT}/Capability_Evolution/SCOPE/external/BrowseComp-Plus"
BCPLUS_CORPUS="${BCPLUS_ROOT}/data/browsecomp_plus_corpus_full.jsonl"
BCPLUS_INDEX="${BCPLUS_ROOT}/indexes/bm25"
TIKTOKEN_ZIP="${JFS_ROOT}/tiktoken-bundle.zip"
TIKTOKEN_BUNDLE="/tmp/tiktoken-bundle"
PYPI_INDEX="https://pkgs.d.xiaomi.net/artifactory/api/pypi/pypi-virtual/simple"

: "${OUT:=/mnt/songzijun/Capability_Evolution/TRIM/outputs/harness_evolve_qwen35_9b_20261022_v5}"
: "${ZERO_ROLLOUT_DIR:=/mnt/songzijun/Capability_Evolution/TRIM/outputs/eval_h1_bcplus_full_Qwen3.5-9B_zero_gpu01234567_qwen35_9b_harness_zero_bcplus_full_8gpu_s1_20261021_memqa_v5}"
export OUT ZERO_ROLLOUT_DIR

echo "=================================================================="
echo "[boot] TRIM harness self-evolve / Qwen3.5-9B / upstream_api / ($(date -Iseconds 2>/dev/null || echo unknown))"
echo "[boot] host=$(hostname) OUT=${OUT} ZERO_ROLLOUT_DIR=${ZERO_ROLLOUT_DIR}"
echo "=================================================================="

export PYTHONUNBUFFERED=1

# ---- 0. cp self to /tmp (dodge JuiceFS heredoc reread) ----
SELF="${BASH_SOURCE[0]:-$0}"
if [[ "${SELF}" == /mnt/songzijun/* ]]; then
  TMP_SELF="/tmp/$(basename "${SELF}")"
  cp -f "${SELF}" "${TMP_SELF}"
  exec bash "${TMP_SELF}" "$@"
fi

# ---- 1. deps install (four-pack + harness import chain) ----
echo "[deps] installing locked runtime from ${PYPI_INDEX}"
python3 -m pip install --no-cache-dir --ignore-installed \
  --upgrade-strategy only-if-needed \
  --index-url "${PYPI_INDEX}" \
  "torch==2.10.0" "transformers==4.57.6" "accelerate==1.13.0" "datasets==4.8.4" \
  "hydra-core==1.3.2" "omegaconf==2.3.0" "vllm==0.19.1" peft structlog chromadb \
  "tqdm" "python-dotenv" "pandas" "pyarrow" "sentencepiece" "pyserini==2.0.0" \
  "openai" "tenacity" "anthropic" "pydantic-settings"
python3 -m pip install --no-cache-dir --no-deps --index-url "${PYPI_INDEX}" \
  "chz" "tinker" "json_repair" "openai_harmony" || echo "[deps] WARN: --no-deps extras issue"

# ---- 2. preflight: GPU / runtime / arch ----
echo "[preflight][gpu] nvidia-smi summary:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>&1 | head -10
python3 - <<'PY'
import os, importlib.metadata, torch, vllm, transformers
from jnius import autoclass
autoclass("java.lang.String")
print(f"[preflight][rt] torch={torch.__version__} cuda={torch.version.cuda} device_count={torch.cuda.device_count()}")
print(f"[preflight][rt] vllm={vllm.__version__} transformers={transformers.__version__} pyserini={importlib.metadata.version('pyserini')}")
assert torch.cuda.device_count() == 8, f"expected 8 GPUs, got {torch.cuda.device_count()}"
assert vllm.__version__ == "0.19.1"
assert transformers.__version__ == "4.57.6"
# arch probe
arch = "Qwen3_5ForConditionalGeneration"
from vllm.model_executor.models.registry import ModelRegistry
assert arch in set(ModelRegistry.get_supported_archs()), f"{arch} NOT registered in vLLM"
print(f"[preflight][arch] {arch} SUPPORTED")
PY

# ---- 2b. vLLM arch probe ----
echo "[preflight][arch] probing vLLM ModelRegistry for Qwen3_5ForConditionalGeneration"
python3 - <<'PY'
import sys
arch = "Qwen3_5ForConditionalGeneration"
try:
    from vllm.model_executor.models.registry import ModelRegistry
    supported = set(ModelRegistry.get_supported_archs())
    if arch in supported:
        print(f"[preflight][arch] SUPPORTED: {arch} is registered in vLLM ModelRegistry")
    else:
        qwen_archs = sorted(a for a in supported if "qwen" in a.lower() or "Qwen" in a)
        print(f"[preflight][arch] NOT SUPPORTED: {arch} is NOT registered")
        print(f"[preflight][arch] qwen-family registered archs: {qwen_archs[:20]}")
        sys.exit(5)
except SystemExit:
    raise
except Exception as e:
    print(f"[preflight][arch] ModelRegistry probe failed: {type(e).__name__}: {e}")
    sys.exit(5)
PY

# ---- 3. preflight: Java / PyJNI / Lucene ----
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

# ---- 3b. ANSERINI_CLASSPATH ----
export ANSERINI_CLASSPATH="$(python3 -c 'import pyserini,os;print(os.path.join(os.path.dirname(pyserini.__file__),"resources","jars"))')"
echo "[preflight][jvm] ANSERINI_CLASSPATH=${ANSERINI_CLASSPATH}"
test -d "${ANSERINI_CLASSPATH}" && echo "[preflight][jvm] anserini jar dir OK" \
  || { echo "[preflight][jvm] ANSERINI_CLASSPATH jar dir MISSING" >&2; exit 2; }

# ---- 4. preflight: offline creds + import chain ----
: "${OPENAI_API_KEY:=**-*****-*******}"
: "${CHROMA_API_KEY:=local-offline}"
: "${CHROMA_DATABASE:=local-offline}"
export OPENAI_API_KEY CHROMA_API_KEY CHROMA_DATABASE
export HARNESS1_FORBID_CHROMA=1
echo "[preflight][import] structlog/chromadb/pyserini/peft/openai_harmony + harness import chain"
export BCPLUS_INDEX_LUCENE="${BCPLUS_INDEX}"
PYTHONPATH="${TRIM_ROOT}:${EASYOPD_ROOT}" python3 - <<'PY'
import os
import structlog, chromadb, pyserini, peft
print(f"[preflight][import] structlog={structlog.__version__} peft={peft.__version__}")
import openai_harmony
print("[preflight][import] openai_harmony OK")
from trim.eval.offline_credentials import ensure_local_offline_credentials
ensure_local_offline_credentials()
from pyserini.search.lucene import LuceneSearcher
idx = os.environ.get("BCPLUS_INDEX_LUCENE")
searcher = LuceneSearcher(idx)
n = int(getattr(searcher, "num_docs", 0) or 0)
hits = searcher.search("test", k=1)
print(f"[preflight][lucene] index={idx} num_docs={n} probe_hits={len(hits)}")
assert n > 0
PY

# ---- 5. preflight: JuiceFS mount + key paths ----
echo "[preflight][fs] JuiceFS mount check:"
mount | grep -E "songzijun|JuiceFS" || echo "[preflight][fs] no songzijun mount line (may be CSI)"
FS_PATHS=("${JFS_ROOT}" "${TRIM_ROOT}" "${EASYOPD_ROOT}" "${MODEL_PATH}" \
         "${VERIFY_MODEL_PATH}" "${BCPLUS_CORPUS}" "${BCPLUS_INDEX}" \
         "${TRIM_ROOT}/scripts/run_harness_evolve.py" "${TRIM_ROOT}/scripts/run_eval.py" \
         "${ZERO_ROLLOUT_DIR}" "${TIKTOKEN_ZIP}")
for p in "${FS_PATHS[@]}"; do
  if [ -e "$p" ]; then
    echo "[preflight][fs] OK  $p"
  else
    echo "[preflight][fs] MISSING  $p" >&2
    exit 2
  fi
done
test -f "${MODEL_PATH}/config.json" && echo "[preflight][fs] model config.json OK"
test -f "${VERIFY_MODEL_PATH}/config.json" && echo "[preflight][fs] verify model config.json OK"
test -f "${ZERO_ROLLOUT_DIR}/upstream_api/SUMMARY.json" \
  && echo "[preflight][fs] H0 baseline upstream_api/SUMMARY.json OK" \
  || { echo "[preflight][fs] H0 baseline SUMMARY.json MISSING" >&2; exit 2; }

# ---- 6. tiktoken ----
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
test -f "${TIKTOKEN_BUNDLE}/share/tiktoken/o200k_base.tiktoken"
test -f "${TIKTOKEN_BUNDLE}/share/tiktoken/cl100k_base.tiktoken"
export TIKTOKEN_ENCODINGS_BASE="${TIKTOKEN_BUNDLE}/share/tiktoken"
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_BUNDLE}/tiktoken_cache"
export TIKTOKEN_RS_CACHE_DIR="${TIKTOKEN_BUNDLE}/tiktoken_rs_cache"

# ---- 7. start vLLM serve actors + verifier (all-mode stack) ----
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="${TRIM_ROOT}:${EASYOPD_ROOT}"
export TRIM_GPU_KEEPALIVE=0
export GPU_UTIL="${GPU_UTIL:-0.90}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}"
export QWEN35_NO_THINKING="${QWEN35_NO_THINKING:-1}"
export QWEN35_MAX_MODEL_LEN="${QWEN35_MAX_MODEL_LEN:-65536}"

# all-mode layout: 7 actor (GPU 0-6) + 1 verifier (GPU 7), ALL_EVAL_TP=64.
export ALL_TP="${ALL_TP:-7}"
export ALL_EVAL_TP="${ALL_EVAL_TP:-64}"
export ALL_ACTOR_GPUS="${ALL_ACTOR_GPUS:-0,1,2,3,4,5,6}"
export ALL_ACTOR_PORTS="${ALL_ACTOR_PORTS:-8042,8044,8046,8048,8050,8052,8054}"
export ALL_VERIFY_GPU="${ALL_VERIFY_GPU:-7}"
export ALL_VERIFY_PORT="${ALL_VERIFY_PORT:-8056}"

export MODEL_PATH BENCHMARK="${BENCHMARK:-bcplus_full}" RUN_ID="${RUN_ID:-harness_evolve_qwen35_9b_20261022_v5}"
export MODEL_SLUG="Qwen3.5-9B" API_MODEL="Qwen3.5-9B"
export VERIFY_MODEL_PATH VERIFY_MODEL_NAME="${VERIFY_MODEL_NAME:-harness-1-verifier}"
# lib_bcplus_gpu8_init validates COMPONENT (:? required). start_all_stack is all-mode (7 actor + verifier).
export COMPONENT="all"

echo "[env] MODEL_PATH=${MODEL_PATH} VERIFY_MODEL_PATH=${VERIFY_MODEL_PATH}"
echo "[env] ALL_ACTOR_GPUS=${ALL_ACTOR_GPUS} ALL_ACTOR_PORTS=${ALL_ACTOR_PORTS} ALL_VERIFY_GPU=${ALL_VERIFY_GPU} ALL_VERIFY_PORT=${ALL_VERIFY_PORT}"
echo "[env] GPU_UTIL=${GPU_UTIL} VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS} QWEN35_NO_THINKING=${QWEN35_NO_THINKING} QWEN35_MAX_MODEL_LEN=${QWEN35_MAX_MODEL_LEN}"

# Source the lib (gives start_all_stack, vllm_extra_for_model, start_vllm_bg, etc.)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
if [[ ! -f "${SCRIPT_DIR}/lib_bcplus_w_harness1_gpu8_eval.sh" ]]; then
  SCRIPT_DIR="${TRIM_ROOT}/scripts"
fi
# shellcheck source=lib_bcplus_w_harness1_gpu8_eval.sh
source "${SCRIPT_DIR}/lib_bcplus_w_harness1_gpu8_eval.sh"

# lib init sets REL_OUT/REL_LOGS/MASTER_LOG + validates MODEL_PATH/MODEL_SLUG/API_MODEL/COMPONENT/RUN_ID.
# Must be called explicitly (it's a function, not auto-run on source). Required before start_all_stack
# (which calls log() → uses MASTER_LOG). Without this, "MASTER_LOG: unbound variable".
export SCRIPT_DIR
lib_bcplus_gpu8_init

# Override vllm_extra_for_model to strip --reasoning-parser qwen3 (no-think, keep tool-call-parser).
_override_vllm_extra_for_model() {
  local slug="${1:-}"
  local trim_root="${TRIM_ROOT}"
  local base extra
  base="$(PYTHONPATH="${trim_root}:${PYTHONPATH:-}" "${PY:-python3}" \
    -m trim.eval.model_profiles vllm-extra "${slug}")"
  extra="${base}"
  if [[ -n "${QWEN35_MAX_MODEL_LEN:-}" ]]; then
    extra="$(printf '%s\n' "${extra}" | sed -E "s#--max-model-len [0-9]+#--max-model-len ${QWEN35_MAX_MODEL_LEN}#")"
  fi
  if [[ "${QWEN35_NO_THINKING}" == "1" ]]; then
    extra="${extra// --reasoning-parser qwen3/}"
  fi
  echo "${extra}"
}
vllm_extra_for_model() { _override_vllm_extra_for_model "$@"; }

echo "=================================================================="
echo "[harness-evolve] starting all-mode serve stack (7 actors + 1 verifier)"
echo "=================================================================="
assert_local_retrieval_ready
ensure_full_corpus
start_all_stack
echo "[harness-evolve] all-mode serve stack ready (actors + verifier up)"

# Build actor_urls + verify_url from ports.
_actor_urls=""
IFS=',' read -ra _ports <<< "${ALL_ACTOR_PORTS}"
for p in "${_ports[@]}"; do
  if [ -z "${_actor_urls}" ]; then
    _actor_urls="http://127.0.0.1:${p}/v1"
  else
    _actor_urls="${_actor_urls},http://127.0.0.1:${p}/v1"
  fi
done
_verify_url="http://127.0.0.1:${ALL_VERIFY_PORT}/v1"
echo "[harness-evolve] actor_urls=${_actor_urls}"
echo "[harness-evolve] verify_url=${_verify_url} verify_model=${VERIFY_MODEL_NAME}"

# ---- 8. exec run_harness_evolve.py ----
echo "=================================================================="
echo "[harness-evolve] exec run_harness_evolve.py (frozen H0, upstream_api, stop before TRIM)"
echo "=================================================================="
cd "${TRIM_ROOT}"
exec python3 "${TRIM_ROOT}/scripts/run_harness_evolve.py" \
  --nnodes 1 \
  --rollout-replicas 8 \
  --evaluation-path upstream_api \
  --api-base-url "${_actor_urls}" \
  --api-model "${API_MODEL}" \
  --verify-base-url "${_verify_url}" \
  --verify-model "${VERIFY_MODEL_NAME}" \
  --model_name "${MODEL_PATH}" \
  --zero-rollout-dir "${ZERO_ROLLOUT_DIR}" \
  --out "${OUT}"
