#!/usr/bin/env bash
# CloudML 启动脚本：TRIM Harness-1 BC+ / GPT-OSS-20B / ADAPTER eval
# (训练后 step100 LoRA, 7nzzn t-20260925093542-7nzzn RL 200step 4node, step100 落盘 9/28 18:38)
# 单节点 8x gm402-80g。基座 gpt-oss-20b-bf16-dequant + 训练后 LoRA adapter (step100, r=8, alpha=16)。
# 8 个单卡 vLLM actor (TP=1) 各加载 base+LoRA (enable_lora + lora-modules)，--tp 32 API-client workers。
# local_bm25 (Pyserini/Lucene)，无 Chroma，COMPONENT=zero（对齐 student 口径）。
#
# === 基座选 bf16-dequant（用户决策，安全）===
# 7nzzn 训练用 mxfp4 原版 (base=/tmp/gpt-oss-20b)，但 eval 用 mxfp4+enable_lora 有 PTX 崩溃风险
# (CLAUDE.md vllm-mxfp4-lora-marlin-ptx-gm402：mxfp4+LoRA Marlin backend 硬编码 PTX vs gm402 sm_90)。
# 改用预 dequant bf16 checkpoint /mnt/songzijun/models/openai/gpt-oss-20b-bf16-dequant (config 无
# quantization_config, 9 safetensors ~40GB)，走标准 bf16+LoRA path，无 PTX。adapter base 虽记录
# /tmp/gpt-oss-20b 但架构一致 GptOssForCausalLM，LoRA delta 与 base dtype 独立，兼容。
#
# === 4 benchmark 共享此 entry，BENCHMARK 由 submit --env 注入 ===
#   bcplus_full (BrowseComp+) / longsealqa / frames / hotpotqa，相同 setting，仅 BENCHMARK + RUN_ID 不同。
#   wrapper run_bcplus_w_harness1_adapter_gpu8_eval.sh 内置 transfer benchmark 路由。
set -euo pipefail

JFS_ROOT="/mnt/songzijun"
TRIM_ROOT="${JFS_ROOT}/Capability_Evolution/TRIM"
EASYOPD_ROOT="${JFS_ROOT}/Capability_Evolution/SCAPE-EasyOPD"
# 基座用 bf16-dequant (安全, 无 mxfp4+LoRA PTX 风险)
MODEL_PATH="${JFS_ROOT}/models/openai/gpt-oss-20b-bf16-dequant"
# 训练产物 step100 adapter (7nzzn RL 200step 4node, step100 落盘 9/28 18:38)
ADAPTER_PATH="${ADAPTER_PATH:-${TRIM_ROOT}/outputs/cloudml_custom_train_harness1_bcplus_full_gpt_oss_20b_rl_ddp_200step_zero_32gpu_4node_20260925/checkpoints/rl/step_000100/adapter}"
BCPLUS_ROOT="${JFS_ROOT}/Capability_Evolution/SCOPE/external/BrowseComp-Plus"
BCPLUS_CORPUS="${BCPLUS_ROOT}/data/browsecomp_plus_corpus_full.jsonl"
BCPLUS_INDEX="${BCPLUS_ROOT}/indexes/bm25"
TIKTOKEN_ZIP="${JFS_ROOT}/tiktoken-bundle.zip"
TIKTOKEN_BUNDLE="/tmp/tiktoken-bundle"
PYPI_INDEX="https://pkgs.d.xiaomi.net/artifactory/api/pypi/pypi-virtual/simple"
EVAL_SCRIPT="${TRIM_ROOT}/scripts/run_bcplus_w_harness1_adapter_gpu8_eval.sh"

: "${BENCHMARK:=bcplus_full}"
: "${RUN_ID:=gpt_oss_20b_adapter_step100_7nzzn_${BENCHMARK}_$(date +%Y%m%d_%H%M%S 2>/dev/null || echo unknown)}"

echo "=================================================================="
echo "[boot] TRIM Harness-1 BC+ / GPT-OSS-20B / ADAPTER eval / 8gpu / BENCHMARK=${BENCHMARK} ($(date -Iseconds 2>/dev/null || echo unknown))"
echo "[boot] base=${MODEL_PATH} (bf16-dequant) adapter=${ADAPTER_PATH} (step100 7nzzn)"
echo "[boot] BENCHMARK=${BENCHMARK} RUN_ID=${RUN_ID} COMPONENT=${COMPONENT:-zero}"
echo "=================================================================="

export PYTHONUNBUFFERED=1

echo "[deps] installing locked runtime from ${PYPI_INDEX}"
python3 -m pip install --no-cache-dir --ignore-installed \
  --upgrade-strategy only-if-needed --index-url "${PYPI_INDEX}" \
  "torch==2.10.0" "transformers==4.57.6" "accelerate" "datasets" \
  "vllm==0.19.1" "hydra-core" "omegaconf" "tensordict" "codetiming" \
  "tqdm" "python-dotenv" "pandas" "pyarrow" "sentencepiece" \
  "pyserini==2.0.0" "structlog" "chromadb" "peft" "openai" "tenacity" "pydantic-settings"

python3 -m pip install --no-cache-dir --no-deps --index-url "${PYPI_INDEX}" \
  "chz" "tinker" "json_repair" || echo "[deps] WARN: chz/tinker/json_repair install had issues"

echo "[deps] resolved:"
python3 - <<'PY'
import importlib.metadata, torch, vllm, transformers
print(f"[deps] torch={torch.__version__} cuda={torch.version.cuda}")
print(f"[deps] vllm={vllm.__version__} transformers={transformers.__version__}")
print(f"[deps] pyserini={importlib.metadata.version('pyserini')}")
PY

echo "[preflight][gpu] nvidia-smi:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv 2>&1 | head -20
python3 - <<'PY'
import os, importlib.metadata, torch, vllm, transformers
print(f"[preflight][rt] torch={torch.__version__} cuda={torch.version.cuda} "
      f"device_count={torch.cuda.device_count()} visible={os.environ.get('CUDA_VISIBLE_DEVICES')}")
assert torch.cuda.device_count() == 8
pv = importlib.metadata.version("pyserini")
assert pv == "2.0.0", f"pyserini must be 2.0.0, got {pv}"
print(f"[preflight][rt] vllm={vllm.__version__} transformers={transformers.__version__} pyserini={pv} OK")
PY

echo "[preflight][jvm] locating libjvm.so"
JVM_PATH="$(find /usr /opt -type f -name libjvm.so 2>/dev/null | head -n 1 || true)"
if [ -z "${JVM_PATH}" ]; then
  apt-get update; apt-get install -y --no-install-recommends openjdk-21-jre-headless
  JVM_PATH="$(find /usr /opt -type f -name libjvm.so 2>/dev/null | head -n 1)"
fi
test -n "${JVM_PATH}" && test -f "${JVM_PATH}"; ldd "${JVM_PATH}" >/dev/null
JAVA_HOME="$(dirname "$(dirname "$(dirname "${JVM_PATH}")")")"
export JVM_PATH JAVA_HOME LD_LIBRARY_PATH="$(dirname "${JVM_PATH}"):${LD_LIBRARY_PATH:-}"
python3 -c 'from jnius import autoclass; print("[preflight][jvm] java_ok", autoclass("java.lang.String"))'

PYSERINI_ROOT=$(python3 -c 'import pathlib, pyserini; print(pathlib.Path(pyserini.__file__).resolve().parent)')
export ANSERINI_CLASSPATH="$(find "${PYSERINI_ROOT}/resources/jars" -type f -name '*.jar' -print | paste -sd: -)"
export CLASSPATH="${ANSERINI_CLASSPATH}"
echo "[preflight][import] ANSERINI_CLASSPATH set"

# 按 BENCHMARK 选 corpus/index
TRANSFER_TB=""
case "${BENCHMARK}" in
  longsealqa|longseal|long_seal|long_seal_qa|longseal-qa) TRANSFER_TB=longsealqa ;;
  frames|frames-benchmark|google_frames|frames_benchmark) TRANSFER_TB=frames ;;
  hotpotqa|hotpot|hotpot_qa|hotpotqa_subset) TRANSFER_TB=hotpotqa ;;
esac
if [ -n "${TRANSFER_TB}" ]; then
  BENCH_INDEX="${TRIM_ROOT}/manifests/transfer_local/${TRANSFER_TB}/indexes/bm25"
  BENCH_CORPUS="${TRIM_ROOT}/manifests/transfer_local/${TRANSFER_TB}/corpus.jsonl"
else
  BENCH_INDEX="${BCPLUS_INDEX}"; BENCH_CORPUS="${BCPLUS_CORPUS}"
fi
for p in "${JFS_ROOT}" "${TRIM_ROOT}" "${EASYOPD_ROOT}" "${MODEL_PATH}" \
         "${ADAPTER_PATH}" "${BENCH_CORPUS}" "${BENCH_INDEX}" \
         "${TRIM_ROOT}/scripts/run_eval.py" "${EVAL_SCRIPT}"; do
  [ -e "$p" ] || { echo "[preflight][fs] MISSING  $p" >&2; exit 2; }
  echo "[preflight][fs] OK  $p"
done
test -f "${MODEL_PATH}/config.json"
test -f "${ADAPTER_PATH}/adapter_config.json"
test -f "${ADAPTER_PATH}/adapter_model.safetensors"

# adapter base 校验放宽：7nzzn 训练 base=/tmp/gpt-oss-20b (mxfp4)，eval 用 bf16-dequant，
# 架构一致 GptOssForCausalLM 即可，LoRA delta 与 base dtype 独立。
python3 - <<PY
import json
cfg = json.load(open("${ADAPTER_PATH}/adapter_config.json"))
base = cfg.get("base_model_name_or_path","")
r = int(cfg.get("r") or 8)
print(f"[preflight][adapter] base_model_name_or_path={base} r={r} alpha={cfg.get('lora_alpha')}")
ok = (base.endswith("gpt-oss-20b") or base == "/tmp/gpt-oss-20b"
      or "gpt-oss-20b" in base)
assert ok, f"adapter base {base} not gpt-oss-20b"
assert r == 8, f"expected r=8, got {r}"
print("[preflight][adapter] base is gpt-oss-20b (mxfp4 train path /tmp/gpt-oss-20b), r=8 OK; eval base=bf16-dequant same arch GptOssForCausalLM, LoRA delta dtype-independent, compatible")
PY

if [ ! -d "${TIKTOKEN_BUNDLE}/share/tiktoken" ]; then
  python3 - <<PY
import zipfile, os, shutil
src="${TIKTOKEN_ZIP}"; dst="${TIKTOKEN_BUNDLE}"
if os.path.exists(dst): shutil.rmtree(dst, ignore_errors=True)
with zipfile.ZipFile(src) as z: z.extractall("/tmp")
print(f"[preflight][tiktoken] extracted -> {dst}")
PY
fi
test -f "${TIKTOKEN_BUNDLE}/share/tiktoken/o200k_base.tiktoken"
export TIKTOKEN_ENCODINGS_BASE="${TIKTOKEN_BUNDLE}/share/tiktoken"
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_BUNDLE}/tiktoken_cache"
export TIKTOKEN_RS_CACHE_DIR="${TIKTOKEN_BUNDLE}/tiktoken_rs_cache"

: "${OPENAI_API_KEY:=**-*****-*******}"
: "${CHROMA_API_KEY:=local-offline}"
: "${CHROMA_DATABASE:=local-offline}"
export OPENAI_API_KEY CHROMA_API_KEY CHROMA_DATABASE
export HARNESS1_FORBID_CHROMA=1

export PYTHONPATH="${TRIM_ROOT}:${EASYOPD_ROOT}"
export RUN_ID ADAPTER="${ADAPTER_PATH}" BASE_MODEL="${MODEL_PATH}" BENCHMARK COMPONENT="${COMPONENT:-zero}"
# ACTOR_GPUS/ACTOR_PORTS: respect env (default 0-7 / 8 ports). If component includes
# verify_tool, auto-remove VERIFY_GPU (default 7) from ACTOR_GPUS AND drop one ACTOR_PORTS
# so counts match (run_bcplus...:432 "ACTOR_GPUS count (N) must match ACTOR_PORTS (M)").
# Avoids msc-type fail (GPU7 conflict + port count mismatch).
ACTOR_GPUS="${ACTOR_GPUS:-0,1,2,3,4,5,6,7}"
ACTOR_PORTS="${ACTOR_PORTS:-8040,8042,8044,8046,8048,8052,8054,8056}"
VERIFY_GPU="${VERIFY_GPU:-7}"
VERIFY_PORT="${VERIFY_PORT:-8050}"
case ",${COMPONENT}," in *,verify_tool,*|*,VERIFY_TOOL,*)
  _vg="${VERIFY_GPU}"
  _ag=""
  IFS=',' read -ra _gpus <<< "${ACTOR_GPUS}"
  for _g in "${_gpus[@]}"; do [ "${_g}" != "${_vg}" ] && _ag="${_ag:+${_ag},}${_g}"; done
  ACTOR_GPUS="${_ag}"
  # drop last port so count matches actor count (verify_tool uses N-1 actors)
  IFS=',' read -ra _ports <<< "${ACTOR_PORTS}"
  _np=""
  _i=0
  for _p in "${_ports[@]}"; do
    _i=$((_i+1))
    [ "${_i}" -lt "${#_ports[@]}" ] && _np="${_np:+${_np},}${_p}"
  done
  ACTOR_PORTS="${_np}"
;; esac
export LAYOUT="multi" ACTOR_GPUS ACTOR_PORTS VERIFY_GPU VERIFY_PORT
export EVAL_TP="${EVAL_TP:-32}" GPU_UTIL="${GPU_UTIL:-0.90}" VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}"
export MAX_TURNS="${MAX_TURNS:-40}" MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}" TEMPERATURE="${TEMPERATURE:-1.0}"
export WORKER_STAGGER="${WORKER_STAGGER:-2}"
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK NODE_RANK MASTER_ADDR MASTER_PORT PET_NPROC_PER_NODE 2>/dev/null || true

echo "[env] ADAPTER=${ADAPTER} BASE=${BASE_MODEL} (bf16-dequant) BENCHMARK=${BENCHMARK} COMPONENT=${COMPONENT}"
echo "[env] LAYOUT=${LAYOUT} ACTOR_GPUS=${ACTOR_GPUS} EVAL_TP=${EVAL_TP} GPU_UTIL=${GPU_UTIL}"
echo "[env] RUN_ID=${RUN_ID}"

echo "=================================================================="
echo "[eval] bash ${EVAL_SCRIPT}  (ADAPTER=${ADAPTER} BASE=${BASE_MODEL} BENCHMARK=${BENCHMARK})"
echo "=================================================================="
cd "${TRIM_ROOT}"
exec bash "${EVAL_SCRIPT}"
