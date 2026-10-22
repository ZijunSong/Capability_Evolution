#!/usr/bin/env bash
# CloudML 启动脚本：TRIM Harness-1 BC+ full / Qwen3.5-27B / ZERO|ALL eval (no-thinking) / 单节点 8x gm402-80g
# 统一 entry（fork boot_qwen35_27b_all_nothink_eval_cloudml_8gpu_20260915.sh + zero boot 的 section 0-6
# preflight 合并）：EVAL_MODE + BENCHMARK + RUN_ID 由 submit --env 注入，按 EVAL_MODE 分支 zero/all
# wrapper。4 benchmark（bcplus_full/longsealqa/frames/hotpotqa）路由由 lib 的 case-BENCHMARK 处理。
#
# 提交日期 2026-10-15，标签 MemQA-General-V5，priority 5，queue 2334。用于 Qwen3.5-27B 训练前
# base × {zero,all} × 4bench × 3seed = 24 任务基线（vs vqqxp/iowty 训练曲线的 base 对照）。
# zero/all 都用 64 worker 统一口径（zero TP=64 / all ALL_EVAL_TP=64）使 3-seed 均值可靠、
# zero vs all 可直接对比。不复用 9 月旧结果（zero fcanl 16-worker / all v2_64k 64-worker 口径
# 不一致且只 1 trial）。
#
# 被测模型 = /mnt/songzijun/models/Qwen3.5-27B（Qwen3_5ForConditionalGeneration dense 27B,
# 51.75GiB bf16/卡, chat_template 两处已改关 thinking, 备份 *.orig_thinking_on_*）。
#
# Thinking 关闭：模型 chat_template 默认 no-thinking（enable_thinking 守卫改 "is not true"），
# vLLM 0.19.1 serve CLI 不支持 --chat-template-kwargs 故靠模板默认。wrapper 的 vllm_extra_for_model
# override strip --reasoning-parser qwen3（no-think 必需，否则吞 tool call→tool_calls=[] content=null）。
#
# 关键风险：vLLM 0.19.1 是否注册 Qwen3_5ForConditionalGeneration，preflight 探测不支持则 exit 5。
# 9 月 fcanl(zero)/iytmg(all) 已验证支持。
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

# EVAL_MODE / BENCHMARK / RUN_ID 由 submit --env 注入（默认值仅本地调试用）。
: "${EVAL_MODE:=zero}"          # zero | all
: "${BENCHMARK:=bcplus_full}"   # bcplus_full | longsealqa | frames | hotpotqa
: "${EVAL_SEED:=42}"            # 42|123|2024 — drives run_eval.py --seed (lib run_eval_cell hook)
case "${EVAL_MODE}" in zero|all) ;; *) echo "FATAL: EVAL_MODE=${EVAL_MODE} (want zero|all)" >&2; exit 1 ;; esac
case "${BENCHMARK}" in bcplus_full|longsealqa|frames|hotpotqa) ;; *) echo "FATAL: BENCHMARK=${BENCHMARK}" >&2; exit 1 ;; esac

# 按 EVAL_MODE 选 wrapper（zero 用新建 thin lib-sourcing wrapper 获 4-bench 路由；all 用已就绪 wrapper）。
if [ "${EVAL_MODE}" = "all" ]; then
  EVAL_SCRIPT="${TRIM_ROOT}/scripts/run_bcplus_w_harness1_qwen35_27b_all_nothink_gpu8_eval.sh"
else
  EVAL_SCRIPT="${TRIM_ROOT}/scripts/run_bcplus_w_harness1_qwen35_27b_zero_nothink_gpu8_eval.sh"
fi

echo "=================================================================="
echo "[boot] TRIM Harness-1 BC+ full / Qwen3.5-27B / ${EVAL_MODE} (no-thinking) / BENCHMARK=${BENCHMARK} / 8gpu  ($(date -Iseconds 2>/dev/null || echo unknown))"
echo "[boot] host=$(hostname) pwd=$(pwd) user=$(whoami) argv=$*"
echo "=================================================================="

export PYTHONUNBUFFERED=1

# ---- 0. 把启动脚本从 JuiceFS cp 到 /tmp 再 exec（规避 JuiceFS heredoc 回读抖动）----
SELF="${BASH_SOURCE[0]:-$0}"
if [[ "${SELF}" == /mnt/songzijun/* ]]; then
  TMP_SELF="/tmp/$(basename "${SELF}")"
  cp -f "${SELF}" "${TMP_SELF}"
  exec bash "${TMP_SELF}" "$@"
fi

# ---- 1. 依赖安装（四件套 + harness import 链）----
echo "[deps] installing locked runtime from ${PYPI_INDEX}"
python3 -m pip install --no-cache-dir --ignore-installed \
  --upgrade-strategy only-if-needed \
  --index-url "${PYPI_INDEX}" \
  "torch==2.10.0" "transformers==4.57.6" "accelerate" "datasets" "vllm==0.19.1" \
  "hydra-core" "omegaconf" "tensordict" "codetiming" "tqdm" "python-dotenv" \
  "pandas" "pyarrow" "sentencepiece" "pyserini==2.0.0" "structlog" "chromadb" \
  "openai" "tenacity" "anthropic" "peft" "openai_harmony" "pydantic-settings"

python3 -m pip install --no-cache-dir --no-deps --index-url "${PYPI_INDEX}" \
  "chz" "tinker" "json_repair" || echo "[deps] WARN: chz/tinker/json_repair install had issues"

echo "[deps] printing resolved versions:"
python3 - <<'PY'
import importlib.metadata, torch, vllm, transformers
def ver(p):
    try: return importlib.metadata.version(p)
    except Exception: return getattr(__import__(p), "__version__", "unknown")
print(f"[deps] torch={torch.__version__} cuda={torch.version.cuda}")
print(f"[deps] vllm={vllm.__version__} transformers={transformers.__version__} pyserini={ver('pyserini')}")
PY

# ---- 2. preflight: GPU / runtime 版本 ----
echo "[preflight][gpu] nvidia-smi summary:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv 2>&1 | head -20 || echo "[preflight][gpu] nvidia-smi unavailable"
python3 - <<'PY'
import os, importlib.metadata
import torch, vllm, transformers
def ver(p):
    try: return importlib.metadata.version(p)
    except Exception: return getattr(__import__(p), "__version__", "unknown")
print(f"[preflight][rt] torch={torch.__version__} cuda={torch.version.cuda} device_count={torch.cuda.device_count()} visible={os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"[preflight][rt] vllm={vllm.__version__} transformers={transformers.__version__} pyserini={ver('pyserini')}")
assert torch.cuda.device_count() == 8, f"expected 8 GPUs, got {torch.cuda.device_count()}"
assert vllm.__version__ == "0.19.1", f"expected vllm 0.19.1, got {vllm.__version__}"
assert transformers.__version__ == "4.57.6", f"expected transformers 4.57.6, got {transformers.__version__}"
assert ver("pyserini") == "2.0.0", f"expected pyserini 2.0.0, got {ver('pyserini')}"
PY

# ---- 2b. preflight: vLLM 是否注册 Qwen3_5ForConditionalGeneration ----
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
    try:
        import vllm.model_executor.models as m, pkgutil, importlib
        found = False
        for _, name, _ in pkgutil.iter_modules(m.__path__):
            if "qwen3" in name.lower():
                mod = importlib.import_module(f"vllm.model_executor.models.{name}")
                src = open(mod.__file__).read() if hasattr(mod, "__file__") and mod.__file__ else ""
                if "Qwen3_5ForConditionalGeneration" in src or "Qwen3_5" in src:
                    print(f"[preflight][arch] fallback: found Qwen3_5 ref in vllm.model_executor.models.{name}")
                    found = True
        if not found:
            print("[preflight][arch] fallback scan: no Qwen3_5 reference found")
            sys.exit(5)
        print("[preflight][arch] SUPPORTED (via fallback scan)")
    except SystemExit:
        raise
    except Exception as e2:
        print(f"[preflight][arch] fallback scan also failed: {type(e2).__name__}: {e2}")
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

# ---- 4. preflight: 离线凭证占位 + 导入链 ----
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

# ---- 5. preflight: JuiceFS 挂载与关键路径（all 才需 verify model）----
echo "[preflight][fs] JuiceFS mount check:"
mount | grep -E "songzijun|JuiceFS" || echo "[preflight][fs] no songzijun mount line (may be CSI)"
FS_PATHS=("${JFS_ROOT}" "${TRIM_ROOT}" "${EASYOPD_ROOT}" "${MODEL_PATH}" \
         "${BCPLUS_CORPUS}" "${BCPLUS_INDEX}" \
         "${TRIM_ROOT}/scripts/run_eval.py" "${EVAL_SCRIPT}" \
         "${TIKTOKEN_ZIP}")
# all 模式额外检查 verify model（zero 无 verifier）
if [ "${EVAL_MODE}" = "all" ]; then
  FS_PATHS+=("${VERIFY_MODEL_PATH}")
fi
for p in "${FS_PATHS[@]}"; do
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
if [ "${EVAL_MODE}" = "all" ]; then
  test -f "${VERIFY_MODEL_PATH}/config.json" \
    && echo "[preflight][fs] verify model config.json OK" \
    || { echo "[preflight][fs] verify model config.json MISSING" >&2; exit 2; }
fi

# ---- 6. preflight: tiktoken 离线词表 ----
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
test -f "${TIKTOKEN_ENCODINGS_BASE}/o200k_base.tiktoken" \
  && echo "[preflight][tiktoken] TIKTOKEN_ENCODINGS_BASE=${TIKTOKEN_ENCODINGS_BASE} (o200k OK)" \
  || { echo "[preflight][tiktoken] o200k not under TIKTOKEN_ENCODINGS_BASE" >&2; exit 2; }

# ---- 7. eval 环境变量（zero/all 分支，统一 64 worker 口径）----
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="${TRIM_ROOT}:${EASYOPD_ROOT}"
export TRIM_GPU_KEEPALIVE=0
# 27B dense 51.75GiB/卡：GPU_UTIL=0.90 (~72GiB) 放权重+KV；MAX_NUM_SEQS=6；MAX_MODEL_LEN=65536
# (消除 BC+ 长轨迹 32k 超限 infra_error，zero/all 同口径)。
export GPU_UTIL="${GPU_UTIL:-0.90}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-6}"
export QWEN35_NO_THINKING="${QWEN35_NO_THINKING:-1}"
export QWEN35_MAX_MODEL_LEN="${QWEN35_MAX_MODEL_LEN:-65536}"
export MAX_TURNS="${MAX_TURNS:-40}"
export WORKER_STAGGER="${WORKER_STAGGER:-0}"

if [ "${EVAL_MODE}" = "zero" ]; then
  # zero 布局：8 个 Qwen3.5-27B actor (TP=1, GPU 0..7, ports 8040..8054)，无 verifier。
  # TP=64 = 64 eval worker（与 all 的 ALL_EVAL_TP=64 统一口径，使 zero/all 可直接对比）。
  export ZERO_ACTOR_GPUS="${ZERO_ACTOR_GPUS:-0,1,2,3,4,5,6,7}"
  export ZERO_ACTOR_PORTS="${ZERO_ACTOR_PORTS:-8040,8042,8044,8046,8048,8050,8052,8054}"
  export TP="${TP:-64}"
else
  # all 布局：7 actor (GPU 0..6) + harness-1 verifier (GPU 7)，ALL_EVAL_TP=64。
  export ALL_TP="${ALL_TP:-7}"
  export ALL_EVAL_TP="${ALL_EVAL_TP:-64}"
  export ALL_ACTOR_GPUS="${ALL_ACTOR_GPUS:-0,1,2,3,4,5,6}"
  export ALL_ACTOR_PORTS="${ALL_ACTOR_PORTS:-8042,8044,8046,8048,8050,8052,8054}"
  export ALL_VERIFY_GPU="${ALL_VERIFY_GPU:-7}"
  export ALL_VERIFY_PORT="${ALL_VERIFY_PORT:-8056}"
fi

# BENCHMARK 传给 wrapper（wrapper source lib，lib 的 case-BENCHMARK 路由到 transfer_local）。
# EVAL_SEED 传给 lib run_eval_cell 的 --seed 钩子（默认 42 向后兼容，多 seed 时传不同值）。
export MODEL_PATH BENCHMARK RUN_ID EVAL_SEED
export MODEL_SLUG="Qwen3.5-27B" API_MODEL="Qwen3.5-27B"

echo "[env] EVAL_MODE=${EVAL_MODE} BENCHMARK=${BENCHMARK} EVAL_SEED=${EVAL_SEED} RUN_ID=${RUN_ID}"
echo "[env] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} PYTHONPATH=${PYTHONPATH}"
echo "[env] HARNESS1_FORBID_CHROMA=${HARNESS1_FORBID_CHROMA} (local_bm25, no Chroma server)"
echo "[env] ANSERINI_CLASSPATH=${ANSERINI_CLASSPATH}"
echo "[env] TIKTOKEN_ENCODINGS_BASE=${TIKTOKEN_ENCODINGS_BASE}"
echo "[env] GPU_UTIL=${GPU_UTIL} VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS} QWEN35_NO_THINKING=${QWEN35_NO_THINKING} QWEN35_MAX_MODEL_LEN=${QWEN35_MAX_MODEL_LEN}"
if [ "${EVAL_MODE}" = "zero" ]; then
  echo "[env] ZERO_ACTOR_GPUS=${ZERO_ACTOR_GPUS} ZERO_ACTOR_PORTS=${ZERO_ACTOR_PORTS} TP=${TP}"
else
  echo "[env] ALL_TP=${ALL_TP} ALL_EVAL_TP=${ALL_EVAL_TP} ALL_ACTOR_GPUS=${ALL_ACTOR_GPUS} ALL_VERIFY_GPU=${ALL_VERIFY_GPU}"
fi
echo "[env] model_path=${MODEL_PATH} $( [ "${EVAL_MODE}" = "all" ] && echo "verify_path=${VERIFY_MODEL_PATH}" )"
echo "[env] run_id=${RUN_ID}"

# ---- 8. 按 EVAL_MODE 启动 eval ----
echo "=================================================================="
echo "[eval] cd ${TRIM_ROOT} && bash scripts/$(basename "${EVAL_SCRIPT}")  (EVAL_MODE=${EVAL_MODE} BENCHMARK=${BENCHMARK})"
echo "=================================================================="
cd "${TRIM_ROOT}"

exec bash "scripts/$(basename "${EVAL_SCRIPT}")"
