#!/usr/bin/env bash
# CloudML 启动脚本：TRIM Harness-1 BC+ full / GLM-4-32B-0414 / ALL (component=all) eval / 单节点 8x gm402-80g
#
# 本任务 = 2026-09-15 筹备的 MemQA-General-V5 8 卡 GLM-4-32B-0414 ALL (component=all) eval。
# 被测模型 = /mnt/songzijun/models/GLM-4-32B-0414
#   config.json 架构 Glm4ForCausalLM / model_type glm4 / dense 32.6B
#   / 61 层 / vocab 151552 / GQA(2 KV heads, head_dim 128) / bfloat16。
#   dense 模型（非 MoE），故 vLLM 不加 --moe-backend；chat_template 无 thinking
#   标记 => 默认非 thinking（不传 --reasoning-parser / enable_thinking）。
#   tool-call-parser=glm45（与 trim.eval.model_profiles 对 GLM-4 系列的路由一致）。
# 跑的是 run_bcplus_full_glm4_32b_0414_all_gpu8_eval.sh（7 actor + 1 verifier + 64 workers）：
#   7 个单卡 vLLM actor (TP=1, GPU 0..6) + 1 harness-1 verifier (GPU 7) + 64 eval workers,
#   retrieval=local_bm25 (Pyserini/Lucene), --reranker none --offline, 无 Chroma 服务。
#
# 镜像: micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto
#   （该镜像基础层缺 torch 等依赖，按 CLAUDE.md 经验在启动阶段显式锁定并安装四包配置）
# GPU 利用：GLM-4-32B-0414 权重 ~65GB/卡（bf16），GQA 2 KV heads => KV 极小；
#   纯推理 eval 无 backward，GPU_UTIL=0.90（~72GB）充分用显存且留足余量（与已验证
#   Qwen3-4B/harness-1 zero GPU_UTIL 0.90 一致）。
# JuiceFS: volume=songzijun, cluster=bj9-cloudml (endpoint), mount=/mnt/songzijun
# 依赖锁定对齐 TRIM Harness-1 BC+ 已验证的 runtime:
#   torch==2.10.0 / transformers==4.57.6 / vllm==0.19.1 / pyserini==2.0.0
#   + structlog/chromadb/peft/openai_harmony/chz/tinker/json_repair
# 由 ImageCommand 调用；保持幂等、可审计；preflight 全量打印后再进 eval。
#
# RUN_ID 与前序 attempt 显式隔离，不覆盖原 attempt 输出。
# GLM-4-32B-0414 zero 走裸模型（不加载 LoRA），与 vLLM lora_shrink 补丁无关（脚本仍幂等应用）。
set -euo pipefail

JFS_ROOT="/mnt/songzijun"
TRIM_ROOT="${JFS_ROOT}/Capability_Evolution/TRIM"
EASYOPD_ROOT="${JFS_ROOT}/Capability_Evolution/SCAPE-EasyOPD"
HARNESS1_EXTERNAL="${TRIM_ROOT}/external/harness-1"
MODEL_PATH="${JFS_ROOT}/models/GLM-4-32B-0414"
TIKTOKEN_ZIP="${JFS_ROOT}/tiktoken-bundle.zip"
TIKTOKEN_BUNDLE="/tmp/tiktoken-bundle"
PYPI_INDEX="https://pkgs.d.xiaomi.net/artifactory/api/pypi/pypi-virtual/simple"

echo "=================================================================="
echo "[boot] TRIM Harness-1 BC+ full / GLM-4-32B-0414 / ALL (component=all) eval / 8gpu  ($(date -Iseconds 2>/dev/null || echo unknown))"
echo "[boot] host=$(hostname) pwd=$(pwd) user=$(whoami) argv=$*"
echo "=================================================================="

export PYTHONUNBUFFERED=1

# ---- 1. 依赖安装（独立诊断 attempt，按 CLAUDE.md 经验镜像连 torch 都缺）----
# 注意：不要把 torch==2.6.0 与 vllm==0.19.1 一起装，后者要求 torch==2.10.0。
# 基础镜像 distutils blinker 1.4 阻断 pip 卸载，用内部 Xiaomi PyPI + --ignore-installed。
# pyserini 必须显式 pin ==2.0.0（裸 pyserini 现在解析到 2.4.0，preflight 断言 2.0.0 必失败）。
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
  "pyserini==2.0.0" \
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

# ---- 1b. vLLM 0.19.1 LoRA shrink op 补丁（assert inputs.is_contiguous -> inputs.contiguous()）----
# ZERO 跑裸 GLM-4-32B 不加载 LoRA，该补丁对本任务无害但保持与已验证 runtime 一致。
echo "[patch] applying vLLM lora_shrink contiguous patch (no-op for zero bare model)"
python3 "${TRIM_ROOT}/scripts/patch_vllm_lora_shrink_contiguous.py" \
  || { echo "[patch] FAILED vLLM lora_shrink patch" >&2; exit 3; }

# ---- 2. preflight: GPU / runtime 版本（任一不符不得进入 eval）----
echo "[preflight][gpu] nvidia-smi summary:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv 2>&1 | head -20 || echo "[preflight][gpu] nvidia-smi unavailable"
python3 - <<'PY'
import os, sys, importlib.metadata
import torch, vllm, transformers, pyserini, structlog, chromadb, peft
def ver(pkg):  # pyserini 2.0.0 不导出 __version__，用 importlib.metadata 兜底
    try:
        return importlib.metadata.version(pkg)
    except Exception:
        return getattr(__import__(pkg), "__version__", "unknown")
print(f"[preflight][rt] torch={torch.__version__} cuda={torch.version.cuda} "
      f"device_count={torch.cuda.device_count()} visible={os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"[preflight][rt] vllm={vllm.__version__} transformers={transformers.__version__} pyserini={ver('pyserini')}")
print(f"[preflight][rt] structlog={structlog.__version__} chromadb={chromadb.__version__} peft={peft.__version__}")
assert torch.cuda.device_count() == 8, f"expected 8 GPUs, got {torch.cuda.device_count()}"
assert vllm.__version__ == "0.19.1", f"expected vllm 0.19.1, got {vllm.__version__}"
assert transformers.__version__ == "4.57.6", f"expected transformers 4.57.6, got {transformers.__version__}"
assert ver("pyserini") == "2.0.0", f"expected pyserini 2.0.0, got {ver('pyserini')}"
import openai_harmony
print("[preflight][import] openai_harmony OK")
# GLM-4-32B-0414 (model_type=glm4) tokeniser has no <|tool_call|> token, so
# vLLM glm45/glm47 parsers cannot extract structured tool_calls.  The model
# emits pythonic func(args) / JSON in content.  The api_adapter
# extract_tool_calls_from_content() fallback recovers them at eval time.
# We keep --tool-call-parser glm45 (matches model_profiles GLM-4 routing);
# the content fallback is the actual parser for this model's output.
import importlib, pkgutil
def _enum_parsers():
    cands = [
        "vllm.tool_parsers.abstract_tool_parser",
        "vllm.v1.entrypoints.openai.tool_parsers",
        "vllm.entrypoints.openai.tool_parsers",
        "vllm.entrypoints.tool_inputs",
    ]
    for mod in cands:
        try:
            m = importlib.import_module(mod)
        except Exception:
            continue
        for attr in ("ToolParserManager", "tool_parsers"):
            obj = getattr(m, attr, None)
            tps = getattr(obj, "tool_parsers", None) if obj is not None else None
            if isinstance(tps, dict):
                return sorted(tps.keys())
    try:
        found = []
        for _, name, ispkg in pkgutil.walk_packages(vllm.__path__, prefix="vllm."):
            if "tool_parser" in name and not ispkg:
                found.append(name)
        return found
    except Exception:
        pass
    return None
registered = _enum_parsers()
if registered is not None:
    print(f"[preflight][parser] candidate tool parsers: {registered}")
    if any("pythonic" in p.lower() for p in registered):
        print("[preflight][parser] pythonic parser found in vLLM 0.19.1")
    else:
        print("[preflight][parser] WARN: no pythonic parser; will let vllm serve report if unsupported")
else:
    print("[preflight][parser] WARN: could not enumerate; will let vllm serve report if unsupported")
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
export JVM_PATH
JAVA_HOME="$(dirname "$(dirname "$(dirname "${JVM_PATH}")")")"
export JAVA_HOME
echo "[preflight][jvm] JVM_PATH=${JVM_PATH} JAVA_HOME=${JAVA_HOME}"
export LD_LIBRARY_PATH="$(dirname "${JVM_PATH}"):${LD_LIBRARY_PATH:-}"
python3 -c 'from jnius import autoclass; print("[preflight][jvm] java_ok", autoclass("java.lang.String"))'

# ---- 3b. ANSERINI_CLASSPATH 指向 pyserini 包内 resources/jars（不能指向包根）----
export ANSERINI_CLASSPATH="$(python3 -c 'import pyserini,os;print(os.path.join(os.path.dirname(pyserini.__file__),"resources","jars"))')"
echo "[preflight][jvm] ANSERINI_CLASSPATH=${ANSERINI_CLASSPATH}"
test -d "${ANSERINI_CLASSPATH}" && echo "[preflight][jvm] anserini jar dir OK" \
  || { echo "[preflight][jvm] ANSERINI_CLASSPATH jar dir MISSING" >&2; exit 2; }

# ---- 4. preflight: 离线凭证占位 + import 链（harness.ultra_core / harness.tools）----
# 必须在首次 import pyserini.search.lucene 之前设 OPENAI_API_KEY 占位（pyserini.encode._openai 模块加载即实例化）。
: "${OPENAI_API_KEY:=**-*****-*******}"
: "${CHROMA_API_KEY:=local-offline}"
: "${CHROMA_DATABASE:=local-offline}"
export OPENAI_API_KEY CHROMA_API_KEY CHROMA_DATABASE
export HARNESS1_FORBID_CHROMA=1   # local_bm25 路径 worker 内部也会设；这里 shell 层先保底
echo "[preflight][import] structlog/chromadb/pyserini/peft/openai_harmony + harness ultra_core/tools"
export BCPLUS_INDEX_LUCENE="${JFS_ROOT}/Capability_Evolution/SCOPE/external/BrowseComp-Plus/indexes/bm25"
PYTHONPATH="${TRIM_ROOT}:${EASYOPD_ROOT}:${HARNESS1_EXTERNAL}" \
  python3 - <<'PY'
import os
import structlog, chromadb, pyserini, peft
print(f"[preflight][import] structlog={structlog.__version__} peft={peft.__version__}")
import openai_harmony
print("[preflight][import] openai_harmony OK")
# Lucene 离线打开（必须在 offline credentials 之后；ANSERINI_CLASSPATH 已 export）
from trim.eval.offline_credentials import ensure_local_offline_credentials
ensure_local_offline_credentials()
from pyserini.search.lucene import LuceneSearcher
idx = os.environ.get("BCPLUS_INDEX_LUCENE")
searcher = LuceneSearcher(idx)
# pyserini 2.0.0: num_docs 是属性（getattr），不是方法 —— 与 browsecomp_retrieval.py 一致
n = int(getattr(searcher, "num_docs", 0) or 0)
hits = searcher.search("test", k=1)
print(f"[preflight][lucene] index={idx} num_docs={n} probe_hits={len(hits)}")
assert n > 0
PY

# ---- 5. preflight: JuiceFS 挂载与关键路径 ----
echo "[preflight][fs] JuiceFS mount check:"
mount | grep -E "songzijun|JuiceFS" || echo "[preflight][fs] no songzijun mount line (may be CSI)"
BCPLUS_INDEX="${JFS_ROOT}/Capability_Evolution/SCOPE/external/BrowseComp-Plus/indexes/bm25"
BCPLUS_CORPUS="${JFS_ROOT}/Capability_Evolution/SCOPE/external/BrowseComp-Plus/data/browsecomp_plus_corpus_full.jsonl"
for p in "${JFS_ROOT}" "${TRIM_ROOT}" "${EASYOPD_ROOT}" "${MODEL_PATH}" \
         "${BCPLUS_CORPUS}" "${BCPLUS_INDEX}" \
         "${TRIM_ROOT}/scripts/run_eval.py" \
         "${TRIM_ROOT}/scripts/run_bcplus_full_glm4_32b_0414_all_gpu8_eval.sh" \
         "${TIKTOKEN_ZIP}"; do
  if [ -e "$p" ]; then
    echo "[preflight][fs] OK  $p"
  else
    echo "[preflight][fs] MISSING  $p" >&2
    exit 2
  fi
done

# ---- 6. preflight: tiktoken 离线词表（o200k_harmony 用 o200k_base.tiktoken）----
# GLM-4-32B-0414 用自有 tokenizer，glm45 parser 不走 openai_harmony；但 harness 导入链
# 仍可能 import openai_harmony，保留离线词表 export 以防 Rust 层联网下载。
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

# ---- 6b. tiktoken 离线路径在 shell 层 export，让 vLLM actor 子进程继承 ----
export TIKTOKEN_ENCODINGS_BASE="${TIKTOKEN_BUNDLE}/share/tiktoken"
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_BUNDLE}/tiktoken_cache"
export TIKTOKEN_RS_CACHE_DIR="${TIKTOKEN_BUNDLE}/tiktoken_rs_cache"
test -f "${TIKTOKEN_ENCODINGS_BASE}/o200k_base.tiktoken" \
  && echo "[preflight][tiktoken] TIKTOKEN_ENCODINGS_BASE=${TIKTOKEN_ENCODINGS_BASE} (o200k OK)" \
  || { echo "[preflight][tiktoken] o200k not under TIKTOKEN_ENCODINGS_BASE" >&2; exit 2; }

# ---- 7. eval 环境变量 ----
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="${TRIM_ROOT}:${EASYOPD_ROOT}:${HARNESS1_EXTERNAL}"
export TRIM_GPU_KEEPALIVE=0
# 固定 RUN_ID，输出目录与前序 attempt 显式隔离（不覆盖原 attempt）。
export RUN_ID="${RUN_ID:-20260915_memqa_general_v5_bcplus_full_glm4_32b_0414_all_gpu8_r1}"

echo "[env] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[env] PYTHONPATH=${PYTHONPATH}"
echo "[env] HARNESS1_FORBID_CHROMA=${HARNESS1_FORBID_CHROMA} (local_bm25, no Chroma server)"
echo "[env] ANSERINI_CLASSPATH=${ANSERINI_CLASSPATH}"
echo "[env] OPENAI_API_KEY set (placeholder, offline)"
echo "[env] TIKTOKEN_ENCODINGS_BASE=${TIKTOKEN_ENCODINGS_BASE}"
echo "[env] model_path=${MODEL_PATH}"
echo "[env] run_id=${RUN_ID}"

# ---- 8. 启动 GLM-4-32B-0414 ALL eval（7 单卡 actor + 1 verifier + 64 eval workers）----
echo "=================================================================="
echo "[eval] cd ${TRIM_ROOT} && bash scripts/run_bcplus_full_glm4_32b_0414_all_gpu8_eval.sh"
echo "=================================================================="
cd "${TRIM_ROOT}"

exec bash scripts/run_bcplus_full_glm4_32b_0414_all_gpu8_eval.sh
