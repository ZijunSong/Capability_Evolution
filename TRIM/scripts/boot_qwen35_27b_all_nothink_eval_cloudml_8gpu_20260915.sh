#!/usr/bin/env bash
# CloudML 启动脚本：TRIM Harness-1 BC+ full / Qwen3.5-27B / ALL eval (no-thinking) / 单节点 8x gm402-80g
# 提交日期 2026-09-15，标签 MemQA-General-V5，priority 8，queue 2334。
#
# 这是 Qwen3.5-27B 的 "all" 组件版（component=all，带 harness-1 verifier），区别于
# 2026-09-14 已成功的 "zero" 版（job t-20260914151008-fcanl，830/830 ok，Recall 32.0%）。
# preflight/依赖/四件套锁定/JVM/tiktoken 全部沿用 zero 成功 boot 脚本
# (run_qwen35_27b_zero_eval_cloudml_8gpu.sh)；仅 EVAL_SCRIPT/RUN_ID/all 布局 env 不同。
#
# 被测模型 = /mnt/songzijun/models/Qwen3.5-27B
#   config.json 架构 Qwen3_5ForConditionalGeneration / model_type qwen3_5
#   / 64 层 linear+full 混合注意力 / dense（无 MoE）/ 11 shards 51.75 GiB bf16
#   / 多模态 VLM（Qwen3VLProcessor），BC+ 走纯文本路径
#   / tokenizer_class=Qwen2Tokenizer（自带 tokenizer.json，非 harmony o200k）。
#
# Thinking 关闭：已修改模型目录 chat_template（chat_template.jinja 独立文件 +
# tokenizer_config.json 内嵌字段两处）的 enable_thinking 守卫为 "is not true"，
# 使默认（worker 不传 enable_thinking）即走 no-thinking 分支。已用 transformers
# apply_chat_template 验证 default == enable_thinking=False。
# vLLM 0.19.1 serve CLI 不支持 --chat-template-kwargs flag（实测报 unrecognized
# arguments），故用模板默认值而非 serve 参数。不动 scorer/runner（CLAUDE.md：官方
# runner 不 fork）。模板备份在 /mnt/songzijun/models/Qwen3.5-27B/*.orig_thinking_on_*。
#
# 关键修复（同 zero 成功经验）：no-thinking 下 wrapper 显式 STRIP --reasoning-parser qwen3。
# recovery1 smoke 失败根因 = reasoning_parser=qwen3 把 tool call 文本吞进 reasoning 字段
# → tool_calls=[] content=null；no-thinking 无独立 reasoning 块，去掉后 tool call 文本留
# content 供 qwen3_coder 解析。verify(harness-1) 不变（openai+triton）。
#
# 镜像: micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto
#   （该镜像基础层缺 torch 等依赖，按 CLAUDE.md 经验在启动阶段显式锁定并安装四包配置）
# JuiceFS: volume=songzijun, endpoint=bj9-cloudml, mount=/mnt/songzijun
#   — JuiceFS 挂载由 CloudML 通过 CLI --juicefs_mount_configs 注入，不在本脚本处理。
# 依赖锁定对齐 TRIM Harness-1 BC+ 已验证的 runtime（四件套口径，CLAUDE.md）：
#   torch==2.10.0 / transformers==4.57.6 / vllm==0.19.1 / pyserini==2.0.0
#   + structlog/chromadb/openai_harmony/chz/tinker/json_repair（chz/tinker --no-deps 保护四件套）
# 由 ImageCommand 调用；保持幂等、可审计；preflight 全量打印后再进 eval。
# eval 脚本：TRIM/scripts/run_bcplus_w_harness1_qwen35_27b_all_nothink_gpu8_eval.sh
#   （6 个 Qwen3.5-27B actor TP=1 on GPU 0-5 + harness-1 verifier on GPU 6，
#    run_eval.py --harness Harness-1 --evaluation-path upstream_api --component all --tp 64，
#    local_bm25 无 Chroma）
#
# 关键风险：vLLM 0.19.1 是否注册 Qwen3_5ForConditionalGeneration 无法本地验证，
#   本脚本 preflight 用 vllm.model_executor.models.registry.ModelRegistry 探测，
#   不支持则 exit 5 干净退出（保留 attempt 供审计，不混入后续）。
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
EVAL_SCRIPT="${TRIM_ROOT}/scripts/run_bcplus_w_harness1_qwen35_27b_all_nothink_gpu8_eval.sh"

echo "=================================================================="
echo "[boot] TRIM Harness-1 BC+ full / Qwen3.5-27B / ALL (no-thinking) eval / 8gpu  ($(date -Iseconds 2>/dev/null || echo unknown))"
echo "[boot] host=$(hostname) pwd=$(pwd) user=$(whoami) argv=$*"
echo "=================================================================="

export PYTHONUNBUFFERED=1

# ---- 0. 把启动脚本从 JuiceFS cp 到 /tmp 再 exec（规避 JuiceFS heredoc 回读抖动）----
# 经验（CLAUDE.md）：ImageCommand 直跑 JuiceFS bash 脚本，heredoc 回读脚本文件时
# JuiceFS 抖动报 "error reading input file"。cp 到本地 /tmp 后 exec 规避。
SELF="${BASH_SOURCE[0]:-$0}"
if [[ "${SELF}" == /mnt/songzijun/* ]]; then
  TMP_SELF="/tmp/$(basename "${SELF}")"
  cp -f "${SELF}" "${TMP_SELF}"
  exec bash "${TMP_SELF}" "$@"
fi

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
  "python-dotenv" \
  "pandas" \
  "pyarrow" \
  "sentencepiece" \
  "pyserini==2.0.0" \
  "structlog" \
  "chromadb" \
  "openai" \
  "tenacity" \
  "anthropic" \
  "peft" \
  "openai_harmony" \
  "pydantic-settings"

# chz/tinker/json_repair 用 --no-deps，避免它们拉取会破坏四件套的依赖（CLAUDE.md 经验）。
python3 -m pip install --no-cache-dir --no-deps \
  --index-url "${PYPI_INDEX}" \
  "chz" \
  "tinker" \
  "json_repair" || echo "[deps] WARN: chz/tinker/json_repair install had issues (all eval may not need tinker)"

echo "[deps] printing resolved versions:"
python3 - <<'PY'
import importlib.metadata, torch, vllm, transformers
def ver(p):
    try:
        return importlib.metadata.version(p)
    except Exception:
        return getattr(__import__(p), "__version__", "unknown")
print(f"[deps] torch={torch.__version__} cuda={torch.version.cuda}")
print(f"[deps] vllm={vllm.__version__} transformers={transformers.__version__} pyserini={ver('pyserini')}")
PY

# ---- 2. preflight: GPU / runtime 版本（任一不符不得进入 eval）----
echo "[preflight][gpu] nvidia-smi summary:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv 2>&1 | head -20 || echo "[preflight][gpu] nvidia-smi unavailable"
python3 - <<'PY'
import os, importlib.metadata
import torch, vllm, transformers
def ver(p):
    try:
        return importlib.metadata.version(p)
    except Exception:
        return getattr(__import__(p), "__version__", "unknown")
print(f"[preflight][rt] torch={torch.__version__} cuda={torch.version.cuda} "
      f"device_count={torch.cuda.device_count()} visible={os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"[preflight][rt] vllm={vllm.__version__} transformers={transformers.__version__} pyserini={ver('pyserini')}")
assert torch.cuda.device_count() == 8, f"expected 8 GPUs, got {torch.cuda.device_count()}"
assert vllm.__version__ == "0.19.1", f"expected vllm 0.19.1, got {vllm.__version__}"
assert transformers.__version__ == "4.57.6", f"expected transformers 4.57.6, got {transformers.__version__}"
assert ver("pyserini") == "2.0.0", f"expected pyserini 2.0.0, got {ver('pyserini')}"
PY

# ---- 2b. preflight: vLLM 是否注册 Qwen3_5ForConditionalGeneration（关键风险，无法本地验证）----
echo "[preflight][arch] probing vLLM ModelRegistry for Qwen3_5ForConditionalGeneration"
python3 - <<'PY'
import sys
arch = "Qwen3_5ForConditionalGeneration"
try:
    from vllm.model_executor.models.registry import ModelRegistry
    # vLLM 0.19.1: get_supported_archs() 返回已注册架构名集合
    supported = set(ModelRegistry.get_supported_archs())
    if arch in supported:
        print(f"[preflight][arch] SUPPORTED: {arch} is registered in vLLM ModelRegistry")
    else:
        # 列出最接近的 qwen 架构，便于诊断
        qwen_archs = sorted(a for a in supported if "qwen" in a.lower() or "Qwen" in a)
        print(f"[preflight][arch] NOT SUPPORTED: {arch} is NOT registered")
        print(f"[preflight][arch] qwen-family registered archs: {qwen_archs[:20]}")
        print(f"[preflight][arch] total registered archs: {len(supported)}")
        sys.exit(5)
except SystemExit:
    raise
except Exception as e:
    # ModelRegistry API 可能因 vLLM 版本差异不同；退到扫描 registry 模块
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
            print("[preflight][arch] fallback scan: no Qwen3_5 reference found in vllm.model_executor.models")
            sys.exit(5)
        print("[preflight][arch] SUPPORTED (via fallback scan)")
    except SystemExit:
        raise
    except Exception as e2:
        print(f"[preflight][arch] fallback scan also failed: {type(e2).__name__}: {e2}")
        sys.exit(5)
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

# ---- 3b. ANSERINI_CLASSPATH 指向 pyserini 包内 resources/jars（不能指向包根）----
export ANSERINI_CLASSPATH="$(python3 -c 'import pyserini,os;print(os.path.join(os.path.dirname(pyserini.__file__),"resources","jars"))')"
echo "[preflight][jvm] ANSERINI_CLASSPATH=${ANSERINI_CLASSPATH}"
test -d "${ANSERINI_CLASSPATH}" && echo "[preflight][jvm] anserini jar dir OK" \
  || { echo "[preflight][jvm] ANSERINI_CLASSPATH jar dir MISSING" >&2; exit 2; }

# ---- 4. preflight: 离线凭证占位 + 导入链（harness ultra_core/tools）----
# 必须在首次 import pyserini.search.lucene 之前设 OPENAI_API_KEY 占位（pyserini.encode._openai 模块加载即实例化）。
: "${OPENAI_API_KEY:=**-*****-*******}"
: "${CHROMA_API_KEY:=local-offline}"
: "${CHROMA_DATABASE:=local-offline}"
export OPENAI_API_KEY CHROMA_API_KEY CHROMA_DATABASE
export HARNESS1_FORBID_CHROMA=1   # local_bm25 路径 worker 内部也会设；这里 shell 层先保底
echo "[preflight][import] structlog/chromadb/pyserini/peft/openai_harmony + harness import chain"
export BCPLUS_INDEX_LUCENE="${BCPLUS_INDEX}"
PYTHONPATH="${TRIM_ROOT}:${EASYOPD_ROOT}" python3 - <<'PY'
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
for p in "${JFS_ROOT}" "${TRIM_ROOT}" "${EASYOPD_ROOT}" "${MODEL_PATH}" "${VERIFY_MODEL_PATH}" \
         "${BCPLUS_CORPUS}" "${BCPLUS_INDEX}" \
         "${TRIM_ROOT}/scripts/run_eval.py" "${EVAL_SCRIPT}" \
         "${TIKTOKEN_ZIP}"; do
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
test -f "${VERIFY_MODEL_PATH}/config.json" \
  && echo "[preflight][fs] verify model config.json OK" \
  || { echo "[preflight][fs] verify model config.json MISSING" >&2; exit 2; }

# ---- 6. preflight: tiktoken 离线词表（harness 链 import openai_harmony 用 o200k_base）----
# Qwen3.5 自身用 Qwen2Tokenizer（自带 tokenizer.json），不需要 harmony tiktoken；
# 但 harness 导入链仍 import openai_harmony，其 Rust 层读 TIKTOKEN_ENCODINGS_BASE 加载
# o200k_base.tiktoken，CloudML 无外网必崩 HarmonyError。故 bundle 仍需解压。
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

# ---- 7. eval 环境变量 ----
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="${TRIM_ROOT}:${EASYOPD_ROOT}"
export TRIM_GPU_KEEPALIVE=0
# all 布局：8 卡全部利用（参考 gpt-oss-20B all 成功经验 7 actor + 1 verify = 8 卡全载）。
# 7 个 Qwen3.5-27B actor (TP=1, GPU 0..6, ports 8042/8044/8046/8048/8050/8052/8054)
# + harness-1 verifier (GPU 7, port 8056, 避开 actor 占用的 8050) + 64 个 API-client worker。
# Qwen3.5-27B dense 权重 51.75GiB；单卡 gm402-80g 需 GPU_UTIL=0.90（~72GiB）放得下权重+KV
# （CLAUDE.md 经验 + zero 成功口径：实测单卡 67/80GiB 已充分利用显存）。
export RUN_ID="${RUN_ID:-20260915_memqa_general_v5_bcplus_full_qwen35_27b_all_nothink_gpu8_recovery_v2_64k}"
export GPU_UTIL="${GPU_UTIL:-0.90}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-6}"
export QWEN35_NO_THINKING="${QWEN35_NO_THINKING:-1}"   # 1 = 关闭 thinking（模型 chat_template 默认值已改）
export QWEN35_MAX_MODEL_LEN="${QWEN35_MAX_MODEL_LEN:-65536}"   # 32768→65536 消除 BC+ 长轨迹 400 infra_error（v2_64k）
export ALL_TP="${ALL_TP:-7}"
export ALL_EVAL_TP="${ALL_EVAL_TP:-64}"
export ALL_ACTOR_GPUS="${ALL_ACTOR_GPUS:-0,1,2,3,4,5,6}"
export ALL_ACTOR_PORTS="${ALL_ACTOR_PORTS:-8042,8044,8046,8048,8050,8052,8054}"
export ALL_VERIFY_GPU="${ALL_VERIFY_GPU:-7}"
export ALL_VERIFY_PORT="${ALL_VERIFY_PORT:-8056}"
export WORKER_STAGGER="${WORKER_STAGGER:-0}"
export MAX_TURNS="${MAX_TURNS:-40}"

echo "[env] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[env] PYTHONPATH=${PYTHONPATH}"
echo "[env] HARNESS1_FORBID_CHROMA=${HARNESS1_FORBID_CHROMA} (local_bm25, no Chroma server)"
echo "[env] ANSERINI_CLASSPATH=${ANSERINI_CLASSPATH}"
echo "[env] OPENAI_API_KEY set (placeholder, offline)"
echo "[env] TIKTOKEN_ENCODINGS_BASE=${TIKTOKEN_ENCODINGS_BASE}"
echo "[env] GPU_UTIL=${GPU_UTIL} VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS} QWEN35_NO_THINKING=${QWEN35_NO_THINKING} QWEN35_MAX_MODEL_LEN=${QWEN35_MAX_MODEL_LEN:-unset}"
echo "[env] ALL_TP=${ALL_TP} ALL_EVAL_TP=${ALL_EVAL_TP} ALL_ACTOR_GPUS=${ALL_ACTOR_GPUS} ALL_VERIFY_GPU=${ALL_VERIFY_GPU}"
echo "[env] model_path=${MODEL_PATH} verify_path=${VERIFY_MODEL_PATH}"
echo "[env] run_id=${RUN_ID}"

# ---- 8. 启动 Qwen3.5-27B ALL eval（6 actor + harness-1 verify，no-thinking）----
echo "=================================================================="
echo "[eval] cd ${TRIM_ROOT} && bash scripts/run_bcplus_w_harness1_qwen35_27b_all_nothink_gpu8_eval.sh"
echo "=================================================================="
cd "${TRIM_ROOT}"

exec bash scripts/run_bcplus_w_harness1_qwen35_27b_all_nothink_gpu8_eval.sh
