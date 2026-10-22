#!/usr/bin/env bash
# CloudML 启动脚本：TRIM Harness-1 BC+ full / Gemma-3-27B-it / ZERO eval / 单节点 8x gm402-80g
# 提交日期 2026-09-17，标签 MemQA-General-V5，priority 8，queue 2334 (gm402-80g)。
#
# Forked from run_gemma3_27b_zero_eval_cloudml_8gpu.sh (9/15 四件套口径 zero boot)
# + boot_gemma3_27b_all_eval_cloudml_8gpu_20260915.sh (9/15 all boot 结构)。本脚本把
# runtime 口径从 CLAUDE.md 四件套 (torch 2.10.0+cu128 / vllm 0.19.1 / transformers 4.57.6 /
# pyserini 2.0.0) 切换为 bishop 口径 (torch 2.11.0+cu130 / vllm 0.25.1 / transformers 5.14.1 /
# pyserini 1.6.0 + jinja2 3.1.6)，与用户在另一台机器跑通 Gemma3-27B-it all eval 的环境一致。
#
# 为何 bishop 口径：HEAD ba7677d 的 GEMMA_PROFILE (trim.eval.model_profiles) 已含
#   --chat-template tool_chat_template_gemma3_pythonic.jinja + tool_call_parser=pythonic
#   + --language-model-only --generation-config vllm（9/17 merge commit 293c2ac 修复 9/15
#   toolcall 失败）。pythonic chat template 是标准 jinja、与 vllm 版本无关；但 Gemma3 VLM
#   架构 (Gemma3ForConditionalGeneration) 的 vLLM 注册 + pythonic parser 在 vllm 0.25.1
#   已被用户实测可用（all eval 跑通），四件套 vllm 0.19.1 对该 VLM 架构的支持未验证。
#   故按用户决策统一用 bishop 口径，保证 zero 与已验证 all 结果同口径可比。
#
# 被测模型 = /mnt/songzijun/models/gemma-3-27b-it
#   config.json 架构 Gemma3ForConditionalGeneration / model_type gemma3
#   / text_config 62 层 hidden 5376 / dense（无 MoE）/ 12 shards ~51.10 GiB bf16
#   / 多模态 VLM（siglip vision encoder），BC+ 走纯文本路径
#   / tokenizer: SentencePiece（tokenizer.model + tokenizer.json，非 harmony o200k）。
#
# Thinking：Gemma3 IT 无 thinking 模式，chat_template 无 enable_thinking 守卫、不输出
#   独立 reasoning 块。GEMMA_PROFILE 无 reasoning_parser，无需 strip。不动 scorer/runner。
#
# 镜像: micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto
#   （基础层缺 torch 等依赖，在启动阶段显式锁定并安装 bishop 口径四包 + 辅助依赖）
# JuiceFS: volume=songzijun, JuiceFSCluster=bj9-cloudml, mount=/mnt/songzijun
#   — JuiceFS 挂载由 CloudML 通过 CLI --juicefs_mount_configs 注入，不在本脚本处理。
# 依赖锁定口径（bishop，与已验证 Gemma3 all run 一致）：
#   torch==2.11.0 / transformers==5.14.1 / vllm==0.25.1 / pyserini==1.6.0 / jinja2==3.1.6
#   + structlog/chromadb/openai_harmony/peft/sentencepiece 等（chz/tinker/json_repair --no-deps
#   保护核心四包）。由 ImageCommand 调用；保持幂等、可审计；preflight 全量打印后再进 eval。
# eval 脚本：TRIM/scripts/run_bcplus_w_harness1_gemma3_27b_zero_cloudml_8gpu_eval.sh
#   （lib-sourced zero wrapper，8 个单卡 Gemma-3-27B-it actor TP=1 GPU 0..7 + 16 个
#    API-client worker, run_eval.py --harness Harness-1 --evaluation-path upstream_api
#    --component zero --tp 16, local_bm25 无 Chroma, 无 verifier。TEMPERATURE=0.0（Gemma3
#    专属：temp 1.0 → 100% implicit_user_text 零搜索）。8 卡全载充分利用显存。）
#
# 关键风险：vLLM 0.25.1 对 Gemma3ForConditionalGeneration 的注册——已在用户 all run 实测
#   可用，本脚本仍保留 ModelRegistry 探测 preflight，不支持则 exit 5 干净退出（保留 attempt）。
set -euo pipefail

JFS_ROOT="/mnt/songzijun"
TRIM_ROOT="${JFS_ROOT}/Capability_Evolution/TRIM"
EASYOPD_ROOT="${JFS_ROOT}/Capability_Evolution/SCAPE-EasyOPD"
MODEL_PATH="${JFS_ROOT}/models/gemma-3-27b-it"
BCPLUS_ROOT="${JFS_ROOT}/Capability_Evolution/SCOPE/external/BrowseComp-Plus"
BCPLUS_CORPUS="${BCPLUS_ROOT}/data/browsecomp_plus_corpus_full.jsonl"
BCPLUS_INDEX="${BCPLUS_ROOT}/indexes/bm25"
TIKTOKEN_ZIP="${JFS_ROOT}/tiktoken-bundle.zip"
TIKTOKEN_BUNDLE="/tmp/tiktoken-bundle"
PYPI_INDEX="https://pkgs.d.xiaomi.net/artifactory/api/pypi/pypi-virtual/simple"
EVAL_SCRIPT="${TRIM_ROOT}/scripts/run_bcplus_w_harness1_gemma3_27b_zero_cloudml_8gpu_eval.sh"

echo "=================================================================="
echo "[boot] TRIM Harness-1 BC+ full / Gemma-3-27B-it / ZERO eval / 8gpu / bishop runtime ($(date -Iseconds 2>/dev/null || echo unknown))"
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
# bishop 口径四包：torch 2.11.0+cu130 / vllm 0.25.1 / transformers 5.14.1 / pyserini 1.6.0。
# 注意：vllm 0.25.1 自带 torch 依赖约束，显式 pin torch==2.11.0 避免被降级。
# 基础镜像 distutils blinker 1.4 阻断 pip 卸载，用内部 Xiaomi PyPI + --ignore-installed。
# pyserini 必须显式 pin ==1.6.0（裸 pyserini 可能解析到 2.4.0，preflight 断言 1.6.0 必失败）。
# Gemma3 tokenizer 用 SentencePiece，已显式装 sentencepiece；jinja2 3.1.6 对齐 bishop。
# 验证（2026-09-17）：torch 2.11.0 / vllm 0.25.1 / transformers 5.14.1 / pyserini 1.6.0 /
#   jinja2 3.1.6 在 Xiaomi PyPI 镜像均有（torch 镜像最高 2.14.0、vllm 0.29.0、transformers 5.17.0）；
#   注意 pip index versions 对该镜像误报 No matching，须用 curl simple index 验证（见记忆）。
echo "[deps] installing locked runtime (bishop 口径) from ${PYPI_INDEX}"
python3 -m pip install --no-cache-dir --ignore-installed \
  --upgrade-strategy only-if-needed \
  --index-url "${PYPI_INDEX}" \
  "torch==2.11.0" \
  "transformers==5.14.1" \
  "accelerate" \
  "datasets" \
  "vllm==0.25.1" \
  "hydra-core" \
  "omegaconf" \
  "tensordict" \
  "codetiming" \
  "tqdm" \
  "python-dotenv" \
  "pandas" \
  "pyarrow" \
  "sentencepiece" \
  "jinja2==3.1.6" \
  "pyserini==1.6.0" \
  "structlog" \
  "chromadb" \
  "openai" \
  "tenacity" \
  "anthropic" \
  "peft" \
  "openai_harmony" \
  "pydantic-settings"

# chz/tinker/json_repair 用 --no-deps，避免它们拉取会破坏 bishop 四包的依赖（CLAUDE.md 经验）。
python3 -m pip install --no-cache-dir --no-deps \
  --index-url "${PYPI_INDEX}" \
  "chz" \
  "tinker" \
  "json_repair" || echo "[deps] WARN: chz/tinker/json_repair install had issues (zero eval may not need tinker)"

echo "[deps] printing resolved versions:"
python3 - <<'PY'
import importlib.metadata, torch, vllm, transformers
def ver(p):
    try:
        return importlib.metadata.version(p)
    except Exception:
        return getattr(__import__(p), "__version__", "unknown")
print(f"[deps] torch={torch.__version__} cuda={torch.version.cuda}")
print(f"[deps] vllm={vllm.__version__} transformers={transformers.__version__} pyserini={ver('pyserini')} jinja2={ver('jinja2')}")
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
print(f"[preflight][rt] vllm={vllm.__version__} transformers={transformers.__version__} pyserini={ver('pyserini')} jinja2={ver('jinja2')}")
assert torch.cuda.device_count() == 8, f"expected 8 GPUs, got {torch.cuda.device_count()}"
assert vllm.__version__ == "0.25.1", f"expected vllm 0.25.1, got {vllm.__version__}"
assert transformers.__version__ == "5.14.1", f"expected transformers 5.14.1, got {transformers.__version__}"
assert ver("pyserini") == "1.6.0", f"expected pyserini 1.6.0, got {ver('pyserini')}"
assert ver("jinja2") == "3.1.6", f"expected jinja2 3.1.6, got {ver('jinja2')}"
PY

# ---- 2b. preflight: vLLM 是否注册 Gemma3ForConditionalGeneration（关键风险）----
echo "[preflight][arch] probing vLLM ModelRegistry for Gemma3ForConditionalGeneration"
python3 - <<'PY'
import sys
arch = "Gemma3ForConditionalGeneration"
try:
    from vllm.model_executor.models.registry import ModelRegistry
    # vLLM 0.25.1: get_supported_archs() 返回已注册架构名集合
    supported = set(ModelRegistry.get_supported_archs())
    if arch in supported:
        print(f"[preflight][arch] SUPPORTED: {arch} is registered in vLLM ModelRegistry")
    else:
        gemma_archs = sorted(a for a in supported if "gemma" in a.lower() or "Gemma" in a)
        print(f"[preflight][arch] NOT SUPPORTED: {arch} is NOT registered")
        print(f"[preflight][arch] gemma-family registered archs: {gemma_archs[:20]}")
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
            if "gemma3" in name.lower():
                mod = importlib.import_module(f"vllm.model_executor.models.{name}")
                src = open(mod.__file__).read() if hasattr(mod, "__file__") and mod.__file__ else ""
                if "Gemma3ForConditionalGeneration" in src or "Gemma3" in src:
                    print(f"[preflight][arch] fallback: found Gemma3 ref in vllm.model_executor.models.{name}")
                    found = True
        if not found:
            print("[preflight][arch] fallback scan: no Gemma3 reference found in vllm.model_executor.models")
            sys.exit(5)
        print("[preflight][arch] SUPPORTED (via fallback scan)")
    except SystemExit:
        raise
    except Exception as e2:
        print(f"[preflight][arch] fallback scan also failed: {type(e2).__name__}: {e2}")
        sys.exit(5)
PY

# ---- 2c. preflight: GEMMA_PROFILE 确实带 pythonic chat template（修复 9/15 toolcall 失败的核心）----
echo "[preflight][profile] verifying GEMMA_PROFILE emits pythonic chat-template"
PYTHONPATH="${TRIM_ROOT}:${EASYOPD_ROOT}" python3 - <<'PY'
from trim.eval.model_profiles import GEMMA_PROFILE, classify_profile_by_name
prof = classify_profile_by_name("gemma-3-27b-it")
assert prof is GEMMA_PROFILE, "gemma-3-27b-it should classify to GEMMA_PROFILE"
assert prof.tool_call_parser == "pythonic", f"expected pythonic parser, got {prof.tool_call_parser}"
extra = list(prof.extra_vllm_args)
assert "--chat-template" in extra, f"GEMMA_PROFILE must carry --chat-template (fix for 9/15 toolcall fail); got {extra}"
tpl = extra[extra.index("--chat-template") + 1]
import os
assert os.path.isfile(tpl), f"pythonic chat template missing: {tpl}"
print(f"[preflight][profile] GEMMA_PROFILE OK: parser={prof.tool_call_parser} chat_template={tpl}")
print(f"[preflight][profile] full extra_vllm_args: {' '.join(extra)}")
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
# pyserini 1.6.0: num_docs 取值兼容（属性或方法）—— getattr 优先，回退调用。
_nd = getattr(searcher, "num_docs", None)
n = int(_nd() if callable(_nd) else (_nd or 0))
hits = searcher.search("test", k=1)
print(f"[preflight][lucene] index={idx} num_docs={n} probe_hits={len(hits)}")
assert n > 0
PY

# ---- 5. preflight: JuiceFS 挂载与关键路径 ----
echo "[preflight][fs] JuiceFS mount check:"
mount | grep -E "songzijun|JuiceFS" || echo "[preflight][fs] no songzijun mount line (may be CSI)"
for p in "${JFS_ROOT}" "${TRIM_ROOT}" "${EASYOPD_ROOT}" "${MODEL_PATH}" \
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

# ---- 6. preflight: tiktoken 离线词表（harness 链 import openai_harmony 用 o200k_base）----
# Gemma3 自身用 SentencePiece tokenizer（tokenizer.model + tokenizer.json），不需要 harmony
# tiktoken；但 harness 导入链仍 import openai_harmony，其 Rust 层读 TIKTOKEN_ENCODINGS_BASE
# 加载 o200k_base.tiktoken，CloudML 无外网必崩 HarmonyError。故 bundle 仍需解压。
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
# zero 布局（充分利用 8 卡显存）：8 个 Gemma-3-27B-it actor (TP=1, GPU 0..7) + 16 worker。
# 8 张卡全载模型（每卡 1 个 TP=1 actor），无 verifier（zero component）。对齐 Qwen3.5-27B
# zero 成功 task t-20260914151008-fcanl 的 8-actor full-load 布局，吞吐 8x 于单 actor。
# 显存/并发参数（27B dense 权重 51.10GiB，单卡 80g + GPU_UTIL 0.90 ~72GiB 装权重+KV）：
#   GPU_UTIL=0.90   — 纯推理 eval 无 backward OOM 风险；27B 权重大，提 KV pool 利吞吐。
#   TP=16            — 16 workers → 2 per GPU（8 actor），与已验证 Qwen3.5-27B zero 一致。
#   MAX_NUM_SEQS=8  — 单卡 27B：权重 51GiB + KV budget ~20GiB，MAX_NUM_SEQS=8 → KV ~16GiB，安全余量足。
#   MAX_MODEL_LEN=32768（profile 默认，由 vllm_extra_for_model 产出；GEMMA3_MAX_MODEL_LEN 留空不覆盖）。
export RUN_ID="${RUN_ID:-20260917_memqa_general_v5_bcplus_full_gemma3_27b_zero_gpu8}"
export TP="${TP:-16}"
export GPU_UTIL="${GPU_UTIL:-0.90}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-8}"
export WORKER_STAGGER="${WORKER_STAGGER:-10}"
export GEMMA3_MAX_MODEL_LEN="${GEMMA3_MAX_MODEL_LEN:-}"
# 8-actor full-load 布局（不走 lib 的单 actor zero；本 wrapper 自管 8 actor）。
export ZERO_ACTOR_GPUS="${ZERO_ACTOR_GPUS:-0,1,2,3,4,5,6,7}"
export ZERO_ACTOR_PORTS="${ZERO_ACTOR_PORTS:-8040,8042,8044,8046,8048,8050,8052,8054}"

echo "[env] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[env] PYTHONPATH=${PYTHONPATH}"
echo "[env] HARNESS1_FORBID_CHROMA=${HARNESS1_FORBID_CHROMA} (local_bm25, no Chroma server)"
echo "[env] ANSERINI_CLASSPATH=${ANSERINI_CLASSPATH}"
echo "[env] OPENAI_API_KEY set (placeholder, offline)"
echo "[env] TIKTOKEN_ENCODINGS_BASE=${TIKTOKEN_ENCODINGS_BASE}"
echo "[env] TP=${TP} GPU_UTIL=${GPU_UTIL} VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS} WORKER_STAGGER=${WORKER_STAGGER}"
echo "[env] ZERO_ACTOR_GPUS=${ZERO_ACTOR_GPUS} ZERO_ACTOR_PORTS=${ZERO_ACTOR_PORTS} (8-card full load)"
echo "[env] model_path=${MODEL_PATH} run_id=${RUN_ID}"
echo "[env] thinking=off (Gemma3 IT has no thinking mode; nothing to strip)"
echo "[env] runtime=bishop (torch 2.11.0+cu130 / vllm 0.25.1 / transformers 5.14.1 / pyserini 1.6.0)"

# ---- 8. 启动 Gemma-3-27B-it ZERO eval（8 单卡 actor + 16 worker，8 卡全载）----
echo "=================================================================="
echo "[eval] cd ${TRIM_ROOT} && bash scripts/run_bcplus_w_harness1_gemma3_27b_zero_cloudml_8gpu_eval.sh"
echo "=================================================================="
cd "${TRIM_ROOT}"

exec bash scripts/run_bcplus_w_harness1_gemma3_27b_zero_cloudml_8gpu_eval.sh
