#!/usr/bin/env bash
# CloudML 启动脚本：TRIM Harness-1 BC+ full / Qwen3-4B-Instruct-2507 / 多节点环境探针
# 资源：2 节点 × 8x gm402-80g = 16 GPU（nodeNumber=2 + RDMA）
# 镜像: micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto
#   （该镜像基础层缺 torch 等依赖，按 CLAUDE.md 经验在启动阶段显式锁定并安装四包配置）
# JuiceFS: volume=songzijun, cluster=bj9-cloudml (endpoint), mount=/mnt/songzijun
#   endpoint 由 CLI --juicefs_mount_configs 显式传入（不在本脚本/yaml 里）。
# 依赖锁定对齐 TRIM Harness-1 BC+ 已验证的 runtime：
#   torch==2.10.0+cu128 / transformers==4.57.6 / vllm==0.19.1 / pyserini 2.0.0 + structlog/chromadb/openai_harmony
#
# 本任务目的：探究如何启动多节点训练，而非完成整个训练任务。
#   RL/trim 训练在框架层是单进程设计（four_cell_runtime._run_four_cell_body 单进程 asyncio 循环
#   + 1 个 vLLM rollout worker 子进程；TP 默认=torch.cuda.device_count()=本节点 GPU 数=8；
#   不读 NNODES/NODE_RANK/RANK/WORLD_SIZE/MASTER_ADDR，不 import ray/verl，RL 路径无 torch.distributed）。
#   因此 2 节点直接 exec run_train.py 会让两节点各自独立跑一遍（重复 + 输出冲突）。
#   本脚本只跑多节点环境探针：依赖 + preflight + 节点身份/RDMA/NCCL/CloudML 注入的分布式 env
#   + torch.distributed.init_process_group 可行性探测，绝不 exec run_train.py。
# 由 ImageCommand 调用；保持幂等、可审计；preflight 全量打印后再进探针。
set -euo pipefail

JFS_ROOT="/mnt/songzijun"
TRIM_ROOT="${JFS_ROOT}/Capability_Evolution/TRIM"
EASYOPD_ROOT="${JFS_ROOT}/Capability_Evolution/SCAPE-EasyOPD"
MODEL_PATH="${JFS_ROOT}/models/Qwen3-4B-Instruct-2507"
RL_DATA="${TRIM_ROOT}/data/harness-1-rl-data"
SEC_CORPUS_ROOT="${TRIM_ROOT}/data/harness-1-sec-corpus"
TIKTOKEN_ZIP="${JFS_ROOT}/tiktoken-bundle.zip"
TIKTOKEN_BUNDLE="/tmp/tiktoken-bundle"
PYPI_INDEX="https://pkgs.d.xiaomi.net/artifactory/api/pypi/pypi-virtual/simple"
PROBE_OUT="${TRIM_ROOT}/outputs/cloudml_2node_16gpu_probe_$(date +%Y%m%d_%H%M%S 2>/dev/null || echo unknown)_$(hostname -s 2>/dev/null || echo host)"

echo "=================================================================="
echo "[boot] TRIM 2-node x 8gpu = 16gpu MULTI-NODE ENV PROBE ($(date -Iseconds 2>/dev/null || echo unknown))"
echo "[boot] host=$(hostname) pwd=$(pwd) user=$(whoami) argv=$*"
echo "[boot] NOTE: probe only — will NOT exec run_train.py"
echo "=================================================================="

export PYTHONUNBUFFERED=1

# ---- 0. 节点身份（最先打印，便于审计多节点到底起了几个 pod）----
echo "[mnode][host] hostname=$(hostname) hostname -s=$(hostname -s 2>/dev/null || true)"
echo "[mnode][host] hostname -I=$(hostname -I 2>/dev/null || true)"
echo "[mnode][host] podname guess from /proc/1/cgroup:" || true
grep -oE '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}' /proc/1/cgroup 2>/dev/null | head -1 || echo "  (no uuid in cgroup)"

# ---- 1. 依赖安装（独立诊断 attempt，按 CLAUDE.md 经验镜像连 torch 都缺）----
# 不要把 torch==2.6.0 与 vllm==0.19.1 一起装，后者要求 torch==2.10.0。
# 基础镜像 distutils blinker 1.4 阻断 pip 卸载，用内部 Xiaomi PyPI + --ignore-installed。
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
  "pyserini" \
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

# ---- 2. preflight: GPU / runtime 版本（任一不符不得进入探针）----
echo "[preflight][gpu] nvidia-smi summary:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv 2>&1 | head -20 || echo "[preflight][gpu] nvidia-smi unavailable"
python3 - <<'PY'
import os, sys
import torch, vllm, transformers
print(f"[preflight][rt] torch={torch.__version__} cuda={torch.version.cuda} "
      f"device_count={torch.cuda.device_count()} visible={os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"[preflight][rt] vllm={vllm.__version__} transformers={transformers.__version__}")
assert torch.cuda.device_count() == 8, f"expected 8 GPUs PER NODE, got {torch.cuda.device_count()}"
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
JAVA_HOME="$(dirname "$(dirname "$(dirname "${JVM_PATH}")")")"
echo "[preflight][jvm] JVM_PATH=${JVM_PATH} JAVA_HOME=${JAVA_HOME}"
export LD_LIBRARY_PATH="$(dirname "${JVM_PATH}"):${LD_LIBRARY_PATH:-}"
python3 -c 'from jnius import autoclass; print("[preflight][jvm] java_ok", autoclass("java.lang.String"))'

# ---- 4. preflight: 导入链（harness.ultra_core / harness.tools 关键依赖）----
echo "[preflight][import] structlog/chromadb/pyserini/peft/openai_harmony"
python3 - <<'PY'
import structlog, chromadb, pyserini, peft
print(f"[preflight][import] structlog={structlog.__version__} peft={peft.__version__}")
import openai_harmony
print("[preflight][import] openai_harmony OK")
PY

# ---- 5. preflight: JuiceFS 挂载与关键路径（本节点视角）----
echo "[preflight][fs] JuiceFS mount check:"
mount | grep -E "songzijun|JuiceFS" || echo "[preflight][fs] no songzijun mount line (may be CSI)"
for p in "${JFS_ROOT}" "${TRIM_ROOT}" "${EASYOPD_ROOT}" "${MODEL_PATH}" "${RL_DATA}" \
         "${SEC_CORPUS_ROOT}/corpora/sec/train" "${SEC_CORPUS_ROOT}/indexes/bm25" \
         "${TRIM_ROOT}/scripts/run_train.py"; do
  if [ -e "$p" ]; then
    echo "[preflight][fs] OK  $p"
  else
    echo "[preflight][fs] MISSING  $p" >&2
    exit 2
  fi
done

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
export TIKTOKEN_RS_CACHE_DIR="${TIKTOKEN_BUNDLE}/tiktoken_rs_cache"
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_BUNDLE}/tiktoken_cache"

# ---- 7. 离线凭证占位 ----
: "${OPENAI_API_KEY:=**-*****-*******}"
: "${CHROMA_API_KEY:=local-offline}"
: "${CHROMA_DATABASE:=local-offline}"
export OPENAI_API_KEY CHROMA_API_KEY CHROMA_DATABASE

# ---- 8. 训练环境变量（与单节点 baseline 对齐；CUDA_VISIBLE_DEVICES 仍为本节点 8 卡）----
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MAX_TURNS=40
export PYTHONPATH="${TRIM_ROOT}:${EASYOPD_ROOT}"
export SCAPE_SEC_RL_DATA="${RL_DATA}"
export SCAPE_SEC_CORPUS_ROOT="${SEC_CORPUS_ROOT}"
export DISABLE_CUSTOM_ALL_REDUCE=1
export SCAPE_LOCAL_OPENAI_EMBEDDINGS=1
export SCAPE_FORCE_LOCAL_HARMONY=1
export SCAPE_CHROMA_PATH=/tmp/empty_chroma
mkdir -p "$SCAPE_CHROMA_PATH"

echo "[env] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[env] MAX_TURNS=${MAX_TURNS}"
echo "[env] PYTHONPATH=${PYTHONPATH}"
echo "[env] DISABLE_CUSTOM_ALL_REDUCE=${DISABLE_CUSTOM_ALL_REDUCE}"

# ==================================================================
# 9. 多节点环境探针（核心目的）
# ==================================================================
mkdir -p "${PROBE_OUT}"

echo "=================================================================="
echo "[probe] multi-node environment discovery"
echo "=================================================================="

echo "[probe][env] all distributed-relevant env vars (host=$(hostname)):"
env | grep -iE 'NODE|RANK|WORLD|MASTER|RDZV|CML_|NNODE|NCCL|RDMA|IB|OFED|TORCH|MI_|XCCL|MPI' | sort || echo "  (none matched)"

echo "[probe][host] full identity:"
echo "  hostname=$(hostname)"
echo "  hostname -s=$(hostname -s 2>/dev/null || true)"
echo "  hostname -I=$(hostname -I 2>/dev/null || true)"
echo "  hostname --all-ip-addresses=$(hostname --all-ip-addresses 2>/dev/null || true)"

echo "[probe][gpu] nvidia-smi -L (per-node GPU list):"
nvidia-smi -L

echo "[probe][rdma] infiniband / roce devices:"
ibv_devinfo 2>/dev/null || echo "[probe][rdma] ibv_devinfo unavailable (no RDMA userland verbs or not installed)"
echo "[probe][rdma] ibstat (if present):"
ibstat 2>/dev/null | head -40 || echo "[probe][rdma] ibstat unavailable"
echo "[probe][rdma] /dev/infiniband:"
ls -la /dev/infiniband/ 2>/dev/null || echo "[probe][rdma] no /dev/infiniband"
echo "[probe][net] ip -o link (NICs):"
ip -o link 2>/dev/null | head -30 || echo "[probe][net] ip unavailable"

echo "[probe][framework] does CloudML inject anything the framework could consume?"
python3 - <<'PY'
import os, torch
print(f"[probe][fw] torch.cuda.device_count()={torch.cuda.device_count()} (框架默认 TP 取此值=本节点 GPU 数)")
keys = [
    "NNODES", "NODE_RANK", "NODE_INDEX", "NODE_NUMBER",
    "RANK", "LOCAL_RANK", "WORLD_SIZE", "GROUP_WORLD",
    "MASTER_ADDR", "MASTER_PORT",
    "CML_NODE_INDEX", "CML_NODE_NUMBER", "CML_NODE_RANK", "CML_WORKER_INDEX",
    "CML_TASK_INDEX", "CML_JOB_NAME", "CML_FRAMEWORK", "CML_RDMA",
    "NCCL_SOCKET_IFNAME", "NCCL_DEBUG", "NCCL_IB_DISABLE",
    "RDZV_BACKEND", "RDZV_ENDPOINT", "RDZV_ID",
    "OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMI_SIZE",
    "TORCH_DISTRIBUTED_DEBUG", "TORCHELASTIC_*",
]
present = {}
for k in keys:
    v = os.environ.get(k)
    if v is not None:
        present[k] = v
print(f"[probe][fw] present distributed env keys ({len(present)}):")
for k in sorted(present):
    print(f"  {k}={present[k]}")
if not present:
    print("[probe][fw] !! NONE of NNODES/NODE_RANK/RANK/WORLD_SIZE/MASTER_ADDR/... are set")
    print("[probe][fw] !! => CloudML pytorch multi-node does NOT auto-inject torchrun-style env")
    print("[probe][fw] !! => 框架若要跨节点需自带 torchrun/ray launcher（当前 RL 路径不带）")
PY

echo "[probe][torch] attempting torch.distributed.init_process_group with env ONLY (no manual set):"
python3 - <<'PY'
import os, traceback
print(f"[probe][ddp] rank={os.environ.get('RANK')} world_size={os.environ.get('WORLD_SIZE')} "
      f"master_addr={os.environ.get('MASTER_ADDR')} master_port={os.environ.get('MASTER_PORT')}")
if not (os.environ.get("RANK") is not None and os.environ.get("MASTER_ADDR") is not None):
    print("[probe][ddp] !! RANK/MASTER_ADDR not injected — torch.distributed cannot init from env alone")
    print("[probe][ddp] !! confirms: a real multi-node train needs an explicit launcher (torchrun/ray) OR framework code change")
else:
    try:
        import torch.distributed as dist
        dist.init_process_group(backend="nccl")
        print(f"[probe][ddp] init_process_group OK: rank={dist.get_rank()} world_size={dist.get_world_size()}")
        dist.barrier()
        print(f"[probe][ddp] barrier passed on rank={dist.get_rank()}")
        dist.destroy_process_group()
    except Exception as e:
        print(f"[probe][ddp] init_process_group FAILED: {e!r}")
        traceback.print_exc()
PY

echo "[probe][nccl] quick NCCL local (8-gpu) all-reduce sanity (per-node):"
python3 - <<'PY'
import os, torch, torch.distributed as dist
try:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29512")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    dist.init_process_group(backend="nccl")
    t = torch.ones(8, device="cuda")
    dist.all_reduce(t)
    print(f"[probe][nccl] local all_reduce OK -> sum={t[0].item()} (expect 8) on rank0")
    dist.destroy_process_group()
except Exception as e:
    print(f"[probe][nccl] local all_reduce FAILED: {e!r}")
PY

echo "[probe][done] writing PROBE_OK marker (host=$(hostname))"
echo "PROBE_OK host=$(hostname) ts=$(date -Iseconds 2>/dev/null || echo unknown)" > "${PROBE_OUT}/PROBE_OK"
echo "[probe][done] marker at ${PROBE_OUT}/PROBE_OK"
echo "[probe][done] !! intentionally NOT exec-ing run_train.py — probe complete"
echo "PROBE_DONE $(hostname)"

# 保持容器短暂存活以便审计 logs，随后正常退出
exit 0
