#!/usr/bin/env bash
# Resume Harness-1 BC+ full RL (CISPO) GPT-OSS-20B (original mxfp4) via
# torch_ddp_lora on FOUR 8-GPU gm402-80g nodes, from step 102 (latest ckpt of
# 7nzzn, which failed at step102 with exit137 OOM-kill on 3 pods + exit1 on
# worker-2). Same config as 7nzzn, only adds --resume (auto-loads latest=step102
# adapter/optimizer/sampler/RNG/step, continues to step 200).
#
# --resume (launch.py:491 store_true) auto-finds latest checkpoint in --out.
# OUT must point at 7nzzn's output dir so latest -> step_000102 is found.
# Multi-node resume: run_train_multinode.py --nnodes 4 --nproc-per-node 8
# self-torchruns, each worker's run_parsed_train loads --resume from --out.
set -euxo pipefail

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TRIM_GPU_KEEPALIVE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/mnt/songzijun/Capability_Evolution/TRIM:/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD

NODE_RANK="${NODE_RANK:-${PET_NODE_RANK:-${NODE_INDEX:-${NODE_NUMBER:-}}}}"
if [ -z "${NODE_RANK:-}" ]; then
  echo "FATAL: NODE_RANK is not set." >&2; exit 1
fi
if [ -z "${MASTER_ADDR:-}" ]; then
  echo "FATAL: MASTER_ADDR is not set." >&2; exit 1
fi
: "${MASTER_PORT:=29500}"
export MASTER_ADDR NODE_RANK MASTER_PORT

unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK \
      GROUP_WORLD_SIZE TORCHELASTIC_RUN_ID TORCHELASTIC_RESTART_COUNT \
      TORCHELASTIC_MAX_RESTARTS TORCHELASTIC_USE_AGENT_STORE \
      PET_MASTER_ADDR PET_MASTER_PORT PET_NNODES PET_NPROC_PER_NODE PET_NODE_RANK

echo "MULTINODE_ENV master_addr=${MASTER_ADDR} node_rank=${NODE_RANK} master_port=${MASTER_PORT} host=$(hostname) RESUME"

if [ -z "${OUT:-}" ]; then
  echo "FATAL: OUT is not set. Resume requires the SAME --out as 7nzzn (latest->step102)." >&2
  exit 1
fi
export OUT
echo "MULTINODE_OUT out=${OUT} host=$(hostname) node_rank=${NODE_RANK}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-**-****-***********-***-*-****-***}"
export SCAPE_LOCAL_OPENAI_EMBEDDINGS=1
export SCAPE_CHROMA_PATH=/tmp/empty_chroma
mkdir -p "$SCAPE_CHROMA_PATH"

python3 -c "import zipfile; zipfile.ZipFile('/mnt/songzijun/tiktoken-bundle.zip').extractall('/tmp')"
test -f /tmp/tiktoken-bundle/share/tiktoken/o200k_base.tiktoken
test -f /tmp/tiktoken-bundle/share/tiktoken/cl100k_base.tiktoken
export TIKTOKEN_ENCODINGS_BASE=/tmp/tiktoken-bundle/share/tiktoken
export TIKTOKEN_RS_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_rs_cache
export TIKTOKEN_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_cache

nvidia-smi -L
test -f /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_train_multinode.py
test -f /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_train.py
test -f /mnt/songzijun/models/openai/gpt-oss-20b/config.json
test -f /mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-rl-data/manifest.json
test -f /mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-sec-corpus/manifest.json
# resume: assert latest ckpt + step100/102 adapter present (resume target)
test -d "$OUT/checkpoints/rl/latest"
test -f "$OUT/checkpoints/rl/latest/STEP_COMPLETE"
echo "RESUME_TARGET latest=$(readlink -f "$OUT/checkpoints/rl/latest")"

JVM_PATH=$(find /usr /opt -type f -name libjvm.so 2>/dev/null | head -1 || true)
if [ -z "${JVM_PATH}" ]; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends openjdk-21-jre-headless
  JVM_PATH=$(find /usr /opt -type f -name libjvm.so 2>/dev/null | head -1 || true)
fi
test -n "${JVM_PATH}"; test -f "${JVM_PATH}"; ldd "${JVM_PATH}"
export JVM_PATH
export JAVA_HOME=$(dirname "$(dirname "$(dirname "${JVM_PATH}")")")
export LD_LIBRARY_PATH=$(dirname "${JVM_PATH}"):${LD_LIBRARY_PATH:-}

python3 -m pip install --no-cache-dir --ignore-installed --upgrade-strategy only-if-needed \
  "torch==2.10.0" "transformers==4.57.6" "accelerate==1.13.0" "datasets==4.8.4" \
  "hydra-core==1.3.2" "omegaconf==2.3.0" "vllm==0.19.1" peft structlog chromadb \
  tqdm regex sympy latex2sympy2 word2number sentencepiece pyserini openai tenacity anthropic

PYSERINI_ROOT=$(python3 -c 'import pathlib, pyserini; print(pathlib.Path(pyserini.__file__).resolve().parent)')
export ANSERINI_CLASSPATH=$(find "$PYSERINI_ROOT/resources/jars" -type f -name '*.jar' -print | paste -sd: -)
test -n "$ANSERINI_CLASSPATH"; export CLASSPATH="$ANSERINI_CLASSPATH"

python3 - <<'PY'
import importlib.metadata, json, os, torch, transformers, vllm
from jnius import autoclass; autoclass("java.lang.String")
v={"torch":torch.__version__,"vllm":vllm.__version__,"transformers":transformers.__version__,
   "pyserini":importlib.metadata.version("pyserini"),"gpu_count":torch.cuda.device_count(),
   "JVM_PATH":os.environ["JVM_PATH"],"java_ok":True,"master_addr":os.environ.get("MASTER_ADDR"),"node_rank":os.environ.get("NODE_RANK")}
print("PREFLIGHT",v,flush=True)
assert v["torch"].startswith("2.10.0") and v["vllm"]=="0.19.1" and v["transformers"]=="4.57.6" and v["pyserini"]=="2.0.0"
assert v["gpu_count"]==8
from pyserini.search.lucene import LuceneSearcher
s=LuceneSearcher("/mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-sec-corpus/indexes/bm25")
n=s.num_docs if not callable(s.num_docs) else s.num_docs(); print("LUCENE_OK",{"num_docs":n},flush=True); assert n>0
from trim.integrations.verl.trainer_adapter import _actor_wrap
assert _actor_wrap("torch_ddp_lora")=="ddp"
import pathlib,re
ta=pathlib.Path("/mnt/songzijun/Capability_Evolution/TRIM/trim/integrations/verl/trainer_adapter.py").read_text()
assert "startup_timeout_s=3600.0" in ta and "EADDRINUSE" in ta
assert re.search(r"train_rows,\s*eval_rows,\s*pool_meta,\s*_?frozen_points\s*=\s*resolve_queries",ta)
print("PATCH_CHECK_OK",flush=True)
PY

# Resume keeps model local copy (same as 7nzzn): each node cp to /tmp.
MODEL_SRC=/mnt/songzijun/models/openai/gpt-oss-20b
MODEL_LOCAL=/tmp/gpt-oss-20b
if [ ! -f "$MODEL_LOCAL/config.json" ]; then
  rm -rf "$MODEL_LOCAL"; mkdir -p "$MODEL_LOCAL"; cp -r "$MODEL_SRC"/. "$MODEL_LOCAL"/
fi
test -f "$MODEL_LOCAL/config.json"; echo "MODEL_LOCAL_READY $MODEL_LOCAL host=$(hostname) node_rank=${NODE_RANK}"

python3 -m py_compile /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_train_multinode.py
python3 -m py_compile /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_train.py
cd /mnt/songzijun/Capability_Evolution
exec python3 /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_train_multinode.py \
  --nnodes 4 \
  --nproc-per-node 8 \
  --node-rank "${NODE_RANK}" \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  --dist-backend nccl \
  --rdzv-backend static \
  --rdzv-id gptoss20b-rl-32 \
  --training-backend torch_ddp_lora \
  --harness Harness-1 \
  --benchmark bcplus_full \
  --model_name /tmp/gpt-oss-20b \
  --train_method rl \
  --component zero \
  --train-env local_legacy \
  --train-data sec \
  --rl-data /mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-rl-data \
  --sec-corpus-root /mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-sec-corpus \
  --train-steps 200 \
  --train-groups-per-step 32 \
  --group-size 8 \
  --train-micro-batch-size 1 \
  --train-heartbeat-every 8 \
  --max-turns 40 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --resume \
  --out "${OUT}"
