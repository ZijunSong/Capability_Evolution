#!/usr/bin/env bash
set -euo pipefail

# Prepared 2026-09-10; NOT SUBMITTED. Recovery2 of t-20260910204048-va5zw (recovery1).
#
# recovery1 (t-20260910204048-va5zw) fixed train_env=upstream/chz by adding
# --train-env local_legacy, passed preflight + vLLM TP=8 init, but failed in the
# first rollout: AttributeError '_LocalConversation' has no attribute 'to_json'.
# Root cause: SCAPE_FORCE_LOCAL_HARMONY=1 (inherited from the old trim-eval
# template) forces harness/ultra_core.py:31 to bind Conversation =
# _LocalConversation (the forbidden _LocalHarmonyEncodingFallback), which lacks
# to_json(), breaking openai_harmony.render_conversation_for_completion. The
# successful eval runs do NOT set this flag. Recovery2 drops it entirely so the
# real openai_harmony Rust encoder + Conversation load offline (tiktoken bundle
# wired via TIKTOKEN_ENCODINGS_BASE/RS_CACHE_DIR/CACHE_DIR), keeping
# --train-env local_legacy from recovery1.
#
# Submits a TRIM Harness-1 BC+ full GPT-OSS-20B training job (train_method=trim,
# component=all) to CloudML workspace 10063 / queue 2334 on one 8-GPU gm402-80g
# node (GUARANTEED_PUBLIC). JuiceFS volume `songzijun` (cluster bj9-cloudml) is
# mounted read-write at /mnt/songzijun through the DIRECT CLI
# --juicefs_mount_configs argument so endpoint=bj9-cloudml is explicit in the
# API request. The YAML intentionally omits the mount; do NOT switch to --filename.

IMAGE_COMMAND=$(cat <<'CLOUDML_IMAGE_COMMAND'
set -euxo pipefail
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MAX_TURNS=40
export PYTHONPATH=/mnt/songzijun/Capability_Evolution/TRIM:/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD
export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1
export OPENAI_API_KEY="${OPENAI_API_KEY:-*****-*******-***********}"
export SCAPE_LOCAL_OPENAI_EMBEDDINGS=1
export SCAPE_CHROMA_PATH=/tmp/empty_chroma
mkdir -p "$SCAPE_CHROMA_PATH"

# Offline tiktoken: extract the bundle to /tmp so harmony_runtime can
# auto-discover it, and also set the env vars explicitly (bundle README "方式 B").
python3 -c "import zipfile; zipfile.ZipFile('/mnt/songzijun/tiktoken-bundle.zip').extractall('/tmp')"
test -f /tmp/tiktoken-bundle/share/tiktoken/o200k_base.tiktoken
test -f /tmp/tiktoken-bundle/share/tiktoken/cl100k_base.tiktoken
export TIKTOKEN_ENCODINGS_BASE=/tmp/tiktoken-bundle/share/tiktoken
export TIKTOKEN_RS_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_rs_cache
export TIKTOKEN_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_cache

nvidia-smi -L
test -f /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_train.py
test -f /mnt/songzijun/models/openai/gpt-oss-20b/config.json
test -f /mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-rl-data/manifest.json
test -f /mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-sec-corpus/manifest.json

JVM_PATH=$(find /usr /opt -type f -name libjvm.so 2>/dev/null | head -1 || true)
if [ -z "${JVM_PATH}" ]; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends openjdk-21-jre-headless
  JVM_PATH=$(find /usr /opt -type f -name libjvm.so 2>/dev/null | head -1 || true)
fi
test -n "${JVM_PATH}"
test -f "${JVM_PATH}"
ldd "${JVM_PATH}"
export JVM_PATH
export JAVA_HOME=$(dirname "$(dirname "$(dirname "${JVM_PATH}")")")
export LD_LIBRARY_PATH=$(dirname "${JVM_PATH}"):${LD_LIBRARY_PATH:-}

python3 -m pip install --no-cache-dir --ignore-installed --upgrade-strategy only-if-needed \
  "torch==2.10.0" "transformers==4.57.6" "accelerate==1.13.0" "datasets==4.8.4" \
  "hydra-core==1.3.2" "omegaconf==2.3.0" "vllm==0.19.1" peft structlog chromadb \
  tqdm regex sympy latex2sympy2 word2number sentencepiece pyserini openai tenacity anthropic

PYSERINI_ROOT=$(python3 -c 'import pathlib, pyserini; print(pathlib.Path(pyserini.__file__).resolve().parent)')
export ANSERINI_CLASSPATH=$(find "$PYSERINI_ROOT/resources/jars" -type f -name '*.jar' -print | paste -sd: -)
test -n "$ANSERINI_CLASSPATH"
export CLASSPATH="$ANSERINI_CLASSPATH"

python3 - <<'PY'
import importlib.metadata
import os
import torch
import transformers
import vllm
from jnius import autoclass
autoclass("java.lang.String")
versions = {
    "torch": torch.__version__,
    "vllm": vllm.__version__,
    "transformers": transformers.__version__,
    "pyserini": importlib.metadata.version("pyserini"),
    "peft": importlib.metadata.version("peft"),
    "structlog": importlib.metadata.version("structlog"),
    "chromadb": importlib.metadata.version("chromadb"),
    "gpu_count": torch.cuda.device_count(),
    "JVM_PATH": os.environ["JVM_PATH"],
    "JAVA_HOME": os.environ["JAVA_HOME"],
    "TIKTOKEN_ENCODINGS_BASE": os.environ.get("TIKTOKEN_ENCODINGS_BASE"),
    "o200k_base_present": os.path.isfile(
        os.path.join(os.environ["TIKTOKEN_ENCODINGS_BASE"], "o200k_base.tiktoken")
    ),
    "java_ok": True,
}
print("PREFLIGHT", versions, flush=True)
assert versions["torch"].startswith("2.10.0")
assert versions["vllm"] == "0.19.1"
assert versions["transformers"] == "4.57.6"
assert versions["pyserini"] == "2.0.0"
assert versions["gpu_count"] == 8
assert versions["o200k_base_present"]
from pyserini.search.lucene import LuceneSearcher
s = LuceneSearcher("/mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-sec-corpus/indexes/bm25")
n_docs = s.num_docs if not callable(s.num_docs) else s.num_docs()
print("LUCENE_OK", {"num_docs": n_docs}, flush=True)
assert n_docs > 0
PY

python3 -m py_compile /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_train.py
cd /mnt/songzijun/Capability_Evolution
exec python3 /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_train.py \
  --harness Harness-1 \
  --benchmark bcplus_full \
  --model_name /mnt/songzijun/models/openai/gpt-oss-20b \
  --train_method trim \
  --component all \
  --train-env local_legacy \
  --rl-data /mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-rl-data \
  --sec-corpus-root /mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-sec-corpus \
  --train-groups-per-step 32 \
  --train-micro-batch-size 4 \
  --train-heartbeat-every 8 \
  --out /mnt/songzijun/Capability_Evolution/TRIM/outputs/cloudml_custom_train_harness1_bcplus_full_gpt_oss_20b_trim_8gpu_20260910_recovery2
CLOUDML_IMAGE_COMMAND
)

cml custom_train submit \
  --context songzijun-cto \
  --workspace_id 10063 \
  --job_name trim-harness1-bcplus-full-gpt-oss-20b-trim-8gpu-20260910-recovery2 \
  --description "Recovery2: Harness-1 BC+ full TRIM training, GPT-OSS-20B, 8x gm402-80g; drops SCAPE_FORCE_LOCAL_HARMONY so the real openai_harmony Conversation (with to_json) is used instead of _LocalConversation, while keeping --train-env local_legacy" \
  --access_type PRIVATE \
  --tag MemQA-General-V5 \
  --image_url micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto \
  --image_command "${IMAGE_COMMAND}" \
  --env PYTHONUNBUFFERED=1 \
  --env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  --env MAX_TURNS=40 \
  --env PYTHONPATH=/mnt/songzijun/Capability_Evolution/TRIM:/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD \
  --env VLLM_DISABLE_CUSTOM_ALL_REDUCE=1 \
  --env SCAPE_LOCAL_OPENAI_EMBEDDINGS=1 \
  --env TIKTOKEN_ENCODINGS_BASE=/tmp/tiktoken-bundle/share/tiktoken \
  --env TIKTOKEN_RS_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_rs_cache \
  --env TIKTOKEN_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_cache \
  --queue_id 2334 \
  --priority 8 \
  --framework pytorch \
  --resource_configs '[{"nodeRole":"worker","nodeNumber":1,"perNodeResourceSpec":{"resourcePriority":"GUARANTEED_PUBLIC","resourceName":"cloudml.ng1gm402-8-8.20-224","resourceNumber":8},"elastic":false,"preemptible":false}]' \
  --juicefs_mount_configs '[{"volume":"songzijun","endpoint":"bj9-cloudml","juiceFsCluster":"bj9-cloudml","subPath":"/","mountPath":"/mnt/songzijun","readOnly":false}]' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}'
