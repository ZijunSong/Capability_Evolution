#!/usr/bin/env bash
set -euo pipefail

# Prepared after recovery1 reproduced vLLM V1 LoRA contiguous-input failure.
IMAGE_COMMAND=$(cat <<'CLOUDML_IMAGE_COMMAND'
set -euxo pipefail
export PYTHONUNBUFFERED=1
export OPENAI_API_KEY=dummy
export VLLM_USE_V1=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MAX_TURNS=40
export PYTHONPATH=/mnt/songzijun/Capability_Evolution:/mnt/songzijun/Capability_Evolution/TRIM:/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD
export SCAPE_LOCAL_OPENAI_EMBEDDINGS=1

nvidia-smi -L
test -f /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_eval.py
test -f /mnt/songzijun/models/openai/gpt-oss-20b/config.json
test -d /mnt/songzijun/Capability_Evolution/TRIM/outputs/cloudml_custom_train_harness1_bcplus_full_gpt_oss_20b_rl_8gpu_20260908_recovery1/adapters/rl
test -f /mnt/songzijun/Capability_Evolution/TRIM/manifests/browsecomp_plus_830/SPLIT.json
test -d /mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus/indexes/bm25

python3 -m pip install --no-cache-dir --ignore-installed --upgrade-strategy only-if-needed \
  'vllm==0.19.1' 'transformers==4.57.6' 'pyserini==2.0.0' \
  tqdm sentence-transformers regex sympy latex2sympy2 word2number structlog chromadb

JVM_PATH=$(find /usr /opt -type f -name libjvm.so 2>/dev/null | head -1 || true)
if [ -z "${JVM_PATH}" ]; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends openjdk-17-jre-headless
  JVM_PATH=$(find /usr /opt -type f -name libjvm.so 2>/dev/null | head -1 || true)
fi
test -n "${JVM_PATH}"
test -f "${JVM_PATH}"
ldd "${JVM_PATH}"
export JVM_PATH
export JAVA_HOME=$(dirname "$(dirname "$(dirname "${JVM_PATH}")")")
export LD_LIBRARY_PATH=$(dirname "${JVM_PATH}"):${LD_LIBRARY_PATH:-}

python3 - <<'PY'
import importlib.metadata
import os
import torch
import transformers
import vllm
from jnius import autoclass

autoclass("java.lang.String")
versions = {
    "torch": torch.__version__, "vllm": vllm.__version__,
    "transformers": transformers.__version__,
    "pyserini": importlib.metadata.version("pyserini"),
    "chromadb": importlib.metadata.version("chromadb"),
    "structlog": importlib.metadata.version("structlog"),
    "gpu_count": torch.cuda.device_count(), "JVM_PATH": os.environ["JVM_PATH"],
    "JAVA_HOME": os.environ["JAVA_HOME"], "java_ok": True,
    "VLLM_USE_V1": os.environ.get("VLLM_USE_V1"),
}
print("PREFLIGHT", versions, flush=True)
assert versions["torch"].startswith("2.10.0")
assert versions["vllm"] == "0.19.1"
assert versions["transformers"] == "4.57.6"
assert versions["pyserini"] == "2.0.0"
assert versions["gpu_count"] == 8
assert versions["VLLM_USE_V1"] == "0"
PY

python3 - <<'PY'
import sys
sys.path.insert(0, "/mnt/songzijun/Capability_Evolution/TRIM/external/harness-1")
import chromadb
import harness.tools
import harness.ultra_core
import structlog
from pyserini.search.lucene import LuceneSearcher
index = "/mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus/indexes/bm25"
searcher = LuceneSearcher(index)
n_docs = searcher.num_docs if not callable(searcher.num_docs) else searcher.num_docs()
hits = searcher.search("history", 1)
print("IMPORT_AND_LUCENE_PREFLIGHT", {"num_docs": n_docs, "probe_hits": len(hits)}, flush=True)
assert n_docs > 0 and hits
PY

python3 - <<'PY'
from pathlib import Path
p=Path("/usr/local/lib/python3.12/dist-packages/vllm/lora/ops/triton_ops/lora_shrink_op.py")
s=p.read_text()
assert "assert inputs.is_contiguous()" in s
p.write_text(s.replace("assert inputs.is_contiguous()", "inputs = inputs.contiguous()", 1))
print("PATCHED_VLLM_LORA_SHRINK_CONTIGUOUS", flush=True)
PY
python3 -m py_compile /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_eval.py
cd /mnt/songzijun/Capability_Evolution
exec python3 /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_eval.py \
  --harness Harness-1 \
  --benchmark bcplus_full \
  --component zero \
  --tp 8 \
  --model_name /mnt/songzijun/models/openai/gpt-oss-20b \
  --adapter /mnt/songzijun/Capability_Evolution/TRIM/outputs/cloudml_custom_train_harness1_bcplus_full_gpt_oss_20b_rl_8gpu_20260908_recovery1/adapters/rl \
  --eval-mode adapter \
  --out /mnt/songzijun/Capability_Evolution/TRIM/outputs/cloudml_trim_run_eval_harness1_bcplus_full_gpt_oss_20b_rl_adapter_8gpu_20260908_memqa_v5_recovery4
CLOUDML_IMAGE_COMMAND
)

cml custom_train submit \
  --context songzijun-cto \
  --workspace_id 10063 \
  --job_name trim-run-eval-harness1-bcplus-full-gpt-oss-20b-rl-adapter-8gpu-20260908-memqa-v5-recovery4 \
  --description "Recovery3: GPT-OSS-20B RL adapter evaluation using vLLM V0 after V1 LoRA kernel assertion" \
  --access_type PRIVATE \
  --tag MemQA-General-V5 \
  --image_url micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto \
  --image_command "${IMAGE_COMMAND}" \
  --env PYTHONUNBUFFERED=1 \
  --env OPENAI_API_KEY=dummy \
  --env VLLM_USE_V1=0 \
  --env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  --env MAX_TURNS=40 \
  --env PYTHONPATH=/mnt/songzijun/Capability_Evolution:/mnt/songzijun/Capability_Evolution/TRIM:/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD \
  --env SCAPE_LOCAL_OPENAI_EMBEDDINGS=1 \
  --queue_id 2334 \
  --priority 8 \
  --framework pytorch \
  --resource_configs '[{"nodeRole":"worker","nodeNumber":1,"perNodeResourceSpec":{"resourcePriority":"GUARANTEED_PUBLIC","resourceName":"cloudml.ng1gm402-8-8.20-224","resourceNumber":8},"elastic":false,"preemptible":false}]' \
  --juicefs_mount_configs '[{"volume":"songzijun","endpoint":"bj9-cloudml","juiceFsCluster":"bj9-cloudml","subPath":"/","mountPath":"/mnt/songzijun","readOnly":false}]' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}'
