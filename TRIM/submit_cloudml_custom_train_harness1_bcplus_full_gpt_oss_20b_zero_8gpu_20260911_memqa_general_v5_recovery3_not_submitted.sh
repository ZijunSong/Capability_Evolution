#!/usr/bin/env bash
set -euo pipefail

# Prepared 2026-09-11; NOT SUBMITTED. Recovery3.
#
# Submits a TRIM Harness-1 BC+ full GPT-OSS-20B ZERO-component evaluation job
# to CloudML workspace 10063 / context songzijun-cto / queue 2334 on one 8-GPU
# gm402-80g node (GUARANTEED_PUBLIC), priority 8, tag MemQA-General-V5.
#
# RECOVERY3: Recovery2 (t-20260911105405-z19v7) passed CHZ_TINKER_OK + PREFLIGHT
# + LUCENE_OK + HARNESS_IMPORT_OK, then the load_upstream_modules() preflight
# caught the next missing module: harness.agent → `import json_repair`. Fix:
# (1) add json_repair (+langchain-text-splitters/peft/trl/pysqlite3-binary) to
# the four-package pip install; (2) make the load_upstream_modules() preflight
# self-healing — on ModuleNotFoundError, pip-install the missing top-level
# module and retry, up to 8 rounds, so any remaining gap is auto-filled at
# startup. Original attempts retained.
#
# Entry: runs scripts/run_bcplus_full_gptoss20b_zero_gpu8_eval.sh, which
# launches 8 single-GPU vLLM actors (TP=1, one per GPU) + 16 upstream_api
# API-client workers (--tp 16, ~2/GPU) over local_bm25 retrieval, no Chroma
# service. This is the fastest layout for all 830 queries on 8 gm402-80g.
#
# JuiceFS volume `songzijun` (cluster bj9-cloudml) is mounted read-write at
# /mnt/songzijun through the DIRECT CLI --juicefs_mount_configs argument so
# endpoint=bj9-cloudml is explicit in the API request. The YAML intentionally
# omits the mount; do NOT switch to --filename for the mount.
#
# Usage:
#   bash submit_cloudml_custom_train_harness1_bcplus_full_gpt_oss_20b_zero_8gpu_20260911_memqa_general_v5_recovery3_not_submitted.sh
#
# To abort after submit: cml custom_train kill <JOB_ID>

IMAGE_COMMAND=$(cat <<'CLOUDML_IMAGE_COMMAND'
set -euxo pipefail
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MAX_TURNS=40
export PYTHONPATH=/mnt/songzijun/Capability_Evolution/TRIM:/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD
export SCAPE_LOCAL_OPENAI_EMBEDDINGS=1
export SCAPE_FORCE_LOCAL_HARMONY=1
export OPENAI_API_KEY="${OPENAI_API_KEY:-*******-***********}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:0/v1}"
export SCAPE_CHROMA_PATH=/tmp/empty_chroma
mkdir -p "$SCAPE_CHROMA_PATH"

python3 -c "import zipfile; zipfile.ZipFile('/mnt/songzijun/tiktoken-bundle.zip').extractall('/tmp')"
test -f /tmp/tiktoken-bundle/share/tiktoken/o200k_base.tiktoken
test -f /tmp/tiktoken-bundle/share/tiktoken/cl100k_base.tiktoken
export TIKTOKEN_ENCODINGS_BASE=/tmp/tiktoken-bundle/share/tiktoken
export TIKTOKEN_RS_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_rs_cache
export TIKTOKEN_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_cache

nvidia-smi -L
test -f /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_bcplus_full_gptoss20b_zero_gpu8_eval.sh
test -f /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_eval.py
test -f /mnt/songzijun/models/openai/gpt-oss-20b/config.json
test -f /mnt/songzijun/Capability_Evolution/TRIM/manifests/browsecomp_plus_830/SPLIT.json
test -d /mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus/indexes/bm25

python3 -m pip install --no-cache-dir --ignore-installed --upgrade-strategy only-if-needed \
  'torch==2.10.0' \
  'vllm==0.19.1' \
  'transformers==4.57.6' \
  'pyserini==2.0.0' \
  'accelerate==1.13.0' \
  tqdm sentence-transformers regex sympy latex2sympy2 word2number \
  structlog chromadb openai tenacity \
  json_repair langchain-text-splitters peft trl pysqlite3-binary

# chz + tinker: required at import time by env_bridge.load_upstream_modules
# → training.train_rl → tinker_cookbook.rl.types. --no-deps AFTER the four-package
# pin so tinker cannot upgrade transformers==4.57.6 (would break vllm 0.19.1).
python3 -m pip install --no-cache-dir --no-deps --ignore-installed 'chz==0.4.0' 'tinker'
python3 -c "import chz, tinker, importlib.metadata as m; print('CHZ_TINKER_OK', m.version('chz'), getattr(tinker,'__version__','?'))"

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
PYSERINI_ROOT=$(python3 -c 'import pathlib, pyserini; print(pathlib.Path(pyserini.__file__).resolve().parent)')
export ANSERINI_CLASSPATH=$(find "$PYSERINI_ROOT/resources/jars" -type f -name '*.jar' -print | paste -sd: -)
test -n "$ANSERINI_CLASSPATH"
export CLASSPATH="$ANSERINI_CLASSPATH"

python3 - <<'PY'
import importlib.metadata
import os
import sys
import torch
import transformers
import vllm
sys.path.insert(0, "/mnt/songzijun/Capability_Evolution/TRIM/external/harness-1")
from jnius import autoclass
autoclass("java.lang.String")
versions = {
    "torch": torch.__version__,
    "vllm": vllm.__version__,
    "transformers": transformers.__version__,
    "pyserini": importlib.metadata.version("pyserini"),
    "chromadb": importlib.metadata.version("chromadb"),
    "structlog": importlib.metadata.version("structlog"),
    "gpu_count": torch.cuda.device_count(),
    "JVM_PATH": os.environ["JVM_PATH"],
    "JAVA_HOME": os.environ["JAVA_HOME"],
    "o200k_base_present": os.path.isfile(
        os.path.join(os.environ["TIKTOKEN_ENCODINGS_BASE"], "o200k_base.tiktoken")
    ),
    "java_ok": True,
}
print("PREFLIGHT", versions, flush=True)
assert versions["torch"].startswith("2.10.0"), versions["torch"]
assert versions["vllm"] == "0.19.1", versions["vllm"]
assert versions["transformers"] == "4.57.6", versions["transformers"]
assert versions["pyserini"] == "2.0.0", versions["pyserini"]
assert versions["gpu_count"] == 8, versions["gpu_count"]
assert versions["o200k_base_present"]
from pyserini.search.lucene import LuceneSearcher
s = LuceneSearcher("/mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus/indexes/bm25")
n_docs = s.num_docs if not callable(s.num_docs) else s.num_docs()
hits = s.search("history", 1)
print("LUCENE_OK", {"num_docs": n_docs, "probe_hits": len(hits)}, flush=True)
assert n_docs > 0 and hits
import chromadb, structlog
import harness.tools, harness.ultra_core  # noqa: F401
print("HARNESS_IMPORT_OK", flush=True)
# End-to-end import of the SAME env_bridge path the upstream_api worker runs,
# with a self-healing loop: any ModuleNotFoundError is pip-installed (top-level
# pkg only) and the import retried, up to 8 rounds. This terminates the
# fix-one-miss-one-resubmit loop. chz+tinker stay --no-deps (never let pip pull
# tinker's deps or it upgrades transformers==4.57.6 and breaks vllm 0.19.1).
import subprocess
from trim.upstream_harness1.env_bridge import load_upstream_modules
os.environ.setdefault("OPENAI_API_KEY", "preflight-placeholder")
os.environ.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:0/v1")
os.environ.setdefault("HARNESS1_FORBID_CHROMA", "1")
_NO_DEPS_PKGS = {"chz", "tinker"}
for _ in range(8):
    try:
        mods = load_upstream_modules()
        assert "SlidingWindowSearchEnv" in mods and "ToolSet" in mods
        print("ENV_BRIDGE_LOAD_OK", {"n_modules": len(mods)}, flush=True)
        break
    except ModuleNotFoundError as mnf:
        raw = str(mnf)
        mod = raw.split("'")[1] if "'" in raw else raw.split()[-1].rstrip("'")
        mod = mod.split(".")[0]
        print(f"ENV_BRIDGE_AUTO_INSTALL round: installing {mod}", flush=True)
        flag = "--no-deps" if mod in _NO_DEPS_PKGS else ""
        subprocess.run(
            ["python3", "-m", "pip", "install", "--no-cache-dir", "--ignore-installed"]
            + ([flag] if flag else []) + [mod],
            check=True,
        )
        import importlib
        importlib.invalidate_caches()
        continue
else:
    raise SystemExit("ENV_BRIDGE_LOAD_FAILED after auto-install rounds; see log above")
PY

python3 -m py_compile /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_eval.py
bash -n /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_bcplus_full_gptoss20b_zero_gpu8_eval.sh

cd /mnt/songzijun/Capability_Evolution/TRIM
exec bash /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_bcplus_full_gptoss20b_zero_gpu8_eval.sh
CLOUDML_IMAGE_COMMAND
)

cml custom_train submit \
  --context songzijun-cto \
  --workspace_id 10063 \
  --job_name trim-run-eval-harness1-bcplus-full-gpt-oss-20b-zero-8gpu-20260911-memqa-general-v5-recovery3 \
  --description "Recovery3: Harness-1 BC+ full zero-component eval of GPT-OSS-20B on 8x gm402-80g; chz+tinker+json_repair (--no-deps for chz/tinker) + self-healing load_upstream_modules preflight; 8 actors + 16 workers, local_bm25, no Chroma" \
  --access_type PRIVATE \
  --tag MemQA-General-V5 \
  --image_url micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto \
  --image_command "${IMAGE_COMMAND}" \
  --env PYTHONUNBUFFERED=1 \
  --env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  --env MAX_TURNS=40 \
  --env PYTHONPATH=/mnt/songzijun/Capability_Evolution/TRIM:/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD \
  --env SCAPE_LOCAL_OPENAI_EMBEDDINGS=1 \
  --env SCAPE_FORCE_LOCAL_HARMONY=1 \
  --env SCAPE_CHROMA_PATH=/tmp/empty_chroma \
  --queue_id 2334 \
  --priority 8 \
  --framework pytorch \
  --resource_configs '[{"nodeRole":"worker","nodeNumber":1,"perNodeResourceSpec":{"resourcePriority":"GUARANTEED_PUBLIC","resourceName":"cloudml.ng1gm402-8-8.20-224","resourceNumber":8},"elastic":false,"preemptible":false}]' \
  --juicefs_mount_configs '[{"volume":"songzijun","endpoint":"bj9-cloudml","juiceFsCluster":"bj9-cloudml","subPath":"/","mountPath":"/mnt/songzijun","readOnly":false}]' \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}'
