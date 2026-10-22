#!/usr/bin/env bash
# Submit TRIM Harness-1 BC+ full Qwen3-4B-Instruct training (8x gm402-80g, queue 2334).
# JuiceFS mount is passed DIRECTLY via --juicefs_mount_configs (endpoint=bj9-cloudml),
# NOT via the YAML --filename path, so the endpoint is explicit in the API request.
#
# Resource YAML used for resource_configs: cloudml_custom_train_harness1_bcplus_full_qwen3_4b_trim_8gpu_20260910_memqa_general_v5_not_submitted.yaml
# Review the full image_command and env in that YAML before submitting.
set -euo pipefail

IMAGE_COMMAND=$(cat <<'CLOUDML_IMAGE_COMMAND'
set -euxo pipefail
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MAX_TURNS=40
export PYTHONPATH=/mnt/songzijun/Capability_Evolution/TRIM/sitecustomize_dir:/mnt/songzijun/Capability_Evolution/TRIM:/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD
export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1
export OPENAI_API_KEY=dummy
export SCAPE_LOCAL_OPENAI_EMBEDDINGS=1
export TIKTOKEN_ENCODINGS_BASE=/mnt/songzijun/tiktoken-bundle/share/tiktoken
export TIKTOKEN_RS_CACHE_DIR=/mnt/songzijun/tiktoken-bundle/tiktoken_rs_cache
export TIKTOKEN_CACHE_DIR=/mnt/songzijun/tiktoken-bundle/tiktoken_cache

nvidia-smi -L
test -f /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_train.py
test -f /mnt/songzijun/models/Qwen3-4B-Instruct-2507/config.json
test -f /mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-rl-data/manifest.json
test -f /mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-sec-corpus/manifest.json
test -f /mnt/songzijun/tiktoken-bundle/share/tiktoken/o200k_base.tiktoken

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends openjdk-21-jre-headless
JVM_PATH=$(find /usr /opt -type f -name libjvm.so 2>/dev/null | head -1 || true)
if [ -z "${JVM_PATH}" ]; then JVM_PATH=/usr/lib/jvm/java-21-openjdk-amd64/lib/server/libjvm.so; fi
test -n "${JVM_PATH}"
test -f "${JVM_PATH}"
ldd "${JVM_PATH}"
export JAVA_HOME=$(dirname "$(dirname "$(dirname "${JVM_PATH}")")")
export JVM_PATH
export LD_LIBRARY_PATH=$(dirname "${JVM_PATH}"):${LD_LIBRARY_PATH:-}

python3 -m pip install --no-cache-dir --ignore-installed --upgrade-strategy only-if-needed \
  "torch==2.10.0" "transformers==4.57.6" "accelerate==1.13.0" "datasets==4.8.4" \
  "hydra-core==1.3.2" "omegaconf==2.3.0" "vllm==0.19.1" \
  peft structlog chromadb tqdm regex sympy latex2sympy2 word2number sentencepiece \
  pyserini openai tenacity anthropic \
  "chz>=0.4.0" "tinker==0.28.1" pyarrow "json-repair>=0.55.0"

# Set ANSERINI_CLASSPATH to the fatjar installed by pyserini so any later
# `import pyserini.search` (in training too) finds io.anserini.* classes. This must
# happen before any jnius autoclass so the JVM starts with the fatjar on the classpath.
ANSERINI_CLASSPATH=$(python3 -c "import pyserini,glob,os; d=os.path.join(os.path.dirname(pyserini.__file__),'resources','jars'); j=glob.glob(os.path.join(d,'anserini-*-fatjar.jar')); assert j, d; print(j[0])")
export ANSERINI_CLASSPATH
echo "ANSERINI_CLASSPATH=${ANSERINI_CLASSPATH}"

# Build the SEC corpus jsonl (chunk_id + document_text) from the 43 parquet shards so
# the local_bm25 retrieval backend (--corpus-path) has full document text. This matches
# trim/local_backend/corpus_store.from_jsonl which expects rows with `id` and `text`.
# Stored on JuiceFS so subsequent recoveries reuse it instead of rebuilding 825MB.
SEC_CORPUS=/mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-sec-corpus
SEC_JSONL=${SEC_CORPUS}/sec_corpus_chunks.jsonl
SEC_INDEX=${SEC_CORPUS}/indexes/bm25
if [ ! -s "${SEC_JSONL}" ]; then
  echo "Building SEC corpus jsonl from parquet -> ${SEC_JSONL}"
  python3 - <<'PY'
import glob, json, os
import pyarrow.parquet as pq
sec_dir = "/mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-sec-corpus"
out = os.path.join(sec_dir, "sec_corpus_chunks.jsonl")
shards = sorted(glob.glob(os.path.join(sec_dir, "corpora/sec/train/train-*.parquet")))
assert shards, "no SEC parquet shards found"
n = 0
with open(out, "w", encoding="utf-8") as f:
    for sp in shards:
        pf = pq.ParquetFile(sp)
        for batch in pf.iter_batches(batch_size=4096):
            cols = batch.to_pydict()
            ids = cols.get("chunk_id") or cols.get("id") or cols.get("docid")
            texts = cols.get("document_text") or cols.get("text") or cols.get("contents")
            assert ids is not None and texts is not None, f"missing chunk_id/document_text in {sp}: {list(cols.keys())}"
            for cid, txt in zip(ids, texts):
                if not cid or not (txt and str(txt).strip()):
                    continue
                f.write(json.dumps({"id": str(cid), "text": str(txt)}, ensure_ascii=False) + "\n")
                n += 1
print(f"SEC_JSONL_BUILT rows={n} path={out}", flush=True)
PY
fi
echo "SEC_JSONL=${SEC_JSONL} ($(wc -l < ${SEC_JSONL} 2>/dev/null) lines)"
test -s "${SEC_JSONL}"
test -d "${SEC_INDEX}"

python3 - <<'PY'
import importlib.metadata
import os
import glob
import torch
import transformers
import vllm
import peft
import structlog
import chromadb

# chz + tinker are required: load_upstream_modules() imports training.train_rl, which
# imports tinker_cookbook.rl.types, which top-level imports `chz` and `tinker`. If
# either is missing, SlidingWindowSearchEnv cannot be imported (train_env=upstream and
# the local_bm25 toolset both go through this import chain).
import chz  # noqa: F401
import tinker  # noqa: F401
print("CHZ_TINKER_OK", {"chz": chz.__version__ if hasattr(chz, "__version__") else "?", "tinker": tinker.__version__ if hasattr(tinker, "__version__") else "?"}, flush=True)

# Pre-check the full SlidingWindowSearchEnv import chain (harness.tools/trajectory/
# agent + training.train_rl + tinker_cookbook.rl.types). This surfaces any remaining
# missing harness-1 deps (e.g. json_repair) BEFORE the training process hits
# open_upstream_train_session, so we fail fast in preflight instead of mid-training.
from trim.upstream_harness1.pin import ensure_harness1_on_path
ensure_harness1_on_path()
from harness.agent import TinkerAgentInferenceModel  # noqa: F401
from training.train_rl import SlidingWindowSearchEnv  # noqa: F401
print("UPSTREAM_IMPORT_CHAIN_OK", flush=True)

# CRITICAL: jnius_config.add_classpath only takes effect BEFORE the JVM starts.
# pyserini.pyclass wraps configure_classpath in a bare try/except that silently swallows
# failures (e.g. if add_options raises), leaving the anserini fatjar OFF the classpath ->
# NoClassDefFoundError on io.anserini.eval.Qrels when pyserini.search is imported. So we
# register the fatjar on jnius_config OURSELVES, before importing jnius, and print the
# resulting classpath so we can verify it in the logs.
import pyserini
_jar_dir = os.path.join(os.path.dirname(pyserini.__file__), "resources", "jars")
_fatjar = glob.glob(os.path.join(_jar_dir, "anserini-*-fatjar.jar"))
assert _fatjar, f"no anserini-*-fatjar.jar under {_jar_dir}"
os.environ["ANSERINI_CLASSPATH"] = _fatjar[0]

import jnius_config
jnius_config.add_classpath(_fatjar[0])
try:
    jnius_config.add_options("--add-modules=jdk.incubator.vector")
except Exception as _opt:
    print("WARN add_options failed:", repr(_opt), flush=True)
_cp = jnius_config.get_classpath()
print("JNIUS_CLASSPATH_BEFORE_JVM:", _cp, flush=True)
assert _fatjar[0] in (_cp or ""), f"classpath missing fatjar: {_cp}"

# Verify sitecustomize auto-loaded at interpreter startup so the TRAINING process
# (run_train.py) also gets the fatjar on the classpath before any jnius import.
import sitecustomize as _sc  # noqa
assert hasattr(_sc, "_register_anserini_classpath"), "sitecustomize not the expected module"
print("SITECUSTOMIZE_LOADED", flush=True)

from jnius import autoclass
from pyserini.search.lucene import LuceneSearcher

JString = autoclass("java.lang.String")  # JVM starts WITH anserini on classpath
s = LuceneSearcher("/mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-sec-corpus/indexes/bm25")
versions = {
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "vllm": vllm.__version__,
    "gpu_count": torch.cuda.device_count(),
    "pyserini": importlib.metadata.version("pyserini"),
    "peft": peft.__version__,
    "chromadb": importlib.metadata.version("chromadb"),
    "structlog": importlib.metadata.version("structlog"),
    "JVM_PATH": os.environ["JVM_PATH"],
    "JAVA_HOME": os.environ["JAVA_HOME"],
    "ANSERINI_CLASSPATH": os.environ["ANSERINI_CLASSPATH"],
    "jnius_classpath": _cp,
    "java_ok": True,
    "num_docs": s.num_docs,
}
print("PREFLIGHT", versions, flush=True)
assert versions["torch"].startswith("2.10.0")
assert versions["vllm"] == "0.19.1"
assert versions["transformers"] == "4.57.6"
assert versions["gpu_count"] == 8
assert versions["num_docs"] > 0
PY

python3 -m py_compile /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_train.py
cd /mnt/songzijun/Capability_Evolution
exec python3 /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_train.py \
  --harness Harness-1 \
  --benchmark bcplus_full \
  --model_name /mnt/songzijun/models/Qwen3-4B-Instruct-2507 \
  --train_method trim \
  --component all \
  --rl-data /mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-rl-data \
  --sec-corpus-root /mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-sec-corpus \
  --retrieval-backend local_bm25 \
  --index-path /mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-sec-corpus/indexes/bm25 \
  --corpus-path /mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-sec-corpus/sec_corpus_chunks.jsonl \
  --reranker none \
  --train-env local_legacy \
  --train-groups-per-step 32 \
  --train-micro-batch-size 4 \
  --train-heartbeat-every 8 \
  --out /mnt/songzijun/Capability_Evolution/TRIM/outputs/cloudml_custom_train_harness1_bcplus_full_qwen3_4b_trim_8gpu_20260910_memqa_general_v5_recovery8
CLOUDML_IMAGE_COMMAND
)

cml custom_train submit \
  --context songzijun-cto \
  --workspace_id 10063 \
  --job_name trim-harness1-bcplus-full-qwen3-4b-trim-8gpu-20260910-memqa-general-v5-recovery8 \
  --description "Recovery8: use --train-env local_legacy to avoid upstream SlidingWindowSearchEnv BC+ probe docid mismatch on SEC corpus" \
  --access_type PRIVATE \
  --tag MemQA-General-V5 \
  --image_url micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto \
  --image_command "${IMAGE_COMMAND}" \
  --framework pytorch \
  --queue_id 2334 \
  --priority 8 \
  --resource_configs '[{"nodeRole":"worker","nodeNumber":1,"perNodeResourceSpec":{"resourcePriority":"GUARANTEED_PUBLIC","resourceName":"cloudml.ng1gm402-8-8.20-224","resourceNumber":8},"elastic":false,"preemptible":false}]' \
  --juicefs_mount_configs '[{"volume":"songzijun","endpoint":"bj9-cloudml","juiceFsCluster":"bj9-cloudml","subPath":"/","mountPath":"/mnt/songzijun","readOnly":false}]' \
  --env PYTHONUNBUFFERED=1 \
  --env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  --env MAX_TURNS=40 \
  --env PYTHONPATH=/mnt/songzijun/Capability_Evolution/TRIM/sitecustomize_dir:/mnt/songzijun/Capability_Evolution/TRIM:/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD \
  --env VLLM_DISABLE_CUSTOM_ALL_REDUCE=1 \
  --env OPENAI_API_KEY=dummy \
  --env SCAPE_LOCAL_OPENAI_EMBEDDINGS=1 \
  --env TIKTOKEN_ENCODINGS_BASE=/mnt/songzijun/tiktoken-bundle/share/tiktoken \
  --env TIKTOKEN_RS_CACHE_DIR=/mnt/songzijun/tiktoken-bundle/tiktoken_rs_cache \
  --env TIKTOKEN_CACHE_DIR=/mnt/songzijun/tiktoken-bundle/tiktoken_cache \
  --diag_options '[{"name":"HangDetection","enable":false}]' \
  --retry_config '{"enableRetry":false,"maxRetryTimes":1,"policySets":["JobFailed"]}' \
  --alert_config '{"enableAlert":true,"alertItems":[{"alertConditions":["FAILED"],"alertLevel":"P2","alertReceivers":{"persons":["songzijun"]}}]}'
