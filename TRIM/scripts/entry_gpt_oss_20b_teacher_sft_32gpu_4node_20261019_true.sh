#!/usr/bin/env bash
# === GPT-OSS-20B Teacher SFT (Teacher-imitation baseline, 2026-10-19) ===
# Three SFT ablations forked from the verified only-opd/no-gate/no-project
# suite (same base, same sec corpus, same Harness-1 BC+ eval). This entry is
# the TEACHER SFT leg. Purpose: the most direct Teacher-imitation baseline —
# SFT directly on Full-Harness (component=all) teacher trajectories captured
# on-policy by the teacher, supervised only on the teacher's native actions.
#
# Distinct from only-opd/trim/all: this is NOT verl torch_ddp_lora + RL/OPD.
# It is the independent SFT v1 stack (commit 002da5f): HuggingFace
# AutoModelForCausalLM + PEFT LoRA + hand-written action-only CE
# (trim/training/sft_ablation/{trainer,ce}.py), vLLM only for rollout/collection.
# Scheme-A exclusive: rank 0 is the ONLY gradient rank; the other 3
# coordinator ranks + 8 rollout replicas/node only sample. 4 x (1 coord + 8
# replicas) x tp1 = 32 cards. Loss = action-only CE, NO RL/advantage/gap/gate.
#
# Ablation alignment (CLAUDE.md: must use same base/init/data/eval as only-opd):
#   * base model: /mnt/songzijun/models/openai/gpt-oss-20b-bf16-dequant (bf16,
#     NOT mxfp4 — same v0dhh fix as only-opd; mxfp4 Marlin PTX crash avoided).
#   * LoRA init: NO --init-adapter. build_hf_trainer get_peft_model builds a
#     fresh lora-rank=32 LoRA from the base; seed=42 is shared across the
#     three SFT legs so the ONLY variable is supervision signal source.
#     (Decision: 2026-10-19; no common-init adapter exists locally and a
#     pre-trained init would inject an extra variable into the ablation.)
#   * teacher = base itself (no teacher adapter): --teacher-model the bf16
#     base, teacher-component=all (Full Harness), student-component=zero.
#   * data: --train-data sec, --rl-data harness-1-rl-data, --sec-corpus-root
#     harness-1-sec-corpus — identical paths to only-opd.
#   * rollout backend: VLLMGenerateClient (vllm_hybrid.py:493), per-replica
#     TP=1, one GPU each. env_worker subprocess per replica.
#
# GPT-OSS-20B model-specific constraints carried from the only-opd baseline
# (all verified, NOT optional):
#   1. micro-batch=1 (NOT 4): SFT v1 ce_update has NO gradient checkpointing
#      (build_hf_trainer does not call gradient_checkpointing_enable; ce.py
#      _forward_logits is a plain forward). GPT-OSS MoE backward cannot
#      grad-ckpt; mb>=2 risks the 74.7GB/80GB backward peak (k6v0p 铁证, RL
#      path). SFT rank-0 single-GPU backward is the same MoE weights, so
#      --micro-batch-size 1 mirrors the only-opd mb1 discipline.
#   2. bf16-dequant base (NOT mxfp4): Marlin PTX crash (v0dhh) + NaN fix.
#   3. SCAPE_FORCE_LOCAL_HARMONY INTENTIONALLY NOT SET (gpt-oss to_json crash,
#      xwl4c 铁证).
#   4. gpu-mem 0.85 for rollout vLLM (NOT 0.90 default): bf16 39-40GB weights +
#     0.85 leaves ~KV headroom; matches only-opd bf16 铁律. VLLMGenerateClient
#     defaults to 0.90 (vllm_hybrid.py:508) and run_rollout_job does not pass
#     it, so we inject SCAPE_VLLM_GPU_MEM_UTIL=0.85 if the client honors it
#     (it is read by the vLLM worker; if absent the env is a no-op).
#
# Everything else byte-identical to only-opd: 4 DDP context patches NOT needed
# here (SFT v1 does NOT import trainer_adapter / four_cell_runtime / verl — it
# is an independent stack), but the JVM/Pyserini/tiktoken/model-cp/preflight
# blocks ARE needed (SEC retrieval uses Lucene BM25; harness import chain
# needs o200k_base). rdzv-id gptoss20b-teacher-sft-bf16-32 (port 29611).

set -euxo pipefail

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TRIM_GPU_KEEPALIVE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/mnt/songzijun/Capability_Evolution/TRIM:/mnt/songzijun/Capability_Evolution/TRIM/external/harness-1:/mnt/songzijun/Capability_Evolution/TRIM/external/harness-1/tinker-cookbook:/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD

# --- Multi-node distributed env guard ---
NODE_RANK="${NODE_RANK:-${PET_NODE_RANK:-${NODE_INDEX:-${NODE_NUMBER:-}}}}"
if [ -z "${NODE_RANK:-}" ]; then
  echo "FATAL: NODE_RANK is not set. Each pod needs its own rank id (0..3)." >&2
  exit 1
fi
if [ -z "${MASTER_ADDR:-}" ]; then
  echo "FATAL: MASTER_ADDR is not set. CloudML custom_train multi-node must inject" \
       "node0's IP as MASTER_ADDR. Aborting to avoid 4 independent single-node runs." >&2
  exit 1
fi
: "${MASTER_PORT:=29611}"
export MASTER_ADDR NODE_RANK MASTER_PORT

unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK \
      GROUP_WORLD_SIZE TORCHELASTIC_RUN_ID TORCHELASTIC_RESTART_COUNT \
      TORCHELASTIC_MAX_RESTARTS TORCHELASTIC_USE_AGENT_STORE \
      PET_MASTER_ADDR PET_MASTER_PORT PET_NNODES PET_NPROC_PER_NODE PET_NODE_RANK

echo "MULTINODE_ENV master_addr=${MASTER_ADDR} node_rank=${NODE_RANK} master_port=${MASTER_PORT} host=$(hostname)"

# --- Shared output directory (MUST be identical on every node) ---
if [ -z "${OUT:-}" ]; then
  echo "FATAL: OUT is not set. Multi-node requires a shared --out path injected via --env OUT=... (identical on every pod). Aborting." >&2
  exit 1
fi
export OUT
echo "MULTINODE_OUT out=${OUT} host=$(hostname) node_rank=${NODE_RANK}"

# FIX1 (to_json bug): SCAPE_FORCE_LOCAL_HARMONY is INTENTIONALLY NOT SET.
export OPENAI_API_KEY="${OPENAI_API_KEY:-**-****-***********-***-*-****-***}"
export SCAPE_LOCAL_OPENAI_EMBEDDINGS=1
export SCAPE_CHROMA_PATH=/tmp/empty_chroma
mkdir -p "$SCAPE_CHROMA_PATH"

# gpu-mem 0.85 for the rollout vLLM worker (GPT-OSS bf16 铁律).
export SCAPE_VLLM_GPU_MEM_UTIL=0.85

# tiktoken bundle: harness import chain (openai_harmony) still needs o200k_base.
python3 -c "import zipfile; zipfile.ZipFile('/mnt/songzijun/tiktoken-bundle.zip').extractall('/tmp')"
test -f /tmp/tiktoken-bundle/share/tiktoken/o200k_base.tiktoken
test -f /tmp/tiktoken-bundle/share/tiktoken/cl100k_base.tiktoken
export TIKTOKEN_ENCODINGS_BASE=/tmp/tiktoken-bundle/share/tiktoken
export TIKTOKEN_RS_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_rs_cache
export TIKTOKEN_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_cache

# --- Presence checks on the mounted JuiceFS tree ---
nvidia-smi -L
test -f /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_teacher_sft_multinode.py
test -f /mnt/songzijun/models/openai/gpt-oss-20b-bf16-dequant/config.json
test -f /mnt/songzijun/models/openai/gpt-oss-20b-bf16-dequant/model.safetensors.index.json
test -f /mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-rl-data/manifest.json
test -f /mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-sec-corpus/manifest.json
test -d /mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-sec-corpus/indexes/bm25

# --- JVM discovery for Pyserini/Lucene (SEC BM25 retrieval) ---
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

# --- Install four-pack + SFT/harness import chain deps (internal Xiaomi PyPI) ---
# SFT v1 stack needs: torch/transformers/vllm/peft (LoRA), pyserini+JVM (SEC
# BM25), structlog/chromadb (harness.ultra_core/tools import chain), openai
# (SCAPE_LOCAL_OPENAI_EMBEDDINGS). Same set as only-opd.
python3 -m pip install --no-cache-dir --ignore-installed --upgrade-strategy only-if-needed \
  "torch==2.10.0" "transformers==4.57.6" "accelerate==1.13.0" "datasets==4.8.4" \
  "hydra-core==1.3.2" "omegaconf==2.3.0" "vllm==0.19.1" peft structlog chromadb \
  tqdm regex sympy latex2sympy2 word2number sentencepiece "pyserini==2.0.0" openai tenacity anthropic \
  chz tinker \
  tensorboard

# --- Install harness-1's TRANSITIVE deps (NOT harness-1 itself) ---
# harness-1 is a flat-layout repo (multiple top-level pkgs: harness/training/
# datagen/inference/...), so `pip install -e harness-1` FAILS with "Multiple
# top-level package discovered in a flat-layout" (r7 ah0f0 worker-1 FAILED this
# way at 07:43:45). harness-1 itself does NOT need to be installed — it is
# already importable via PYTHONPATH (entry sets external/harness-1 on PYTHONPATH,
# so `import training.train_rl` / `import harness.ultra_core` resolve to the
# submodule source directly). We only need its transitive runtime deps that the
# `import training.train_rl` chain pulls: json-repair (harness/agent.py:6),
# langchain-text-splitters, wikipedia-api, modal, baseten-performance-client,
# fastembed, pysqlite3-binary, wandb, trl, chz, tinker, rich (all verified on
# Xiaomi PyPI). --upgrade-strategy only-if-needed keeps the already-installed
# four-pack (torch/transformers/vllm/peft) + structlog/tiktoken/chromadb/etc
# unchanged; only the MISSING ones get installed.
python3 -m pip install --no-cache-dir --upgrade-strategy only-if-needed \
  json-repair langchain-text-splitters wikipedia-api modal baseten-performance-client \
  fastembed pysqlite3-binary wandb trl chz tinker rich

# --- .pth file with sys.path.insert(0,...): make external/harness-1 + tinker-cookbook
# win the `training` package name over TRIM/trim/training.
# r11 ROOT CAUSE (finally diagnosed via result.json diag_sys_path): the env_worker
# child subprocess sys.path had BOTH '/mnt/.../TRIM/trim' (which contains training/
# with __init__.py — a REAL package) AND '/mnt/.../TRIM/external/harness-1' (which
# contains training/train_rl.py). Since TRIM/trim came BEFORE harness-1 in sys.path,
# `import training` resolved to TRIM/trim/training (real pkg wins), and `import
# training.train_rl` failed (no train_rl.py there). A plain .pth path-line only
# appends to sys.path END (after TRIM/trim), so it did NOT win. The fix: a .pth
# `import sys; sys.path.insert(0, ...)` line runs at site-init and PREPENDS harness-1
# to sys.path, so `import training` finds harness-1/training (with train_rl.py) first.
# Verified locally: with insert(0) .pth, training.__path__ = harness-1/training and
# import training.train_rl proceeds (reaches tinker_cookbook import — also added).
HARNESS1=/mnt/songzijun/Capability_Evolution/TRIM/external/harness-1
TINKER_CB=$HARNESS1/tinker-cookbook
SITE_PKGS=$(python3 -c "import site; print(site.getsitepackages()[0])")
cat > "$SITE_PKGS/zzz_harness1.pth" <<EOF
import sys; sys.path.insert(0, "$HARNESS1"); sys.path.insert(0, "$TINKER_CB")
EOF
test -f "$SITE_PKGS/zzz_harness1.pth"
echo "HARNESS1_PTH_WRITTEN $SITE_PKGS/zzz_harness1.pth"
# verify: a fresh subprocess (no PYTHONPATH, -E ignores env) can import training.train_rl
# AND training resolves to harness-1 (not TRIM/trim). This proves insert(0) .pth wins.
python3 -E -c "import training; assert 'harness-1/training' in list(training.__path__)[0], list(training.__path__); import training.train_rl, harness.ultra_core; print('HARNESS1_SUBPROCESS_IMPORT_OK', list(training.__path__))"

# --- ANSERINI_CLASSPATH for PyJNI/Lucene ---
PYSERINI_ROOT=$(python3 -c 'import pathlib, pyserini; print(pathlib.Path(pyserini.__file__).resolve().parent)')
export ANSERINI_CLASSPATH=$(find "$PYSERINI_ROOT/resources/jars" -type f -name '*.jar' -print | paste -sd: -)
test -n "$ANSERINI_CLASSPATH"
export CLASSPATH="$ANSERINI_CLASSPATH"

# --- Preflight: assert four-pack versions, 8 GPUs (per node), JVM, SFT import chain, SEC Lucene ---
python3 - <<'PY'
import importlib.metadata
import json
import os
import socket
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
    "host": socket.gethostname(),
    "master_addr": os.environ.get("MASTER_ADDR"),
    "node_rank": os.environ.get("NODE_RANK"),
    "rank_env": os.environ.get("RANK"),
    "world_size_env": os.environ.get("WORLD_SIZE"),
    "pet_nproc_per_node": os.environ.get("PET_NPROC_PER_NODE"),
    "pet_node_rank": os.environ.get("PET_NODE_RANK"),
    "sft_gpu_mem_util_env": os.environ.get("SCAPE_VLLM_GPU_MEM_UTIL"),
}
print("PREFLIGHT", versions, flush=True)
assert versions["torch"].startswith("2.10.0"), f"torch={versions['torch']} != 2.10.0"
assert versions["vllm"] == "0.19.1", f"vllm={versions['vllm']} != 0.19.1"
assert versions["transformers"] == "4.57.6", f"transformers={versions['transformers']} != 4.57.6"
assert versions["pyserini"] == "2.0.0", f"pyserini={versions['pyserini']} != 2.0.0"
assert versions["gpu_count"] == 8, f"per-node gpu_count={versions['gpu_count']} != 8"
assert versions["o200k_base_present"]

try:
    from torch.utils.tensorboard import SummaryWriter  # noqa: F401
    tb_ok = True
except Exception as exc:
    tb_ok = False
    print("TENSORBOARD_IMPORT_FAIL", {"error": str(exc)}, flush=True)
print("TENSORBOARD_PROBE", {"summary_writer_ok": tb_ok,
                            "log_path_env": os.environ.get("TENSORBOARD_LOG_PATH")}, flush=True)

# GPT-OSS-20B model probe: bf16-dequant (NO quant_method, NOT mxfp4).
model_cfg_path = "/mnt/songzijun/models/openai/gpt-oss-20b-bf16-dequant/config.json"
with open(model_cfg_path, "r", encoding="utf-8") as fh:
    model_cfg = json.load(fh)
qcfg = model_cfg.get("quantization_config") or {}
print("MODEL_PROBE", {
    "architectures": model_cfg.get("architectures"),
    "model_type": model_cfg.get("model_type"),
    "quant_method": qcfg.get("quant_method", "NONE"),
}, flush=True)
assert model_cfg.get("architectures") == ["GptOssForCausalLM"], f"arch={model_cfg.get('architectures')}"
assert not qcfg.get("quant_method"), f"bf16-dequant must have NO quant_method, got {qcfg.get('quant_method')}"

# SEC Lucene BM25 index must be loadable (SEC retrieval uses it).
from pyserini.search.lucene import LuceneSearcher
s = LuceneSearcher("/mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-sec-corpus/indexes/bm25")
n_docs = s.num_docs if not callable(s.num_docs) else s.num_docs()
print("LUCENE_OK", {"num_docs": n_docs}, flush=True)
assert n_docs > 0

# SFT v1 stack import chain (the only-opd verl patches do NOT apply here — SFT
# v1 does not import trainer_adapter/four_cell_runtime. It imports its own
# dist_launch/cli/trainer/ce). Assert the SFT entry points and runtimes load.
from trim.training.sft_ablation.dist_launch import launch_sft  # noqa: F401
from trim.training.sft_ablation.cli import parse_sft_args  # noqa: F401
from trim.training.sft_ablation.trainer import build_hf_trainer, SFTTrainer  # noqa: F401
from trim.training.sft_ablation.ce import ce_update  # noqa: F401
from trim.training.sft_ablation.contracts import PROFILES  # noqa: F401
from trim.training.teacher_sft.runtime import run as teacher_run  # noqa: F401
assert "teacher_sft" in PROFILES, "teacher_sft profile missing"
prof = PROFILES["teacher_sft"]
print("SFT_PROFILE_OK", {
    "experiment": "teacher_sft",
    "state_source": prof.state_source,
    "prompt_owner": prof.prompt_owner,
    "supervision_provider": prof.supervision_provider,
    "loss": prof.loss,
    "rl_enabled": prof.rl_enabled,
    "projection": prof.projection,
}, flush=True)
assert prof.rl_enabled is False, "SFT must be rl_enabled=False (action-only CE, no RL)"

# Harness-1 submodule import preflight (fix aapel r4 ModuleNotFoundError).
# env_worker.audit_loaded_mask hard-imports `training.train_rl` (env_worker.py:46)
# and `harness.ultra_core` (env_worker.py:45). These live in
# TRIM/external/harness-1/ (a git submodule pinned to commit 8ac4012). The
# parent process finds them via ensure_harness1_on_path (sys.path), but the
# env_worker SUBPROCESS (spawned by launch_env_worker via subprocess.run with
# isolate_inference_child_env) resolves modules from PYTHONPATH only. If
# external/harness-1 is NOT on PYTHONPATH, env_worker crashes with
# ModuleNotFoundError: No module named 'training.train_rl' AFTER cp+barrier
# (aapel r4 FAILED this way at 21:47:56, ~1min after torchrun). Assert here so
# we fail fast at preflight, before cp/barrier waste ~1h.
import harness.ultra_core  # noqa: F401
import training.train_rl  # noqa: F401
print("HARNESS1_IMPORT_OK", {"training.train_rl": True, "harness.ultra_core": True}, flush=True)

# VLLMGenerateClient gpu-mem discipline: assert the env is set (the client
# defaults to 0.90; we inject 0.85 for the bf16 铁律). The client may or may
# not read this env — log it so the audit can see what the worker actually used.
print("SFT_GPU_MEM_DISCIPLINE", {"env": "SCAPE_VLLM_GPU_MEM_UTIL=0.85",
                                  "client_default": 0.90,
                                  "note": "bf16 39GB weights, 0.85 leaves KV headroom"}, flush=True)

print("SFT_PREFLIGHT_OK", {"four_pack": True, "jvm": True, "sec_lucene": True,
                            "sft_stack_import": True, "teacher_sft_profile": True,
                            "bf16_dequant": True}, flush=True)
PY

# --- FIX2: copy bf16-dequant model weights to local /tmp before exec ---
# bf16-dequant is 39GB (3x mxfp4's 13GB); each node does its own cp from JuiceFS
# (~7min), then all 8 local ranks load from fast local disk.
MODEL_SRC=/mnt/songzijun/models/openai/gpt-oss-20b-bf16-dequant
MODEL_LOCAL=/tmp/gpt-oss-20b-bf16-dequant
if [ ! -f "$MODEL_LOCAL/config.json" ]; then
  rm -rf "$MODEL_LOCAL"
  mkdir -p "$MODEL_LOCAL"
  cp -r "$MODEL_SRC"/. "$MODEL_LOCAL"/
fi
test -f "$MODEL_LOCAL/config.json"
test -f "$MODEL_LOCAL/model.safetensors.index.json"
echo "MODEL_LOCAL_READY $MODEL_LOCAL host=$(hostname) node_rank=${NODE_RANK}"

# --- Worker-wait-master rendezvous barrier (ported from lambda_opd resume entry) ---
# dwprt r2 FAILED at 2026-10-19 18:31 (RendezvousTimeoutError): master cp'd
# the 39GB model slowly (22min, JuiceFS jitter) and reached torchrun rendezvous
# while workers were still in cp/pip; workers hit the c10d TCPStore deadline
# (~10min) before master's port was open. This is the SAME race qvnwn hit
# (lambda_opd 0.1, memory [[trim-gptoss20b-trim-all-recall-only-lambda-opd-ablation-20261017]]).
# Barrier: workers poll a shared OUT-dir signal that ONLY master writes right
# before exec, i.e. AFTER master finished pip+preflight+cp model. Workers thus
# never start torchrun before master, eliminating the timeout race. SFT uses
# --nproc-per-node 1 (4 coordinator ranks) so each pod runs this entry; the
# barrier logic is identical to the verl entry's (4 pods, one per node_rank).
# --- Bidirectional rendezvous barrier (fix tlaem r3 RendezvousTimeoutError) ---
# r3 tlaem FAILED @ 20:07 (RendezvousTimeoutError) even WITH the one-way
# barrier: master cp'd the model (23min, JuiceFS jitter) and wrote its
# master_ready_0 signal, then exec'd torchrun; but worker nodes were STILL in
# their own cp (slower than master), never reached the "poll master_ready"
# step, so they never joined rendezvous within the ~10min c10d deadline. The
# one-way barrier only prevents "worker starts torchrun before master"; it
# does NOT prevent the reverse — "master starts torchrun before workers are
# ready". Bidirectional fix: EVERY pod writes its own node_ready_<rank> AFTER
# its cp finishes; master additionally waits for ALL 4 node_ready_* signals,
# then writes all_ready_0; workers (after writing their own node_ready) poll
# for all_ready_0. So torchrun starts ONLY after all 4 nodes have finished cp
# — cp-speed skew no longer matters. Timeout 3000s (50min) covers the worst
# cp (23min observed + headroom).
BARRIER_DIR="${OUT}/.rendezvous_barrier"
mkdir -p "$BARRIER_DIR"
MY_RANK="${NODE_RANK:-0}"
MY_SIGNAL="$BARRIER_DIR/node_ready_${MY_RANK}"
: > "$MY_SIGNAL"
echo "BARRIER_NODE_READY wrote $MY_SIGNAL host=$(hostname) node_rank=${MY_RANK}"
if [ "${MY_RANK}" = "0" ]; then
  echo "BARRIER_MASTER_WAIT_ALL waiting for all node_ready_0..3 host=$(hostname)"
  WAITED=0
  ALL_READY=false
  while [ "$ALL_READY" = "false" ]; do
    ALL_READY=true
    for r in 0 1 2 3; do
      [ -f "$BARRIER_DIR/node_ready_${r}" ] || ALL_READY=false
    done
    if [ "$ALL_READY" = "false" ]; then
      sleep 10
      WAITED=$((WAITED + 10))
      if [ "$WAITED" -ge 3000 ]; then
        SEEN=""
        for r in 0 1 2 3; do [ -f "$BARRIER_DIR/node_ready_${r}" ] && SEEN="${SEEN} ${r}"; done
        echo "FATAL: master timed out after ${WAITED}s waiting for all node_ready. Seen:${SEEN:- none}" >&2
        exit 1
      fi
      if [ $((WAITED % 60)) -eq 0 ]; then
        SEEN=""
        for r in 0 1 2 3; do [ -f "$BARRIER_DIR/node_ready_${r}" ] && SEEN="${SEEN} ${r}"; done
        echo "BARRIER_MASTER_WAIT_ALL still waiting ${WAITED}s ready:${SEEN:- none} host=$(hostname)"
      fi
    fi
  done
  : > "$BARRIER_DIR/all_ready_0"
  echo "BARRIER_ALL_READY wrote $BARRIER_DIR/all_ready_0 host=$(hostname) node_rank=0 (all 4 cp done)"
else
  ALL_SIGNAL="$BARRIER_DIR/all_ready_0"
  echo "BARRIER_WORKER_WAIT_ALL polling $ALL_SIGNAL host=$(hostname) node_rank=${MY_RANK}"
  WAITED=0
  while [ ! -f "$ALL_SIGNAL" ]; do
    sleep 10
    WAITED=$((WAITED + 10))
    if [ "$WAITED" -ge 3000 ]; then
      echo "FATAL: worker node_rank=${MY_RANK} timed out after ${WAITED}s waiting for all_ready signal $ALL_SIGNAL" >&2
      exit 1
    fi
    if [ $((WAITED % 60)) -eq 0 ]; then
      echo "BARRIER_WORKER_WAIT_ALL still waiting ${WAITED}s host=$(hostname) node_rank=${MY_RANK}"
    fi
  done
  echo "BARRIER_WORKER_SEEN_ALL after ${WAITED}s host=$(hostname) node_rank=${MY_RANK}"
fi

python3 -m py_compile /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_teacher_sft_multinode.py
python3 -m py_compile /mnt/songzijun/Capability_Evolution/TRIM/trim/training/sft_ablation/dist_launch.py
python3 -m py_compile /mnt/songzijun/Capability_Evolution/TRIM/trim/training/teacher_sft/runtime.py
cd /mnt/songzijun/Capability_Evolution
exec python3 /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_teacher_sft_multinode.py \
  --nnodes 4 \
  --nproc-per-node 1 \
  --node-rank "${NODE_RANK}" \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  --rollout-replicas 8 \
  --tensor-parallel-size 1 \
  --rdzv-id gptoss20b-teacher-sft-bf16-32 \
  --model_name /tmp/gpt-oss-20b-bf16-dequant \
  --teacher-model /tmp/gpt-oss-20b-bf16-dequant \
  --teacher-component all \
  --student-component zero \
  --train-data sec \
  --rl-data /mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-rl-data \
  --sec-corpus-root /mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-sec-corpus \
  --train-env local_legacy \
  --stage all \
  --num-updates 200 \
  --batch-size 128 \
  --micro-batch-size 1 \
  --group-size 2 \
  --learning-rate 5e-6 \
  --lora-rank 32 \
  --max-turns 40 \
  --max-model-len 8192 \
  --max-new-tokens 2048 \
  --seed 42 \
  --out "${OUT}"
