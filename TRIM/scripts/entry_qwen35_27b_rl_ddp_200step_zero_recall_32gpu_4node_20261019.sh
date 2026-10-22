#!/usr/bin/env bash
# === Qwen3.5-27B rl/zero + final-answer-recall reward (2026-10-19) ===
# Fork of entry_qwen3_4b_rl_ddp_200step_zero_32gpu_4node_20260925.sh (the
# verified Qwen3-4B multi-node rl/zero baseline, 9ixnl, 57.9h/200step SUCCEED)
# with TWO changes vs that baseline:
#   1. Model: Qwen3-4B-Instruct-2507 (dense 4B, 7.6GB) -> Qwen3.5-27B (dense 27B,
#      51.75GB bf16, Qwen3_5ForConditionalGeneration, mixed linear/full attention).
#   2. Reward: 9ixnl ran the DEFAULT 4-term reward (no --rl-reward). THIS entry
#      adds --rl-reward final-answer-recall (the EXPLICIT path per CLAUDE.md
#      2026-10-15 澄清: terminal_reward_breakdown(rl_reward="final-answer-recall")
#      -> reward_policy.score_explicit_reward, pure final_answer_recall metric,
#      NO env read, NO compute_logprobs, NO extra logits forward). This is the
#      "only fa recall" reward the user asked for, on the REAL rl/zero path
#      (rl_enabled=True, reward fed to CISPO policy-gradient backward).
#
# === PATH: rl/zero (H_zero RL baseline, NOT trim/all distill, NOT only-opd) ===
# --train_method rl --component zero = pure on-policy RL with H_zero (no teacher
# projection). rl/zero does NOT touch the curate-clipping code path (x2j3u
# FAILED@step0 was trim/all's teacher curate projection; rl/zero has no teacher
# curate refs). Hence PATCH_CHECK below asserts only the three DDP-generic
# patches (NOT the trim/all-specific curate-clipping patch). This matches the
# 9ixnl entry's PATCH_CHECK shape. The reward-explicit-path asserts are added
# on top (ported from entry_gpt_oss_20b_only_opd_recall_32gpu_4node_20261015.sh,
# which verified REWARD_EXPLICIT_PATH_OK).
#
# === Qwen3.5-27B vs Qwen3-4B differences (this entry) ===
#   * Dense bf16 27B (51.75GB), NOT 4B (7.6GB). Same Qwen3 family but the 27B
#     uses Qwen3_5ForConditionalGeneration (model_type qwen3_5, text_config
#     mixed linear_attention/full_attention layers, attn_output_gate,
#     full_attention_interval=4, image_token_id present = multimodal-capable
#     base, but training is text-only). vLLM 0.19.1 in this image already has
#     Qwen3_5 registered in ModelRegistry (eval jobs fcanl/iytmg/bg3xs verified
#     2026-10-15). Training uses the SAME vllm_rollout_worker.py:126-170 LLM()
#     construction (per-rank TP=1), so vLLM model loading behaves identically to
#     eval. The ARCH_OK ModelRegistry probe below is the authoritative check.
#     Training rollout does NOT use vLLM's serve tool-calling API (the env parses
#     tool calls from generated text itself), so the eval-only
#     --reasoning-parser qwen3 strip concern does NOT apply here.
#   * mb=2 (NOT 4, NOT 1): user-requested. 27B dense CAN enable gradient
#     checkpointing (fsdp2_actor.py:321-327 calls gradient_checkpointing_enable
#     unconditionally for all HF models, including Qwen3_5; unlike GPT-OSS MoE
#     where grad-ckpt is insufficient for expert bmm peaks). With grad-ckpt, the
#     backward activation footprint is sqrt-L-tier (layer-boundary checkpoints)
#     not full-L展开, so mb2 is safe: 51.75GB weights + ~8GB mb2 grad-ckpt
#     storage + ~3GB recompute peak + ~12GB LoRA/grad/CUDA/fragmentation overhead
#     ~= 75GB/80GB (5GB headroom). mb4 would be ~83GB (risky). mb1 (71GB) is the
#     conservative fallback if mb2 OOMs at step0. This is DENSE so mb CAN scale
#     (vs GPT-OSS MoE locked to mb1) — the user's expectation that 27B dense
#     trains faster than GPT-OSS MoE holds for the BACKWARD stage (mb2 > mb1
#     throughput); rollout is still slower than 4B (27B vLLM gen is ~30% slower
#     than 20B, and 0.80 gpu-mem leaves only ~12GB KV vs 0.85's ~28GB).
#   * gpu-mem 0.90 (FULL utilization, NOT 0.80): affects ONLY the rollout phase
#     (vLLM KV cache budget). torch_ddp_lora is scheme-A exclusive so vLLM is
#     shutdown during backward and FULLY releases its memory — 9ixnl 铁证: it
#     ran 0.90 gpu-mem but backward peak was only 18-28GB (vLLM's KV pool was
#     released, not held). So gpu-mem has ZERO effect on backward OOM; raising
#     it only buys rollout KV cache (higher concurrency = faster rollout).
#     0.90 -> ~72GB budget, 51.75GB weights leaves ~20GB KV (vs 0.80's ~12GB);
#     27B eval (boot_qwen35_27b_*_eval*.sh) used 0.90 single-actor and loaded
#     fine, so 0.90 is verified for 27B vLLM loading. This is the "充分利用显存"
#     choice — max KV for fastest rollout, no backward cost.
#   * SCAPE_FORCE_LOCAL_HARMONY=1 is SET (carried from 9ixnl; Qwen3.5 is HF chat
#     family, does NOT hit the openai_harmony to_json crash that gpt-oss hits, so
#     the env is harmless and matches the verified Qwen path).
#   * Model cp to /tmp: 27B is 51.75GB; each node does its own cp from JuiceFS
#     (~5-7min), then all 8 local ranks load from fast local disk. Runs BEFORE
#     vLLM worker spawn so it does NOT count against the 3600s worker-start
#     window. 4 nodes x 8 ranks = 32 concurrent JuiceFS readers risk IO contention
#     on first step; the per-node local cp avoids the t-cucac-style concurrent-
#     read hang.
#
# === All known torch_ddp_lora pitfalls (carried from 9ixnl + 7nzzn) ===
#   * trainer_adapter.py:478 startup_timeout_s=3600.0 (FIX for 900s worker-start
#     timeout when DDP ranks each start their own vLLM engine from disk; 27B
#     51.75GB cp + load is heavier than 4B, 3600s headroom is essential).
#   * trainer_adapter.py:264 resolve_queries 4-tuple unpack (FIX for 724d9fa
#     ValueError: too many values to unpack).
#   * trainer_adapter.py:506 client.start() EADDRINUSE retry x3 (FIX for the
#     per-rank vLLM TP=1 TCPStore port race at step>=1; 32 ranks across 4 nodes
#     make the port race MORE likely, the retry is essential).
#
# torch_ddp_lora path: NO CUDA_LAUNCH_BLOCKING (DDP is real multi-proc torchrun).
# NO DISABLE_CUSTOM_ALL_REDUCE (per-rank vLLM TP=1, no TP all-reduce to fix).
# trainer_adapter.py:181 hardcodes init_dist_if_needed(backend="nccl"); 7nzzn
# verified PyTorch auto-uses Gloo for rendezvous + object collectives
# (broadcast_object_list / all_gather_object) across 4 nodes, nccl only for DDP
# gradient all-reduce. ([Gloo] Rank 0-31 is connected to 31 peer ranks.)
set -euxo pipefail

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TRIM_GPU_KEEPALIVE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/mnt/songzijun/Capability_Evolution/TRIM:/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD

# --- Multi-node distributed env guard ---
# CloudML custom_train framework=pytorch nodeNumber=4 must give each pod a
# distinct NODE_RANK and a shared MASTER_ADDR (node0's service DNS). If these
# are absent the launcher cannot form a 4-node rendezvous.
#
# 7nzzn/9ixnl verified CloudML injects MASTER_ADDR + NODE_RANK but NOT RANK/
# WORLD_SIZE/PET_*. The defensive unset below is kept as insurance: if a future
# CRD change injects RANK/WORLD_SIZE, under_torchrun() in run_train_multinode.py
# would short-circuit main() straight into _worker WITHOUT launching torchrun,
# so the 4 nodes would never rendezvous into a world_size=32 group. To make the
# launch deterministic regardless of what CloudML injects, we:
#   1. Capture MASTER_ADDR + NODE_RANK (the two things we actually need from the
#      scheduler) BEFORE clearing injected rank env.
#   2. unset RANK / WORLD_SIZE / LOCAL_RANK / LOCAL_WORLD_SIZE / PET_* / the
#      torchelastic markers, so under_torchrun() returns False and main() takes
#      the needs_torchrun() -> _exec_torchrun branch (self-launched torchrun).
# We then pass --node-rank / --master-addr explicitly, so parse_dist_argv does
# not rely on any injected env.
NODE_RANK="${NODE_RANK:-${PET_NODE_RANK:-${NODE_INDEX:-${NODE_NUMBER:-}}}}"
if [ -z "${NODE_RANK:-}" ]; then
  echo "FATAL: NODE_RANK is not set. Each pod needs its own rank id (0..3)." >&2
  exit 1
fi
if [ -z "${MASTER_ADDR:-}" ]; then
  echo "FATAL: MASTER_ADDR is not set. CloudML custom_train multi-node must inject" \
       "node0's IP as MASTER_ADDR. If this pod has no MASTER_ADDR, the job will" \
       "fail to rendezvous. Aborting to avoid 4 independent single-node runs." >&2
  exit 1
fi
: "${MASTER_PORT:=29500}"
export MASTER_ADDR NODE_RANK MASTER_PORT

# Strip any CloudML-injected torchrun-style rank env so run_train_multinode.py's
# under_torchrun() (which keys on WORLD_SIZE>1 or non-auto PET_NPROC_PER_NODE)
# returns False and main() self-launches torchrun via _exec_torchrun. This makes
# the 4-node rendezvous deterministic instead of dependent on injection flavor.
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK \
      GROUP_WORLD_SIZE TORCHELASTIC_RUN_ID TORCHELASTIC_RESTART_COUNT \
      TORCHELASTIC_MAX_RESTARTS TORCHELASTIC_USE_AGENT_STORE \
      PET_MASTER_ADDR PET_MASTER_PORT PET_NNODES PET_NPROC_PER_NODE PET_NODE_RANK

echo "MULTINODE_ENV master_addr=${MASTER_ADDR} node_rank=${NODE_RANK} master_port=${MASTER_PORT} host=$(hostname)"

# --- Shared output directory (MUST be identical on every node) ---
# `run_train_multinode.py` writes per-rank rollout shards to <out>/tmp/rollout_shards
# and rank 0 interleaves them via all_gather_via_disk. If each node uses a different
# <out> (e.g. a per-pod $(date) timestamp), the shards are invisible across nodes and
# step 0 backward crashes with FileNotFoundError. OUT is injected via submit --env
# so all 4 pods share the exact same path. Never use $(date) here.
if [ -z "${OUT:-}" ]; then
  echo "FATAL: OUT is not set. Multi-node requires a shared --out path injected via --env OUT=... (identical on every pod). Aborting." >&2
  exit 1
fi
export OUT
echo "MULTINODE_OUT out=${OUT} host=$(hostname) node_rank=${NODE_RANK}"

# Qwen3.5-27B: SCAPE_FORCE_LOCAL_HARMONY=1 is SET (harmless on Qwen3.5 HF chat
# family, matches the verified 9ixnl Qwen3-4B path; Qwen3.5 does NOT go through
# openai_harmony Rust encoder so no _LocalConversation to_json crash). Placeholder
# key so pyserini.encode._openai module-level openai.OpenAI() does not raise
# OpenAIError at import (RL uses local BM25, never calls OpenAI API).
export OPENAI_API_KEY="${OPENAI_API_KEY:-**-****-***********-***-*-****-***}"
export SCAPE_LOCAL_OPENAI_EMBEDDINGS=1
export SCAPE_FORCE_LOCAL_HARMONY=1
export SCAPE_CHROMA_PATH=/tmp/empty_chroma
mkdir -p "$SCAPE_CHROMA_PATH"

# tiktoken bundle: harness import chain (openai_harmony) still needs o200k_base.
python3 -c "import zipfile; zipfile.ZipFile('/mnt/songzijun/tiktoken-bundle.zip').extractall('/tmp')"
test -f /tmp/tiktoken-bundle/share/tiktoken/o200k_base.tiktoken
test -f /tmp/tiktoken-bundle/share/tiktoken/cl100k_base.tiktoken
export TIKTOKEN_ENCODINGS_BASE=/tmp/tiktoken-bundle/share/tiktoken
export TIKTOKEN_RS_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_rs_cache
export TIKTOKEN_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_cache

# --- Presence checks on the mounted JuiceFS tree ---
nvidia-smi -L
test -f /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_train_multinode.py
test -f /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_train.py
test -f /mnt/songzijun/models/Qwen3.5-27B/config.json
test -f /mnt/songzijun/models/Qwen3.5-27B/model.safetensors.index.json
test -f /mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-rl-data/manifest.json
test -f /mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-sec-corpus/manifest.json

# --- JVM discovery for Pyserini/Lucene (local BM25 retrieval) ---
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

# --- Install four-pack + harness import chain deps (internal Xiaomi PyPI) ---
python3 -m pip install --no-cache-dir --ignore-installed --upgrade-strategy only-if-needed \
  "torch==2.10.0" "transformers==4.57.6" "accelerate==1.13.0" "datasets==4.8.4" \
  "hydra-core==1.3.2" "omegaconf==2.3.0" "vllm==0.19.1" peft structlog chromadb \
  tqdm regex sympy latex2sympy2 word2number sentencepiece pyserini openai tenacity anthropic

# --- ANSERINI_CLASSPATH for PyJNI/Lucene ---
PYSERINI_ROOT=$(python3 -c 'import pathlib, pyserini; print(pathlib.Path(pyserini.__file__).resolve().parent)')
export ANSERINI_CLASSPATH=$(find "$PYSERINI_ROOT/resources/jars" -type f -name '*.jar' -print | paste -sd: -)
test -n "$ANSERINI_CLASSPATH"
export CLASSPATH="$ANSERINI_CLASSPATH"

# --- Preflight: assert four-pack versions, 8 GPUs (per node), JVM, import chain, Lucene ---
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
}
print("PREFLIGHT", versions, flush=True)
assert versions["torch"].startswith("2.10.0"), f"torch={versions['torch']} != 2.10.0"
assert versions["vllm"] == "0.19.1", f"vllm={versions['vllm']} != 0.19.1"
assert versions["transformers"] == "4.57.6", f"transformers={versions['transformers']} != 4.57.6"
assert versions["pyserini"] == "2.0.0", f"pyserini={versions['pyserini']} != 2.0.0"
# Per-node: each pod must see 8 GPUs (pinned 1-per-rank later by pin_local_cuda_device).
assert versions["gpu_count"] == 8, f"per-node gpu_count={versions['gpu_count']} != 8"
assert versions["o200k_base_present"]

# Qwen3.5-27B model probe: dense bf16 (NOT quantized). model_type qwen3_5,
# architectures Qwen3_5ForConditionalGeneration (multimodal-capable base, but
# training is text-only). No triton/Marlin/backend dance needed (dense bf16).
model_cfg_path = "/mnt/songzijun/models/Qwen3.5-27B/config.json"
with open(model_cfg_path, "r", encoding="utf-8") as fh:
    model_cfg = json.load(fh)
qcfg = model_cfg.get("quantization_config") or {}
tc = model_cfg.get("text_config") or {}
print("MODEL_PROBE", {
    "architectures": model_cfg.get("architectures"),
    "model_type": model_cfg.get("model_type"),
    "text_config.model_type": tc.get("model_type"),
    "torch_dtype": model_cfg.get("torch_dtype") or tc.get("dtype"),
    "quant_method": qcfg.get("quant_method", "none"),
}, flush=True)
assert model_cfg.get("architectures") == ["Qwen3_5ForConditionalGeneration"], \
    f"unexpected architectures={model_cfg.get('architectures')}"
assert not qcfg.get("quant_method"), \
    f"Qwen3.5-27B must be dense bf16, got quant_method={qcfg.get('quant_method')}"

# CRITICAL: probe vLLM ModelRegistry for Qwen3_5ForConditionalGeneration.
# vLLM 0.19.1 in this image is expected to already have Qwen3_5 registered
# (eval jobs fcanl/iytmg/bg3xs verified 2026-10-15). Training uses the same
# vllm_rollout_worker.py LLM() construction (per-rank TP=1) as eval, so vLLM
# model loading behaves identically. This probe is authoritative — if the arch
# is NOT registered, vLLM init will crash at step0 rollout and we fail fast here
# instead of wasting ~20min of 32-GPU rollout time.
arch = "Qwen3_5ForConditionalGeneration"
try:
    from vllm.model_executor.models.registry import ModelRegistry
    supported = set(ModelRegistry.get_supported_archs())
    if arch in supported:
        print("ARCH_OK", {"arch": arch, "registered": True}, flush=True)
    else:
        qwen_archs = sorted(a for a in supported if "Qwen" in a)
        print("ARCH_NOT_SUPPORTED", {"arch": arch, "qwen3_family_registered": qwen_archs[:15]}, flush=True)
        raise SystemExit(f"vLLM 0.19.1 does NOT have {arch} registered; training cannot load Qwen3.5-27B")
except SystemExit:
    raise
except Exception as exc:
    print("ARCH_PROBE_ERROR", {"arch": arch, "error": str(exc)}, flush=True)
    raise

from pyserini.search.lucene import LuceneSearcher
s = LuceneSearcher("/mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-sec-corpus/indexes/bm25")
n_docs = s.num_docs if not callable(s.num_docs) else s.num_docs()
print("LUCENE_OK", {"num_docs": n_docs}, flush=True)
assert n_docs > 0
# DDP-LoRA wrap probe: torch_ddp_lora backend must select the plain "ddp" actor
# wrap (full replica per rank), NOT the FSDP2 sharded wrap.
from trim.integrations.verl.trainer_adapter import _actor_wrap
assert _actor_wrap("torch_ddp_lora") == "ddp"
print("DDP_LORA_WRAP_OK", {"wrap": _actor_wrap("torch_ddp_lora")}, flush=True)
# Verify the three code-tree patches are in place (so we don't silently regress).
# rl/zero path has NO teacher curate projection, so the curate-clipping patch
# (trim/all-specific, ethqp fix) is NOT required here — only the 3 DDP-generic
# patches. This matches the 9ixnl entry's PATCH_CHECK.
import pathlib, re
ta = pathlib.Path("/mnt/songzijun/Capability_Evolution/TRIM/trim/integrations/verl/trainer_adapter.py").read_text()
assert "startup_timeout_s=3600.0" in ta, "startup_timeout_s=3600 patch missing"
assert re.search(r"train_rows,\s*eval_rows,\s*pool_meta,\s*_?frozen_points\s*=\s*resolve_queries", ta), "4-tuple unpack patch missing"
assert "EADDRINUSE" in ta, "EADDRINUSE retry patch missing"

# Verify the reward path. Per CLAUDE.md 2026-10-15 澄清 there are TWO paths:
#   (A) EXPLICIT --rl-reward final-answer-recall -> terminal_reward_breakdown()
#       at four_cell_runtime.py:1144 dispatches to reward_policy.score_explicit_reward
#       (pure final_answer_recall metric, reads NO env, no compute_logprobs). This
#       is what we run. rl/zero has rl_enabled=True (resolve_trim_ablation("rl")
#       returns None -> default rl_enabled=True at launch.py:1312-1313), so the
#       reward IS fed to the CISPO policy-gradient backward (unlike only-opd
#       where rl_enabled=False and reward only ranks OPD experiences).
#   (B) LEGACY rl_reward=None -> _legacy_terminal_reward_breakdown() reads
#       TRIM_REWARD_MODE env (recall / recall_turn). NOT used here (no env set).
# Assert path A's wiring is present; do NOT assert legacy 'total = rec' text
# (that lives in the legacy fn we deliberately don't take, and reward_policy
# legitimately has a "total": -0.2 invalid-episode branch, so asserting its
# absence would false-positive).
fc = pathlib.Path("/mnt/songzijun/Capability_Evolution/TRIM/trim/training/four_cell_runtime.py").read_text()
assert "def terminal_reward_breakdown(" in fc, "explicit-reward dispatch fn missing"
assert "from trim.training.reward_policy import score_explicit_reward" in fc, "explicit path not wired"
assert "rl_reward is None" in fc, "explicit/legacy dispatch branch missing"
rp = pathlib.Path("/mnt/songzijun/Capability_Evolution/TRIM/trim/training/reward_policy.py").read_text()
assert "EXPLICIT_RL_REWARDS" in rp and '"final-answer-recall"' in rp, "final-answer-recall not registered explicit"
assert "PROTOCOL_FA_ONLY" in rp, "FA-only protocol constant missing"
assert "def score_explicit_reward" in rp, "score_explicit_reward missing"
from trim.training.reward_policy import EXPLICIT_RL_REWARDS, reward_protocol_for, score_explicit_reward  # noqa: E402
assert "final-answer-recall" in EXPLICIT_RL_REWARDS
assert reward_protocol_for("final-answer-recall") == "trim_final_answer_recall_only_v1"
print("REWARD_EXPLICIT_PATH_OK", {"rl_reward": "final-answer-recall",
                                  "path": "score_explicit_reward (no env, no logprobs)",
                                  "rl_enabled_for_rl_method": True}, flush=True)
print("PATCH_CHECK_OK", {"startup_timeout_s": True, "resolve_queries_4tuple": True,
                          "eaddrinuse_retry": True, "reward_explicit_path": True,
                          "qwen35_arch": True}, flush=True)
PY

# --- Copy Qwen3.5-27B model weights to local /tmp before exec (defensive) ---
# 27B is 51.75GB; each node does its own cp from JuiceFS (~5-7min), then all 8
# local ranks load from fast local disk. Runs BEFORE vLLM worker spawn so it
# does NOT count against the 3600s worker-start window. 4 nodes x 8 ranks = 32
# concurrent JuiceFS readers risk IO contention on first step; the per-node
# local cp avoids the t-cucac-style concurrent-read hang.
MODEL_SRC=/mnt/songzijun/models/Qwen3.5-27B
MODEL_LOCAL=/tmp/qwen35-27b
if [ ! -f "$MODEL_LOCAL/config.json" ]; then
  rm -rf "$MODEL_LOCAL"
  mkdir -p "$MODEL_LOCAL"
  cp -r "$MODEL_SRC"/. "$MODEL_LOCAL"/
fi
test -f "$MODEL_LOCAL/config.json"
test -f "$MODEL_LOCAL/model.safetensors.index.json"
echo "MODEL_LOCAL_READY $MODEL_LOCAL host=$(hostname) node_rank=${NODE_RANK}"

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
  --rdzv-id qwen35-27b-rl-zero-recall-32 \
  --training-backend torch_ddp_lora \
  --harness Harness-1 \
  --benchmark bcplus_full \
  --model_name /tmp/qwen35-27b \
  --train_method rl \
  --component zero \
  --train-env local_legacy \
  --train-data sec \
  --rl-data /mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-rl-data \
  --sec-corpus-root /mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-sec-corpus \
  --rl-reward final-answer-recall \
  --train-steps 200 \
  --train-groups-per-step 32 \
  --group-size 8 \
  --train-micro-batch-size 2 \
  --train-heartbeat-every 8 \
  --max-turns 40 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --out "${OUT}"
