#!/usr/bin/env bash
# Container entry for Harness-1 BC+ full RL (CISPO, H_zero baseline,
# --train_method rl --component zero) Qwen3-4B-Instruct-2507 (original bf16,
# dense 4B, NOT dequant) via torch_ddp_lora backend on FOUR 8-GPU gm402-80g
# nodes (4 nodes x 8 cards = 32 GPUs, 200 steps).
#
# === THIS VARIANT: final-answer recall + turn-count penalty reward ===
# Built on the 9ixnl-verified Qwen3-4B rl/zero 32gpu entry
# (entry_qwen3_4b_rl_ddp_200step_zero_32gpu_4node_20260925.sh). The only
# behavioral change vs 9ixnl is the reward formula:
#
#   * The TRIM torch_ddp_lora path computes per-episode terminal reward in
#     trim.training.four_cell_runtime.terminal_reward_breakdown. That function
#     was already collapsed to pure final-answer recall (total = rec,
#     protocol "final_answer_recall_only") in the working tree; this entry
#     additionally enables the turn-count penalty by exporting
#     TRIM_REWARD_MODE=recall_turn. The reward becomes:
#
#         total = final_answer_recall - TRIM_TURN_PENALTY * (n_turns / TRIM_TURN_NORM)
#
#     where n_turns = len(actions) (the episode's executed action count,
#     truncated when the env hits `ended`), capped at 1.0 when
#     n_turns >= TRIM_TURN_NORM. Defaults: TRIM_TURN_PENALTY=0.15,
#     TRIM_TURN_NORM=40 (matches --max-turns 40). So a full-recall episode
#     still scores highest; the penalty only opens intra-group reward spread
#     across rollouts of different lengths so CISPO/advantage normalization
#     can prefer more concise trajectories.
#   * The legal/shaping components of the legacy 4-term reward are already
#     decoupled from `total` (kept only as diagnostic fields). This entry
#     does NOT re-enable them.
#   * harness-1's own external/train_rl.py 4-term compute_reward
#     (external/harness-1/harness/ultra_core.py) is a SEPARATE training loop
#     that this torch_ddp_lora path does NOT call; only four_cell_runtime's
#     terminal_reward_breakdown is on the active path.
#
# === TensorBoard + JSONL metrics ===
# trainer_adapter._make_tensorboard_writer already honors the CloudML-injected
# $TENSORBOARD_LOG_PATH (when enableTensorboard=true) and falls back to
# <out>/<cell>/tb. It writes reward/ loss/ cispo/ tokens/ timing_s/ mem/
# scalar blocks every optimizer step (plus <out>/<cell>/metrics.jsonl,
# step_timing.jsonl, turn_samples.jsonl as JSONL fallback). This entry
# additionally surfaces turn_penalty_mean / n_turns_mean in the reward block
# so the turn-penalty effect is visible in the UI and in metrics.jsonl.
#
# === Everything else identical to 9ixnl ===
#   * Dense bf16 4B (7.6G), NOT mxfp4. NO triton_kernels / Marlin / PTX dance.
#   * SCAPE_FORCE_LOCAL_HARMONY=1 SET (harmless on Qwen3-4B is_hf_chat_family).
#   * mb=4 (Qwen3-4B dense backward can grad-ckpt; NOT the gpt-oss mb1 floor).
#   * Model cp to /tmp per node (4-way JuiceFS read, not 32-way).
#   * PATCH_CHECK asserts the 3 DDP-generic patches (startup_timeout_s=3600,
#     resolve_queries 4-tuple, EADDRINUSE retry). rl/zero has no teacher
#     curate projection, so the trim/all curate-clipping patch is NOT required.
#   * Multi-node: entry unsets CloudML-injected RANK/WORLD_SIZE/PET_* (after
#     capturing MASTER_ADDR/NODE_RANK) so run_train_multinode.py main() takes
#     the needs_torchrun() -> _exec_torchrun branch and self-launches
#     torchrun --nnodes 4 --nproc_per_node 8 (world_size=32). --out is shared
#     via --env OUT (multi-node铁律; never per-pod $(date)).
set -euxo pipefail

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TRIM_GPU_KEEPALIVE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/mnt/songzijun/Capability_Evolution/TRIM:/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD

# --- Reward formula: final-answer recall + turn-count penalty ---
# TRIM_REWARD_MODE=recall_turn enables `total = recall - turn_penalty*(n_turns/norm)`
# in trim.training.four_cell_runtime.terminal_reward_breakdown. Leaving it
# unset (or =recall) keeps the pure-recall baseline. TRIM_TURN_PENALTY is the
# max penalty subtracted (saturates at n_turns>=TRIM_TURN_NORM); TRIM_TURN_NORM
# normalizes the turn count (40 == --max-turns 40, so 40 actions => full penalty).
export TRIM_REWARD_MODE=recall_turn
export TRIM_TURN_PENALTY=0.15
export TRIM_TURN_NORM=40

# --- Multi-node distributed env guard ---
# 9ixnl/7nzzn verified CloudML custom_train framework=pytorch nodeNumber=4
# injects MASTER_ADDR (master pod service DNS) + NODE_RANK but NOT
# RANK/WORLD_SIZE/PET_*. The defensive unset below is insurance: if a future
# CRD change injects RANK/WORLD_SIZE, under_torchrun() in
# run_train_multinode.py would short-circuit main() straight into _worker
# WITHOUT launching torchrun, so the 4 nodes would never rendezvous. To make
# the launch deterministic we capture MASTER_ADDR + NODE_RANK first, then
# clear the rank env so main() self-launches torchrun via _exec_torchrun.
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
: "${MASTER_PORT:=29500}"
export MASTER_ADDR NODE_RANK MASTER_PORT

unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK \
      GROUP_WORLD_SIZE TORCHELASTIC_RUN_ID TORCHELASTIC_RESTART_COUNT \
      TORCHELASTIC_MAX_RESTARTS TORCHELASTIC_USE_AGENT_STORE \
      PET_MASTER_ADDR PET_MASTER_PORT PET_NNODES PET_NPROC_PER_NODE PET_NODE_RANK

echo "MULTINODE_ENV master_addr=${MASTER_ADDR} node_rank=${NODE_RANK} master_port=${MASTER_PORT} host=$(hostname)"
echo "REWARD_ENV reward_mode=${TRIM_REWARD_MODE} turn_penalty=${TRIM_TURN_PENALTY} turn_norm=${TRIM_TURN_NORM}"

# --- Shared output directory (MUST be identical on every node) ---
# run_train_multinode.py writes per-rank rollout shards to <out>/tmp/rollout_shards
# and rank 0 interleaves them via all_gather_via_disk. If each node uses a
# different <out> the shards are invisible across nodes and step 0 backward
# crashes with FileNotFoundError. OUT is injected via submit --env so all 4
# pods share the exact same path. Never use $(date) here.
if [ -z "${OUT:-}" ]; then
  echo "FATAL: OUT is not set. Multi-node requires a shared --out path injected via --env OUT=... (identical on every pod). Aborting." >&2
  exit 1
fi
export OUT
echo "MULTINODE_OUT out=${OUT} host=$(hostname) node_rank=${NODE_RANK}"

# Qwen3-4B: SCAPE_FORCE_LOCAL_HARMONY=1 is SET (harmless on Qwen). Placeholder
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
test -f /mnt/songzijun/models/Qwen3-4B-Instruct-2507/config.json
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
# tensorboard is REQUIRED for real SummaryWriter impl: torch.utils.tensorboard
# is a stub that imports the real `tensorboard` package; without it installed,
# _make_tensorboard_writer's `from torch.utils.tensorboard import SummaryWriter`
# raises ModuleNotFoundError and the try/except degrades to None (training
# survives but no TB curves). The image has no tensorboard, so we install it
# here. CloudML `--enable_tensorboard` injects $TENSORBOARD_LOG_PATH which the
# writer reads (we must NOT export that env ourselves — see header note).
python3 -m pip install --no-cache-dir --ignore-installed --upgrade-strategy only-if-needed \
  "torch==2.10.0" "transformers==4.57.6" "accelerate==1.13.0" "datasets==4.8.4" \
  "hydra-core==1.3.2" "omegaconf==2.3.0" "vllm==0.19.1" peft structlog chromadb \
  tqdm regex sympy latex2sympy2 word2number sentencepiece pyserini openai tenacity anthropic \
  tensorboard

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
    "reward_mode": os.environ.get("TRIM_REWARD_MODE"),
    "turn_penalty": os.environ.get("TRIM_TURN_PENALTY"),
    "turn_norm": os.environ.get("TRIM_TURN_NORM"),
    "tensorboard_log_path": os.environ.get("TENSORBOARD_LOG_PATH"),
}
print("PREFLIGHT", versions, flush=True)
assert versions["torch"].startswith("2.10.0"), f"torch={versions['torch']} != 2.10.0"
assert versions["vllm"] == "0.19.1", f"vllm={versions['vllm']} != 0.19.1"
assert versions["transformers"] == "4.57.6", f"transformers={versions['transformers']} != 4.57.6"
assert versions["pyserini"] == "2.0.0", f"pyserini={versions['pyserini']} != 2.0.0"
assert versions["gpu_count"] == 8, f"per-node gpu_count={versions['gpu_count']} != 8"
assert versions["o200k_base_present"]
assert versions["reward_mode"] == "recall_turn", \
    f"reward_mode must be recall_turn, got {versions['reward_mode']}"
# Qwen3-4B model probe: dense bf16 (NOT mxfp4). No triton/Marlin backend needed.
model_cfg_path = "/mnt/songzijun/models/Qwen3-4B-Instruct-2507/config.json"
with open(model_cfg_path, "r", encoding="utf-8") as fh:
    model_cfg = json.load(fh)
qcfg = model_cfg.get("quantization_config") or {}
print("MODEL_PROBE", {
    "architectures": model_cfg.get("architectures"),
    "model_type": model_cfg.get("model_type"),
    "torch_dtype": model_cfg.get("torch_dtype"),
    "quant_method": qcfg.get("quant_method", "none"),
}, flush=True)
assert model_cfg.get("architectures") == ["Qwen3ForCausalLM"], \
    f"unexpected architectures={model_cfg.get('architectures')}"
assert not qcfg.get("quant_method"), \
    f"Qwen3-4B must be dense bf16, got quant_method={qcfg.get('quant_method')}"
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
# (trim/all-specific) is NOT required here — only the 3 DDP-generic patches.
import pathlib, re
ta = pathlib.Path("/mnt/songzijun/Capability_Evolution/TRIM/trim/integrations/verl/trainer_adapter.py").read_text()
assert "startup_timeout_s=3600.0" in ta, "startup_timeout_s=3600 patch missing"
assert re.search(r"train_rows,\s*eval_rows,\s*pool_meta,\s*_?frozen_points\s*=\s*resolve_queries", ta), "4-tuple unpack patch missing"
assert "EADDRINUSE" in ta, "EADDRINUSE retry patch missing"
print("PATCH_CHECK_OK", {"startup_timeout_s": True, "resolve_queries_4tuple": True, "eaddrinuse_retry": True}, flush=True)
# Reward patch probe: confirm the turn-penalty env-gated branch is present in
# four_cell_runtime.terminal_reward_breakdown (so a stale image that lacks the
# edit would fail loudly instead of silently running pure-recall).
fcr = pathlib.Path("/mnt/songzijun/Capability_Evolution/TRIM/trim/training/four_cell_runtime.py").read_text()
assert 'TRIM_REWARD_MODE' in fcr and 'recall_turn' in fcr and 'turn_penalty' in fcr, \
    "recall+turn-penalty reward patch missing in four_cell_runtime.terminal_reward_breakdown"
print("REWARD_PATCH_OK", {"reward_mode_env": True, "recall_turn_branch": True, "turn_penalty_field": True}, flush=True)
# TensorBoard writer probe: confirm _make_tensorboard_writer honors
# $TENSORBOARD_LOG_PATH (CloudML enableTensorboard=true injects it), and that
# the real SummaryWriter impl is importable (torch.utils.tensorboard is a stub
# that needs the `tensorboard` package installed above; without it the writer
# degrades to None and TB curves silently never appear).
assert "TENSORBOARD_LOG_PATH" in ta, "tensorboard writer must honor $TENSORBOARD_LOG_PATH"
# NOTE: we read $TENSORBOARD_LOG_PATH only for the preflight line below; we
# never export/overwrite it (the platform injects it; manual export would
# shadow the platform path and the UI would be empty).
print("TENSORBOARD_WRITER_OK", {
    "honors_env": True,
    "platform_injected_path": os.environ.get("TENSORBOARD_LOG_PATH"),
}, flush=True)
try:
    from torch.utils.tensorboard import SummaryWriter as _SW  # noqa: F401
    print("TENSORBOARD_IMPORT_OK", {"summary_writer": True}, flush=True)
except Exception as exc:
    # Don't hard-fail: training survives with tb_writer=None, but surface it
    # loudly so an empty TB UI is diagnosable post-hoc as "missing pkg".
    print("TENSORBOARD_IMPORT_FAILED", {"error": str(exc)}, flush=True)
# RL monitoring metrics: 5 TB scalar groups + their data sources must be wired
# (performance/agent/optimization/learning_signal/trim). Asserted so a stale
# code tree fails loudly instead of silently writing empty TB blocks.
tb_groups = [
    '"performance"', '"agent"', '"optimization"', '"learning_signal"', '"trim"',
]
for _grp in tb_groups:
    assert _grp in ta, f"tb scalar group {_grp} missing in trainer_adapter"
print("TB_SCALAR_GROUPS_OK", {"groups": tb_groups}, flush=True)
# Actor part fields (source of optimization/ + cispo/ + tokens/ blocks). These
# are computed in fsdp2_actor.update() — entropy (log_softmax over vocab
# window), approx_kl (0.5*(ratio-1)^2 from CISPO ratio, zero extra forward),
# grad_norm (clip_grad_norm_ return value), policy_loss, ratio_clip_fraction,
# gate_mean. If any is missing the TB block silently skips (None guard).
fa = pathlib.Path("/mnt/songzijun/Capability_Evolution/TRIM/trim/integrations/verl/fsdp2_actor.py").read_text()
actor_keys = ["policy_loss", "entropy", "approx_kl", "grad_norm",
             "ratio_clip_fraction", "gate_mean", "rl_real_tokens",
             "n_optimizer_steps", "skipped_empty_supervision"]
_missing_actor = [k for k in actor_keys if k not in fa]
assert not _missing_actor, f"actor metrics keys missing in fsdp2_actor: {_missing_actor}"
print("ACTOR_METRICS_OK", {"keys": actor_keys}, flush=True)
# reward_parts_group_stats fields (source of performance/ + learning_signal/ +
# trim/ blocks): turn_penalty/n_turns/reward_mode (this run's additions) +
# success_rate/group_reward_std_mean/zero_variance_group_ratio/selected_ratio/
# task_recall_mean/n_const_reward_groups.
rom = pathlib.Path("/mnt/songzijun/Capability_Evolution/TRIM/trim/training/rl_opd_metrics.py").read_text()
audit_keys = ["turn_penalty", "n_turns", "reward_mode", "success_rate",
              "group_reward_std_mean", "zero_variance_group_ratio",
              "selected_ratio", "task_recall_mean", "n_const_reward_groups"]
_missing_audit = [k for k in audit_keys if k not in rom]
assert not _missing_audit, f"audit fields missing in rl_opd_metrics: {_missing_audit}"
print("AUDIT_FIELDS_OK", {"keys": audit_keys}, flush=True)
PY

# --- Copy Qwen3-4B model weights to local /tmp before exec (defensive) ---
# 4 nodes x 8 ranks = 32 concurrent JuiceFS readers risk IO contention on the
# first step; the per-node local cp is cheap (~1min for 7.6G) and avoids the
# t-cucac-style concurrent-read hang. Runs BEFORE vLLM worker spawn so it does
# NOT count against the 3600s worker-start window.
MODEL_SRC=/mnt/songzijun/models/Qwen3-4B-Instruct-2507
MODEL_LOCAL=/tmp/qwen3-4b
if [ ! -f "$MODEL_LOCAL/config.json" ]; then
  rm -rf "$MODEL_LOCAL"
  mkdir -p "$MODEL_LOCAL"
  cp -r "$MODEL_SRC"/. "$MODEL_LOCAL"/
fi
test -f "$MODEL_LOCAL/config.json"
test -f "$MODEL_LOCAL/model.safetensors.index.json" || true
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
  --rdzv-id qwen3-4B-rl-recall-turn-punish-32 \
  --training-backend torch_ddp_lora \
  --harness Harness-1 \
  --benchmark bcplus_full \
  --model_name /tmp/qwen3-4b \
  --train_method rl \
  --component zero \
  --train-env local_legacy \
  --train-data sec \
  --rl-data /mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-rl-data \
  --sec-corpus-root /mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-sec-corpus \
  --train-steps 200 \
  --train-groups-per-step 32 \
  --group-size 8 \
  --train-micro-batch-size 4 \
  --train-heartbeat-every 8 \
  --max-turns 40 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --out "${OUT}"
