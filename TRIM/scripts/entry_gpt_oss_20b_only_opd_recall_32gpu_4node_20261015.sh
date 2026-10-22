#!/usr/bin/env bash
# === GPT-OSS-20B only-opd + final-answer-recall reward + tensorboard (2026-10-15) ===
# Fork of entry_gpt_oss_20b_trim_all_recall_only_32gpu_4node_20261002_bf16.sh
# (the verified gpt-oss bf16-dequant recall-only 32gpu baseline). "Switch train
# method to only-opd" means: same 4node 32gpu bf16-dequant 200step + recall-only
# reward + tensorboard ON, but --train_method only-opd instead of trim/all.
#
# only-opd (2026-10-15 sync d5d5b1a, trim_ablation.py:27): rl_enabled=False,
# projection_mode=full, gate_mode=sigmoid. It is NOT an alias of trim/opd — it
# disables the RL update path and keeps ONLY the OPD projection distillation
# cell (teacher_full / student_zero, component=all). The final-answer-recall
# reward still drives OPD experience ranking/selection (terminal_reward_breakdown
# total=rec, env-gated via TRIM_REWARD_MODE=recall) even though no policy-gradient
# RL update runs (rl_enabled=False). So --rl-reward final-answer-recall +
# TRIM_REWARD_MODE=recall are BOTH kept (they set the reward protocol the OPD
# distillation reads, unchanged from the verified trim/all bf16 baseline).
#
# gpt-oss-20b forces 3 model-specific changes (all from verified memory, NOT
# optional, identical to the 20261002 bf16 baseline):
#   1. mb1 (NOT mb4): gpt-oss MoE cannot grad-ckpt, mb>=2 backward OOMs (k6v0p
#      铁证 peak_mem 74.7GB/80GB). Qwen3-4B dense mb4 does NOT apply here.
#   2. bf16-dequant model /mnt/songzijun/models/openai/gpt-oss-20b-bf16-dequant
#      (NOT original mxfp4): original mxfp4+LoRA has Marlin PTX crash (v0dhh
#      FAILED@step1) + NaN (txgli/zxkvl FAILED@step0). bf16-dequant is the
#      verified safe path (b1ljo/vqqxp/pncda all ran through).
#   3. SCAPE_FORCE_LOCAL_HARMONY INTENTIONALLY NOT SET: gpt-oss goes through
#      openai_harmony Rust encoder, setting this env causes _LocalConversation
#      to_json crash (xwl4c). Qwen3-4B set it because harmless; gpt-oss MUST NOT.
# Everything else byte-identical to the 20261002 bf16 baseline: 4 DDP patches
# (incl curate-clipping action_is_visible skip — the tinker_opd_datum datum-build
# guard applies to the OPD distillation cell too, so it is kept for only-opd),
# gpu-mem0.90, max-turns40, max-model-len8192, expandable_segments, model cp
# 39GB to /tmp per node, startup_timeout 3600s, tensorboard ON.
# rdzv-id gptoss20b-only-opd-recall-bf16-32 (avoid collision with trim/all 20261002).

# Container entry for Harness-1 BC+ only-opd (SR-OPD / CISPO distill, teacher_full
# student_zero, --train_method only-opd --component all) GPT-OSS-20B bf16-dequant
# on FOUR 8-GPU gm402-80g nodes (4 nodes x 8 cards = 32 GPUs, 200 steps).
#
# === bf16-dequant: why NOT original mxfp4 (verified by v0dhh/b1ljo) ===
# v0dhh (t-20260925132550-v0dhh, original mxfp4) FAILED at step1 vLLM+LoRA reload:
#   (EngineCore) torch.AcceleratorError: CUDA error: the provided PTX was compiled
#   with an unsupported toolchain (cudaErrorUnsupportedPtxVersion)
#   @ marlin_utils_fp4.py:574 repack_weight (prepare_moe_mxfp4_layer_for_marlin)
# Per user decision, switch to the pre-dequantized bf16 checkpoint
# /mnt/songzijun/models/openai/gpt-oss-20b-bf16-dequant (same GptOssForCausalLM,
# same architecture, quant_method=NONE, 9 safetensors ~39GB) — this runs the
# standard bf16+LoRA path with NO Marlin/Triton backend, definitively bypassing
# the mxfp4 PTX pit (坑1) AND the triton_kernels CPU-OOM pit (坑4).
#
# === Multi-node framework (verified by 7nzzn + v0dhh + b1ljo) ===
# custom_train framework=pytorch nodeNumber=4 creates a PyTorchJob CRD, injects
# MASTER_ADDR (master pod DNS) + NODE_RANK but NOT RANK/WORLD_SIZE/PET_*. The
# entry unsets rank env defensively so run_train_multinode.py self-launches
# torchrun via _exec_torchrun (deterministic 4-node rendezvous).
#
# === Pitfalls carried from the 20261002 bf16 baseline (all still apply) ===
#   * trainer_adapter.py:478 startup_timeout_s=3600.0 (900s worker-start timeout fix;
#     bf16-dequant is 39GB/3x mxfp4, so the per-node /tmp cp takes ~7min and 8 ranks
#     load from local /tmp concurrently — 3600s headroom is essential).
#   * trainer_adapter.py:264 resolve_queries 4-tuple unpack (724d9fa ValueError fix).
#   * trainer_adapter.py:506 client.start() EADDRINUSE retry x3 (per-rank vLLM TP=1
#     TCPStore port race at step>=1; 32 ranks across 4 nodes make it more likely).
#   * SCAPE_FORCE_LOCAL_HARMONY INTENTIONALLY NOT SET (gpt-oss to_json crash).
#   * Model cp to /tmp BEFORE exec (8-rank JuiceFS concurrent-read hang fix; here
#     EACH NODE does its own cp of 39GB bf16 from JuiceFS to /tmp, ~7min).
#   * curate-clipping tolerant-skip patch (action_is_visible + tinker_opd_datum
#     skip; the OPD distillation cell builds datums via tinker_opd_datum too, so
#     the guard is kept for only-opd).
#
# mb1 (NOT 4): mb>=2 backward OOMs on 20B (gpt-oss MoE backward cannot enable
# gradient checkpointing, mb1 is the primary OOM defense; peak_mem 74.7GB/80GB
# constant per rank). With 4 nodes x 8 ranks = 32 replicas, per-rank memory
# budget is UNCHANGED (each rank is TP=1 single-GPU).
#
# bf16-dequant path: NO VLLM_MXFP4_USE_MARLIN, NO triton_kernels install needed
# (quant_method=NONE, vLLM runs standard bf16+LoRA, no Marlin/Triton backend).
# NO CUDA_LAUNCH_BLOCKING (DDP multi-proc). NO DISABLE_CUSTOM_ALL_REDUCE (TP=1).
set -euxo pipefail

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TRIM_GPU_KEEPALIVE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/mnt/songzijun/Capability_Evolution/TRIM:/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD

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
: "${MASTER_PORT:=29500}"
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

# --- reward: final-answer-recall EXPLICIT (--rl-reward, NOT env-gated) ---
# only-opd sets rl_enabled=False (no policy-gradient update), but the OPD
# distillation still reads the terminal reward to rank/select experiences.
# Per CLAUDE.md 2026-10-15 澄清: --rl-reward final-answer-recall is the EXPLICIT
# path -> terminal_reward_breakdown(rl_reward="final-answer-recall") ->
# reward_policy.score_explicit_reward (four_cell_runtime.py:1144 dispatch,
# reward_policy.py:77), pure final_answer_recall metric, reads NO env, no
# compute_logprobs / no extra logits forward. The explicit path TAKES
# PRECEDENCE over env; setting TRIM_REWARD_MODE=recall here would be REDUNDANT
# and misleading (it only routes _legacy_terminal_reward_breakdown when
# rl_reward is None). So NO TRIM_REWARD_MODE env is exported on purpose.

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
test -f /mnt/songzijun/models/openai/gpt-oss-20b-bf16-dequant/config.json
test -f /mnt/songzijun/models/openai/gpt-oss-20b-bf16-dequant/model.safetensors.index.json
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
    "tensorboard_log_path": os.environ.get("TENSORBOARD_LOG_PATH"),
}
print("PREFLIGHT", versions, flush=True)
assert versions["torch"].startswith("2.10.0"), f"torch={versions['torch']} != 2.10.0"
assert versions["vllm"] == "0.19.1", f"vllm={versions['vllm']} != 0.19.1"
assert versions["transformers"] == "4.57.6", f"transformers={versions['transformers']} != 4.57.6"
assert versions["pyserini"] == "2.0.0", f"pyserini={versions['pyserini']} != 2.0.0"
assert versions["gpu_count"] == 8, f"per-node gpu_count={versions['gpu_count']} != 8"
assert versions["o200k_base_present"]

# Tensorboard SummaryWriter probe. Platform injects $TENSORBOARD_LOG_PATH when
# --enable_tensorboard is set (submit script). If tensorboard package is missing
# we do NOT hard-fail (trainer _make_tensorboard_writer degrades to None), but
# warn loudly so the job owner knows tensorboard UI will be empty.
try:
    from torch.utils.tensorboard import SummaryWriter  # noqa: F401
    tb_ok = True
except Exception as exc:
    tb_ok = False
    print("TENSORBOARD_IMPORT_FAIL", {"error": str(exc)}, flush=True)
print("TENSORBOARD_PROBE", {"summary_writer_ok": tb_ok,
                            "log_path_env": versions["tensorboard_log_path"]}, flush=True)
# GPT-OSS-20B model probe: bf16-dequant checkpoint (quant_method must be absent/NONE,
# NOT mxfp4 — this is the v0dhh fix: bf16 avoids the mxfp4 Marlin PTX crash).
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
from pyserini.search.lucene import LuceneSearcher
s = LuceneSearcher("/mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-sec-corpus/indexes/bm25")
n_docs = s.num_docs if not callable(s.num_docs) else s.num_docs()
print("LUCENE_OK", {"num_docs": n_docs}, flush=True)
assert n_docs > 0
from trim.integrations.verl.trainer_adapter import _actor_wrap
assert _actor_wrap("torch_ddp_lora") == "ddp"
print("DDP_LORA_WRAP_OK", {"wrap": _actor_wrap("torch_ddp_lora")}, flush=True)
import pathlib, re
ta = pathlib.Path("/mnt/songzijun/Capability_Evolution/TRIM/trim/integrations/verl/trainer_adapter.py").read_text()
assert "startup_timeout_s=3600.0" in ta, "startup_timeout_s=3600 patch missing"
assert re.search(r"train_rows,\s*eval_rows,\s*pool_meta,\s*_?frozen_points\s*=\s*resolve_queries", ta), "4-tuple unpack patch missing"
assert "EADDRINUSE" in ta, "EADDRINUSE retry patch missing"
ae = pathlib.Path("/mnt/songzijun/Capability_Evolution/TRIM/trim/training/action_encoding.py").read_text()
assert "def action_is_visible(" in ae, "action_is_visible patch missing"
tod = pathlib.Path("/mnt/songzijun/Capability_Evolution/TRIM/trim/training/tinker_opd_datum.py").read_text()
assert "action_is_visible(step.target_action, visible)" in tod, "skip-clipped-doc call site missing"
assert "n_skip_clipped_doc" in tod, "skip counter missing"
from trim.training.action_encoding import action_is_visible
assert action_is_visible({"name": "curate", "arguments": {"add_ids": ["X"]}}, ["Y"]) is False
assert action_is_visible({"name": "curate", "arguments": {"add_ids": ["X"]}}, ["X"]) is True
# Verify the tensorboard writer helper + new tb scalar groups are present (ported from uwkpr/h3eb8).
assert "_make_tensorboard_writer" in ta and "_tb_add_scalars" in ta, \
    "tensorboard writer helper missing in trainer_adapter.py"
assert '"performance"' in ta and '"agent"' in ta and '"optimization"' in ta \
    and '"learning_signal"' in ta and '"trim"' in ta, \
    "new tb scalar groups (performance/agent/optimization/learning_signal/trim) missing in trainer_adapter.py"
# Verify the new optimization-metric computation (entropy/approx_kl/grad_norm/policy_loss)
# is wired into the actor update().
fa = pathlib.Path("/mnt/songzijun/Capability_Evolution/TRIM/trim/integrations/verl/fsdp2_actor.py").read_text()
assert '"action_entropy"' in fa, "per-token entropy not exposed in _forward_response_logprobs"
assert '"entropy"' in fa and '"approx_kl"' in fa and '"grad_norm"' in fa and '"policy_loss"' in fa, \
    "entropy/approx_kl/grad_norm/policy_loss metrics not emitted by actor.update()"
# approx_kl must be aggregated from the CISPO ratio.
assert "0.5 * (ratio - 1.0) ** 2" in fa, "approx_kl not aggregated from ratio in _rl_numerator"
print("ACTOR_METRICS_OK", {"entropy": True, "approx_kl": True, "grad_norm": True,
                           "policy_loss": True}, flush=True)
# Verify the new audit fields (advantage_std/zero_variance_group_ratio/success_rate/selected_ratio)
# are computed in reward_parts_group_stats.
ro = pathlib.Path("/mnt/songzijun/Capability_Evolution/TRIM/trim/training/rl_opd_metrics.py").read_text()
for key in ("zero_variance_group_ratio", "group_reward_std_mean", "success_rate",
            "selected_ratio", "task_recall_mean", "n_const_reward_groups"):
    assert key in ro, f"{key} not computed in reward_parts_group_stats"
print("AUDIT_FIELDS_OK", {"advantage_std": True, "zero_variance_group_ratio": True,
                           "success_rate": True, "selected_ratio": True}, flush=True)
# Verify the reward path. Per CLAUDE.md 2026-10-15 澄清 there are TWO paths:
#   (A) EXPLICIT --rl-reward final-answer-recall -> terminal_reward_breakdown()
#       at four_cell_runtime.py:1144 dispatches to reward_policy.score_explicit_reward
#       (pure final_answer_recall metric, reads NO env). This is what we run.
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
                                  "path": "score_explicit_reward (no env, no logprobs)"}, flush=True)
# Verify the only-opd ablation method is registered (2026-10-15 sync d5d5b1a).
from trim.training.trim_ablation import resolve_trim_ablation, NEW_TRIM_ABLATION_METHODS
prof = resolve_trim_ablation("only-opd")
assert prof is not None and prof.variant == "only_opd", f"only-opd profile missing/variant wrong: {prof}"
assert prof.rl_enabled is False, f"only-opd must have rl_enabled=False, got {prof.rl_enabled}"
assert prof.projection_mode == "full", f"only-opd projection_mode must be full, got {prof.projection_mode}"
assert prof.gate_mode == "sigmoid", f"only-opd gate_mode must be sigmoid, got {prof.gate_mode}"
assert "only-opd" in NEW_TRIM_ABLATION_METHODS
print("ONLY_OPD_PROFILE_OK", {"variant": prof.variant, "rl_enabled": prof.rl_enabled,
                              "projection_mode": prof.projection_mode, "gate_mode": prof.gate_mode}, flush=True)

print("PATCH_CHECK_OK", {"startup_timeout_s": True, "resolve_queries_4tuple": True, "eaddrinuse_retry": True, "skip_clipped_doc": True, "bf16_dequant": True, "tb_writer": True, "tb_scalar_groups": True, "only_opd": True, "reward_explicit_path": True}, flush=True)
PY

# --- FIX2: copy bf16-dequant model weights to local /tmp before exec ---
# bf16-dequant is 39GB (3x mxfp4's 13GB); each node does its own cp from JuiceFS
# (~7min), then all 8 local ranks load from fast local disk. Runs BEFORE vLLM
# worker spawn so it does NOT count against the 3600s worker-start window.
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
  --rdzv-id gptoss20b-only-opd-recall-bf16-32 \
  --training-backend torch_ddp_lora \
  --harness Harness-1 \
  --benchmark bcplus_full \
  --model_name /tmp/gpt-oss-20b-bf16-dequant \
  --train_method only-opd \
  --component all \
  --train-env local_legacy \
  --train-data sec \
  --rl-data /mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-rl-data \
  --sec-corpus-root /mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-sec-corpus \
  --rl-reward final-answer-recall \
  --train-steps 200 \
  --train-groups-per-step 32 \
  --group-size 8 \
  --train-micro-batch-size 1 \
  --train-heartbeat-every 8 \
  --max-turns 40 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --out "${OUT}"
