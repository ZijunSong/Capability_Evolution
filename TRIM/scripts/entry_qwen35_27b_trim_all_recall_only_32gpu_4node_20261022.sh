#!/usr/bin/env bash
# Harness-1 BC+ trim/all (SR-OPD/CISPO distill, --train_method trim --component
# all) Qwen3.5-27B (dense bf16 51.75GB, Qwen3_5ForConditionalGeneration)
# torch_ddp_lora 4node32gpu 200step. REWARD=final-answer recall ONLY.
#
# = uwkpr entry (entry_qwen3_4b_trim_all_recall_only_32gpu_4node_20260930.sh,
#   verified Qwen3-4B trim/all recall-only 32gpu 4node) with 4 changes:
#   1. MODEL Qwen3-4B-Instruct-2507 -> Qwen3.5-27B (all probes/paths updated).
#   2. mb 4 -> 2 (27B 51.75GB weights; dense grad-ckpt; rvt6r estimate mb2 peak
#      ~76GB/80GB; mb4 ~83GB OOM; if step0 backward OOMs resubmit mb1).
#   3. REWARD ACTIVE path = explicit --rl-reward final-answer-recall ->
#      reward_policy.score_explicit_reward (CLAUDE.md 2026-10-15 澄清 explicit
#      > legacy env). Legacy code state (`total = rec`) asserted as insurance.
#   4. ARCH_OK vLLM ModelRegistry probe for Qwen3_5ForConditionalGeneration
#      (fail-fast before wasting 32-GPU rollout; eval fcanl/iytmg/bg3xs proved
#      vLLM 0.19.1 registers Qwen3_5).
# Otherwise identical to uwkpr: gpu-mem0.90 max-turns40 max-model-len8192
# SCAPE_FORCE_LOCAL_HARMONY=1 tensorboard ON 4 DDP patches (startup_timeout
# 3600 / 4-tuple unpack / EADDRINUSE retry / curate-clipping tolerant-skip).
#
# trim/all path note: curate-clipping patch (action_is_visible +
# tinker_opd_datum skip, ethqp fix) is REQUIRED or step0 datum build aborts
# ("projected curate refs docs not visible after clipping", x2j3u). iowty
# verified trim/all multi-node world_size=32; uwkpr verified + recall-only.
#
# Multi-node (7nzzn/iowty verified): CloudML pytorch nodeNumber=4 injects
# MASTER_ADDR + NODE_RANK only (NOT RANK/WORLD_SIZE/PET_*); defensive unset
# below + explicit --master-addr/--node-rank make rendezvous deterministic.
# OUT must be identical on all pods (submit --env); never per-pod $(date).
#
# Qwen3.5-27B: dense bf16, no mxfp4/Marlin dance. cp 51.75GB to /tmp per node
# BEFORE vLLM spawn (outside the 3600s worker-start window) to avoid the
# 32-concurrent-JuiceFS-reader hang (t-cucac class).
#
# torch_ddp_lora: NO CUDA_LAUNCH_BLOCKING, NO DISABLE_CUSTOM_ALL_REDUCE
# (per-rank vLLM TP=1).
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
       "node0's IP as MASTER_ADDR. If this pod has no MASTER_ADDR, the job will" \
       "fail to rendezvous. Aborting to avoid 4 independent single-node runs." >&2
  exit 1
fi
: "${MASTER_PORT:=29500}"
export MASTER_ADDR NODE_RANK MASTER_PORT

# Strip any CloudML-injected torchrun-style rank env so run_train_multinode.py's
# under_torchrun() returns False and main() self-launches torchrun via
# _exec_torchrun. This makes the 4-node rendezvous deterministic.
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK \
      GROUP_WORLD_SIZE TORCHELASTIC_RUN_ID TORCHELASTIC_RESTART_COUNT \
      TORCHELASTIC_MAX_RESTARTS TORCHELASTIC_USE_AGENT_STORE \
      PET_MASTER_ADDR PET_MASTER_PORT PET_NNODES PET_NPROC_PER_NODE PET_NODE_RANK

echo "MULTINODE_ENV master_addr=${MASTER_ADDR} node_rank=${NODE_RANK} master_port=${MASTER_PORT} host=$(hostname)"

# --- Shared output directory (MUST be identical on every node) ---
# run_train_multinode.py writes per-rank rollout shards to <out>/tmp/rollout_shards
# and rank 0 interleaves them via all_gather_via_disk. If each node uses a
# different <out> (e.g. a per-pod $(date) timestamp), the shards are invisible
# across nodes and step 0 backward crashes with FileNotFoundError. OUT is
# injected via submit --env so all 4 pods share the exact same path.
# NOTE: $TENSORBOARD_LOG_PATH is auto-injected by the platform when
# --enable_tensorboard is passed at submit; the trainer reads it and writes
# event files there. We do NOT set it here.
if [ -z "${OUT:-}" ]; then
  echo "FATAL: OUT is not set. Multi-node requires a shared --out path injected via --env OUT=... (identical on every pod). Aborting." >&2
  exit 1
fi
export OUT
echo "MULTINODE_OUT out=${OUT} host=$(hostname) node_rank=${NODE_RANK}"

# Qwen3.5-27B: SCAPE_FORCE_LOCAL_HARMONY=1 is SET (harmless on Qwen3.5 HF chat
# family, matches the verified Qwen paths). Placeholder key so
# pyserini.encode._openai module-level openai.OpenAI() does not raise
# OpenAIError at import (trim/RL uses local BM25, never calls OpenAI API).
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

# --- Install four-pack + harness import chain deps + tensorboard (internal PyPI) ---
python3 -m pip install --no-cache-dir --ignore-installed --upgrade-strategy only-if-needed \
  "torch==2.10.0" "transformers==5.17.0" "accelerate==1.13.0" "datasets==4.8.4" \
  "hydra-core==1.3.2" "omegaconf==2.3.0" "vllm==0.19.1" peft structlog chromadb \
  tqdm regex sympy latex2sympy2 word2number sentencepiece "pyserini==2.0.0" openai tenacity anthropic \
  tensorboard

# --- ANSERINI_CLASSPATH for PyJNI/Lucene ---
PYSERINI_ROOT=$(python3 -c 'import pathlib, pyserini; print(pathlib.Path(pyserini.__file__).resolve().parent)')
export ANSERINI_CLASSPATH=$(find "$PYSERINI_ROOT/resources/jars" -type f -name '*.jar' -print | paste -sd: -)
test -n "$ANSERINI_CLASSPATH"
export CLASSPATH="$ANSERINI_CLASSPATH"

# --- Preflight: assert four-pack versions, 8 GPUs (per node), JVM, import chain,
#     Lucene, tensorboard SummaryWriter, reward=recall-only (explicit + legacy
#     code state), new optimization/audit metrics, Qwen3.5 arch (config +
#     vLLM ModelRegistry), AND the 4 trim/all DDP patches (incl.
#     curate-clipping) ---
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
assert versions["transformers"] == "5.17.0", f"transformers={versions['transformers']} != 5.17.0"
assert versions["pyserini"] == "2.0.0", f"pyserini={versions['pyserini']} != 2.0.0"
assert versions["gpu_count"] == 8, f"per-node gpu_count={versions['gpu_count']} != 8"
assert versions["o200k_base_present"]

# Tensorboard SummaryWriter probe. The platform injects $TENSORBOARD_LOG_PATH
# when --enable_tensorboard is set (submit script); the trainer writes event
# files there. If the tensorboard package is missing we do NOT hard-fail (the
# trainer's _make_tensorboard_writer degrades to None), but we warn loudly so
# the job owner knows tensorboard UI will be empty.
try:
    from torch.utils.tensorboard import SummaryWriter  # noqa: F401
    tb_ok = True
except Exception as exc:
    tb_ok = False
    print("TENSORBOARD_IMPORT_FAIL", {"error": str(exc)}, flush=True)
print("TENSORBOARD_PROBE", {"summary_writer_ok": tb_ok,
                            "log_path_env": versions["tensorboard_log_path"]}, flush=True)

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

# HF-side arch probe: tf 4.57.6 raised KeyError 'qwen3_5' at step0 actor init
# (72rhl 2026-10-23) — vLLM had the arch but Transformers did not. 5.17.0
# registers it in MODEL_FOR_CAUSAL_LM_MAPPING. Fail fast here if that regresses.
from transformers import AutoConfig
hf_cfg = AutoConfig.from_pretrained("/mnt/songzijun/models/Qwen3.5-27B", trust_remote_code=True)
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
_ok = hf_cfg.__class__ in MODEL_FOR_CAUSAL_LM_MAPPING
if not _ok and getattr(hf_cfg, "text_config", None) is not None:
    _ok = hf_cfg.text_config.__class__ in MODEL_FOR_CAUSAL_LM_MAPPING
assert _ok, f"{hf_cfg.__class__.__name__} not in MODEL_FOR_CAUSAL_LM_MAPPING"
print("HF_ARCH_OK", {"config": hf_cfg.__class__.__name__}, flush=True)
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

# Verify the code-tree patches are in place (so we don't silently regress).
import pathlib, re
ta = pathlib.Path("/mnt/songzijun/Capability_Evolution/TRIM/trim/integrations/verl/trainer_adapter.py").read_text()
assert "startup_timeout_s=3600.0" in ta, "startup_timeout_s=3600 patch missing"
assert re.search(r"train_rows,\s*eval_rows,\s*pool_meta,\s*_?frozen_points\s*=\s*resolve_queries", ta), "4-tuple unpack patch missing"
assert "EADDRINUSE" in ta, "EADDRINUSE retry patch missing"

# trim/all-specific patch: curate-clipping tolerant-skip.
# assert_action_visible used to abort the whole run when a teacher curate
# referenced docs dropped by context clipping (deterministic crash on the
# 0000950103-25-007493_26 / 0001213900-25-049013_54 episode in step0, x2j3u
# FAILED). Now action_is_visible returns bool and the caller (tinker_opd_datum)
# skips the unprojectable step. Verify the new function + the skip path exist;
# this is the ONLY trim/all-specific patch and is REQUIRED for this entry's path.
ae = pathlib.Path("/mnt/songzijun/Capability_Evolution/TRIM/trim/training/action_encoding.py").read_text()
assert "def action_is_visible(" in ae, "action_is_visible patch missing"
tod = pathlib.Path("/mnt/songzijun/Capability_Evolution/TRIM/trim/training/tinker_opd_datum.py").read_text()
assert "action_is_visible(step.target_action, visible)" in tod, "skip-clipped-doc call site missing"
assert "n_skip_clipped_doc" in tod, "skip counter missing"
from trim.training.action_encoding import action_is_visible
assert action_is_visible({"name": "curate", "arguments": {"add_ids": ["X"]}}, ["Y"]) is False
assert action_is_visible({"name": "curate", "arguments": {"add_ids": ["X"]}}, ["X"]) is True

# Verify the tensorboard writer helper + new tb scalar groups are present (from h3eb8).
assert "_make_tensorboard_writer" in ta and "_tb_add_scalars" in ta, \
    "tensorboard writer helper missing in trainer_adapter.py"
assert '"performance"' in ta and '"agent"' in ta and '"optimization"' in ta \
    and '"learning_signal"' in ta and '"trim"' in ta, \
    "new tb scalar groups (performance/agent/optimization/learning_signal/trim) missing in trainer_adapter.py"
# Verify the new optimization-metric computation (entropy/approx_kl/grad_norm/
# policy_loss) is wired into the actor update().
fa = pathlib.Path("/mnt/songzijun/Capability_Evolution/TRIM/trim/integrations/verl/fsdp2_actor.py").read_text()
assert '"action_entropy"' in fa, "per-token entropy not exposed in _forward_response_logprobs"
assert '"entropy"' in fa and '"approx_kl"' in fa and '"grad_norm"' in fa and '"policy_loss"' in fa, \
    "entropy/approx_kl/grad_norm/policy_loss metrics not emitted by actor.update()"
# approx_kl must be aggregated from the CISPO ratio.
assert "0.5 * (ratio - 1.0) ** 2" in fa, "approx_kl not aggregated from ratio in _rl_numerator"
print("ACTOR_METRICS_OK", {"entropy": True, "approx_kl": True, "grad_norm": True,
                           "policy_loss": True}, flush=True)
# Verify the new audit fields (advantage_std/zero_variance_group_ratio/
# success_rate/selected_ratio) are computed in reward_parts_group_stats.
ro = pathlib.Path("/mnt/songzijun/Capability_Evolution/TRIM/trim/training/rl_opd_metrics.py").read_text()
for key in ("zero_variance_group_ratio", "group_reward_std_mean", "success_rate",
            "selected_ratio", "task_recall_mean", "n_const_reward_groups"):
    assert key in ro, f"{key} not computed in reward_parts_group_stats"
print("AUDIT_FIELDS_OK", {"advantage_std": True, "zero_variance_group_ratio": True,
                           "success_rate": True, "selected_ratio": True}, flush=True)

# === Reward = final-answer recall ONLY (ACTIVE path = explicit --rl-reward) ===
# Per CLAUDE.md 2026-10-15 澄清 there are TWO paths:
#   (A) EXPLICIT --rl-reward final-answer-recall -> terminal_reward_breakdown()
#       dispatches to reward_policy.score_explicit_reward (pure
#       final_answer_recall metric, reads NO env, no compute_logprobs). THIS is
#       the active path for this entry (flag passed in the exec line below).
#   (B) LEGACY rl_reward=None -> _legacy_terminal_reward_breakdown() (the uwkpr
#       code state: `total = rec`, protocol "final_answer_recall_only", no -0.2
#       invalid penalty). NOT taken here (flag is passed), but asserted as
#       regression insurance — if the legacy fn ever regresses to the 4-term
#       composite, the next job that omits the flag would silently change reward.
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
                                  "fed_to": "CISPO+sr_opd_ce joint backward (trim/all rl_enabled)"}, flush=True)
# Legacy path code state (uwkpr REWARD_RECALL_ONLY_OK asserts): regression
# insurance only — the active dispatch is the explicit flag above.
assert '"final_answer_recall_only"' in fc, "reward protocol not flipped to final_answer_recall_only"
assert 'total = rec' in fc, "reward total not set to pure recall"
assert '"total": -0.2' not in fc, "legacy -0.2 invalid-episode penalty still present"
print("REWARD_RECALL_ONLY_OK", {"protocol": "final_answer_recall_only"}, flush=True)

print("PATCH_CHECK_OK", {"startup_timeout_s": True, "resolve_queries_4tuple": True,
                         "eaddrinuse_retry": True, "skip_clipped_doc": True,
                         "tb_writer": True, "tb_scalar_groups": True,
                         "qwen35_arch": True, "reward_explicit_path": True}, flush=True)
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
  --rdzv-id qwen35-27b-trim-all-recall-only-32 \
  --training-backend torch_ddp_lora \
  --harness Harness-1 \
  --benchmark bcplus_full \
  --model_name /tmp/qwen35-27b \
  --train_method trim \
  --component all \
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
