#!/usr/bin/env bash
# === GPT-OSS-20B trim/all + recall-only + tensorboard + lambda_opd=1.0 (2026-10-17) ===
# Fork of entry_gpt_oss_20b_trim_all_recall_only_32gpu_4node_20261002_bf16.sh (pncda,
# the verified gpt-oss bf16-dequant trim/all recall-only 32gpu baseline whose
# 3-seed step100/step200 FA Recall is the user's reference:
#   step100 FA mean = BrowseComp+ 24.79% / FRAMES 77.79% / LongSealQA 69.77% / HotpotQA 91.82%
#   step200 FA mean = BrowseComp+ 24.27% / FRAMES 79.76% / LongSealQA 71.94% / HotpotQA 93.51%
# ).
# ONLY CHANGE vs pncda: lambda_opd 0.01 (trim default) -> 1.0 via explicit
# --lambda-opd 1.0. Everything else byte-identical to pncda:
#   - same --train_method trim --component all, same recall-only reward
#     (four_cell_runtime.py terminal_reward_breakdown total=rec, unchanged code tree),
#   - same mb1 + bf16-dequant + NO SCAPE_FORCE_LOCAL_HARMONY + gpu-mem0.90
#     + max-turns40 + max-model-len8192 + expandable_segments + 4 DDP patches
#     (incl curate-clipping) + startup_timeout 3600s + model cp 39GB to /tmp,
#   - same 4 nodes x 8 GPUs = 32 ranks, torch_ddp_lora, 200 steps, train-data sec.
#
# lambda_opd mechanism (verified, NOT a guess):
#   launch.py:760-768  --lambda-opd  type=float  default=None  dest=lambda_opd
#   launch.py:1307-1308  if args.lambda_opd is None:
#                          args.lambda_opd = SCAPE_RL_LAMBDA_OPD if seed_scale else 0.1
#   rl_opd_types.py:59  SCAPE_RL_LAMBDA_OPD = 0.01   # trim is seed_scale -> default 0.01
#   launch.py:1233-1234  assert lambda_opd > 0       # 1.0 satisfies this
# So pncda (no --lambda-opd, --train_method trim) got 0.01; THIS job explicitly
# passes --lambda-opd 1.0 to override. No code-tree change needed.
#
# lambda_opd scales the OPD distillation loss (a scalar coefficient on the OPD
# CE term); it does NOT change forward/backward activation shapes, so the pncda
# peak_mem=74.7GB/80GB budget under mb1+gpu-mem0.90 still holds (no extra OOM risk
# from raising the coefficient 100x — only the gradient magnitude changes).
# A larger OPD coefficient (1.0 = 100x pncda's 0.01) weights the teacher
# projection more heavily vs the recall RL gradient; the first 1-3 steps are the
# main place a too-large OPD loss could destabilize the policy, so the step0-2
# rollout/grad_norm/entropy is the thing to watch post-launch.
#
# rdzv-id gptoss20b-trim-all-recall-only-bf16-lop10-32 (avoids collision with
# pncda's gptoss20b-trim-all-recall-only-bf16-32 rendezvous endpoint).

# Container entry for Harness-1 BC+ trim (SR-OPD / CISPO distill, --train_method
# trim --component all) GPT-OSS-20B on FOUR 8-GPU gm402-80g nodes (4 nodes x 8
# cards = 32 GPUs, 200 steps), lambda_opd=1.0 variant.
#
# All inherited comments from pncda (recov1_bf16 / v0dhh fix lineage) still apply:
#   * bf16-dequant avoids mxfp4 Marlin PTX crash (坑1) and triton_kernels OOM (坑4).
#   * mb1 (NOT 4): gpt-oss MoE cannot grad-ckpt, mb>=2 backward OOMs (k6v0p peak 74.7GB/80GB).
#   * SCAPE_FORCE_LOCAL_HARMONY INTENTIONALLY NOT SET (gpt-oss to_json crash, xwl4c).
#   * startup_timeout_s=3600 (bf16-dequant 39GB /tmp cp ~7min per node).
#   * resolve_queries 4-tuple unpack, EADDRINUSE retry x3, curate-clipping tolerant-skip.
#   * 4-node torchrun self-launched (CloudML injects MASTER_ADDR/NODE_RANK but NOT
#     RANK/WORLD_SIZE/PET_*; entry unsets rank env defensively).
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
# Verify the reward is final-answer-recall ONLY (ported from uwkpr/h3eb8).
fc = pathlib.Path("/mnt/songzijun/Capability_Evolution/TRIM/trim/training/four_cell_runtime.py").read_text()
assert '"final_answer_recall_only"' in fc, "reward protocol not flipped to final_answer_recall_only"
assert 'total = rec' in fc, "reward total not set to pure recall"
assert '"total": -0.2' not in fc, "legacy -0.2 invalid-episode penalty still present"
print("REWARD_RECALL_ONLY_OK", {"protocol": "final_answer_recall_only"}, flush=True)

print("PATCH_CHECK_OK", {"startup_timeout_s": True, "resolve_queries_4tuple": True, "eaddrinuse_retry": True, "skip_clipped_doc": True, "bf16_dequant": True, "tb_writer": True, "tb_scalar_groups": True}, flush=True)
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

# RDZV_ID for this run (shared by barrier signal filename + --rdzv-id). Making
# the barrier signal filename carry the rdzv-id fixes the stale-signal bug
# that killed lhsyc (0.1 group): lhsyc reused the same OUT as obkzm, and
# obkzm's leftover .rendezvous_barrier/master_ready_0 (same fixed name) was
# read by lhsyc's worker BEFORE lhsyc's master reached the barrier (master was
# still cp'ing the 39GB model), so the worker launched torchrun early and hit
# TCPStore 900s timeout. With rdzv-id in the filename, each run's signal is
# uniquely named, so a stale signal from a prior run can never be read by
# this run's workers. r4 uses a fresh rdzv-id (...-r4-resume) distinct from r2/r3.
RDZV_ID="gptoss20b-trim-all-recall-only-bf16-lop10-32-r4-resume"
# Master clears ITS OWN stale signal early (right after cd, before pip/preflight/
# cp model), so even if a prior r4 attempt left a signal it won't be read while
# master is still in pip/cp. Workers below poll for this run's uniquely-named
# signal and will NOT match any prior run's signal.
BARRIER_DIR="${OUT}/.rendezvous_barrier"
mkdir -p "$BARRIER_DIR"
if [ "${NODE_RANK:-0}" = "0" ]; then
  rm -f "$BARRIER_DIR/master_ready_${RDZV_ID}" 2>/dev/null || true
fi

# --- Resume preflight (wufak 1.0 r2 kill@step9, resume from step_000009) ---
# wufak (lambda_opd=1.0) was stopped by user 2026-10-19 ~16:01 at step9
# (latest->step_000009, STEP_COMPLETE+optimizer.pt+adapter all present).
# --resume makes run_train_multinode read $OUT/checkpoints/scape_seed/latest
# and continue from step10. assert the resume target is intact before exec.
RESUME_LATEST="${OUT}/checkpoints/scape_seed/latest"
test -L "$RESUME_LATEST" || { echo "FATAL: resume needs $RESUME_LATEST symlink (cell=scape_seed)" >&2; exit 1; }
RESUME_TARGET=$(readlink -f "$RESUME_LATEST")
test -f "$RESUME_TARGET/STEP_COMPLETE" || { echo "FATAL: $RESUME_TARGET/STEP_COMPLETE missing" >&2; exit 1; }
test -f "$RESUME_TARGET/optimizer.pt"   || { echo "FATAL: $RESUME_TARGET/optimizer.pt missing" >&2; exit 1; }
test -f "$RESUME_TARGET/STEP_MANIFEST.json" || { echo "FATAL: $RESUME_TARGET/STEP_MANIFEST.json missing" >&2; exit 1; }
echo "RESUME_TARGET latest=$RESUME_TARGET cell=scape_seed (wufak 1.0 r2, resume from step10)"

# --- Worker-wait-master rendezvous barrier (fix qvnwn failure) ---
# qvnwn (the 0.1 group) FAILED at 2026-10-17 23:58 (runTime 2864s): master
# (NODE_RANK=0) pip install took ~42min (pip dependency-resolver backtracked on
# latex2sympy2/pyserini version combinations), while workers finished in ~17min
# and reached run_train_multinode's internal torchrun rendezvous first. Master
# was still in pip/cp-model when workers tried to connect to its TCPStore port
# (20672); the port wasn't open yet, so workers hit "client socket timed out
# after 900000ms" twice (30min total) and crashed with
# torch.distributed.DistNetworkError. Job died before step0 — lambda_opd was
# correctly passed but never executed; this was a pure rendezvous timing race,
# NOT a lambda_opd or training issue.
#
# This barrier decouples worker torchrun start from master's: workers poll a
# shared OUT-dir signal file that ONLY master writes, right before exec, i.e.
# after master has finished pip+preflight+cp model and is about to run
# run_train_multinode (which spawns the master torchrun that opens the
# TCPStore port). Workers thus never start torchrun before master, eliminating
# the 900s timeout race. Master clears stale signals from prior failed
# attempts in this OUT dir first; workers only read.
BARRIER_DIR="${OUT}/.rendezvous_barrier"
mkdir -p "$BARRIER_DIR"
if [ "${NODE_RANK:-0}" = "0" ]; then
  # master: signal ready (signal file is uniquely named per rdzv-id, so it
  # cannot be a stale signal from a prior run — and we cleared it above).
  : > "$BARRIER_DIR/master_ready_${RDZV_ID}"
  echo "BARRIER_MASTER_READY wrote $BARRIER_DIR/master_ready_${RDZV_ID} host=$(hostname) node_rank=${NODE_RANK} ts=$(date -u +%FT%TZ)"
else
  # worker: poll for THIS run's master ready signal (uniquely named by rdzv-id,
  # so a stale signal from a prior run can never match — fixes lhsyc bug).
  # Tolerate a slow master (pip backtracking can take ~42min, allow 50min).
  MASTER_SIGNAL="$BARRIER_DIR/master_ready_${RDZV_ID}"
  echo "BARRIER_WORKER_WAIT polling $MASTER_SIGNAL host=$(hostname) node_rank=${NODE_RANK}"
  WAITED=0
  while [ ! -f "$MASTER_SIGNAL" ]; do
    sleep 10
    WAITED=$((WAITED + 10))
    if [ "$WAITED" -ge 3000 ]; then
      echo "FATAL: worker node_rank=${NODE_RANK} timed out after ${WAITED}s waiting for master_ready signal $MASTER_SIGNAL" >&2
      exit 1
    fi
    # progress heartbeat every 60s so stdout mtime advances (avoid false-hang detection)
    if [ $((WAITED % 60)) -eq 0 ]; then
      echo "BARRIER_WORKER_WAIT still waiting ${WAITED}s host=$(hostname) node_rank=${NODE_RANK}"
    fi
  done
  echo "BARRIER_WORKER_SEEN_MASTER after ${WAITED}s host=$(hostname) node_rank=${NODE_RANK}"
fi

exec python3 /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_train_multinode.py \
  --nnodes 4 \
  --nproc-per-node 8 \
  --node-rank "${NODE_RANK}" \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  --dist-backend nccl \
  --rdzv-backend static \
  --rdzv-id "${RDZV_ID}" \
  --training-backend torch_ddp_lora \
  --harness Harness-1 \
  --benchmark bcplus_full \
  --model_name /tmp/gpt-oss-20b-bf16-dequant \
  --train_method trim \
  --component all \
  --lambda-opd 1.0 \
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
  --out "${OUT}" \
  --resume
