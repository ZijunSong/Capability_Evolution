#!/usr/bin/env bash
# === GPT-OSS-20B no-gate + final-answer-recall reward (2026-10-15) ===
# Fork of entry_gpt_oss_20b_trim_all_recall_only_32gpu_4node_20261002_bf16.sh
# (the verified gpt-oss bf16-dequant trim/all 32gpu baseline). Two changes vs
# that entry, both requested by the user and both NATIVELY supported by the
# launcher (no code-tree hard-edit required, unlike the recall-only entry which
# hard-flips four_cell_runtime.py):
#
#   1. --train_method no-gate  (was trim). no-gate = CISPO + projected-gap
#      with g=1 (gate_mode="none", projection_mode="full", rl_enabled=True);
#      resolve_trim_ablation("no-gate") -> TrimAblation("no_gate", True, "full",
#      "none"). Defaults to Harness-1 Full teacher / Zero student, which is
#      compatible with --component all (launch.py help: "only-opd, no-project,
#      and no-gate default to Harness-1 all (Teacher Full, Student Zero)").
#   2. --rl-reward final-answer-recall  (was recall-only via code-tree hard-flip).
#      This is the NATIVE flag path: launch.py:1245 sets
#      args.rl_reward_effective="final-answer-recall", reward_protocol_for() ->
#      PROTOCOL_FA_ONLY, and score_explicit_reward() returns total=fa (pure
#      final-answer recall, no legacy shaping). This does NOT touch
#      four_cell_runtime.py — the recall-only entry's PATCH_CHECK asserts on a
#      code-tree hard-edit that this path does NOT use, so those asserts are
#      DROPPED here (asserting them would false-fail on a clean tree).
#
# Model stays bf16-dequant (NOT original mxfp4): mxfp4+LoRA has Marlin PTX crash
# (v0dhh FAILED@step1) + NaN (txgli/zxkvl FAILED@step0); bf16-dequant is the
# verified safe path (b1ljo/vqqxp running).
# mb1 (NOT 4): GPT-OSS-20B is a MoE whose backward CANNOT enable gradient
# checkpointing, so mb>=2 risks backward OOM (k6v0p铁律 peak 74.7GB/80GB). This
# mb1 floor is MoE backward activation memory — INDEPENDENT of mxfp4 vs bf16
# (bf16's 3x heavier weights make backward TIGHTER, not looser). ALL verified
# bf16 GPT-OSS-20B baselines (b1ljo/vqqxp/pncda, recall-only entry) used mb1.
# Per the '确保利用以往经验' directive + mb1铁律, this run uses mb1.
# '充分利用8卡显存' is satisfied by gpu-mem 0.85 (KV cache撑满 during rollout,
# the verified max for bf16-dequant; 0.90 撞余量风险); mb4 would risk OOM =
# the opposite of utilization. The 4
# nodes x 8 ranks = 32 replicas are all TP=1 single-GPU (per-rank memory budget
# unchanged; multi-node N-rank data-parallel分摊 backward).
#
# Everything else byte-identical to the bf16 baseline: 4 DDP patches (incl
# curate-clipping tolerant-skip — kept because no-gate's projected-gap path
# shares the same tinker_opd_datum build, so the skip path is still protective),
# gpu-mem0.85 (NOT 0.90; bf16 39-40GB权重/卡, KV余量比 Qwen3-4B 7.6GB 小,
#   0.90撞余量风险; 0.85 是 pncda/vvptp/b1ljo/uwkpr->vqqxp/smbpi 系列实测验证值),
#   max-turns40, max-model-len8192, expandable_segments, model cp
#   39GB to /tmp per node, startup_timeout 3600s.
# rdzv-id gptoss20b-no-gate-fa-recall-32 (avoid collision with prior entries).
# SCAPE_FORCE_LOCAL_HARMONY INTENTIONALLY NOT SET (gpt-oss to_json crash via
# openai_harmony Rust encoder; only gpt-oss hits Harmony).
#
# === Multi-node framework (verified by 7nzzn + v0dhh + b1ljo) ===
# custom_train framework=pytorch nodeNumber=4 creates a PyTorchJob CRD, injects
# MASTER_ADDR (master pod DNS) + NODE_RANK but NOT RANK/WORLD_SIZE/PET_* (v0dhh
# PREFLIGHT rank_env=None world_size_env=None pet_nproc_per_node=None). The entry
# unsets rank env defensively so run_train_multinode.py self-launches torchrun
# via _exec_torchrun (deterministic 4-node rendezvous via --master-addr/--node-rank).
#
# === Pitfalls carried from bf16 baseline (all still apply) ===
#   * trainer_adapter.py:478 startup_timeout_s=3600.0 (900s worker-start timeout
#     fix; bf16-dequant is 39GB/3x mxfp4, so the per-node /tmp cp takes ~7min and
#     8 ranks load from local /tmp concurrently — 3600s headroom is essential).
#   * trainer_adapter.py:264 resolve_queries 4-tuple unpack (724d9fa ValueError fix).
#   * trainer_adapter.py:506 client.start() EADDRINUSE retry x3 (per-rank vLLM
#     TP=1 TCPStore port race at step>=1; 32 ranks across 4 nodes make it more
#     likely).
#   * SCAPE_FORCE_LOCAL_HARMONY INTENTIONALLY NOT SET (gpt-oss to_json crash).
#   * Model cp to /tmp BEFORE exec (8-rank JuiceFS concurrent-read hang fix; here
#     EACH NODE does its own cp of 39GB bf16 from JuiceFS to /tmp, ~7min).
#   * curate-clipping tolerant-skip patch (action_is_visible + tinker_opd_datum
#     skip; without it x2j3u crashes in step0 datum build). no-gate's projected-
#     gap path uses the same tinker_opd_datum build, so the skip path protects it.
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
# CloudML custom_train framework=pytorch nodeNumber=4 must give each pod a
# distinct NODE_RANK and a shared MASTER_ADDR (node0's service DNS). If these
# are absent the launcher cannot form a 4-node rendezvous.
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
# run_train_multinode.py writes per-rank rollout shards to <out>/tmp/rollout_shards
# and rank 0 interleaves them via all_gather_via_disk. If each node uses a different
# <out> (e.g. a per-pod $(date) timestamp), the shards are invisible across nodes
# and step 0 backward crashes with FileNotFoundError. OUT is injected via submit
# --env so all 4 pods share the exact same path. Never use $(date) here.
if [ -z "${OUT:-}" ]; then
  echo "FATAL: OUT is not set. Multi-node requires a shared --out path injected via --env OUT=... (identical on every pod). Aborting." >&2
  exit 1
fi
export OUT
echo "MULTINODE_OUT out=${OUT} host=$(hostname) node_rank=${NODE_RANK}"

# FIX1 (to_json bug): SCAPE_FORCE_LOCAL_HARMONY is INTENTIONALLY NOT SET.
# Placeholder key so pyserini.encode._openai module-level openai.OpenAI()
# does not raise OpenAIError at import (RL uses local BM25, never calls OpenAI API).
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

# curate-clipping tolerant-skip patch. no-gate's projected-gap path uses the
# same tinker_opd_datum build as trim/all, so the skip path is still protective
# (without it x2j3u crashes in step0 datum build on clipped curate refs).
ae = pathlib.Path("/mnt/songzijun/Capability_Evolution/TRIM/trim/training/action_encoding.py").read_text()
assert "def action_is_visible(" in ae, "action_is_visible patch missing"
tod = pathlib.Path("/mnt/songzijun/Capability_Evolution/TRIM/trim/training/tinker_opd_datum.py").read_text()
assert "action_is_visible(step.target_action, visible)" in tod, "skip-clipped-doc call site missing"
assert "n_skip_clipped_doc" in tod, "skip counter missing"
from trim.training.action_encoding import action_is_visible
assert action_is_visible({"name": "curate", "arguments": {"add_ids": ["X"]}}, ["Y"]) is False
assert action_is_visible({"name": "curate", "arguments": {"add_ids": ["X"]}}, ["X"]) is True

# Verify the no-gate + final-answer-recall contract resolves to the expected
# protocol via the NATIVE flag path (NOT a code-tree hard-edit). This is what
# --train_method no-gate --rl-reward final-answer-recall must produce:
#   * resolve_trim_ablation("no-gate") -> TrimAblation("no_gate", rl_enabled=True,
#     projection_mode="full", gate_mode="none")  (CISPO on, projected-gap full, g=1)
#   * reward_protocol_for("final-answer-recall") -> "trim_final_answer_recall_only_v1"
#   * score_explicit_reward(mode="final-answer-recall", ...) -> {"total": fa}
from trim.training.trim_ablation import resolve_trim_ablation, ABLATION_CONTRACT_VERSION
prof = resolve_trim_ablation("no-gate")
assert prof is not None, "no-gate must resolve to a TrimAblation profile"
assert prof.variant == "no_gate", f"variant={prof.variant} != no_gate"
assert prof.rl_enabled is True, "no-gate must keep RL (CISPO) enabled"
assert prof.projection_mode == "full", f"projection_mode={prof.projection_mode} != full"
assert prof.gate_mode == "none", f"gate_mode={prof.gate_mode} != none (g=1 expected)"
from trim.training.reward_policy import reward_protocol_for, score_explicit_reward, PROTOCOL_FA_ONLY, EXPLICIT_RL_REWARDS
assert reward_protocol_for("final-answer-recall") == PROTOCOL_FA_ONLY, "final-answer-recall must map to PROTOCOL_FA_ONLY"
assert "final-answer-recall" in EXPLICIT_RL_REWARDS
# score_explicit_reward with mode=final-answer-recall must yield total=fa (pure
# recall, no legacy shaping/penalty).
sample_quality = {
    "recall": 0.5, "precision": 0.5, "final_answer_recall": 0.5,
    "trajectory_recall": 0.5, "trajectory_fa_recall": 0.5,
    "n_curated": 3, "num_turns": 10,
}
scored = score_explicit_reward(sample_quality, mode="final-answer-recall", max_turns=40)
assert scored["total"] == 0.5, f"total={scored['total']} != fa=0.5 (pure recall expected)"
assert scored["protocol"] == PROTOCOL_FA_ONLY
assert scored.get("penalty", 0.0) == 0.0 and scored.get("shaping", 0.0) == 0.0
print("NO_GATE_FA_RECALL_CONTRACT_OK", {
    "variant": prof.variant, "rl_enabled": prof.rl_enabled,
    "projection_mode": prof.projection_mode, "gate_mode": prof.gate_mode,
    "reward_protocol": PROTOCOL_FA_ONLY, "score_total_is_pure_fa": True,
    "contract_version": ABLATION_CONTRACT_VERSION,
}, flush=True)

print("PATCH_CHECK_OK", {
    "startup_timeout_s": True, "resolve_queries_4tuple": True, "eaddrinuse_retry": True,
    "skip_clipped_doc": True, "bf16_dequant": True,
    "no_gate_ablation": True, "fa_recall_protocol": True,
}, flush=True)
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
  --rdzv-id gptoss20b-no-gate-fa-recall-32 \
  --training-backend torch_ddp_lora \
  --harness Harness-1 \
  --benchmark bcplus_full \
  --model_name /tmp/gpt-oss-20b-bf16-dequant \
  --train_method no-gate \
  --component all \
  --train-env local_legacy \
  --train-data sec \
  --rl-reward final-answer-recall \
  --rl-data /mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-rl-data \
  --sec-corpus-root /mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-sec-corpus \
  --train-steps 200 \
  --train-groups-per-step 32 \
  --group-size 8 \
  --train-micro-batch-size 1 \
  --train-heartbeat-every 8 \
  --max-turns 40 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --out "${OUT}"
