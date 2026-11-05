#!/usr/bin/env bash
# Probe C2 在线确认训练：Qwen3.5-9B 8-step 2node16gpu，正式 HarnessOPD 语义
# （rollout refresh / 真实 old logp / CISPO / projected-gap 全保持，仅步数=8）。
# COND ∈ harnessopd_m | rl_only | same_view：
#   harnessopd_m = --train_method trim + teacher=trieve 精确联盟（TRIM_TEACHER_COMPONENTS）
#   rl_only      = --train_method rl（无 OPD）
#   same_view    = --train_method zero-placebo（同机制无组件特权，Teacher prompt=Student）
# 从 θ0 全新开始（不承接 C1）；训练 query 排除 C_select/C_audit（--train-query-ids 白名单）。
set -euxo pipefail

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TRIM_GPU_KEEPALIVE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/mnt/songzijun/Capability_Evolution/TRIM:/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD
export SCAPE_SEC_RL_DATA=/mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-rl-data
export SCAPE_SEC_CORPUS_ROOT=/mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-sec-corpus

: "${COND:?FATAL: COND must be set (harnessopd_m|rl_only|same_view)}"
: "${OUT:?FATAL: OUT must be set}"
: "${STATES_OUT:?FATAL: STATES_OUT must be set (collection OUT with SPLIT_ASSIGNMENT.json)}"

# --- Multi-node distributed env guard（同训练 entry 铁律）---
NODE_RANK="${NODE_RANK:-${PET_NODE_RANK:-${NODE_INDEX:-${NODE_NUMBER:-}}}}"
if [ -z "${NODE_RANK:-}" ] || [ -z "${MASTER_ADDR:-}" ]; then
  echo "FATAL: NODE_RANK/MASTER_ADDR not set (CloudML multi-node inject)." >&2
  exit 1
fi
: "${MASTER_PORT:=29500}"
export MASTER_ADDR NODE_RANK MASTER_PORT
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK \
      GROUP_WORLD_SIZE TORCHELASTIC_RUN_ID TORCHELASTIC_RESTART_COUNT \
      TORCHELASTIC_MAX_RESTARTS TORCHELASTIC_USE_AGENT_STORE \
      PET_MASTER_ADDR PET_MASTER_PORT PET_NNODES PET_NPROC_PER_NODE PET_NODE_RANK
echo "C2_TRAIN cond=${COND} out=${OUT} node_rank=${NODE_RANK}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-**-****-***********-***-*-****-***}"
export SCAPE_LOCAL_OPENAI_EMBEDDINGS=1
export SCAPE_FORCE_LOCAL_HARMONY=1
export TRIM_VLLM_GENERATE_RETRIES=2
export SCAPE_CHROMA_PATH=/tmp/empty_chroma
mkdir -p "$SCAPE_CHROMA_PATH"
python3 -c "import zipfile; zipfile.ZipFile('/mnt/songzijun/tiktoken-bundle.zip').extractall('/tmp')"
export TIKTOKEN_ENCODINGS_BASE=/tmp/tiktoken-bundle/share/tiktoken
export TIKTOKEN_RS_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_rs_cache
export TIKTOKEN_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_cache

# --- JVM + 运行时依赖（与训练 entry 同配方）---
JVM_PATH=$(find /usr /opt -type f -name libjvm.so 2>/dev/null | head -1 || true)
if [ -z "${JVM_PATH}" ]; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends openjdk-21-jre-headless
  JVM_PATH=$(find /usr /opt -type f -name libjvm.so 2>/dev/null | head -1 || true)
fi
test -f "${JVM_PATH}"
export JVM_PATH
export JAVA_HOME=$(dirname "$(dirname "$(dirname "${JVM_PATH}")")")
export LD_LIBRARY_PATH=$(dirname "${JVM_PATH}"):${LD_LIBRARY_PATH:-}
python3 -m pip install --no-cache-dir --ignore-installed --upgrade-strategy only-if-needed \
  "torch==2.10.0" "transformers==5.17.0" "accelerate==1.13.0" "datasets==4.8.4" \
  "hydra-core==1.3.2" "omegaconf==2.3.0" "vllm==0.19.1" peft structlog chromadb \
  tqdm regex sympy latex2sympy2 word2number sentencepiece "pyserini==2.0.0" openai tenacity anthropic \
  tensorboard
PYSERINI_ROOT=$(python3 -c 'import pathlib, pyserini; print(pathlib.Path(pyserini.__file__).resolve().parent)')
export ANSERINI_CLASSPATH=$(find "$PYSERINI_ROOT/resources/jars" -type f -name '*.jar' -print | paste -sd: -)
test -n "$ANSERINI_CLASSPATH"
export CLASSPATH="$ANSERINI_CLASSPATH"

# --- Teacher 联盟（仅 harnessopd_m）+ TEACHER_MASK_OK 断言 ---
# zero-placebo 契约强制 --component zero（launch.py:1231 fail-fast），不能用 all。
METHOD="trim"
COMP="all"
case "${COND}" in
  harnessopd_m)
    # R1/R2 收敛：仅 auto_populate_first_search 有 READY 监督通道（C0/C1 铁证），
    # chunk_neighbors 未注册不伪造；teacher=单成员精确联盟。
    export TRIM_TEACHER_COMPONENTS="auto_populate_first_search"
    python3 -c "
import os
from trim.training.four_cell_runtime import teacher_mask_for, student_mask_for
t = teacher_mask_for('all'); s = student_mask_for('all')
assert sorted(k for k, v in t.items() if v) == ['auto_populate_first_search'], t
assert not any(s.values()), s
print('TEACHER_MASK_OK single-member')
"
    ;;
  rl_only)      METHOD="rl" ;;
  same_view)    METHOD="zero-placebo"; COMP="zero" ;;
  *) echo "FATAL: COND=${COND}" >&2; exit 1 ;;
esac
echo "C2_METHOD cond=${COND} method=${METHOD} component=${COMP}"

nvidia-smi -L
test -f /mnt/songzijun/models/Qwen3.5-9B/config.json
test -f "${STATES_OUT}/SPLIT_ASSIGNMENT.json"

# C2 训练/评测 query 隔离：训练白名单 = 池 − C_select − C_audit。
mkdir -p "${OUT}"
python3 -m trim.probes.probe_c_c2_eval make-train-ids \
  --states-out "${STATES_OUT}" --out "${OUT}/train_ids.txt"
test -s "${OUT}/train_ids.txt"

# 模型 cp /tmp（JuiceFS 并发读铁律）。
MODEL_SRC=/mnt/songzijun/models/Qwen3.5-9B
MODEL_LOCAL=/tmp/qwen35-9b
echo "MODEL_CP_START $(date -Is)"
cp -r "${MODEL_SRC}"/. "${MODEL_LOCAL}"/
echo "MODEL_CP_DONE $(date -Is)"

python3 -m py_compile /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_train_multinode.py
cd /mnt/songzijun/Capability_Evolution

# 训练成功后写 READY_MANIFEST（eval entry 的就绪判定依据；guide §12.4：
# manifest 含 checksum 与 checkpoint_committed=true，等待方不用 find 非空推断）。
cat > /tmp/c2_train_wrapper.sh <<'WRAPPER'
#!/usr/bin/env bash
set -euxo pipefail
# HOST 变量经环境传递（heredoc 为引号形式，不展开）：C2_* 由 entry export。
: "${C2_NODE_RANK:?}"; : "${C2_MASTER_ADDR:?}"; : "${C2_MASTER_PORT:?}"
: "${C2_OUT:?}"; : "${C2_COND:?}"; : "${C2_METHOD:?}"; : "${C2_COMP:?}"; : "${C2_MODEL_LOCAL:?}"
python3 /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_train_multinode.py \
  --nnodes 2 \
  --nproc-per-node 8 \
  --node-rank "${C2_NODE_RANK}" \
  --master-addr "${C2_MASTER_ADDR}" \
  --master-port "${C2_MASTER_PORT}" \
  --dist-backend nccl \
  --rdzv-backend static \
  --rdzv-id "probe-c-c2-9b-${C2_COND}-8step-16" \
  --training-backend torch_ddp_lora \
  --harness Harness-1 \
  --benchmark bcplus_full \
  --model_name "${C2_MODEL_LOCAL}" \
  --train_method "${C2_METHOD}" \
  --component "${C2_COMP}" \
  --train-env local_legacy \
  --train-data sec \
  --rl-data /mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-rl-data \
  --sec-corpus-root /mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-sec-corpus \
  --train-query-ids "${C2_OUT}/train_ids.txt" \
  --rl-reward final-answer-recall \
  --train-steps 8 \
  --train-groups-per-step 32 \
  --group-size 8 \
  --train-micro-batch-size 4 \
  --train-heartbeat-every 8 \
  --max-turns 40 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --seed 42 \
  --out "${C2_OUT}"
TRAIN_RC=$?
if [ "${TRAIN_RC}" = "0" ]; then
  ADAPTER=$( { find "${C2_OUT}"/checkpoints -maxdepth 5 -type d -name adapter -path "*step_00008*" 2>/dev/null || true; } | sort | tail -1)
  test -n "${ADAPTER}"
  test -f "${ADAPTER}/adapter_config.json"
  python3 - <<PY
import json, hashlib, pathlib
ad = pathlib.Path("${ADAPTER}")
files = sorted(p.name for p in ad.iterdir())
blob = json.dumps(files, sort_keys=True).encode()
meta = {
    "condition": "${C2_COND}",
    "method": "${C2_METHOD}",
    "component": "${C2_COMP}",
    "adapter_dir": str(ad),
    "adapter_files": files,
    "files_hash": hashlib.sha256(blob).hexdigest(),
    "checkpoint_committed": True,
    "written_at": __import__("time").strftime("%F %T"),
}
out = pathlib.Path("${C2_OUT}") / "READY_MANIFEST.json"
tmp = out.with_suffix(".json.tmp")
tmp.write_text(json.dumps(meta, indent=2, ensure_ascii=False, sort_keys=True))
tmp.replace(out)
print("READY_MANIFEST_OK", meta["adapter_dir"])
PY
else
  echo "TRAIN_FAILED rc=${TRAIN_RC}" >&2
fi
exit ${TRAIN_RC}
WRAPPER
export C2_NODE_RANK="${NODE_RANK}" C2_MASTER_ADDR="${MASTER_ADDR}" C2_MASTER_PORT="${MASTER_PORT}"
export C2_OUT="${OUT}" C2_COND="${COND}" C2_METHOD="${METHOD}" C2_COMP="${COMP}" C2_MODEL_LOCAL="${MODEL_LOCAL}"
chmod +x /tmp/c2_train_wrapper.sh
# heredoc 内嵌脚本回读可能撞 JuiceFS 抖动：cp 到 /tmp 已天然满足（脚本就在 /tmp）。
exec bash /tmp/c2_train_wrapper.sh
