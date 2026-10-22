#!/usr/bin/env bash
# Gemma-3-27B-it × zero — bcplus_full (830 queries), upstream_api + local_bm25.
# CloudML 单节点 8 × gm402-80g, FULL 8-CARD LOAD (8 single-GPU actors, no verifier).
#
# Forked from run_bcplus_w_harness1_gemma3_27b_all_gpu8_eval.sh (9/17 CloudML all wrapper,
# lib-sourced, bishop-runtime-ready via ${PY:-python}). 把 COMPONENT 切到 zero + 8-actor
# 全载布局 + TEMPERATURE=0.0（Gemma3 专属：temp 1.0 产生 100% implicit_user_text 零搜索，
# 见 9/17 gemma 脚本注释）。复用 lib_bcplus_w_harness1_gpu8_eval.sh 的 lib_bcplus_gpu8_run_zero
# 路径，该路径 start_zero_stack 按 ZERO_ACTOR_GPUS 数组起多 actor（多 actor 模式用
# actor_${slug}_zero_gpu${gpu}.log），传 8 个 GPU 即 8-actor 全载。
#
# 与 9/15 standalone 脚本 run_bcplus_w_harness1_gemma3_27b_zero_gpu8_eval.sh 的差异：
#   - 本脚本 source lib（与 all 一致），不重复实现 actor 启停逻辑；9/15 standalone 自带
#     --temperature 1.0 硬编码（line 349），与 Gemma3 必须 0.0 冲突，故不用它。
#   - 本脚本 export TEMPERATURE=0.0 让 lib run_eval_cell 正确传参。
#
# 被测模型 = /mnt/songzijun/models/gemma-3-27b-it
#   config.json 架构 Gemma3ForConditionalGeneration / model_type gemma3
#   / text_config 62 层 hidden 5376 / dense / 12 shards ~51.10 GiB bf16 / VLM, BC+ 纯文本
#   / tokenizer: SentencePiece（tokenizer.model + tokenizer.json，非 harmony o200k）。
#
# vLLM 参数由 trim.eval.model_profiles vllm-extra 自动产出（GEMMA_PROFILE @ HEAD ba7677d）：
#   --enable-auto-tool-choice --tool-call-parser pythonic
#   --language-model-only --generation-config vllm
#   --chat-template trim/eval/templates/tool_chat_template_gemma3_pythonic.jinja
#   --max-model-len N --trust-remote-code
# （GEMMA_PROFILE 无 reasoning_parser；Gemma3 IT 无 thinking 模式，无需 strip。）
#
# Why 8-card full load (充分利用 8 卡显存，对齐 Qwen3.5-27B zero 成功 task t-...fcanl)：
#   * zero component 无 verifier，故 8 张卡全用来跑被测模型 actor（每卡 1 个 TP=1 actor）。
#   * 27B dense 权重 51.10 GiB；单卡 gm402-80g + GPU_UTIL 0.90 ~72 GiB，权重 51 GiB + KV ~16 GiB
#     = ~67 GiB/80 GiB，充分利用显存（同 Qwen3.5-27B zero 实测口径）。
#   * 8 actor (TP=1) + 16 worker → 每 GPU 2 worker；避免 TP=8 all-reduce 开销（27B 单卡放得下）。
#
# Optional env (defaults shown; boot 脚本已 export 大部分):
#   ZERO_ACTOR_GPUS=0,1,2,3,4,5,6,7  ZERO_ACTOR_PORTS=8040,8042,...,8054
#   TP=16  MAX_NUM_SEQS=8  MAX_MODEL_LEN=32768  GPU_UTIL=0.90  WORKER_STAGGER=10  TEMPERATURE=0.0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# When this wrapper is copied to /tmp (to dodge JuiceFS client-cache lag) and exec'd
# from there, BASH_SOURCE[0] resolves to /tmp and the sibling lib script is not found.
# Fall back to the canonical JuiceFS scripts dir in that case.
if [[ ! -f "${SCRIPT_DIR}/lib_bcplus_w_harness1_gpu8_eval.sh" ]]; then
  SCRIPT_DIR="/mnt/songzijun/Capability_Evolution/TRIM/scripts"
fi
# shellcheck source=lib_bcplus_w_harness1_gpu8_eval.sh
source "${SCRIPT_DIR}/lib_bcplus_w_harness1_gpu8_eval.sh"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_gemma3_27b_zero_cloudml_8gpu}"
MODEL_PATH="${MODEL_PATH:-/mnt/songzijun/models/gemma-3-27b-it}"
MODEL_SLUG="gemma-3-27b-it"
API_MODEL="${API_MODEL:-gemma-3-27b-it}"
COMPONENT="zero"

# CloudML boot 已 pip 安装 bishop runtime 并把 python/vllm 放进 PATH；不硬编码 conda 路径。
PY="${PY:-python}"
VLLM="${VLLM:-vllm}"

# 8 single-GPU actors (TP=1 each). No tensor parallel → no all-reduce cost. 8 卡全载。
export ZERO_ACTOR_GPUS="${ZERO_ACTOR_GPUS:-0,1,2,3,4,5,6,7}"
export ZERO_ACTOR_PORTS="${ZERO_ACTOR_PORTS:-8040,8042,8044,8046,8048,8050,8052,8054}"
export TP="${TP:-16}"               # 16 API-client workers → 2 per GPU
export GPU_UTIL="${GPU_UTIL:-0.90}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-8}"
export WORKER_STAGGER="${WORKER_STAGGER:-10}"
# Gemma3 专属：TEMPERATURE 必须 0.0（temp 1.0 → 100% implicit_user_text 零搜索）。
export TEMPERATURE="${TEMPERATURE:-0.0}"

lib_bcplus_gpu8_run_zero
