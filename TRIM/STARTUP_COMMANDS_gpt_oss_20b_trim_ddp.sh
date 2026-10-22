#!/bin/bash
# =============================================================================
# GPT-OSS-20B mxfp4 trim DDP 100step 训练 - 完整启动指令
# 任务: t-20260919095305-qxmz5 (recov2)
# 生成时间: 2026-09-22 (已完成 22/100 step 时打包)
# =============================================================================

# -----------------------------------------------------------------------------
# 1. 任务基本信息
# -----------------------------------------------------------------------------
JOB_ID="t-20260919095305-qxmz5"
JOB_NAME="trim-harness1-bcplus-full-gpt-oss-20b-trim-ddp-100step-8gpu-20260919-recov2"
QUEUE_ID="2334"                          # gm402-80g, 青海集群 kscn-hd-cloudml-prod
PRIORITY="8"
TAG="MemQA-General-V5"
IMAGE="micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto"
POD="tj-3050749-t-20260919095305-qxmz5-master-0"
OUT="/mnt/songzijun/Capability_Evolution/TRIM/outputs/cloudml_custom_train_harness1_bcplus_full_gpt_oss_20b_trim_ddp_100step_8gpu_20260919_recov2_20260919_100609"

# -----------------------------------------------------------------------------
# 2. YAML 文件
# -----------------------------------------------------------------------------
YAML="/mnt/songzijun/Capability_Evolution/TRIM/scripts/cloudml_custom_train_harness1_bcplus_full_gpt_oss_20b_trim_ddp_100step_8gpu_20260919_recov2.yaml"
# (本打包内含该 YAML 副本)

# -----------------------------------------------------------------------------
# 3. 最终 cml custom_train 提交命令
# -----------------------------------------------------------------------------
# JuiceFS 通过 CLI --juicefs_mount_configs 传 JSON (endpoint 字段名=juiceFSCluster, 值=bj9-cloudml)
# 不通过 YAML --filename 方式挂载, 避免 YAML 内 juiceFsMountConfigs 双重挂载
cml custom_train submit \
  --filename "$YAML" \
  --access_type PRIVATE \
  --juicefs_mount_configs '[{"volume":"songzijun","juiceFSCluster":"bj9-cloudml","subPath":"/","mountPath":"/mnt/songzijun","readOnly":false}]'

# -----------------------------------------------------------------------------
# 4. 核心训练参数 (YAML 内 imageCommand 末尾的 exec python3 run_train.py 段)
# -----------------------------------------------------------------------------
# exec python3 /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_train.py \
#   --training-backend torch_ddp_lora \        # DDP LoRA 后端, commit 6e8dea4 修复 world_size=8 真启动
#   --nproc-per-node 8 \                      # 8 rank, per-rank vLLM TP=1
#   --harness Harness-1 --benchmark bcplus_full \
#   --model_name /tmp/gpt-oss-20b \            # 指向本地盘副本 (cp 自 JuiceFS, 修复 8 路并发读卡死)
#   --train_method trim --component all \
#   --train-env local_legacy --train-data sec \
#   --rl-data /mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-rl-data \
#   --sec-corpus-root /mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-sec-corpus \
#   --train-steps 100 \                        # 100 step
#   --train-groups-per-step 32 --group-size 8 \  # 256 episodes/step
#   --train-micro-batch-size 1 \              # mb1: 20B mxfp4 backward 不 OOM 的必要约束
#   --train-heartbeat-every 8 \
#   --max-turns 40 \                           # 深轨迹
#   --max-model-len 8192 \
#   --gpu-memory-utilization 0.70 \            # 折中水位: 权重13.6G + KV40.3G = 54G/80G, 留26G给backward
#   --out <OUT>

# -----------------------------------------------------------------------------
# 5. 三层关键修复 (相对原始 Qwen3-4B 模板的改动)
# -----------------------------------------------------------------------------
# 修复1 (recov1): JuiceFS 8 路并发读 13GB 模型卡死
#   imageCommand 加: cp -r /mnt/songzijun/models/openai/gpt-oss-20b/. /tmp/gpt-oss-20b/
#   + --model_name 改指向 /tmp/gpt-oss-20b (本地盘, 8 rank 秒加载)
#
# 修复2 (recov2): harmony to_json 崩溃
#   删除 SCAPE_FORCE_LOCAL_HARMONY=1 (imageCommand export + envConfigs 两处)
#   原因: 该 env 触发 ultra_core.py:31 本地 fallback, Conversation=_LocalConversation(无 to_json)
#         传给 openai_harmony:470 conversation.to_json() 崩. gpt-oss 走 harmony 必触发, Qwen 不走故无害
#
# 修复3 (配置): 20B mxfp4 backward OOM 规避
#   --train-micro-batch-size 1 (非默认 4) + --gpu-memory-utilization 0.70 (非默认 0.90)
#   原因: 20B MoE backward 不能开 gradient checkpointing (hf_tool_opd.py:243 注释), mb≥2/gpu0.90 必 OOM
#   实测峰值 74774MB/80GB (93.5%), 1275 microbatch 全程无 OOM

# -----------------------------------------------------------------------------
# 6. provenance 审计三件套 (world_size=8 真启动, 已全部验证通过)
# -----------------------------------------------------------------------------
# (1) stdout: [verl-fsdp2-launch] banner + torch.distributed.run --nproc_per_node 8
# (2) LAUNCH.json: dist.world_size=8 (本打包内含 LAUNCH.json)
# (3) vllm_sessions/: 每组 8 个 session scape_seed_b{N}_r0~r7 (本打包内含)

# -----------------------------------------------------------------------------
# 7. 实测节奏与 ETA (截至打包时 2026-09-24)
# -----------------------------------------------------------------------------
# 已完成: 36/100 step, runTime 121h (已运行超 5 天)
# 节奏: ~3.3h/step (稳态无退化, 实测 ~3.14h/step 略快), = rollout ~1.1h + backward ~2.2h
# 近期 step32-35 间隔: 3h22m / 3h24m / 3h26m / 3h17m (极稳定无退化)
# 剩余: 64 step × 3.3h ≈ 211h ≈ 8.8 天
# 预计完成: ~2026-10-02
# 详见 SPEED_ANALYSIS_gpt_oss_20b_trim_ddp.md
