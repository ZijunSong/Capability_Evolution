---
name: trim-gptoss20b-rl-4step-20260911
description: "CloudML job t-20260911191937-y8uax, 8x gm402-80g, GPT-OSS-20B RL 4-step training, 2026-09-11 监控中"
metadata: 
  node_type: memory
  type: project
  originSessionId: 5cd5c673-f50c-4fdf-81a3-91285e2d1ae0
  modified: 2026-09-11T14:23:09.613Z
---

2026-09-11 提交 TRIM `run_train.py` RL 训练（GPT-OSS-20B, train_method=rl, train-data=sec, train-steps=4）到 CloudML。

- **Job**: `t-20260911191937-y8uax`，job_name `trim-harness1-bcplus-full-gpt-oss-20b-8gpu-rl-4step-20260911`
- **workspace** 10063，**队列 2334**（记忆对话, kscn-hd-cloudml-prod），8×gm402-80g，GUARANTEED_PUBLIC，priority 8，tag MemQA-General-V5，非抢占
- **镜像** `micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto`
- **JuiceFS**: volume=songzijun, juiceFsCluster=bj9-cloudml, subPath=/, mountPath=/mnt/songzijun, 通过 CLI `--juicefs_mount_configs` 直传（endpoint 字段被服务端忽略，juiceFsCluster=bj9-cloudml 生效）。describe 确认 endpoint 未保留，仅 juiceFSCluster。
- **输出目录**: `Capability_Evolution/TRIM/outputs/trim_train_harness1_bcplus_full_gpt_oss_20b_8gpu_rl_4step_20260911`（独立，不混入旧 attempt）
- **YAML/脚本**: `TRIM/cloudml_custom_train_harness1_bcplus_full_gpt_oss_20b_rl_4step_20260911_not_submitted.yaml` + `submit_cloudml_custom_train_harness1_bcplus_full_gpt_oss_20b_rl_4step_20260911_not_submitted.sh`
- **pod**: `tj-3050749-t-20260911191937-y8uax-master-0`

**与 recovery8（Qwen3-4B trim, train-steps=8, 108 chunks/step）的关键差异**：本任务 train-steps=4，每 step 只有 **1 chunk（256 episodes，32 queries × group-size 8）**，不是 108 chunks。日志显示 `chunk 1/1`。单 step = 1 chunk rollout + 1 optim，节奏快得多。

**启动时间线（19:19 提交 → 全程无报错）**：
- 19:19:45 创建 → 排队（8-GPU gang 调度，Unschedulable waiting resource）
- 19:20 左右 running，runTime 计时
- 19:21 apt openjdk-21 + JVM 探测（`/usr/lib/jvm/java-21-openjdk-amd64/lib/server/libjvm.so`）+ ldd
- 19:21–19:31 pip install（torch 2.10.0+cu128 / vllm 0.19.1 / transformers 4.57.6 / pyserini 2.0.0 / peft 0.20.0 / chromadb 1.5.9 / structlog 26.1.0，verl 可选依赖告警不阻断）
- 19:31 ANSERINI_CLASSPATH→`pyserini/resources/jars/anserini-2.0.0-fatjar.jar`，py_compile（run_train/vllm_rollout_worker/rl_opd_types 全过）
- 19:31:28 **PREFLIGHT** 全断言过（gpu_count=8, java_ok, o200k_base_present）
- 19:32:03 **LUCENE_OK** num_docs=2115106
- 19:34:20 `[train] runtime wiring pass student_on=0 teacher_on=10 opd_steps=None`
- 19:36:52 `[train] cell=rl on-policy rollout step=0 policy=v0`（32 queries）
- 19:36:53 `vLLM start backend=vllm tp=8 lora=None tag=rl_rollout0`
- 19:36:54–19:39:54 vLLM worker start（elapsed 0→180s，约每 45s 心跳）
- 19:42:01 **`[rollout] chunk 1/1 queries=32 episodes=256 prep=91.8s backend=vllm`** → vLLM READY
- 19:42:04+ vLLM generate 分批推进（每批 ~90s，elapsed 0→90→reset→0→90…）

**LAUNCH.json 注意点**: `vllm_disable_custom_all_reduce: null`（CLI 未显式传该 arg，默认 None）。但 worker 补丁读 `os.environ.get("DISABLE_CUSTOM_ALL_REDUCE","1")=="1"` 设 True，envConfigs 里已 export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1，故 TP8 仍走 NCCL。环境变量路径生效（见 [[trim-vllm-rollout-worker-disable-custom-all-reduce]]）。

**监控**: 已设 session-only cron `8120da40`，每小时 :17/:47 触发，汇报 step/rollout 进度并更新 ETA。仍为 running partial，不得作为 formal artifact。关联 [[cloudml-trim-train-job-config]]、[[trim-gptoss20b-train-20260910]]、[[trim-rlopd-types-dataclass-field-order-bug]]。

**2026-09-11 20:28 监控更新（非卡死确认 + 实测节奏）**:
- runTime=4088s(~68min)，state=running，无 Traceback/OOM/EngineDeadError（error scan 仅命中 pip 阶段 verl 依赖告警，不阻断）。
- step0 rollout 仍在 generate：19:42 启动 chunk 1/1 后至 20:28 未打印完成标记，但 **vLLM 在持续产出，非卡死**。判据：`vllm_sessions/001_rl_rollout0/vllm_worker.log` 持续增长（421KB→445KB），mtime 滚动到 20:28:35；`result.json` 已 9.8MB 累积中间结果。
- **实测节奏**：vLLM 多轮 generate，每轮 `Processed prompts: 253/253→254→255→256`（batch size 随未完成 episode 数缓慢回升，说明部分长 episode 仍在跑），每轮约 **104s**（output ~2488 toks/s, input ~14000 toks/s）。主日志的 `vllm_wait what=vLLM generate elapsed 0→90→reset` 心跳正好对应每轮 batch。
- **关键认知**：单 RL rollout 的 episodes 是多轮 agent 对话（max-turns=40 + 多轮检索），需要十几~几十轮 vLLM batch 才能让所有 256 episode 达到终止条件。至 20:28 已跑 ~18 轮 batch ≈ 31min 纯生成，仍未完成 step0。这与 recovery8 的"108 chunks/step、每 chunk 一轮"不同——**本任务单 chunk 内是多轮 agent rollout**，节奏比预期长。
- 主日志末尾仍只滚动 `vllm_wait generate` 心跳，未见 step0 done/optim_step_done/step1。下次监控重点确认 step0 何时闭合、optim 时长，据此校准 4-step 总 ETA。
