# TRIM Harness-1 BC+ full GPT-OSS-20B 训练任务归档说明

归档时间:2026-09-11
归档任务:CloudML custom_train `trim-harness1-bcplus-full-gpt-oss-20b-trim-8gpu-20260910`(recovery0/1/2 三个 attempt)

## 任务概况

- workspace `10063`,context `songzijun-cto`,队列 `2334`(记忆对话,集群 kscn-hd-cloudml-prod/青海),priority 8,`GUARANTEED_PUBLIC`,单节点 8 卡 `cloudml.ng1gm402-8-8.20-224`(gm402-80g)。
- 镜像 `micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto`,标签 `MemQA-General-V5`。
- JuiceFS 卷 `songzijun`/集群 `bj9-cloudml`/subPath `/`/挂载 `/mnt/songzijun`/读写,通过直接 CLI `--juicefs_mount_configs` 传入(YAML 不含挂载)。
- 启动脚本:`run_train.py --harness Harness-1 --benchmark bcplus_full --model_name /mnt/songzijun/models/openai/gpt-oss-20b --train_method trim --component all --train-groups-per-step 32 --train-micro-batch-size 4 --train-heartbeat-every 8`。
- 四包锁定(preflight 断言):torch==2.10.0 / transformers==4.57.6 / vllm==0.19.1 / pyserini==2.0.0;附加速 accelerate/datasets/hydra/omegaconf/peft/structlog/chromadb/openjdk-21。
- tiktoken 离线:`/mnt/songzijun/tiktoken-bundle.zip` 解压到 `/tmp/tiktoken-bundle`,设 `TIKTOKEN_ENCODINGS_BASE/RS_CACHE_DIR/CACHE_DIR`(o200k_base.tiktoken,Harmony 用)。

## 三个 attempt 与问题演进

### attempt 0: `t-20260910202309-keyys`(原始)— failed (runTime 643s)
- preflight 四包/tiktoken/Lucene 全过。
- **根因**:`train_env=upstream`(默认)在 `four_cell_runtime.py:1706-1729` 调 `open_upstream_train_session`,需镜像缺的 `chz` 模块,报 `No module named 'chz'`,抛 SystemExit。该 block 不 fallback 到 TRIM local env。
- **修复**:加 `--train-env local_legacy`(`trim/cli/launch.py:689`,choices 仅 `upstream|local_legacy`),跳过 upstream-only block。SEC 训练检索 `open_train_retrieval`(`four_cell_runtime.py:1581`)本来就用本地 Lucene BM25 索引(经 `--sec-corpus-root`),独立于 `--retrieval-backend`。

### attempt 1 (recovery1): `t-20260910204048-va5zw` — failed (runTime 947s)
- 过了 preflight + vLLM TP=8 初始化,进入首个 rollout。
- **根因**:imageCommand 设了 `SCAPE_FORCE_LOCAL_HARMONY=1`(从旧 trim-eval 模板继承),`external/harness-1/harness/ultra_core.py:31` 检测该 flag 强制 `Conversation = _LocalConversation`(走 `_LocalHarmonyEncodingFallback`),`_LocalConversation` 无 `to_json`,破坏 `openai_harmony.render_conversation_for_completion`。报错 `AttributeError: '_LocalConversation' object has no attribute 'to_json'`。
- **修复**:删 `SCAPE_FORCE_LOCAL_HARMONY`(imageCommand export + envConfigs key 都删),走 `ultra_core.py:57-71` 真 `openai_harmony.Conversation`。成功 eval 任务不设该 flag。

### attempt 2 (recovery2): `t-20260910210458-f0wx5` — step0 成功 + step1 hang(02:50 kill)
- 过了两处失败点,稳定训练:
  - PREFLIGHT ✓(torch 2.10.0+cu128/vllm 0.19.1/transformers 4.57.6/pyserini 2.0.0/gpu_count=8/o200k_base_present/java_ok)
  - LUCENE_OK ✓(num_docs=2115106)
  - LAUNCH.json `train_env=local_legacy`
  - `[train] runtime wiring pass student_on=0 teacher_on=10 opd_steps=1`
  - `[train] cell=scape_seed on-policy rollout step=0 policy=v0` + `vLLM start tp=8`
- **step 0 完整成功**:rollout chunk 1-108(21:20:15→01:56:24,稳态 ~2.55 min/chunk)→ 加载 HF 训练模型 + save theta0 LoRA → `optim_step_start step=1 n_steps=1 n_pool_groups=3453 n_step_groups=32 sampled=true micro_batch_size=4`(01:59:31)→ `hf_fb phase=cispo`(student fwd-bwd,1536 datum,291s)→ `hf_fb phase=opd_gap`(student+teacher fwd)→ `hf_fb sampled_gap`(415s)→ `hf_optim_step`(LoRA 更新,0.08s)→ **`optim_step_done step=1 elapsed_s=892.846 update_type=rl_opd_joint n_rl_datums=1536 n_opd_datums=1536 n_rl_tokens=589619 n_opd_tokens=313481`**(02:14:24)。
- **step 1 hang**:02:14:50 `cell=scape_seed on-policy rollout step=1 policy=v1` → 02:14:58 `[rollout] chunk 1/108 queries=32 episodes=256 prep=0.0s -> vLLM`(prep=0=vLLM 引擎复用未 reload)→ **33 min 无新日志**,runTime 仍涨(20662s)= 进程活但 vLLM 死锁。无 Traceback/OOM/NCCL 报错。02:50 用 `cml custom_train kill` 停止(保留输出)。

## 结构理解(关键更正)

- `--train-steps` 默认 **8**(`trim/cli/launch.py:580`),`n_train=8`,`for step in range(8)`(step 0..7)。
- `optim_step_done` 的 `n_steps=1` 是 `train_cell` 内**每个 step 的 optim 次数**(=1),**非总 step 数**。全程 = 8 step × ~51min(rollout 36 + optim 15)≈ 6.8h。
- step 0 的 vLLM chunk 1 是冷启动(prep=103.6s,全新引擎);step 1 复用引擎(prep=0.0s)后卡死。
- SR-OPD+CISPO 协议(CLAUDE.md `sr_opd_ce`+CISPO),`opd_loss=sr_opd_projected_gap`,teacher_kind=upstream。

## step1 hang 根因(初步,未最终定位)

- step 边界 `load_adapter_weights`(`vllm_hybrid.py:611`)把 LoRA `load_state_dict` 到 `backend.model`(HF PEFT),**不是 vLLM engine**;vLLM engine 复用(prep=0.0)。
- 疑似:vLLM TP=8 引擎在 step 边界复用 + policy v0→v1 LoRA adapter swap 后生成死锁(`enforce_eager=True` 已设但仍 hang)。
- 深查 vLLM step1 hang 根因的 agent 在完成前被停止(用户改为归档)。

## 归档内容

```
trim_gptoss20b_train_20260910_archive/
├── README_ARCHIVE.md                      # 本说明
├── 01_startup_scripts/                    # 启动脚本(YAML + submit sh)
│   ├── cloudml_..._20260910_not_submitted.yaml        # attempt0 原始
│   ├── submit_..._20260910_not_submitted.sh
│   ├── cloudml_..._20260910_recovery1_not_submitted.yaml  # attempt1
│   ├── submit_..._20260910_recovery1_not_submitted.sh
│   ├── cloudml_..._20260910_recovery2_not_submitted.yaml  # attempt2(最新成功过step0)
│   └── submit_..._20260910_recovery2_not_submitted.sh
├── 02_training_output/                    # recovery2 训练产出(step0 有效)
│   └── cloudml_custom_train_harness1_bcplus_full_gpt_oss_20b_trim_8gpu_20260910_recovery2/
│       ├── LAUNCH.json
│       ├── RUN_MANIFEST.json
│       ├── RUNTIME_EFFECT_AUDIT.json
│       ├── adapters/   (theta0 + scape_seed LoRA)
│       └── vllm_sessions/
└── 03_problem_notes/                      # 问题记录
    └── (本 README_ARCHIVE.md 的副本)
```

## 当前状态与定级

- recovery2 的 step0 是**有效 partial 训练产出**(rollout+rl_opd_joint optim 成功),保留在 `02_training_output/`。
- 三个 attempt 均保留原状,不覆盖、不混入 formal artifact。
- step1 hang 未最终定位;如需继续修复,方向:深查 vLLM step 边界引擎复用/adapter swap 死锁,或开 DiagOptions HangDetection、或降到 `--train-steps 1` 只保 step0。
