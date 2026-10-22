# CloudML TRIM Harness-1 BC+ full Qwen3-4B-Instruct 训练恢复记录 (2026-09-10)

## 最终成功任务
- **Job**: `t-20260910223920-j4xxy` (recovery8)
- **job_name**: `trim-harness1-bcplus-full-qwen3-4b-trim-8gpu-20260910-memqa-general-v5-recovery8`
- **提交脚本**: `Capability_Evolution/TRIM/submit_cloudml_custom_train_harness1_bcplus_full_qwen3_4b_trim_8gpu_20260910_memqa_general_v5_not_submitted.sh`
- **资源 YAML**: `Capability_Evolution/TRIM/cloudml_custom_train_harness1_bcplus_full_qwen3_4b_trim_8gpu_20260910_memqa_general_v5_not_submitted.yaml`
- **队列**: 2334 (记忆对话, kscn-hd-cloudml-prod), 8×gm402-80g, GUARANTEED_PUBLIC, priority 8, tag MemQA-General-V5
- **镜像**: `micr.cloud.mioffice.cn/songzijun/songzijun-cto:3050749-songzijun-cto`
- **JuiceFS**: volume=songzijun, endpoint=bj9-cloudml, mountPath=/mnt/songzijun (通过 --juicefs_mount_configs 直接传，非 --filename)
- **输出目录**: `Capability_Evolution/TRIM/outputs/cloudml_custom_train_harness1_bcplus_full_qwen3_4b_trim_8gpu_20260910_memqa_general_v5_recovery8`
- **状态 (22:53)**: running, step=0 on-policy rollout chunk 1/108, vLLM tp=8 已就绪并产出
- **证明为 partial/running，非 formal artifact**：仍需独立审计后续阶段，不得混入 formal 汇总

## 启动命令关键参数
```
python3 TRIM/scripts/run_train.py --harness Harness-1 --benchmark bcplus_full \
  --model_name /mnt/songzijun/models/Qwen3-4B-Instruct-2507 \
  --train_method trim --component all \
  --rl-data /mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-rl-data \
  --sec-corpus-root /mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-sec-corpus \
  --retrieval-backend local_bm25 \
  --index-path .../harness-1-sec-corpus/indexes/bm25 \
  --corpus-path .../harness-1-sec-corpus/sec_corpus_chunks.jsonl \
  --reranker none --train-env local_legacy \
  --train-groups-per-step 32 --train-micro-batch-size 4 --train-heartbeat-every 8 \
  --out ...recovery8
```

## 恢复历程 (recovery0=首次 → recovery8=成功)
- recovery0 (`t-20260910202401-2pztj`): preflight `from pyserini.search.lucene import LuceneSearcher` 报 `NoClassDefFoundError: io/anserini/eval/Qrels`。根因：pyserini.pyclass 的 configure_classpath 被 bare try/except 吞，anserini fatjar 没进 JVM classpath。
- recovery1 (`t-20260910203913-cfu7h`): 设 ANSERINI_CLASSPATH env + `import pyserini.pyclass`，仍失败（同样 NoClassDefFoundError，因 pyclass try/except 吞了 add_options 错误）。
- recovery2 (`t-20260910204910-aarzv`): 改为手动 `jnius_config.add_classpath(fatjar)` 并打印 `JNIUS_CLASSPATH_BEFORE_JVM` 验证。preflight 通过（classpath 含 fatjar, num_docs=2115106）。但训练在 `resolve_queries` 失败：`FileNotFoundError: SEC RL queries not found at /data/ppnm/harness-1-rl-data.tar.gz`。根因：未传 --rl-data。
- recovery3 (`t-20260910210702-mdwbl`): 加 `--rl-data` + `--sec-corpus-root`。过了 resolve_queries，但失败在 `open_train_retrieval`→`PyseriniBackend`：训练进程的 PyseriniBackend 线程没拿到 classpath（preflight 设的 classpath 只在该进程生效，run_train.py 新进程没有）。
- recovery4 (`t-20260910212907-vbnbz`): 加 sitecustomize.py 自动注册 classpath，但 sitecustomize 调 `jnius_config.get_classpath()` 触发 `import jnius`→JVM 早启动，崩在解释器启动。
- recovery5 (`t-20260910214054-63fbr`): sitecustomize 改为只调 `add_classpath`（不调 get_classpath），preflight + Lucene JVM + open_train_retrieval 全过。但失败在 `open_upstream_train_session`：`No module named 'chz'`。
- recovery6 (`t-20260910220442-n3kyv`): 装 chz+tinker，过了 chz。但失败在 `No module named 'json_repair'`（harness/agent.py 顶层 import）。
- recovery7 (`t-20260910222244-2l5sa`): 装 json-repair + preflight 加完整 upstream import 链预检（UPSTREAM_IMPORT_CHAIN_OK）。过了 import 链，但失败在 `local_bm25 corpus missing probe documents: ['59931','69324','44797']`。根因：`validate_corpus_against_index` 默认 probe docid 是 BC+ 的纯数字 id，但 SEC corpus docid 是 `0001140361-25-047061_0` 格式，probe 域不匹配。
- recovery8 (`t-20260910223920-j4xxy`): 改用 `--train-env local_legacy`，绕过 upstream SlidingWindowSearchEnv 的 BC+ probe（走 TRIM local_search_env + SEC sec_pyserini searcher）。**成功**：vLLM tp=8 启动，step=0 rollout chunk 1/108 产出。

## 关键修复产物（持久化到 JuiceFS，后续 recovery 复用）
1. **sitecustomize.py**: `Capability_Evolution/TRIM/sitecustomize_dir/sitecustomize.py` — 解释器启动时自动 `jnius_config.add_classpath(anserini fatjar)`，只调 add_classpath（不调 get_classpath/vm_running，避免早启动 JVM）。PYTHONPATH 首位加 sitecustomize_dir。
2. **SEC corpus jsonl**: `Capability_Evolution/TRIM/data/harness-1-sec-corpus/sec_corpus_chunks.jsonl` — 从 43 parquet shards 生成，2115106 行（chunk_id→id, document_text→text），供 local_bm25 --corpus-path。
3. **tiktoken bundle**: `/mnt/songzijun/tiktoken-bundle/`（share/tiktoken/o200k_base.tiktoken 等）。
4. **额外依赖**: chz>=0.4.0, tinker==0.28.1, json-repair>=0.55.0, pyarrow（随 pip install 装）。

## 关键经验
- pyserini 2.0.0 的 anserini fatjar 在 `pyserini/resources/jars/anserini-*-fatjar.jar`。`jnius_config.add_classpath` 必须在 JVM 启动前调用；pyserini.pyclass 的 configure_classpath 被 bare try/except 吞错，不可靠，必须自己手动 add_classpath。
- sitecustomize 不能调 `jnius_config.get_classpath()` 或 `vm_running()`——新版 pyjnius 里这会触发 `import jnius`→JVM 早启动。只调 `add_classpath`（纯列表操作）。
- harness-1 的 SlidingWindowSearchEnv import 链需 chz + tinker（tinker_cookbook.rl.types 顶层 import）+ json_repair（harness.agent 顶层 import）。baseten_performance_client/pysqlite3/tinker 在 harness.config 是 try-except 可选。
- SEC 训练的 retrieval：`open_train_retrieval` 用 SEC corpus 的 sec_pyserini（--sec-corpus-root/indexes/bm25）；`open_upstream_train_session`(train_env=upstream) 用 --retrieval-backend local_bm25 + BC+ probe，probe docid 域与 SEC 不匹配。用 `--train-env local_legacy` 绕过，走 TRIM local_search_env + SEC searcher。
- SEC corpus 的 Lucene index docid 与 BC+ 不同（SEC: `0001140361-25-047061_0`; BC+: 纯数字）。validate_corpus_against_index 默认 probe 是 BC+ docid，不适用于 SEC。
- 每次恢复用全新 job_name + --out 目录，保留 failed attempt 日志不覆盖不混入。
- 队列 2334 有时排队（pending=1），即使总空闲 GPU 够，因 priority fifo/fragment/8-GPU gang 调度；等待几分钟通常能调度。

## 监控更新 (2026-09-10 23:19)
- state=running, runTime=2507s (~42min), 无报错/OOM
- step=0 on-policy rollout, chunk 8/108 已进入 vLLM 生成
- 单 chunk 节奏(稳态 ~165s): c1→2=356s(预热), c2→3=268s, c3→4=215s, c4→5=201s, c5→6=169s, c6→7=168s, c7→8=158s; prep 已降至 5-6s
- 预计 rollout 阶段完成: 剩余 100 chunks × ~165s ≈ 4.6h → 约 2026-09-11 03:50；之后还有 step1+(teacher/OPD) 阶段待确认
- 每半小时(:17/:47)自动监控中

## 监控更新 (2026-09-10 23:49)
- state=running, runTime=4308s (~72min), 无报错/OOM
- step=0 on-policy rollout, chunk 21/108 进入 vLLM 生成
- 稳态单 chunk 收敛到 ~130-142s (c16→17=133s, c17→18=129s, c18→19=150s, c19→20=131s, c20→21=142s); prep 降至 3-4s
- 从 c9(23:21)→c21(23:49): 12 chunks/28.5min → ~142s/chunk
- 预计 rollout 阶段完成: 剩余 87 chunks × ~142s ≈ 3.4h → 约 2026-09-11 03:10；之后 step1+(teacher/OPD) 阶段待确认

## 监控更新 (2026-09-11 00:21)
- state=running, runTime=6110s (~102min), 无报错/OOM
- step=0 on-policy rollout, chunk 35/108 进入 vLLM 生成
- 稳态单 chunk ~131-143s (c30→31=131s, c31→32=143s, c32→33=137s, c33→34=137s, c34→35=131s); prep 2.5-3.3s
- c21(23:49)→c35(00:21): 14 chunks/32min → ~137s/chunk
- 预计 rollout 阶段完成: 剩余 73 chunks × ~137s ≈ 2.8h → 约 2026-09-11 03:08（与上次估算一致，稳态可靠）

## 监控更新 (2026-09-11 00:50)
- state=running, runTime=7907s (~132min), 无报错/OOM
- step=0 on-policy rollout, chunk 48/108 进入 vLLM 生成（接近半程）
- 稳态单 chunk ~125-149s (c44→45=125s, c45→46=149s, c46→47=137s, c47→48=134s); prep 2.3-3.9s
- c35(00:21)→c48(00:50): 13 chunks/29min → ~134s/chunk
- 预计 rollout 阶段完成: 剩余 60 chunks × ~134s ≈ 2.2h → 约 2026-09-11 03:02（与历次 03:08/03:10 估算一致，稳态可靠）

## 监控更新 (2026-09-11 01:20)
- state=running, runTime=9707s (~162min), 无报错/OOM
- step=0 on-policy rollout, chunk 61/108 进入 vLLM 生成（过半）
- 稳态单 chunk ~131-139s (c57→58=131s, c58→59=137s, c59→60=137s, c60→61=139s); prep 2.5-3.2s
- c48(00:50)→c61(01:20): 13 chunks/30min → ~138s/chunk
- 预计 rollout 阶段完成: 剩余 47 chunks × ~138s ≈ 1.8h → 约 2026-09-11 03:08（与历次 03:02/03:10 一致，稳态可靠）

## 监控更新 (2026-09-11 01:50)
- state=running, runTime=11498s (~192min), 无报错/OOM
- step=0 on-policy rollout, chunk 74/108 进入 vLLM 生成
- 稳态单 chunk ~135-146s (c70→71=135s, c71→72=146s, c72→73=146s, c73→74=140s); prep 2.4-3.0s
- c61(01:20)→c74(01:50): 13 chunks/30.5min → ~141s/chunk
- 预计 rollout 阶段完成: 剩余 34 chunks × ~141s ≈ 1.3h → 约 2026-09-11 03:10（与历次 03:02/03:08 一致，稳态可靠）

## 监控更新 (2026-09-11 02:20)
- state=running, runTime=13296s (~222min), 无报错/OOM
- step=0 on-policy rollout, chunk 87/108 进入 vLLM 生成
- 稳态单 chunk ~130-139s (c83→84=139s, c84→85=138s, c85→86=130s, c86→87=137s); prep 2.3-2.8s
- c74(01:50)→c87(02:20): 13 chunks/30.5min → ~141s/chunk
- 预计 rollout 阶段完成: 剩余 21 chunks × ~141s ≈ 49min → 约 2026-09-11 03:10（与历次高度一致，稳态可靠）

## 监控更新 (2026-09-11 02:49)
- state=running, runTime=15099s (~252min), 无报错/OOM
- step=0 on-policy rollout, chunk 99/108 进入 vLLM 生成（接近 rollout 阶段尾声）
- 稳态单 chunk ~137-160s (c95→96=151s, c96→97=147s, c97→98=137s, c98→99=160s); prep 1.8-2.9s
- c87(02:20)→c99(02:49): 12 chunks/29min → ~145s/chunk
- 预计 rollout 阶段完成: 剩余 9 chunks × ~145s ≈ 22min → 约 2026-09-11 03:12（与历次 03:10 高度一致）
- 下次监控留意是否进入 step1（teacher projection/OPD）阶段（cell=/step=1/teacher/opd 标志）

## 监控更新 (2026-09-11 03:22) — 重大进展：进入优化阶段
- state=running, runTime=16901s (~282min, ~4.7h), 无报错/OOM
- **rollout 108/108 完成**（step=0 rollout: 22:49→03:19, ~4.6h，与历次 03:10 估算吻合）
- **step=1 优化完成** (03:22:07 optim_step_done, update_type=rl_opd_joint):
  - phase=cispo: 1536/1536 done (73s, student_fwd 32s)
  - phase=opd_gap: 1536/1536 done (99s, student_fwd 30s, teacher_fwd 28s)
  - phase=sampled_gap: 1536/1536 done (100s)
  - hf_optim_step: 0.08s; optim_step 总 elapsed 476s (~8min)
  - n_rl_datums=1536, n_opd_datums=1536, n_rl_tokens=176524, n_opd_tokens=265226
- **train_steps=8**（CLI 默认，非 smoke），每 step = 1 轮 rollout(108 chunks ~4.6h) + 优化(~8min)
- **总时长重新估算**: 8 steps × ~4.75h ≈ 38h；已完成 1 step(~4.7h)，剩余 7 steps ≈ 33h
- **预计全部完成: 约 2026-09-12 13:00（明日下午）** —— 比早期单步估算大幅延长，因需 8 轮 on-policy rollout
- 下次监控留意 step=2 rollout 是否启动（vLLM start / chunk 1/108 step=1 policy=v1）

## 监控更新 (2026-09-11 03:23) — step=1 rollout 启动（vLLM 复用）
- state=running, runTime=18711s (~312min, ~5.2h), 无报错/OOM
- step=0: rollout 22:49→03:19 (~4.6h, 含 vLLM 首次启动预热) → optim 03:19→03:22 (~3min, optim_step_done step=1)
- step=1 (policy=v1) rollout 已启动 03:22:55, chunk 1/108 prep=0.0s（vLLM 复用，无需重启，比 step=0 快）
- 编号说明: rollout step=N 对应 policy vN；optim_step_done step=N 是第 N 个优化；共 8 个 optim step（v0..v7）
- 单 step 节奏修正: step=0 含 vLLM 预热 ~4.6h；后续 step vLLM 复用，rollout ~4.2h + optim ~3min ≈ 4.25h/step
- **ETA 修正**: 剩余 7 steps × ~4.25h ≈ 29.8h → 预计全部完成 约 2026-09-12 09:00（比上次 13:00 略提前，vLLM 复用省预热）

## 监控更新 (2026-09-11 03:42)
- state=running, runTime=20535s (~5.7h), pod 健康，无 OOM/异常/failed
- step=0 全流程完成（rollout 4.6h + optim 3min）
- step=1 (policy=v1) rollout chunk 1/108 生成中（03:23 启动，vLLM 复用 prep=0s）
- 日志缓冲未刷新（chunk1 vLLM 生成中），但 runTime 持续增长（20508→20535）确认进程健康；step=0 chunk1 也曾 6min 无 chunk2
- ETA 维持: 剩余 7 steps × ~4.25h ≈ 29.8h → 约 2026-09-12 09:00

## 监控更新 (2026-09-11 04:11) — step=1 chunk1 生成偏慢（疑似模型行为变化）
- state=running, runTime=22410s (~6.2h), 进程确认活着(40s 内 runTime +41s)
- step=1 (policy=v1) rollout chunk 1/108 生成中，已 50+min 无 chunk 2（step=0 时 chunk1→2 仅 6min）
- 无 error/OOM/Traceback，pod 健康
- 分析: policy=v1 是训练后模型，可能生成更长搜索轨迹(MAX_TURNS=40, 更多轮检索)导致单 episode 耗时大增；runTime 持续涨确认在算非死锁
- ETA 暂维持 2026-09-12 09:00；若下次监控 step=1 仍无 chunk2 推进，重新评估是否卡住需介入

## 监控更新 (2026-09-11 04:39) — 关键发现：step=1+ 用 HF generate（非 vLLM），极慢
- state=running, runTime=24188s (~6.7h), 进程活着(60s 内 runTime +61s, CPU 忙)
- step=1 (policy=v1) rollout chunk 1/108 仍无 chunk2（76min 无进展，step=0 仅 6min）
- **根因定位（非挂起）**: four_cell_runtime.py:1916 `if vllm_on and not (lora_path and name != "theta0")`
  - step=0 用 theta0 adapter → name=="theta0" → 走 vLLM（快，批量生成）
  - step=1+ 用训练后 adapter（非 theta0）→ 走 HFGenerateClient（第 1922 行），因"此 vLLM release 无法加载 Harness-1 attention LoRA targets"
  - HF generate 单卡逐 episode 生成，比 vLLM 批量慢 1-2 数量级
  - 03:22:59 的 "Loading checkpoint shards" 是 HF 加载 v1 adapter 模型，非 vLLM
- **影响**: HF 模式下每 chunk 可能 1-2h，108 chunks × 7 剩余 steps(step1-7) 极其漫长（可能数百-上千小时），不现实
- 输出目录: outputs/...recovery8/ 已有 adapters/(theta0)、LAUNCH.json、RUN_MANIFEST.json、RUNTIME_EFFECT_AUDIT.json
- **需用户决策**: 是否继续等待 HF 慢生成，还是调整配置（如减少 train_steps / 改 train_method / 接受更少 step）

## 监控更新 (2026-09-11 06:11) — chunk2 推进，实测 HF 单 chunk ~168min
- state=running, runTime=31372s (~8.7h)
- **step=1 chunk 2/108 出现 (06:11:45)**: chunk1 03:23 启动 → chunk2 06:11，单 chunk ~168min (2.8h)
  - 对比 step=0 vLLM 模式 ~140s/chunk，HF 慢 ~72 倍
- 用户决策: 继续等待观察（不干预）
- **实测 ETA（HF 慢生成）**: 每 step 108 chunks × 168min ≈ 302h/step
  - 剩余: step1 余 106 chunks + step2-7 各 108 = 754 chunks × 168min ≈ 2111h ≈ 88 天（完全不可接受）
- 继续每半小时监控，确认节奏稳定；若后续仍按 ~168min/chunk 推进，需再次与用户确认是否调整

## 监控更新 (2026-09-11 06:41)
- state=running, runTime=31419s (~8.7h), 进程健康, 无 error/OOM
- step=1 (policy=v1) rollout chunk 2/108 生成中 (06:11 启动, HF 模式)
- 节奏稳定: chunk1→2=168min, chunk2 预计 09:00 左右出 chunk3
- ETA 维持实测: 剩余 754 chunks × 168min ≈ 88 天（用户已选继续观察，不干预）

## 监控更新 (2026-09-11 07:11)
- state=running, runTime=33103s (~9.2h), 进程健康, 无 error/OOM
- step=1 chunk 2/108 生成中 (06:11 启动, 已 1h, HF 模式)
- 节奏稳定, chunk3 预计 ~09:00 出现 (符合 168min/chunk)
- ETA 维持: 剩余 754 chunks × 168min ≈ 88 天（用户继续观察中）

## 监控更新 (2026-09-11 07:41)
- state=running, runTime=34900s (~9.7h), 进程健康, 无 error/OOM
- step=1 chunk 2/108 生成中 (06:11 启动, 已 1.5h, HF 模式)
- 节奏稳定, chunk3 预计 ~09:00 (符合 168min/chunk)
- ETA 维持: 剩余 754 chunks × 168min ≈ 88 天（用户继续观察中）
