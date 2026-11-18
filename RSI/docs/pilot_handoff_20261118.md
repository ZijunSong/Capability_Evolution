# HarnessOPD Pilot 任务交接文档（转接给下一个 agent）

**日期**：2026-11-18 11:20
**分支**：rsi-implementation @ afea7ac（所有修复已推 GitHub）
**详版状态文档**：`RSI/docs/pilot_status_and_issues_20261118.md`

---

## 1. 当前排队的两个实验 ID（均 pri8，4node32gpu，50 step，LoRA r32）

| 域 | Job ID | Job 名 | 状态 |
|---|---|---|---|
| **ALFWorld** | **t-20261118111750-hnjyj** | rsi-pilot-harnessopd-alfworld-9b-4n32g-20261115-r1 | enqueued（priority 8 已确认） |
| **WebShop** | **t-20261118111753-u6ghy** | rsi-pilot-harnessopd-webshop-9b-4n32g-20261115-r1 | enqueued（priority 8 已确认） |

两者 entry 已含全部 14 项修复（见 §3）。预计 gang 排队数小时~1 天（4node32gpu）。注意：这两个 job 都可能因陈旧 enqueued job 占配额而延迟——如排队过久，用 `cml custom_train list` 查同名旧 job 并 stop 释放配额。

## 2. 教师确认数据（E1-E3 已完成，双域 READY_FOR_PILOT）

| 域 | 教师候选 | confirm | probe | readiness |
|---|---|---|---|---|
| ALFWorld | 622625de（bank c31e0e88，alfworld_checker + completion_gate） | **+45.8pp strict**（11W/0L/13T） | admitted：143/143 fork、125/125 legal、grad 52.5 | READY_FOR_PILOT |
| WebShop | bd397b10（bank a691cf12，5 组件） | **+27.4pp reward / +12.5pp strict**（零 strict 回退） | admitted：34/34 fork、20/20 legal、grad 8.5 | READY_FOR_PILOT |
| AppWorld | 无候选 | — | — | **NO_TEACHER 诚实结案**（9B proposer 出不了合规候选），不阻塞双域 |

准备链验收物路径：`runs/{alfworld,webshop}_confirm_r1/`（confirm/verdict/teacher_readiness）、`runs/{domain}_probe_r1/opd/probe_report.json`。

## 3. Entry 已固化的 14 项修复（新 agent 不要重复排查）

1. **4 节点 JuiceFS barrier**（deps+copy 后写 ready 标记，4 齐才 torchrun）——bkrd9 gang 偏差 901s 修复；**只清本节点 hostname 标记**（szlwn rm-rf 竞态修复）；**per-job 子目录 key=MASTER_ADDR**（xny4h 陈旧标记修复）
2. **GPU keepalive**：preflight + barrier 循环内每 10s 触 cuda:0（平台 30min-zero-GPU 强杀）
3. **pip() TimeoutExpired→returncode 124**（重试循环曾从未生效）+ **全部 pip limit=2400**
4. **spawn 子进程按节点串行锁**（RSI_SPAWN_EPISODE_LOCK，janah 32 路内存风暴修复）→ **已废弃**，见 §4
5. **tilelang 卸载**（直接 subprocess -m pip uninstall；foqm1 JIT INVALID_IMAGE 修复）+ **triton==3.8.0 pin** + **fla/utils/env.py 就地 patch**（TRITON_ABOVE_3_7_1=True，r17 守卫复现修复）
6. **磁盘守卫**：model pre-copy 前 df 检查（df6od /tmp 14G 静默死）
7. **NCCL timeout 240min**（pmhlp 600s episode 方差超时）
8. **PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True**（riilf NCCL 内部分配碎片化 OOM）
9. **AppWorld v3 契约**（appworld_v3.md + typed Finish + APPWORLD_ROOT 重定向节点本地盘——五连 hang 根因是 JuiceFS 元数据停顿）
10. **ALFWorld 依赖链**（fast-downward-textworld 等，e1 已验证序列）

## 4. 关键架构演进（新 agent 必读，避免重走弯路）

**episode 隔离方案已系统性排除**（每个都有实证）：
- fork 隔离：破坏 CUDA（mqg3o）
- spawn-CPU：Triton kernel 无法访问 CPU tensor（envpp）
- spawn-GPU：GPU0 双份 19G 模型 OOM（ypw5 连续 8 episode FAILED）

**最终架构 = 方案 B**：episode 回到父进程内跑（r2 实证能真实推进 6.5h、SeqNum=1460、数十 step），单 rank 卡死风险由 **torchrun --max-restarts=5 elastic restart** 兜底，且 `run_train` 新增 **resume 逻辑**（restart 后从 run_dir 最新 checkpoint 恢复 global_step/adapter/optimizer，checkpoint_every=5 → 最多丢 5 步）。collector.py 已恢复 r2 模式（spawn 包装已删）。

## 5. 监控判据（接手后怎么做）

```bash
cml custom_train describe t-20261118111750-hnjyj   # ALFWorld
cml custom_train describe t-20261118111753-u6ghy   # WebShop
# 日志：cml custom_train logs <job> --pod_name tj-3050749-<jobid>-master-0
```

**健康链**：deps OK → GPU keepalive → barrier 'all 4 nodes ready'（计数应从 0/4 递增）→ prepare frozen → torchrun 32 rank → episode rollout（父进程内，10-60min/episode）→ value=/step 行。

**若 SUCCEED**：读 `runs/{domain}_confirm_r1/train/train_result.json`（注意 ALFWorld 的 run_dir 是 `runs/alfworld_confirm_r1/train/`）报 global_steps/reward_trace/adapter_digest/export → **E4 达成**，下一步 E5（多 seed 冻结评测）。

**若 FAILED**：拉 history_logs（`cml custom_train history_logs <job> --pod_name <pod> -r`）诊断。已知模式速查：pip TIMEOUT=镜像慢（重试应生效）；NCCL watchdog=看是哪类（straggle vs stuck 480s）；'episode FAILED' spawn 类=不应再出现（已回退）。**pri8 重提无需确认；每轮 patch 后必须 bash -n + ast 验证 + describe 确认 priority=8**。

**风险预判**：r2 的 6.5h 运行说明单 rank 卡死约每 2.5h 一次——50 步期望卡死 2-4 次，5 次 restart 预算足够；若 restart 耗尽仍卡，上报两难决断（提高 restarts 或 episode 级 skip 的两难已在 §7 记录）。

## 6. 任务目标锚点

用户 goal：**"准备阶段都完成后，就分别以优先级为8,4节点32卡提交HarnessOPD训练，持续监控直至顺利完成这一步"**——提交✅，剩"顺利完成"（SUCCEED + train_result.json）。E4 完成后进入 E5（冻结方案与预算、多训练 seed、封存最终评测）——那需要用户新决策，不在本 goal 内。

**AppWorld 后续**（不阻塞）：如重启教师路线走 assisted 上界对照或工具 SFT 分支（guide §4.5 末段）。
