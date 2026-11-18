# HarnessOPD 双域 Pilot 现状与问题汇总

**日期**：2026-11-18
**分支**：rsi-implementation @ afea7ac
**面向**：用户审阅。文中所有失败均已在代码或流程上修复，最末节给出当前状态与剩余风险。

---

## 1. 一句话现状

**准备阶段全部完成**（ALFWorld / WebShop 双域的 confirm + probe + readiness + 训练门验证 G1-G5 全绿），**4node32gpu pilot 训练已提交并多轮重试**——最后一轮（ALFWorld r16=xzibu / WebShop r23=shzji）采用"方案 B"（进程内 episode + elastic restart + checkpoint 恢复），均已进入排队/运行。**尚无一个 pilot 跑完 50 step**。

---

## 2. 已完成且验证过的准备链（双域一致）

| 链路步骤 | ALFWorld | WebShop | 证据 |
|---|---|---|---|
| fresh confirm（门 1） | ✅ +45.8pp strict（11W/0L/13T，24 新 group 零回退） | ✅ reward +27.4pp / strict +12.5pp（12W/1L/11T；3W/0L/21T） | confirm_h_r0.json + verdict |
| probe-opd（门 2/3） | ✅ admitted（143/143 fork、125 native anchors、125/125 legal、grad 52.5、覆盖 87.4%） | ✅ admitted（34/34 fork、20/20 legal、grad 8.5、覆盖 58.8%） | probe_report.json |
| teacher readiness | ✅ READY_FOR_PILOT（零 problems） | ✅ READY_FOR_PILOT（零 problems） | teacher_readiness.json（docs 副本入库） |
| 训练门验证 G1-G5 | ✅ r6（LoRA 真变化/重载 parity/tokenizer/digests/190 OPD rows） | ✅ r3（80 OPD + 526 RL rows，同五项全过） | train_verify_verdict.json |

AppWorld 教师路线按 guide §4.5 #6 诚实结案：`CANDIDATES_REPEATEDLY_INVALID → NO_TEACHER`（9B proposer 在证据契约下无法产出合规候选），不阻塞双域。

---

## 3. Pilot 失败史与修复链（每轮死点均在更深层，全部已修）

| # | Job | 死点 | 根因 | 修复 |
|---|---|---|---|---|
| r1 | pmhlp | NCCL watchdog 600s | rank 间 episode 时长方差 > 600s | NCCL timeout → 240min；eps/rank 2→1 |
| r2 | 8qsy3 | 单 rank 卡死 4h → 31 rank 等 240min watchdog | **真单 rank 卡死**（TextWorld env / generate C 层，零日志） | 见 §4 方案 B |
| r3 | kp6dj | pip 900s | 旧 entry | pip limit 提高 |
| r4/r8 | vip7p/vzhfp | keepalive 在 torch 安装前 import torch | 插入位置错误 | 移至 torch import 后 |
| r5 | q8jdc | bash 嵌套 heredoc 语法错 | patch 打成嵌套 heredoc | 改 inline python + 提交前 ast 验证 |
| r6 | tpoie | 复现 r2 卡死签名 | 同 r2 | 主动停止，触发方案探索 |
| r7 | mqg3o | fork 子进程全死 | **fork 破坏 CUDA**（父进程持 CUDA） | 改 spawn |
| WebShop r12 | janah | NCCL 'watchdog got stuck 480s'（监控线程饿死） | **32 路并发 spawn 各加载 19G 模型**内存风暴 | spawn 子进程按节点文件锁串行 |
| WebShop r14 | cnuiq | 'pip torch 2400s 超时' 且重试从未生效 | **subprocess.run 超时抛异常而非返回码**——重试循环从未跑过 | pip() 捕获 TimeoutExpired → 返回 124 |
| WebShop r15 | xzlex | worker 跑旧 entry（'No module named bs4'） | **imageCommand 固化时序陷阱**：提交(08:28)早于加固推送(10:26) | 全部残留 limit=900→2400；流程教训：先 commit 后 submit |
| ALFWorld r9 | bkrd9 | 'DistStoreError 901s 1/4 clients' | **gang 启动偏差**：master 先完成 deps 进 torchrun，3 冷节点还在 pip | 4 节点 JuiceFS barrier（deps+copy 后写 ready 标记，4 齐才 torchrun） |
| ALFWorld r16 | xny4h | '3/4 clients joined'（尽管有 barrier） | **barrier 陈旧标记**：目录按 entry 名，4 个跨 run 旧标记被计数 | per-job 目录（key=MASTER_ADDR）+ 启动清空 |
| ALFWorld r17? | szlwn | 'barrier: 1/4' 卡 55min（4 pod 全活） | **rm -rf 竞态**：每 pod 启动删整目录，晚启动 pod 删掉早启动 pod 的标记 | 只清本节点 hostname 标记，绝不删目录 |
| r15 | foqm1 | 训练 backward `cuModuleLoadData CUDA_ERROR_INVALID_IMAGE` | **tilelang JIT 与 gm402 不兼容**；triton 3.8.0 已满足守卫 → tilelang 不必要 | 卸载 tilelang（经 2 轮修正卸载调用写法） |
| r19/r22 | ypw5/i7zvf | spawn 子进程在 cuda:0 加载模型 OOM（连续 8 episode FAILED） | **GPU0 双份 19G 模型**（父 rank 训练 + spawn rollout） | 见 §4 方案 B 最终形态 |

---

## 4. 当前架构（方案 B）与理由

**隔离方案的系统性排除**：
- fork 隔离：fork 破坏 CUDA（mqg3o 实证）
- spawn-CPU 隔离：Triton kernel 无法访问 CPU tensor（envpp 实证 'Pointer argument cannot be accessed from Triton'）
- spawn-GPU 隔离：GPU0 双份 19G 模型 OOM（ypw5 实证连续 8 episode FAILED）

**最终架构**：episode 回到**父进程内**跑（r2 已验证能真实推进 6.5h、SeqNum=1460、数十 step 完成）；单 rank 卡死风险改由 **torchrun elastic restart** 兜底：
- entry 加 `--max-restarts=5`
- `run_train` 新增 **resume 逻辑**：restart 后从 run_dir 最新 checkpoint 恢复 global_step/adapter/optimizer（此前 restart 会静默从 step 0 重来并覆盖 step 目录）
- `checkpoint_every=5` → 每次重启最多丢 5 步

---

## 5. fla/triton 层修复（独立于上面的调度层问题）

| 问题 | 修复 | 验证 |
|---|---|---|
| fla Hopper 守卫（triton 3.4-3.7.1 禁用 gated kernel） | triton==3.8.0 pip pin | pip 确认装上 |
| tilelang JIT 产物 INVALID_IMAGE（gm402） | 卸载 tilelang（经 2 轮修正调用写法：pip('pip uninstall…') 是错的，直接 subprocess -m pip uninstall） | 'tilelang uninstall rc: 0' |
| **fla TRITON_ABOVE_3_7_1 常量在 rank 进程为 False**（import 时计算，launcher override 不可见） | **就地 patch 节点 fla/utils/env.py**（TRITON_ABOVE_3_7_1=True，fla #640 只禁 3.4-3.7.0） | 'env.py patched' + fresh subprocess 'verify: 3.8.0 True' |

---

## 6. 当前状态（2026-11-18 10:25）

| Pilot | Job | 状态 |
|---|---|---|
| ALFWorld r16 | t-20261118101302-xzibu | **running**（runTime ~4.6min） |
| WebShop r23 | t-20261118101319-shzji | enqueued（pri8） |

上一轮（spawn 版 r19/r22）已停；本轮是 plan B 完整形态首次提交。

---

## 7. 剩余风险与判定点

1. **单 rank 卡死若在 <5 步内频繁发生**：5 次 restart 耗尽后 job 失败。届时选项：(a) 提高 max-restarts；(b) episode 级 skip（把卡死 episode 标 reward 0）——但进程内无法安全中断 C 层 generate，需回到隔离方案，而隔离方案已系统性排除。**若 restart 5 次耗尽，将上报并提议两难决断**。
2. **r2 的 6.5h 运行说明卡死是低频事件**（~2.5h/次）：50 步 × 1 eps/rank，期望卡死次数 ≈ 2-4 次，5 次 restart 预算应足够。
3. **NCCL 240min timeout** 在 elastic restart 下仍生效——重启后 barrier/rendezvous 重新进行，已由 barrier 保证 4 节点齐。
4. **WebShop r23 依赖链冷节点 pip 时长**：barrier 2h 上限内应完成（r14/xzlex 实测 deps ~35min）。

## 8. 判定标准（不变）

- SUCCEED + train_result.json → 报 global_steps/reward_trace/adapter_digest/export → **E4 达成**
- FAILED → history_logs 诊断 → pri8 重提（无需确认）
- restart 5 次耗尽 → 上报两难决断
