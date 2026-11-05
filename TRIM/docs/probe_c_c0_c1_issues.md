# Probe C 可学习性探针 C0/C1 问题记录（Qwen3.5-9B 四联盟）

> 生成于 2026-11-03。数据源：`outputs/probe-c-qwen35-9b-coalitions-8gpu_20261101_r3/runner/Qwen_Qwen3.5-9B/`（AUDIT.json / CACHE.json / SCREEN.json / PROBE_C_REPORT.md）。全历程见 memory `probe-c-9b-coalitions-job-20261101` / `probe-c-implementation-20261031`。

## 一、任务链概况

| attempt | job | 结果 |
|---|---|---|
| r1 | t-20261101145144-ics4k | failed@105s（镜像 python3 裸无 transformers，漏抄训练 entry 的运行时依赖 bootstrap） |
| r2 | t-20261101145740-p06y6 | failed@449s（audit 之后 collect 阶段中断，runner 目录只有 AUDIT/无 CACHE） |
| **r3** | **t-20261101145740-p06y6（重提，OUT `_r3`）** | **成功**：audit→collect→runner collect→cache(C0)→screen(C1) 全链完成 |

## 二、C0（pilot 16q×2split×全 turn，128 状态）问题

### 2.1 三联盟 verdict 非 READY —— 注册标签 ≠ 执行证据

| 联盟 | verdict | 漏斗关键数 | 问题定位 |
|---|---|---|---|
| **trieve** | ✅ READY | 32/32 状态可投影，6531 监督 token，全 query 覆盖 | 无 |
| **ccso** | ⚠️ NO_SUPPORT | event_emitted=95/128，**projection_direct=95 但 realizable=仅 4**（valid_over_attempt=3.1%） | sentence_compress 触发正常，但 91/95 卡在投影：**observation 级变换（压缩/去重/图化）映射不出学生的 token 级动作** |
| **esp** | ❌ NO_TRIGGER | triggered=0/128 | adaptive_rerank_instruction 是 synthetic 门控启发式，真实 H0 轨迹不触发 |
| **msc** | ❌ NO_TRIGGER | triggered=0/128 | verify_tool=`no_real_verifier`（无真实 verifier side branch），token_budget_marker 是 runtime anchor |

**根因 1：Teacher 注册缺口（AUDIT.json 记录，C0 前已知但被低估）**

| 联盟 | 成员 | 未注册 |
|---|---|---|
| esp | subtractive_curation, importance_tagging, adaptive_rerank_instruction | **2/3**（前两者） |
| ccso | sentence_compress, content_dedup, evidence_graph | **2/3**（后两者） |
| trieve | auto_populate_first_search, chunk_neighbors | **1/2**（chunk_neighbors） |
| msc | verify_tool, token_budget_marker | 0（但 verify_tool 的 branch 是 no_real_verifier 占位） |

注册缺口只解释 esp（2/3 未注册）；**ccso 的 sentence_compress 已注册且触发 95 次仍卡投影**——说明问题不止注册，还有投影通道本身。

**根因 2：三类组件的可内化性天然分层**
- trieve（auto_populate_first_search）= **动作级**组件：teacher 分支直接改变学生可选动作 → 投影 1:1 可行
- ccso = **observation 级**组件：改写的是输入观测而非动作 → 学生没有对应动作可学（DIRECT 投影无目标）
- esp/msc = **决策门控级**组件：触发依赖 synthetic 启发式/外部 verifier，真实 rollout 中根本不出场

**后果**：C1 只有 trieve 一家有监督信号可学，四联盟对比实验设计（4 组同口径）**降级为单组案例研究**。

### 2.2 修复链消耗（r1→r3 每修先本地验证再重提）
依赖 bootstrap→SEC env→语料重复≠observation 回填→pilot turn 分布→TeacherTarget.get→跨状态 cid→regenerate 未解包→NLL 评测状态→4→8 契约。

## 三、C1（trieve 4→8 步微更新，seed 11/23）问题

### 3.1 四行结果

| seed | budget | verdict | NLL 0.546→ | R_N | D_K | R_D | invalid post |
|---|---|---|---|---|---|---|---|
| 11 | 4 | PROMOTE | 0.503 | +0.044 | 0.1407 | +0.0004 | 0.73 |
| 11 | 8 | DEPRIORITIZE | 0.439 | **+0.107** | 0.1440 | -0.0029 | 0.70 |
| 23 | 4 | DEPRIORITIZE | 0.497 | +0.049 | 0.1411 | -0.0000 | 0.75 |
| 23 | 8 | DEPRIORITIZE | 0.437 | **+0.109** | 0.1428 | -0.0017 | 0.73 |

### 3.2 遇到的问题

1. **verdict 与实际口径冲突（"双改善门"机械判定）**：verdict@8 全 DEPRIORITIZE，但 D_K 改善（R_D<0）量级 0.002-0.003、低于测量噪声，且方向跨 run 翻转（r7/r8）。按任务书 §9 实为 **BORDERLINE** 而非 DEPRIORITIZE。当前阈值 `delta_dk=0.0` 对噪声不设容差，机械触发了降权。
2. **NLL 改善与 D_K 不动背离**：R_N 随步数翻倍（+0.045→+0.11，seed 复现好）→ 投影目标短程可学性强；D_K（候选分布级 routing）8 步内完全不动 → 动作偏好内化 ≠ 检索策略收敛。两个指标的因果链没有打通（NLL 降了 D_K 不降，无法回答"学会的动作是否改变了分布"）。
3. **invalid 率高位徘徊**：generated_invalid 0.70-0.75（greedy 重生成 7 成不合法），虽预算门（eps_invalid=0.02）未触发拦截，但说明生成的动作分布仍在 guard 频繁纠偏下运行，可学性信号部分被合法性掩蔽。
4. **单组件单 pair 证据面窄**：C1 只吃到 trieve 注册成员（auto_populate_first_search）的 DIRECT 监督；chunk_neighbors 未注册。即使 trieve 的结论，也只是"半个联盟"。

## 四、C2 中断（与 C0/C1 结论的衔接）

- C2 在线确认四 job（训练×3 + eval×1，2026-11-02 提交）**已全部停止**（用户要求），OUT `_r2` 保留。r1 曾踩三坑：CLI 子命令返回 Path→SystemExit 杀 entry；C2 eval 漏传 SEC searcher→64 题 recall 全 0（空环境假数据）；等待循环 find+pipefail 自杀。same_view 第 4 坑：`--train_method zero-placebo` 契约强制 `--component zero`。
- 重提命令：`PRIORITY=5 bash scripts/submit_probe_c_c2_train_16gpu_2node_20261102.sh <COND>` ×3 + `submit_probe_c_c2_eval_8gpu_20261102.sh`（entry 三坑均已修）。

## 五、遗留问题与建议顺序

1. **esp/ccso/msc 补真实 Teacher 分支**（根本性）——否则 C0 永远只有 trieve 一家 READY，四联盟对比无从谈起：
   - esp：subtractive_curation/importance_tagging 需要真实 side branch（当前 adaptive_rerank 是 synthetic 启发式）
   - ccso：关键不是注册而是**投影通道**——observation 级变换需要定义"学生视角的可学目标"（如压缩后的检索重写动作），或接受该类组件走 C2 闭环而非 C1 直投影
   - msc：verify_tool 需接真实 verifier
2. **C1 verdict 规则加噪声容差**：delta_dk 从 0.0 改为 ≥测量噪声下界（如 0.005），避免 BORDERLINE 被机械判成 DEPRIORITIZE；或引入 §9 的显式 BORDERLINE 分支。
3. **C2 闭环确认 trieve**（borderline 扩展 16 步 + G_m^8/E_m^8 在线增益），C3 用 rieve 50step(succeed) 校准 C1 预测力。
4. **与训练结果交叉验证**（2026-11-03 新证据）：ccso C0 判 NO_SUPPORT，但 50step HarnessOPD eval 里 ccso longsealqa 65.66 高于 esp 64.22——组件蒸馏通道（C0/C1 测的）与 RL reward 通道增益解耦，C0 NO_SUPPORT 不预测训练无增益。C2 的 E_m（vs rl_only）正是为拆解这个设计的，建议优先重提。

## 附：9B 四联盟 50step 训练现状（2026-11-01 核对）

| 组 | job | state | 产物 |
|---|---|---|---|
| esp | t-20261028084241-mlmro(r2) | failed@10-29 | metrics 1→50 步全 + 800 sessions |
| ccso | t-20261028084244-urnt4(r2) | failed@10-29 → **11-03 old1l 续跑 step50 SUCCEED** | step_000050/adapter 完整，12/12 formal eval 已收官 |
| rieve | t-20261028072409-9g7ul(r2) | succeed | 800 sessions 50 步完整 |
| msc | t-20261027163853-a1tye | stopped | metrics 1→50 步全 + 800 sessions |
