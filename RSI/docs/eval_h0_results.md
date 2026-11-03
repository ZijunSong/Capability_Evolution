# RSI H0 基线评测结果（formal）

> ⚠️ **2026-11-03 测量审计更正（`measurement_status: LEGACY_REQUIRES_REAUDIT`）**
> 外部审查（`Capability_Evolution_RSI_代码改进与实验路线_20261103.md` §2）识别出本文件三处
> 测量风险，本文全部数字降级为 historical development snapshot，`usable_for_method_claim: false`：
> 1. **clock_freeze**：旧 runtime 用 `perf_counter` 计时，AppWorld 的 freezegun 全局 patch 会把
>    墙钟测成任务模拟日期——"100% budget_wall、n_decisions 全 0"更可能是**模型从未被调用**，
>    而非本文所称"记录层缺口"。`runtime/clock.py` 的解释与本文件冲突，以 clock.py 为准。
> 2. **metric_semantics**：旧 `value = pass_percentage/100` 是**连续测试通过比例**，不是官方
>    TGC（官方 TGC = `tracker.success` 全通过均值）。"TGC 23.51%"应读作 partial 均值 23.51%，
>    strict success 为 0。
> 3. **artifact_scoring_path**：旧 scorer 对同名 experiment 重新构造 AppWorld 并 initialize
>    （close_all + preamble + save_state 重跑）——318 对任务分数逐位相同的头号嫌疑是
>    **评分读到了重置后的状态**而非"H0/Hk 轨迹同分"。已修（commit e918f67，官方
>    `evaluate_task` 离线评分），但旧数字不能因此自动恢复有效。
> 重跑口径见 v2 config（`configs/appworld_self_evolve_v2_9b.yaml`）与 E0/E1 实验计划。

## Qwen3.5-9B · 完整 AppWorld · minimal harness · official TGC（2026-11-02）

- **job**：`t-20261102144730-dhlol`（rsi-eval-h0-appworld-qwen35-9b-8gpu-20261102-r2），succeed，runTime≈106min，8×M402 单节点
- **产物**：`runs/eval_h0_appworld_qwen35_9b_bench_r2/eval/eval_<split>_minimal_1beba6d6_17.json`（7 文件）
- **provenance**：seed=17，harness=minimal（空 bank），bank_digest=1beba6d6…，policy_digest=63eea503…，config_digest=35ba0001…，synthetic=False
- **口径**：official TGC `task_goal_completion` 逐 task 评分；4 个 infra `task_error` 从 formal 均值剔除（记 0 仅供参考）
- **前史**：首跑 `t-20261102103014-wtmif` 静默挂死已 kill（fla 核验证错导入名 + 8worker OOM + Pool.map 等死，修复见 [[rsi-eval-h0-wtmif-hang-dhlol-r2-20261102]]）

> 📊 跨实验总对照表（9B/27B × 4 实验组 × 4 指标 × 4 bench）：`results_table.csv`（同目录，后续结果持续并入）

### 核心结果（TGC 均值，valid-only）

| split | n_total | n_valid | n_err | **TGC valid** | TGC all(含err=0) | max | wall_s/task |
|---|---|---|---|---|---|---|---|
| work_train | 43 | 41 | 2 | **28.13%** | 26.82% | 0.50 | 31.6 |
| harness_select | 44 | 42 | 2 | **33.10%** | 31.59% | 0.50 | 36.3 |
| weight_dev | 44 | 44 | 0 | **15.32%** | 15.32% | 0.50 | 32.1 |
| confirm_h | 39 | 39 | 0 | **25.47%** | 25.47% | 0.50 | 33.8 |
| confirm_w | 36 | 36 | 0 | **33.33%** | 33.33% | 0.50 | 34.1 |
| final_test | 53 | 53 | 0 | **23.64%** | 23.64% | 0.50 | 33.3 |
| transfer_test | 66 | 66 | 0 | **13.41%** | 13.41% | 0.40 | 31.8 |
| **ALL** | **325** | **321** | **4** | **23.51%** | 23.23% | 0.50 | 33.1 |

### 说明与异常
- **完成数审计**：325/325 全部有 per-task 记录；有效 321，infra error 4。
- **termination census**：budget_wall 321 + task_error 4。全部 valid task 以 `budget_wall`（动作/墙钟预算耗尽）终止，**0 个 finish**——minimal harness 下模型主动收敛能力弱，score 分布集中在部分目标完成度（众数 0.5，79 例；0.0，53 例；无 1.0 满分）。
- **score 语义**：值呈 1/6、1/8、1/5、1/4、1/2 分数（AppWorld TGC 按目标子句计分），评分真实非占位。
- **task_error 4 例（infra，非模型问题）**：
  - work_train `appworld:229360a_5/6`：任务 DB 目录缺失（`.../229360a_6/dbs does not exist`，数据冻结包缺口）
  - harness_select `appworld:d0b1f43_2`：FileNotFoundError（同源数据缺口）
  - harness_select `appworld:b7a9ee9_3`：`ModuleNotFoundError: generate`（模型输出触发 eval 侧执行路径问题，待查）
- **`n_decisions=0` 记录缺口**：per_task 的 `n_decisions` 字段全 0 但 wall_ms≈31-36s/任务、score 有区分度——该计数字段未接线（trace 计数未写入 per_task），不影响 TGC 评分，待修记录层。

### 分析（初判）
1. **confirm 池最高（25-33%）、transfer_test 最低（13.4%）**：分布内迁移（transfer）泛化差，符合 zero-shot H0 预期；confirm_w/harness_select 是对 harness 选型最敏感的池，后续 harness 进化收益应重点看这两池的 delta。
2. **budget_wall=100%**：动作预算（max_actions=24）是当前硬顶，harness 进化（更好的收敛/finish 信号）预期先提升 finish 率再抬 TGC。
3. weight_dev（15.3%）显著低于 work_train（28.1%）：留出分布难度差异，作后续 Student 训练的难度参照。

---

## exp_hk 对照：自进化 harness H_k vs H0 minimal（2026-11-03）

> **formal artifact**：`t-20261102210523-kafnl`（rsi-eval-hk-appworld-qwen35-9b-8gpu-20261102-r1），succeed，runTime≈5h13m，8×M402 单节点。
> **H_k**：fh46b（evolve_appworld_qwen35_9b_p1_r6，4 代 PLATEAU）champion bank `73ddc436…`，唯一组件 `ContextCompressor`（len(history)>20 时注入"聚焦最近 5 轮/勿重复执行/卡死换路或 finish"指令）；dev panel macro 0.2591→0.3273（+6.8pp）。
> **产物**：`runs/eval_hk_appworld_qwen35_9b_bench_r1/eval/`（7 文件 `eval_<role>_bank_73ddc436_17.json`）。
> **provenance**：seed=17、harness=bank、bank_digest=73ddc436…、policy_digest=63eea503…（与 H0 完全一致）、config 仅 campaign_id 不同。除 harness 外与 dhlol r2 逐字段同口径（同 dep 钉版、同 runtime budget、同 split 确定性切分）。

### 逐池 TGC 对比（valid-only）

| pool | n | valid | err | **HK TGC** | H0 TGC | Δ(pp) |
|---|---|---|---|---|---|---|
| work_train | 43 | 42 | 1 | 28.25% | 28.13% | +0.12 |
| harness_select | 44 | 40 | 4 | 32.88% | 33.10% | -0.23 |
| weight_dev | 44 | 44 | 0 | 15.32% | 15.32% | 0.00 |
| confirm_h | 39 | 39 | 0 | 25.47% | 25.47% | 0.00 |
| confirm_w | 36 | 36 | 0 | 33.33% | 33.33% | 0.00 |
| final_test | 53 | 53 | 0 | 23.64% | 23.64% | 0.00 |
| transfer_test | 66 | 66 | 0 | 13.41% | 13.41% | 0.00 |
| **ALL** | **325** | **320** | **5** | **23.46%** | **23.51%** | **-0.05** |

**paired 同任务（valid-in-both，n=318）：H0=23.34% vs HK=23.34%，d=+0.00pp。**

### 结论：dev-panel 增益未迁移

1. **TGC 与 H0 严格持平**（paired +0.00pp；含 err 记 0 口径 23.10% vs 23.23%）。Success Rate 同为 0（0/320 vs 0/321）。score 分布众数同为 0.5（78 例）。
2. **行为确实改变，分数没变**：
   - 终止策略翻转：H0 全部 budget_wall（costs.completed=0）→ HK **251/320 finish**（主动收敛），budget_wall 仅 69。
   - HK 修复了 H0 的 `n_decisions` 记录缺口（H0 全 0 → HK 正常记录，均值 13.8、max 24）；HK actor_output_tokens=1.27M（~3929/task），H0 该账本也是 0（同款记录缺口）。
   - compute 总量 20.25h vs H0 3.0h（6.75×）——更长 episode + finish 语义下完整跑满决策，但**多花的算力没有换到 TGC**。
3. **机制解读**：ContextCompressor 只在 len(history)>20（episode 末尾 1-3 个决策）注入指令；96 个触发任务 score 与 H0 全部一致（95 same + 0 diff）。即组件真正改变的是**收尾行为（提前 finish、省预算）**，不改变 goal 完成度。dev panel 的 +6.8pp 增益是 harness_select 池上的选择效应/评测噪声，未在 7 池全量复现。
4. **err 5 例 vs H0 4 例**：AppWorld 冻结数据缺口（FileNotFoundError）在不同 worker/时序下随机落点（HK 修复了 229360a_5、d0b1f43_2，新增 82e2fac_6/7、b7a9ee9_1）；`b7a9ee9_3` ModuleNotFoundError generate 两轮同现（eval 侧执行路径问题，与 harness 无关）。valid-only 口径不受影响。
5. **对后续的启示**：① harness 进化要产生 TGC 增益，候选组件需作用于**中途决策**（检索策略/工具选择/状态追踪），仅收尾压缩不够；② evolution 的 dev panel（harness_select 44 task、tolerance 0.01）区分度不足，+6.8pp 的 panel 增益可能是噪声——后续可加大 dev panel 或收紧 tolerance；③ kafnl 的记录层（n_decisions/tokens/finish census）是完整可信的，可作为后续 eval 的记录层基准。

### 审计备注
- bank preflight `BANK_OK digest match 73ddc436`；7 文件 bank_digest/harness/seed 全部核验。
- 逐 task 配对核验：318/318 valid-in-both 分值一致到浮点精度；termination 翻转但分值不变的 41/43（work_train）为同轨迹不同收尾，符合"压缩指令只改收尾"机制。
- per-task wall_ms 口径两轮不可比（H0 未含模型调用时间，HK 含；用 costs.wall_ms_total 比较才公平：3.0h vs 20.25h）。
