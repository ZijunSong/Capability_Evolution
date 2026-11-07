# RSI HarnessOPD 重构完成备忘（PR-01~PR-10）

> 日期：2026-11-07。依据 `HarnessOPD_9B_Three_Benchmarks_Route_and_Code_Guide.md`（固定基线 30580de）。
> 实现分支：`rsi-implementation`，提交 `0c6830f`。**全部 10 个 PR 已实现并通过测试。**

## 1. 测试与验证状态

| 层 | 结果 |
|---|---|
| unit（79 旧 + 52 新） | **131 passed** |
| e2e（two_round_campaign 3 用例） | **3 passed** |
| vendor integration（AppWorld 真实环境，6 用例） | **6 passed**（~4 min） |
| 真实 AppWorld H0 v3 smoke（发现调用/错误反馈/typed finish/snapshot） | **通过** |
| 真实 AppWorld fork-at-old-decision | **通过**（child 无未来变量、base 完好） |
| audit_runtime_contracts 脚本 | v1 doc → FAIL（4 问题）；v3 doc/retry config → PASS |

## 2. 每个 PR 落了什么（一句话版）

- **PR-01**：`appworld_v3.md` 只写 vendor 验证过的调用（三个发现调用 + print 规则 + 无通用 authenticate）；Finish 变成唯一提交入口，`completion_submit_failed` 可重试；python 运行时错误 → `ok=False`。新配置 `configs/appworld_retry_v1.yaml`（4 代 × 2 候选，≤8 评估，repair 1，patience 2）。
- **PR-02**：`runtime/run_identity.py`——RunContext 给 A/B 臂独立 experiment 名和 episode 目录（原子写、防覆盖）；`stable_seed` 派生五条 RNG 流；`HFPolicy.sample` 真正使用 `params.seed`（fork_rng 局部作用域）。
- **PR-03**：`evolution/outcomes.py` 统一三处 outcome 判定（strict bool 必填、partial_failure 不再进 success 案例池、EVAL_INVALID 阻断）；selection 改用本代 paired；confirm 同时报 value + strict 两个 ΔH。
- **PR-04**：`evolution/failure_windows.py` 分类失败窗口（错误步 + 前驱步 + 机制标签 + event refs）；proposer 证据契约（generic-advice 假设直接拒）；`max_candidate_repairs` 从 YAML 接线。
- **PR-05**：`opd/readiness.py` + `schemas/teacher_readiness.schema.json`——8 状态准入产物绑定 bank/policy/runtime/env 身份，任一变化即失效；`run_train` 在 `require_teacher_readiness: true` 时硬门；confirm 在 manifest 无 `confirm_h` split 时拒绝运行（不 fallback）。
- **PR-06**：`DecisionRecord.public_event_end` 记录采样时刻真实公开前缀边界；三个真实 adapter 的 `fork()` 改为按快照位置重放（`replay:<n>`）——修复前 fork 旧决策拿到的是当前状态；`collect_opd_rows` 用决策自己的 snapshot+精确前缀，legacy 记录大声 skip。
- **PR-07**：teacher 分支 parity——veto 产生私有反馈事件并计费；Finish 走 before_finish 门控；parse 错误计 action 预算。
- **PR-08**：`opd/validators/{base,alfworld,webshop,appworld}.py` 三值判定（valid/invalid/uncertain，uncertain 正式路径 skip）；ALFWorld 对当前 admissible 集匹配；WebShop 按页面类型验 Buy Now；AppWorld AST 检查会话变量/动态构造。
- **PR-09**：teacher 打分用目标前因果前缀（`pre_target_events`）——目标自身执行反馈不再进打分 prompt；`run_train` 在分支前统一构建 runtime_parts（修掉分布式分支 UnboundLocalError 被宽 except 吞成 reward-0 的坑），编程错误直接中止；零信号批次跳过 optimizer（`NO_TRAINING_SIGNAL`）；logprob 禁止零填充（raw step logits 做 log_softmax，失败大声报错）；采样分布显式钉死（top_k=0、repetition_penalty=1）；export 保留源 tokenizer；checkpoint 带 trainable 身份 + 非空 runtime/env digest。
- **PR-10**：5 个脚本（audit_runtime_contracts / export_teacher_bundle / build_teacher_readiness / audit_decision_alignment / run_9b_internalization_pilot）+ 52 个新测试。

## 3. 过程中的关键修复/发现

1. **toy_smoke.yaml 缺 `runtime.profile`**：e2e 三用例在 30580de 上就挂（此前 PR 引入的必填校验），补声明后修复——非本次重构引入。
2. **`_resolve_decision_snapshot` 首版清空 digest**：导致 toy fork 的 digest 校验全部失败、probe 全 skip（e2e `INSUFFICIENT_OPD_SUPPORT` 暴露）。内容寻址 ref（`toy_file:<digest16>`）必须透传自身 digest；`replay:<n>` ref 由 adapter fork 时重验。**教训：e2e 闭环是 OPD 链路回归的真正守门员。**
3. **ast.Exec/Eval 在 py3.12 不存在**：动态构造检测改为只查 Call 名单（eval/exec/getattr/globals/locals/__import__）。
4. **AppWorld `api_calls` 取错属性层级**：`apis.spotify.search()` 应取 root 后第一个属性（app 名），不是最深 attr。
5. **vendor 验证结论**（0.2.0.dev0，本机数据）：`show_app_apis` 确实不存在；`show_app_descriptions`/`show_api_descriptions`/`show_api_doc` 全部返回可用内容；`Execution failed. Traceback:` 前缀是真实运行时错误信号。

## 4. 还没做（诚实边界）

- **未做真实 9B 训练/评测**：`run_train` 的所有路径已修复并有 CPU 测试，但 guide §13 要求的「真实 9B 1-2 update 验证 → ALFWorld 20-50 step pilot」需要 GPU job（CloudML 提交），本会话只覆盖代码与 CPU/vendor 验证。
- **confirm_h fresh split manifest（v2）未生成**：三个 e1_*_9b.json manifest 都没有 confirm_h 角色；PR-05 的 confirm 拒绝逻辑就位，但跑 ALFWorld/WebShop 确认前需先出 v2 manifest（scripts 有 readiness builder，manifest 生成器待做或手工）。
- **AppWorld 有界重试未启动**：`appworld_retry_v1.yaml` 就绪，预算已固定（4×2/8/1/2）；按 guide §4.5 应先修完 PR-01/02/03 后重测 H0（本会话 smoke 已过），重试本身是下一个 GPU 任务。
- **PR-06 的 AppWorld session 状态恢复**：fork 用合法前缀重放（已验证变量/交互计数正确），guide §6.3 提到的非确定性重放 fail-closed 已有 digest 校验，但「Python 全量解释器状态」等价性仍未在长轨迹上做压力测试。
- **`test_appworld_completion_types.py`**（int/float/str/null 全类型保真矩阵）未单列——null 保真已测（`test_answer_type_fidelity_null`），数值/字符串提交在 smoke 里验证过，完整矩阵待补。

## 5. 下一步（按 guide 顺序）

1. **AppWorld H0 重测**（work_train + harness_select，新契约）→ 确认 API 发现/登录/状态变更/完成提交可行（guide §4.5 #1）。
2. **AppWorld 有界重试**：`appworld_retry_v1.yaml`，结束状态必须是 `BASELINE_RETAINED / NO_CONFIRMED_GAIN / ...` 之一，job success ≠ 方法成功。
3. **ALFWorld 优先**：导出 622625de 完整 bank（`export_teacher_bundle.py`）→ v2 manifest（含 confirm_h）→ 同协议 confirm → probe-opd → readiness → 20-50 step pilot。
4. **WebShop**：冻结 trace-r11 bd397b10 为待确认候选，确认后再进第二个内化环境。
5. 训练侧第一轮：真实 9B 1-2 update 验证（参数变化 + 重载 logprob 一致）后才允许 pilot。
