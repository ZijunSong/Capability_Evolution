# 汇总当前 10 个 Harness 组件现有实验结果为 CSV

## 0. 任务目标

请基于当前 SCAPE / Beyond Textual Privilege 项目中**已经存在的实验结果与 artifacts**，不要重新训练、不要重新采集、不要补跑任何实验，整理出一张统一的 10-component 结果总表，并以 CSV 格式输出。

本任务只做：

```text
READ
AUDIT
RESOLVE LATEST CANONICAL RESULT
AGGREGATE
EXPORT CSV
```

禁止：

```text
TRAIN
ROLLOUT
NEW EVAL
NEW DATA COLLECTION
NEW SEED
NEW BOOTSTRAP
NEW CLAIM
```

目标输出：

```text
outputs/current_component_results_summary/
├── COMPONENT_10_CURRENT_RESULTS.csv
├── COMPONENT_10_CURRENT_RESULTS.md
├── COMPONENT_10_RESULT_PROVENANCE.csv
├── COMPONENT_10_RESULT_GAPS.md
├── RUN_MANIFEST.json
└── SHA256SUMS
```

---

# 1. 需要汇总的 10 个组件

顺序必须固定为：

```text
1. verify_tool
2. importance_tagging
3. subtractive_curation
4. auto_populate_first_search
5. content_dedup
6. chunk_neighbors
7. evidence_graph
8. sentence_compress
9. token_budget_marker
10. adaptive_rerank_instruction
```

最终 `COMPONENT_10_CURRENT_RESULTS.csv` 必须恰好 10 行。

即使某个组件当前没有完整结果，也必须保留该行并填：

```text
N/A
```

同时在 `status` / `reason` 字段说明原因。

---

# 2. 数据来源

首先读取当前主 checkout 和相关 worktree 中所有已有的结果文件。

优先从当前项目最新汇总文件开始：

```text
result-simplified*.md
result-record*.md
```

然后按每个组件查找：

```text
outputs/**/HANDOFF*.json
outputs/**/RESULT*.csv
outputs/**/RESULT*.md
outputs/**/CLOSED_LOOP*.csv
outputs/**/CLOSED_LOOP*.md
outputs/**/MAIN_TABLE*.csv
outputs/**/SUMMARY*.json
outputs/**/GATE*.json
outputs/**/RUN_MANIFEST.json
```

特别关注当前 0818 相关目录：

```text
outputs/0818_projected_action_auto/
outputs/0818_projected_curation_bundle/
outputs/0818_retrieval_hygiene_bundle/
outputs/0818_actual_baselines_novelty/
outputs/component_sweep_0818/            # 若已经存在
```

同时需要读取历史上每个具体组件已有的 canonical artifacts，例如：

```text
verify_tool
importance_tagging
subtractive_curation
auto_populate_first_search
structured/textual comparison
real influence
K4/K8 fork replay
actual-LoRA closed-loop
route-head diagnostics
```

不要只依赖单个 `result-simplified.md` 中的文字总结；如果存在对应 handoff / CSV / JSON，优先读取 artifact 中的原始数值。

---

# 3. 冲突结果的优先级

项目中存在大量历史结果、修复后的结果、旧 evaluator 结果和 provisional result。

必须按以下优先级解析。

## Priority 1：最新 canonical / final handoff

最高优先级：

```text
latest dated canonical output
FINAL
final
paper-grade
completed
latest provenance correction
H*_HANDOFF.json
```

若 handoff 明确写：

```text
supersedes previous result
discard old handoff
latest canonical
provenance correction
```

则旧结果不得用于主表。

---

## Priority 2：actual-model + real multi-step closed-loop

如果同一个组件同时存在：

```text
route proxy
same-state proxy
route-head
actual LoRA real closed-loop
```

主表优先使用：

```text
actual model weights
student_inference_privilege=false
real multi-step closed-loop
```

即：

```text
actual-LoRA / actual-model real closed-loop
>
route-head real loop
>
same-state proxy
>
route JS / CE / agreement
```

---

## Priority 3：修复后的 evaluator / runtime

任何明确标记为以下问题的结果必须排除：

```text
broken load
wrong adapter stacking
parser contract bug
Harmony bug
double PEFT wrap
constant-zero evaluator artifact
missing gold contract
parsable_rate=0 bridge
mock / synthetic
```

这些结果可以进入 provenance，但不能进入主指标。

---

## Priority 4：没有 paper-grade metric 时，保留最佳“当前已有层级”

如果某组件没有 actual-model closed-loop：

可以退而使用：

```text
route-head real closed-loop
same-state K4/K8 utility
route proxy
```

但必须显式填写：

```text
metric_level
```

例如：

```text
ACTUAL_MODEL_REAL_CLOSED_LOOP
ROUTE_HEAD_REAL_CLOSED_LOOP
SAME_STATE_VALUE
ROUTE_PROXY_ONLY
AUDIT_ONLY
NO_RESULT
```

不允许把 proxy 数字伪装成 paper-grade result。

---

# 4. Teacher / Student Before / Student After 的统一定义

本任务只汇总已有结果。

不要重新定义或重跑。

尽最大可能从现有 artifacts 中提取：

```text
Teacher metric
Student Before OPD metric
Student After OPD metric
```

---

## 4.1 Teacher

Teacher 定义优先为：

```text
same model / same initialization
target component ON
```

若当前已有实验只提供：

```text
Full Harness
full-module teacher
component-on branch
K4/K8 full branch
```

则根据实际 artifact 填写，并记录：

```text
teacher_definition
```

不要错误地把：

```text
Full Harness
```

和：

```text
single-component Teacher
```

混为一谈。

如果没有单组件 Teacher real closed-loop metric：

```text
teacher_overall_reward = N/A
```

可以保留：

```text
teacher_value_metric
```

等已有指标。

---

## 4.2 Student Before OPD

优先使用同一实验 contract 下的：

```text
Base Student
CLEAN_BASE
BASE_REDUCED
student_before
```

如果没有严格 matched Base：

```text
student_before_overall_reward = N/A
```

不要从其他实验抄一个 Base 数字。

---

## 4.3 Student After OPD

优先使用：

```text
actual model / actual LoRA
no privilege inference
real closed-loop
```

如果有多个 seeds：

默认汇总：

```text
mean over completed formal seeds
```

同时记录：

```text
best_seed
best_value
n_seeds
```

如果只有 route-level result：

填到对应 proxy 字段，不填 actual-model reward。

---

# 5. 主 CSV schema

生成：

```text
COMPONENT_10_CURRENT_RESULTS.csv
```

必须包含以下列。

```text
component
component_category
realizability
current_status
metric_level

teacher_definition
teacher_overall_reward
teacher_trajectory_recall
teacher_curated_evidence_recall
teacher_final_answer_recall

student_before_definition
student_before_overall_reward
student_before_trajectory_recall
student_before_curated_evidence_recall
student_before_final_answer_recall

student_after_method
student_after_overall_reward
student_after_trajectory_recall
student_after_curated_evidence_recall
student_after_final_answer_recall

student_after_delta_vs_before
student_after_best_seed
student_after_n_seeds

pure_opd_overall_reward
pure_opd_delta_vs_before
rl_plus_opd_overall_reward
rl_plus_opd_delta_vs_before

same_state_value_K4
same_state_value_K8
route_proxy_metric

event_support
student_inference_privilege
actual_model_weights

decision
reason
source_path
source_date
```

---

# 6. `pure_opd` / `rl_plus_opd` 字段处理

当前很多组件还没有新统一 EasyOPD sweep 的：

```text
PURE_OPD
RL_PLUS_OPD
```

结果。

因此：

如果当前 artifact 中不存在严格对应的统一实验：

```text
pure_opd_overall_reward = N/A
rl_plus_opd_overall_reward = N/A
```

不要把历史上：

```text
reverse KL
action CE
route-KL
value-weighted KL
```

自动重命名为新的统一 `PURE_OPD`。

只有 experiment / manifest 明确满足当前定义时才填写。

---

# 7. 每个组件的特殊处理规则

## 7.1 `verify_tool`

已知它改变 action space。

主表应优先记录：

```text
realizability = NON_REALIZABLE_ACTION_SPACE_MISMATCH
```

如果历史上只有 route-proxy distillation：

可以记录 proxy，但不能把它写成成功 internalization。

如果没有合法 Student After：

```text
student_after_* = N/A
```

---

## 7.2 `importance_tagging`

必须优先使用最新 proper same-`xi_t` K4/K8 formal fork result。

旧 approximate influence positive 若被最新 formal gate supersede：

不得作为当前主结论。

---

## 7.3 `subtractive_curation`

区分：

```text
旧 all-zero result
evaluator/data-contract audit
projected curation bundle
```

如果最新结果明确 discard / missing contract：

以最新 canonical decision 为准。

不要把旧 zero reward 直接解释成真实组件 utility=0。

---

## 7.4 `auto_populate_first_search`

需要区分至少：

```text
旧 route-KL actual-LoRA closed-loop
H20 clean-init AUTO
PROJECTED_ACTION_AUTO
```

当前 Projected-Action 若只有：

```text
training completed
closed-loop pending
```

则不能把它写成 Student After positive/negative。

主表应该采用：

```text
当前最新“已完成闭环”的结果
```

同时在 `reason` 写：

```text
PROJECTED_ACTION newer training exists but closed-loop pending
```

---

## 7.5 `content_dedup`

如果最新 retrieval bundle 中：

```text
event trigger = 0
```

必须显式：

```text
event_support = 0
```

如果 actual-LoRA 结果存在但数据并没有真实 dedup event：

不得把微小 reward gain解释为 component internalization success。

decision 应体现：

```text
NO_ACTIVE_EVENT_SUPPORT
```

或最新 canonical discard。

---

## 7.6 `chunk_neighbors`

如果没有正式独立 component result：

```text
NO_RESULT
```

不要从组合实验推断单组件数值。

---

## 7.7 `evidence_graph`

如果只有历史 influence / route-level / component-on result：

按最高可用 metric level 填。

不要从 Full Harness 总收益拆出 evidence_graph 的收益。

---

## 7.8 `sentence_compress`

同上。

---

## 7.9 `token_budget_marker`

同上。

特别注意：

如果它出现在 Full Harness flag 中，但没有单组件 isolation：

不能把 Full Harness metric当作 token_budget_marker Teacher metric。

---

## 7.10 `adaptive_rerank_instruction`

如果只有 bundle 中：

```text
AUTO_DEDUP_RERANK
```

结果：

不得直接当作 adaptive_rerank 单组件 Student After。

可以在 `reason` 写：

```text
only bundle-level evidence available
```

主数值填 N/A，除非存在独立 single-component artifact。

---

# 8. 10-component 主表的决策标签

`decision` 只能使用以下标准标签之一：

```text
PASS_STUDENT_AFTER_GT_BEFORE
FAIL_STUDENT_AFTER_LE_BEFORE
TEACHER_COMPONENT_NO_POSITIVE_UTILITY
NON_REALIZABLE_ACTION_SPACE_MISMATCH
NO_ACTIVE_EVENT_SUPPORT
BLOCKED_DATA_CONTRACT
TRAINING_COMPLETED_CLOSED_LOOP_PENDING
ROUTE_PROXY_ONLY
AUDIT_ONLY
DISCARD_COMPONENT
NO_RESULT
INCONCLUSIVE
```

如果最新 canonical artifact 已有明确：

```text
DISCARD
STOP
REDESIGN
BLOCKED
```

优先映射到最接近的上述标签。

---

# 9. 生成一个简洁 Markdown 表

同时生成：

```text
COMPONENT_10_CURRENT_RESULTS.md
```

只保留核心列：

```text
Component
Status
Metric Level
Teacher
Student Before
Student After
Delta
K4
K8
Event Support
Decision
```

必须恰好 10 行。

数值保留 4 位小数。

没有结果：

```text
N/A
```

---

# 10. Provenance 表

生成：

```text
COMPONENT_10_RESULT_PROVENANCE.csv
```

每个组件至少一行，允许多行。

字段：

```text
component
selected_as_main
result_type
date
output_dir
artifact_file
metric_level
status
supersedes
excluded_reason
notes
```

目的：

明确为什么最终主表选了这个结果，而不是历史上的另一个。

例如：

```text
AUTO old route proxy
AUTO actual-LoRA closed-loop
AUTO H20 clean-init
AUTO projected-action training pending
```

都可以保留 provenance，但只有一个 current main result。

---

# 11. Gap 报告

生成：

```text
COMPONENT_10_RESULT_GAPS.md
```

必须回答：

```text
1. 哪些组件已经有 Teacher / Before / After 三者完整 real closed-loop？
2. 哪些只有 Teacher utility？
3. 哪些只有 route proxy？
4. 哪些有 actual-LoRA 但没有合法闭环？
5. 哪些完全没有独立实验？
6. 哪些只有 bundle-level evidence，不能拆成单组件？
7. 哪些因为 action-space mismatch 原则上不可内化？
8. 哪些因为 event support = 0 暂时无法评估？
9. 哪些已有结果被 evaluator/runtime bug 判废？
10. 当前距离完整 10-component sweep 还缺哪些实验？
```

---

# 12. 数据一致性检查

在输出 CSV 前必须检查：

```text
no duplicated component rows
exactly 10 main rows
no missing component names
no NaN string ambiguity
all missing values use N/A
all numeric values parseable
source_path exists
source_date consistent with artifact
```

并检查：

```text
Student Before 与 Student After 是否来自同一 evaluator contract
Teacher 是否为 single-component isolation
actual_model_weights 是否真实
student_inference_privilege 是否 false
```

若不匹配：

```text
不要计算 delta
```

对应 delta 填：

```text
N/A
```

并写 reason。

---

# 13. 不允许做的推断

禁止：

```text
1. 用 Full Harness 总指标作为单组件 Teacher 指标
2. 用 route-head result 代替 actual-model result
3. 用 bundle result 拆成单组件 result
4. 用 proxy reward 当 external task reward
5. 用旧 broken evaluator 数字覆盖最新修复结果
6. 把训练完成但 closed-loop pending 写成成功
7. 把 positive K4 当作 Student After > Before
8. 把 historical action CE / route-KL 自动映射成新 PURE_OPD
9. 把 N/A 写成 0
10. 根据趋势“估计”缺失数值
```

---

# 14. 建议的执行流程

```text
Step 1
扫描 result-simplified / result-record 最新文件。

Step 2
为 10 个组件建立 candidate artifact list。

Step 3
读取 candidate handoff / CSV / JSON。

Step 4
按 precedence rule 选择 current canonical result。

Step 5
填写 provenance table。

Step 6
填写 10-row main CSV。

Step 7
运行 consistency audit。

Step 8
生成 markdown summary 和 gap report。

Step 9
生成 SHA256SUMS。
```

---

# 15. 最终交付

必须提供：

```text
outputs/current_component_results_summary/COMPONENT_10_CURRENT_RESULTS.csv
```

这是最主要的用户交付。

并同时提供：

```text
COMPONENT_10_CURRENT_RESULTS.md
COMPONENT_10_RESULT_PROVENANCE.csv
COMPONENT_10_RESULT_GAPS.md
RUN_MANIFEST.json
SHA256SUMS
```

最终 handoff 中必须明确：

```text
status = CURRENT_10_COMPONENT_RESULTS_AGGREGATED
n_components = 10
n_complete_teacher_before_after = ?
n_actual_model_closed_loop = ?
n_proxy_only = ?
n_no_result = ?
```

不要把“表成功生成”解释为“研究已经完成”。

本任务只负责准确地把**当前已有证据**整理成统一、可审计的 10-component CSV。
