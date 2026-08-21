# H100-1：Action / Projectable Components Sweep

## 1. 本机组件

按顺序：

```text
1. auto_populate_first_search
2. importance_tagging
3. subtractive_curation
```

这三类都属于“Harness effect 需要被转换成 Student-native behavior”的组件，因此集中在一台机器上，复用 projected-action / argument supervision 代码路径。


# 本轮 5K 数据扩容目标

本轮相对上一版的核心变化只有一个：正式训练数据统一扩容到“1,000–2,000 unique queries × 每 query 2–4 次独立 Student on-policy rollout → 5,000 unique event-active states/可训练组件”。其他 Base / Teacher / Student 定义、loss、evaluator、seed、公平性和组件语义全部保持原协议。

硬性禁止：复制 state、复制 query、重复同一 trajectory、伪造 event、修改组件触发条件来凑 5K。

---

# 全局实验协议（四台 H100 必须完全一致）

## A. 启动前硬门槛

只有在统一框架 handoff 明确：

```text
SCAPE_EASYOPD_READY
```

后才允许开始正式实验。

首先读取：

```text
/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD/outputs/scape_easyopd/framework/H1003_SCAPE_EASYOPD_HANDOFF.json
```

若不存在或 `status != SCAPE_EASYOPD_READY`：

```text
STOP_FRAMEWORK_NOT_READY
```

不得退回 legacy trainer。

所有机器使用同一个 frozen framework commit、同一个 EasyOPD SHA、同一个 verl SHA、同一个 Python/CUDA lock。

---

## B. 统一 Base / Teacher / Student 定义

四台机器必须从同一个：

```text
CANONICAL_STUDENT_BASE
```

开始。该 checkpoint 从 framework handoff 中读取，不允许每台机器自行选择。

### Student Before OPD

```text
weights = CANONICAL_STUDENT_BASE
target component = OFF
all other V8D components = OFF
student_inference_privilege = false
```

这是所有 10 个组件共用的 `Student Before OPD`。

如果 evaluator/query manifest 完全相同，应只运行一次并冻结结果；其他机器只引用同一 canonical row，不重复产生不同 Base。

### Teacher

Teacher 不是“更大的模型”，而是：

```text
weights = CANONICAL_STUDENT_BASE
target component = ON
all other V8D components = OFF
```

也就是说，对每个组件只打开这一个 Harness component，隔离得到该组件本身能带来的 teacher/reference metric。

对于组件需要 privileged view 的情况，Teacher 看到该组件提供的完整 runtime/context；Student 不看。

### Student After OPD

```text
weights = CANONICAL_STUDENT_BASE + trained adapter
target component = OFF
all other V8D components = OFF
student_inference_privilege = false
```

必须分别训练两条：

```text
PURE_OPD
RL_PLUS_OPD
```

不要求 Student After 超过 Teacher。

第一判据始终是：

```text
Student After > Student Before
```

---

## C. 统一数据、5K event-active collection 与 evaluator

四台机器统一采用“大 query 覆盖 + 多次独立 on-policy rollout + event-active state 去重”的训练数据协议。DEV / TEST 仍然是全局冻结、query-disjoint 的真实 closed-loop evaluator；TRAIN 改成“共享训练 query universe + component-specific event-conditioned manifest”。

### C1. 全局 query manifests

四台机器共同冻结：

```text
manifests/COMPONENT_SWEEP_TRAIN_POOL.json
manifests/COMPONENT_SWEEP_DEV.json
manifests/COMPONENT_SWEEP_TEST.json
```

要求：

```text
TRAIN_POOL / DEV / TEST query-disjoint
TRAIN_POOL 至少支持 2,000 unique queries；若真实 corpus 不足，使用全部真实 unique queries并记录
same retriever
same max_steps
same reward
same final-answer scorer
same parser/tool runtime
same temperature/top-p
same generation budget
```

DEV / TEST 默认：

```text
DEV  = 128 unique queries
TEST = 256 unique queries
```

正式主表只使用：

```text
real multi-step closed-loop
```

route proxy / same-state KL / agreement 只做 diagnostic。

### C2. 每个可训练组件的数据目标

每个可训练组件必须独立构建：

```text
1,000–2,000 unique TRAIN queries
每个 query 2–4 次独立 Student on-policy rollout
最终抽取并冻结 5,000 unique event-active states
```

这里的 `5,000` 指去重后的真实 event-active Student states，不是 5,000 条重复样本，也不是 5,000 个 teacher token span。

每个组件写：

```text
manifests/component_sweep_5k/<COMPONENT>/TRAIN_QUERIES.json
manifests/component_sweep_5k/<COMPONENT>/ROLLOUT_MANIFEST.jsonl
manifests/component_sweep_5k/<COMPONENT>/EVENT_ACTIVE_STATES_ALL.jsonl
manifests/component_sweep_5k/<COMPONENT>/TRAIN_STATES_5K.jsonl
manifests/component_sweep_5k/<COMPONENT>/DATA_STATS.json
```

PURE_OPD 与 RL_PLUS_OPD、seed42 与 seed43 必须使用同一个 frozen component query manifest 和同一个 5K state budget contract；不得给某个 loss/seed 偷换更好的数据。

### C3. 独立 rollout 定义

同一 query 的 2–4 次 rollout 必须真正独立：

```text
same CANONICAL_STUDENT_BASE at collection start
same query / environment contract
different rollout_seed
independent sampling trajectory
no trajectory replay
no teacher-forced action during Student rollout
```

Teacher 不独立生成另一条 trajectory；仍然只在 Student 实际访问到的 state/prefix 上构造对应 privileged view / Harness effect 并给出 supervision。

### C4. 自适应扩容到 5K

统一按以下顺序扩容，避免一开始无谓生成 8K rollout：

```text
Stage A: 1,000 unique queries × 2 rollouts/query
Stage B: 若 unique event-active states < 5,000，扩到 1,500 queries × 2
Stage C: 若仍不足，扩到 2,000 queries × 2
Stage D: 若仍不足，对已选 queries 增加第 3 次独立 rollout
Stage E: 若仍不足，再增加第 4 次独立 rollout
```

允许在 Stage B–E 使用“真实 event-conditioned query prioritization”提高触发率，但只能依据 train corpus 中可合法预先计算的结构信号或已观察到的真实 trigger，不得修改 component 语义、伪造 trigger、使用 DEV/TEST gold/qrel、未来 observation 或答案泄漏。

### C5. unique event-active state 去重规则

为每个 event-active state 计算：

```text
state_uid = SHA256(
    component
  + query_id
  + normalized_student_visible_prefix
  + normalized_tool_history
  + normalized_student_observable_env_state
  + normalized_event_or_projectable_target
)
```

明确：

```text
rollout_seed 不进入 state_uid
teacher privileged-only text / hidden runtime bookkeeping 不进入 student-visible state fingerprint
完全相同的 Student-visible state 即使来自不同 rollout 也只计 1 次
```

同时记录：

```text
query_id
rollout_id
rollout_seed
step_id
event_type
state_uid
projectable_target（若有）
terminal_reward（若已有）
```

为防止少数长轨迹垄断 5K 数据，最终抽样优先做 query-balanced / event-subtype-balanced selection；同一 query 的 states 数量必须在 `DATA_STATS.json` 中报告分布（min/median/p95/max）。

### C6. 5K 选择规则

若去重后：

```text
n_unique_event_active >= 5,000
```

则从 `EVENT_ACTIVE_STATES_ALL.jsonl` 中按固定 seed 做可复现的分层抽样，恰好冻结：

```text
TRAIN_STATES_5K.jsonl = 5,000 unique states
```

优先覆盖更多 query、event subtype、trajectory depth 与 reward outcome，而不是简单按文件前 5,000 条截断。

若 2,000 unique queries × 4 rollouts/query 后仍然：

```text
n_unique_event_active < 5,000
```

则记录：

```text
INSUFFICIENT_5K_EVENT_SUPPORT
```

不得重复 state、重复 query 文本或人工制造 event 来凑 5K。对 trainable component，正式 5K sweep 不进入训练；Teacher / Student Before / event-support metric 仍照常写入主表。


---

## D. 两种训练 loss

### D1. `PURE_OPD`

默认：

```text
L_OPD = exact reverse KL
```

要求：

```text
token-level
same-tokenizer full-vocab exact reverse KL
response/tool span mask correct
teacher scores Student on-policy prefix
```

对于 `PROJECTABLE` 组件，Harness effect 本身不是 teacher 显式 token action，因此不允许硬做无意义的 teacher-token reverse KL。

这类组件统一使用框架中已经验证过的 canonical：

```text
L_OPD_PROJECTABLE =
    L_projected_action_CE
  + lambda_next * L_next_turn_reverse_KL
```

默认：

```text
lambda_next = 1.0
```

除非 framework acceptance 给出了另一个固定值；四台机器必须一致。

### D2. `RL_PLUS_OPD`

固定：

```text
L_total =
    lambda_rl  * L_GRPO
  + lambda_opd * L_OPD
```

默认：

```text
lambda_rl  = 1.0
lambda_opd = 1.0
```

其中 `L_OPD` 与 PURE_OPD 使用完全相同的 component-specific objective。

禁止每个组件分别调 lambda 来追结果。

RL reward 必须与 real closed-loop evaluator 的主 reward 同源；不得为不同组件发明不同 reward。

使用 EasyOPD/verl 已验证的 GRPO implementation，不自行重写。

---

## E. 训练 seeds、5K 数据预算与公平性

每个可训练组件都跑：

```text
PURE_OPD seeds 42,43
RL_PLUS_OPD seeds 42,43
```

即每组件 4 个 actual-model cells。

每个 cell 的数据预算必须完全一致：

```text
same component TRAIN_QUERIES.json
same 5,000 unique event-active state budget
same state selection seed / selection algorithm
same optimizer step budget
same epochs/effective tokens where applicable
same LoRA rank/alpha
same reference/anchor config
```

默认 LoRA/config 从 framework acceptance 的 canonical config 读取。

不得单独为某组件、某 seed 或某 loss 增加 epochs/LR，也不得让 RL+OPD 看更多 query 或更多 OPD states。

`RL_PLUS_OPD` 中 GRPO 自身需要的 online environment rollout 仍按 EasyOPD/verl canonical implementation 运行；但 OPD supervision 的 coverage/budget 仍受同一 `TRAIN_QUERIES.json + 5K unique state contract` 约束，并在 run manifest 中分开报告：

```text
n_opd_unique_queries
n_opd_rollouts_collected
n_opd_unique_event_states_raw
n_opd_train_states = 5000
n_grpo_online_rollouts
```

禁止把 GRPO 额外 rollout 偷算成“5K OPD states”而不去重、不审计。


---

## F. 5K Event support gate

正式训练前，每个组件先运行：

```bash
python scripts/scape_component_opd.py audit --component <NAME>
python scripts/scape_component_opd.py collect \
  --component <NAME> \
  --event-conditioned \
  --query-min 1000 \
  --query-max 2000 \
  --rollouts-min 2 \
  --rollouts-max 4 \
  --target-unique-event-states 5000
```

如果当前 CLI 尚无这些参数，必须在统一 framework 中补齐等价配置；不得用 shell 层重复执行后简单 concat 来冒充去重 collector。

至少输出：

```text
n_queries_available
n_queries_selected
n_rollouts_total
n_states_raw
n_event_active_raw
n_unique_event_active
event_rate_per_state
event_rate_per_rollout
n_queries_with_event
query_event_coverage
n_projectable
n_valid_args
n_terminal_reward
state_uid_collision_count
```

正式 5K 训练硬门槛：

```text
1,000 <= n_queries_selected <= 2,000
每个 selected query 的 rollout_count ∈ [2,4]
n_unique_event_active >= 5,000
TRAIN_STATES_5K 恰好 5,000 unique state_uid
```

若自然 support 不足，可以做真实 event-conditioned sampling，例如：

- subtractive：优先采样 curated set 接近容量上限、且后续会出现新 evidence 的真实 query/trajectory；
- content_dedup：先离线定位真实 duplicate clusters，再优先采样能检索到这些 cluster 的真实 query；
- chunk_neighbors：优先采样真实 chunk-hit event；
- AUTO：第一次成功 search event；
- importance：真实 curate event；
- evidence_graph：真实跨文档 bridge/entity relation event；
- sentence_compress：真实 noisy observation compression event；
- token_budget_marker：真实预算变化/终止决策相关 state。

禁止修改 component 语义、复制文档、伪造 trigger、注入未来信息或用 DEV/TEST 反向挑 query。

若达到 2,000 queries × 4 rollouts/query 后仍不足 5K：

```text
INSUFFICIENT_5K_EVENT_SUPPORT
```

主表中 Student After 填 `N/A`，同时保留 Teacher / Student Before metric 和真实 event support；不得退回旧的 256/512 小样本门槛完成“10/10 可训练”。


---

## G. 每组件固定执行顺序

```text
1. audit realizability
2. freeze global TRAIN_POOL / DEV / TEST query-disjoint contract
3. select first 1,000 component TRAIN queries
4. collect 2 independent Student on-policy rollouts/query
5. extract event-active states + compute state_uid + deduplicate
6. adaptively expand queries/rollouts until 5K unique states or 2,000×4 ceiling
7. freeze TRAIN_QUERIES.json + TRAIN_STATES_5K.jsonl + DATA_STATS.json
8. run Teacher metric
9. run Student Before metric / verify canonical Base
10. train PURE_OPD seed42/43 with identical 5K data budget
11. train RL_PLUS_OPD seed42/43 with identical OPD data budget
12. reload actual adapters
13. run DEV real closed-loop
14. run TEST real closed-loop
15. aggregate two seeds
16. paired bootstrap vs Student Before
17. 20-case mechanism audit
18. write component handoff + data provenance
```

暂时：

```text
DO NOT RUN SHUFFLED CAUSAL CONTROL
```


---

## H. 主指标

每个组件至少报告：

```text
overall_reward                # primary
trajectory_recall
curated_evidence_recall
final_answer_recall
invalid_tool_rate
mean_turns
mean_tool_calls
```

如果某 benchmark 没有某项 gold：

```text
N/A
```

不能写 0。

### 组件机制指标

另存，不作为主表唯一成功标准：

```text
AUTO:
  search_to_curate_delay
  immediate_curate_rate
  relevant_added_rate

importance/subtractive:
  valid_add_rate
  valid_remove_rate
  irrelevant_removed_rate
  curated_set_churn

content_dedup:
  duplicate_trigger_rate
  duplicate_read_rate
  duplicate_curate_rate

chunk_neighbors:
  neighbor_recovery_rate
  followup_read/search_rate
  relevant_neighbor_usage

evidence_graph:
  bridge_entity_followup_rate
  new_entity_discovery

sentence_compress:
  useful-evidence retention
  redundant-read/search rate

token_budget_marker:
  termination timing
  late-step waste
  unnecessary tool calls

adaptive_rerank:
  topK overlap
  qrel recall@K
  query refinement quality
```

---

## I. 每组件必须输出统一 row

每个组件写：

```text
COMPONENT_RESULT.json
COMPONENT_RESULT.csv
COMPONENT_RESULT.md
```

JSON 至少：

```json
{
  "component": "...",
  "realizability": "...",
  "data": {
    "n_train_unique_queries": 0,
    "n_rollouts_total": 0,
    "n_event_active_raw": 0,
    "n_unique_event_active": 0,
    "n_train_unique_states": 0,
    "target_train_unique_states": 5000,
    "collection_status": "..."
  },
  "event_support": 0,
  "teacher": {
    "overall_reward": 0.0,
    "trajectory_recall": 0.0,
    "curated_evidence_recall": 0.0,
    "final_answer_recall": null
  },
  "student_before": {},
  "pure_opd": {
    "seed42": {},
    "seed43": {},
    "mean": {},
    "paired_bootstrap_vs_before": {}
  },
  "rl_plus_opd": {
    "seed42": {},
    "seed43": {},
    "mean": {},
    "paired_bootstrap_vs_before": {}
  },
  "decision": "..."
}
```

---

## J. 单组件判定

不缩 claim。

### PASS_PURE_OPD

```text
PURE_OPD mean > Student Before
>=2 seeds same positive direction
paired bootstrap CI vs Before supports positive effect
```

### PASS_RL_PLUS_OPD

同理。

### PASS_BOTH

两条都过。

### FAIL_COMPONENT_INTERNALIZATION

两条都不能稳定超过 Student Before。

### NON_REALIZABLE

action-space/effect 不存在 Student-native realization。

### INSUFFICIENT_5K_EVENT_SUPPORT

在 1,000–2,000 unique queries、每 query 2–4 次独立 rollout 的上限内，仍不足 5,000 unique event-active states，不能进入正式 5K 训练。

不要因为 Teacher 本身没提升就强行蒸馏；如果：

```text
Teacher <= Student Before
```

则记录：

```text
TEACHER_COMPONENT_NO_POSITIVE_UTILITY
```

并原则上停止该组件正式训练，除非用户另有指示。


# H100-1 专属执行

## 1. `auto_populate_first_search`

预期 classification：

```text
AUTOMATIC_SIDE_EFFECT
PROJECTABLE
```

Harness：

```text
successful first search
-> automatically add top-K docs to curated set
```

Student-native target：

```text
curate(add_ids=delta_add, remove_ids=[])
```

只使用 Student 当前可见 doc ids。

### PURE_OPD

```text
projected_action_CE + next_turn_reverse_KL
```

### RL_PLUS_OPD

```text
GRPO + projected_action_CE + next_turn_reverse_KL
```

必须重新用统一 framework collector 采集，不直接沿用旧 legacy trainer artifacts 作为正式数据。

5K 数据采集优先级：AUTO 以真实 `first successful search` 为 event；importance 以真实 importance-informed curate decision 为 event；subtractive 以真实 add/remove delta 为 event。三者都必须先完成 1K–2K queries × 2–4 independent rollout 的统一 collector，再进入下面训练命令。

命令模板：

```bash
python scripts/scape_component_opd.py run \
  --component auto_populate_first_search \
  --mode teacher_eval

python scripts/scape_component_opd.py run \
  --component auto_populate_first_search \
  --mode pure_opd \
  --seeds 42,43

python scripts/scape_component_opd.py run \
  --component auto_populate_first_search \
  --mode rl_plus_opd \
  --seeds 42,43

python scripts/scape_component_opd.py eval \
  --component auto_populate_first_search \
  --split dev,test
```

必须额外比较：

```text
Teacher AUTO metric
Student Before
Student After PURE_OPD
Student After RL+OPD
```

---

## 2. `importance_tagging`

预期：

```text
ARGUMENT_PRIVILEGE
PARTIAL/PROJECTABLE
```

不要监督 `importance` 数值本身。

正确 supervision：

```text
importance-informed teacher curation decision
-> realized add/remove choice
-> Student-native curate(add_ids, remove_ids)
```

若没有真实 valid add/remove 支持：

```text
INSUFFICIENT_5K_EVENT_SUPPORT
```

不要训练。

### PURE_OPD

```text
projected_action_CE + next_turn_reverse_KL
```

### RL_PLUS_OPD

```text
GRPO + same L_OPD
```

必须单独运行 Teacher utility：

```text
importance ON only
all other V8D OFF
```

若 Teacher 本身不能超过 Student Before：

```text
TEACHER_COMPONENT_NO_POSITIVE_UTILITY
```

停止正式蒸馏，并在总表中留下该 row。

---

## 3. `subtractive_curation`

预期：

```text
AUTOMATIC_SIDE_EFFECT
PROJECTABLE
```

必须先解决真实 event support：

```text
curated_ids_pre
curated_ids_post
added = post - pre
removed = pre - post
```

合法性：

```text
all added ids visible to Student
all removed ids in current Student curated set
```

优先采样：

```text
curated_count >= 24
```

重点：

```text
curated_count == capacity
new candidate evidence arrives
```

不允许伪造满集合。

### PURE_OPD

```text
projected_action_CE + next_turn_reverse_KL
```

### RL_PLUS_OPD

```text
GRPO + same L_OPD
```

---

## 4. GPU 调度建议

每组件 4 个正式训练 cells：

```text
PURE_OPD seed42
PURE_OPD seed43
RL_PLUS_OPD seed42
RL_PLUS_OPD seed43
```

若单 cell 单 H100 可运行：

### Wave 1

```text
GPU0-3: AUTO 4 cells
GPU4-7: importance 4 cells
```

### Wave 2

```text
GPU0-3: subtractive 4 cells
GPU4-7: evaluator / bootstrap / case analysis
```

若 EasyOPD/verl 需要多卡 placement，使用 framework 自己的 Ray placement group，不得为了并行改变 batch/update budget。

---

## 5. 本机输出

### 5K data artifacts（每个可训练组件目录内必须存在）

```text
TRAIN_QUERIES.json
ROLLOUT_MANIFEST.jsonl
EVENT_ACTIVE_STATES_ALL.jsonl
TRAIN_STATES_5K.jsonl
DATA_STATS.json
DATA_PROVENANCE.md
```

其中 `TRAIN_STATES_5K.jsonl` 必须恰好 5,000 个不同 `state_uid`；若组件被 realizability / teacher-utility / 5K-support gate 阻断，则写明阻断原因并保留已采集的真实数据统计，不得伪造该文件。



```text
outputs/component_sweep_0818/h100_1/
├── auto_populate_first_search/
├── importance_tagging/
├── subtractive_curation/
├── H1001_COMPONENT_ROWS.csv
├── H1001_COMPONENT_ROWS.json
├── H1001_COMPONENT_HANDOFF.json
└── SHA256SUMS
```

`H1001_COMPONENT_ROWS.csv` 恰好 3 行。

每行必须有：

```text
component
n_train_unique_queries
n_rollouts_total
n_event_active_raw
n_unique_event_active
n_train_unique_states
collection_status
teacher_reward
student_before_reward
pure_opd_reward
pure_opd_delta_vs_before
rl_plus_opd_reward
rl_plus_opd_delta_vs_before
teacher_trajectory_recall
before_trajectory_recall
pure_opd_trajectory_recall
rl_plus_opd_trajectory_recall
teacher_final_answer_recall
before_final_answer_recall
pure_opd_final_answer_recall
rl_plus_opd_final_answer_recall
event_support
decision
```

最终只允许基于 real closed-loop 写结论。
