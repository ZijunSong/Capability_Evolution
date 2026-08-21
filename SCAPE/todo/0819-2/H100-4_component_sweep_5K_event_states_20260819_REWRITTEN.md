# H100-4：Control Components + 10-Component Aggregation — 5K Unblock Protocol

# 0. 本轮状态：先解除 5K collection 阻断，再运行正式实验

当前不得直接启动任何正式 `PURE_OPD` / `RL_PLUS_OPD` 训练。上一轮已经确认，正式 5K collection 被以下硬门槛阻断：

```text
BLOCKER-1  当前 TRAIN query 只有 446 个，低于正式 collector 的 query-min=1000
BLOCKER-2  framework handoff 没有 CANONICAL_STUDENT_BASE
BLOCKER-3  handoff 中引用的 /opt/scape-easyopd-smoke7 不存在
BLOCKER-4  EasyOPD 环境中没有可导入的真实 Harness-1 / SCAPE runtime
BLOCKER-5  scripts/scape_component_opd.py collect 仍是 4 行 synthetic smoke collector
BLOCKER-6  因上述问题，当前没有合法 TRAIN_STATES_5K.jsonl；这是正确行为，不得补写伪造文件
```

本轮任务分成两个严格阶段：

```text
Phase U  = UNBLOCK / integration / real 5K collection
Phase E  = formal experiment / training / evaluation
```

只有 Phase U 全部通过后，才能进入 Phase E。

本轮禁止：

```text
- 为了让实验“跑起来”而保留 synthetic collector
- 重复 446 个 query、做字符串改写后当新 query
- 复制 state / trajectory / event 来凑 5K
- 把 smoke artifact 混入正式 manifest
- 给 Student 暗中打开 Harness privilege
- 用 DEV/TEST gold、qrel、答案或未来 observation 构造 TRAIN
- 因 /opt 路径不存在而自行创建一个空目录冒充旧环境
```

---

# 1. 统一根目录与环境：四台 H100 完全一致

## 1.1 固定目录

```bash
export CAP_ROOT=/mnt/songzijun/Capability_Evolution
export SCAPE_ROOT=/mnt/songzijun/Capability_Evolution/SCAPE
export EASYOPD_ROOT=/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD
```

Harness-1 的真实实现以：

```text
/mnt/songzijun/Capability_Evolution/SCAPE
```

为唯一 source of truth。

EasyOPD/verl 集成、collector、trainer 和 evaluator 的工作目录以：

```text
/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD
```

为准。

## 1.2 环境配置：废弃 `/opt/scape-easyopd-smoke7`

每台机器、每个 shell、每个 Ray worker 启动正式任务前，都必须先执行：

```bash
source /mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD/scripts/setup_scape_easyopd_smoke7_env.sh
```

不得再要求：

```text
/opt/scape-easyopd-smoke7
```

存在。

环境脚本必须负责或验证：

```text
Python executable
PYTHONPATH / editable-install paths
CUDA / NCCL / Ray 基础环境
EasyOPD dependencies
verl dependencies
SCAPE/Harness-1 可导入路径
model cache/HF cache（若已配置）
```

在正式任务开始前保存：

```bash
which python
python -V
python -c 'import torch; print(torch.__version__, torch.version.cuda)'
python -c 'import verl; print("VERL_IMPORT_OK")'
```

并写入：

```text
outputs/component_sweep_0818/preflight/ENVIRONMENT.txt
```

若环境脚本执行失败：

```text
STOP_ENV_SETUP_FAILED
```

不得降级到另一个临时 conda/venv 后继续跑正式实验。

---

# 2. CANONICAL_STUDENT_BASE：先统一为 `openai/gpt-oss-20b`

本轮不再等待旧 handoff 提供空缺的 `CANONICAL_STUDENT_BASE`。先统一固定：

```text
CANONICAL_STUDENT_BASE = openai/gpt-oss-20b
```

如果机器使用 Hugging Face cache 中的本地 snapshot，允许把实际加载路径解析到本地 snapshot，但必须同时记录：

```text
logical_model_id = openai/gpt-oss-20b
resolved_model_path = <actual local snapshot or HF id>
model_revision = <revision if available>
model_config_sha256 = <sha256>
tokenizer_config_sha256 = <sha256>
```

四台机器的 logical model、revision、tokenizer 和加载 contract 必须一致。

## 2.1 gpt-oss 加载硬约束

gpt-oss 不能被当成普通 chat-template 模型随意拼 prompt。SCAPE/EasyOPD bridge 必须沿用 gpt-oss 已验证的 Harmony context/render contract；不得为了省事换成一个普通 `apply_chat_template` 路径后宣称是同一 Base。

至少加入：

```text
GptOssHarmonyAdapter
- build_context(...)
- render(...)
- tool schema / assistant action rendering
- response span identification
- tokenizer consistency check
```

并做 10-case deterministic serialization regression test。失败则：

```text
STOP_GPT_OSS_HARMONY_CONTRACT_FAILED
```

## 2.2 Student / Teacher / Student After

### Student Before OPD

```text
weights = openai/gpt-oss-20b
Harness-1 target component = OFF
all other target V8D components = OFF
student_inference_privilege = false
```

### Teacher

Teacher 仍然不是更大的模型：

```text
weights = openai/gpt-oss-20b
target component = ON
all other V8D components = OFF
```

Teacher 只在 Student 实际访问到的 state/prefix 上获得该 target component 的 privileged context / side effect / runtime signal。

### Student After OPD

```text
weights = openai/gpt-oss-20b + trained adapter
target component = OFF
all other V8D components = OFF
student_inference_privilege = false
```

第一判据仍是：

```text
Student After > Student Before
```

---

# 3. Harness-1 → EasyOPD：必须完成真实 runtime 集成

当前正式 collection 不能继续的核心原因不是“少一个 import”，而是 EasyOPD collector 没有真正执行 Harness-1 的 closed-loop runtime。本轮先完成 bridge。

## 3.1 集成原则

```text
SCAPE repository = Harness-1 runtime/component semantics 的 source of truth
SCAPE-EasyOPD     = training/collection/evaluation orchestration
```

优先：

```text
import / editable install / thin adapter
```

禁止：

```text
复制 Harness-1 组件代码到 EasyOPD 后自行改写语义
做一个 FakeHarness1 / DummyRuntime / MockSearchEnv 来满足接口
```

## 3.2 先做代码审计，不允许猜 API

Agent 必须在 `/mnt/songzijun/Capability_Evolution/SCAPE` 中定位并记录：

```text
1. Harness-1 / SCAPE 主 runtime 入口
2. environment reset / step / tool dispatch 入口
3. Student-visible state/prefix 的构造位置
4. search/read/curate/verify 等 tool schema 与 parser
5. 10 个目标 component 的真实 enable/disable 配置与 hook
6. event 发生前后的 runtime state
7. component side effect / privileged context 的实际生成位置
8. trajectory / reward / terminal state 的来源
```

输出：

```text
outputs/scape_easyopd/framework/HARNESS1_RUNTIME_INVENTORY.md
```

其中每个映射必须写真实文件路径、类/函数名，不得只写概念描述。

## 3.3 Bridge 的目标接口

在 EasyOPD 中实现一个 thin bridge，名称可为：

```text
scape_easyopd/harness1_bridge/
```

但实际目录以当前 repo 结构为准。必须暴露等价能力：

```python
runtime = Harness1Bridge(component=..., enabled=...)
obs = runtime.reset(query_record, rollout_seed)
step = runtime.step(student_action)
snapshot = runtime.snapshot_student_visible_state()
event = runtime.get_component_event()
teacher_view = runtime.build_teacher_view_from_same_state()
```

关键要求：

```text
- reset/step 真正调用 SCAPE/Harness-1
- tool call 真正进入原 parser/tool runtime
- snapshot 来自真实当前 state
- event 来自真实 component hook
- teacher_view 只能基于同一个 Student state 构造
- teacher 不另外 rollout 一条 trajectory
```

## 3.4 Import / runtime acceptance

必须新增并通过：

```text
TEST-1 import SCAPE/Harness-1 runtime
TEST-2 load openai/gpt-oss-20b adapter contract
TEST-3 run 1 real query end-to-end with all components OFF
TEST-4 run same query with exactly one target component ON
TEST-5 capture at least one real tool call
TEST-6 capture real pre/post runtime state
TEST-7 component event hook 不靠 synthetic flag 触发
TEST-8 Teacher view 与 Student prefix 的 query_id/rollout_id/step_id 对齐
```

输出：

```text
outputs/scape_easyopd/framework/HARNESS1_EASYOPD_ACCEPTANCE.json
```

只有：

```json
{"status":"HARNESS1_EASYOPD_READY","synthetic_fallback":false}
```

才允许进入正式 collection。

---

# 4. 446 → 2,000：正式 TRAIN query universe 的构造方法

当前 446 个 query 不足以满足正式 gate。不能复制，也不能仅做 paraphrase 扩容。本轮直接构造一个最终可扩到 Stage C 的：

```text
COMPONENT_SWEEP_TRAIN_POOL = 2,000 unique corpus-grounded queries
```

其中已有 446 个合法 train query 保留；新增目标为：

```text
至少 1,554 个通过验证的新 unique query
```

实际应先产生约 3,000–4,000 个 candidate，再经过严格验证/去重冻结 2,000 个，避免最后数量不足。

## 4.1 数据源边界

新 query 只能从 TRAIN corpus / TRAIN-side metadata 构造。开始构造前先冻结：

```text
DEV_QUERY_IDS
TEST_QUERY_IDS
DEV/TEST evidence doc ids（若 benchmark 可得）
DEV/TEST qrels / gold answers
```

这些内容只能用于 exclusion，不允许用于生成或筛选“更容易触发”的 TRAIN query。

优先采用 document-disjoint；若现有 benchmark 结构无法严格 document-disjoint，至少保证：

```text
query_id disjoint
normalized query text disjoint
answer/evidence leakage audit
no DEV/TEST qrel-driven prioritization
```

## 4.2 Corpus-grounded query synthesis，而非凭空出题

新增 query 采用“先取证据结构，再生成问题”的流程：

```text
TRAIN documents
  -> evidence bundle mining
  -> query specification
  -> natural-language query realization
  -> answer/evidence validation
  -> dedup/leakage audit
  -> frozen TRAIN_POOL
```

### Step Q1：从 TRAIN corpus 挖 evidence bundle

至少覆盖：

```text
A. single-document multi-chunk evidence
B. two-document bridge / entity relation
C. multi-document evidence aggregation
D. near-duplicate / overlapping documents
E. parent-child / neighboring chunk relation
F. noisy/redundant observation scenarios
G. evidence replacement / curation-pressure scenarios
```

注意：这些只是 query 构造时的 corpus 结构标签，不是人为触发 Harness event。真正 event 必须在 Student on-policy rollout 中自然发生。

### Step Q2：形成 query specification

每个 candidate 先保存结构化 spec：

```json
{
  "candidate_id": "...",
  "construction_method": "bridge|multi_chunk|multi_doc|duplicate_cluster|neighbor|...",
  "source_doc_ids": ["..."],
  "evidence_spans": [{"doc_id":"...","span":"..."}],
  "reference_answer": "...",
  "required_facts": ["..."],
  "forbidden_dev_test_overlap": false
}
```

### Step Q3：自然语言 query realization

可以使用当前统一模型/已有生成器把 spec 转成自然语言 query，但 generator 不能发明 spec 之外的新事实。若使用 `openai/gpt-oss-20b` 生成 query，只把它当 query writer；最终是否保留由后面的 corpus validator 决定。

query 必须：

```text
- 可由 source/evidence docs 回答
- 不在问题文本中泄漏答案
- 不是已有 446 query 的简单同义改写
- 需要真实 search/read/curate 过程，而不是纯常识即可稳定作答
- 不显式提到 component 名称或暗示“请触发某组件”
```

### Step Q4：可回答性与 evidence validation

至少做两层验证：

```text
1. lexical/structural validator
   - evidence doc ids 存在
   - evidence span 存在于对应 doc
   - reference answer 非空
   - query/reference 没有 obvious answer-copy leakage

2. retrieval/closed-loop feasibility validator
   - 使用与正式实验相同 retriever
   - 在不打开 target component 的情况下，相关 evidence 原则上可被 Student-native tool action 访问
   - 无不存在的 doc/chunk/tool id
```

如果 benchmark 有 gold/qrel 生成机制，可额外产生 train-only qrel；如果没有，不得伪造 evaluator gold，只把 corpus provenance 当训练 query 的 construction evidence。

### Step Q5：严格去重

对 446 个旧 query + 新 candidate 一起做：

```text
exact normalized-text dedup
query_id dedup
source/evidence bundle dedup audit
semantic near-duplicate filtering
```

至少保存：

```text
normalized_query_sha256
source_bundle_sha256
```

语义去重阈值必须冻结并写入 config；不得为某组件临时改变。

### Step Q6：冻结 2,000 个 query

选择时优先覆盖不同：

```text
construction_method
evidence-doc count
retrieval depth
source domains/types（若有）
answer types（若有）
```

而不是按 component trigger rate 直接选最终 2,000 个全局 pool。

输出：

```text
manifests/COMPONENT_SWEEP_TRAIN_POOL.json
manifests/COMPONENT_SWEEP_TRAIN_POOL_PROVENANCE.jsonl
manifests/COMPONENT_SWEEP_TRAIN_POOL_STATS.json
manifests/COMPONENT_SWEEP_QUERY_LEAKAGE_AUDIT.md
```

最终 gate：

```text
n_train_pool_unique_queries == 2000
n_exact_duplicate_queries == 0
n_dev_test_query_overlap == 0
```

若不能构造到 2,000，但已 >=1,000，可允许 Stage A 启动；不过要继续补 pool，直到能支持 Stage C。正式 collection 不得通过复制 query 补齐。

---

# 5. 5K 数据到底如何构造：真实 Student rollout → event state → Teacher supervision

本节取代原来“只有目标数字但没有可执行构造链”的描述。

每个可训练 component 的 `5K` 是：

```text
5,000 unique event-active Student states
```

不是：

```text
5,000 queries
5,000 trajectories
5,000 teacher responses
5,000 duplicated rows
```

## 5.1 Component-specific query manifest

从全局 2,000 TRAIN_POOL 中，为每个 component 固定一个有序 query list：

```text
manifests/component_sweep_5k/<COMPONENT>/TRAIN_QUERIES.json
```

初始选择 1,000 个。选择规则可以利用 TRAIN-only corpus 结构信号做 event-conditioned prioritization，但不能使用 future rollout information 预先伪造 event。

合法例子：

```text
content_dedup: duplicate-cluster index
chunk_neighbors: parent/neighbor chunk topology
subtractive: train-side evidence bundle 较多、可能形成 curation pressure
sentence_compress: long/noisy retrieved passages
```

非法例子：

```text
“这个 query 在 DEV 上高分”
“gold 显示第 3 步必触发”
手工改 query 要求模型调用某 component
```

## 5.2 Stage A–E rollout 扩容

严格按：

```text
Stage A: 1,000 queries × 2 independent Student rollouts/query
Stage B: 1,500 queries × 2
Stage C: 2,000 queries × 2
Stage D: 2,000 queries × 3
Stage E: 2,000 queries × 4
```

每完成一个 stage 都立即：

```text
extract real event-active states
compute state_uid
deduplicate
count n_unique_event_active
```

一旦：

```text
n_unique_event_active >= 5000
```

停止继续扩容，进入 5K freeze。

同一 query 的不同 rollout 必须：

```text
same openai/gpt-oss-20b base
same query/environment contract
different rollout_seed
independent sampling
no replay
no teacher-forced action
```

## 5.3 真实 collector 的逐步行为

对每个 `<query, rollout_seed>`：

```text
1. Harness1Bridge.reset(query, seed), target component OFF for Student behavior generation
2. gpt-oss Student 根据当前真实 Student-visible context 生成 action
3. Harness-1 parser/tool runtime 执行 action
4. 保存 Student-visible pre-state / action / tool result / post-state
5. 检查 target component 在“Teacher ON semantics”下，对同一 state 是否存在真实 event/effect
6. 若 event-active：
   a. 保存 Student-visible state fingerprint
   b. target component ON，仅基于同一 Student state 构造 privileged view/effect
   c. 构造 DIRECT 或 PROJECTABLE supervision
7. Student 继续自己的 on-policy trajectory；Teacher 不接管 trajectory
8. 直到 terminal/max_steps
```

对于某些 automatic side effect，实际实现可通过“双 runtime snapshot / isolated component hook replay”得到 Teacher effect，但必须保证：

```text
same Student pre-event state
same underlying environment data
Teacher component 是唯一差异
no future observation
```

## 5.4 每个 event row 最低 schema

`EVENT_ACTIVE_STATES_ALL.jsonl` 每行至少：

```json
{
  "component": "...",
  "query_id": "...",
  "rollout_id": "...",
  "rollout_seed": 0,
  "step_id": 0,
  "event_type": "...",
  "student_visible_prefix": "...",
  "tool_history": [],
  "student_observable_env_state": {},
  "event_payload_student_visible": {},
  "teacher_privileged_view_ref": "...",
  "projectable_target": null,
  "terminal_reward": null,
  "state_uid": "sha256...",
  "collector_mode": "real_harness1"
}
```

Teacher-only text不进入 Student-visible fingerprint。

## 5.5 `state_uid`

统一：

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
teacher privileged-only text 不进入 state_uid
hidden runtime bookkeeping 不进入 state_uid
相同 Student-visible state 跨 rollout 只算 1 个
```

## 5.6 从 raw unique states 冻结恰好 5,000

当 `n_unique_event_active >= 5000`：

先按 `state_uid` 去重，再做 deterministic stratified selection。优先级：

```text
1. 最大化 unique query coverage
2. 平衡 event subtype
3. 平衡 trajectory depth bins
4. 平衡 terminal reward/outcome bins（若已有）
5. 限制单一 query 对 5K 的垄断
```

建议默认单 query cap：

```text
soft cap = 8 states/query
```

若不足 5K，可逐级放宽，但必须在 `DATA_STATS.json` 记录实际 cap 与 query-state 分布。

使用固定：

```text
selection_seed = 20260818
```

最终：

```text
TRAIN_STATES_5K.jsonl 恰好 5000 行
len(unique(state_uid)) == 5000
collector_mode 全部 == real_harness1
synthetic_row_count == 0
```

## 5.7 5K 文件集

每个 trainable component 必须生成：

```text
TRAIN_QUERIES.json
ROLLOUT_MANIFEST.jsonl
EVENT_ACTIVE_STATES_ALL.jsonl
TRAIN_STATES_5K.jsonl
DATA_STATS.json
DATA_PROVENANCE.md
```

其中 `DATA_PROVENANCE.md` 必须回答：

```text
query 从哪里来
哪些是原 446，哪些是新构造
query synthesis/validation 版本
SCAPE commit
SCAPE-EasyOPD commit
model id/revision
collector config
retriever config
rollout seeds
state_uid schema version
selection seed
是否存在任何 synthetic/smoke rows（正式数据必须为 0）
```

---

# 6. 重写 `scape_component_opd.py collect`：从 smoke collector 变成 paper-grade collector

当前 4 行 synthetic collector 只能保留为单元测试 fixture，不允许继续作为 `collect` 默认实现。

## 6.1 CLI contract

正式 CLI 至少支持：

```bash
python scripts/scape_component_opd.py collect \
  --component <NAME> \
  --runtime harness1 \
  --student-base openai/gpt-oss-20b \
  --query-pool manifests/COMPONENT_SWEEP_TRAIN_POOL.json \
  --event-conditioned \
  --query-min 1000 \
  --query-max 2000 \
  --rollouts-min 2 \
  --rollouts-max 4 \
  --target-unique-event-states 5000 \
  --selection-seed 20260818
```

如果环境中无法加载 Harness-1：

```text
STOP_REAL_HARNESS_RUNTIME_UNAVAILABLE
```

而不是自动 fallback synthetic。

## 6.2 smoke 与 formal 必须物理隔离

允许：

```text
scripts/tests/fixtures/synthetic_smoke_*.jsonl
--mode smoke
```

但 `--mode formal` / 默认正式 collector 必须满足：

```text
synthetic_fallback = false
runtime = harness1
```

formal output 路径发现 `synthetic=true` 或 `collector_mode!=real_harness1` 时直接失败。

## 6.3 Collector audit 输出

至少：

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
synthetic_row_count
runtime_name
model_id
```

正式 5K gate：

```text
1000 <= n_queries_selected <= 2000
2 <= rollout_count_per_selected_query <= 4
n_unique_event_active >= 5000
TRAIN_STATES_5K rows == 5000
unique(state_uid) == 5000
synthetic_row_count == 0
runtime_name == harness1
model_id == openai/gpt-oss-20b
```

若 Stage E 后仍不足：

```text
INSUFFICIENT_5K_EVENT_SUPPORT
```

不得生成伪造 `TRAIN_STATES_5K.jsonl`。

---

# 7. Framework handoff 重定义

旧 handoff 的 `SCAPE_EASYOPD_READY` 不能只表示“loss smoke test 能跑”。本轮必须升级为包含：

```json
{
  "status": "SCAPE_EASYOPD_READY",
  "canonical_student_base": "openai/gpt-oss-20b",
  "environment_setup_script": "/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD/scripts/setup_scape_easyopd_smoke7_env.sh",
  "scape_root": "/mnt/songzijun/Capability_Evolution/SCAPE",
  "easyopd_root": "/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD",
  "harness_runtime": "Harness-1/SCAPE",
  "harness1_easyopd_ready": true,
  "formal_collector": "real_harness1",
  "synthetic_fallback": false,
  "train_pool_unique_queries": 2000
}
```

建议由 H100-3/framework owner 更新：

```text
/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD/outputs/scape_easyopd/framework/H1003_SCAPE_EASYOPD_HANDOFF.json
```

只有上述字段通过 audit 才允许四台机器共同进入 Phase E。

---

# 8. 正式训练 loss 与公平性（Phase E）

## 8.1 PURE_OPD

DIRECT component：

```text
L_OPD = exact token-level reverse KL
same tokenizer full-vocab
teacher scores Student on-policy prefix
response/tool span mask correct
```

PROJECTABLE component：

```text
L_OPD_PROJECTABLE = L_projected_action_CE + lambda_next * L_next_turn_reverse_KL
lambda_next = 1.0
```

## 8.2 RL_PLUS_OPD

```text
L_total = 1.0 * L_GRPO + 1.0 * L_OPD
```

`L_OPD` 必须与 PURE_OPD 的 component-specific objective 完全相同。

每个 trainable component：

```text
PURE_OPD     seeds 42,43
RL_PLUS_OPD  seeds 42,43
```

四个 cell 使用完全相同的：

```text
TRAIN_QUERIES.json
TRAIN_STATES_5K.jsonl
state selection seed
optimizer step budget
epoch/effective token budget
LoRA rank/alpha
reference/anchor config
```

GRPO online rollout 单独统计，禁止把它偷算进 5K OPD states。

---

# 9. 统一执行顺序

任何机器都按下面顺序，不允许跳步：

```text
U1. source setup_scape_easyopd_smoke7_env.sh
U2. resolve/freeze openai/gpt-oss-20b model contract
U3. inventory Harness-1 runtime in SCAPE
U4. integrate Harness1Bridge into EasyOPD
U5. pass Harness-1/EasyOPD acceptance
U6. replace 4-row synthetic formal collector
U7. build 2,000-query corpus-grounded TRAIN_POOL from 446 + >=1,554 validated new queries
U8. freeze DEV/TEST leakage exclusions and query provenance
U9. audit component realizability
U10. Stage A-E real Student on-policy collection
U11. dedup and freeze exactly 5,000 real event-active states
U12. write DATA_PROVENANCE / DATA_STATS
E1. Teacher metric
E2. canonical Student Before metric
E3. PURE_OPD seed42/43
E4. RL_PLUS_OPD seed42/43
E5. reload actual adapters
E6. DEV real closed-loop
E7. TEST real closed-loop
E8. aggregate two seeds
E9. paired bootstrap vs Before
E10. 20-case mechanism audit
E11. component handoff
```

当前仍然：

```text
DO NOT RUN SHUFFLED CAUSAL CONTROL
```

---

# 10. 主指标与通用判定

主指标：

```text
overall_reward
trajectory_recall
curated_evidence_recall
final_answer_recall
invalid_tool_rate
mean_turns
mean_tool_calls
```

没有 gold 的指标填 `N/A`，不能填 0。

判定：

```text
PASS_PURE_OPD:
  mean > Student Before
  seed42/43 同方向为正
  paired bootstrap CI 支持正效应

PASS_RL_PLUS_OPD:
  同上

PASS_BOTH:
  两条都过

FAIL_COMPONENT_INTERNALIZATION:
  两条都不能稳定超过 Student Before

NON_REALIZABLE:
  Harness effect 无 Student-native realization

INSUFFICIENT_5K_EVENT_SUPPORT:
  2,000 queries × 4 rollouts/query 后仍 <5000 unique event-active states

TEACHER_COMPONENT_NO_POSITIVE_UTILITY:
  Teacher <= Student Before
```

`TEACHER_COMPONENT_NO_POSITIVE_UTILITY` 原则上停止该组件正式蒸馏，但保留 Teacher / Before / event support 结果。

---

# 11. H100-4 专属：Control Components + Master Aggregation

本机组件：

```text
1. token_budget_marker
2. verify_tool
```

同时负责 10-component master aggregation。

## 11.1 `token_budget_marker`

分类预期：

```text
PRIVILEGED_CONTEXT / RUNTIME_ACCOUNTING
DIRECT 或 PARTIAL
```

先审计 Harness-1 中 marker 的真实语义：

```text
它只是精确 runtime bookkeeping？
还是会产生可从 Student-visible history 学到的 termination/continuation behavior？
```

Teacher：marker ON；Student：marker OFF。

只有 Teacher positive 且 event support >= 5K 才进入正式训练。

DIRECT 时：

```text
PURE_OPD = exact reverse KL
RL+OPD = GRPO + exact reverse KL
```

PARTIAL 时必须明确哪些信息不可 internalize，不能把精确 hidden counter 泄漏给 Student。

额外指标：

```text
termination_timing
late_step_waste
mean_tool_calls
mean_search_calls
reward_per_tool_call
```

## 11.2 `verify_tool`

这是明确的 action-space boundary：

```text
Teacher action space includes verify(doc_ids, claim)
Student action space has no verify interface
```

主协议：

```text
realizability = NON_REALIZABLE_ACTION_SPACE_MISMATCH
```

仍运行：

```text
Teacher metric
Student Before metric
event support
verify utilization
```

但：

```text
Student After PURE_OPD = N/A
Student After RL_PLUS_OPD = N/A
```

禁止：

```text
给 Student 偷加 verify tool
把 verify outcome 文本化为 Student privilege
用 route imitation 假装 internalization
```

可选 `INTERFACE_EQUALIZED_UPPER_BOUND` 只能写入 diagnostic，不能进入 10-component 主表。

## 11.3 H100-4 对 Phase U 的额外检查

聚合前必须确认四台机器共同引用：

```text
openai/gpt-oss-20b
同一个 Harmony adapter contract
同一个 setup_scape_easyopd_smoke7_env.sh
同一个 SCAPE commit
同一个 SCAPE-EasyOPD commit
同一个 2,000 TRAIN_POOL SHA
同一个 state_uid schema
同一个 selection_seed=20260818
```

如果任一机器仍显示：

```text
/opt/scape-easyopd-smoke7
synthetic collector
n_train_pool_unique_queries = 446
CANONICAL_STUDENT_BASE missing
Harness runtime unavailable
```

则：

```text
MASTER_TABLE_BLOCKED_PHASE_U_INCOMPLETE
```

不能聚合正式主表。

## 11.4 10-component master table

固定顺序：

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

输出：

```text
outputs/component_sweep_0818/master/COMPONENT_10_MAIN_TABLE.csv
outputs/component_sweep_0818/master/COMPONENT_10_MAIN_TABLE.md
outputs/component_sweep_0818/master/COMPONENT_10_FULL_METRICS.csv
outputs/component_sweep_0818/master/COMPONENT_10_DECISIONS.md
```

主表恰好 10 行，列至少：

```text
Component
Type
Realizability
Collection Status
Event Support (unique states)
Train Queries
Rollouts
Teacher Reward
Student Before Reward
Student After PURE_OPD Reward
Delta PURE vs Before
Student After RL+OPD Reward
Delta RL+OPD vs Before
Best After
Best Delta
Decision
```

`After` 填两个 seeds 的 mean，不填最好 seed。

## 11.5 Base consistency audit

检查：

```text
logical model id == openai/gpt-oss-20b
resolved revision/model config SHA
tokenizer SHA
Harmony adapter SHA
query manifest SHA
evaluator config SHA
reward config SHA
retriever config SHA
parser/tool runtime SHA
SCAPE commit
```

不一致：

```text
MASTER_TABLE_BLOCKED_BASE_MISMATCH
```

不得平均不同 Base。

## 11.6 Collector consistency audit

新增：

```text
collector_mode == real_harness1
synthetic_row_count == 0
TRAIN_STATES_5K unique(state_uid) == 5000 for every trained component
query pool source SHA identical
```

失败 row：

```text
INVALID_DATA_COLLECTION_CONTRACT
```

不得进入 paper-grade main result。

## 11.7 Teacher / loss consistency audit

Teacher：

```text
same base weights
only target component ON
all other components OFF
```

失败：`INVALID_TEACHER_ISOLATION`。

DIRECT：

```text
PURE = exact reverse KL
HYBRID = GRPO + exact reverse KL
```

PROJECTABLE：

```text
PURE = projected action CE + next-turn reverse KL
HYBRID = GRPO + same projected OPD
```

若出现 forward KL、route-head KL、不同 lambda/epochs 等：

```text
INVALID_LOSS_CONTRACT
```

## 11.8 最终结论问题

`COMPONENT_10_DECISIONS.md` 回答：

```text
1. 10 个组件中多少 Teacher > Before？
2. 多少组件可合法 internalize？
3. PURE_OPD 有多少 After > Before？
4. RL+OPD 有多少 After > Before？
5. RL+OPD vs PURE 的 win/tie/loss？
6. DIRECT / PROJECTABLE / PARTIAL 哪类最容易内化？
7. 哪些能力应该长期保留 runtime？
8. 是否存在 Teacher 有收益但 Student 无法内化的 placement boundary？
9. 是否有组件因真实 event support <5K 而无法形成 paper-grade training set？
```

## 11.9 本机输出

```text
outputs/component_sweep_0818/h100_4/
├── token_budget_marker/
├── verify_tool/
├── H1004_COMPONENT_ROWS.csv
├── H1004_COMPONENT_ROWS.json
├── H1004_COMPONENT_HANDOFF.json
└── SHA256SUMS

outputs/component_sweep_0818/master/
├── COMPONENT_10_MAIN_TABLE.csv
├── COMPONENT_10_MAIN_TABLE.md
├── COMPONENT_10_FULL_METRICS.csv
├── COMPONENT_10_DECISIONS.md
├── BASE_CONSISTENCY_AUDIT.md
├── COLLECTOR_CONSISTENCY_AUDIT.md
├── TEACHER_ISOLATION_AUDIT.md
├── LOSS_CONSISTENCY_AUDIT.md
├── RUN_MANIFEST.json
└── SHA256SUMS
```

最终只有在 10 行均有合法状态（numeric 或 N/A + explicit reason），并且 Phase U / Base / Collector / Teacher / Loss audits 全通过后，才能写：

```text
MASTER_COMPONENT_SWEEP_READY
```
