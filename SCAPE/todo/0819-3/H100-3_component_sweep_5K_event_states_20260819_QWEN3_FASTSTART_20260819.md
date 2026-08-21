# H100-3：Framework Owner / Privileged Context Sweep — Qwen3 Fast-Start OPD + Harness-1 Integration + 5K Protocol

# 0. 本轮状态：先解除 5K collection 阻断，再运行正式实验

上一轮正式 5K collection 的确被多个工程门槛阻断；其中 `CANONICAL_STUDENT_BASE` 本轮已明确解决：直接使用本地 `/mnt/songzijun/models/Qwen3-30B-A3B-Instruct-2507`。本轮目标是先把环境、真实 Harness-1 bridge 和 collector 配好，然后**尽快启动真实 OPD_PILOT**，同时继续推进 5K formal collection。

```text
BLOCKER-1  当前 TRAIN query 只有 446 个，低于正式 collector 的 query-min=1000
RESOLVED-2 CANONICAL_STUDENT_BASE 统一改为 /mnt/songzijun/models/Qwen3-30B-A3B-Instruct-2507
RESOLVED-3 旧 /opt/scape-easyopd-smoke7 路径废弃；统一 source repo 内 setup_scape_easyopd_smoke7_env.sh
BLOCKER-4  EasyOPD 环境中没有可导入的真实 Harness-1 / SCAPE runtime
BLOCKER-5  scripts/scape_component_opd.py collect 仍是 4 行 synthetic smoke collector
BLOCKER-6  因上述问题，当前没有合法 TRAIN_STATES_5K.jsonl；这是正确行为，不得补写伪造文件
```

## 0.1 本轮最高优先级：尽快打通一次真实 OPD 闭环

本轮不再把“5K 全部收满”作为第一次看到 OPD 结果的前置条件。采用两条并行轨道：

```text
Track P = OPD_PILOT：尽快验证 pipeline 是否闭环
Track F = FORMAL_5K：继续构造 paper-grade 5K 数据并跑正式主实验
```

`OPD_PILOT` 只用于工程/方法闭环验证，不进入最终主表。允许条件：

```text
- setup_scape_easyopd_smoke7_env.sh 已成功 source
- /mnt/songzijun/models/Qwen3-30B-A3B-Instruct-2507 已真实加载
- Harness1Bridge acceptance 已通过
- collector_mode == real_harness1
- synthetic_row_count == 0
- 某组件已获得 >= 256 unique real event-active states
```

一旦满足，立即对该组件启动：

```text
P1. 冻结最先获得的 256~512 unique real event states
P2. 跑 PURE_OPD seed=42 小预算训练
P3. 保存真实 adapter
P4. 从磁盘重新加载 adapter
P5. 在固定 DEV-smoke subset 上跑 Student Before vs Student After
P6. 输出 loss curve / reward delta / invalid_tool_rate / adapter reload proof
```

如果 pilot 能稳定训练、adapter 可 reload、DEV 行为发生可测变化，即认为“OPD pipeline engineering loop 已打通”；随后继续 Track F。若 pilot 失败，优先修 pipeline，不要继续大规模烧 5K collection/GPU。

正式论文结论仍只允许来自 `TRAIN_STATES_5K.jsonl` 和完整 seeds 42/43 协议。

---

本轮正式 paper-grade 任务仍分为两个阶段：

```text
Phase U  = UNBLOCK / integration / real 5K collection
Phase E  = formal experiment / training / evaluation
```

Phase E（paper-grade 正式训练）仍要求 Phase U 全部通过；但满足 0.1 的真实 runtime + real states 条件后，可以在 Phase U 中途先跑 `OPD_PILOT`。

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
export CANONICAL_STUDENT_BASE=/mnt/songzijun/models/Qwen3-30B-A3B-Instruct-2507
export SCAPE_ROOT=/mnt/songzijun/Capability_Evolution/SCAPE
export EASYOPD_ROOT=/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD
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

# 2. CANONICAL_STUDENT_BASE：统一切换到本地 Qwen3-30B-A3B-Instruct-2507

H100 上没有此前计划使用的旧 Base，因此本轮不再等待/下载旧模型；直接使用已经存在的本地 Qwen3。四台机器统一固定：

```text
CANONICAL_STUDENT_BASE = /mnt/songzijun/models/Qwen3-30B-A3B-Instruct-2507
logical_model_id        = Qwen3-30B-A3B-Instruct-2507
```

开始任何 rollout / teacher scoring / OPD training 前必须验证该目录是完整、可加载的本地模型目录：

```bash
export CANONICAL_STUDENT_BASE=/mnt/songzijun/models/Qwen3-30B-A3B-Instruct-2507
test -d "$CANONICAL_STUDENT_BASE"
test -f "$CANONICAL_STUDENT_BASE/config.json"
python - <<'PYMODEL'
from transformers import AutoConfig, AutoTokenizer
p = "/mnt/songzijun/models/Qwen3-30B-A3B-Instruct-2507"
cfg = AutoConfig.from_pretrained(p, trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(p, trust_remote_code=True)
print("MODEL_CONFIG_OK", getattr(cfg, "model_type", None))
print("TOKENIZER_OK", type(tok).__name__)
print("HAS_CHAT_TEMPLATE", bool(getattr(tok, "chat_template", None)))
PYMODEL
```

必须记录：

```text
logical_model_id = Qwen3-30B-A3B-Instruct-2507
resolved_model_path = /mnt/songzijun/models/Qwen3-30B-A3B-Instruct-2507
model_config_sha256 = <sha256 of config.json>
tokenizer_config_sha256 = <sha256 if file exists>
chat_template_sha256 = <sha256 of tokenizer chat_template serialization>
```

四台机器必须使用同一路径/同一份 config/tokenizer/chat-template contract。不得某台机器自动换成 Hugging Face 同名仓库或其他 revision。

## 2.1 Qwen3 加载与 prompt/tool contract

本轮完全删除旧模型专属的 serialization 依赖。Qwen3 必须使用本地 tokenizer 自带的原生 chat template / tool rendering contract；不得手工拼接一套与 tokenizer 不一致的 prompt 后继续声明为同一 Base。

EasyOPD bridge 至少提供一个等价的：

```text
Qwen3NativeChatAdapter
- build_context(...)
- render(...)
- tool schema / assistant action rendering
- response/tool span identification
- tokenizer consistency check
```

要求：

```text
- 优先调用 tokenizer 原生 chat template
- 不猜测不存在的 Qwen3 特殊 kwargs
- tool schema 与 SCAPE/Harness-1 parser 一致
- collector / teacher scorer / trainer / evaluator 使用完全相同 serialization
- 做 10-case deterministic serialization regression
```

失败则：

```text
STOP_QWEN3_CHAT_CONTRACT_FAILED
```

## 2.2 Student / Teacher / Student After

### Student Before OPD

```text
weights = /mnt/songzijun/models/Qwen3-30B-A3B-Instruct-2507
Harness-1 target component = OFF
all other target V8D components = OFF
student_inference_privilege = false
```

### Teacher

Teacher 仍然不是更大的模型，而是同一个本地 Qwen3 Base：

```text
weights = /mnt/songzijun/models/Qwen3-30B-A3B-Instruct-2507
target component = ON
all other V8D components = OFF
```

Teacher 只在 Student 实际访问到的同一个 state/prefix 上获得该 target component 的 privileged context / side effect / runtime signal。

### Student After OPD

```text
weights = /mnt/songzijun/models/Qwen3-30B-A3B-Instruct-2507 + trained adapter
target component = OFF
all other V8D components = OFF
student_inference_privilege = false
```

第一判据仍是：

```text
Student After > Student Before
```

## 2.3 单卡显存探针：先测，不允许暗中换模型

由于 Qwen3-30B-A3B-Instruct-2507 明显大于此前 smoke Base，正式并行调度前先做一次真实显存探针：

```text
1. 单 H100 完成 tokenizer/config load
2. 单 H100 完成一次真实 inference forward/generation
3. 用与正式训练相同的 LoRA/precision 做 1 optimizer-step train probe
4. 保存 peak allocated/reserved memory
```

若单卡无法安全训练：允许把一个 training cell 改为 2-GPU FSDP/TP/verl 已有的真实分布式方案并降低同机并发；**不允许**为了保持“1 GPU = 1 cell”而换小模型、silent quantization、截断模型层或使用 mock trainer。

输出：

```text
outputs/component_sweep_0818/preflight/QWEN3_MODEL_ACCEPTANCE.json
```

只有真实加载和至少一次训练 step 通过，才标记：

```text
QWEN3_BASE_READY
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
TEST-2 load /mnt/songzijun/models/Qwen3-30B-A3B-Instruct-2507 adapter contract
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

可以使用当前统一模型/已有生成器把 spec 转成自然语言 query，但 generator 不能发明 spec 之外的新事实。若使用本地 `/mnt/songzijun/models/Qwen3-30B-A3B-Instruct-2507` 生成 query，只把它当 query writer；最终是否保留仍由后面的 corpus validator 决定。

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
same Qwen3-30B-A3B-Instruct-2507 base loaded from the frozen local path
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
2. Qwen3 Student 根据当前真实 Student-visible context 生成 action
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
  --student-base /mnt/songzijun/models/Qwen3-30B-A3B-Instruct-2507 \
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
logical_model_id
resolved_model_path
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
logical_model_id == Qwen3-30B-A3B-Instruct-2507
resolved_model_path == /mnt/songzijun/models/Qwen3-30B-A3B-Instruct-2507
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
  "canonical_student_base": "/mnt/songzijun/models/Qwen3-30B-A3B-Instruct-2507",
  "logical_model_id": "Qwen3-30B-A3B-Instruct-2507",
  "environment_setup_script": "/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD/scripts/setup_scape_easyopd_smoke7_env.sh",
  "scape_root": "/mnt/songzijun/Capability_Evolution/SCAPE",
  "easyopd_root": "/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD",
  "harness_runtime": "Harness-1/SCAPE",
  "harness1_easyopd_ready": true,
  "qwen3_base_ready": true,
  "formal_collector": "real_harness1",
  "synthetic_fallback": false,
  "train_pool_unique_queries": 2000
}
```

建议由 H100-3/framework owner 更新：

```text
/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD/outputs/scape_easyopd/framework/H1003_SCAPE_EASYOPD_HANDOFF.json
```

H100-3 应在正式 2,000-query pool 完成之前先发布一个早期 handoff：

```json
{
  "status": "SCAPE_EASYOPD_PILOT_READY",
  "pilot_ready": true,
  "canonical_student_base": "/mnt/songzijun/models/Qwen3-30B-A3B-Instruct-2507",
  "harness1_easyopd_ready": true,
  "qwen3_base_ready": true,
  "real_collector_ready": true,
  "synthetic_fallback": false
}
```

H100-1/2/4 看到 `SCAPE_EASYOPD_PILOT_READY` 后就可以开始真实 collection，并在达到 256 unique states 后立即跑 `OPD_PILOT`；**不需要等待** 2,000-query pool。

注意：`OPD_PILOT` 可以在 `train_pool_unique_queries` 尚未达到 2000 时先跑，只要使用的 states 全部来自真实 Harness-1 rollout 且不含 synthetic；但正式 Phase E 仍必须满足完整 5K / query-pool / leakage audit gate。

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

# 9. 统一执行顺序：环境通过后立即启动真实 collection，并尽早插入 OPD_PILOT

任何机器按下面顺序推进；能并行的工作不要串行等待：

```text
U1. source setup_scape_easyopd_smoke7_env.sh
U2. verify/freeze /mnt/songzijun/models/Qwen3-30B-A3B-Instruct-2507 model + tokenizer/chat-template contract
U3. run Qwen3 inference + 1-step train memory probe
U4. inventory Harness-1 runtime in SCAPE
U5. integrate/consume the shared Harness1Bridge
U6. pass Harness-1/EasyOPD acceptance
U7. replace 4-row synthetic formal collector

PARALLEL-DATA:
D1. build 2,000-query corpus-grounded TRAIN_POOL from 446 + >=1,554 validated new queries
D2. freeze DEV/TEST leakage exclusions and query provenance

AS SOON AS REAL COLLECTION WORKS:
C1. start component-specific real Student on-policy collection immediately
C2. once >=256 unique event-active states exist for a component, launch OPD_PILOT seed42
C3. reload pilot adapter and run fixed DEV-smoke evaluation
C4. if pilot fails, fix pipeline before scaling collection/training

FORMAL:
F1. continue Stage A-E collection until >=5,000 unique event-active states/component
F2. dedup and freeze exactly 5,000 real event-active states
F3. write DATA_PROVENANCE / DATA_STATS
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

# 11. H100-3 专属：Framework Owner + Privileged-Context Components

本机组件：

```text
1. evidence_graph
2. sentence_compress
```

H100-3 同时是本轮 Phase U 的 framework owner。其他机器不得各自实现不同的 Harness bridge / collector。

## 11.1 H100-3 先完成 framework unblock，但不要让 2,000-query builder 阻塞第一次 OPD

本机按两个 milestone 推进。第一优先级是尽快让四台机器拥有同一个真实可训练 pipeline：

```text
Milestone A — PILOT_READY
1. source setup_scape_easyopd_smoke7_env.sh
2. load /mnt/songzijun/models/Qwen3-30B-A3B-Instruct-2507
3. SCAPE Harness-1 runtime inventory
4. Harness1Bridge
5. Qwen3 native chat/tool adapter regression
6. real closed-loop collector
7. collector acceptance（real_harness1 / no synthetic）
8. exact reverse-KL numeric regression
9. Qwen3 inference + 1-step LoRA train probe
10. update H1003_SCAPE_EASYOPD_HANDOFF.json -> SCAPE_EASYOPD_PILOT_READY
```

只要以下全部通过：

```text
HARNESS1_EASYOPD_READY
REAL_COLLECTOR_PILOT_READY
QWEN3_BASE_READY
QWEN3_CHAT_READY
FRAMEWORK_NUMERICS_READY
```

立即写 `SCAPE_EASYOPD_PILOT_READY`。H100-1/2/4 不再等待后续 query-pool 构造，可立刻用现有合法 TRAIN queries 开始真实 collection，并达到 256 states 后启动 OPD_PILOT。

随后 H100-3 与 pilot 并行完成：

```text
Milestone B — FORMAL_READY
11. 446 -> 2,000 TRAIN_POOL builder
12. query leakage audit
13. freeze 2,000-query manifest + SHA
14. formal collector/query-pool acceptance
15. update H1003_SCAPE_EASYOPD_HANDOFF.json -> SCAPE_EASYOPD_READY
```

只有 Milestone B 通过后，才能进入 paper-grade 5K Phase E；但这不应阻塞 OPD_PILOT。

## 11.2 `evidence_graph`

分类：

```text
PRIVILEGED_CONTEXT / DIRECT
```

Teacher：

```text
same Qwen3-30B-A3B-Instruct-2507 weights loaded from /mnt/songzijun/models/Qwen3-30B-A3B-Instruct-2507
evidence_graph ON
all other V8D OFF
```

Student：

```text
evidence_graph OFF
raw reduced state
```

5K event 定义必须来自 Harness-1 的真实 graph enrichment / bridge relation context 生成，而不是看到两个 doc 就手工标 `event=true`。

Teacher 只在 Student 当前 state/prefix 上构造 graph-privileged view。

PURE_OPD：

```text
exact token-level reverse KL
```

RL+OPD：

```text
GRPO + exact reverse KL
```

额外指标：

```text
bridge_entity_followup_rate
new_entity_discovery
repeated_search_rate
```

## 11.3 `sentence_compress`

分类：

```text
PRIVILEGED_CONTEXT / DIRECT
```

Teacher：

```text
compressed current search observations
```

Student：

```text
original noisy current observations
```

5K event row 必须同时关联：

```text
original Student-visible observation
compression input doc ids
compressed Teacher view
state_uid
```

必须写：

```text
SENTENCE_COMPRESS_INFORMATION_AUDIT.md
```

检查：

```text
same underlying retrieved docs
same environment state
no future observation
no gold/qrel leakage
compression only uses current observation
```

PURE_OPD：exact reverse KL。

RL+OPD：GRPO + exact reverse KL。

额外指标：

```text
useful_evidence_retention
redundant_search_rate
redundant_read_rate
context_tokens_consumed
```

## 11.4 数值核验

正式训练前再次运行：

```text
exact reverse KL brute-force reference
gradient finite-difference
BF16 vs FP32 sanity
response mask
tool span mask
Qwen3 chat serialization mask alignment
```

任一失败：

```text
STOP_FRAMEWORK_NUMERICS_REGRESSION
```

## 11.5 GPU 调度

以下 GPU 映射只是“单卡 1-step train probe 通过”时的 nominal schedule。若 2.3 的显存探针显示单 H100 不安全，立即改为 2-GPU/cell 或 verl 当前已支持的分布式配置，优先保证 `OPD_PILOT -> adapter reload -> DEV` 闭环；不得为了维持并发度改用更小 Base。

`OPD_PILOT` 是唯一提前训练例外：达到 256 unique real event states 后即可占用必要的 1–2 张 GPU。下面的 Wave/8-cell 映射只针对正式 5K 训练。

仅在 Phase U 完成且两个 component 各自 5K gate 通过后，才启动下面的正式 8-cell 训练：

```text
GPU0 evidence_graph PURE42
GPU1 evidence_graph PURE43
GPU2 evidence_graph HYBRID42
GPU3 evidence_graph HYBRID43
GPU4 sentence_compress PURE42
GPU5 sentence_compress PURE43
GPU6 sentence_compress HYBRID42
GPU7 sentence_compress HYBRID43
```


每个本机 trainable component 一旦跑过 `OPD_PILOT`，额外保存：

```text
OPD_PILOT/
├── PILOT_TRAIN_STATES.jsonl
├── PILOT_CONFIG.json
├── PILOT_TRAIN_LOG.jsonl
├── adapter/
├── ADAPTER_RELOAD_ACCEPTANCE.json
└── DEV_SMOKE_BEFORE_AFTER.json
```

这些文件不得替代正式 5K artifacts；它们只证明 pipeline 已经可以真实训练和 reload。

## 11.6 本机输出

Framework：

```text
outputs/scape_easyopd/framework/
├── HARNESS1_RUNTIME_INVENTORY.md
├── HARNESS1_EASYOPD_ACCEPTANCE.json
├── QWEN3_CHAT_ACCEPTANCE.json
├── FORMAL_COLLECTOR_ACCEPTANCE.json
├── QUERY_POOL_BUILD_STATS.json
├── H1003_SCAPE_EASYOPD_HANDOFF.json
└── SHA256SUMS
```

Component：

```text
outputs/component_sweep_0818/h100_3/
├── evidence_graph/
├── sentence_compress/
├── H1003_COMPONENT_ROWS.csv
├── H1003_COMPONENT_ROWS.json
├── H1003_COMPONENT_HANDOFF.json
└── SHA256SUMS
```
