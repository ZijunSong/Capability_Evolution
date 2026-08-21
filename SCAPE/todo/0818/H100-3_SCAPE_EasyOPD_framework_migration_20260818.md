# H100-3：基于 EasyOPD/verl 重构 SCAPE 的统一 Harness-Component OPD 框架

## 0. 任务目标

本任务不是继续跑某个具体组件的实验，而是停止依赖当前由 Coding Agent 零散搭建的 OPD trainer，将一套经过公开论文与开源实现验证的 OPD 基础设施改造成 **完全适配 SCAPE / Beyond Textual Privilege 的统一训练框架**。

选定底座：

```text
EasyOPD
upstream: https://github.com/lds-ustc/EasyOPD
paper:    https://arxiv.org/abs/2607.11012
backend:  verl
verl:     https://github.com/verl-project/verl
```

内部工程名：

```text
SCAPE-EasyOPD
```

本轮使用：

```text
Machine: H100-3
GPU:     8 × H100
```

H100-3 之前的 `RETRIEVAL_HYGIENE_BUNDLE` 已经完成并正式 discard，不要继续该实验分支。

---

# 1. 为什么选择 EasyOPD，而不是继续自建 trainer

先阅读 EasyOPD 论文、README、README_DEV，以及 verl 的 Agentic RL / Multi-turn / AgentLoop 文档，然后在：

```text
FRAMEWORK_SELECTION_AUDIT.md
```

中完成技术审计。

必须至少比较：

```text
EasyOPD
verl 原生 OPD
KDFlow
SOD
OpenRLHF
TRL/GKD
```

最终默认选择 EasyOPD，除非实际代码审核发现无法满足下面的硬约束。

## 1.1 EasyOPD 适配本项目的关键原因

EasyOPD 已经把 OPD 方法拆成显式 extension boundaries：

```text
loss
rollout metadata
reward
alignment
teacher-sidecar
```

SCAPE 中不同 Harness component 恰好不是同一种 supervision：

- 有的是 privileged context；
- 有的是新增 tool/argument；
- 有的是 Harness 自动 side-effect；
- 有的是可投影成 Student 原生 action；
- 有的是外部实时信息，原则上不应直接内化。

因此需要的是一个 **method-local / component-local supervision framework**，而不是一个固定 reverse-KL trainer。

EasyOPD 底层直接复用 verl，后者已经提供：

```text
distributed rollout
FSDP/Megatron
vLLM / SGLang
multi-turn agent loop
custom tool calls
reward
optimization
checkpointing
Ray worker orchestration
```

不要重新实现这些系统能力。

## 1.2 KDFlow 只作为第二候选

KDFlow 的优点：

```text
native on-policy KD
FSDP2 + SGLang
full/reverse/JS/adaptive KL
LoRA
chunked loss
multi-teacher
EMA self-teacher
```

但它更偏通用 language-model KD stack；本项目还需要 Harness state、tool loop、component event、state fork/restore 和 step-wise agentic supervision，因此将 KDFlow 作为 KD numerics / efficiency 的参考实现，而不是主框架。

因此：

```text
EasyOPD/verl = primary
KDFlow       = reference implementation for KD numerics / efficiency
SOD          = reference implementation for step-wise agentic OPD
```

不要把三套框架拼在一起。

---

# 2. 最重要的工程原则

## 2.1 不再自行重写已有基础设施

以下能力优先使用 EasyOPD/verl 原生实现：

```text
Ray distributed workers
rollout engine
FSDP
vLLM / SGLang
checkpoint
optimizer
gradient accumulation
actor weight sync
teacher sidecar
KL/logit infrastructure
multi-turn agent loop
tool scheduling
```

只有 SCAPE 特有逻辑放进新模块。

## 2.2 不直接魔改 verl core

新增代码优先放：

```text
easyopd/methods/scape_component_opd/
```

若不得不修改 `verl/`：

1. 必须是最小增量；
2. 不允许删除原逻辑；
3. 所有修改使用：

```python
# ============ [EasyOPD:SCAPE_COMPONENT_OPD] ============
...
# ============ [EasyOPD:SCAPE_COMPONENT_OPD END] ========
```

4. 在 `VERL_PATCH_AUDIT.md` 中逐文件解释原因；
5. 所有新增 config 必须有 safe default，关闭后行为与 upstream 一致。

## 2.3 不覆盖当前 SCAPE

当前：

```text
/mnt/songzijun/Capability_Evolution/SCAPE
```

只能读取、import、调用。

新框架单独放：

```text
/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD
```

Python/GPU 环境必须放 `/opt`：

```text
/opt/scape-easyopd
```

禁止在 `/mnt` 创建 torch/transformers/vLLM Python runtime。

---

# 3. Phase A：冻结 upstream 与环境

## A1. Clone

```bash
cd /mnt/songzijun/Capability_Evolution
git clone https://github.com/lds-ustc/EasyOPD.git SCAPE-EasyOPD
cd SCAPE-EasyOPD

git rev-parse HEAD | tee UPSTREAM_EASYOPD_SHA.txt
git submodule update --init --recursive
```

立刻记录：

```text
EasyOPD SHA
embedded verl SHA
recipe submodule SHA
CUDA
driver
Python
torch
transformers
vllm
sglang
ray
flash-attn
peft
```

写入：

```text
UPSTREAM_LOCK.md
UPSTREAM_LOCK.json
```

从这一刻开始，本轮实验禁止 `git pull` 漂移版本。

## A2. 安装隔离环境

优先参考 EasyOPD 官方：

```text
scripts/install_easyopd_env.sh
requirements.txt
```

但不要污染系统 Python。

目标：

```bash
python3.11 -m venv /opt/scape-easyopd
source /opt/scape-easyopd/bin/activate
```

随后按 upstream installer 安装。

若 installer 默认创建 conda 环境，允许修改 installer 使其安装到 `/opt/scape-easyopd`，但不要改变依赖版本逻辑。

安装后：

```bash
python - <<'PY'
import torch, transformers, peft, ray
import easyopd, verl
print(torch.__version__)
print(transformers.__version__)
print(peft.__version__)
print(ray.__version__)
print(torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

必须：

```text
8 GPUs visible
BF16 matmul pass on all 8
import easyopd pass
import verl pass
```

---

# 4. Phase B：先验证 upstream，不要马上改代码

必须运行：

```bash
python scripts/run_easyopd.py --list-methods
python scripts/run_easyopd.py --method gkd --dry-run
python scripts/run_easyopd.py --method sod --dry-run
python scripts/run_easyopd.py --method opcd --dry-run
```

然后运行 upstream unit tests 中与：

```text
distillation
loss
rollout
agent loop
tool
```

相关的测试。

若 upstream test 本身失败：

```text
STOP_UPSTREAM_ENV_BROKEN
```

先定位环境/版本问题，不允许一边 framework 自身坏着一边接 SCAPE。

输出：

```text
UPSTREAM_SMOKE.md
UPSTREAM_TEST_RESULTS.txt
```

---

# 5. Phase C：审计当前 SCAPE 自建 OPD

读取当前 SCAPE 中所有与 OPD 相关代码，至少包括：

```text
scripts/train_route_opd.py
scripts/eval_route_opd.py
scape/training/route_opd.py
scape/training/hf_tool_opd.py
PROJECTED_ACTION_AUTO 新 trainer / collector
real closed-loop evaluator
dual-view / component mask
Harmony/tool renderer/parser
fork/replay scripts
```

不要默认这些实现正确。

输出：

```text
LEGACY_OPD_CODE_AUDIT.md
```

逐项回答：

```text
1. rollout 是否真正来自 current Student？
2. teacher 是否在 Student state/prefix 上重新评分？
3. teacher/student token positions 是否严格对齐？
4. forward KL 定义是否正确？
5. reverse KL 定义和 gradient 是否正确？
6. response/tool-call mask 是否正确？
7. JSON argument token 是否真的进入 loss？
8. LoRA 是否 double-wrap？
9. base + SFT + OPD adapter merge/load 顺序是否正确？
10. shuffled control 是否保持 marginal/update budget？
11. query split 是否真正 disjoint？
12. state fork 是否是 same-xi_t？
13. teacher privilege 是否泄漏进 Student inference？
14. route proxy 是否被错误当成 real closed-loop？
15. parser/Harmony contract 是否统一？
```

发现 bug 必须记录，但本轮**不要优先修 legacy trainer**。

新框架将取代它。

---

# 6. Phase D：实现统一 `scape_component_opd`

新目录：

```text
easyopd/methods/scape_component_opd/
├── __init__.py
├── README.md
├── core.py
├── config.py
├── types.py
├── component_spec.py
├── component_registry.py
├── rollout_hook.py
├── teacher_sidecar.py
├── loss_hook.py
├── reward_hook.py
├── alignment_hook.py
├── state_snapshot.py
├── state_delta.py
├── action_projection.py
├── tool_span.py
├── controls.py
├── diagnostics.py
├── scape_agent_loop.py
├── losses/
│   ├── forward_kl.py
│   ├── reverse_kl.py
│   ├── jsd.py
│   ├── action_ce.py
│   ├── projected_action_ce.py
│   ├── next_turn_kl.py
│   ├── step_weighted_kl.py
│   └── hybrid_rl_opd.py
└── components/
    ├── verify_tool.py
    ├── importance_tagging.py
    ├── subtractive_curation.py
    ├── auto_populate_first_search.py
    ├── content_dedup.py
    ├── chunk_neighbors.py
    ├── evidence_graph.py
    ├── sentence_compress.py
    ├── token_budget_marker.py
    └── adaptive_rerank_instruction.py
```

EasyOPD registry 名：

```text
scape_component_opd
```

调用形式最终必须类似：

```bash
python scripts/run_easyopd.py \
  --method scape_component_opd \
  --config easyopd/config/scape_component_opd.yaml \
  component.name=auto_populate_first_search
```

以及：

```bash
python scripts/run_easyopd.py \
  --method scape_component_opd \
  --config easyopd/config/scape_component_opd.yaml \
  component.name=evidence_graph
```

**换组件不允许复制一套 trainer。**

---

# 7. `ComponentSpec`：整个框架的核心抽象

每一个 Harness component 必须实现同一个 typed contract。

建议 dataclass：

```python
@dataclass
class ComponentSpec:
    name: str

    # Placement / realizability
    effect_type: str
    realizability: str

    # Runtime
    enable_teacher_component: Callable
    disable_student_component: Callable
    event_detector: Callable

    # Views
    build_teacher_view: Callable
    build_student_view: Callable

    # State
    snapshot_state: Callable
    restore_state: Callable
    effect_extractor: Callable | None

    # Supervision
    projection_builder: Callable | None
    supervision_builder: Callable
    default_loss_mode: str

    # Validation
    visibility_validator: Callable
    action_schema_validator: Callable
    leakage_validator: Callable

    # Metrics
    mechanism_metrics: list[str]
```

---

# 8. 先明确“什么组件能以什么形式内化”

框架不能默认 10 个组件全都用 reverse KL。

为每个组件声明：

```text
DIRECT
PROJECTABLE
PARTIAL
NON_REALIZABLE
```

## 8.1 `verify_tool`

```text
effect_type    = ACTION_SPACE_CHANGE
realizability  = NON_REALIZABLE
```

Teacher 多了：

```text
verify(doc_ids, claim)
```

Student 连这个 interface 都没有。

默认行为：

```text
framework refuses training
```

必须输出：

```text
NON_REALIZABLE_ACTION_SPACE_MISMATCH
```

除非用户显式给 Student 增加同一 verify tool，否则不允许所谓“蒸馏 verify_tool”。

## 8.2 `importance_tagging`

```text
effect_type   = ARGUMENT_PRIVILEGE
realizability = PROJECTABLE/PARTIAL
```

Teacher 的 curate：

```text
curate(..., importance=...)
```

Student：

```text
curate(add_ids, remove_ids)
```

默认不要蒸馏 importance 数值。

应支持：

```text
teacher importance decision
  -> realized add/remove selection
  -> Student-native add_ids/remove_ids supervision
```

若没有 valid add/remove support：

```text
SUPPORT_GATE_FAIL
```

禁止 fabrication。

## 8.3 `subtractive_curation`

```text
effect_type   = AUTOMATIC_SIDE_EFFECT
realizability = PROJECTABLE
```

Harness：

```text
curated_ids_pre -> curated_ids_post
```

映射：

```text
add_ids    = post - pre
remove_ids = pre - post
```

只有在：

```text
all add ids visible to Student
all remove ids currently in Student curated set
```

时才是合法 projected action。

## 8.4 `auto_populate_first_search`

```text
effect_type   = AUTOMATIC_SIDE_EFFECT
realizability = PROJECTABLE
```

真实 effect：

```text
successful first search
-> harness adds top-K to curated set
```

投影：

```text
curate(add_ids=delta_add)
```

必须使用真实 pre/post state delta。

## 8.5 `content_dedup`

```text
effect_type   = AUTOMATIC_POOL_FILTER
realizability = PARTIAL
```

框架必须做到：

```text
event_support == 0
-> STOP_NO_ACTIVE_EVENT_SUPPORT
```

不能拿 inactive rows 训练。

若未来有真实 duplicate cluster：

- 可以监督 canonical read/curate choice；
- 不能让 Student “删除一个根本不可操作的 environment object”。

## 8.6 `chunk_neighbors`

```text
effect_type   = EXTERNAL_INFORMATION_AUGMENTATION
realizability = PARTIAL / RUNTIME_PREFERRED
```

Harness 会额外提供 Student 本来没有观察到的邻居 chunk。

默认：

```text
do not distill hidden neighbor content as if Student observed it
```

只有当可以映射成合法：

```text
read_document(...)
search_corpus(...)
```

行为序列，且所需 id/信息在 Student action space 可取得时，才允许 behavioral projection。

否则标记：

```text
KEEP_RUNTIME
```

## 8.7 `evidence_graph`

```text
effect_type   = PRIVILEGED_CONTEXT
realizability = DIRECT
```

Student rollout 用 reduced view。

Teacher 对 Student 当前 prefix/state 使用：

```text
full view + evidence_graph
```

重新评分相同 Student-generated tokens/actions。

默认 loss：

```text
token-level reverse KL
```

同时支持：

```text
forward KL
JSD
SOD step weighting
hybrid RL + OPD
```

## 8.8 `sentence_compress`

同样：

```text
PRIVILEGED_CONTEXT / DIRECT
```

Teacher 看压缩后 observation；Student 看原始 noisy observation。

必须保证：

```text
teacher and student refer to same underlying environment state
```

不能把未来信息压进去。

## 8.9 `token_budget_marker`

```text
PRIVILEGED_CONTEXT / DIRECT-PARTIAL
```

它包含精确 runtime accounting。

必须提供两个模式：

```text
semantic_internalization
runtime_anchor
```

默认实验模式：

```text
semantic_internalization
```

Student 无 marker，teacher 有 marker。

若结果依赖精确 token budget 数字，则应标记它更适合 runtime，不要强行内化。

## 8.10 `adaptive_rerank_instruction`

```text
PRIVILEGED_RETRIEVAL_CONTEXT / DIRECT or PARTIAL
```

必须区分：

```text
instruction-only effect
candidate-ranking effect
```

如果它改变了真实 retrieved documents/order，则 teacher/student state 不再完全相同。

此时不能简单 same-prefix KL。

需要记录：

```text
retrieval_delta
topK_overlap
qrel_recall_delta
```

并根据是否可在 Student action space 通过更好的 search query / curate behavior实现，选择 DIRECT 或 PROJECTABLE。

---

# 9. `SCAPEAgentLoop`：接入真实 Search Harness

不要把 SCAPE 的 Search runtime 重写一份。

实现：

```text
easyopd/methods/scape_component_opd/scape_agent_loop.py
```

优先继承 verl：

```python
AgentLoopBase
```

如果实际审核证明原生 `ToolAgentLoop` 足够，则复用它。

目标：

```text
SCAPE runtime remains source of truth
verl/EasyOPD only orchestrates rollout/training
```

必须支持 SCAPE 当前工具：

```text
fan_out_search
search_corpus
grep_corpus
read_document
review_docs
curate
verify       # only when tool exists in the current action space
end_search
```

---

# 10. 极重要：token-in-token-out，不得 decode/re-encode

自定义 AgentLoop 时坚持：

```text
never re-encode tokens you've decoded
```

这对本项目尤其关键，因为之前已经出现 Harmony/parser contract bug。

必须建立：

```text
SCAPE_TOKEN_CONTRACT.md
```

并测试：

```text
rollout token ids
teacher-rescore token ids
training token ids
tool-call span token ids
```

完全一致。

对于 gpt-oss：

```text
使用官方 Harmony build_context / render contract
```

对于 Harness-1/Qwen：

```text
使用模型自身官方 chat template
```

禁止手写字符串 parser 后再 tokenize 作为 paper-grade training path。

---

# 11. Dual View / Same-State Teacher

框架必须原生支持：

```text
Student view = reduced/no-privilege
Teacher view = same state + selected Harness component
```

Student 先 rollout。

在每个监督位置记录：

```text
query_id
trajectory_id
turn
state_hash
student_prefix_token_ids
student_action_token_ids
student_view_hash
teacher_view_hash
component_event
```

Teacher 必须针对 Student occupancy 提供 supervision。

不要先 rollout 一条独立 teacher trajectory，再叫它 OPD。

---

# 12. State Fork / Restore

实现标准接口：

```python
snapshot = state.snapshot()

student_branch = restore(snapshot)
teacher_branch = restore(snapshot)
```

必须 snapshot：

```text
working memory
documents
curated ids
curated importance
tool history
remaining budget
component masks
retrieval state
evidence state
verified state
```

生成：

```text
state_hash_pre
state_hash_student
state_hash_teacher
```

Same-state test 必须确认在执行 component effect 之前：

```text
student_branch.state_hash == teacher_branch.state_hash
```

---

# 13. 标准数据结构：`ComponentTransitionRecord`

统一保存为 parquet/jsonl。

至少字段：

```text
query_id
trajectory_id
turn_id
component_name
component_effect_type
realizability

student_checkpoint_sha
teacher_checkpoint_sha

state_hash_pre
student_view_hash
teacher_view_hash

student_prompt_token_ids
student_response_token_ids
response_mask
tool_span_mask
argument_span_mask

student_logprobs
teacher_logprobs
reference_logprobs

teacher_route_distribution      # optional diagnostic
teacher_action                  # if available

component_event_active
component_effect
projected_action
projection_valid
visibility_valid

reward_before
reward_after
trajectory_reward
terminal_reward

visible_doc_ids
curated_ids_pre
curated_ids_post

query_split
seed
```

不要把 reward/future labels 放进 teacher privileged input，除非某个 baseline 的论文定义明确需要，并单独标记。

---

# 14. Loss 实现：全部重新做数值单测

不要因为 EasyOPD/verl 已有 loss 就盲信。

必须为所有使用的 loss 写 PyTorch brute-force reference tests。

## 14.1 Exact Forward KL

实现：

```text
KL(P_teacher || P_student)
```

同 tokenizer 时默认 full-vocab exact。

测试：

```text
manual probability calculation
random logits
extreme logits
BF16
FP32 reference
gradient finite
gradient direction
```

## 14.2 Exact Reverse KL

实现：

```text
KL(P_student || P_teacher)
```

尤其检查：

```text
NO accidental detach of student log-ratio
```

必须做：

```text
autograd gradient vs brute-force exact reference
finite-difference check on tiny logits
```

若 top-k 近似版本存在：

```text
reverse_kl_topk_approx
```

必须单独命名，禁止和 exact reverse KL 混为一谈。

## 14.3 JSD

支持：

```text
alpha-JSD
```

用于 GKD/ablation。

## 14.4 Tool/action CE

支持 mask：

```text
tool_name
JSON syntax
argument key
argument value
doc ids
```

必须可配置：

```text
tool_only
args_only
tool_plus_args
```

## 14.5 Projected Action CE

用于：

```text
AUTO
subtractive
importance+subtractive
future projectable components
```

必须显式标记：

```text
on_policy_state = true
target_source = harness_effect_projection
```

科学上不要把它伪装成“teacher token KL”。

## 14.6 Next-turn KL

用于：

```text
projected action
-> execute
-> next state
-> teacher/student KL
```

支持 continuation closure。

## 14.7 SOD Step-Weighted OPD

复用 EasyOPD/SOD 的设计接口。

使任何 component 可以配置：

```yaml
step_weighting:
  type: sod
```

## 14.8 Hybrid RL + OPD

后续要比较：

```text
L_rl + L_opd
```

基于 verl 的 GRPO/PPO flow 支持：

```text
loss = lambda_rl * L_rl + lambda_opd * L_opd
```

必须分别 log：

```text
L_rl
L_opd
total_loss
gradient_norm
KL
reward
```

不要把 RL advantage 和 OPD advantage 混成不可审计的单变量。

---

# 15. Teacher modes

框架统一支持：

```text
same_weights_privileged_view
frozen_external_teacher
ema_teacher
full_harness_runtime_teacher
```

当前 Beyond Textual Privilege 主线默认：

```text
same_weights_privileged_view
```

即：

```text
same model family / same weight initialization
Student = reduced view
Teacher = selected privileged Harness component
```

若具体实验使用别的 teacher，YAML 必须明确写出。

---

# 16. Reference / anchor policy

支持：

```text
none
frozen_init
ema
explicit_checkpoint
```

不要硬编码 `lambda_anchor=0.05`。

YAML：

```yaml
reference:
  mode: frozen_init
  coef: 0.05
```

---

# 17. Component YAML：以后逐组件只改配置

创建：

```text
easyopd/config/scape/
├── base.yaml
├── verify_tool.yaml
├── importance_tagging.yaml
├── subtractive_curation.yaml
├── auto_populate_first_search.yaml
├── content_dedup.yaml
├── chunk_neighbors.yaml
├── evidence_graph.yaml
├── sentence_compress.yaml
├── token_budget_marker.yaml
└── adaptive_rerank_instruction.yaml
```

示例：

```yaml
method:
  name: scape_component_opd

component:
  name: evidence_graph
  realizability: DIRECT
  event_only: true

teacher:
  mode: same_weights_privileged_view
  component_enabled: true

student:
  component_enabled: false
  inference_privilege: false

rollout:
  policy: current_student
  multi_turn: true
  max_steps: 8

distillation:
  loss: reverse_kl
  token_mask: response
  step_weighting: none

reference:
  mode: frozen_init
  coef: 0.05

controls:
  shuffled_target: true
  matched_text: false

evaluation:
  real_closed_loop: true
```

---

# 18. 组合组件也要原生支持，但不能简单 stack LoRA

YAML 支持：

```yaml
component:
  names:
    - importance_tagging
    - subtractive_curation
```

以及：

```yaml
component:
  names:
    - auto_populate_first_search
    - content_dedup
```

组合时：

```text
重新 collect on-policy data
重新 teacher sidecar
重新训练
```

禁止：

```text
LoRA A + LoRA B 简单 stack
```

然后称其为 component composition。

---

# 19. Event-conditioned collector

所有 component 都必须先统计 event support。

输出：

```text
EVENT_SUPPORT.csv
```

至少：

```text
n_queries
n_states
n_event_active
event_rate
n_projectable
n_valid_args
n_terminal_reward
```

Gate：

```text
event support too low
-> do not train
```

不要再发生 `content_dedup` 0 trigger 仍然训练完整矩阵的情况。

---

# 20. Controls 必须成为框架一级对象

统一支持：

## 20.1 Shuffled target

保持：

```text
same states
same update budget
same target marginal
```

破坏：

```text
state-target pairing
```

对于 doc-id arguments，shuffle 后必须仍然合法。

## 20.2 Matched Text

将同一 structured privilege deterministic textualize。

必须做：

```text
round-trip information equivalence audit
```

不能额外加入 reasoning/future info。

## 20.3 Event heuristic

例如 AUTO：

```text
first search -> top-K curate
```

作为非 learned mechanism baseline。

## 20.4 Standard OPD

同一 Student rollout：

```text
teacher logits
reverse/forward KL
```

作为基础 baseline。

---

# 21. Diagnostics：不能只看 loss

每个 method/component 至少输出：

```text
train loss
grad norm
teacher-student KL
event support
supervised token count
tool-span supervised token count
argument-span supervised token count
projection validity
visibility validity
invalid-tool rate
teacher/student action agreement
```

组件机制指标：

### AUTO

```text
search_to_curate_delay
immediate_curate_rate
relevant_added_rate
```

### subtractive

```text
valid_remove_rate
irrelevant_removed_rate
curated_churn
```

### dedup

```text
duplicate_trigger_rate
duplicate_read_rate
duplicate_curate_rate
```

### evidence graph

```text
bridge-entity search rate
new-entity discovery
```

### token budget

```text
termination timing
tool calls
late-step waste
```

---

# 22. Real closed-loop evaluator 只能有一套 contract

把当前 SCAPE paper-grade evaluator 包装为：

```text
SCAPERealClosedLoopEvaluator
```

以后所有：

```text
Base
Ours
shuffled
matched text
OPSD
OPHSD
SEED/OPID
```

必须走同一个 evaluator contract。

统一：

```text
query manifest
retriever
max_steps
reward
termination
parser
tool runtime
final answer scoring
```

route proxy 只能：

```text
diagnostic=true
```

禁止：

```text
recommended_for_main_table=true
```

---

# 23. Tool-call / Harmony contract tests

必须针对当前项目历史 bug 写 regression tests。

至少：

```text
test_gpt_oss_harmony_roundtrip
test_harness1_qwen_tool_roundtrip
test_curate_add_ids_span
test_curate_remove_ids_span
test_end_search_span
test_invalid_tool_rejected
test_tool_schema_matches_runtime
test_no_double_peft_wrap
test_clean_sft_then_opd_adapter_load_order
```

特别是：

```text
tool span parsable rate = 100%
```

任何 `parsable_rate=0` 的 bridge 类结果必须在新框架中直接 hard fail。

---

# 24. On-policy correctness tests

必须写：

```text
test_rollout_checkpoint_equals_current_student
test_teacher_scores_student_prefix
test_no_independent_teacher_rollout_for_opd
test_same_state_before_component_fork
test_student_inference_has_no_privilege
test_query_disjoint_split
test_shuffle_preserves_marginal
```

任何一项失败：

```text
PAPER_GRADE=false
```

---

# 25. No-leakage tests

对于 privileged context：

```text
teacher-only field
```

必须检查没有出现在：

```text
student prompt
student runtime state
student inference config
student evaluation config
```

Projected action：

```text
all doc ids visible
all remove ids legal
no gold/qrel used in target generation
```

---

# 26. Phase E：四类代表性组件做 framework acceptance test

本轮不是正式 paper experiment。

只验证框架可以覆盖四种内化类型。

## E1. DIRECT：`evidence_graph` 或 `token_budget_marker`

选择 event support 更高的一个。

跑：

```text
16 real queries
current Student on-policy rollout
Teacher privileged re-score
1 tiny update
real closed-loop smoke
```

证明：

```text
teacher sidecar works
token KL works
Student inference no privilege
```

## E2. PROJECTABLE：`auto_populate_first_search`

复用当前真实 projection schema，但由新框架重新 collect 少量数据。

跑：

```text
16-32 real queries
projected curate action
tool-span CE
next-turn KL smoke
```

证明：

```text
state delta
-> legal action
-> loss
-> model update
-> actual reload
```

## E3. ARGUMENT PRIVILEGE：`importance_tagging`

只做 schema/support smoke。

如果没有足够真实 valid add/remove：

```text
SUPPORT_GATE_FAIL
```

这本身视为框架正确行为。

## E4. NON_REALIZABLE：`verify_tool`

执行：

```text
component.name=verify_tool
```

必须自动拒绝训练并返回：

```text
NON_REALIZABLE_ACTION_SPACE_MISMATCH
```

这也是 acceptance test。

---

# 27. 8×H100 acceptance resource plan

本阶段不追最终效果。

建议：

```text
GPU0-1: DIRECT component smoke
GPU2-3: AUTO projected-action smoke
GPU4:   loss/numerics GPU tests
GPU5:   multi-turn/Harmony regression
GPU6:   real closed-loop evaluator smoke
GPU7:   reserved for Ray/teacher-sidecar debugging
```

如果 verl 默认会动态占用全部 8 卡，则按其 placement group 正常运行，不要强行拆卡。

---

# 28. Acceptance criteria

本任务只有满足以下全部条件才算完成：

```text
[ ] EasyOPD upstream installation/test pass
[ ] 8 H100 usable
[ ] SCAPEAgentLoop can run real multi-turn Search
[ ] Student rollout is truly on-policy
[ ] Same-state dual view works
[ ] Teacher sidecar works
[ ] Full/reverse KL numeric tests pass
[ ] Reverse KL gradient reference test pass
[ ] Tool-call span mask test pass
[ ] Projected action visibility test pass
[ ] Student inference privilege=false
[ ] actual LoRA checkpoint reload pass
[ ] no double PEFT wrap
[ ] real closed-loop evaluator pass
[ ] DIRECT component smoke pass
[ ] PROJECTABLE component smoke pass
[ ] NON_REALIZABLE component correctly rejected
[ ] event support zero correctly blocks training
[ ] query-disjoint split test pass
[ ] shuffled control contract test pass
```

只有全部通过：

```text
SCAPE_EASYOPD_READY
```

否则：

```text
SCAPE_EASYOPD_NOT_READY
```

并明确 blocking issue。

---

# 29. 不要在本轮做的事

不要：

```text
1. 追求论文最好 reward
2. 扫几十个 seed
3. 重新跑 RETRIEVAL_HYGIENE_BUNDLE
4. 重新救 importance_tagging
5. 把 route-head 接进新框架当主模型
6. 复制旧 trainer 后改名字叫 EasyOPD
7. 为了跑通而 mock Search/Harness
8. 用 synthetic data 替代最终 acceptance
9. 修改已有 paper-grade artifact
10. 宣称新框架已经证明某组件有效
```

本轮目标是：

```text
TRUSTWORTHY INFRASTRUCTURE
```

不是：

```text
POSITIVE SCIENTIFIC RESULT
```

---

# 30. 最终以后逐组件运行的标准流程

框架完成后，每个 component 都必须按完全相同的 pipeline：

```text
1. ComponentSpec audit
2. realizability classification
3. event support collection
4. same-state/value diagnostic
5. on-policy Student rollout
6. teacher sidecar / effect projection
7. matched controls
8. actual model training
9. checkpoint reload
10. real closed-loop DEV
11. real closed-loop TEST
12. paired bootstrap
13. mechanism case analysis
14. GO / REDESIGN / DISCARD
```

不能再每个 component 临时写一套脚本。

---

# 31. 标准运行 CLI

最终至少支持：

```bash
# 查看组件
python scripts/scape_component_opd.py list-components

# 审计某组件
python scripts/scape_component_opd.py audit \
  --component auto_populate_first_search

# 收集
python scripts/scape_component_opd.py collect \
  --component auto_populate_first_search \
  --config easyopd/config/scape/auto_populate_first_search.yaml

# 训练
python scripts/scape_component_opd.py train \
  --component auto_populate_first_search \
  --loss projected_action_ce

# closed-loop
python scripts/scape_component_opd.py eval \
  --component auto_populate_first_search \
  --split dev

# 一键 paper-grade pipeline
python scripts/scape_component_opd.py run \
  --component evidence_graph \
  --config easyopd/config/scape/evidence_graph.yaml
```

所有 CLI 必须支持：

```text
--dry-run
--seed
--output-dir
--resume
```

---

# 32. 输出目录

统一：

```text
outputs/scape_easyopd/
├── framework/
├── audits/
├── tests/
├── acceptance/
└── components/
    ├── auto_populate_first_search/
    ├── importance_tagging/
    ├── subtractive_curation/
    ├── content_dedup/
    ├── chunk_neighbors/
    ├── evidence_graph/
    ├── sentence_compress/
    ├── token_budget_marker/
    ├── adaptive_rerank_instruction/
    └── verify_tool/
```

每个正式实验必须自行创建 versioned run directory，不允许覆盖。

---

# 33. 必须产出的 artifacts

```text
UPSTREAM_EASYOPD_SHA.txt
UPSTREAM_LOCK.md
UPSTREAM_LOCK.json

FRAMEWORK_SELECTION_AUDIT.md
UPSTREAM_SMOKE.md
UPSTREAM_TEST_RESULTS.txt
LEGACY_OPD_CODE_AUDIT.md
VERL_PATCH_AUDIT.md

SCAPE_EASYOPD_ARCHITECTURE.md
SCAPE_COMPONENT_SPEC.md
SCAPE_TOKEN_CONTRACT.md
SCAPE_DATA_CONTRACT.md
SCAPE_REAL_CLOSED_LOOP_CONTRACT.md

COMPONENT_REALIZABILITY_MATRIX.csv
COMPONENT_REALIZABILITY_MATRIX.md

LOSS_NUMERICS_TEST.md
TOOL_SPAN_TEST.md
ON_POLICY_CONTRACT_TEST.md
LEAKAGE_TEST.md
STATE_FORK_TEST.md

ACCEPTANCE_DIRECT.md
ACCEPTANCE_PROJECTABLE.md
ACCEPTANCE_ARGUMENT_PRIVILEGE.md
ACCEPTANCE_NON_REALIZABLE.md
ACCEPTANCE_REAL_CLOSED_LOOP.md

RUN_MANIFEST.json
STATUS_LIVE.md
H1003_SCAPE_EASYOPD_HANDOFF.json
SHA256SUMS
```

---

# 34. `COMPONENT_REALIZABILITY_MATRIX` 最低要求

最终必须得到类似：

| Component | Effect type | Student-native realizability | Default training mode | Current support | Decision |
|---|---|---|---|---|---|
| verify_tool | action-space change | NON_REALIZABLE | none | N/A | keep runtime |
| importance_tagging | argument privilege | PARTIAL/PROJECTABLE | projected args | audit | gated |
| subtractive_curation | auto state effect | PROJECTABLE | projected action | audit | gated |
| auto_populate_first_search | auto state effect | PROJECTABLE | projected action + optional next-turn KL | available | trainable |
| content_dedup | pool filter | PARTIAL | event-conditioned behavior | currently zero trigger | block |
| chunk_neighbors | external info | PARTIAL | action-sequence projection if legal | audit | runtime preferred |
| evidence_graph | privileged context | DIRECT | token KL | audit | candidate |
| sentence_compress | privileged context | DIRECT | token KL | audit | candidate |
| token_budget_marker | privileged context/runtime accounting | DIRECT/PARTIAL | token KL | audit | candidate/runtime |
| adaptive_rerank_instruction | retrieval privilege | DIRECT/PARTIAL | KL or projected retrieval behavior | audit | candidate |

不要机械照抄这张表。

必须依据真实代码和事件数据校正。

---

# 35. 代码质量要求

必须：

```text
ruff
type hints
pytest
deterministic config serialization
SHA256
structured JSON handoff
```

禁止：

```text
silent fallback
except Exception: pass
hard-coded GPU ids
hard-coded absolute training output
magic global state
```

所有 fallback 必须：

```text
log warning
record in manifest
paper_grade=false if it changes scientific contract
```

---

# 36. 最终 handoff

`H1003_SCAPE_EASYOPD_HANDOFF.json` 至少：

```json
{
  "status": "SCAPE_EASYOPD_READY",
  "easyopd_sha": "...",
  "verl_sha": "...",
  "environment": "...",
  "gpu_count": 8,
  "upstream_tests_pass": true,
  "loss_numeric_tests_pass": true,
  "on_policy_contract_pass": true,
  "tool_span_contract_pass": true,
  "state_fork_pass": true,
  "no_privilege_inference_pass": true,
  "real_closed_loop_smoke_pass": true,
  "direct_component_smoke_pass": true,
  "projectable_component_smoke_pass": true,
  "non_realizable_guard_pass": true,
  "ready_components": [],
  "blocked_components": [],
  "recommended_next_component": null
}
```

最后根据真实 audit 给出：

```text
recommended_next_component
```

优先选择：

```text
event support 足够
effect realizable
teacher signal 非退化
能做 actual-model closed-loop
```

不要因为历史上某组件“看起来 promising”就固定推荐它。

---

# 37. 本轮最终判定

只有两个合法结论：

```text
SCAPE_EASYOPD_READY
```

或：

```text
SCAPE_EASYOPD_NOT_READY
```

如果 NOT READY：

必须精确指出：

```text
framework bug
environment bug
SCAPE agent-loop incompatibility
tokenization/tool parser incompatibility
state fork incompatibility
teacher-sidecar incompatibility
loss numerics bug
```

不要用 route proxy 或某个 smoke reward 来掩盖基础设施问题。
