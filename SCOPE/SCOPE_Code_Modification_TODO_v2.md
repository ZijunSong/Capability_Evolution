# SCOPE 代码重构 TODO（基于新版方法与当前实验结论）

> 适用仓库：`/data/ppnm/SCOPE`  
> 上游继承：`harness-1-upstream/` → `/data/ppnm/BiSHOP` → `/data/ppnm/SCOPE`  
> 本文目标：将当前已经打通的  
> `DecisionState → typed shadow → Endorse/Correct → OPDTransitionV2 → RL+OPD`  
> 重构为新版 SCOPE：
>
> **Capability decomposition → information-safe same-state supervision → verified local correction → recovery state augmentation → capability retirement**
>
> 当前实验结论必须作为实现约束：
>
> - `Duplicate Evidence`：Go25 Precision / Recall = **1.00 / 1.00**，可进入第一轮正式训练；
> - `Premature Stop`：Go25 Precision / Recall = **1.00 / 0.96**，可进入第一轮正式训练；
> - `Irrelevant Evidence`：当前 GT 与 shadow 同源，存在 circular labeling，**第一轮训练必须禁用**；
> - `Invalid Citation`：样本量过少，暂不作为主训练 capability；
> - bare terminal replay 几乎只有 `CORRECT`，因此主数据必须继续来自 **online multi-step DecisionState**；
> - Full Harness v1/v2 当前绝对任务表现较弱，现阶段首先把它视为 **状态/能力监督来源与系统对照**，不能把 Full Harness reward 当成唯一“教师正确性”依据；
> - 尚无真实 BrowseComp+ 上的正式 SCOPE 训练增益，因此所有重构应优先服务于“尽快得到第一个可信训练闭环”。

---

# 0. 首先冻结新版 SCOPE 的工程边界

## 0.1 继续保留的主干

以下代码和设计已经通过 smoke / audit 验证，不应推倒重写：

```text
SlidingWindowSearchEnv
    ↓
Student rollout
    ↓
DecisionState
    ↓
CriticalStateSelector
    ↓
Typed Shadow Module
    ↓
PrivilegedArtifact
    ↓
audit / verifier / telemetry
```

重构重点放在：

1. `PrivilegedArtifact` 的语义与 provenance；
2. Endorse / Correct 后的数据协议；
3. action-level training loss；
4. information-safe gate；
5. capability statistics；
6. recovery branch；
7. inference retirement evaluation。

---

## 0.2 明确不恢复 BiSHOP 的两部分

### [ ] 不解冻旧 `harness/lifecycle/`

新版 SCOPE 的 `module retirement` **不是 Harness–Model co-evolution**。

旧 BiSHOP lifecycle 的语义是：

```text
model 改进
  → harness 更新
  → 再训练 model
  → harness/model 双向演化
```

新版 SCOPE 只需要：

```text
固定 harness capability
  → 测量 model 是否已经内化
  → inference 时关闭 / 简化对应认知 capability
```

因此：

- `harness/lifecycle/` 保持 legacy；
- `train_coevolution_round.py` 保持冻结；
- 不允许模型训练结果反向修改 Harness prompt / graph / module behavior；
- 新增轻量 `capability retirement evaluator`，只负责**评测与生成部署 manifest**。

### [ ] 不恢复旧 M4 Recovery

新版 `Recovery-on-Demand` **不是运行时 Recovery Harness module**。

两者必须严格区分：

```text
BiSHOP M4 Recovery:
    deployment/runtime capability
    负责真实 Agent 执行过程中的回滚、修复

SCOPE recovery branch:
    training-only state augmentation
    从学生真实访问的 DecisionState fork
    只执行一次 verified corrective action
    再交还学生继续 K 步
```

建议代码命名统一使用：

```text
recovery_branch
recovery_rollout
recovery_state
```

不要使用：

```text
RecoveryModule
M4
runtime_recovery
```

---

# 1. 代码结构目标

建议把当前 SCOPE 专属代码最终整理为：

```text
SCOPE/
├── harness/
│   ├── capability/
│   │   ├── decision_state.py
│   │   ├── action_space.py
│   │   ├── selector.py
│   │   ├── adapters.py
│   │   ├── capability_id.py              # NEW
│   │   └── distillability.py             # NEW
│   │
│   ├── shadow/
│   │   ├── base.py
│   │   ├── registry.py
│   │   ├── evidence_shadow.py
│   │   ├── verification_shadow.py
│   │   ├── budget_shadow.py
│   │   └── action_realizer.py             # NEW
│   │
│   ├── artifacts/
│   │   ├── schema.py
│   │   ├── reason_codes.py
│   │   ├── visibility.py
│   │   ├── validators.py
│   │   └── provenance.py                  # NEW
│   │
│   └── telemetry/
│       ├── events.py
│       ├── writer.py
│       └── state_hash.py                  # NEW
│
├── training/
│   ├── train_scope.py
│   ├── scope/                             # NEW：新版 SCOPE 主训练逻辑
│   │   ├── schema.py
│   │   ├── dataset_builder.py
│   │   ├── routing.py
│   │   ├── losses.py
│   │   ├── weighting.py
│   │   ├── capability_stats.py
│   │   ├── internalization.py
│   │   ├── recovery_branch.py
│   │   └── sampler.py
│   │
│   ├── opd_v2/                            # legacy / compatibility
│   ├── audit_scope_shadow_bare.py
│   ├── audit_scope_chat_online.py
│   ├── offline_relabel_audit.py
│   ├── evaluate_audit_go_nogo.py
│   ├── build_scope_dataset.py             # NEW
│   ├── probe_distillability.py            # NEW
│   └── evaluate_scope_internalization.py  # NEW
│
├── inference/
│   └── scope/
│       ├── evaluate_minimal_runtime.py
│       ├── evaluate_capability.py          # NEW
│       ├── retirement_eval.py              # NEW
│       └── runtime_manifest.py             # NEW
│
├── configs/
│   └── scope/
│       ├── sdi_dup_premature.yaml          # NEW，第一轮主配置
│       ├── sdi_uniform.yaml                # NEW
│       ├── sdi_adaptive.yaml               # NEW
│       ├── sdi_recovery.yaml               # NEW
│       ├── distillability_probe.yaml        # NEW
│       ├── retirement_eval.yaml             # NEW
│       ├── dual_mode.yaml                  # LEGACY baseline
│       └── ...
│
└── artifacts/
    ├── baselines/
    ├── capability/
    │   ├── distillability_map.json         # NEW
    │   ├── internalization.json            # NEW
    │   └── retirement_manifest.json        # NEW
    └── datasets/
        └── scope_v3/
```

---

# 2. P0：先完成 Phase 0 和实验冻结

> 优先级：**BLOCKER**  
> 在正式训练前完成。否则训练后没有可信的 before/after。

## [ ] P0.1 跑完 Harness rollout v2

当前：

```text
modules_full_v2.yaml
≈463 / 830 completed
```

需要完成：

```text
830 / 830
```

落盘至少包括：

```text
episodes.jsonl
events.jsonl
summary.json
resolved_config.yaml
errors.jsonl
```

### Acceptance

- episode 数 = 830；
- errors 单独统计；
- 所有 summary 指标可由原始 episode 重算；
- 保存模型、retriever、index、Harness config、git commit。

---

## [ ] P0.2 正式冻结以下 baseline

至少生成：

```text
artifacts/baselines/
├── bare_metrics.json
├── minimal_runtime_metrics.json
├── full_harness_v1_metrics.json
├── full_harness_v2_metrics.json
├── manifest.json
└── README.md
```

`manifest.json` 必须记录：

```json
{
  "model": "Qwen2.5-7B-Instruct",
  "dataset": "BrowseComp+",
  "n_queries": 830,
  "retriever": "BM25",
  "index_version": "...",
  "max_turns": 35,
  "git_commit": "...",
  "configs": {
    "bare": "...",
    "minimal": "...",
    "full_v1": "...",
    "full_v2": "..."
  }
}
```

### 注意

Full Harness 当前弱不影响冻结。

Phase 0 的目的不是声称 Harness 很强，而是：

> 后续所有代码修改和训练至少不能让已有执行能力出现无法解释的 regression。

---

## [ ] P0.3 固定第一版 paper capability set

新建：

```text
harness/capability/capability_id.py
```

第一轮只允许：

```python
DUPLICATE_EVIDENCE
PREMATURE_STOP
```

显式禁用：

```python
IRRELEVANT_EVIDENCE
INVALID_CITATION
```

建议 capability ID 与 module ID 分离：

```text
module=evidence_state
capability=duplicate_evidence

module=budget_control / verification_control
capability=premature_stop
```

原因：

> 新版 SCOPE 的最小分析单位应是 **capability-bearing decision**，而不是整个模块。

未来 `M1 Evidence` 内可能同时存在：

```text
duplicate_evidence
evidence_prioritization
subtractive_curation
missing_primary_source
```

这些能力的可靠性和 internalization 不一定相同。

---

# 3. P1：升级 DecisionState 为可审计状态协议

> 优先级：**BLOCKER**

当前 `DecisionState` 已经能驱动 online audit，但新版 SCOPE 需要更严格的 provenance。

## [ ] P1.1 增加 `DecisionStateV2`

修改：

```text
harness/capability/decision_state.py
```

建议至少包含：

```json
{
  "schema_version": "scope.decision_state.v2",

  "episode_id": "...",
  "event_id": "...",
  "turn": 12,

  "goal": "...",
  "active_subgoal": "...",

  "supported_claims": [],
  "unsupported_claims": [],
  "conflicting_claims": [],

  "candidate_evidence_ids": [],
  "curated_evidence_ids": [],
  "observed_ids": [],

  "last_action_type": "curate",
  "last_action_arguments": {},
  "last_query": "...",

  "repeated_query_count": 0,

  "remaining_search_calls": 3,
  "remaining_open_calls": 2,
  "remaining_turns": 8,

  "student_action": {
    "type": "...",
    "arguments": {}
  }
}
```

---

## [ ] P1.2 所有字段必须标记来源

新增：

```text
harness/artifacts/provenance.py
```

把字段分为：

```text
OBSERVED
RUNTIME
DERIVED_VISIBLE
PRIVILEGED_FORBIDDEN
```

例如：

```text
remaining_turns          → RUNTIME
curated_evidence_ids     → OBSERVED / runtime state
repeated_query_count     → DERIVED_VISIBLE
hidden_relevance_label   → PRIVILEGED_FORBIDDEN
gold_answer              → PRIVILEGED_FORBIDDEN
```

### Acceptance

必须能够自动检查：

```text
Info(DecisionState) ⊆ Info(student runtime state)
```

不能仅靠人工约定。

---

## [ ] P1.3 增加 DecisionState hash

新增：

```text
harness/telemetry/state_hash.py
```

对以下内容计算 canonical hash：

```text
WorkingMemory
observed_ids
evidence state
budget counters
student-visible history
```

用途：

1. 检查 shadow 前后状态是否改变；
2. recovery fork 后检查恢复点一致；
3. offline replay 检查是否真的回到同一状态。

---

# 4. P2：把 PrivilegedArtifact 升级成 LocalDecisionArtifact

> 优先级：**BLOCKER**

名字可以继续保留 `PrivilegedArtifact` 兼容旧代码，但 schema 语义必须升级。

## [ ] P2.1 Artifact V3

建议 schema：

```json
{
  "schema_version": "scope.artifact.v3",

  "module_id": "evidence_state",
  "capability_id": "duplicate_evidence",

  "target": "obs_17",
  "diagnosis": "duplicate",

  "recommended_operation": "skip_curate",
  "operation_args": {},

  "evidence_ids": ["obs_17", "obs_8"],
  "runtime_fields_used": [],

  "reason_code": "semantic_duplicate",
  "confidence": 0.98,

  "teacher_source": "EvidenceShadow"
}
```

对 `PREMATURE_STOP`：

```json
{
  "module_id": "budget_control",
  "capability_id": "premature_stop",

  "diagnosis": "insufficient_evidence",
  "recommended_operation": "continue_search",

  "operation_args": {
    "query_intent": "fill_missing_claim"
  },

  "evidence_ids": ["obs_3", "obs_9"],
  "runtime_fields_used": ["remaining_turns"],

  "reason_code": "coverage_insufficient"
}
```

---

## [ ] P2.2 Artifact 禁止保存 teacher 后续轨迹

新增 validator：

```text
forbid_future_observation
forbid_teacher_trace
forbid_hidden_answer
forbid_hidden_verifier_text
```

允许：

```text
diagnosis
typed operation
operation arguments
reason code
visible evidence references
runtime metadata references
```

不允许：

```text
teacher 后续网页正文
teacher completion trajectory
隐藏 gold answer
teacher CoT
external verifier 的隐藏事实结论
```

---

## [ ] P2.3 `reason_code` 变成有限枚举

首轮至少：

```text
duplicate_evidence:
    exact_duplicate
    normalized_url_duplicate
    semantic_duplicate

premature_stop:
    unsupported_claim_remaining
    insufficient_coverage
    unresolved_conflict
    answer_not_grounded
```

禁止用自由文本 reason 直接参与训练。

自由文本可以保留在：

```text
debug_reason
```

但：

```text
debug_reason.loss_mask = 0
```

---

# 5. P3：新增 Information-Safe Gate

> 优先级：**BLOCKER**  
> 这是新版 SCOPE 与旧 Dual-mode OPD 的核心结构差异之一。

新增：

```text
harness/artifacts/validators.py
```

或：

```text
training/scope/gates.py
```

建议前者负责 schema/provenance，后者负责训练样本 gate。

---

## [ ] P3.1 Visibility Gate

检查：

```text
artifact.evidence_ids ⊆ decision_state.observed_ids
```

输出：

```json
{
  "visible": true,
  "unknown_evidence_ids": []
}
```

---

## [ ] P3.2 Runtime Provenance Gate

检查：

```text
artifact.runtime_fields_used
```

只能引用 DecisionState 中真实提供的 runtime metadata。

禁止 shadow 自己估计：

```text
remaining_search_calls
remaining_turns
token budget
tool budget
```

---

## [ ] P3.3 Module Responsibility Gate

例如：

```text
duplicate_evidence
    可以建议:
        skip_curate
        replace_evidence

    不可以建议:
        final_answer
        invent_query_answer
```

```text
premature_stop
    可以建议:
        continue_search
        verify
        open_source
        stop_and_answer

    不可以:
        直接输出未知 factual answer
```

---

## [ ] P3.4 Executability Gate

所有 `recommended_operation + operation_args` 必须经过：

```text
ActionSchema.validate()
```

而不是只检查字符串。

---

## [ ] P3.5 Shadow purity audit

每次 shadow call：

```text
hash_before = state_hash(env)
step_before = env.step
wm_before   = working_memory_hash

artifact = shadow(decision_state)

hash_after = state_hash(env)
```

必须满足：

```text
hash_before == hash_after
env.step unchanged
WorkingMemory unchanged
tool call count unchanged
observation count unchanged
```

否则：

```text
sample.mask = 0
audit_error = SHADOW_MUTATED_ENV
```

---

# 6. P4：新增 ActionRealizer，分离“诊断”与“动作”

> 优先级：**BLOCKER**

新版 artifact 不应该直接等价于一段 teacher completion。

新增：

```text
harness/shadow/action_realizer.py
```

接口：

```python
realize(
    decision_state,
    artifact
) -> CandidateAction
```

例如：

```text
Artifact:
    diagnosis=duplicate
    operation=skip_curate

→

CandidateAction:
    type=continue
```

或者：

```text
Artifact:
    diagnosis=insufficient_coverage
    operation=continue_search
    query_intent=find_primary_source

→

CandidateAction:
    type=search
    args={query: ...}
```

---

## [ ] P4.1 第一轮尽量使用 deterministic realizer

对 Dup：

```text
skip / do-not-curate
```

尽量 deterministic。

对 Premature Stop：

可以让 LLM 生成 query，但必须拆成：

```text
artifact:
    决定 “继续搜什么类型的信息”

realizer:
    将 query_intent 具体化为 query string
```

这样未来可以单独分析：

```text
stop decision 是否正确
query formulation 是否正确
```

避免两个 capability 混为一个标签。

---

# 7. P5：把 Dual-mode OPD 改成 Verified Decision Routing

> 优先级：**BLOCKER**

这是训练层最大的代码修改。

当前：

```text
Endorse
Correct
Ignore
  ↓
OPDTransitionV2
  ↓
endorse loss + correct loss
```

新版：

```text
student action
+
artifact
+
verifier
  ↓
routing
  ↓
verified target action
  ↓
DecisionSupervisionSampleV3
  ↓
统一 action-level imitation
```

---

## [ ] P5.1 新增统一 sample schema

新建：

```text
training/scope/schema.py
```

推荐：

```json
{
  "schema_version": "scope.supervision.v3",

  "episode_id": "...",
  "event_id": "...",
  "turn": 12,

  "branch_type": "MAIN",
  "capability_id": "premature_stop",
  "module_id": "budget_control",

  "decision_state": {},

  "student_action": {},
  "target_action": {},

  "route": "CORRECT",

  "artifact": {},

  "gates": {
    "visible": true,
    "schema_valid": true,
    "module_valid": true,
    "executable": true
  },

  "verification": {
    "student_valid": false,
    "target_valid": true,
    "score_student": null,
    "score_target": null
  },

  "weight_terms": {
    "procedural_purity": 1.0,
    "reliability": 1.0,
    "internalization": 0.0,
    "local_gain": 1.0
  },

  "sample_weight": 1.0
}
```

---

## [ ] P5.2 Endorse / Correct 只作为 route metadata

逻辑：

```python
if module_endorses(student_action) and verify(student_action):
    target_action = student_action
    route = ENDORSE

elif module_rejects(student_action):
    candidate = realize(artifact)

    if verify(candidate):
        target_action = candidate
        route = CORRECT

    else:
        route = IGNORE

else:
    route = IGNORE
```

不要再维护两个本质不同的主训练目标。

---

## [ ] P5.3 `OPDTransitionV2` 进入兼容层

建议：

```text
training/opd_v2/
```

保留用于：

```text
legacy baseline
旧 smoke
论文 ablation
```

但 `train_scope.py` 主路径改为：

```python
from training.scope import ...
```

避免论文已经改成 `Verified Decision Routing / SDI`，代码仍到处叫 `dual_mode_opd`。

---

# 8. P6：主损失改为 Action-Level SDI

> 优先级：**BLOCKER**

新版主方法不要依赖 teacher logits。

## [ ] P6.1 新增 Action Span Mask

`dataset_builder.py` 必须为每个 sample 标记：

```text
prompt tokens     loss_mask = 0
state text        loss_mask = 0
artifact text     不进入 student prompt 或 loss
tool observation  loss_mask = 0
target action     loss_mask = 1
```

确保训练的是：

> **当前 DecisionState 下应该执行什么 action**

而不是：

> 模仿 Harness 的解释文本和序列化格式。

---

## [ ] P6.2 主 loss：Corrective / Endorse CE 统一

实现：

```text
L_SDI = -w * log π(target_action | student_state)
```

Endorse：

```text
target = student_action
```

Correct：

```text
target = verified corrective action
```

---

## [ ] P6.3 Pairwise loss 降为 ablation

旧：

```text
-log σ(logπ(a+) - logπ(a-))
```

保留为：

```text
configs/scope/pairwise_ablation.yaml
```

但不作为第一版主方法。

原因：

1. 当前最重要的是先证明 local verified action label 能训练；
2. CE 与 DAgger-style expert action supervision 更直接；
3. black-box Harness 不需要 logits；
4. 少一个训练不稳定来源。

---

## [ ] P6.4 首轮建议先不叠加 RL

当前：

```text
train_scope.py = mock RL + OPD 已接线
```

但尚无真实训练。

第一轮建议：

```text
L = L_SDI + ξ L_KL
```

第二轮再比较：

```text
L = L_RL + λ L_SDI + ξ L_KL
```

理由：

> 当前首先要验证的是 “Harness capability 能否被局部内化”，而不是同时验证 RL reward、OPD 和 shadow 三套机制。

否则即使结果变好，也难回答：

```text
收益到底来自 RL 还是 SCOPE？
```

---

# 9. P7：重新定义 capability statistics

> 优先级：**HIGH**

旧版主要：

```text
G_m
U_m
ρ_m
```

新版：

```text
P_c : procedural purity
U_c : supervision reliability
ρ_c : internalization
δ_t : local decision gain / signal strength
```

这里建议按 `capability_id=c` 统计，而不是只按 module。

---

## [ ] P7.1 Reliability `U_c`

```text
verified targets / proposed targets
```

必须区分：

```text
U_correct
U_endorse
U_total
```

避免一个 capability 因大量简单 endorse 把 reliability 冲高。

---

## [ ] P7.2 Local gain `δ_t`

若 verifier 有标量：

```text
δ = max(0, score(target) - score(student))
```

若只有 binary：

Correct：

```text
student invalid
target valid
→ delta = 1
```

Endorse 不应该使用 `delta=0` 后把样本权重乘成 0。

因此代码上定义统一：

```text
signal_strength
```

```python
if route == CORRECT:
    signal_strength = local_gain

elif route == ENDORSE:
    signal_strength = endorsement_confidence
```

这是对新版公式实现时必须修正的细节。

---

## [ ] P7.3 Internalization `ρ_c`

新增：

```text
training/scope/internalization.py
training/evaluate_scope_internalization.py
```

只能在：

```text
held-out student DecisionStates
```

上测。

建议输出：

```json
{
  "duplicate_evidence": {
    "action_type_agreement": 0.83,
    "full_action_agreement": 0.72,
    "n": 500
  },
  "premature_stop": {
    "stop_continue_accuracy": 0.79,
    "full_action_agreement": 0.61,
    "n": 500
  }
}
```

不要直接用 training-set agreement 当 `ρ`。

---

## [ ] P7.4 第一轮训练暂时不要启用全动态权重

第一轮：

```text
P = 1
U = audit frozen value / sample verified
ρ = 0
w = 1
```

即：

```text
uniform verified supervision
```

目标：验证核心 mechanism。

第二轮：

```text
U
```

第三轮：

```text
U * (1-rho)
```

最后：

```text
P * U * (1-rho) * signal_strength
```

这样实验天然对应 E3 ablation。

---

# 10. P8：实现 Distillability Probe，但不要阻塞第一轮训练

> 优先级：**MEDIUM-HIGH**

新版理论增加：

```text
procedural purity P
```

但当前 Full Harness 本身较弱，立刻用 task reward 做复杂 module ablation 可能噪声很大。

因此拆成两步。

---

## [ ] P8.1 第一阶段：safe-by-construction P

对当前两个 capability：

### Duplicate Evidence

不需要新网页，只比较当前 observed evidence。

暂标：

```text
P_duplicate = 1.0
P_source = SAFE_BY_CONSTRUCTION
```

### Premature Stop

只允许使用：

```text
student-visible evidence state
runtime budget
```

满足 information-safe gate 时暂标：

```text
P_premature = 1.0
P_source = SAFE_BY_CONSTRUCTION
```

这只是工程先验，不作为论文最终实验结论。

---

## [ ] P8.2 第二阶段：正式 `probe_distillability.py`

新增三种模式：

```text
OFF
PROCEDURAL_ONLY
FULL
```

例如：

```text
M2 full:
    可以真实调用 external verifier

M2 procedural:
    只能看到学生已经拥有的 verification record
    可以决定 “是否应该 verify / 下一步做什么”
    不允许新增 verifier result

M2 off:
    不使用该 module
```

输出：

```text
artifacts/capability/distillability_map.json
```

格式：

```json
{
  "verification_control": {
    "reward_off": 0.10,
    "reward_proc": 0.14,
    "reward_full": 0.20,
    "procedural_share": 0.40
  }
}
```

### 注意

当 denominator 很小：

```text
R(full) - R(off) ≈ 0
```

不要直接产生一个不稳定的 `P`。

增加：

```text
min_effect_size
confidence_interval
n_episodes
probe_valid
```

---

# 11. P9：修复第一轮训练数据的最大风险——Premature Stop 标签失衡

> 优先级：**BLOCKER**

当前审计：

```text
online stop:
    几乎全部 bad stop → CORRECT

targeted valid-stop:
    8/8 → ENDORSE
```

这证明 selector **能判断好/坏 stop**，但不代表第一轮训练数据已经平衡。

若真实训练集里几乎只有：

```text
stop → CORRECT → continue
```

模型很可能学成：

> **永远不要 stop。**

这是当前第一轮训练最需要防止的 failure mode。

---

## [ ] P9.1 统计自然在线 stop 分布

必须输出：

```text
n_stop_total
n_stop_endorse
n_stop_correct
n_stop_ignore
```

并按：

```text
query
turn bucket
evidence coverage bucket
remaining budget bucket
```

分层。

---

## [ ] P9.2 targeted probe 不直接当主训练数据

Go25 的 targeted valid-stop 首要用途是：

```text
验证 selector / verifier precision
```

除非这些状态本身来自：

```text
student actually visited state
```

否则不要混进主 SCOPE train set。

可以单独标：

```text
source = TARGETED_PROBE
train_mask = 0
```

---

## [ ] P9.3 从真实 student states 中补 valid-stop

优先顺序：

1. student rollout 中已经得到充分证据的状态；
2. student rollout 中最终任务成功前的 stop / answer states；
3. recovery branch 后学生自然进入的充分证据状态；
4. 后续更强 student checkpoint 的 on-policy states。

目标不是人工制造 50/50，而是确保：

> 模型既看到 “现在不能停”，也看到 “现在确实应该停”。

---

## [ ] P9.4 单独评测 Stop calibration

增加指标：

```text
Premature Stop Rate
Over-search Rate
Valid Stop Recall
Stop Precision
Mean extra calls after sufficient evidence
```

第一轮训练后必须同时看：

```text
premature stop 是否下降
AND
over-search 是否上升
```

不能只报告前者。

---

# 12. P10：构建 SCOPE v3 训练数据

> 优先级：**BLOCKER**

新增：

```text
training/build_scope_dataset.py
```

输入：

```text
online chat audit events
DecisionState
Artifact
student action
verifier result
```

输出：

```text
artifacts/datasets/scope_v3/train.jsonl
artifacts/datasets/scope_v3/valid.jsonl
artifacts/datasets/scope_v3/stats.json
artifacts/datasets/scope_v3/manifest.json
```

---

## [ ] P10.1 第一版只使用

```text
Duplicate Evidence
Premature Stop
```

过滤：

```text
Irrelevant Evidence
Invalid Citation
unknown capability
failed gate
failed verifier
shadow mutation
```

---

## [ ] P10.2 数据 split 按 query，不按 event

错误：

```text
同一个 query 的 turn 5 在 train
turn 8 在 valid
```

正确：

```text
query-level split
```

否则 `ρ` 和 validation loss 会有严重泄漏。

---

## [ ] P10.3 保存 dataset provenance

每条 sample 保存：

```text
source rollout id
source event id
git commit
model checkpoint
harness config hash
retriever/index version
artifact schema version
verifier version
```

---

## [ ] P10.4 stats.json

至少：

```json
{
  "n_samples": 0,
  "route": {
    "ENDORSE": 0,
    "CORRECT": 0
  },
  "capability": {
    "duplicate_evidence": {},
    "premature_stop": {}
  },
  "gate_rejection": {},
  "action_type": {},
  "turn_bucket": {}
}
```

---

# 13. P11：第一轮真实训练配置

> 优先级：**BLOCKER**

新增：

```text
configs/scope/sdi_dup_premature.yaml
```

建议第一轮实验只验证：

> information-safe same-state verified action supervision 是否可以产生可测内化。

配置原则：

```yaml
capabilities:
  duplicate_evidence: true
  premature_stop: true
  irrelevant_evidence: false
  invalid_citation: false

training:
  objective: action_ce
  recovery_branch: false
  adaptive_weighting: false
  outcome_rl: false
  stabilization_kl: true

data:
  main_branch_only: true
  require_information_safe: true
  require_target_verified: true
```

---

## [ ] P11.1 第一轮必须保留三个 ablation

```text
A0 Base Qwen2.5-7B
A1 Duplicate only
A2 Premature only
A3 Duplicate + Premature
```

这样能回答：

```text
具体哪个 capability 被学到了？
联合训练是否互相干扰？
```

---

## [ ] P11.2 再增加旧方法对照

至少：

```text
Legacy endorse-only
Legacy dual-mode / pairwise
New unified corrective CE
```

用于证明新版改动不仅是改名。

---

# 14. P12：Recovery-on-Demand

> 优先级：**第二阶段**
>
> 第一轮 shadow-only 训练跑通以后再实现。

当前 Premature Stop 非常适合验证 recovery：

```text
student:
    stop prematurely

shadow:
    continue_search

main branch:
    仍保留 student stop，保证 on-policy 数据定义

recovery branch:
    从 stop 前 state fork
    执行 corrective search
    student 再继续 K turns
```

---

## [ ] P12.1 首先验证环境是否可 fork

新增 smoke：

```text
tests/scope/test_env_fork_equivalence.py
```

方案 A：

```text
deep snapshot
```

需要 snapshot：

```text
WorkingMemory
trajectory
budget counters
opened docs
curated evidence
search history
random state
```

方案 B：

若环境不易 deepcopy：

```text
episode seed
+
action prefix
→ deterministic replay to turn t
```

然后要求：

```text
state_hash(replayed) == state_hash(original)
```

---

## [ ] P12.2 recovery branch 必须独立 telemetry

每个 event：

```json
{
  "branch_type": "RECOVERY",
  "parent_event_id": "...",
  "fork_turn": 7,
  "intervention_capability": "premature_stop"
}
```

不能和 main trajectory 混成一条 episode。

---

## [ ] P12.3 只执行一次 teacher correction

规则：

```text
fork state
→ execute target action once
→ student owns policy again
```

禁止：

```text
teacher takeover K turns
```

否则方法会退化为 DAgger-style teacher mixing / completion。

---

## [ ] P12.4 recovery 权重低于 main

配置：

```text
recovery_weight < main_weight
```

并报告：

```text
main samples
recovery samples
main/recovery ratio
```

---

## [ ] P12.5 第一版只给 Premature Stop 开 recovery

Duplicate Evidence 通常不会直接截断后续状态空间。

Premature Stop 则会：

```text
stop at t
→ t+1... 不再存在
```

所以先只给：

```text
PREMATURE_STOP
```

开启 recovery。

这是最干净的机制实验。

---

# 15. P13：Internalization 与 capability retirement

> 优先级：**第三阶段**

新增：

```text
training/evaluate_scope_internalization.py
inference/scope/retirement_eval.py
```

---

## [ ] P13.1 不要仅凭 `ρ` 自动退役

退役必须同时满足：

```text
agreement high
AND
removal test no significant drop
AND
capability does not require irreducible external state
```

即：

```text
rho_c > threshold
performance_drop_when_disabled < tolerance
external_dependency == false / hybrid
```

---

## [ ] P13.2 生成 retirement manifest，不改 Harness 本体

例如：

```json
{
  "duplicate_evidence": {
    "status": "RETIRED",
    "rho": 0.94
  },
  "premature_stop": {
    "status": "HYBRID_RUNTIME",
    "rho": 0.88,
    "retain": ["budget_counter"]
  },
  "external_verifier": {
    "status": "RUNTIME_ONLY"
  }
}
```

路径：

```text
artifacts/capability/retirement_manifest.json
```

inference 启动时读取。

---

## [ ] P13.3 minimal runtime 评测矩阵

必须比较：

```text
1. Bare model
2. Minimal executor
3. Minimal executor + hard state/budget runtime
4. Partially retired Harness
5. Full Harness
```

指标：

```text
Task Success
Evidence quality
Search calls
Harness LLM calls
latency
token cost
Premature Stop Rate
Repeated/Duplicate Evidence Rate
```

最终形成：

```text
Task Quality
vs
Runtime Complexity / Cost
```

Pareto curve。

---

# 16. P14：Irrelevant Evidence 重新进入训练前的必要条件

> 优先级：**暂缓**

当前结论：

```text
Irrelevant GT 与 shadow 同源
offline proxy precision = 0 / spotcheck fail
```

因此保持：

```yaml
irrelevant_evidence:
  train_enabled: false
  audit_enabled: true
```

---

## [ ] P14.1 独立 GT

必须构造不依赖同一个 heuristic 的 ground truth。

推荐来源优先级：

```text
1. supporting-doc / gold-evidence 标注
2. 独立 cross-source verifier
3. 人工 spotcheck subset
4. 与 shadow 不同模型/规则的 judge
```

不要：

```text
shadow heuristic
→ 用同一 heuristic 生成 GT
→ 再评 shadow
```

---

## [ ] P14.2 等新 events 字段积累后重跑

你已经增加：

```text
action_arguments
recommended_action
query
rendered / reconstructed context
```

后续还必须确保：

```text
candidate evidence id
curated evidence id
claim id
observed ids
```

都可离线重建。

满足后再跑：

```text
offline_relabel_audit.py
evaluate_audit_go_nogo.py
```

只有独立 Precision / Recall 过阈值后再 enable。

---

# 17. P15：Telemetry / Audit 升级

> 优先级：**HIGH**

旧 telemetry 主要回答：

```text
shadow 有没有触发？
correct / endorse 比例是多少？
```

新版必须能回答：

```text
这个 supervision 为什么允许训练？
它是否真的 information-safe？
它来自哪个 capability？
是否经过 verifier？
是否 main/recovery？
训练后该 capability 是否真的被内化？
```

---

## [ ] P15.1 Event schema v3

至少：

```text
episode_id
event_id
turn
branch_type
decision_state_hash
module_id
capability_id
student_action
artifact
target_action
route
gate_results
verifier_results
sample_weight
```

---

## [ ] P15.2 新增核心 audit 指标

全局：

```text
shadow_mutation_rate
visibility_violation_rate
schema_failure_rate
unexecutable_target_rate
verified_target_rate
```

每 capability：

```text
calls
endorse
correct
ignore
precision
recall
route balance
action distribution
```

训练后：

```text
agreement rho
error rate before/after
retention
```

---

# 18. P16：配置迁移

## [ ] 新建主配置

```text
configs/scope/sdi_dup_premature.yaml
configs/scope/sdi_uniform.yaml
configs/scope/sdi_adaptive.yaml
configs/scope/sdi_recovery.yaml
configs/scope/distillability_probe.yaml
configs/scope/retirement_eval.yaml
```

---

## [ ] legacy 配置继续保留

```text
dual_mode.yaml
endorse_only.yaml
verification_only.yaml
fixed_weight.yaml
adaptive_weight.yaml
```

但顶部加入：

```yaml
method_status: legacy_baseline
paper_main_method: false
```

这样旧实验仍可复现。

---

# 19. P17：建议的正式实验顺序

不要直接：

```text
Full SCOPE = gate + recovery + adaptive P/U/rho + RL + retirement
```

一次全开。

推荐严格按下面顺序。

---

## Round 0 — Pipeline Freeze

完成：

```text
Harness v2 830
Phase 0 baseline
DecisionStateV2
ArtifactV3
information-safe gates
DecisionSupervisionSampleV3
```

Go 条件：

```text
shadow_mutation_rate = 0
visibility_violation_rate = 0
target executable rate ≈ 1
Dup / Premature audit 仍通过
```

---

## Round 1 — First Real SCOPE Training

```text
capabilities:
    Duplicate
    Premature

objective:
    action CE + KL

no:
    RL
    recovery
    adaptive weighting
    Irrelevant
```

回答唯一问题：

> **verified same-state local decision supervision 本身能不能把 capability 写进模型？**

---

## Round 2 — Capability Ablation

```text
Duplicate only
Premature only
Duplicate + Premature
```

评估：

```text
duplicate error slice
stop calibration slice
overall BrowseComp+
```

---

## Round 3 — Routing / Loss Ablation

比较：

```text
endorse-only
reject-mask-only
legacy pairwise correct
corrective CE
unified verified target CE
```

回答：

> Corrective target 是否必要，以及为什么新版 routing 优于旧 Dual-mode 表述。

---

## Round 4 — Recovery

只给：

```text
Premature Stop
```

开启 recovery。

比较：

```text
shadow-only
shadow + recovery
```

重点：

```text
recovery state coverage
Premature Stop Rate
success
teacher/shadow calls
```

---

## Round 5 — Adaptive Capability Weighting

逐级：

```text
uniform
U
U(1-rho)
P U (1-rho) signal_strength
```

---

## Round 6 — Distillability

正式跑：

```text
OFF
PROCEDURAL_ONLY
FULL
```

获得：

```text
P_c / P_m
```

并做 fresh-corpus transfer。

---

## Round 7 — Retirement

训练后：

```text
rho
removal test
minimal runtime
runtime cost
```

输出 capability retirement map。

---

# 20. 第一轮训练前最终 Checklist

只有下面全部满足，才建议正式 `train_scope.py`。

## Baseline

- [ ] Harness v2 830 跑完
- [ ] bare baseline 冻结
- [ ] minimal-runtime baseline 冻结
- [ ] full-harness baseline 冻结
- [ ] git commit / config / model / index version 记录完整

## Schema

- [ ] DecisionStateV2
- [ ] ArtifactV3
- [ ] DecisionSupervisionSampleV3
- [ ] capability_id 与 module_id 分离

## Safety / Validity

- [ ] shadow mutation audit = 0
- [ ] visibility gate
- [ ] runtime provenance gate
- [ ] module responsibility gate
- [ ] action executability gate
- [ ] target verifier

## Capability

- [ ] Duplicate enabled
- [ ] Premature enabled
- [ ] Irrelevant disabled
- [ ] Invalid Citation disabled / audit only

## Data

- [ ] online multi-step states 为主
- [ ] query-level train/valid split
- [ ] route distribution 已统计
- [ ] Premature valid-stop coverage 已检查
- [ ] targeted probes 默认 train_mask=0
- [ ] dataset manifest 完整

## Training

- [ ] action-span loss mask
- [ ] unified CE target
- [ ] stabilization KL
- [ ] no teacher logits dependency
- [ ] no recovery in Round 1
- [ ] no adaptive weighting in Round 1
- [ ] no outcome RL in Round 1

---

# 21. 当前已有文件的具体处理表

| 当前文件 / 目录 | 操作 | 新版角色 |
|---|---|---|
| `harness/capability/DecisionState` | **升级** | `DecisionStateV2` + provenance + hash |
| `harness/capability/selector` | **保留并细化** | capability-bearing event selector |
| `harness/shadow/*` | **保留并改输出** | typed local artifact，不输出 teacher trace |
| `harness/artifacts/*` | **重点升级** | ArtifactV3 + visibility/provenance gates |
| `harness/telemetry/*` | **升级** | main/recovery/gate/capability telemetry |
| `training/train_scope.py` | **重构** | 主入口：SDI → optional RL/recovery |
| `training/opd_v2/` | **降为 legacy** | 旧 dual-mode baseline |
| `training/opd/` | **保留 legacy** | full-trace OPD baseline |
| `training/audit_scope_shadow_bare.py` | **保留** | terminal / replay diagnostic，不作为主数据源 |
| `training/audit_scope_chat_online.py` | **核心保留** | 主 DecisionState 数据来源 |
| `offline_relabel_audit.py` | **升级** | 独立 GT 后重新启用 Irrelevant |
| `evaluate_audit_go_nogo.py` | **升级** | capability-wise Go/No-Go |
| `harness/lifecycle/` | **冻结** | 不属于 paper-1 |
| `train_coevolution_round.py` | **冻结** | BiSHOP legacy |
| `modules/recovery` | **冻结** | 不等于新版 training recovery branch |
| `configs/scope/dual_mode.yaml` | **legacy** | baseline |
| `configs/scope/minimal_runtime.yaml` | **保留并扩展** | retirement / minimal-runtime eval |
| `artifacts/baselines/` | **立即补齐** | Phase 0 |
| `inference/scope/` | **扩展** | capability / retirement / transfer eval |

---

# 22. 新版 SCOPE 的代码语义迁移表

| 旧概念 | 新概念 | 代码动作 |
|---|---|---|
| Dual-mode OPD | Verified Decision Routing | Endorse/Correct 只保留 route metadata |
| `a-` vs `a+` pairwise | verified target action | 主损失改 action CE |
| `OPDTransitionV2` | `DecisionSupervisionSampleV3` | 新 schema |
| teacher logits | optional | 主方法不依赖 |
| module weight `G U (1-rho)` | capability weight `P U (1-rho) strength` | 第二阶段实现 |
| whole module | capability-bearing decision | 新增 `capability_id` |
| shadow output | local decision artifact | 增 provenance / visibility |
| Full Harness advantage | capability + information | 新增 distillability probe |
| no second trajectory | main trajectory 无第二条 | recovery 允许 training-only fork |
| M4 Recovery | 不启用 | 新建独立 recovery branch |
| BiSHOP lifecycle | 不启用 | 新建 retirement evaluator |
| Full Harness → No Harness | Full → Hybrid → Minimal Runtime | inference manifest |

---

# 23. 当前最应该先改的 10 个事项

按实际执行顺序：

1. [ ] **跑完 Harness v2 830，并冻结 Phase 0。**
2. [ ] **新增 `capability_id`，第一版锁死 Dup + Premature。**
3. [ ] **DecisionState 升 V2，补 observed IDs / runtime provenance / action arguments。**
4. [ ] **Artifact 升 V3，禁止 future observation / hidden verifier result。**
5. [ ] **加入 information-safe + executability + no-mutation gates。**
6. [ ] **新增 ActionRealizer，把 artifact 与 executable action 分开。**
7. [ ] **把 `OPDTransitionV2` 主路径替换成 `DecisionSupervisionSampleV3`。**
8. [ ] **把 endorse/correct loss 统一为 verified target action CE。**
9. [ ] **构建 Dup + Premature 的 query-level split SCOPE v3 dataset，并检查 stop 标签失衡。**
10. [ ] **先跑不含 RL / recovery / adaptive weighting 的第一轮真实训练与前后 slice evaluation。**

只有第 10 步已经产生可信 positive signal 后，再继续：

```text
Recovery
→ adaptive P/U/rho
→ distillability probe
→ module retirement
```

---

# 24. 第一篇论文代码完成的最终 Definition of Done

SCOPE paper-1 不需要重新实现 BiSHOP 的完整共演化。

代码层面的完成条件建议定义为：

### Core mechanism

- [ ] pure student / minimal-runtime rollout
- [ ] DecisionStateV2
- [ ] typed local artifact
- [ ] information-safe gate
- [ ] verified action routing
- [ ] action-level SDI
- [ ] Dup + Premature 正式训练闭环
- [ ] recovery-on-demand
- [ ] capability weighting
- [ ] internalization evaluation

### Scientific evidence

- [ ] Phase 0 baseline
- [ ] same-state local vs full-trace
- [ ] endorse-only vs corrective
- [ ] shadow-only vs recovery
- [ ] uniform vs adaptive weighting
- [ ] fresh-corpus / cross-harness
- [ ] capability-specific error slices
- [ ] module/capability distillability map

### Deployment

- [ ] retirement evaluator
- [ ] retirement manifest
- [ ] partial-retirement Harness
- [ ] minimal-runtime evaluation
- [ ] quality–runtime-cost Pareto curve

### 明确不属于 paper-1

- [ ] ~~Harness 自我更新~~
- [ ] ~~Model→Harness proposal~~
- [ ] ~~多轮 co-evolution~~
- [ ] ~~BiSHOP lifecycle~~
- [ ] ~~M4 runtime Recovery~~
- [ ] ~~强行内化 external verifier / live retrieval~~

---

# 25. 最重要的工程判断

当前不是“需要重新写一套 SCOPE”。

真正需要做的是：

```text
已经验证的：
student rollout
DecisionState
shadow module
audit / Go-No-Go

        ↓ 保留

旧：
Endorse / Correct 两套 OPD loss
模块整体加权
不产生第二轨迹
lifecycle 全冻结

        ↓ 精确替换

新：
Information-safe local artifact
Verified target action
Action-level SDI
Capability-level P/U/rho
Training-only local recovery branch
Post-training capability retirement
```

因此工作量最大的不是 Harness 环境本身，而是三个接口：

1. **`DecisionState → Artifact` 的信息边界；**
2. **`Artifact → verified target action` 的监督协议；**
3. **`target action → training / recovery / retirement statistics` 的数据闭环。**

只要这三个接口稳定，现有 BrowseComp+、BM25、Ultra 环境、chat driver、shadow audit 和大部分 telemetry 都可以继续复用。

---

## 推荐下一步

第一条真正应该开始写的代码不是 recovery，也不是 adaptive weighting，而是：

```text
DecisionStateV2
    ↓
ArtifactV3
    ↓
InformationSafeGate
    ↓
DecisionSupervisionSampleV3
```

然后立即用现有 Go25 已通过的：

```text
Duplicate Evidence
Premature Stop
```

重新生成一版 SCOPE v3 数据。

这一步完成后，你当前项目会第一次在**代码语义上与新版论文叙事完全一致**，同时又最大限度复用已经做完的实验与审计基础。
