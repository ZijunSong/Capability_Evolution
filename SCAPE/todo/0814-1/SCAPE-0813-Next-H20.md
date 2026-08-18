# SCAPE-0813-Next-H20.md
## H20（8×H20）下一轮：先修正 Learnability 测量，再做 Evidence-Graph Hybrid Migration

> 依据：`result-record(20260813-131646).md`
>
> 当前主结论：
>
> - `evidence_graph`：uniform + weighted Stage L 双次 FAIL；
> - `subtractive_curation / importance_tagging / verify_tool`：micro tournament 全 FAIL；
> - 但现有 `d_pre / d_post` 多次出现负值，而 SCAPE 方法定义的 KL/JS divergence 理论上应非负；
> - H100-1 已确认 `evidence_graph` 更合理的 placement 是 **graph state 保留在 runtime，graph-aware decision / renderer 影响再迁移**，不是整个 evidence graph 退役；
> - 官方 Chroma 仍被凭证阻塞，本轮继续标记 `LOCAL_COMPAT_ONLY`，不得宣称 official parity。
>
> 本机的任务不是继续“换一个 component 再训”，而是：
>
> 1. **审计并修正 Gate-L measurement contract；**
> 2. **用修正后的 metric 重评已有 checkpoint；**
> 3. 若旧 checkpoint 中其实存在有效 learnability，立即做 closed-loop Stage S；
> 4. 若仍全部失败，启动新的 **Graph-Hybrid V2→V3 migration**；
> 5. Hybrid 仍失败后，才进入 `gpt-oss-20b + Harness-1 public SFT` clean-mechanism setting。

---

# 0. 全局约束

Repo：

```bash
cd /data/ppnm/Capability_Evolution/SCAPE
```

开始前记录：

```bash
git status
git rev-parse HEAD
nvidia-smi
pytest -q
```

必须：

```text
legacy_scope_path_used=false
```

禁止：

```text
- 继续旧 SCOPE rollback / KEEP-SKIP / P_m
- 再跑 evidence_graph full-removal 的第三次 objective rescue
- 在 metric audit 未完成前启动新的 8K 训练
- 因 H100 utility/influence 强而越过 learnability gate
- 将 local BM25/HF 结果写成 official Chroma parity
```

输出根：

```text
outputs/0813_next_h20/
```

必须持续更新：

```text
STATUS_LIVE.md
RUN_MANIFEST.json
DECISION_STATE.json
result-record.md
```

---

# 1. Phase A — Gate-L Metric Validity Audit（最高优先级）

## 1.1 为什么必须先做

当前记录把 `d_pre / d_post` 称为 divergence，但出现例如：

```text
evidence_graph:
d_pre = -0.0107
d_post = -0.0485

subtractive_curation:
d_pre = -0.1344
d_post = -0.0234
```

如果该量被用于实现 SCAPE 文档中的：

```text
D_KL(p_T || p_S)
```

则负值不合法。

Cursor Agent 必须先定位：

```text
scape.training.hf_tool_opd
Gate-L evaluator
aggregate_candidate_b_tournament.py
evidence_graph Stage-L evaluator
```

确认目前的 `d_*` 究竟是：

```text
A. 真 forward KL
B. reverse KL
C. chosen-token logprob difference
D. teacher-minus-student score
E. 其他 signed proxy
```

**不要先假设代码是对的。**

---

# 2. 建立新的 Learnability Measurement Contract

新增：

```text
scape/eval/learnability_metrics_v2.py
tests/test_learnability_metrics_v2.py
scripts/rescore_existing_stage_l_v2.py
```

至少计算以下指标。

## M1. Tool-name JS

合法工具：

```text
fan_out_search
search_corpus
grep_corpus
read_document
review_docs
curate
verify
end_search
```

对 full/reduced view 分别计算合法 tool-name 的 sequence logprob，归一化后：

```text
JS_name >= 0
```

主要求：

```text
identity input => JS_name ~= 0
field-order null => close to null baseline
```

## M2. Teacher-sequence Cross Entropy

对 full-view teacher greedy / frozen teacher tool call：

```text
CE_T_on_S = - mean log p_S(y_T,j | y_T,<j, reduced_view)
```

这是稳定、可比较的非负量。

训练后 learnability：

```text
CE_post < CE_pre
```

## M3. Exact token forward-KL（若显存/吞吐允许）

在 tool span 的每个 teacher-forced position：

```text
KL(p_T || p_S) >= 0
```

至少在：

```text
name tokens
argument-key tokens
argument-value tokens
```

分别报告。

如果全词表 KL 太贵：

```text
先在 top-k union + tail correction 上实现，
但必须额外用小样本 full-vocab KL 验证近似误差。
```

## M4. Action agreement

```text
tool-name agreement
exact tool-call agreement
argument key agreement
argument string / doc-id set similarity
invalid_tool_rate
```

注意：

**signed logprob delta 可以继续作为 diagnostic，但必须改名，禁止再叫 divergence，也禁止单独作为 Gate L。**

---

# 3. Unit / Positive / Negative Controls

必须先跑：

```text
C0: teacher == student, same view
    => KL≈0, JS≈0

C1: teacher == student, field-order perturbation
    => small non-zero or ≈0

C2: teacher full view vs reduced view, base checkpoint
    => positive gap expected on known influence-positive states

C3: artificially perturb student logits / temperature
    => KL/JS 必须严格 > C0

C4: duplicate computation through two independent code paths
    => agreement tolerance <= 1e-5 on 32 states
```

若 C0/C3 失败：

```text
STOP_ALL_TRAINING=true
```

只修 evaluator。

---

# 4. Phase A 的 8 卡并行计划

使用已有 held-out data 与已有 checkpoints，**不训练**。

## GPU0 — evidence_graph seed42 audit

顺序：

```text
1. base θ0
2. uniform L8K s42
3. weighted L8K s42
4. name_only L8K s42
```

数据：

```text
EG_VALID_1K
EG_TEST_1K
```

输出：

```text
gpu0_eg_s42_metrics_v2.json
```

---

## GPU1 — evidence_graph seed43 audit

顺序：

```text
1. base θ0
2. uniform L8K s43
3. weighted L8K s43
4. 若存在对应 name_only checkpoint 一并测
```

---

## GPU2 — subtractive_curation audit

顺序：

```text
SC base
SC L512 s42/s43
SC L2K s42/s43
SC action_ce 2K
SC name_only 2K
```

---

## GPU3 — importance_tagging audit

```text
IT base
IT L512 s42/s43
IT L2K s42/s43
```

---

## GPU4 — verify_tool audit

```text
VT base
VT L512 s42/s43
VT L2K s42/s43
```

---

## GPU5 — metric controls

运行：

```text
C0/C1/C2/C3/C4
```

额外检查：

```text
token mask 边界
teacher-forced alignment
BOS/EOS
JSON punctuation
tool name tokenization
arg key/value tokenization
```

---

## GPU6 — old-vs-new metric correlation

对所有旧 Stage-L cell：

```text
old d_pre/d_post
vs
JS_name
vs
CE_T_on_S
vs
KL_name
vs
KL_args
```

生成：

```text
OLD_NEW_METRIC_CORRELATION.csv
OLD_GATE_REINTERPRETATION.md
```

---

## GPU7 — independent evaluator / smoke

禁止 import 原 Gate-L evaluator 的核心计算函数。

从 HF logits 独立实现 64-state scorer，生成：

```text
INDEPENDENT_METRIC_CHECK.md
```

用于发现“训练 evaluator 与独立 evaluator 同一个 bug”的风险。

---

# 5. Phase A Decision Gate

生成：

```text
LEARNABILITY_GATE_V2.json
LEARNABILITY_GATE_V2.md
```

### Case A — 发现 metric contract bug

若：

```text
旧 d_* 实际是 signed score
或 old Gate 对 improvement 方向判断错误
```

则：

1. 不重训；
2. 用 V2 metric 重评所有已有 checkpoint；
3. 按 component / objective / seed 重新判定；
4. 只有 V2 两 seed 一致改善的 setting 可进入 Stage S。

### Case B — old Gate 本质正确，四组件仍均失败

则：

```text
STOP_FULL_COMPONENT_MIGRATION=true
```

进入 Phase B：Graph Hybrid。

---

# 6. Phase A.5 — 如果已有 checkpoint 被 V2 Gate 翻转为 PASS

只选择 **一个** strongest candidate。

必须优先依据：

```text
held-out V2 metric
+ seed consistency
+ invalid_tool_rate
```

而非旧 `L_m`。

然后做真实四格：

```text
S0 = θ0 + H_full
S1 = θ0 + H_-m
S2 = θ' + H_-m
S3 = θ' + H_full
```

使用：

```text
fresh held-out 128q
```

若：

```text
S2 >= S0 - noninferiority_delta
AND runtime cost lower
```

才扩 400q。

若 S2 明显低：

```text
不要扩 seed，不进入 Stage M。
```

---

# 7. Phase B — Evidence-Graph Hybrid Target（新的主训练方向）

## 7.1 新 placement

不再训练：

```text
GRAPH_OFF student
<- GRAPH_FULL teacher
```

改成：

```text
Student runtime V2:
GRAPH_STATE_ONLY

Teacher view V3:
GRAPH_STATE_PLUS_MINIMAL_RENDER
```

可加辅助 teacher：

```text
V4 = GRAPH_FULL_RENDER
```

但 V3 是主 teacher，因为 H100-1 已显示：

```text
V3 接近 V4
且 V2 -> V3 policy influence 最大
```

最终想证明：

```text
graph store/state 保留
+
renderer/controller 进一步变轻
+
model 学会 graph-aware tool decision
```

这才是 SCAPE 的 hybrid placement。

---

# 8. 构造 Graph-Hybrid 数据

新增：

```text
outputs/0813_next_h20/graph_hybrid/data/
```

冻结：

```text
GH_TRAIN_8K
GH_VALID_1K
GH_TEST_1K
```

要求：

```text
student occupancy = V2
same xi_t
teacher view = V3
no teacher future rollout
query-disjoint
```

保存额外字段：

```text
graph_state_size
minimal_render_tokens
full_render_tokens
turn
student_tool
teacher_tool
teacher_entropy
JS_name
```

---

# 9. Graph-Hybrid Micro：8 卡队列

只先跑 512 / 2K。

## GPU0

```text
tool_name_only_KL
seed42
L512 -> L2K
```

## GPU1

```text
tool_name_only_KL
seed43
L512 -> L2K
```

## GPU2

```text
uniform name+args KL
seed42
L512 -> L2K
```

## GPU3

```text
uniform name+args KL
seed43
L512 -> L2K
```

## GPU4

```text
same-state action CE
seed42
L2K
```

## GPU5

```text
same-state full-response KL
seed42
L2K
```

## GPU6

```text
off-policy V3 trajectory SFT/OPD matched-budget baseline
seed42
L2K
```

## GPU7

```text
no-training V2/V3/V4 closed-loop baseline
+
runtime profiler
```

主线优先 `tool_name_only_KL`，原因是旧 evidence_graph ablation 中 name-only 离线表现明显优于 uniform / args-only；但这次 Gate 必须使用修正后的 V2 metrics。

---

# 10. Graph-Hybrid Gate

必须同时满足：

```text
JS_name_post < JS_name_pre
CE_T_on_S_post < CE_T_on_S_pre
两 seed 方向一致
invalid_tool_rate 不升
```

建议不要求 args KL 同时改善，因为本轮 target 被明确收窄为 graph-aware routing / tool decision。

若 PASS：

```text
扩 8K × seeds42/43/44
然后跑 Stage S:
theta0+V3
theta0+V2
theta'+V2
theta'+V3
```

最终目标：

```text
Q(theta', V2) >= Q(theta0, V3)
C(V2) < C(V3)
```

这比“删掉整个 graph”更合理。

---

# 11. Phase C — Clean Mechanism Setting（仅在 Hybrid 仍失败后）

不要立刻执行，只有：

```text
Metric V2 valid
AND
Graph-Hybrid micro 两个合理 objective 都 FAIL
```

才启动。

建立：

```text
openai/gpt-oss-20b
+
Harness-1 public SFT trajectories
```

先做 tool/search SFT，使模型具备基本 tool syntax，但不要使用 Harness-1 RL checkpoint。

创建：

```text
CLEAN_SFT_BASE
```

然后重新测：

```text
Contribution
Influence
Hybrid Learnability
```

目的：

区分：

```text
“SCAPE migration 不可行”
vs
“published Harness-1 checkpoint 已经处于不适合继续同模型自蒸馏的局部区域”
```

### Clean-setting 8 卡建议

若进入此阶段：

```text
GPU0: CLEAN_SFT seed42
GPU1: CLEAN_SFT seed43
GPU2: EG-hybrid micro s42
GPU3: EG-hybrid micro s43
GPU4: SC micro s42
GPU5: IT micro s42
GPU6: VT micro s42
GPU7: base/full/reduced evaluation + runtime
```

---

# 12. 本轮 H20 自动停止规则

1. `Metric V2` 未通过 identity / positive control：**禁止训练**。
2. 已有 checkpoint V2 Gate PASS：先 Stage S，不要重训。
3. Graph-Hybrid 两 seed × 两 objective 均 FAIL：停止 published-checkpoint migration。
4. Clean setting 也 FAIL：论文需要接受“某些 Harness capability 当前难以迁入 weights”，转向 placement boundary / hybrid runtime 结论。
5. 未出现一个 single-component/hybrid Stage-S positive 之前，**禁止 Stage M / multi-component annealing**。
6. 不再因为某个 component 的 Influence 高而无限救 loss。

---

# 13. 本机最终必须交付

```text
METRIC_CONTRACT_V2.md
LEARNABILITY_GATE_V2.md
OLD_GATE_REINTERPRETATION.md
GRAPH_HYBRID_MICRO_REPORT.md
GRAPH_HYBRID_STAGE_S.md        # 只有 gate pass 才有真实 S2/S3
CLEAN_SETTING_STATUS.md        # 若触发
NEXT_DECISION.json
SHA256SUMS
```

`NEXT_DECISION.json` 只能是以下之一：

```text
EXISTING_CHECKPOINT_STAGE_S
GRAPH_HYBRID_8K
CLEAN_MECHANISM_SETTING
PLACEMENT_BOUNDARY_RESULT
BLOCKED_BY_METRIC_BUG
```
