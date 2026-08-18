# H20：Clean-Init 泛化主线——修复 gpt-oss Harness Tool Channel → AUTO Structured OPD → Actual-Student Real Closed-Loop

> 时间基线：2026-08-17  
> 机器：8×H20  
> Repo：`/data/ppnm/Capability_Evolution/SCAPE`  
> 当前 clean base：`openai/gpt-oss-20b` + Harness-1 public SFT only  
> 本轮定位：**不与 H100-1/2/3/4 重复。专门回答“Beyond Textual Privilege / OPD+Harness 的正结果能否从 released Harness-1 初始化泛化到 clean gpt-oss 初始化？”**

---

## 0. 本轮唯一目标

H20 上一轮已经完成：

```text
gpt-oss-20b
→ Harness-1 public SFT only
→ CLEAN-SFT-FULL / CLEAN-SFT-TOOL
```

但当前状态是：

```text
CLEAN-SFT-FULL free-generation tool parse ≈ 0.75
CLEAN-SFT-TOOL free-generation tool parse ≈ 0.25
C1 V2/V3 same-state gap exists
C2 OPD not started
NEXT_DECISION = CLEAN_BASE_BLOCKED
```

因此本轮**禁止直接在不合格 clean base 上跑旧 Graph-Hybrid C2/8K**。

本轮要闭合的新链条是：

```text
clean gpt-oss initialization
→ repair/validate actual Harness tool channel
→ fresh on-policy H_-auto states
→ auto_populate_first_search same-xi_t causal/value confirmation
→ reverse 8-way Route-KL into actual gpt-oss Student weights
→ remove auto privilege at inference
→ real multi-step Search execution
→ Student > clean Base
→ unshuffled > shuffled-target control
```

如果成功，这会成为主论文非常重要的一条泛化证据：

> Harness-native high-level control privilege 的可蒸馏性并不依赖 released `pat-jj/harness-1` checkpoint 的特殊初始化，而可以在只经过 public SFT 的 clean `gpt-oss-20b` Student 上重新建立。

---

# 1. 与四台 H100 的职责边界

当前四台 H100 已分别负责：

```text
H100-1: AUTO actual-LoRA real closed-loop + shuffle causal control
H100-2: importance_tagging proper K4/K8 → actual LoRA → real closed-loop
H100-3: Structured privilege redesign →争取 Structured > Matched Text
H100-4: Full Harness / Matched Text / OPHSD actual-model baselines
```

因此 H20 不重复这些 exact jobs。

H20 本轮只负责：

```text
clean gpt-oss base recovery
+
AUTO cross-initialization / cross-model transfer
```

如果 H100-1 已经冻结 final real-eval contract，则 H20 应尽可能同步该 contract：

```text
query manifest
retriever/index
tool set
max steps
termination
reward
evidence/qrel metric
parser
```

如果 H100-1 尚未产出 final handoff，不等待；先按本文件冻结 H20 本地 contract，并把所有 manifest 写全，之后允许离线重算统计，但禁止偷偷换 test split。

---

# 2. Step A：先判断上一轮 Base Gate 是“真实模型问题”还是“评测/模板合同问题”

## 2.1 不要直接重新训练 69 小时 SFT

优先复用已完成 checkpoint：

```text
outputs/0814_clean_mechanism/sft/gpu0/full_s42_full/lora_checkpoint
outputs/0814_clean_mechanism/sft/gpu1/full_s43_full/lora_checkpoint
outputs/0814_clean_mechanism/sft/gpu2/tool_s42_full/lora_checkpoint
outputs/0814_clean_mechanism/sft/gpu3/tool_s43_full/lora_checkpoint
```

主候选优先：

```text
FULL seed42
FULL seed43
```

TOOL 只作诊断，不再作为默认 base。

---

## 2.2 审计 Harmony / tool-call runtime contract

必须逐项检查：

```text
1. tokenizer/chat-template 是否与 gpt-oss 官方 Harmony 格式一致
2. tool schema 是否在 generation prompt 中正确注入
3. assistant channel / recipient / tool-call token 是否正确
4. parser 是否接受模型实际产生的 canonical Harmony tool call
5. 是否错误把 analysis prose 当作 invalid tool
6. stop token / end-of-turn / end_search 是否配置正确
7. LoRA merge/load 是否正确
8. evaluator 是否真的加载对应 FULL checkpoint
9. sampling 是否 deterministic 且没有被错误 stop
10. raw gpt-oss / FULL / TOOL 是否使用完全相同 runtime contract
```

增加一个最小 synthetic contract test：

```text
canonical valid Harmony tool call -> parser must pass
canonical end_search call         -> parser must pass
malformed tool name               -> parser must fail
```

输出：

```text
outputs/h20_clean_auto_0817/base_recovery/HARMONY_RUNTIME_AUDIT.md
outputs/h20_clean_auto_0817/base_recovery/HARMONY_RUNTIME_TESTS.json
```

---

## 2.3 把 n=4 smoke 扩成正式 n=128

在固定 query manifest 上分别跑：

```text
RAW_GPT_OSS
CLEAN_FULL_S42
CLEAN_FULL_S43
CLEAN_TOOL_S42
CLEAN_TOOL_S43
```

记录：

```text
tool_parse_rate
legal_tool_rate
invalid_tool_rate
tool_name histogram
search/read/curate/verify/end_search coverage
mean generated tokens
termination reason
first-action distribution
```

写：

```text
BASE_EVAL_128.csv
BASE_EVAL_128.md
BASE_QUERY_MANIFEST.json
```

### Base Gate

优先使用：

```text
parse_rate >= 0.99
legal_tool_rate >= 0.99
invalid_tool_rate <= 0.01
non-degenerate tool coverage
```

如果 FULL s42/s43 中至少一个通过，立即进入 Step C，不做额外 SFT。

---

# 3. Step B：如果 Base Gate 仍失败，只允许一次“工具通道恢复”训练，不重跑旧 FULL/TOOL

当前 TOOL-only mask 已经证明不是正确修复方向。

本轮禁止：

```text
再次原样 FULL 25112×3 epoch
再次原样 TOOL-only 25112×3 epoch
直接降低 Base Gate 后继续 OPD
用 constrained decoder 掩盖模型不会产生正确 tool syntax
```

---

## 3.1 FORMAT-REPAIR 数据

从 public SFT 中筛出真实合法 tool-call turn，构造：

```text
FORMAT_REPAIR_TRAIN
FORMAT_REPAIR_VALID
```

要求：

```text
- 只使用 public SFT，RL records=0
- 保留完整 user/context prefix
- 保留 assistant 从 channel transition 到 tool call 的完整结构
- 对 8 个合法工具尽可能做 tool-balanced sampling
- 保留 end_search examples
- 不引入 synthetic/mock answer
- 不引入 future reward/gold
```

核心不是只训练 JSON arguments，而是训练：

```text
assistant turn
→ correct Harmony channel transition
→ correct recipient/tool name
→ valid arguments serialization
→ correct turn termination
```

建议 loss mask：

```text
FORMAT_AWARE_TOOL_TURN
```

即从 assistant tool-action turn 的必要 Harmony/control tokens 开始，到完整 tool call 结束，而不是仅 mask tool-name/args。

---

## 3.2 只从 FULL checkpoint 做短程恢复

优先：

```text
base = CLEAN_FULL_S42
```

训练 budget 先控制在：

```text
2K / 4K unique-or-explicitly-resampled tool-action turns
1 epoch
LoRA r=8 or 16
small lr: 2e-6 ~ 5e-6
```

并行最多跑以下四个 cell：

```text
FR_A: format-aware, balanced tools, seed42
FR_B: format-aware, balanced tools, seed43
FR_C: format-aware + end_search upweight, seed42
FR_D: full-assistant-action-turn CE diagnostic, seed42
```

不要开大规模 loss sweep。

---

## 3.3 Format Recovery Gate

每个 cell 重新跑同一个 `BASE_EVAL_128` manifest。

只有满足：

```text
parse >= 0.99
legal >= 0.99
invalid <= 0.01
```

才允许成为本轮 `CLEAN_AUTO_BASE`。

若多个通过：

```text
1. 优先 parse/legal 更高
2. 再比较真实 search tool coverage
3. 再比较 held-out SFT CE
```

写：

```text
FORMAT_REPAIR_TRAINING.csv
FORMAT_REPAIR_EVAL.csv
CLEAN_AUTO_BASE.json
```

---

## 3.4 Auto-stop

如果：

```text
原 FULL 两 seed FAIL
+
一次 substantive FORMAT_REPAIR 仍全部 FAIL
```

则写：

```text
CLEAN_GPT_OSS_TOOL_CHANNEL_UNRESOLVED
```

停止在 gpt-oss clean-init 上做 AUTO OPD。

不得为了“继续主线”绕过 Base Gate。

---

# 4. Step C：为 clean gpt-oss 重新收集 AUTO on-policy 数据

Base Gate PASS 后，不要直接拿 H100 Harness-1 的 AUTO states 训练 gpt-oss。

必须重新按 clean Student occupancy 收集。

Component：

```text
auto_populate_first_search
```

Student runtime：

```text
H_-auto_populate_first_search
auto_populate_first_search=false
```

Teacher/full view：

```text
same xi_t
auto_populate_first_search=true
```

只允许 state-time structured privilege：

```text
auto_seed presence / metadata
full/reduced mask state
step
first-search-pending
prior-search count
tool history
```

禁止：

```text
future reward
gold answer
future trajectory
terminal outcome
```

---

## 4.1 Fresh on-policy collection

目标：

```text
>= 512 unique relevant states
最好 1024+
query-disjoint
```

只保留：

```text
first-search / first-evidence-population relevant windows
```

记录：

```text
snapshot_hash
query_id
step
reduced view
full structured view
student action
teacher route distribution
teacher action
AUTO effect-active flag
```

输出：

```text
AUTO_CLEAN_RAW.jsonl
AUTO_CLEAN_TRAIN.jsonl
AUTO_CLEAN_VALID.jsonl
AUTO_CLEAN_TEST.jsonl
AUTO_CLEAN_SPLIT_MANIFEST.json
AUTO_CLEAN_PRIVILEGE_SCHEMA.md
AUTO_CLEAN_DATA_AUDIT.md
```

不得复制 unique state 冒充规模；如需要 update-budget resampling，必须显式写 `resampled_duplicate=true`。

---

# 5. Step D：clean gpt-oss 上重新做 AUTO causal/value confirmation

不要直接假设 H100 上 value-positive 在 clean gpt-oss occupancy 上仍成立。

对 same `xi_t` 做 fork/replay：

```text
Teacher/full branch:
  AUTO ON
  executes its own next action

Student/reduced branch:
  AUTO OFF
  executes its own next action

Continuation:
  both continue under reduced/no-AUTO policy
```

跑：

```text
K4
K8
seeds >= 2
```

建议：

```text
512 states × 2 seeds × K4/K8
```

记录：

```text
A_K4
A_K8
replay noise
bootstrap CI
effect-active stratum
natural first-search stratum
```

输出：

```text
AUTO_CLEAN_VALUE_PER_STATE.jsonl
AUTO_CLEAN_VALUE_GATE.json
AUTO_CLEAN_VALUE_REPORT.md
```

### Value Gate

至少：

```text
mean value > replay_noise
CI_low > 0 on >= 1 main stratum
K4/K8 direction consistent
two seeds direction consistent
```

如果 value 本身不 positive：

```text
STOP_CLEAN_AUTO_VALUE_NOT_TRANSFERRED
```

不要靠调 loss 强救。

---

# 6. Step E：Actual gpt-oss Student LoRA OPD

主 recipe 直接沿用当前 AUTO 最强设计，不再大 sweep：

```text
objective       = reverse 8-way Route-KL
state selection = relevant / component-active
lambda_args     = 0
lambda_anchor   = 0.05
LoRA            = actual gpt-oss LLM weights
Student input   = reduced/no privilege
```

Route space 固定：

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

训练：

```text
AUTO_CLEAN_UNSHUFFLED
seeds 42,43,44,45
```

优先 4 卡并行。

其余 GPU 同时做：

```text
- evaluator smoke
- route distribution audit
- checkpoint reload test
- real closed-loop dry-run
```

---

## 6.1 必要训练检查

每个 cell 必须：

```text
finite loss
finite grad
params changed
checkpoint reloadable
route distribution normalized
invalid tool sanity pass
student inference privilege=false
```

写：

```text
AUTO_CLEAN_LORA_TRAINING_CELLS.csv
```

不允许把 auxiliary `route_head.pt` 当主模型。

---

# 7. Step F：marginal-preserving shuffled-target causal control

这是 H20 clean-init 线必须补的关键因果对照。

在完全相同的 training states 上：

```text
state fixed
teacher target marginal fixed
state-target pairing shuffled
```

要求与 unshuffled 完全匹配：

```text
same unique states
same query ids
same update budget
same target marginal
same reverse Route-KL
same LoRA rank/alpha
same lr
same epochs
same seeds
```

训练：

```text
AUTO_CLEAN_SHUFFLED
seeds 42,43,44,45
```

记录 fixed points，尽量为 0。

输出：

```text
AUTO_CLEAN_SHUFFLE_TRAINING_CELLS.csv
AUTO_CLEAN_SHUFFLE_AUDIT.md
```

---

# 8. Step G：Actual-model Real Multi-Step Closed-Loop

本轮主结果必须来自真实 gpt-oss Student model weights，而不是 same-state route proxy。

## 8.1 冻结真实评测合同

写：

```text
AUTO_CLEAN_REAL_EVAL_CONTRACT.md
```

必须记录：

```text
query source
query manifest
BM25/index path/version
qrel/evidence gold availability
final-answer gold availability
reward definition
tool cost
max steps
termination
tool parser
tool set
student inference privilege=false
```

如果 final-answer gold 不存在：

```text
final_answer = N/A
```

禁止填 0。

---

## 8.2 16-query deterministic smoke

至少比较：

```text
CLEAN_BASE
BEST_UNSHUFFLED
```

每个 query 保存完整 trajectory。

必须确认：

```text
LoRA actually loaded
weights changed
AUTO privilege absent
actual tool parser/executor used
state mutates after tool execution
multi-step behavior exists
terminal scorer non-constant
```

若 Base 与 Student action sequence 完全一致，先修 evaluator/checkpoint，不进入大评测。

输出：

```text
AUTO_CLEAN_REAL_SMOKE_CASES.jsonl
AUTO_CLEAN_REAL_SMOKE_AUDIT.md
```

---

## 8.3 DEV / TEST

冻结：

```text
DEV  = 128 query
TEST = 256 query
```

如果 unique query 不足，使用全部可用，不允许重复 query。

至少比较：

```text
CLEAN_BASE
AUTO_CLEAN_UNSHUFFLED seed42
AUTO_CLEAN_UNSHUFFLED seed43
AUTO_CLEAN_UNSHUFFLED seed44
AUTO_CLEAN_UNSHUFFLED seed45
AUTO_CLEAN_SHUFFLED seed42
AUTO_CLEAN_SHUFFLED seed43
AUTO_CLEAN_SHUFFLED seed44
AUTO_CLEAN_SHUFFLED seed45
CLEAN_FULL_HARNESS
```

主设置：

```text
max_steps = 6
```

额外做少量：

```text
max_steps = 10/12
```

sanity，排除“更早停止、工具调用更少所以 reward 看似更高”的假收益。

记录：

```text
external/task reward
evidence/qrel recall
trajectory success
tool calls
search calls
invalid-tool rate
termination
token cost
wall time
```

做 query-level paired bootstrap 95% CI。

输出：

```text
AUTO_CLEAN_REAL_CLOSED_LOOP_DEV.csv
AUTO_CLEAN_REAL_CLOSED_LOOP_TEST.csv
AUTO_CLEAN_PAIRED_BOOTSTRAP.csv
AUTO_CLEAN_REAL_CLOSED_LOOP.md
```

---

# 9. Step H：必要 case analysis

自动抽取：

```text
Base fail -> AUTO success
Base success -> AUTO fail
AUTO success -> Shuffle fail
AUTO fail -> Full Harness success
```

每类尽量 20–25 个。

保存：

```text
query
all tool calls
arguments
runtime-state summary
termination
external reward
qrel/gold evidence
```

重点回答：

```text
1. clean gpt-oss 的增益是否真的来自 better first-search control？
2. 改进是否传播到第 2~6 step？
3. 是否主要只是更早 end_search？
4. shuffle 为什么失效/不失效？
5. clean gpt-oss 与 released Harness-1 上的 AUTO failure/success mode 是否一致？
```

输出：

```text
AUTO_CLEAN_CASE_ANALYSIS.md
AUTO_CLEAN_CASES.jsonl
```

---

# 10. 最终 Go / Redesign / Stop

## GO

只有同时满足：

```text
A. clean gpt-oss tool Base Gate PASS
B. AUTO proper value K4/K8 positive
C. actual gpt-oss LoRA real closed-loop > clean Base
D. >= 2 unshuffled seeds same direction
E. unshuffled > shuffled on real closed-loop
F. invalid-tool no material regression
G. student inference privilege=false
```

才写：

```text
CLEAN_INIT_AUTO_TRANSFER_PASS
```

这时 H20 线可以形成论文中的：

```text
cross-initialization / cross-model generalization
```

证据。

---

## REDESIGN ONCE

如果：

```text
same-state route metrics improve
but real closed-loop does not improve
```

只允许一次实质 redesign，优先根据 case 分析选择：

```text
continuation-aware relevant-state selection
multi-step component-active windows
balanced hard negatives
route + legal argument supervision
```

只有当当前 runtime state 中存在合法 argument target 时才允许最后一项。

重新训练时必须重新做 shuffled-target control。

---

## STOP

如果以下任一成立：

```text
1. 两轮 tool-channel recovery 后 clean Base 仍无法稳定调用工具
2. proper K4/K8 value 在 clean occupancy 上不 positive
3. 一次 substantive OPD redesign 后 actual Student 仍不优于 clean Base
4. unshuffled 与 shuffled 无可分辨差异
```

则停止该 clean-init recipe，不追 seed 噪声。

---

# 11. 明确禁止事项

本轮 H20 禁止：

```text
- 直接恢复旧 Graph-Hybrid C2/8K
- 在 Base Gate FAIL 的 checkpoint 上做大规模 OPD
- 再做大范围 loss sweep
- 把 same-state JS/CE 当论文主结果
- 把 route_head.pt 当 actual Student
- 用 constrained decoding 隐藏 tool-format incompetence
- 把缺失 final-answer gold 写成 0
- 把 duplicated/resampled rows 说成 unique states
- 引入 future reward/gold 作为 privilege
- 用 Full Harness takeover continuation
- 因为 H100 上 AUTO positive 就跳过 clean gpt-oss proper value confirm
```

---

# 12. GPU 建议调度

## Phase A/B：Base recovery

```text
GPU0: FULL s42 n=128 eval / format-repair A
GPU1: FULL s43 n=128 eval / format-repair B
GPU2: TOOL s42 n=128 diagnostic / format-repair C
GPU3: TOOL s43 n=128 diagnostic / format-repair D
GPU4: raw gpt-oss n=128 + Harmony parser contract
GPU5: evaluator/template audit + independent replay
GPU6: data build / tool-balance audit
GPU7: monitor + tests + checksum + free slot
```

## Phase C/D：AUTO fresh data + value

```text
GPU0-3: on-policy reduced rollout / same-xi_t scoring shards
GPU4-7: K4/K8 fork-replay value shards
```

## Phase E/F：LoRA + shuffle

```text
GPU0-3: unshuffled seeds 42/43/44/45
GPU4-7: shuffled seeds 42/43/44/45
```

## Phase G：real closed-loop

按模型显存和吞吐实际情况分 shard；不要为了并行度改变 evaluator contract。

---

# 13. 必须产出

统一根目录：

```text
outputs/h20_clean_auto_0817/
```

至少包含：

```text
RUN_MANIFEST.json
STATUS_LIVE.md

base_recovery/
  HARMONY_RUNTIME_AUDIT.md
  HARMONY_RUNTIME_TESTS.json
  BASE_QUERY_MANIFEST.json
  BASE_EVAL_128.csv
  BASE_EVAL_128.md
  FORMAT_REPAIR_TRAINING.csv
  FORMAT_REPAIR_EVAL.csv
  CLEAN_AUTO_BASE.json

auto_data/
  AUTO_CLEAN_PRIVILEGE_SCHEMA.md
  AUTO_CLEAN_DATA_AUDIT.md
  AUTO_CLEAN_RAW.jsonl
  AUTO_CLEAN_TRAIN.jsonl
  AUTO_CLEAN_VALID.jsonl
  AUTO_CLEAN_TEST.jsonl
  AUTO_CLEAN_SPLIT_MANIFEST.json

value/
  AUTO_CLEAN_VALUE_PER_STATE.jsonl
  AUTO_CLEAN_VALUE_GATE.json
  AUTO_CLEAN_VALUE_REPORT.md

training/
  AUTO_CLEAN_LORA_TRAINING_CELLS.csv
  AUTO_CLEAN_SHUFFLE_TRAINING_CELLS.csv
  AUTO_CLEAN_SHUFFLE_AUDIT.md

real_eval/
  AUTO_CLEAN_REAL_EVAL_CONTRACT.md
  AUTO_CLEAN_REAL_SMOKE_CASES.jsonl
  AUTO_CLEAN_REAL_SMOKE_AUDIT.md
  AUTO_CLEAN_REAL_CLOSED_LOOP_DEV.csv
  AUTO_CLEAN_REAL_CLOSED_LOOP_TEST.csv
  AUTO_CLEAN_PAIRED_BOOTSTRAP.csv
  AUTO_CLEAN_REAL_CLOSED_LOOP.md
  AUTO_CLEAN_CASE_ANALYSIS.md
  AUTO_CLEAN_CASES.jsonl

BEST_CLEAN_AUTO_STUDENT.json
H20_CLEAN_AUTO_HANDOFF.json
SHA256SUMS
```

---

# 14. Handoff 必须明确回答

`H20_CLEAN_AUTO_HANDOFF.json` 至少包含：

```text
clean_base_checkpoint
clean_base_gate_pass
clean_base_parse_rate
clean_base_invalid_tool_rate

auto_value_k4_positive
auto_value_k8_positive
auto_value_seed_consistent

actual_model_weights=true/false
student_inference_privilege=false
student_beats_clean_base
unshuffled_beats_shuffled
full_harness_reference_available

external_metric_contract_valid
final_answer_gold_available
best_checkpoint
recommended_for_main_table

final_decision:
  CLEAN_INIT_AUTO_TRANSFER_PASS
  or CLEAN_GPT_OSS_TOOL_CHANNEL_UNRESOLVED
  or STOP_CLEAN_AUTO_VALUE_NOT_TRANSFERRED
  or STOP_CLEAN_AUTO_REAL_TASK_NO_GAIN
```

---

# 15. result-record 更新规则

实验完成后，把本轮结果追加到：

```text
result-record.md
```

必须写清：

```text
Setting
Results
Paired/controls
Gate
Decision
Artifacts
```

尤其禁止把：

```text
format repair success
same-state route improvement
value-positive
```

单独包装成最终正结果。

最终主结果只认：

```text
actual gpt-oss Student weights
+
no-privilege inference
+
real multi-step Search
+
external metric > clean Base
+
unshuffled > shuffled
```
