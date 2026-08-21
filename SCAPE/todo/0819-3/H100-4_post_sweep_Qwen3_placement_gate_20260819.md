# H100-4 后续执行指令：Qwen3 Reload 标准化 + Capability Placement Gate + Master Aggregation

> 日期：2026-08-19  
> 机器：H100-4  
> Base：`/mnt/songzijun/models/Qwen3-30B-A3B-Instruct-2507`  
> 目标：H100-4 不再启动新的正式组件 OPD 主实验，转为 **Qwen3 OPD 基础设施验证机 + Capability Placement Gate 验证机 + 10-component master aggregation 节点**。

---

## 0. 当前已确认结论

H100-4 已完成真实 Harness-1 collection：

```text
token_budget_marker:
  2000 queries
  8000 rollouts
  8000 unique event-active states
  TRAIN_STATES_5K = 5000
  synthetic_row_count = 0

verify_tool:
  2000 queries
  8000 rollouts
  8000 unique event-active states
  TRAIN_STATES_5K = 5000
  synthetic_row_count = 0
```

当前科学 gate：

```text
token_budget_marker:
  marker_present_rate = 1.0
  actionable_marker_rate = 0.0
  5000/5000 states 位于 low_under_60
  decision = TEACHER_COMPONENT_NO_POSITIVE_UTILITY

verify_tool:
  verify_action_available_rate = 1.0
  student_has_verify_tool = false
  decision = NON_REALIZABLE_ACTION_SPACE_MISMATCH
```

当前工程 gate：

```text
Qwen3 OPD_PILOT:
  256 real_harness1 states
  4 LoRA steps
  adapter saved

native PEFT reload:
  failed at WeightConverter.__init__(distributed_operation)

manual safetensors reload:
  384/384 LoRA tensors loaded
  post-reload forward passed
```

因此当前严格禁止：

```text
- 不对 token_budget_marker 启动正式 PURE_OPD / RL_PLUS_OPD
- 不对 verify_tool 启动 Student After OPD
- 不给 Student 增加 verify tool 来“修复” realizability
- 不补 synthetic states
- 不改变原 5K 主分布来制造 positive utility
- 不擅自增加第 11 个 component 做新的主实验
```

---

# 1. 执行优先级

严格按：

```text
P0. Preflight + 锁定当前 H100-4 canonical artifacts
P1. 修复/标准化 Qwen3 LoRA canonical reload
P2. 实现统一 Capability Placement Gate
P3. 审计 H100-1/2/3 最新 handoff
P4. 更新 10-component master table
P5. 只有 P0-P4 完成且 GPU 空闲时，做 token_budget_marker stress diagnostic
```

P5 只允许是 diagnostic，不得替代当前 5K main result。

---

# 2. 固定环境

```bash
export CAP_ROOT=/mnt/songzijun/Capability_Evolution
export SCAPE_ROOT=/mnt/songzijun/Capability_Evolution/SCAPE
export EASYOPD_ROOT=/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD
export MODEL_PATH=/mnt/songzijun/models/Qwen3-30B-A3B-Instruct-2507

cd "$EASYOPD_ROOT"
source scripts/setup_scape_easyopd_smoke7_env.sh
mkdir -p outputs/component_sweep_0818/h100_4/post_phase_u
```

记录：

```bash
{
  echo "DATE=$(date -Is)"
  echo "HOST=$(hostname)"
  echo "PYTHON=$(which python)"
  echo "MODEL_PATH=$MODEL_PATH"
  python -V
  python - <<'PY'
import torch, transformers
print('torch', torch.__version__)
print('cuda', torch.version.cuda)
print('transformers', transformers.__version__)
print('gpu_count', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
} | tee outputs/component_sweep_0818/h100_4/post_phase_u/ENVIRONMENT.txt
```

检查模型：

```bash
test -d "$MODEL_PATH" || exit 1
python - <<'PY'
from transformers import AutoTokenizer, AutoConfig
p='/mnt/songzijun/models/Qwen3-30B-A3B-Instruct-2507'
AutoTokenizer.from_pretrained(p, local_files_only=True, trust_remote_code=True)
AutoConfig.from_pretrained(p, local_files_only=True, trust_remote_code=True)
print('QWEN3_LOCAL_LOAD_OK')
PY
```

失败则：

```text
STOP_H1004_POST_SWEEP_PREFLIGHT_FAILED
```

---

# 3. P0：锁定 canonical H100-4 结果

确认：

```text
outputs/component_sweep_0818/h100_4/H1004_COMPONENT_HANDOFF.json
outputs/component_sweep_0818/h100_4/H1004_TEACHER_BEFORE_DIAGNOSTICS.json
outputs/component_sweep_0818/h100_4/token_budget_marker/TRAIN_STATES_5K.jsonl
outputs/component_sweep_0818/h100_4/token_budget_marker/OPD_PILOT/ADAPTER_RELOAD_ACCEPTANCE.json
outputs/component_sweep_0818/h100_4/verify_tool/TRAIN_STATES_5K.jsonl
outputs/component_sweep_0818/h100_4/SHA256SUMS
```

执行：

```bash
cd "$EASYOPD_ROOT/outputs/component_sweep_0818/h100_4"
sha256sum -c SHA256SUMS
cd "$EASYOPD_ROOT"
```

失败：

```text
STOP_H1004_CANONICAL_ARTIFACT_DRIFT
```

后续所有新输出只写：

```text
outputs/component_sweep_0818/h100_4/post_phase_u/
```

不得覆盖已完成的 H100-4 canonical result。

---

# 4. P1：Qwen3 LoRA canonical reload 标准化

## 4.1 目标

必须打通：

```text
Qwen3 base
-> LoRA train
-> save_pretrained
-> 新 Python process
-> canonical adapter reload
-> adapter active
-> deterministic forward
-> real Harness-1 smoke
```

最终输出：

```text
post_phase_u/qwen3_reload/
├── QWEN3_RELOAD_ROOT_CAUSE.md
├── QWEN3_RELOAD_ENV.json
├── QWEN3_RELOAD_ACCEPTANCE.json
├── QWEN3_RELOAD_NUMERIC_CHECK.json
├── QWEN3_RELOAD_REAL_LOOP_SMOKE.json
└── SHA256SUMS
```

## 4.2 复现 native reload bug

新增：

```text
scripts/h1004_reproduce_qwen3_peft_reload.py
```

要求：

```text
- 只使用已有 OPD_PILOT adapter，不重新训练
- clean process 加载 base
- 分别测试 PeftModel.from_pretrained 与 model.load_adapter
- 保存完整 stack trace
- 记录 torch / transformers / peft / accelerate 版本
- 记录 model class、adapter_config、target_modules
```

adapter 路径从：

```text
outputs/component_sweep_0818/h100_4/token_budget_marker/OPD_PILOT/
```

自动寻找 `adapter_config.json + adapter_model.safetensors`，不得猜目录。

## 4.3 Root-cause audit

逐项检查：

```text
A. Qwen3 remote-code/custom class 与 PEFT WeightConverter API mismatch
B. transformers / peft 版本 mismatch
C. adapter_config / target_modules 是否正确
D. save 与 clean reload 的 model class 是否一致
E. FSDP/Accelerate wrapper 是否污染保存 contract
F. state_dict key prefix 是否不一致
G. MoE expert/shared module 是否有特殊映射
```

优先修复顺序：

```text
1. 正确 API / 版本组合
2. adapter save/load metadata
3. 最薄 Qwen3 PEFT compatibility shim
4. manual safetensors mapping 只能作为最后 fallback
```

若写 shim：

```text
scape_easyopd/qwen3_peft_compat.py
```

它只能修 loader/mapping，不得改变权重数值、target modules 或训练目标。

## 4.4 Reload acceptance

新增：

```text
scripts/h1004_validate_qwen3_reload.py
```

至少通过：

```text
TEST-1 clean process base load
TEST-2 canonical adapter load
TEST-3 expected LoRA tensors 全部 active
TEST-4 adapter trainable tensors 非零
TEST-5 disable_adapter 前后输出不同
TEST-6 re-enable 后输出恢复
TEST-7 fixed prompt 保存前/重载后 logits 一致
TEST-8 real Harness-1 单 query closed-loop smoke
TEST-9 student_inference_privilege=false
```

推荐数值门槛：

```text
cosine(logits_before_save, logits_after_reload) >= 0.9999
```

若 BF16 需要放宽，必须在 acceptance 中记录原因。

理想状态：

```json
{
  "status": "QWEN3_CANONICAL_ADAPTER_RELOAD_READY",
  "manual_tensor_mapping_required": false,
  "real_loop_smoke_pass": true
}
```

若最终必须兼容 fallback：

```json
{
  "status": "QWEN3_ADAPTER_RELOAD_READY_WITH_COMPAT_FALLBACK",
  "manual_tensor_mapping_required": true,
  "native_peft_issue_root_caused": true,
  "real_loop_smoke_pass": true
}
```

禁止在 root cause 未明确时把 manual mapping 当成正式修复。

---

# 5. P2：实现统一 Capability Placement Gate

新增：

```text
scripts/h1004_capability_placement_gate.py
```

核心判定链：

```text
Event Support
    ↓
Positive Utility
    ↓
Student-native Realizability
    ↓
OPD Learnability
    ↓
Internalize / Keep Runtime
```

必须明确：

```text
event support != utility support
utility support != realizability
realizability != learnability
```

## 5.1 Event Support Gate

检查：

```text
collector_mode == real_harness1
synthetic_row_count == 0
TRAIN_STATES_5K exists
rows == 5000
unique(state_uid) == 5000
student_inference_privilege == false
```

输出之一：

```text
EVENT_SUPPORT_PASS
EVENT_SUPPORT_INSUFFICIENT
INVALID_DATA_COLLECTION_CONTRACT
```

## 5.2 Positive Utility Gate

优先读取真实 Teacher/Before 或 same-state K-step fork：

```text
Teacher reward - Student Before reward
K4 Teacher - Student
K8 Teacher - Student
paired CI
tool cost
terminal/trajectory reward
```

明确禁止：

```text
marker_present_rate 高 != positive utility
event_rate 高 != positive utility
Teacher context 更长 != positive utility
```

输出：

```text
POSITIVE_UTILITY_PASS
TEACHER_COMPONENT_NO_POSITIVE_UTILITY
UTILITY_NOT_YET_MEASURED
```

## 5.3 Realizability Gate

分类：

```text
DIRECT
PROJECTABLE
PARTIAL
NON_REALIZABLE
```

检查：

```text
- Teacher action 是否属于 Student action space
- Harness side effect 是否可投影成 Student-native action
- projected args 是否全部引用 Student-visible ids
- 是否依赖 hidden counter / hidden object / future observation
- 是否依赖 Student 不拥有的 tool/interface
```

输出：

```text
REALIZABLE_DIRECT
REALIZABLE_PROJECTABLE
REALIZABLE_PARTIAL
NON_REALIZABLE_ACTION_SPACE_MISMATCH
NON_REALIZABLE_EXTERNAL_INFORMATION
NON_REALIZABLE_HIDDEN_RUNTIME_STATE
```

`verify_tool` 必须继续得到：

```text
NON_REALIZABLE_ACTION_SPACE_MISMATCH
```

不得为了使它 PASS 给 Student 增加 verify。

## 5.4 OPD Learnability Gate

只有：

```text
Positive Utility == PASS
且
Realizability != FAIL
```

才允许读取/评价 Student After。

要求：

```text
PURE_OPD seeds
RL+OPD seeds
DEV real closed-loop
TEST real closed-loop
paired bootstrap
adapter reload acceptance
student_inference_privilege=false
```

输出：

```text
OPD_LEARNABILITY_PASS
OPD_LEARNABILITY_FAIL
OPD_NOT_RUN_GATE_BLOCKED
```

## 5.5 Placement Decision

统一映射：

```text
Event fail
  -> DATA_INSUFFICIENT

Utility fail
  -> KEEP_RUNTIME_OR_DROP_COMPONENT

Realizability fail
  -> KEEP_RUNTIME_PLACEMENT_BOUNDARY

Utility pass + Realizable pass + OPD fail
  -> REALIZABLE_BUT_NOT_LEARNED

Utility pass + Realizable pass + OPD pass
  -> INTERNALIZATION_CANDIDATE
```

每个 component 写：

```text
CAPABILITY_PLACEMENT_AUDIT.json
CAPABILITY_PLACEMENT_AUDIT.md
```

---

# 6. P3：审计 H100-1/2/3 handoff

新增：

```text
scripts/h1004_discover_component_handoffs.py
```

搜索范围仅：

```text
outputs/component_sweep_0818/
outputs/scape_easyopd/framework/
```

发现：

```text
*HANDOFF*.json
*COMPONENT_ROWS*.json
*ACCEPTANCE*.json
DATA_STATS.json
DATA_PROVENANCE.md
ADAPTER_RELOAD_ACCEPTANCE.json
```

根据文件内容中的：

```text
machine_role
component
logical_model_id
resolved_model_path
```

识别来源，不要只靠目录名猜。

## 6.1 Base consistency

必须一致：

```text
logical_model_id == Qwen3-30B-A3B-Instruct-2507
resolved_model_path == /mnt/songzijun/models/Qwen3-30B-A3B-Instruct-2507
SCAPE commit 相同
SCAPE-EasyOPD commit 相同
TRAIN_POOL SHA 相同
state_uid schema 相同
selection_seed == 20260818
chat/tool rendering contract 相同
```

失败：

```text
MASTER_TABLE_BLOCKED_BASE_MISMATCH
```

绝不平均不同 Base。

## 6.2 Collector consistency

正式训练组件要求：

```text
collector_mode == real_harness1
synthetic_row_count == 0
TRAIN_STATES_5K rows == 5000
unique(state_uid) == 5000
query-pool source SHA 一致
```

失败：

```text
INVALID_DATA_COLLECTION_CONTRACT
```

## 6.3 Teacher isolation

检查：

```text
same Qwen3 weights
only target component ON
all other target components OFF
same Student state/prefix
no future observation
no DEV/TEST leakage
```

失败：

```text
INVALID_TEACHER_ISOLATION
```

## 6.4 Student After

检查：

```text
student_inference_privilege=false
actual LoRA/model weights loaded
adapter reload acceptance pass
DEV/TEST 都是 real closed-loop
retriever/reward/max-steps contract 一致
```

如果只是 same-state proxy：

```text
AUXILIARY_ONLY_NOT_MAIN_RESULT
```

不得写入 main After 列。

---

# 7. P4：更新 10-component master table

实现/更新：

```text
scripts/h1004_build_capability_placement_master.py
```

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

必须新增四个核心列：

```text
Positive Utility
Realizability
OPD Learnability
Placement Decision
```

完整推荐列：

```text
Component
Type
Event Support
Positive Utility
Realizability
OPD Learnability
Placement Decision
Train Queries
Unique Event States
Teacher Reward
Student Before Reward
Student After PURE_OPD
Delta PURE
Student After RL+OPD
Delta RL+OPD
DEV Status
TEST Status
Adapter Reload
Student Inference Privilege
Decision
Reason
```

H100-4 当前两行先固定为：

```text
token_budget_marker:
  Event Support = PASS
  Positive Utility = FAIL
  Realizability = PARTIAL
  OPD Learnability = NOT_RUN_GATE_BLOCKED
  Placement Decision = KEEP_RUNTIME_OR_DROP_COMPONENT

verify_tool:
  Event Support = PASS
  Positive Utility = DIAGNOSTIC_ONLY
  Realizability = NON_REALIZABLE_ACTION_SPACE_MISMATCH
  OPD Learnability = N/A
  Placement Decision = KEEP_RUNTIME_PLACEMENT_BOUNDARY
```

输出：

```text
outputs/component_sweep_0818/master/
├── COMPONENT_10_MAIN_TABLE.csv
├── COMPONENT_10_MAIN_TABLE.md
├── COMPONENT_10_FULL_METRICS.csv
├── COMPONENT_10_CAPABILITY_PLACEMENT.csv
├── COMPONENT_10_CAPABILITY_PLACEMENT.md
├── COMPONENT_10_DECISIONS.md
├── BASE_CONSISTENCY_AUDIT.md
├── COLLECTOR_CONSISTENCY_AUDIT.md
├── TEACHER_ISOLATION_AUDIT.md
├── ADAPTER_RELOAD_AUDIT.md
├── RUN_MANIFEST.json
└── SHA256SUMS
```

如果 H100-1/2/3 尚未完成，也生成 partial table，但缺失项统一：

```text
PENDING_EXTERNAL
```

或：

```text
N/A + explicit reason
```

绝不填 0。

master 状态写：

```text
MASTER_TABLE_BLOCKED_PHASE_E_INCOMPLETE
```

直到所有主结果齐全。

---

# 8. P5：可选 token_budget_marker high-pressure stress diagnostic

仅在 P0-P4 完成后允许启动。

目标不是训练，而是回答：

```text
当前 token_budget_marker 无 utility，
是 component 本身低价值，
还是当前 Search occupancy 从未进入真正 budget pressure？
```

## 8.1 数据

收集：

```text
128-512 unique high-pressure real Harness-1 states
```

允许为了 stress workload：

```text
- 提高 max_steps
- 选更长 context / noisy evidence query
- 增加 search/read history
```

但必须：

```text
student_inference_privilege=false
collector_mode=real_harness1
synthetic=false
no gold leakage
no future observation
```

禁止：

```text
手工改 token counter
伪造 remaining budget
复制 state
把 marker 文本直接给 Student
```

## 8.2 Pressure bins

先审计 Harness-1 当前 token-budget implementation，再定义：

```text
LOW
MID
HIGH
CRITICAL
```

阈值必须来自真实 runtime 语义，不能拍脑袋沿用 60/80/90。

## 8.3 Same-state fork

对每个 state：

```text
Student: token_budget_marker OFF
Teacher: token_budget_marker ON
same xi_t
same environment
same continuation policy
```

至少测：

```text
K4
K8
```

指标：

```text
reward delta
tool-call delta
search-call delta
late-step-waste delta
termination timing
invalid-tool
context tokens consumed
```

如果 HIGH/CRITICAL 仍无正 utility：

```text
TOKEN_BUDGET_MARKER_INTRINSICALLY_LOW_UTILITY_UNDER_TESTED_SEARCH_SETTING
```

如果 HIGH/CRITICAL 出现正 utility：

```text
TOKEN_BUDGET_MARKER_UTILITY_IS_OCCUPANCY_CONDITIONAL
```

即便后者成立，也不得直接开始正式 OPD，只记录：

```text
candidate_for_future_occupancy_conditioned_internalization=true
```

另开新 protocol。

---

# 9. GPU 原则

先：

```bash
nvidia-smi
```

动态读取 GPU 数，不假设一定 4 卡或 8 卡。

```text
P1 reload/debug: 单 GPU 优先
P2/P3/P4: CPU 为主
P5 stress: 按空闲 GPU 分 shard
```

禁止为了“跑满 GPU”而重新训练 H100-4 已被 gate 截停的两个组件。

---

# 10. 自动入口

新增：

```text
scripts/run_h1004_post_sweep.py
```

调用：

```bash
python scripts/run_h1004_post_sweep.py \
  --model "$MODEL_PATH" \
  --mode all
```

顺序：

```text
1. preflight
2. canonical artifact lock
3. Qwen3 reload audit/fix/acceptance
4. discover H100-1/2/3 current handoffs
5. placement audit for every discovered component
6. build/update partial master
7. checksum
8. write final H100-4 handoff
```

默认不自动运行 P5。

只有显式：

```bash
python scripts/run_h1004_post_sweep.py \
  --model "$MODEL_PATH" \
  --mode all \
  --enable-token-budget-stress
```

才允许做 stress diagnostic。

---

# 11. 最终 H100-4 handoff

输出：

```text
outputs/component_sweep_0818/h100_4/post_phase_u/H1004_POST_SWEEP_HANDOFF.json
```

至少包含：

```json
{
  "machine_role": "H100-4",
  "canonical_student_base": "/mnt/songzijun/models/Qwen3-30B-A3B-Instruct-2507",
  "qwen3_adapter_reload_status": "...",
  "token_budget_marker": {
    "event_support": "PASS",
    "positive_utility": "FAIL",
    "realizability": "PARTIAL",
    "opd_learnability": "NOT_RUN_GATE_BLOCKED",
    "placement_decision": "KEEP_RUNTIME_OR_DROP_COMPONENT"
  },
  "verify_tool": {
    "event_support": "PASS",
    "realizability": "NON_REALIZABLE_ACTION_SPACE_MISMATCH",
    "opd_learnability": "N/A",
    "placement_decision": "KEEP_RUNTIME_PLACEMENT_BOUNDARY"
  },
  "n_external_components_discovered": 0,
  "n_external_components_utility_pass": 0,
  "n_external_components_realizable": 0,
  "n_external_components_opd_pass": 0,
  "master_status": "MASTER_TABLE_BLOCKED_PHASE_E_INCOMPLETE",
  "token_budget_stress_status": "NOT_RUN"
}
```

---

# 12. 完成标准

只有全部满足才算 H100-4 后续任务完成：

```text
DONE-1 Qwen3 adapter reload 有 canonical acceptance；或 fallback 已明确 root cause
DONE-2 Capability Placement Gate 可对任意 component handoff 运行
DONE-3 H100-4 两个组件已写入 Utility / Realizability / Learnability / Placement 分类
DONE-4 当前可发现的 H100-1/2/3 handoff 已全部 audit
DONE-5 partial master table 已更新
DONE-6 未对 token_budget_marker / verify_tool 启动被禁止的正式 OPD
DONE-7 student_inference_privilege=false 未被破坏
DONE-8 新 artifact 不覆盖 canonical H100-4 已完成结果
DONE-9 SHA256SUMS 完整并验证通过
```

最终状态：

```text
H1004_POST_SWEEP_INFRA_AND_PLACEMENT_READY
```

若 H100-1/2/3 尚未结束：

```text
H1004_POST_SWEEP_READY_WAITING_EXTERNAL_PHASE_E
```

这不是失败，只表示 H100-4 自身职责已完成，最终科学主表等待其他机器的正式 Phase E 结果。

---

# 13. 给执行 Agent 的最终约束

```text
你的目标不是让 GPU 跑满，也不是让 10 个组件全部训练。

你的目标是：
1. 标准化 Qwen3 OPD save/reload；
2. 把 Event Support、Positive Utility、Realizability、OPD Learnability 分开；
3. 独立审计 H100-1/2/3 的结果；
4. 只让 Utility + Realizability 均通过的组件进入 Learnability 判断；
5. 不允许通过修改 Student interface、伪造 event、补 synthetic state 或改变主分布制造正结果；
6. token_budget_marker 与 verify_tool 的当前结论没有新证据时不得改写；
7. 最终回答：哪些 capability 应进入 model weights，哪些应该长期留在 Harness/runtime。
```
