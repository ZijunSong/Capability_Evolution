# SCAPE-0813-Next-H100-4.md
## H100-4（4×H100）下一轮：独立 Learnability Metric Audit / Gate-L Revalidation

> H100-4 本轮不要再做 Candidate-B utility 排名。
>
> 四张卡最适合做一个真正独立的验证：
>
> **对 evidence_graph / subtractive_curation / importance_tagging / verify_tool 各占一张卡，独立重算 base 与训练后 checkpoint 的 learnability metric，判断 H20 的“全 FAIL”是否是真实现象。**
>
> 这是当前最重要的 cross-node reproducibility check。

本机不训练。

---

# 1. Repo / Environment

```bash
cd /mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-4/SCAPE
```

GPU Python：

```text
必须 /opt 环境
```

建议新建独立 evaluator：

```text
/opt/scape-metric-audit
```

输出：

```text
outputs/h100_4_learnability_metric_audit/
outputs/scape_prestage_v4/H1004_LEARNABILITY_AUDIT_HANDOFF.json
```

---

# 2. 关键独立性要求

本机 evaluator：

```text
不得 import H20 Gate-L 的 divergence 实现
不得 import旧 aggregate 中对 d_pre/d_post 的判定函数
```

可以复用：

```text
snapshot schema
token span parser
model loading
```

但 KL/JS/CE 必须独立实现。

---

# 3. 从 H20 同步最小 artifacts

只同步必要内容，不复制整个 merged model 仓库。

优先：

```text
LoRA adapter
checkpoint metadata
VALID/TEST split
token masks / raw same-state examples
```

base model 使用本机已有：

```text
pat-jj/harness-1
```

需要的对象：

### evidence_graph

```text
base
uniform L8K s42
uniform L8K s43
weighted L8K s42
weighted L8K s43
name_only L8K s42
```

### subtractive_curation

```text
base
uniform L512/L2K s42/s43
action_ce 2K s42
name_only 2K s42
```

### importance_tagging

```text
base
uniform L512/L2K s42/s43
```

### verify_tool

```text
base
uniform L512/L2K s42/s43
```

如果某 adapter 不存在：

```text
记录 MISSING_ARTIFACT
不要用其他 checkpoint 替代。
```

---

# 4. 独立指标

对每个 checkpoint：

```text
Tool-name JS
Teacher-sequence CE
Tool-name forward KL
Arg-key forward KL
Arg-value forward KL
Tool-name agreement
Exact tool-call agreement
Invalid tool rate
```

所有真正 KL/JS：

```text
must_be_nonnegative = true
```

允许数值误差：

```text
>= -1e-6
```

出现更低负值：

```text
METRIC_IMPLEMENTATION_BUG
```

---

# 5. 4 卡并行

## GPU0 — evidence_graph

顺序：

```text
1. identity/null controls
2. base
3. uniform s42
4. uniform s43
5. weighted s42
6. weighted s43
7. name_only s42
```

heldout：

```text
EG_VALID_1K
EG_TEST_1K
```

---

## GPU1 — subtractive_curation

```text
1. identity/null
2. base
3. L512 s42
4. L512 s43
5. L2K s42
6. L2K s43
7. action_ce 2K
8. name_only 2K
```

---

## GPU2 — importance_tagging

```text
1. identity/null
2. base
3. L512 s42/s43
4. L2K s42/s43
```

---

## GPU3 — verify_tool

```text
1. identity/null
2. base
3. L512 s42/s43
4. L2K s42/s43
```

完成后本卡负责全局 aggregate。

---

# 6. Positive Controls

每卡都必须：

```text
teacher == student + same view
=> KL/JS ≈ 0
```

至少 GPU0 再做：

```text
student logits temperature perturbation
or fixed small logit bias
=> KL/JS > 0
```

用于验证 scorer 不是“永远输出 0”。

---

# 7. Cross-node Compare

读取 H20：

```text
old d_pre/d_post
new V2 metric
```

生成：

```text
H20_OLD_VS_H1004_NEW.csv
```

分类：

### A. `GATE_BUG_CONFIRMED`

如果：

```text
旧 metric 的 improvement 方向与独立 KL/CE 结论系统性冲突
```

### B. `GATE_FAIL_CONFIRMED`

如果：

```text
独立 evaluator 也显示训练后没有一致改善
```

### C. `OBJECTIVE_SPECIFIC_PASS`

例如：

```text
name_only pass
uniform fail
```

这对 H20 Graph-Hybrid objective 选择非常关键。

---

# 8. 特别检查：signed score

现有 H100 记录里 `I_args_raw` 也可能为负。

检查代码命名：

如果它实际是：

```text
teacher logprob - student logprob
或其他 signed delta
```

则：

```text
保留 raw diagnostic
重命名为 *_signed_delta
```

禁止在报告中写成：

```text
KL divergence
JS divergence
```

这一点也同步给 H20。

---

# 9. Handoff

生成：

```json
{
  "metric_contract_valid": true,
  "old_gate_bug": false,
  "component_results": {
    "evidence_graph": "...",
    "subtractive_curation": "...",
    "importance_tagging": "...",
    "verify_tool": "..."
  },
  "recommended_h20_action": "..."
}
```

`recommended_h20_action` 只能为：

```text
RESCORE_EXISTING_AND_STAGE_S
GRAPH_HYBRID_NAME_ONLY
GRAPH_HYBRID_UNIFORM
CLEAN_MECHANISM_SETTING
FIX_METRIC_IMPLEMENTATION_FIRST
```

---

# 10. 禁止事项

```text
不训练
不再做 B-utility 排名
不再扩 verify targeted influence
不跑 official Chroma
不把本机结果覆盖 H20 closed-loop result
```

本机只负责：

```text
metric validity
+
cross-node reproducibility
```

---

# 11. 最终产物

```text
RUN_MANIFEST.json
STATUS_LIVE.md
INDEPENDENT_METRIC_IMPLEMENTATION.md
PER_CHECKPOINT_METRICS.csv
PER_CHECKPOINT_METRICS.json
H20_OLD_VS_H1004_NEW.csv
GATE_L_REVALIDATION.md
outputs/scape_prestage_v4/H1004_LEARNABILITY_AUDIT_HANDOFF.json
SHA256SUMS
```
