# SCOPE 实验记录

> 按 `method.md` pipeline 组织。状态标记：`✅ 已完成` · `🔄 进行中` · `📋 TODO` · `⏸ 暂缓` · `❌ 不做`

---

## Stage 0 — Module Distillability Probe（E0）

**method 对应：** 估计各模块 \(P_m\)，分类 procedural / hybrid / runtime-only。

**状态：** ✅ 已完成（100q audit 全部 100/100；Map 已冻结 2026-07-29 10:55）

**Setting**

| 项 | 值 |
|----|-----|
| Model | `Qwen2.5-7B-Instruct`（base，未训） |
| Benchmark | BrowseComp+，固定 audit 100q（`artifacts/datasets/e0_audit_100q`，SEED=42） |
| Retriever | BM25 |
| Harness base | `modules_full_v2.yaml`；FULL 复用 `outputs/harness_rollout_browsecomp_full_v2` |
| max_turns / max_tokens | 35 / 2048 |
| temperature | 1.0 |
| GPU / vLLM | GPU4，port 8776，`e0-harness-policy` |
| 对比形态 | capability-level OFF / PROC / FULL（`deterministic_truncation` 无 PROC） |
| 编排脚本 | `run_e0_distillability_nohup.sh` + `e0_watch_and_rerun.sh` + `e0_status.sh` |
| 产物目录 | `outputs/scope_e0_distillability/` · `artifacts/capability/distillability_map.json` · `E0_REPORT.md` |

#### ✅ 2026-07-29 — E0 100q 正式冻结（10:55）

**续跑时间线（07-29）**

| 时间 | 任务 | 结果 |
|------|------|------|
| 08:44 | kill 卡住 `external_verification/proc`（23/100） | 旧编排器误标 DONE |
| 08:44–09:18 | deterministic_truncation/off | ✅ 100/100（~34 min） |
| 09:19–10:13 | duplicate_evidence/proc | ✅ 100/100（~54 min） |
| 10:13–10:15 | verification_decision/proc（补 2 题） | ✅ 100/100 |
| 10:15–10:55 | external_verification/proc（续跑 77 题） | ✅ 100/100（~40 min） |
| 10:55 | `build_map.py` → Map + E0_REPORT 冻结 | ✅ |

**完成度（全部 100/100）**

| Capability | OFF | PROC | FULL |
|------------|----:|-----:|-----:|
| duplicate_evidence | ✅ | ✅ | ✅（复用） |
| stop_decision | ✅ | ✅ | ✅ |
| evidence_curation | ✅ | ✅ | ✅ |
| verification_decision | ✅ | ✅ | ✅ |
| external_verification | ✅ | ✅ | ✅ |
| deterministic_truncation | ✅ | n/a | ✅ |

**主指标 recall（paired；FULL 复用 Full v2 = 0.0506）**

| Capability | \(R_{\text{off}}\) | \(R_{\text{proc}}\) | \(R_{\text{full}}\) | \(\Delta^{\text{proc}}\) | \(\Delta^{\text{full}}\) | \(P_{\text{raw}}\) | CI(\(P\)) | W/L/T | Decision |
|------------|-------------------:|--------------------:|--------------------:|-------------------------:|-------------------------:|-------------------:|----------|------:|----------|
| duplicate_evidence | 0.0255 | 0.0343 | 0.0506 | +0.0088 | **+0.0250** | 0.35 | [−1.23, 1.27] | 12/7/81 | INCONCLUSIVE |
| stop_decision | 0.0282 | 0.0239 | 0.0506 | −0.0043 | +0.0224 | −0.19 | [−6.02, 0.84] | 8/10/82 | RUNTIME |
| evidence_curation | 0.0383 | 0.0169 | 0.0506 | −0.0213 | +0.0123 | −1.74 | [−24.0, 18.5] | 4/12/84 | RUNTIME |
| verification_decision | 0.0424 | 0.0198 | 0.0506 | −0.0226 | +0.0082 | −2.76 | [−22.8, 28.7] | 3/10/87 | RUNTIME |
| external_verification | 0.0152 | 0.0203 | 0.0506 | +0.0051 | **+0.0353** | 0.15 | [−0.52, 0.60] | 5/6/89 | RUNTIME |
| deterministic_truncation† | 0.0269 | — | 0.0506 | — | ~+0.0237 | — | — | — | INCONCLUSIVE |

† `build_map` 对 truncation 报 `no_overlap_queries`（map 中 R=0 为 builder bug）；episodes/summary 实测 \(R_{\text{off}}=0.0269\)，`truncation_events=0`（100q 未触发截断）。

**PROC audit**

| Capability | interventions | shadow_calls | info-safe | 备注 |
|------------|-------------:|-------------:|-----------|------|
| duplicate_evidence | 606 | 606 | ❌ | visibility_violation_rate=3%；PROC 部分恢复 \(P=0.35\) |
| stop_decision | 0 | 3 | ✅ | 几乎无干预 |
| evidence_curation | 700 | 700 | ✅ | 有干预但 \(\Delta^{\text{proc}}<0\) |
| verification_decision | 0 | 0 | ✅ | 续跑后 audit 通过（首轮曾记 external_call=12） |
| external_verification | 0 | 0 | ✅ | PROC 不暴露 verify tool |
| deterministic_truncation | — | — | — | PROC 不支持 |

**解读**

- **RUNTIME（4/6）**：stop / evidence_curation / verification_decision / external_verification — PROC 未能恢复 FULL 增量（\(P\le 0\) 或极低），或 benefit 依赖外部 verifier 信息。
- **INCONCLUSIVE（2/6）**：
  - `duplicate_evidence`：\(\Delta^{\text{full}}=+2.50\)pp 最强 sanity-check 信号之一；PROC 部分恢复（\(P=0.35\)）但 info-safe 失败（3% visibility violation）→ 待定 HYBRID。
  - `deterministic_truncation`：OFF recall 正常但 100q 无 truncation 事件，probe 无判别力；map builder 另有 `no_overlap_queries` bug。
- **置信度**：除 `external_verification`（HIGH）外，其余 paired CI 极宽 / LOW_CONFIDENCE；`stop`/`verification`/`external` 的 `n_proc_interventions=0`，RUNTIME 更像「PROC 探针弱」而非强证伪可蒸馏。
- **Dup sanity check**：PROC \(R=3.43\%\) > OFF \(2.55\%\) < FULL \(5.06\%\)，方向正确但未达 FULL；paired 12W/7L/81T。
- **最大 FULL 增益**：`external_verification` \(\Delta^{\text{full}}=+3.53\)pp（但 OFF recall 最低 1.52%）。

**830q Go/No-Go（不自动启动）**

| Capability | 建议 | 理由 |
|------------|------|------|
| external_verification | **优先** | \(\Delta^{\text{full}}\) 最大（+3.53pp）；CI 相对可用 |
| duplicate_evidence | **优先** | method sanity check；\(\Delta^{\text{full}}=+2.50\)pp；PROC 有方向性恢复 |
| stop_decision | 可选 | \(\Delta^{\text{full}}=+2.24\)pp；PROC 无干预，RUNTIME 分类 |
| evidence_curation | 低优先 | \(\Delta^{\text{full}}\) 仅 +1.23pp；PROC 显著差于 OFF |
| verification_decision | 低优先 | \(\Delta^{\text{full}}=+0.82\)pp，接近 noise |
| deterministic_truncation | **跳过** | 100q 无 truncation 事件；builder pairing bug |

**产物**

- `artifacts/capability/distillability_map.json`（2026-07-29 10:55）
- `outputs/scope_e0_distillability/E0_REPORT.md`（2026-07-29 10:55）
- 全量 episodes：`outputs/scope_e0_distillability/{cap}/{off,proc,full}/episodes.jsonl`

**备注：** Round-1 Dup-SDI 训练独立于 E0；E0 用 base model。830q **需人工批准后启动**（Round 2 结论：`RECOMMEND_830=false`，见下节）。

---

## Round 2 — Dup Behavioral Audit（0729-todo1）

**method 对应：** 诊断 Round-1 teacher-forced 拟合 vs \(H_{\min}\) 闭环行为脱节；引入 \(H_{\min,\text{v2}}\)、compact operation target、sample-normalized CE，做 100q 对照与训练消融。

**状态：** ✅ 代码 + Wave 1–3 完成 · ❌ Wave 4 闭环 eval 未完成 · `RECOMMEND_830=false`

**分支 / 产物根目录：** `scope/dup-round2-behavioral` · `outputs/scope_round2/` · `artifacts/datasets/round2_audit_100q/`

### 代码改动（Barrier 0，96 tests pass）

| Task | 内容 | 关键文件 |
|------|------|----------|
| A | Round-1 loss-mass audit | `training/scope/analyze_loss_mass.py` |
| B | Capability-level eval（operation P/R/F1、confusion） | `training/scope/eval_dup_capability.py` |
| C | \(H_{\min,\text{v2}}\) runtime | `harness/configs/modules_minimal_v2.yaml` |
| D | Compact target `KEEP_EVIDENCE` / `SKIP_DUPLICATE` | `training/scope/compact_target.py` |
| E | ActionRealizer operation→runtime | `harness/shadow/action_realizer.py` |
| F | `sample_normalized_action_ce` + `legacy_token_ce` + route balance | `training/scope/losses.py` |
| G | Stop 四象限统计 | `harness/capability/stop_calibration.py` |

编排：`scripts/scope_round2/run_all.sh` · 报告：`outputs/scope_round2/ROUND2_REPORT.md`

### 全局 GPU / 协议 Setting

| 项 | 值 |
|----|-----|
| Model | `Qwen2.5-7B-Instruct`（Base）；Round1 对照 `outputs/dup_sdi_round1/merged_hf` |
| Benchmark | BrowseComp+，冻结 **100q**（`artifacts/datasets/round2_audit_100q/query_manifest.json`，`SEED=42`，4×25 shard） |
| Retriever | BM25 |
| Rollout runtime | **`modules_minimal_v2.yaml`**（\(H_{\min,\text{v2}}\)：保留 search/BM25/verify tool + hard truncate；关闭 cognitive dedup/curation/stop policy） |
| vLLM | **1 model / 1 GPU，TP=1**；GPU0–7 → port 8800–8807 |
| max_turns / max_tokens / temperature | 35 / 2048 / 1.0 |
| 禁止项（本轮未跑） | 830q eval · E0 830 · capability weighting · Recovery · RL · Irrelevant |

---

#### ✅ Barrier 0 — 代码 + 诊断（2026-07-29）

**Round-1 Loss-Mass Audit**（755 samples，含 `dup_sdi_round1` + natural_100q）

| Route | Sample share | Target-token share |
|-------|-------------:|-------------------:|
| ENDORSE | 45.0% | **21.0%** |
| CORRECT | 54.8% | **79.0%** |

- 诊断：**sample balance ≠ loss-token balance**
- `verify_claim` 单独占 **60%** target token mass
- 产物：`outputs/scope_round2/diagnostics/round1_loss_mass.md`

---

#### ✅ Wave 1 — H_min_v2 100q Closed-loop（Base vs Old Round1）

**Setting**

| 项 | 值 |
|----|-----|
| Base | `Qwen2.5-7B-Instruct` |
| Old Round1 | `outputs/dup_sdi_round1/merged_hf` |
| Runtime | `modules_minimal_v2.yaml` |
| 分片 | GPU0–3 Base shard0–3；GPU4–7 Round1 shard0–3 |
| 脚本 | `training/scope_round2/hmin_v2_rollout.py` |

**结果 — 100q paired（Barrier 1）**

| 指标 | Base | Old Round1 | Δ | Paired W/L/T |
|------|------|------------|---|--------------|
| recall | 2.29% | 3.48% | +1.18pp | 12/7/81 |
| reward | 0.122 | 0.220 | +0.099 | 13/14/73 |
| trajectory_recall | 24.2% | 25.3% | +1.1pp | — |
| final_answer_recall | 2.95% | 6.37% | +3.42pp | — |
| mean_turns | 33.29 | 34.43 | +1.14 | — |
| **mean_n_curated** | **14.35** | **17.73** | **+3.38** | **62/32/6** |
| mean_n_pool | 288.94 | 286.93 | −2.01 | — |
| unique_evidence_ratio | 0.061 | 0.068 | +0.007 | 62/38/0 |
| duplicate_curate_rate | 0 | 0 | 0 | —（H_min_v2 未打点） |

**解读（Q2）**

- **curation 膨胀仍在**：mean_n_curated +3.38，paired CI 全为正 → 复现 smoke20「多 curate」方向（smoke20：11.15→20.65）
- **task 指标本切片未崩塌**：recall/reward 反而高于 Base（与 smoke20 的 recall/reward 下降不同，100q 方差大）
- DupCurateRate 指标在 \(H_{\min,\text{v2}}\) 上未 instrumented，**mean_n_curated 为行为代理**

产物：`outputs/scope_round2/hmin_v2_base/merged/` · `hmin_v2_round1/merged/` · `eval/base_vs_round1_100q.md`

---

#### ✅ Wave 2 — Same-State Shadow + Stop Calibration + Round1 Capability Re-eval

**Dup shadow（Base @ H_min_v2 decision states）**

- 四 shard 并行 labeling → `outputs/scope_round2/dup_shadow/shard0–3/`
- 修复 compact target 解析后重建数据集

**Stop Calibration 100q（H_min_v2）**

| 象限 | Count |
|------|------:|
| STOP→STOP | 0 |
| STOP→CONTINUE | 13 |
| CONTINUE→STOP | 0 |
| CONTINUE→CONTINUE | 3316 |
| n_decision_points | 3329 |

- **bilateral_coverage: False** — 仍缺 CONTINUE→STOP 监督质量
- 产物：`outputs/scope_round2/stop_calibration/stop_calibration_100q.md`

**Round1 capability re-eval（新指标，valid 77）**

| 指标 | 值 |
|------|-----|
| teacher_forced_token_acc | 93.9% |
| action_match_rate | 26.0% |
| route CORRECT accuracy | 94.9% |
| route ENDORSE accuracy | 82.8% |

---

#### ✅ Barrier 2 — Round 2 数据集

**Setting**

| 项 | 值 |
|----|-----|
| 来源 | Base @ H_min_v2 same-state shadow |
| Capability | `duplicate_evidence` only |
| Target 格式 | compact `SKIP_DUPLICATE` JSON |
| Split | query-level，train 257 / valid 30 |

**分布（局限）**

| 项 | 值 |
|----|-----|
| KEEP / SKIP | 0 / **287** |
| ENDORSE / CORRECT | **0 / 287** |
| visibility / schema violations | 0 / 0 |

⚠️ 仅捕获 duplicate-curate **CORRECT** 点，无 ENDORSE/KEEP → endorse-only 消融无法运行

产物：`artifacts/datasets/dup_sdi_round2/`

---

#### ✅ Wave 3 — 8 路训练消融（Barrier 3）

**共同 Setting**

| 项 | 值 |
|----|-----|
| Base model | `Qwen2.5-7B-Instruct` |
| Method | LoRA r=16, α=32 |
| lr / epochs / batch / grad_accum | 2e-5 / 3 / 4 / 4 |
| max_length | 4096 |
| KL coef | 0.01 |
| Dataset | `dup_sdi_round2`（同上） |

**Variant 分配**

| GPU | Variant | loss_mode | compact | route_balance | 备注 |
|-----|---------|-----------|---------|---------------|------|
| 0 | round2_main | sample_normalized | ✅ | ✅ | 主模型 |
| 1 | round2_legacy_token_ce | legacy_token_ce | ❌ | ❌ | 长序列 token CE 对照 |
| 2 | round2_full_action_sample_norm | sample_normalized | ❌ | ❌ | 完整 action + sample norm |
| 3 | round2_no_route_balance | sample_normalized | ✅ | ❌ | |
| 4 | round2_endorse_only | sample_normalized | ✅ | — | **FAILED**（0 train samples） |
| 5 | round2_correct_only | sample_normalized | ✅ | — | CORRECT only filter |
| 6 | round2_main_seed43 | 同 main | ✅ | ✅ | seed=43 |
| 7 | round2_main_seed44 | 同 main | ✅ | ✅ | seed=44 |

**结果 — round2_main offline capability（30 valid，compact prompt）**

| 指标 | 值 |
|------|-----|
| valid loss | 0.275 |
| parse_rate | **1.0** |
| operation_accuracy | 0.50 |
| SKIP_DUPLICATE recall / F1 | **0.50 / 0.67** |
| KEEP_EVIDENCE recall | n/a（valid 全 CORRECT） |
| teacher_forced_token_acc | 94.7%（不作为成功标准） |

\* 其余 variant Barrier-3 批量 eval 因 prompt 不匹配 greedy_parse=0；仅 main 经 compact prompt 重评。

产物：`outputs/scope_round2/training/round2_*/` · `eval/round2_training_comparison.md`

---

#### ❌ Wave 4 — Closed-loop 100q（未完成）

**计划：** 8 模型 × 同 manifest × \(H_{\min,\text{v2}}\) × merge LoRA → rollout

**阻塞：** 推理路径未接入 ActionRealizer（compact operation → runtime action）；训练后的 LoRA 未在闭环中验证 Dup 主指标（DupCurateRate / FalseSkipRate）

产物占位：`outputs/scope_round2/eval/round2_closed_loop_100q.md`（partial）

---

### Round 2 五问结论（0729-todo1 §十三）

| # | 问题 | 结论 |
|---|------|------|
| Q1 | sample balance ≠ loss-token balance？ | **是**。CORRECT 79% token vs 55% sample；verify_claim 占 60% tokens |
| Q2 | Round1 @ H_min_v2 复现 smoke20？ | **部分**。mean_n_curated +3.38 确认多 curate；本 100q recall/reward 未降 |
| Q3 | compact target 改善 operation？ | **离线部分**。SKIP recall 50%，闭环未测 |
| Q4 | legacy vs sample-norm vs compact？ | **方向支持 compact+sample-norm**；缺公平重评与 ENDORSE 数据 |
| Q5 | Endorse vs Correct 贡献？ | **无法回答**（0 ENDORSE） |

### 最终判定

```text
ROOT_CAUSE_ROUND1 = 长序列 token CE（尤其 verify_claim）使 loss mass 偏向 CORRECT；
                    teacher-forced 高拟合未转化为 H_min_v2 闭环行为，Round1 仍抬高 mean_n_curated

ROUND2_POSITIVE_SIGNAL = false

RECOMMEND_830 = false

NEXT_ACTION = (1) 修复 same-state shadow 产生 ENDORSE/KEEP 双侧标签
              (2) 推理接入 ActionRealizer 后完成 Wave 4
              (3) Stop selector 提升 CONTINUE→STOP 覆盖
              (4) 平衡数据后重训再议 830
```

---

## Round 3 — Bilateral Duplicate Capability Internalization（0729-todo2）

**method 对应：** 在 student 真实访问的 evidence-admission decision points 上构造 KEEP/SKIP 双侧监督，用 `operation_ce` 直接优化 operation decision，并统一 train/inference 的 typed action interface（`DupOperationRuntime` + `ActionRealizer`），验证能否在 \(H_{\min,\text{v2}}\) closed-loop 中降低 DuplicateCurateRate 且不显著恶化 FalseSkipRate。

**状态：** ✅ Barrier A–C 完成 · 🔄 Wave4 / Closed-loop 100q 进行中 · `RECOMMEND_830=false`

**分支 / 产物根目录：** `scope/dup-round3-bilateral` · `outputs/scope_round3/` · `artifacts/datasets/dup_sdi_round3/` · git `3e95fad`

### 代码改动（Barrier A，109 tests pass）

| 模块 | 内容 | 关键文件 |
|------|------|----------|
| Selector | error-triggered → **decision-triggered**（curate 时 `evidence_admission`） | `harness/capability/selectors.py` |
| Decision point | `DupDecisionPoint` 元数据（capability_id / decision_type / candidate_id） | `harness/capability/dup_decision_point.py` |
| Shadow | 双侧 `KEEP_EVIDENCE` / `SKIP_DUPLICATE` only | `harness/shadow/dup_bilateral_shadow.py` |
| ActionRealizer | `realize_operation`：KEEP→curate，SKIP→不写入 curated | `harness/shadow/action_realizer.py` |
| Operation objective | 长度归一化 verbalizer CE（train/inference 共用 `score_operations`） | `training/scope/operation_scorer.py` |
| Loss | 新增 `operation_ce`；保留 `legacy_token_ce` / `sample_normalized_action_ce` | `training/scope/losses.py` |
| Inference | `DupOperationRuntime` + vLLM scorer（无第二 LLM） | `training/scope/dup_operation_runtime.py` |
| Telemetry | admission events → DupCurateRate / FalseSkipRate | `training/scope/dup_telemetry.py` |

编排：`scripts/scope_round3/run_all_8gpu.sh` · 报告：`outputs/scope_round3/ROUND3_REPORT.md`

### 全局 GPU / 协议 Setting

| 项 | 值 |
|----|-----|
| Model | `Qwen2.5-7B-Instruct`（Base）；Round1 `outputs/dup_sdi_round1/merged_hf`；Round2 `outputs/scope_round2/training/round2_*` |
| Benchmark | BrowseComp+，冻结 **100q**（`artifacts/datasets/round2_audit_100q/query_manifest.json`） |
| Retriever | BM25 |
| Rollout runtime | **`modules_minimal_v2.yaml`**（\(H_{\min,\text{v2}}\)） |
| vLLM | **1 model / 1 GPU，TP=1**；Round3 port **8900–8907**（Wave4）/ **8910–8927**（closed-loop） |
| max_turns / max_tokens / temperature | 35 / 2048 / 1.0 |
| 禁止项（本轮未跑） | 830q · E0 830 · capability weighting · Recovery · RL · Premature Stop 训练 · Irrelevant |

---

#### ✅ Barrier A — 代码 + 单测（2026-07-29）

- `pytest tests/scope/`：**109 passed**
- 验证项：unique→KEEP、duplicate→SKIP、ENDORSE/CORRECT 路由、ActionRealizer 确定性映射、visibility=0、train/inference 共用 scorer
- 首轮 Wave4 因 `CurateTool` import 路径错误失败，已修复（`training.train_rl.CurateTool`）

---

#### ✅ Barrier B — 双侧数据集（2026-07-29）

**Setting**

| 项 | 值 |
|----|-----|
| 来源 | Base @ H_min_v2 decision states（Round2 100q rollout 重切 8 shard） |
| Shadow | `DupBilateralShadow`（decision-triggered，非 duplicate-suspect 触发） |
| Split | query-level：**train 80q / valid 20q**（1807 / 522 events） |
| Gate | visibility_violation=0（3 条预过滤）· shadow_mutation=0 · schema_invalid=0 |

**分布（双侧，Round2 单侧问题已修复）**

| 项 | Count |
|----|------:|
| KEEP_EVIDENCE | **1784** |
| SKIP_DUPLICATE | **545** |
| ENDORSE | **1784** |
| CORRECT | **545** |
| keep/skip ratio | 3.27:1 |
| endorse/correct ratio | 3.27:1 |

```text
ROUND3_DATA_GO = true
```

产物：`artifacts/datasets/dup_sdi_round3/` · `bilateral_dataset_report.md` · `bilateral_dataset_stats.json`

---

#### ✅ Barrier C — 8 路训练消融（2026-07-29）

**共同 Setting**

| 项 | 值 |
|----|-----|
| Base model | `Qwen2.5-7B-Instruct` |
| Method | LoRA r=16, α=32 |
| lr / epochs / batch / grad_accum | 2e-5 / 3 / 4 / 4 |
| max_length | 4096 |
| KL coef | 0.01（`operation_ce` 路径 KL≈0） |
| Dataset | `dup_sdi_round3`（1807 train / 522 valid） |
| 优化 steps | ~1350 / variant（operation_ce）；correct-only ~330 |

**Variant 分配**

| GPU | Variant | loss_mode | 备注 |
|-----|---------|-----------|------|
| 0 | round3_op_main_seed42 | **operation_ce** | route+class balance，seed=42 |
| 1 | round3_op_main_seed43 | operation_ce | 同 main，seed=43 |
| 2 | round3_op_main_seed44 | operation_ce | 同 main，seed=44 |
| 3 | round3_compact_json_sample_norm | sample_normalized | compact JSON 对照 |
| 4 | round3_legacy_full_action_token_ce | legacy_token_ce | 表面形式 imitation 对照 |
| 5 | round3_correct_only_op | operation_ce | CORRECT only |
| 6 | round3_endorse_only_op | operation_ce | ENDORSE only |
| 7 | round3_op_no_balance | operation_ce | 无 class/route balance |

**训练 loss（epoch 3 末）**

| Variant | final_train_loss |
|---------|-----------------:|
| round3_op_main_seed42/43/44 | ~0.50 |
| round3_op_no_balance | 0.519 |
| round3_compact_json | 0.226 |
| round3_legacy_token_ce | 0.227 |
| round3_correct_only | **0.002** |
| round3_endorse_only | **≈0** |

产物：`outputs/scope_round3/training/round3_*/` · merged：`outputs/scope_round3/merged/`

---

#### ✅ 训练前 Baselines + Offline Capability Eval（valid 522）

**B0 — Majority（永远 KEEP）**

| KEEP F1 | SKIP F1 | macro-F1 |
|--------:|--------:|---------:|
| 1.000 | 0.000 | 0.500 |

**B1 — Base operation_ce（未训练，restricted verbalizer scorer）**

| KEEP F1 | SKIP F1 | macro-F1 | op_acc |
|--------:|--------:|---------:|-------:|
| 1.000 | 0.000 | 0.500 | 81.0% |

**B2 — Round2 main（compact JSON，公平 operation eval）**

| KEEP F1 | SKIP F1 | macro-F1 | SKIP recall |
|--------:|--------:|---------:|------------:|
| 0.000 | **0.697** | 0.349 | **53.5%** |

**Round3 全 variant offline（operation-level，522 valid）**

| Variant | KEEP F1 | SKIP F1 | macro-F1 | balanced acc | 备注 |
|---------|--------:|--------:|---------:|-------------:|------|
| round3_op_main seed42/43/44 | 1.000 | **0.000** | 0.500 | 0.500 | **≡ Base，全预测 KEEP** |
| round3_op_no_balance | 1.000 | 0.000 | 0.500 | 0.500 | 同上 |
| round3_endorse_only | 1.000 | 0.000 | 0.500 | 0.500 | 对照：全 KEEP |
| round3_correct_only | 0.000 | **1.000** | 0.500 | 0.500 | 对照：全 SKIP |
| **round3_compact_json** | 0.983 | **0.061** | **0.522** | 0.522 | 唯一略优于 Base |
| round3_legacy_token_ce | 0.986 | 0.020 | 0.503 | 0.503 | token CE，SKIP 极弱 |

**解读**

- **operation_ce 主模型训练塌缩为 majority KEEP**：三 seed macro-F1=0.500，SKIP recall=0%，与 Base B1 完全相同；correct-only / endorse-only 对照证明训练管线本身可学单侧
- **compact JSON + sample-norm 略优**：macro-F1 0.522，SKIP recall 6.1%，但仍远低于 Round2 main 的 53.5% SKIP recall（Round2 valid 仅 30 CORRECT-only 样本，不可直接比）
- **teacher_forced_token_acc 不可信**：compact/legacy ~91%，operation_ce main ~47%，与 operation 行为脱节

产物：`outputs/scope_round3/eval/baselines.json` · `outputs/scope_round3/eval/offline_capability.json`

---

#### 🔄 Wave 4 Diagnostic — 四 checkpoint plumbing（进行中 / 首轮失败已修复）

**计划：** Base / Round1 / Round2-main / Round2-legacy × dup-operation + ActionRealizer + telemetry，GPU0–7 各 2 shard（port 8900–8907）

| 阶段 | 状态 |
|------|------|
| 首轮（07-29 14:52） | ❌ `ImportError: CurateTool from harness.tools` |
| 修复后重启 | 🔄 rollout 启动；`comparison.json` variants 仍为空（shard 未完成） |

**Barrier（待满足）：** typed operation 可执行 · telemetry 完整 · ActionRealizer 工作 · 无 hidden fallback

产物占位：`outputs/scope_round3/wave4_diagnostic/`

---

#### 🔄 Closed-loop 100q — Dup 行为主指标（进行中）

**Setting**

| 项 | 值 |
|----|-----|
| 协议 | 同 manifest · BM25 · \(H_{\min,\text{v2}}\) · dup-operation + ActionRealizer |
| 顺序 | Base 100q（GPU0）→ 8 trained variant 并行（GPU0–7） |
| 脚本 | `scripts/scope_round3/run_post_train_8gpu.sh` |

**当前进度（07-29 17:15）**

- Base closed-loop：**shard0 进行中**（query 631，~turn 24）
- 8 variant closed-loop：**未开始**（待 Base 完成后并行）
- 暂无完整 `closed_loop/*/merged/summary.json`

**代理参考（Round2 H_min_v2 Base，无 dup-operation telemetry）**

| 指标 | Base（Round2 Wave1） |
|------|---------------------:|
| recall | 2.29% |
| reward | 0.122 |
| mean_n_curated | 14.35 |
| duplicate_curate_rate | 未 instrumented |

产物（进行中）：`outputs/scope_round3/closed_loop/` · 日志：`outputs/scope_round3/logs/post_train_master.log`

---

### Round 3 研究问题结论（0729-todo2 §一）

> 双侧监督 + operation_ce + 统一 action interface 能否让 duplicate_evidence 在 \(H_{\min,\text{v2}}\) closed-loop 中真正降低 DuplicateCurateRate？

| 层面 | 结论 | 依据 |
|------|------|------|
| **数据 / Selector（H3）** | ✅ **已解决** | KEEP=1784, SKIP=545, ENDORSE/CORRECT 全非零；`ROUND3_DATA_GO=true` |
| **Train/Inference 一致（H2）** | ✅ **已实现** | 统一 `operation_ce` scorer + `DupOperationRuntime` + `ActionRealizer`；Wave4/CL 验证中 |
| **Offline capability** | ❌ **未通过** | main operation_ce macro-F1=0.500 ≡ Base；SKIP recall=0% |
| **Closed-loop behavior** | 🔄 **待完成** | DupCurateRate / FalseSkipRate 尚无 trained-model 结果 |
| **Task retention** | 🔄 **待完成** | 需 paired 100q |

### 根因假设更新（0729-todo2 §十七）

| 假设 | 判定 | 说明 |
|------|------|------|
| H1 token-loss-mass distortion | **PARTIALLY_SUPPORTED** | legacy/compact JSON 仍高 token acc，SKIP recall 弱 |
| H2 training/inference action mismatch | **SUPPORTED** | Round3 已统一 interface；Round2 根因之一 |
| H3 selector-induced one-sided supervision | **SUPPORTED** | Round2 0 KEEP/0 ENDORSE → Round3 双侧修复 |
| H4 operation-value supervision weakness | **SUPPORTED** | operation_ce 主模型未学到 SKIP，塌缩 majority |

### 最终判定（暂定，待 closed-loop 完成后更新）

```text
ROUND3_POSITIVE_SIGNAL = false   # offline：operation_ce 未超 Base；SKIP recall=0%

RECOMMEND_830 = false

NEXT_ACTION = (1) 调查 operation_ce 塌缩根因（verbalizer prior / 梯度 / eval 一致性）
              (2) 完成 Wave4 + closed-loop 100q，更新 DupCurateRate / FalseSkipRate
              (3) 对比 compact JSON vs operation_ce 在闭环行为上的差异
              (4) 在 Dup 最小 positive signal 前不扩 830 / 多能力 / weighting
```

---

<details>
<summary>历史记录：07-28 首轮失败 → 07-29 卡住恢复（已归档）</summary>

#### 首轮失败（07-28 13:42–14:30）

- 大量 OFF/PROC：`Connection error` / PROC `run_information_safe_gates(... artifact=)` 签名 bug。
- 旧 `E0_REPORT`（14:30）不可信（多模式 `errors=1.0`、`turns=0`）。

#### nohup 续跑（07-28 17:14 起）

| 时间 | 任务 | 结果 |
|------|------|------|
| 17:15 | duplicate_evidence/proc | OOM Kill，误标 DONE，episodes 空 |
| 17:15–17:19 | stop_decision/proc | ✅ 100 |
| 17:19–17:52 | evidence_curation/proc | ✅ 100 |
| 17:52–18:02 | verification_decision/off | ✅ 100 |
| 18:02–18:46 | verification_decision/proc | 98/100（2 error） |
| 18:46–19:19 | external_verification/off | ✅ 100 |
| 19:19–19:27 | external_verification/proc | 推进至 ~20–23/100 |
| 19:27–07-29 08:44 | external_verification/proc | **卡住** 23/100（PID 3960060，~13h） |

#### 中间态（07-29 08:43）

仅 `stop_decision` / `evidence_curation` 三模式齐全；dup PROC 空；truncation OFF 失败；external PROC 卡住。08:44 kill 后 truncation OFF 续跑，09:19 watch relaunch 补齐其余缺格，10:55 冻结（见上）。

</details>

---

## 训练主循环（每 iteration）

### Step 1 — Pure Student Rollout（\(\tau^- \sim \pi_\theta \mid H_{\min}\)）

**method 对应：** 学生在 Minimal Runtime 上 on-policy rollout，收集真实访问状态。

**状态：** 📋 TODO（正式数据管线）

**Setting（计划）**

| 项 | 值 |
|----|-----|
| Model | `Qwen2.5-7B-Instruct` |
| Benchmark | BrowseComp+，`LIMIT=100` → 全量 830 |
| Retriever | BM25 |
| Rollout runtime | `modules_minimal.yaml` |
| max_turns / max_tokens | 35 / 2048 |
| temperature | 1.0 |
| Scope config | `configs/scope/sdi_dup_premature.yaml` |
| Capabilities | `duplicate_evidence`, `premature_stop` |
| 用途 | 替代 Full v2 轨迹，对齐 student-state supervision |

**已完成的相关实验（偏离 method，仅作协议验证）：**

#### ✅ 2026-07-28 10:28 — 协议 Smoke（20 题，Full v2 rollout）

**Setting**

| 项 | 值 |
|----|-----|
| Model | `Qwen2.5-7B-Instruct` |
| Benchmark | BrowseComp+，`LIMIT=20`，`SEED=42` |
| Retriever | BM25 |
| Harness（rollout） | `modules_full_v2.yaml` ⚠️ 非 \(H_{\min}\) |
| Scope config | `configs/scope/sdi_dup_premature.yaml` |
| Capabilities | `duplicate_evidence`, `premature_stop` |
| max_turns / max_tokens | 35 / 2048 |
| temperature | 1.0 |
| GPU | 0–3，vLLM port 8774 |

**结果**

| 指标 | 值 |
|------|-----|
| 完成题数 | 20/20 |
| events / trainable | 123 / 123 |
| Dup | 103（ENDORSE 56, CORRECT 47） |
| Premature | 20（**ENDORSE 0, CORRECT 20**） |
| leakage / mutation | 0 / 0 |

产物：`outputs/scope_v3_protocol_smoke20/`

#### ✅ 2026-07-28 11:53 — Natural 100q Audit（Full v2 rollout）

**Setting**

| 项 | 值 |
|----|-----|
| Model | `Qwen2.5-7B-Instruct` |
| Benchmark | BrowseComp+，`LIMIT=100`，`SEED=42` |
| Retriever | BM25 |
| Harness（rollout） | `modules_full_v2.yaml` ⚠️ 非 \(H_{\min}\) |
| Scope config | `configs/scope/sdi_dup_premature.yaml` |
| Capabilities | `duplicate_evidence`, `premature_stop` |
| max_turns / max_tokens | 35 / 2048 |
| temperature | 1.0 |
| GPU | 0–3，vLLM port 8775 |

**结果 — 协议质量**

| 指标 | 值 |
|------|-----|
| 完成题数 | 100/100 |
| events / trainable | 755 / 754 |
| visibility_violation / shadow_mutation | 0 / 0 |

**结果 — Capability 监督**

| Capability | calls | P/R | 分布 |
|------------|-------|-----|------|
| Duplicate Evidence | 655 | 1.00 / 1.00 | ENDORSE 340 + CORRECT 315 |
| Premature Stop | 100 | 1.00 / 0.99 | **ENDORSE 0 + CORRECT 99** + IGNORE 1 |
| Irrelevant Evidence | 773 | — | 全部 IGNORE（Round-1 不训） |

Premature 细分：`n_valid_stop=0`，`n_bad_stop=100`，`premature_all_correct_risk=true`

产物：`outputs/scope_v3_audit_100q/natural_100q/`

---

### Step 2 — DecisionState 构建（\(d_t = \psi(s_t)\)）

**method 对应：** 将交互状态压缩为统一 DecisionState，满足 \(\operatorname{Info}(d_t) \subseteq \operatorname{Info}(s_t)\)。

**状态：** ✅ 已随 v3 协议在线验证（见 Step 1 两条 rollout 实验）

---

### Step 3 — Same-State Shadow Guidance（\(z_t^m = h_m(d_t)\)）

**method 对应：** 同状态查询 typed module，产出局部 artifact（非完整轨迹）。

**状态：** ✅ 已验证（Smoke 20q + Audit 100q）

当前已接入 capabilities：`duplicate_evidence`，`premature_stop`（selector 待修，见 Stop Calibration）

---

### Step 4 — Information-Safe Gate（\(M_t^m\)）

**method 对应：** visibility / schema / executable / module 四重 mask。

**状态：** ✅ 已验证

| 实验 | leakage | shadow_mutation |
|------|---------|-----------------|
| Smoke 20q | 0 | 0 |
| Audit 100q | 0 | 0 |

#### ✅ 2026-07-28 11:06 — Verifier 可靠性探针（离线，不进训练集）

验证 \(V_m\) 在 synthetic valid-stop 上能否给出 ENDORSE（隔离 selector 问题）。

**Setting**

| 项 | 值 |
|----|-----|
| Model | —（离线合成，无 rollout） |
| 数据 | 合成 `DecisionState` 探针，`n=24` |
| Harness | — |
| Scope config | `configs/scope/sdi_dup_premature.yaml` |
| Capability | `premature_stop` only |
| train_mask | 0（明确不进训练） |

**结果**

| 指标 | 值 |
|------|-----|
| ENDORSE / CORRECT | 24 / 0（100% ENDORSE） |
| trainable | 0 |

**结论：** Verifier 可靠；自然轨迹缺 positive-stop 是 selector 单侧触发，非 verifier 失效。

产物：`outputs/scope_v3_audit_100q/targeted_valid_stop/`

---

### Step 5 — Verified Decision Routing（ENDORSE / CORRECT）

**method 对应：** module endorse → \(\tilde a_t = a_t^-\)；reject + verified → \(\tilde a_t = a_t^+\)。

**状态：** 🔄 部分完成

| Capability | 监督质量 | 状态 |
|------------|----------|------|
| `duplicate_evidence` | P/R=1.0，双向均衡，655 samples | ✅ 可用于训练 |
| `premature_stop` | 99% CORRECT，0 ENDORSE | ⚠️ 分布不可靠，待 Stop Calibration |
| `irrelevant_evidence` | 全 IGNORE | ⏸ Round-1 不训 |

#### 🔄 2026-07-28 — Stop Calibration 重构

**目的：** Premature selector 从单侧 `bad stop → continue` 改为双侧 Stop-vs-Continue 四象限监督。

**Setting（计划）**

| 项 | 值 |
|----|-----|
| Model | `Qwen2.5-7B-Instruct` |
| Benchmark | BrowseComp+，`LIMIT=20` smoke → `LIMIT=100` audit |
| Retriever | BM25 |
| Harness | `modules_full_v2.yaml` |
| Scope config | `configs/scope/sdi_dup_premature.yaml`（`stop_calibration: true`） |
| Capabilities | `duplicate_evidence`, `premature_stop` |
| max_turns / max_tokens | 35 / 2048 |
| temperature | 1.0 |
| 代码改动 | `stop_calibration.py`, `selectors.py`, `verification_shadow.py` |

| 项 | 状态 |
|----|------|
| 代码 | ✅ 单测 96/96 通过（含四象限） |
| 100q audit @ H_min_v2 | ✅ 2026-07-29（见 Round 2 §Wave 2） |
| 20q smoke → Full v2 audit 重跑 | 📋 TODO |

**2026-07-29 — H_min_v2 Stop Calibration 100q 结果**

- n_decision_points: 3329；STOP→CONTINUE: 13；其余 CONTINUE→CONTINUE: 3316
- **bilateral_coverage: False**（CONTINUE→STOP = 0）
- 本轮仅 audit，**不参与 Round 2 Dup 训练**

❌ **不做：** 原样扩大 Premature audit（Full v2 自然轨迹已知分布问题）；本轮不训 Premature

---

### Step 6 — Capability Weighting（\(w_t^m = P_m U_m (1-\rho_m)\)）

**method 对应：** 更新 \(U_m, G_m, \rho_m\)，动态样本/模块权重。

**状态：** 📋 TODO

**Setting（计划）**

| 项 | 值 |
|----|-----|
| Model | `Qwen2.5-7B-Instruct` |
| Benchmark | BrowseComp+ |
| 权重方案 | uniform · \(U_m\) · \(U_m(1-\rho_m)\) · \(P_m U_m (1-\rho_m)\) |
| 对比实验 | E3 |

| 子项 | 状态 |
|------|------|
| 估计各 capability 的 \(U_m\)（audit 已有粗统计） | 🔄 Dup ✅；Premature 待修 |
| Held-out \(\rho_m\)（内化率） | 📋 TODO |
| 加权训练 vs uniform filter（E3） | 📋 TODO |

**当前替代方案：** Dup Round 1 仅用 `capability=duplicate_evidence, train_mask=1` filter，无 \(P_m U_m(1-\rho_m)\) 权重。

---

### Step 7 — Shadow-first, Recovery-on-Demand

**method 对应：** 高影响纠正处 fork，执行一次 \(a_t^+\)，学生继续 \(K\) 步。

**状态：** ⏸ 暂缓（method 建议首版 toy 可先不加 recovery）

**Setting（计划）**

| 项 | 值 |
|----|-----|
| Model | `Qwen2.5-7B-Instruct` |
| Benchmark | BrowseComp+ |
| Rollout runtime | `modules_minimal.yaml` |
| Recovery 步数 \(K\) | TBD |
| 触发条件 | \(\delta_t^m > \tau_{\text{recover}}\) |
| 对比实验 | E4 |

📋 TODO：当 Minimal Runtime rollout 出现明显 premature stop / dead-end 后再启用

---

### Step 8 — Optimize（\(\mathcal{L} = \mathcal{L}_{\text{SDI}} + \xi \mathcal{L}_{\text{stab}}\)）

**method 对应：** Action-level CE + 可选 KL；首版不含 \(\mathcal{L}_{\text{RL}}\)、\(\mathcal{L}_{\text{rec}}\)。

#### ✅ 2026-07-28 12:38 — Dup-only SDI Round 1（训练 + Capability Eval）

**目的：** 验证「可学习 Harness decision 能否独立内化」（method 首版 \(\mathcal{L}_{\text{SDI}}+\mathcal{L}_{\text{stab}}\)）。

**Setting — 监督数据**

| 项 | 值 |
|----|-----|
| Model（rollout 来源） | `Qwen2.5-7B-Instruct` |
| Benchmark | BrowseComp+，`LIMIT=100`，`SEED=42` |
| 来源 | `natural_100q/samples.jsonl` |
| Filter | `duplicate_evidence`, `train_mask=1` |
| Split | query-level，`valid_fraction=0.1`，`seed=42` |
| n_samples / train / valid | 655 / 578 / 77 |
| Route | ENDORSE 340 + CORRECT 315 |

**Setting — 训练**

| 项 | 值 |
|----|-----|
| Base model | `Qwen2.5-7B-Instruct` |
| Method | LoRA（r=16, α=32） |
| Loss | Action-level CE + KL（coef=0.01） |
| lr / epochs | 2e-5 / 3 |
| batch / grad_accum | 4 / 4 |
| max_length | 4096 |
| Scope config | `configs/scope/sdi_dup_only.yaml` |
| GPU | 4（单卡），~13 min（430 steps） |

**Setting — Capability Eval**

| 项 | 值 |
|----|-----|
| 脚本 | `training/scope/eval_dup_capability.py` |
| Eval set | valid 77 条 Dup 样本 |
| Decode | greedy，`max_new_tokens=64`，首行 JSON 截取 |
| 加载方式 | base + LoRA adapter |

**结果 — 训练曲线（节选）**

| 阶段 | loss |
|------|------|
| epoch 1 初期 | ~1.25 |
| epoch 1 末期 | ~0.19–0.42 |
| epoch 3 末期 | ~0.13–0.45 |

**结果 — Capability Eval**

| 指标 | 值 |
|------|-----|
| valid loss | **0.227** |
| teacher_forced_token_acc | **93.9%** |
| greedy_parse_rate | 100% |
| action_match_rate | 26.0% |
| endorse_accuracy（38 条） | 13.2% |
| correct_accuracy（39 条） | 38.5% |

**解读：** SDI 目标下 teacher-forced 93.9% 说明内化信号明确；greedy 精确匹配偏低与 doc_id 列表长、decode 易偏有关，不等同训练失败。首轮 eval 曾误将 adapter 当 base 加载（loss 1.54 / match 0%），已修复。

产物：
- `artifacts/datasets/dup_sdi_round1/`
- `outputs/dup_sdi_round1/`（LoRA adapter）
- `outputs/dup_sdi_round1/capability_eval.json`

#### ✅ 2026-07-28 13:42 — Minimal Runtime Smoke20（Dup-SDI vs Base）

**目的：** 在 \(H_{\min}\) 下对比 Dup Round 1 训练后模型与 base；观察重复 curate 是否减少、总体 recall 是否不伤。

**Setting**

| 项 | 值 |
|----|-----|
| Base model | `Qwen2.5-7B-Instruct` |
| Trained model | `outputs/dup_sdi_round1/merged_hf`（LoRA merge） |
| Benchmark | BrowseComp+ 前 **20** 题（`LIMIT=20`，`SPLIT=all`） |
| Retriever | BM25 |
| Runtime | `modules_minimal.yaml`（V8D 全关） |
| Scope config | `configs/scope/minimal_runtime.yaml` |
| max_turns / max_tokens / temperature | 35 / 2048 / 1.0 |
| parallel | 2 |
| GPU | 4–7（vLLM TP=4） |
| 脚本 | `scripts/run_dup_sdi_minimal_runtime_smoke20.sh` |
| Phase 0 全量对照 | Minimal recall **2.45%**，reward **0.121**（830 题，非同批） |

**结果 — 同 20 题对比**

| 指标 | Base | Dup-SDI Trained | Δ |
|------|------|-----------------|---|
| recall | **3.06%** | 0.71% | −2.35pp |
| reward | **0.137** | 0.013 | −0.124 |
| trajectory_recall | 17.0% | 15.3% | −1.7pp |
| final_answer_recall | 5.0% | 0.0% | −5.0pp |
| mean_turns | 30.1 | 34.6 | +4.5 |
| **mean_n_curated** | **11.15** | **20.65** | **+9.5** |
| mean_n_pool | 276.2 | 296.5 | +20.3 |
| error_rate | 0% | 0% | — |

**解读（smoke20，高方差）**

```text
本轮未观察到 Dup 能力内化的 positive signal：
  - recall / reward 均低于同批 base
  - mean_n_curated 反而上升（11.2 → 20.7），未见重复 curate 减少
可能原因：655 条局部监督 + 3 epoch LoRA；action-span CE 未直接优化 runtime curate 行为；样本量仅 20
需全量 830 或更大 smoke 才能下结论；当前不支持「Dup-only 已改善 Minimal Runtime」
```

**2026-07-29 更新 — Round 2 @ H_min_v2 100q 确认 curation 膨胀方向**

- 同冻结 100q manifest：mean_n_curated **14.35 → 17.73**（+3.38，paired W/L/T=62/32/6）
- Loss-mass 根因：CORRECT 占 79% token mass（verify_claim 60%）
- **RECOMMEND_830=false**；830q 暂缓（见 Round 2 节）

产物：
- `outputs/dup_sdi_round1/minimal_runtime_smoke20/base/`
- `outputs/dup_sdi_round1/minimal_runtime_smoke20/trained/`
- `outputs/dup_sdi_round1/minimal_runtime_smoke20/compare_smoke20.json`

#### 📋 TODO — Minimal Runtime 全量 830（E6 / Retention）

**状态：** ⏸ 暂缓（Round 2：`RECOMMEND_830=false`）

**Setting（计划）**

| 项 | 值 |
|----|-----|
| Model | `Qwen2.5-7B-Instruct` + Dup Round 1 LoRA |
| Benchmark | BrowseComp+ 全量 830 题 |
| Retriever | BM25 |
| Runtime | `modules_minimal.yaml` |
| max_turns / max_tokens | 35 / 2048 |
| temperature | 1.0 |
| 对比 | trained LoRA vs base on Minimal Runtime |
| 基线 | Phase 0 Minimal recall **2.45%**，Full v2 **3.80%** |
| 指标 | recall，reward，\(\text{Retention}_{\text{dup}}\) |

---

### Step 9 — Module Lifecycle（内化 → 降权 → 退役）

**method 对应：** 据 \(P_m, U_m, \rho_m\) 决定 internalize / retire / hybrid runtime。

**状态：** 📋 TODO（E0 \(P_m\) 已有 100q 冻结结果，见 Stage 0；仍缺 held-out \(\rho_m\) 与正式 Minimal eval）

---

## 实验设计消融（method §4）

### E0 — Module Distillability Map

**状态：** ✅ 已完成（100q audit，2026-07-29 10:55 冻结；见 Stage 0）

---

### E1 — Full Harness Distillation vs Same-State Local Distillation

**状态：** 📋 TODO

**Setting（计划）**

| 项 | 值 |
|----|-----|
| Model | `Qwen2.5-7B-Instruct` |
| Benchmark | BrowseComp+ |
| Retriever | BM25 |
| max_turns / max_tokens | 35 / 2048 |
| temperature | 1.0 |
| 对比方法 | SFT on Harness trace · OPHSD-style full context · same-state local label · same-state + info-safe gate |

重点：fresh corpus · unseen facts · citation hallucination · action decision accuracy。

---

### E2 — 为什么需要 Correct，而不只是 Endorse

**状态：** 🔄 进行中

| 子项 | 状态 |
|------|------|
| Dup 双向 ENDORSE/CORRECT 监督 | ✅ Audit 验证（Round-1 natural 100q） |
| Premature 四象限监督 | 🔄 Stop Cal 代码 ✅；H_min_v2 100q audit ✅；bilateral 仍不足 |
| endorse-only / correct-only / main 对比 | 🔄 Round 2 训练跑通；**endorse-only 失败（0 样本）**；闭环未测 |
| loss-mass / compact / sample-norm 消融 | ✅ Round 2 Wave 3（7/8 variant）；main offline SKIP recall 50% |

**Setting（计划，endorse 消融）**

| 项 | 值 |
|----|-----|
| Model | `Qwen2.5-7B-Instruct` |
| Benchmark | BrowseComp+ |
| Retriever | BM25 |
| max_turns / max_tokens | 35 / 2048 |
| temperature | 1.0 |
| 对比 | endorse-only · reject/mask only · corrective CE · pairwise preference |

---

### E3 — Capability Weighting vs Privilege Illusion

**状态：** 📋 TODO

**Setting（计划）**

| 项 | 值 |
|----|-----|
| Model | `Qwen2.5-7B-Instruct` |
| Benchmark | BrowseComp+ |
| 对比 | uniform · \(U_m\) · \(U_m(1-\rho_m)\) · \(P_m U_m (1-\rho_m)\) |

---

### E4 — DAgger-style Mixing vs Shadow-first Recovery

**状态：** 📋 TODO（Recovery 暂不加）

**Setting（计划）**

| 项 | 值 |
|----|-----|
| Model | `Qwen2.5-7B-Instruct` |
| Benchmark | BrowseComp+ |
| Retriever | BM25 |
| max_turns / max_tokens | 35 / 2048 |
| temperature | 1.0 |
| 对比 | pure student OPD · DAgger mixture · student-prefix → teacher completion · SCOPE shadow-only · SCOPE shadow + recovery |

---

### E5 — Black-box Teacher Compatibility

**状态：** 📋 TODO

**Setting（计划）**

| 项 | 值 |
|----|-----|
| Model | `Qwen2.5-7B-Instruct` |
| Benchmark | BrowseComp+ |
| Teacher A | white-box local model（可返回 logits） |
| Teacher B | API / rule / retriever / verifier 混合 Harness |
| 对比 | logit-OPD vs action-level SCOPE |

---

### E6 — Module Retirement / Minimal Runtime Pareto

**状态：** 🔄 部分完成

#### ✅ 2026-07-28 上午 — Phase 0 基线冻结（830 题）

**Setting**

| 项 | 值 |
|----|-----|
| Model | `Qwen2.5-7B-Instruct` |
| Benchmark | BrowseComp+ 全量 830 题 |
| Retriever | BM25 |
| max_turns / max_tokens | 35 / 2048 |
| temperature | 1.0 |
| parallel | 2 |

| Runtime | Harness config | 说明 |
|---------|----------------|------|
| Bare | 无 Harness | 单轮 freeform 作答 |
| Full Harness v1 | `modules_full.yaml` | 旧 TokenBudget API agent |
| Full Harness v2 | `modules_full_v2.yaml` | Ultra ChatDecisionDriver + 全模块 |
| Minimal Runtime | `modules_minimal.yaml` | 仅 executor，V8D 全关 |

**结果**

| Runtime | recall | reward | mean_turns | mean_n_curated |
|---------|--------|--------|------------|----------------|
| Bare | — (acc 1.33%) | — | 1.0 | — |
| Full Harness v1 | 1.07% | 0.011 | 5.2 | 1.1 |
| **Full Harness v2** | **3.80%** | **0.181** | 33.7 | 26.5 |
| **Minimal Runtime** | **2.45%** | **0.121** | 32.4 | 13.2 |

产物：`artifacts/baselines/compare_phase0_full830.json`，`outputs/minimal_runtime_browsecomp_full830/`

| 配置 | recall | 状态 |
|------|--------|------|
| 1. Bare Model | acc 1.33% | ✅ 2026-07-28 上午 |
| 2. Minimal Executor | 2.45% | ✅ 2026-07-28 上午 |
| 3. Minimal + hard verifier / state store | — | 📋 TODO |
| 4. Partially retired Harness | — | 📋 TODO |
| 5. Full Harness v2 | 3.80% | ✅ 2026-07-28 上午 |
| 6. Trained + Minimal Runtime | smoke20：trained 劣于 base；**100q H_min_v2：n_curated +3.38** | 🔄 Round 2 确认问题 / 830 ⏸ |

---

### Fresh-corpus / Cross-Harness 泛化

**状态：** 📋 TODO

**Setting（计划）**

| 项 | 值 |
|----|-----|
| Model | `Qwen2.5-7B-Instruct` |
| Fresh Corpus | BM25 → dense；train corpus → new index；fixed source → new distribution |
| Cross-Harness | JSON 字段顺序 · reason code · evidence renderer · context serialization 扰动 |

---

## 全局待办（按优先级）

```text
[P0] Round 3 后续：完成 Wave4 plumbing + closed-loop 100q → 更新 DupCurateRate/FalseSkipRate
[P0] Round 3：调查 operation_ce 塌缩（全 KEEP）根因；对比 compact JSON 闭环行为
[P0] Stop Calibration：提升 CONTINUE→STOP 象限覆盖后再训 Premature
[P1] E0 distillability map ✅ 100q 冻结；830q 待 Round3 Dup positive signal 后人工批准
[P2] 多能力联合 SDI + capability weighting（E3）
[P2] Recovery branch（E4，按需）
[P3] E1/E5 基线对比 · Fresh-corpus / Cross-Harness 泛化

⏸ 暂缓：830q Minimal Runtime eval（RECOMMEND_830=false，Round2+Round3）
```

---

## 进度总览

```text
Stage 0  P_m probe                           [✓] 100q 冻结  2026-07-29 10:55
R2       Dup Behavioral Audit (0729)        [✓] Wave1–3 ✓ / Wave4 ✗  RECOMMEND_830=false
R3       Dup Bilateral Internalization      [~] A–C ✓ / Wave4+CL 🔄  RECOMMEND_830=false
Step 1   rollout @ H_min                    [~] H_min_v2 100q Base rollout ✓（Round 2/3）
Step 2   DecisionState                        [✓] 2026-07-28；R3 DupDecisionPoint ✓
Step 3   shadow guidance                      [✓] R3 DupBilateralShadow 双侧 ✓
Step 4   info-safe gate                       [✓] 2026-07-28
Step 5   ENDORSE/CORRECT                      [✓] R3 bilateral KEEP/SKIP+ENDORSE/CORRECT
Step 6   capability weighting                 [ ] TODO
Step 7   recovery                             [—] 暂缓
Step 8   L_SDI + L_stab                       [✓] R1 Dup  2026-07-28；R3 operation_ce ✓
Step 8↓  Round 3 train+offline                [✓] 8 variant；main SKIP recall=0%
Step 8↓  Round 3 closed-loop                  [~] Base shard0 进行中
Step 9   module lifecycle                     [ ] TODO

E0–E6    消融实验                              [~] E0 ✅ / E2 Round2+3 部分，其余 TODO
```
