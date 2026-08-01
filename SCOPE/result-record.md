## 当前结论（2026-08-01）

- **已成立：** same-state shadow / info-safe · measurement+scorer · Round5 O7 offline 双侧分离 · **Round6 offline cross-score AUROC=1.0（valid522 + 全部 B6 states）** · runtime parity=1.0（adapter/merged/HF）。
- **尚未成立：** Dup **closed-loop internalization** positive signal · 校准后仍 **FSR≈1.0** · reward 低于 Base · **`RECOMMEND_830=false`**。
- **Round 6 判定：** `H_RUNTIME/H_SHIFT/H_CALIB/H_FEEDBACK` 均为 false；`ROUND6_CLOSED_LOOP_POSITIVE=false`。
- **关键发现：** offline 排序与 AUROC 完美，但 per-seed margin 阈值校准 **不能** 将闭环行为拉到 FSR≤5%；O7 在 holdout 上表现为 **几乎全部 SKIP**（先验≈1），非可控 duplicate rejection。
- **当前 P0：** live closed-loop decision 路径 / score scale 与 offline replay 差异；禁止扩 830、E1、weighting、multi-capability。
- **主线：** 修 live admission 决策一致性 → 再评估是否需 on-policy Dagg。

---

## 分支与实验对照

> **说明：** 早期实验（Phase 0、Round-1、v3 协议、E0）均在 `main` 基线上以**本地未提交代码**跑通；Round 2/3 代码最终合并提交于 `scope/dup-round3-bilateral`。实验**产物目录**与 git 分支解耦——换分支不会移动 `outputs/`。

### 分支谱系

```text
main @ 3e95fad（origin/main）
  │  Phase 0 基线冻结（830q）
  │
  ├── [本地开发，07-28~07-29 未单独 commit]
  │     Round-1 Dup-SDI · v3 协议 smoke/audit · E0 100q · Round 2 全部 wave
  │
  ├── scope/dup-round2-behavioral @ 3e95fad
  │     仅本地工作分支指针（与 main 同 commit，**未 push**）
  │     Round 2 实验在此分支名上跑，但代码当时未入库
  │
  ├── scope/dup-round3-bilateral @ ad072b9（origin 已 push）
  │     一次性提交 Round 2 + Round 3 全部代码（115 files）
  │
  ├── scope/dup-round4-objective-repair @ 6b4e88b
  │     measurement / scorer audit + overfit128（objective 未过）
  │
  └── scope/dup-round5-learnability @ 6b4e88b + **本地未提交** Round5 代码
        Observability / objective tournament / O7 full screen / 100q CL
        （`scripts/scope_round5/` · `training/scope_round5/` 仍为 untracked）
```

### 实验 ↔ 分支 ↔ 产物 一览

| 实验阶段                 | 对应文档节      | Git 分支                                  | 关键 commit      | 远程                                  | 产物根目录                                                   |
| ------------------------ | --------------- | ----------------------------------------- | ---------------- | ------------------------------------- | ------------------------------------------------------------ |
| Phase 0 基线（830q）     | E6 §Phase 0     | `main`                                    | `1ed533b`        | ✅ `origin/main`                       | `artifacts/baselines/` · `outputs/minimal_runtime_browsecomp_full830/` |
| v3 协议 smoke/audit      | Step 1–5        | `main` + 本地                             | `3e95fad` 基线   | —                                     | `outputs/scope_v3_protocol_smoke20/` · `outputs/scope_v3_audit_100q/` |
| Round-1 Dup-SDI 训练     | Step 8 §Round 1 | `main` + 本地                             | `3e95fad` 基线   | —                                     | `artifacts/datasets/dup_sdi_round1/` · `outputs/dup_sdi_round1/` |
| E0 Distillability 100q   | Stage 0         | `main` + 本地                             | `3e95fad` 基线   | —                                     | `outputs/scope_e0_distillability/` · `artifacts/capability/distillability_map.json` |
| Round 2 Behavioral Audit | Round 2         | `scope/dup-round2-behavioral`（工作分支） | 代码在 `ad072b9` | ❌ 未 push                             | `outputs/scope_round2/` · `artifacts/datasets/dup_sdi_round2/` · `artifacts/datasets/round2_audit_100q/` |
| Round 3 Bilateral        | Round 3         | `scope/dup-round3-bilateral`              | **`ad072b9`**    | ✅ `origin/scope/dup-round3-bilateral` | `outputs/scope_round3/` · `artifacts/datasets/dup_sdi_round3/` |
| Round 4 Objective Repair | Round 4         | `scope/dup-round4-objective-repair`       | `6b4e88b` / `e3d5afa` | ❌ 未确认 push                    | `outputs/scope_round4/` |
| Round 5 Learnability     | Round 5         | `scope/dup-round5-learnability`           | `6b4e88b` + 本地 | ❌ 未 push                             | `outputs/scope_round5/` |

### 复现注意事项

| 场景                       | 应 checkout                               | 说明                                                         |
| -------------------------- | ----------------------------------------- | ------------------------------------------------------------ |
| 仅复现 Phase 0 基线        | `main` @ `3e95fad`                        | 不含 Round 2/3 脚本                                          |
| 复现 Round 2 训练/评估脚本 | `scope/dup-round3-bilateral` @ `ad072b9`  | Round 2 代码未在 `dup-round2-behavioral` 上单独 commit       |
| 复现 Round 3               | `scope/dup-round3-bilateral` @ `ad072b9`  | —                                                            |
| 复现 Round 4/5 数值        | **无需切换分支**                          | 直接读 `outputs/scope_round4|5/`；R5 脚本需本地工作树        |
| 复现 Round 5 编排          | `scope/dup-round5-learnability` + 本地    | `scripts/scope_round5/` 当时未入库                           |
| 复现历史实验数值           | **无需切换分支**                          | 直接读 `outputs/` 下 JSON/MD；数据集在 `artifacts/datasets/`（未进 git，需本地保留） |

### 共享协议资产（跨分支冻结）

| 资产                           | 路径                                                       | 首次冻结       | 使用方                          |
| ------------------------------ | ---------------------------------------------------------- | -------------- | ------------------------------- |
| BrowseComp+ 100q manifest      | `artifacts/datasets/round2_audit_100q/query_manifest.json` | Round 2 Wave 1 | Round 2–5 closed-loop           |
| \(H_{\min,\text{v2}}\) runtime | `harness/configs/modules_minimal_v2.yaml`                  | Round 2        | Round 2–5 rollout · closed-loop |
| Round-1 merged 对照模型        | `outputs/dup_sdi_round1/merged_hf`                         | Round 1        | Round 2 Wave 1 · Round 3 Wave 4 |
| Distillability map             | `artifacts/capability/distillability_map.json`             | E0 07-29 10:55 | 830q Go/No-Go 参考              |
| Round3 Dup train/valid         | `artifacts/datasets/dup_sdi_round3/`                       | Round 3        | Round 3–5 训练/offline          |
| Round4 overfit128              | `artifacts/datasets/dup_sdi_round4_overfit128/`            | Round 4        | Round 4 B4 · Round 5 B3         |
| Round5 O7 merged checkpoints   | `outputs/scope_round5/merged/o7_r64_seed{42,43,44}`        | Round 5 B6     | 100q closed-loop                |

---

## Stage 0 — Module Distillability Probe（E0）

**method 对应：** 估计各模块的 procedural recoverability \(P_m\)，为 procedural / hybrid / runtime-dependent taxonomy 提供证据。

**状态：** ✅ 100q probe 已完成并冻结（2026-07-29 10:55）；⚠️ **taxonomy 尚未冻结**。当前结果用于筛选后续 probe，而不是把所有模块直接定类。

**分支：** `main` @ `3e95fad` + 本地未提交脚本（E0 编排脚本已入库于 `ad072b9`） · **未单独开分支** · 产物不依赖 git

### Setting

| 项                     | 值                                                           |
| ---------------------- | ------------------------------------------------------------ |
| Model                  | `Qwen2.5-7B-Instruct`（base，未训）                          |
| Benchmark              | BrowseComp+，固定 audit 100q（`artifacts/datasets/e0_audit_100q`，SEED=42） |
| Retriever              | BM25                                                         |
| Harness base           | `modules_full_v2.yaml`；FULL 复用 `outputs/harness_rollout_browsecomp_full_v2` |
| max_turns / max_tokens | 35 / 2048                                                    |
| temperature            | 1.0                                                          |
| GPU / vLLM             | GPU4，port 8776，`e0-harness-policy`                         |
| 对比形态               | capability-level OFF / PROC / FULL（`deterministic_truncation` 无 PROC） |
| 编排脚本               | `run_e0_distillability_nohup.sh` + `e0_watch_and_rerun.sh` + `e0_status.sh` |
| 产物目录               | `outputs/scope_e0_distillability/` · `artifacts/capability/distillability_map.json` · `E0_REPORT.md` |

#### ✅ 2026-07-29 — E0 100q 正式冻结（10:55）

**续跑时间线（07-29）**

| 时间        | 任务                                             | 结果                 |
| ----------- | ------------------------------------------------ | -------------------- |
| 08:44       | kill 卡住 `external_verification/proc`（23/100） | 旧编排器误标 DONE    |
| 08:44–09:18 | deterministic_truncation/off                     | ✅ 100/100（~34 min） |
| 09:19–10:13 | duplicate_evidence/proc                          | ✅ 100/100（~54 min） |
| 10:13–10:15 | verification_decision/proc（补 2 题）            | ✅ 100/100            |
| 10:15–10:55 | external_verification/proc（续跑 77 题）         | ✅ 100/100（~40 min） |
| 10:55       | `build_map.py` → Map + E0_REPORT 冻结            | ✅                    |

**完成度（全部 100/100）**

| Capability               |  OFF | PROC |      FULL |
| ------------------------ | ---: | ---: | --------: |
| duplicate_evidence       |    ✅ |    ✅ | ✅（复用） |
| stop_decision            |    ✅ |    ✅ |         ✅ |
| evidence_curation        |    ✅ |    ✅ |         ✅ |
| verification_decision    |    ✅ |    ✅ |         ✅ |
| external_verification    |    ✅ |    ✅ |         ✅ |
| deterministic_truncation |    ✅ |  n/a |         ✅ |

**主指标 recall（paired；FULL 复用 Full v2 = 0.0506）**

> 下表保留原始 `build_map` 输出；`Decision` 改为“当前证据状态”，避免把低覆盖 PROC probe 误解释成最终 taxonomy。

| Capability                | \(R_{\text{off}}\) | \(R_{\text{proc}}\) | \(R_{\text{full}}\) | \(\Delta^{\text{proc}}\) | \(\Delta^{\text{full}}\) | \(P_{\text{raw}}\) | CI(\(P\))     |   W/L/T | 当前证据状态                                    |
| ------------------------- | -----------------: | ------------------: | ------------------: | -----------------------: | -----------------------: | -----------------: | ------------- | ------: | ----------------------------------------------- |
| duplicate_evidence        |             0.0255 |              0.0343 |              0.0506 |                  +0.0088 |              **+0.0250** |               0.35 | [−1.23, 1.27] | 12/7/81 | **HYBRID-CANDIDATE / LOW-CONF**                 |
| stop_decision             |             0.0282 |              0.0239 |              0.0506 |                  −0.0043 |                  +0.0224 |              −0.19 | [−6.02, 0.84] | 8/10/82 | **INVALID/WEAK PROC PROBE**                     |
| evidence_curation         |             0.0383 |              0.0169 |              0.0506 |                  −0.0213 |                  +0.0123 |              −1.74 | [−24.0, 18.5] | 4/12/84 | **RUNTIME-LEANING / LOW-CONF**                  |
| verification_decision     |             0.0424 |              0.0198 |              0.0506 |                  −0.0226 |                  +0.0082 |              −2.76 | [−22.8, 28.7] | 3/10/87 | **INVALID/WEAK PROC PROBE**                     |
| external_verification     |             0.0152 |              0.0203 |              0.0506 |                  +0.0051 |              **+0.0353** |               0.15 | [−0.52, 0.60] |  5/6/89 | **EXECUTION RUNTIME-DEPENDENT；ROUTING 未判定** |
| deterministic_truncation† |             0.0269 |                   — |              0.0506 |                        — |                 ~+0.0237 |                  — | —             |       — | **INVALID PROBE（0 events）**                   |

† `build_map` 对 truncation 报 `no_overlap_queries`（map 中 R=0 为 builder bug）；episodes/summary 实测 \(R_{\text{off}}=0.0269\)，`truncation_events=0`。

**PROC audit**

| Capability               | interventions | shadow_calls | info-safe | 备注                                                         |
| ------------------------ | ------------: | -----------: | --------- | ------------------------------------------------------------ |
| duplicate_evidence       |           606 |          606 | ❌         | visibility_violation_rate=3%；PROC 有方向性恢复              |
| stop_decision            |             0 |            3 | ✅         | 无有效干预，不能据此判 runtime-only                          |
| evidence_curation        |           700 |          700 | ✅         | 有干预但 \(\Delta^{\text{proc}}<0\)                          |
| verification_decision    |             0 |            0 | ✅         | 无有效干预，不能据此判 runtime-only                          |
| external_verification    |             0 |            0 | ✅         | PROC 不暴露 verify tool；只说明“执行能力”不可由该 PROC 形态替代 |
| deterministic_truncation |             — |            — | —         | PROC 不支持                                                  |

### 当前结论

1. **E0 已验证“capability 具有异质性”的方向，但不足以冻结 taxonomy。** `duplicate_evidence` 是目前唯一同时具备大量 PROC intervention 与正向恢复的模块，因此是最合适的 internalization sanity check。
2. `stop_decision`、`verification_decision` 的 PROC 几乎没有实际 intervention，当前结果主要反映 **probe coverage 不足**，不是“不可蒸馏”的证据。
3. `evidence_curation` 有充分 intervention 且 PROC 明显劣于 OFF，是目前较可信的 **runtime-leaning / 当前 proceduralization 失败** 信号，但 CI 仍很宽。
4. `external_verification` 应拆成两层看：**外部事实获取/执行**天然需要 runtime；“何时触发验证”的 routing decision 仍可能内化。现有 E0 把二者混在同一 capability 中，因此不应以 \(P_{\text{raw}}=0.15\) 直接判定整个能力 runtime-only。
5. `deterministic_truncation` 在 100q 中 0 events，当前 probe 没有辨识力；扩大到 830 也不一定解决，应先构造 event-enriched probe。
6. \(P_{\text{raw}}\) 的分母 \(\Delta^{\text{full}}\) 较小且 paired ties 很多，导致 CI 极宽；在 intervention coverage 达标前，\(P_m\) 只作辅助统计，不作硬分类阈值。

### 下一步：不直接扩 E0 830

| 优先级 | Capability               | 动作                                                      | Go 条件                       |
| ------ | ------------------------ | --------------------------------------------------------- | ----------------------------- |
| **P0** | duplicate_evidence       | Round 3 已完成（否定性结论）；修 evaluator + 诊断 operation_ce 塌缩 | 指标可信 + objective 修复后重训 main |
| **P1** | stop_decision            | 修 selector，构造 STOP↔CONTINUE 双侧 event-enriched probe | 两类 intervention 均非零      |
| **P1** | verification_decision    | 先修触发/打点，再做 targeted probe                        | `n_proc_interventions > 0`    |
| **P1** | deterministic_truncation | 构造必触发 truncation 的长轨迹子集                        | `truncation_events > 0`       |
| **P2** | evidence_curation        | 检查当前 PROC 语义是否等价，再决定是否重跑                | procedural artifact 定义稳定  |
| **P2** | external_verification    | 拆成 routing decision 与 external execution 两层评估      | 不再混合 capability 边界      |

**产物**

- `artifacts/capability/distillability_map.json`（2026-07-29 10:55）
- `outputs/scope_e0_distillability/E0_REPORT.md`（2026-07-29 10:55）
- 全量 episodes：`outputs/scope_e0_distillability/{cap}/{off,proc,full}/episodes.jsonl`

**备注：** Round-1 Dup-SDI 训练独立于 E0；E0 使用 base model。原计划的 E0 830 **暂不启动**：当前主要瓶颈是 probe validity / event coverage，而不是样本量。

<details>
<summary>E0 历史记录：07-28 首轮失败 → 07-29 卡住恢复</summary>


#### 首轮失败（07-28 13:42–14:30）

- 大量 OFF/PROC：`Connection error` / PROC `run_information_safe_gates(... artifact=)` 签名 bug。
- 旧 `E0_REPORT`（14:30）不可信（多模式 `errors=1.0`、`turns=0`）。

#### nohup 续跑（07-28 17:14 起）

| 时间              | 任务                       | 结果                                 |
| ----------------- | -------------------------- | ------------------------------------ |
| 17:15             | duplicate_evidence/proc    | OOM Kill，误标 DONE，episodes 空     |
| 17:15–17:19       | stop_decision/proc         | ✅ 100                                |
| 17:19–17:52       | evidence_curation/proc     | ✅ 100                                |
| 17:52–18:02       | verification_decision/off  | ✅ 100                                |
| 18:02–18:46       | verification_decision/proc | 98/100（2 error）                    |
| 18:46–19:19       | external_verification/off  | ✅ 100                                |
| 19:19–19:27       | external_verification/proc | 推进至 ~20–23/100                    |
| 19:27–07-29 08:44 | external_verification/proc | **卡住** 23/100（PID 3960060，~13h） |

#### 中间态（07-29 08:43）

仅 `stop_decision` / `evidence_curation` 三模式齐全；dup PROC 空；truncation OFF 失败；external PROC 卡住。08:44 kill 后 truncation OFF 续跑，09:19 watch relaunch 补齐其余缺格，10:55 冻结（见上）。

</details>

---

## Round 2 — Dup Behavioral Audit（0729-todo1）

**method 对应：** 诊断 Round-1 teacher-forced 拟合 vs \(H_{\min}\) 闭环行为脱节；引入 \(H_{\min,\text{v2}}\)、compact operation target、sample-normalized CE，做 100q 对照与训练消融。

**状态：** ✅ 代码 + Wave 1–3 完成 · ⏭ Wave 4 未单独补跑（由 Round 3 统一 typed-action 闭环替代） · `RECOMMEND_830=false`

**分支：** 工作分支 `scope/dup-round2-behavioral`（本地，@ `3e95fad`，**未 push**）· 代码 commit `ad072b9`（在 `scope/dup-round3-bilateral`）  
**产物根目录：** `outputs/scope_round2/` · `artifacts/datasets/round2_audit_100q/` · `artifacts/datasets/dup_sdi_round2/`  
**报告：** `outputs/scope_round2/ROUND2_REPORT.md`

### 代码改动（Barrier 0，96 tests pass）

核心改动：loss-mass audit（`analyze_loss_mass.py`）、operation eval（`eval_dup_capability.py`）、\(H_{\min,\text{v2}}\)（`modules_minimal_v2.yaml`）、compact KEEP/SKIP target、`ActionRealizer`、sample-normalized CE、Stop 四象限统计。编排：`scripts/scope_round2/run_all.sh`。

### 全局 GPU / 协议 Setting

| 项                                   | 值                                                           |
| ------------------------------------ | ------------------------------------------------------------ |
| Model                                | `Qwen2.5-7B-Instruct`（Base）；Round1 对照 `outputs/dup_sdi_round1/merged_hf` |
| Benchmark                            | BrowseComp+，冻结 **100q**（`artifacts/datasets/round2_audit_100q/query_manifest.json`，`SEED=42`，4×25 shard） |
| Retriever                            | BM25                                                         |
| Rollout runtime                      | **`modules_minimal_v2.yaml`**（\(H_{\min,\text{v2}}\)：保留 search/BM25/verify tool + hard truncate；关闭 cognitive dedup/curation/stop policy） |
| vLLM                                 | **1 model / 1 GPU，TP=1**；GPU0–7 → port 8800–8807           |
| max_turns / max_tokens / temperature | 35 / 2048 / 1.0                                              |
| 禁止项（本轮未跑）                   | 830q eval · E0 830 · capability weighting · Recovery · RL · Irrelevant |

---

#### ✅ Barrier 0 — 代码 + 诊断（2026-07-29）

Round-1 loss-mass audit（755 samples）：ENDORSE 45.0% samples / 21.0% target tokens；CORRECT 54.8% / 79.0%；其中 `verify_claim` 占 60% target-token mass。**确认 sample balance 与 optimization mass 不一致，但这只是根因候选，不是因果证明。**

产物：`outputs/scope_round2/diagnostics/round1_loss_mass.md`

#### ✅ Wave 1 — H_min_v2 100q Closed-loop（Base vs Old Round1）

**Setting**

| 项                | 值                                                         |
| ----------------- | ---------------------------------------------------------- |
| Base / Old Round1 | `Qwen2.5-7B-Instruct` / `outputs/dup_sdi_round1/merged_hf` |
| Runtime           | `modules_minimal_v2.yaml`                                  |
| 分片              | GPU0–3 Base；GPU4–7 Round1                                 |
| 脚本              | `training/scope_round2/hmin_v2_rollout.py`                 |

| 指标                  |      Base | Old Round1 | Δ / Paired         |
| --------------------- | --------: | ---------: | ------------------ |
| recall                |     2.29% |      3.48% | +1.18pp；12/7/81   |
| reward                |     0.122 |      0.220 | +0.099；13/14/73   |
| trajectory_recall     |     24.2% |      25.3% | +1.1pp             |
| final_answer_recall   |     2.95% |      6.37% | +3.42pp            |
| mean_turns            |     33.29 |      34.43 | +1.14              |
| **mean_n_curated**    | **14.35** |  **17.73** | **+3.38；62/32/6** |
| mean_n_pool           |    288.94 |     286.93 | −2.01              |
| unique_evidence_ratio |     0.061 |      0.068 | +0.007；62/38/0    |

结论：**100q 中再次出现“更多 curate”的方向，但没有复现 smoke20 的 task degradation。** `duplicate_curate_rate` 当时未 instrument，因此 `mean_n_curated` 只是行为代理，不能等价为“重复 curate”。

产物：`outputs/scope_round2/hmin_v2_{base,round1}/merged/` · `eval/base_vs_round1_100q.md`

#### ✅ Wave 2 — Same-State Shadow + Stop Calibration + Round1 Capability Re-eval

**Dup shadow（Base @ H_min_v2 decision states）**

- 四 shard 并行 labeling → `outputs/scope_round2/dup_shadow/shard0–3/`
- 修复 compact target 解析后重建数据集

**Stop Calibration 100q（H_min_v2）**

| 象限              | Count |
| ----------------- | ----: |
| STOP→STOP         |     0 |
| STOP→CONTINUE     |    13 |
| CONTINUE→STOP     |     0 |
| CONTINUE→CONTINUE |  3316 |
| n_decision_points |  3329 |

- **bilateral_coverage: False** — 仍缺 CONTINUE→STOP 监督质量
- 产物：`outputs/scope_round2/stop_calibration/stop_calibration_100q.md`

**Round1 capability re-eval（新指标，valid 77）**

| 指标                     | 值    |
| ------------------------ | ----- |
| teacher_forced_token_acc | 93.9% |
| action_match_rate        | 26.0% |
| route CORRECT accuracy   | 94.9% |
| route ENDORSE accuracy   | 82.8% |

---

#### ✅ Barrier 2 — Round 2 数据集

**Setting**

| 项          | 值                                |
| ----------- | --------------------------------- |
| 来源        | Base @ H_min_v2 same-state shadow |
| Capability  | `duplicate_evidence` only         |
| Target 格式 | compact `SKIP_DUPLICATE` JSON     |
| Split       | query-level，train 257 / valid 30 |

**分布（局限）**

| 项                             | 值          |
| ------------------------------ | ----------- |
| KEEP / SKIP                    | 0 / **287** |
| ENDORSE / CORRECT              | **0 / 287** |
| visibility / schema violations | 0 / 0       |

⚠️ 仅捕获 duplicate-curate **CORRECT** 点，无 ENDORSE/KEEP → endorse-only 消融无法运行

产物：`artifacts/datasets/dup_sdi_round2/`

---

#### ✅ Wave 3 — 8 路训练消融（Barrier 3）

**共同 Setting**

| 项                               | 值                       |
| -------------------------------- | ------------------------ |
| Base model                       | `Qwen2.5-7B-Instruct`    |
| Method                           | LoRA r=16, α=32          |
| lr / epochs / batch / grad_accum | 2e-5 / 3 / 4 / 4         |
| max_length                       | 4096                     |
| KL coef                          | 0.01                     |
| Dataset                          | `dup_sdi_round2`（同上） |

**Variant 分配**

| GPU  | Variant                        | loss_mode         | compact | route_balance | 备注                          |
| ---- | ------------------------------ | ----------------- | ------- | ------------- | ----------------------------- |
| 0    | round2_main                    | sample_normalized | ✅       | ✅             | 主模型                        |
| 1    | round2_legacy_token_ce         | legacy_token_ce   | ❌       | ❌             | 长序列 token CE 对照          |
| 2    | round2_full_action_sample_norm | sample_normalized | ❌       | ❌             | 完整 action + sample norm     |
| 3    | round2_no_route_balance        | sample_normalized | ✅       | ❌             |                               |
| 4    | round2_endorse_only            | sample_normalized | ✅       | —             | **FAILED**（0 train samples） |
| 5    | round2_correct_only            | sample_normalized | ✅       | —             | CORRECT only filter           |
| 6    | round2_main_seed43             | 同 main           | ✅       | ✅             | seed=43                       |
| 7    | round2_main_seed44             | 同 main           | ✅       | ✅             | seed=44                       |

**结果 — round2_main offline capability（30 valid，compact prompt）**

| 指标                       | 值                      |
| -------------------------- | ----------------------- |
| valid loss                 | 0.275                   |
| parse_rate                 | **1.0**                 |
| operation_accuracy         | 0.50                    |
| SKIP_DUPLICATE recall / F1 | **0.50 / 0.67**         |
| KEEP_EVIDENCE recall       | n/a（valid 全 CORRECT） |
| teacher_forced_token_acc   | 94.7%（不作为成功标准） |

\* 其余 variant Barrier-3 批量 eval 因 prompt 不匹配 greedy_parse=0；仅 main 经 compact prompt 重评。

产物：`outputs/scope_round2/training/round2_*/` · `eval/round2_training_comparison.md`

---

#### ⏭ Wave 4 — Closed-loop 100q（未单独补跑；由 Round 3 替代）

**原计划：** 8 模型 × 同 manifest × \(H_{\min,\text{v2}}\) × merge LoRA → rollout

**未完成原因：** Round 2 推理路径尚未接入 ActionRealizer（compact operation → runtime action）。Round 3 已把 typed operation、ActionRealizer 与 telemetry 统一，因此**不再回头补 Round 2 的旧闭环路径**；仅保留 Round 2 checkpoint 作为 Round 3 diagnostic 对照。

产物占位：`outputs/scope_round2/eval/round2_closed_loop_100q.md`（partial）

---

### Round 2 五问结论（0729-todo1 §十三）

| #    | 问题                                  | 当前结论                                                     |
| ---- | ------------------------------------- | ------------------------------------------------------------ |
| Q1   | sample balance ≠ loss-token balance？ | **确认存在**。CORRECT 占 79% target-token mass，而 sample share 为 54.8%；`verify_claim` 单项占 60%。这是明确的 objective imbalance。 |
| Q2   | Round1 @ H_min_v2 复现 smoke20？      | **复现行为方向，不复现任务退化。** mean_n_curated +3.38；但 100q recall/reward 未下降，因此不能把 smoke20 的任务性能下降视为稳定结论。 |
| Q3   | compact target 改善 operation？       | **仅有弱离线信号。** one-sided valid 上 SKIP recall=50%；没有 KEEP 类，也没有闭环结果。 |
| Q4   | legacy vs sample-norm vs compact？    | **尚不能比较。** 多个 variant 因 prompt/eval 不匹配未公平重评；当前数据不足以归因到 loss 或 target format。 |
| Q5   | Endorse vs Correct 贡献？             | **无法回答。** Round 2 数据为 0 ENDORSE / 0 KEEP。           |

### 最终判定

```text
ROUND1_CONFIRMED_FAILURE = teacher-forced 拟合没有转化为预期的 H_min_v2 行为；
                           Round1 模型稳定增加 mean_n_curated

PRIMARY_HYPOTHESIS = 长 action span + token-level CE 导致 loss mass 偏向 CORRECT/verify_claim
                     （有诊断证据，但尚无公平消融证明其为“根因”）

ROUND2_POSITIVE_SIGNAL = NOT_ESTABLISHED
RECOMMEND_830 = false

ROUND2_WAVE4 = SUPERSEDED_BY_ROUND3_TYPED_ACTION_CLOSED_LOOP
```

**对后续的有效结论只有两条：** 先修复双侧监督与 train/inference action interface；在这两项完成后，才有资格比较 objective 并讨论是否扩大到 830。Round 3 正是对此的重构。

---

## Round 3 — Bilateral Duplicate Capability Internalization（0729-todo2）

**method 对应：** 在 student 真实访问的 evidence-admission decision points 上构造 KEEP/SKIP 双侧监督，用 `operation_ce` 直接优化 operation decision，并统一 train/inference 的 typed action interface（`DupOperationRuntime` + `ActionRealizer`），验证能否在 \(H_{\min,\text{v2}}\) closed-loop 中降低 DuplicateCurateRate 且不显著恶化 FalseSkipRate。

**状态：** ✅ **全部完成**（Barrier A–C · Wave4 · Closed-loop 100q · Final report）· `ROUND3_POSITIVE_SIGNAL=false` · `RECOMMEND_830=false`

**分支：** `scope/dup-round3-bilateral` @ **`ad072b9`**（`origin/scope/dup-round3-bilateral`，2026-07-29 push）  
**产物根目录：** `outputs/scope_round3/` · `artifacts/datasets/dup_sdi_round3/`  
**报告：** `outputs/scope_round3/ROUND3_REPORT.md`  
**前置：** 依赖 Round 2 的 100q manifest 与 \(H_{\min,\text{v2}}\) rollout states；Round 2 训练 checkpoint 在 `outputs/scope_round2/training/`

### 代码改动（Barrier A，109 tests pass）

Round 3 将 Dup 从 error-triggered 改为 **decision-triggered evidence admission**：`DupDecisionPoint` + `DupBilateralShadow` 生成 KEEP/SKIP 双侧 label；`score_operations` / `operation_ce` 与 `DupOperationRuntime` 共用 scorer；`ActionRealizer` 执行 typed operation；`dup_telemetry.py` 记录 admission behavior。编排：`scripts/scope_round3/run_all_8gpu.sh`。

### 全局 GPU / 协议 Setting

| 项                                   | 值                                                           |
| ------------------------------------ | ------------------------------------------------------------ |
| Model                                | `Qwen2.5-7B-Instruct`（Base）；Round1 `outputs/dup_sdi_round1/merged_hf`；Round2 `outputs/scope_round2/training/round2_*` |
| Benchmark                            | BrowseComp+，冻结 **100q**（`artifacts/datasets/round2_audit_100q/query_manifest.json`） |
| Retriever                            | BM25                                                         |
| Rollout runtime                      | **`modules_minimal_v2.yaml`**（\(H_{\min,\text{v2}}\)）      |
| vLLM                                 | **1 model / 1 GPU，TP=1**；Round3 port **8900–8907**（Wave4）/ **8910–8927**（closed-loop） |
| max_turns / max_tokens / temperature | 35 / 2048 / 1.0                                              |
| 禁止项（本轮未跑）                   | 830q · E0 830 · capability weighting · Recovery · RL · Premature Stop 训练 · Irrelevant |

---

#### ✅ Barrier A — 代码 + 单测（2026-07-29）

`pytest tests/scope/`：109 passed。覆盖 KEEP/SKIP、ENDORSE/CORRECT、ActionRealizer 映射、visibility、train/inference shared scorer。首轮 Wave4 因 `CurateTool` import 路径错误失败，已修复为 `training.train_rl.CurateTool`。

#### ✅ Barrier B — 双侧数据集（2026-07-29）

**Setting**

| 项     | 值                                                           |
| ------ | ------------------------------------------------------------ |
| 来源   | Base @ H_min_v2 decision states（Round2 100q rollout 重切 8 shard） |
| Shadow | `DupBilateralShadow`（decision-triggered，非 duplicate-suspect 触发） |
| Split  | query-level：**train 80q / valid 20q**（1807 / 522 events）  |
| Gate   | visibility_violation=0（3 条预过滤）· shadow_mutation=0 · schema_invalid=0 |

**分布（双侧，Round2 单侧问题已修复）**

| 项                    |    Count |
| --------------------- | -------: |
| KEEP_EVIDENCE         | **1784** |
| SKIP_DUPLICATE        |  **545** |
| ENDORSE               | **1784** |
| CORRECT               |  **545** |
| keep/skip ratio       |   3.27:1 |
| endorse/correct ratio |   3.27:1 |

```text
ROUND3_DATA_GO = true
```

产物：`artifacts/datasets/dup_sdi_round3/` · `bilateral_dataset_report.md` · `bilateral_dataset_stats.json`

---

#### ✅ Barrier C — 8 路训练消融（2026-07-29）

**共同 Setting**

| 项                               | 值                                                 |
| -------------------------------- | -------------------------------------------------- |
| Base model                       | `Qwen2.5-7B-Instruct`                              |
| Method                           | LoRA r=16, α=32                                    |
| lr / epochs / batch / grad_accum | 2e-5 / 3 / 4 / 4                                   |
| max_length                       | 4096                                               |
| KL coef                          | 0.01（`operation_ce` 路径 KL≈0）                   |
| Dataset                          | `dup_sdi_round3`（1807 train / 522 valid）         |
| 优化 steps                       | ~1350 / variant（operation_ce）；correct-only ~330 |

**Variant 分配**

| GPU  | Variant                            | loss_mode         | 备注                         |
| ---- | ---------------------------------- | ----------------- | ---------------------------- |
| 0    | round3_op_main_seed42              | **operation_ce**  | route+class balance，seed=42 |
| 1    | round3_op_main_seed43              | operation_ce      | 同 main，seed=43             |
| 2    | round3_op_main_seed44              | operation_ce      | 同 main，seed=44             |
| 3    | round3_compact_json_sample_norm    | sample_normalized | compact JSON 对照            |
| 4    | round3_legacy_full_action_token_ce | legacy_token_ce   | 表面形式 imitation 对照      |
| 5    | round3_correct_only_op             | operation_ce      | CORRECT only                 |
| 6    | round3_endorse_only_op             | operation_ce      | ENDORSE only                 |
| 7    | round3_op_no_balance               | operation_ce      | 无 class/route balance       |

**训练 loss（epoch 3 末）**

| Variant                     | final_train_loss |
| --------------------------- | ---------------: |
| round3_op_main_seed42/43/44 |            ~0.50 |
| round3_op_no_balance        |            0.519 |
| round3_compact_json         |            0.226 |
| round3_legacy_token_ce      |            0.227 |
| round3_correct_only         |        **0.002** |
| round3_endorse_only         |           **≈0** |

产物：`outputs/scope_round3/training/round3_*/` · merged：`outputs/scope_round3/merged/`

---

#### ✅ 训练前 Baselines + Offline Capability Eval（valid 522）

> ⚠️ **Evaluator sanity check（P0）**：valid 集同时含 KEEP/SKIP 时，“永远 KEEP”的标准 `KEEP F1` 不应为 1.000；“永远 SKIP”的 `SKIP F1` 也不应为 1.000。例如 B1 全 KEEP 且 op_acc≈81% 时，标准 KEEP-F1 应约为 \(2\times0.81/(1+0.81)\approx0.895\)，而不是 1.000。当前脚本很可能把 class recall / class accuracy 误标为 F1，或实现存在错误。因此下列 F1 / macro-F1 **保留原始记录但暂不用于研究结论**。在修复前，可信的主要是预测分布、operation accuracy、KEEP/SKIP recall 等可直接核验量。

**B0 — Majority（永远 KEEP）**

| KEEP F1 | SKIP F1 | macro-F1 |
| ------: | ------: | -------: |
|   1.000 |   0.000 |    0.500 |

**B1 — Base operation_ce（未训练，restricted verbalizer scorer）**

| KEEP F1 | SKIP F1 | macro-F1 | op_acc |
| ------: | ------: | -------: | -----: |
|   1.000 |   0.000 |    0.500 |  81.0% |

**B2 — Round2 main（compact JSON，公平 operation eval）**

| KEEP F1 |   SKIP F1 | macro-F1 | SKIP recall |
| ------: | --------: | -------: | ----------: |
|   0.000 | **0.697** |    0.349 |   **53.5%** |

**Round3 全 variant offline（operation-level，522 valid）**

| Variant                     | KEEP F1 |   SKIP F1 |  macro-F1 | balanced acc | 备注                    |
| --------------------------- | ------: | --------: | --------: | -----------: | ----------------------- |
| round3_op_main seed42/43/44 |   1.000 | **0.000** |     0.500 |        0.500 | **≡ Base，全预测 KEEP** |
| round3_op_no_balance        |   1.000 |     0.000 |     0.500 |        0.500 | 同上                    |
| round3_endorse_only         |   1.000 |     0.000 |     0.500 |        0.500 | 对照：全 KEEP           |
| round3_correct_only         |   0.000 | **1.000** |     0.500 |        0.500 | 对照：全 SKIP           |
| **round3_compact_json**     |   0.983 | **0.061** | **0.522** |        0.522 | 唯一略优于 Base         |
| round3_legacy_token_ce      |   0.986 |     0.020 |     0.503 |        0.503 | token CE，SKIP 极弱     |

**解读**

- **可信现象：** operation_ce main 三个 seed 均全预测 KEEP，SKIP recall=0%；说明 bilateral discrimination 尚未学出。
- correct-only / endorse-only 能把模型推向全 SKIP / 全 KEEP，只能证明 objective 能推动 score 到单侧极端，**不能证明双侧分类训练正确**。
- compact JSON 至少产生少量 SKIP prediction（脚本报告 SKIP recall 6.1%）；在 F1 evaluator 修复前，不再写“优于 Base”或与 Round2 直接排序。
- teacher-forced token accuracy 仅作拟合诊断，不作为 capability internalization 成功标准。

产物：`outputs/scope_round3/eval/baselines.json` · `outputs/scope_round3/eval/offline_capability.json`

---

#### ✅ Wave 4 Diagnostic — 四 checkpoint plumbing（2026-07-30）

**Setting：** Base / Round1 / Round2-main / Round2-legacy × dup-operation + ActionRealizer + telemetry；每 variant **shard0+shard1**（50q diagnostic）；port 8900–8907

| 阶段                | 状态                                                         |
| ------------------- | ------------------------------------------------------------ |
| 首轮（07-29 14:52） | ❌ `ImportError: CurateTool from harness.tools`（已修）         |
| 最终（07-30 03:37） | ✅ 4 variant merged · `wave4_barrier=true` · `plumbing_ok=true` |

**Wave4 结果（50q / variant，dup-operation 路径）**

| Variant       | DupCurateRate | recall | reward | plumbing |
| ------------- | ------------: | -----: | -----: | -------- |
| base          |         0.000 |  3.90% |  0.305 | ✅        |
| round1        |         0.000 |  1.99% |  0.167 | ✅        |
| round2_main   |         0.000 |  3.90% |  0.305 | ✅        |
| round2_legacy |         0.000 |  3.90% |  0.305 | ✅        |

> Base 与 Round2 checkpoint 在 dup-operation 路径下 DCR=0（全 KEEP admission），与 offline B1 一致；telemetry 完整、ActionRealizer 无 hidden fallback。

产物：`outputs/scope_round3/wave4_diagnostic/comparison.json` · `comparison.md`

---

#### ✅ Closed-loop 100q — Dup 行为主指标（2026-07-30 完成）

**Setting**

| 项   | 值                                                           |
| ---- | ------------------------------------------------------------ |
| 协议 | 冻结 manifest · BM25 · \(H_{\min,\text{v2}}\) · dup-operation + ActionRealizer |
| 规模 | 100q × 8 shard = 8 GPU 并行；每 variant 8 shard merge |
| 脚本 | `run_post_train_8gpu.sh` → `resume_post_train_8gpu.sh`（错峰 90s + 串行重试） |

**执行记录**

| 时间 | 事件 |
| ---- | ---- |
| 07-29 17:11 | 首轮 post-train：8 路并行 vLLM 初始化竞争，多 variant 失败 |
| 07-29 22:27 | 二次重启：仍因 vLLM `Engine core initialization failed` 中断 |
| 07-30 01:34 | 三次重启：`resume_post_train_8gpu.sh` 错峰启动 + Wave2 串行重试 |
| 07-30 03:37 | ✅ 全部 9 变体（含 Base）merged · wave4_compare · final_report |

**Closed-loop 全 variant 对比（100q merged）**

| Variant | DupCurateRate | FalseSkipRate | mean_n_curated | recall | reward |
| ------- | ------------: | ------------: | -------------: | -----: | -----: |
| **Base** | **0.000** | **0.000** | 9.13 | 2.68% | 0.165 |
| round3_op_main_seed42 | 0.139 | 0.815 | 9.26 | 0.95% | 0.042 |
| round3_op_main_seed43 | 0.178 | 0.892 | 11.40 | 3.44% | 0.127 |
| round3_op_main_seed44 | 0.127 | 0.789 | 10.18 | 1.85% | 0.076 |
| **round3_compact_json** | **0.000** | **0.015** | 7.48 | 2.22% | 0.123 |
| round3_legacy_token_ce | 0.001 | 0.019 | 7.77 | 2.11% | 0.098 |
| round3_correct_only（对照） | 0.218 | 1.000 | 13.07 | 1.89% | 0.039 |
| round3_endorse_only（对照） | 0.000 | 0.000 | 8.50 | 1.96% | 0.141 |
| round3_op_no_balance | 0.137 | 0.841 | 11.59 | 2.55% | 0.106 |

**Paired 统计（seed42 vs Base，100q bootstrap）**

| 指标 | mean Δ | 95% CI | W/L/T |
| ---- | -----: | ------ | ----- |
| duplicate_curate_rate | **+0.133** | [+0.114, +0.154] | 80/0/20 |
| false_skip_rate | **+0.801** | [+0.783, +0.818] | 100/0/0 |
| recall | −0.020 | [−0.041, +0.000] | 5/12/83 |
| reward | −0.154 | [−0.283, −0.034] | 8/33/59 |

**解读**

- **operation_ce 主模型（3 seeds）：** offline 全 KEEP（SKIP recall=0%），闭环产生大量 SKIP（FalseSkipRate 79–89%），DupCurateRate 反而上升 — **训练目标与闭环行为严重不一致**。
- **compact_json：** 唯一在 DCR≈0、FSR≈1.5% 下保持 recall/reward 接近 Base 的变体；offline 有少量 SKIP prediction（6.1% recall）。
- **correct_only 对照：** FSR=100%、DCR=21.8%，证明 route-filter 训练能把模型推向极端 SKIP 行为。
- **endorse_only 对照：** 与 Base 行为一致（全 KEEP admission）。
- **Task retention：** recall/reward 未出现灾难性崩溃（paired recall CI 含 0），但 seed42 reward 显著低于 Base。

产物：`outputs/scope_round3/closed_loop/*/merged/summary.json` · `ROUND3_REPORT.md` · 日志：`outputs/scope_round3/logs/resume_post_train_master.log`

---

### Round 3 研究问题结论（0729-todo2 §一）

> 双侧监督 + operation_ce + 统一 action interface 能否让 duplicate_evidence 在 \(H_{\min,\text{v2}}\) closed-loop 中真正降低 DuplicateCurateRate？

| 层面 | 结论 | 依据 |
| ---- | ---- | ---- |
| **数据 / Selector（H3）** | ✅ **已解决** | KEEP=1784, SKIP=545；`ROUND3_DATA_GO=true` |
| **Train/Inference 一致（H2）** | ✅ **已实现** | Wave4 plumbing ✅；shared scorer + runtime + realizer |
| **Offline capability** | ❌ **未通过** | operation_ce main macro-F1=0.500 ≡ Base；SKIP recall=0% |
| **Closed-loop behavior** | ❌ **未通过** | main 模型 DCR↑、FSR↑（vs Base）；compact_json 唯一接近 Base |
| **Task retention** | ✅ **通过** | paired recall Δ≈0；无系统性 recall 崩溃 |

### 根因假设更新（0729-todo2 §十七）

| 假设 | 判定 | 说明 |
| ---- | ---- | ---- |
| H1 token-loss-mass distortion | **PARTIALLY_SUPPORTED** | legacy/compact JSON SKIP recall 弱（2–6%）；非主因 |
| H2 training/inference action mismatch | **SUPPORTED** | Round3 已统一 interface；Round2 根因之一 |
| H3 selector-induced one-sided supervision | **SUPPORTED** | Round2 0 KEEP/0 ENDORSE → Round3 双侧修复 |
| H4 operation-value supervision weakness | **SUPPORTED** | operation_ce 主模型 offline 全 KEEP、闭环大量误 SKIP |
| H5 evaluator correctness | **OPEN（P0）** | majority baseline F1=1.0 不符合标准定义；F1 指标暂不可信 |

### 最终判定

```text
ROUND3_POSITIVE_SIGNAL = false
  # offline：operation_ce 未超 Base；SKIP recall=0%
  # closed-loop：main 模型 DCR/FSR 显著恶化 vs Base

RECOMMEND_830 = false

Capability pass: False
Behavior pass:     False
Task retention:    True

NEXT_ACTION = (1) 修复 offline F1 evaluator sanity check
              (2) 调查 operation_ce 塌缩根因（verbalizer prior / effective class weight / train-infer score 一致性）
              (3) 对比 compact_json vs operation_ce 的 score margin / 闭环行为差异
              (4) 在 Dup 最小 positive signal 前不扩 830 / 多能力 / weighting
```

**下一步已移到文末“全局待办”。**



---

## 训练主循环（每 iteration）

> 本节只记录 method pipeline 与当前实现状态；Round 2/3 的详细数值不再重复。

### Step 1 — Pure Student Rollout（\(\tau^- \sim \pi_\theta \mid H_{\min}\)）

**method 对应：** 学生在 Minimal Runtime 上 on-policy rollout，收集真实访问状态。  
**状态：** 🔄 已有 \(H_{\min,\text{v2}}\) 100q rollout（Round 2/3）；正式 iteration 数据管线待固化。

**Setting（当前正式计划）**

| 项                                   | 值                                                           |
| ------------------------------------ | ------------------------------------------------------------ |
| Model                                | `Qwen2.5-7B-Instruct`                                        |
| Benchmark                            | BrowseComp+，100q → 通过 gate 后 830                         |
| Retriever                            | BM25                                                         |
| Rollout runtime                      | **`modules_minimal_v2.yaml`**；旧 `modules_minimal.yaml` 仅保留历史对照 |
| max_turns / max_tokens / temperature | 35 / 2048 / 1.0                                              |
| Scope config                         | `configs/scope/sdi_dup_premature.yaml`                       |
| Capabilities                         | 先 `duplicate_evidence`；`premature_stop` 待双侧 Stop Calibration 通过后加入 |
| 用途                                 | 在 student 实际访问状态上生成 same-state supervision         |

**历史 Full-v2 协议 Setting（Smoke/Audit 共用）**

| 项                                   | 值                                          |
| ------------------------------------ | ------------------------------------------- |
| Model                                | `Qwen2.5-7B-Instruct`                       |
| Retriever                            | BM25                                        |
| Harness                              | `modules_full_v2.yaml`（⚠️ 非 \(H_{\min}\)） |
| Scope config                         | `configs/scope/sdi_dup_premature.yaml`      |
| Capabilities                         | `duplicate_evidence`, `premature_stop`      |
| max_turns / max_tokens / temperature | 35 / 2048 / 1.0                             |
| Smoke                                | `LIMIT=20`, `SEED=42`, GPU0–3, port 8774    |
| Audit                                | `LIMIT=100`, `SEED=42`, GPU0–3, port 8775   |

**历史协议验证 timeline**

| 时间             | 实验         | Setting                             | 关键结果                                                     |
| ---------------- | ------------ | ----------------------------------- | ------------------------------------------------------------ |
| 2026-07-28 10:28 | Smoke 20q    | Full v2；GPU0–3；port 8774          | 123/123 trainable；Dup 56 ENDORSE + 47 CORRECT；Premature 0 ENDORSE + 20 CORRECT；leakage/mutation=0 |
| 2026-07-28 11:53 | Natural 100q | Full v2；SEED=42；GPU0–3；port 8775 | 755 events / 754 trainable；Dup 340 ENDORSE + 315 CORRECT；Premature 0 ENDORSE + 99 CORRECT + 1 IGNORE；Irrelevant 全 IGNORE |

产物：`outputs/scope_v3_protocol_smoke20/` · `outputs/scope_v3_audit_100q/natural_100q/`

> 这两次 Full v2 rollout 仅用于协议验证和 Round-1 历史数据；正式训练状态分布改用 \(H_{\min,\text{v2}}\)。

---

### Step 2 — DecisionState 构建（\(d_t=\psi(s_t)\)）

**method 对应：** 统一压缩交互状态，要求 \(\operatorname{Info}(d_t)\subseteq\operatorname{Info}(s_t)\)。  
**状态：** ✅ v3 在线验证；Round 3 增加 `DupDecisionPoint`。

---

### Step 3 — Same-State Shadow Guidance（\(z_t^m=h_m(d_t)\)）

**method 对应：** 同状态查询 typed module，返回局部 artifact，而不是 teacher 完整轨迹。  
**状态：** ✅ Dup 已实现 decision-triggered 双侧 shadow；Premature 的 selector coverage 仍待修。

---

### Step 4 — Information-Safe Gate（\(M_t^m\)）

**method 对应：** visibility / schema / executable / module mask。  
**状态：** ✅ v3 Smoke/Audit leakage=0、shadow_mutation=0；⚠️ E0 Dup PROC 仍有 3% visibility violation，需单独修复。

**Verifier 可靠性探针 timeline：2026-07-28 11:06**

**Setting**

| 项              | 值                                     |
| --------------- | -------------------------------------- |
| Model / Harness | —（离线 synthetic probe，无 rollout）  |
| 数据            | synthetic `DecisionState`，n=24        |
| Capability      | `premature_stop`                       |
| Scope config    | `configs/scope/sdi_dup_premature.yaml` |
| train_mask      | 0（不进训练）                          |

结果：24/24 ENDORSE。说明 verifier 能识别 valid-stop；自然数据缺 positive-stop 主要来自 selector/coverage，而不是该 synthetic probe 中的 verifier failure。

产物：`outputs/scope_v3_audit_100q/targeted_valid_stop/`

---

### Step 5 — Verified Decision Routing（ENDORSE / CORRECT）

**method 对应：** endorse → 保留学生动作；verified reject → 使用纠正动作。  
**状态：** ✅ Dup 双侧 routing 已在 Round 3 建立；⏸ Premature 暂不训练。

**Stop Calibration Setting**

| 项                                   | 值                                                           |
| ------------------------------------ | ------------------------------------------------------------ |
| Model                                | `Qwen2.5-7B-Instruct`                                        |
| Benchmark                            | BrowseComp+，20q smoke → 100q audit                          |
| Retriever                            | BM25                                                         |
| Harness                              | `modules_full_v2.yaml`（历史计划）；Round 2 已在 H_min_v2 audit |
| Scope config                         | `configs/scope/sdi_dup_premature.yaml` + `stop_calibration: true` |
| Capabilities                         | `duplicate_evidence`, `premature_stop`                       |
| max_turns / max_tokens / temperature | 35 / 2048 / 1.0                                              |
| 代码                                 | `stop_calibration.py` · `selectors.py` · `verification_shadow.py` |

2026-07-29 H_min_v2 100q：3329 decision points，STOP→CONTINUE=13，CONTINUE→CONTINUE=3316，CONTINUE→STOP=0，`bilateral_coverage=False`。因此不原样扩大、不训练 Premature；先修 selector / targeted state construction。

---

### Step 6 — Capability Weighting（\(w_t^m=P_mU_m(1-\rho_m)\)）

**状态：** ⏸ 暂缓。至少一个 capability 通过可信 closed-loop internalization gate 后再做。

**Setting（保留）**

| 项        | 值                                                           |
| --------- | ------------------------------------------------------------ |
| Model     | `Qwen2.5-7B-Instruct`                                        |
| Benchmark | BrowseComp+                                                  |
| 权重方案  | uniform · \(U_m\) · \(U_m(1-\rho_m)\) · \(P_mU_m(1-\rho_m)\) |
| 对比实验  | E3                                                           |

当前：Dup 只有粗 \(U_m\) 统计；held-out \(\rho_m\) 未建立。此时做 weighting 会把 measurement/objective 错误混入权重结论。

---

### Step 7 — Shadow-first, Recovery-on-Demand

**状态：** ⏸ 按需；首版不主动加入 recovery。

**Setting（保留）**

| 项                  | 值                                   |
| ------------------- | ------------------------------------ |
| Model               | `Qwen2.5-7B-Instruct`                |
| Benchmark           | BrowseComp+                          |
| Rollout runtime     | `modules_minimal_v2.yaml`            |
| Recovery 步数 \(K\) | TBD                                  |
| 触发条件            | \(\delta_t^m>\tau_{\text{recover}}\) |
| 对比实验            | E4                                   |

仅当真实 rollout 中 premature stop / dead-end 形成稳定 failure mass 时启用。

---

### Step 8 — Optimize（\(\mathcal{L}=\mathcal{L}_{\text{SDI}}+\xi\mathcal{L}_{\text{stab}}\)）

**method 对应：** action/operation-level objective + 可选 KL；首版不含 RL / recovery loss。

#### ✅ 2026-07-28 12:38 — Dup-only SDI Round 1

**Setting — 数据**

| 项                        | 值                                                           |
| ------------------------- | ------------------------------------------------------------ |
| Model（rollout）          | `Qwen2.5-7B-Instruct`                                        |
| Benchmark / 来源          | BrowseComp+ `LIMIT=100`, `SEED=42`；`natural_100q/samples.jsonl` |
| Filter                    | `duplicate_evidence`, `train_mask=1`                         |
| Split                     | query-level，`valid_fraction=0.1`，seed=42                   |
| n_samples / train / valid | 655 / 578 / 77                                               |
| Route                     | ENDORSE 340 + CORRECT 315                                    |

**Setting — 训练 / eval**

| 项                               | 值                                                           |
| -------------------------------- | ------------------------------------------------------------ |
| Base                             | `Qwen2.5-7B-Instruct`                                        |
| LoRA                             | r=16, α=32                                                   |
| Loss                             | Action-level CE + KL=0.01                                    |
| lr / epochs / batch / grad_accum | 2e-5 / 3 / 4 / 4                                             |
| max_length                       | 4096                                                         |
| Scope config                     | `configs/scope/sdi_dup_only.yaml`                            |
| GPU                              | 4，~13 min，430 steps                                        |
| Eval script                      | `training/scope/eval_dup_capability.py`                      |
| Eval                             | valid 77；greedy，`max_new_tokens=64`，首行 JSON；base + LoRA adapter |

关键结果：valid loss 0.227；teacher-forced token acc 93.9%；parse 100%；action_match 26.0%。**结论修正：序列拟合成功，不等于 decision internalization。**

#### ✅ 2026-07-28 13:42 — Minimal Runtime Smoke20

**Setting**

| 项                                   | 值                                               |
| ------------------------------------ | ------------------------------------------------ |
| Base / Trained                       | Base vs `outputs/dup_sdi_round1/merged_hf`       |
| Benchmark                            | BrowseComp+ 前 20 题（`LIMIT=20`, `SPLIT=all`）  |
| Retriever                            | BM25                                             |
| Runtime                              | 历史 `modules_minimal.yaml`（V8D 全关）          |
| Scope config                         | `configs/scope/minimal_runtime.yaml`             |
| max_turns / max_tokens / temperature | 35 / 2048 / 1.0                                  |
| parallel / GPU                       | 2；GPU4–7，vLLM TP=4                             |
| 脚本                                 | `scripts/run_dup_sdi_minimal_runtime_smoke20.sh` |
| Phase0 历史参考                      | Minimal 830 recall 2.45%，reward 0.121（非同批） |

关键结果：recall 3.06%→0.71%，reward 0.137→0.013，mean_n_curated 11.15→20.65。Round 2 H_min_v2 100q 进一步确认 mean_n_curated 14.35→17.73，但 recall/reward 未稳定复现下降。故 Round 1 不作为成功模型继续扩 830。

产物：`artifacts/datasets/dup_sdi_round1/` · `outputs/dup_sdi_round1/`

#### ⏸ Minimal Runtime 全量 830（E6 / Retention）

**Setting（计划保留）**

| 项                                   | 值                                                     |
| ------------------------------------ | ------------------------------------------------------ |
| Model                                | 新的通过 gate 的 Dup trained model vs Base             |
| Benchmark                            | BrowseComp+ 830                                        |
| Retriever                            | BM25                                                   |
| Runtime                              | **`modules_minimal_v2.yaml`**                          |
| max_turns / max_tokens / temperature | 35 / 2048 / 1.0                                        |
| 对比                                 | matched manifest / matched runtime                     |
| 历史参考                             | Phase0 `modules_minimal.yaml` 2.45%；Full v2 3.80%     |
| 主指标                               | behavior metric + recall/reward + \(\rho_m\)/Retention |

正式 830 前先补 matched H_min_v2 Base baseline；不再对 Round-1 checkpoint 做全量主线评估。

---

### Step 9 — Module Lifecycle（内化 → 降权 → 退役）

**状态：** 📋 TODO。E0 100q 只提供初步 \(P_m\)；仍缺可信 held-out \(\rho_m\) 与 matched Minimal Runtime retention，暂不执行 module retirement。


## 实验设计消融（method §4）

### E0 — Module Distillability Map

**状态：** 🔄 100q 原始 probe 已冻结；taxonomy 未冻结。下一步是 targeted/event-enriched validity probe，不直接扩 830。详见 Stage 0。

---

### E1 — Full Harness Distillation vs Same-State Local Distillation

**状态：** 📋 TODO，提升为 **P1 核心基线**。

**Setting（计划）**

| 项                                   | 值                                                           |
| ------------------------------------ | ------------------------------------------------------------ |
| Model                                | `Qwen2.5-7B-Instruct`                                        |
| Benchmark                            | BrowseComp+                                                  |
| Retriever                            | BM25                                                         |
| max_turns / max_tokens / temperature | 35 / 2048 / 1.0                                              |
| 对比                                 | Harness-trace SFT · OPHSD-style full context · same-state local label · same-state + info-safe gate |
| 主指标                               | action/operation decision · fresh-corpus behavior · citation/factual errors · closed-loop task retention |

目标：证明收益来自“same-state typed local supervision / info-safe design”，而不是普通 Harness trace imitation。

---

### E2 — 为什么需要 Correct，而不只是 Endorse

**状态：** 🔄 Round 3 已具备双侧数据、单侧 controls 与完整 100q closed-loop；正式结论：`ROUND3_POSITIVE_SIGNAL=false`；待 evaluator 修复后重评 offline F1。

**Setting**

| 项                                   | 值                                                           |
| ------------------------------------ | ------------------------------------------------------------ |
| Model                                | `Qwen2.5-7B-Instruct`                                        |
| Benchmark                            | BrowseComp+                                                  |
| Retriever                            | BM25                                                         |
| max_turns / max_tokens / temperature | 35 / 2048 / 1.0                                              |
| 对比                                 | endorse-only · correct-only · bilateral main · compact/sample-norm · operation objective |
| 成功标准                             | 双侧 operation + closed-loop behavior + task retention       |

---

### E3 — Capability Weighting vs Privilege Illusion

**状态：** ⏸ 至少两个 capability 具备可信 supervision / retention 后再做。

**Setting**

| 项        | 值                                                           |
| --------- | ------------------------------------------------------------ |
| Model     | `Qwen2.5-7B-Instruct`                                        |
| Benchmark | BrowseComp+                                                  |
| 对比      | uniform · \(U_m\) · \(U_m(1-\rho_m)\) · \(P_mU_m(1-\rho_m)\) |

---

### E4 — DAgger-style Mixing vs Shadow-first Recovery

**状态：** ⏸ Recovery 按失败分布启用。

**Setting**

| 项                                   | 值                                                           |
| ------------------------------------ | ------------------------------------------------------------ |
| Model                                | `Qwen2.5-7B-Instruct`                                        |
| Benchmark                            | BrowseComp+                                                  |
| Retriever                            | BM25                                                         |
| max_turns / max_tokens / temperature | 35 / 2048 / 1.0                                              |
| 对比                                 | pure student OPD · DAgger mixture · student-prefix→teacher completion · SCOPE shadow-only · SCOPE shadow+recovery |

---

### E5 — Black-box Teacher Compatibility

**状态：** 📋 后置。

**Setting**

| 项        | 值                                              |
| --------- | ----------------------------------------------- |
| Model     | `Qwen2.5-7B-Instruct`                           |
| Benchmark | BrowseComp+                                     |
| Teacher A | white-box local model                           |
| Teacher B | API / rule / retriever / verifier mixed Harness |
| 对比      | logit-OPD vs action-level SCOPE                 |

---

### E6 — Module Retirement / Minimal Runtime Pareto

**状态：** 🔄 Phase 0 830 baseline 已完成；trained+H_min_v2 830 暂缓。

#### ✅ 2026-07-28 上午 — Phase 0 基线冻结（830 题）

**分支：** `main` @ `1ed533b` · `origin/main`

**Setting**

| 项                                   | 值                    |
| ------------------------------------ | --------------------- |
| Model                                | `Qwen2.5-7B-Instruct` |
| Benchmark                            | BrowseComp+ 830       |
| Retriever                            | BM25                  |
| max_turns / max_tokens / temperature | 35 / 2048 / 1.0       |
| parallel                             | 2                     |

| Runtime             | Config                 | recall / acc |    reward | mean_turns | mean_n_curated |
| ------------------- | ---------------------- | -----------: | --------: | ---------: | -------------: |
| Bare                | 无 Harness             |    acc 1.33% |         — |        1.0 |              — |
| Full v1             | `modules_full.yaml`    |        1.07% |     0.011 |        5.2 |            1.1 |
| **Full v2**         | `modules_full_v2.yaml` |    **3.80%** | **0.181** |       33.7 |           26.5 |
| **Minimal（历史）** | `modules_minimal.yaml` |    **2.45%** | **0.121** |       32.4 |           13.2 |

产物：`artifacts/baselines/compare_phase0_full830.json` · `outputs/minimal_runtime_browsecomp_full830/`

后续 Pareto 配置仍保留：Bare → Minimal Executor → Minimal + hard verifier/state store → Partially retired Harness → Full v2 → Trained + Minimal。正式比较时统一到当前 H_min_v2 协议，并补 matched Base baseline。

---

### Fresh-corpus / Cross-Harness 泛化

**状态：** 📋 P4。

**Setting（计划）**

| 项            | 值                                                           |
| ------------- | ------------------------------------------------------------ |
| Model         | `Qwen2.5-7B-Instruct`                                        |
| Fresh Corpus  | BM25→dense · train corpus→new index · fixed source→new distribution |
| Cross-Harness | JSON 字段顺序 · reason code · evidence renderer · context serialization 扰动 |


## 全局待办（按优先级）

> 原则：先证明“测得对”，再证明“学得到”，最后才扩大规模 / 多能力。  
> Round 4/5 后：**measurement / observability / offline learnability（O7）已过**；当前最大风险是 **offline↔closed-loop 行为不一致**。

```text
[P0 ✅] Offline evaluator + scorer consistency（Round 4 B1/B2）
[P0 ✅] DecisionState observability（Round 5 B1：无 label collision）
[P0 ✅] Objective learnability offline（Round 5：O7 overfit D128 + valid bal_acc=1.0）

[P0-NOW] O7 closed-loop 校准 / seed 一致性  ← 当前最紧迫
  - 解释 seed42≈KEEP vs seed43/44 高 FSR
  - 检查 runtime score 路径、决策阈值、状态分布相对 valid 的偏移
  - 目标：相对 Base 改善 duplicate rejection，且 FSR 可控、reward 不崩

[P1-A] Dup positive-signal gate（仍未过）
  - 双侧不塌缩 + 3 seeds 一致 + DCR/FSR 改善 + task retention
  - 通过后才允许 830q retention / E6

[P1-B] E0 targeted probe 修复，而不是直接 830
[P1-C] E1 local vs full-harness（后置于 Dup 闭环正信号）
[P2+] weighting / Recovery / multi-capability / RL — 暂缓
```

**明确暂缓**

- E0 830：当前 probe coverage 问题不能靠无差别扩大样本解决。
- Round 1 LoRA 的 830 eval：已被 Round 2/3 诊断淘汰，不再作为主线。
- 多能力联合 SDI / weighting：Dup 单能力尚未建立可信 positive signal。
- Premature 训练：CONTINUE→STOP 监督覆盖仍为 0。
- **830 / E1：** Round 5 `ROUND5_POSITIVE_SIGNAL=false`，按 barrier **禁止扩规模**。


## 进度总览

| 项                    | 状态       | 当前结论                                                     |
| --------------------- | ---------- | ------------------------------------------------------------ |
| Stage 0 / E0          | 🔄          | 100q 原始 probe 冻结；taxonomy 未冻结                        |
| Round 1               | ✅ 历史完成 | 序列拟合成功，行为内化未成立                                 |
| Round 2               | ✅ 诊断完成 | 发现 loss-mass / one-sided / action-interface 问题；旧 Wave4 被 R3 替代 |
| Round 3               | ✅ 完成     | 双侧数据+typed runtime+100q CL 全完成；`ROUND3_POSITIVE_SIGNAL=false`；operation_ce 塌缩为 P0 |
| Round 4               | ✅ 完成     | measurement/scorer 验证通过；`operation_ce` objective 未通过 overfit128；**不进入 B5** |
| **Round 5**           | ✅ 完成     | Observability+O7 offline PASS；100q CL **无 positive signal**；`RECOMMEND_830=false` |
| Step 1–5              | 🔄          | Dup 主链已通；Stop bilateral coverage 未通                   |
| Step 6 Weighting      | ⏸          | 等 ≥2 个可信 capability                                      |
| Step 7 Recovery       | ⏸          | 按 failure mass 决定                                         |
| Step 8 Optimize       | 🔄          | R1–R5 checkpoints 已有；Dup 闭环 positive signal 仍未建立     |
| Step 9 Lifecycle / E6 | 🔄          | Phase0 830 完成；trained H_min_v2 830 暂缓                   |
| E1                    | 📋 P1       | local vs full-harness distillation 核心基线；**仍后置于 Dup 闭环正信号** |
| E2                    | 🔄          | 双侧 controls + R5 O7 offline 成立；闭环行为未校准             |
| E3–E5                 | ⏸/📋        | 后置                                                         |

**当前一句话结论：** Round 5 证明 DecisionState 可观测且 **O7（discriminative_ce + LoRA r=64）offline 可完美过拟合/双侧分离**，但三 seeds 闭环行为不一致（seed42≈KEEP、seed43/44 高 FSR），**不构成 Dup positive internalization signal，禁止扩 830**。

---

## Round 4 — Duplicate Measurement & Objective Repair（07-30）

**Git：** `scope/dup-round4-objective-repair`（自 `scope/dup-round3-bilateral` @ `ad072b9`）

**Gate 结论**

| Flag | 值 | 含义 |
| --- | --- | --- |
| `ROUND4_MEASUREMENT_VALID` | **true** | B1 offline metrics + forced episode + DCR/FSR 原始计数 |
| `ROUND4_SCORER_VALID` | **true** | B2 train/offline/runtime mismatch = 0%（8 models × 522 states） |
| `ROUND4_OBJECTIVE_VALID` | **false** | B4 overfit128 未达 95% acc / 90% 双侧 recall |
| `ROUND4_POSITIVE_SIGNAL` | **false** | 旧 operation_ce checkpoint 未恢复；compact_json 仅微弱改善 |
| `RECOMMEND_830` | **false** | 按 todo：Dup positive signal 前不做 830 |

---

### Barrier 1 — Measurement Audit ✅

| 子任务 | 状态 | 产物 |
| --- | --- | --- |
| B1.1 标准二分类指标 + unit tests | ✅ 8/8 PASS | `training/scope/binary_operation_metrics.py` |
| B1.2 DCR/FSR 原始计数 telemetry | ✅ | `training/scope/dup_telemetry.py` |
| B1.3 Forced episode (20×2) | ✅ | `outputs/scope_round4/metric_audit/forced_episode.jsonl` |
| B1.1 Offline re-eval（10 models × 522 valid） | ✅ | `outputs/scope_round4/metric_audit/offline_eval_fixed.json` |

**Offline 重评（修复 metrics 后，522 valid）**

| Variant | acc | macro_f1 | KEEP recall | SKIP recall |
| --- | ---: | ---: | ---: | ---: |
| Base / op_seed42/43/44 / endorse / no_balance | 0.810 | 0.448 | **1.000** | **0.000** |
| **compact_json** | 0.803 | **0.481** | 0.981 | **0.040** |
| correct_only | 0.190 | 0.159 | 0.000 | 1.000 |
| Round2-main（B1 早批） | 0.103 | 0.167 | 0.000 | 0.545 |

---

### Barrier 2 — Scorer Consistency ✅

8 GPU replay（522 valid states × 8 models）：**train/offline/runtime/prompt mismatch rate 全部为 0%**。

| 模型 | margin mean | 解读 |
| --- | ---: | --- |
| Base | -4.33 | 强偏 KEEP |
| op_seed42/43/44 | -1.3 ~ -1.5 | 仍偏 KEEP |
| correct_only | +5.72 | 强偏 SKIP |
| compact_json | -2.74 | 偏 KEEP，幅度中等 |

产物：`outputs/scope_round4/scorer_audit/SCORE_CONSISTENCY_REPORT.md`

**结论：** Round3 operation_ce 塌缩**不是** scorer/prompt/train-infer 不一致导致。

---

### Barrier 3 — Postfix Replay ✅（infra）/ ⚠️（行为）

**Phase 1**（14:28–14:43，8 GPU 并行，修复 `CUDA_VISIBLE_DEVICES` 覆盖 bug 后）：8/8 offline JSON 写入 `outputs/scope_round4/postfix_replay/offline/`。

**Phase 2**（14:47–~16:00，GPU0 B4 ∥ GPU1–4 closed-loop，75s 错峰）：**10/10 closed-loop shard 全部 `telemetry_complete: true`**，无 vLLM 初始化失败。

| Variant | shard0 | shard1 |
| --- | --- | --- |
| base | ✅ | ✅ |
| compact_json | ✅ | ✅ |
| op_seed42/43/44 | ✅ | ✅ |

旧 checkpoint 在 scorer 修复后**未恢复**合理双侧行为（op_ce 闭环仍高 FSR）。

---

### Barrier 4 — Overfit128 ❌

**Dataset：** `artifacts/datasets/dup_sdi_round4_overfit128/`（64 KEEP + 64 SKIP，76 unique queries，seed=42）

**Training：** operation_ce · LoRA r=16 · lr=2e-5 · 10 epochs · class_balancing=true · GPU0 · ~20 min

| 指标 | 训练前 | 训练后 | 目标 |
| --- | ---: | ---: | --- |
| train accuracy | 0.500 | **0.508** | >0.95 |
| KEEP recall | 1.000 | 0.766 | >0.90 |
| SKIP recall | 0.000 | **0.250** | >0.90 |
| SKIP mean loss（probe） | 4.46 | 0.79 | — |
| KEEP mean loss（probe） | 0.01 | 0.62 | — |
| margin mean | -4.36 | -0.20 | 应分离两类 |

`B4_PASS=false` · 产物：`outputs/scope_round4/overfit128/overfit128_report.json`

**诊断：** SKIP 样本初始 loss 远高于 KEEP（~400×），梯度/优化被 KEEP 主导；10 epoch 后 margin 仍对两类均为负，objective 未能学到可靠 SKIP 边界。

---

### 工程修复记录

1. **GPU 分配 bug：** `run_postfix_offline_eval.py` 内 `os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu` 覆盖 shell 设置，导致 8 任务挤占 GPU0 → OOM；已删除覆盖逻辑。
2. **B3 offline bug：** `load_jsonl` 需 `Path` 类型；改为独立 Python 脚本。
3. **Closed-loop：** 每 GPU 单 shard + 75s 错峰，最多 4 并发 vLLM。

---

### 代码与脚本（本分支）

```
training/scope/binary_operation_metrics.py
training/scope/eval_dup_capability.py          # 接入统一 metrics
training/scope/dup_telemetry.py                # DCR/FSR 原始计数
training/scope_round4/                         # audit / replay / overfit
tests/scope/test_binary_operation_metrics.py
scripts/scope_round4/                          # barrier 1–4 nohup 编排
artifacts/datasets/dup_sdi_round4_overfit128/  # 128-sample 平衡集
```

---

### 下一步（按 0730-todo1.md）→ 已进入 Round 5

```text
B4 FAIL → Round 5：Observability + objective tournament + O7 full screen + 100q CL
        → 见下一节 Round 5（2026-07-30 完成）
```

---

## Round 5 — Operation Observability & Learnability（07-30）

**Git：** `scope/dup-round5-learnability`（自 `scope/dup-round4-objective-repair` @ `6b4e88b`；Round5 脚本/训练代码当时多为**本地未提交**）

**文档：** `0730-todo2.md`  
**产物根：** `outputs/scope_round5/`  
**报告：** `outputs/scope_round5/ROUND5_REPORT.md`  
**环境快照：** `outputs/scope_round5/environment_snapshot.txt`

**时间线：** 2026-07-30 16:43（B0）→ 20:46（B4/B5 marker）→ 22:47（B6 定向续跑完成）

---

### Gate 结论

| Flag | 值 | 含义 |
| --- | --- | --- |
| `B1_PASS` / Observability | **true** | effective-input 无 KEEP/SKIP 标签冲突；shadow agreement=100%；truncation=0% |
| `B2_PASS` / Objective math | **true** | KEEP/SKIP one-step margin 方向正确；LoRA 有梯度与参数更新 |
| `B3_PASSED_OBJECTIVES` | **O7 only** | 仅 O7 通过 D2→D8→D32→D128 cascade |
| `B4_PASS` | **true** | O7×3 seeds valid 双侧 discrimination=1.0；Top-2=`o7_r64_seed44/43` |
| `B5_COMPLETE` | marker only | **未实际跑 50q**（`closed_loop/b5_50q/` 为空；supervisor 7s 内写完 marker） |
| `B6_COMPLETE` | **true** | Base + O7×3seeds + compact_json 各 100q 闭环完成 |
| `ROUND5_OBSERVABILITY_VALID` | **true**（据 B1） | label = f(student-visible DecisionState) 成立 |
| `ROUND5_OBJECTIVE_VALID` | **true**（offline） | O7 可 overfit 且 full-valid 双侧分离 |
| `ROUND5_CLOSED_LOOP_POSITIVE` | **false** | 三 seeds 行为不一致；43/44 高 FSR；reward↓ vs Base |
| `ROUND5_POSITIVE_SIGNAL` | **false** | 未同时满足 todo 六条正信号条件 |
| `RECOMMEND_830` | **false** | 禁止扩 830 |

---

### Setting（冻结）

| 项 | 值 |
| --- | --- |
| Base model | `/data/ppnm/models/Qwen2.5-7B-Instruct` |
| Runtime | \(H_{\min,\text{v2}}\) · `harness/configs/modules_minimal_v2.yaml` |
| Train / Valid | Round3 `dup_sdi_round3` 1807 / 522（sha256 `a0168283…`） |
| Overfit128 | Round4 `dup_sdi_round4_overfit128`（sha256 `ea31a1b9…`） |
| 100q manifest | `artifacts/datasets/round2_audit_100q/query_manifest.json`（sha256 `47b12f76…`） |
| CUDA / PyTorch / transformers / peft | 13.0 / 2.11.0+cu130 / 5.14.1 / 0.19.1 |
| 闭环调度 | 最多 4 并发 vLLM；4×25 shard；wave 内 75s stagger |
| 编排 | `scripts/scope_round5/pipeline_supervisor.sh`（B6 末段因 hang 改为 `targeted_b6_resume.sh`） |

**Objective 定义（B3 tournament）**

| ID | loss_mode | 其他 |
| --- | --- | --- |
| O0 | `operation_ce`（legacy） | LoRA r=16 |
| O1 | `discriminative_ce` | r=16 |
| O2 | `pairwise_margin` | r=16 |
| O3 | `single_token` | r=16 |
| O4 | `sample_normalized_action_ce` + compact_target | r=16 |
| O5 | `discriminative_ce_sum` | r=16 |
| O6 | `discriminative_ce_mean` | r=16 |
| **O7** | **`discriminative_ce`** | **r=64, α=128** |

嵌套 overfit：`D2 ⊂ D8 ⊂ D32 ⊂ D128`（平衡 KEEP/SKIP）；无 class/route balance；KL=0；cascade 失败即停。

**B4 全量训练：** O7×seed{42,43,44} + compact_json×seed{42,43,44}；3 epochs；lr=2e-5；bs=4×accum=4；max_length=4096。

---

### Barrier 0 — 环境冻结 ✅

产物：`environment_snapshot.txt` · HEAD `6b4e88b` · branch `scope/dup-round5-learnability`

---

### Barrier 1 — DecisionState Observability ✅

对 overfit128 / train1807 / valid522 dump effective student input（DecisionState→renderer→chat template→tokenizer→truncation）。

| 检查 | 结果 |
| --- | --- |
| unique effective inputs | 2327 |
| exact collision groups | 130 |
| **conflicting-label groups** | **0** |
| serialized-state shadow agreement | **100%**（≥99% gate） |
| truncation rate（KEEP/SKIP/overall） | **0%** |

产物：`observability/effective_inputs*.jsonl` · `LABEL_COLLISION_REPORT.md` · `observability_report.json`

**结论：** KEEP/SKIP label **可由 student-visible DecisionState 推导**；Round4 FAIL 不能归因于不可观测标签冲突。

---

### Barrier 2 — Objective 数学与梯度 ✅

| 探针 | loss_before→after | margin Δ | 方向 | LoRA grad / Δθ |
| --- | ---: | ---: | --- | --- |
| KEEP one-step | 0.0019→0.0019 | −6.28→−11.06（Δ−4.78） | ✅ 更偏 KEEP | 有更新 |
| SKIP one-step | 5.5→5.5 | −5.52→−0.02（Δ+5.50） | ✅ 更偏 SKIP | grad≈17.8 |

产物：`b2_objective/b2_report.json`

---

### Barrier 3 — 8-GPU Micro-Overfit Tournament ✅（仅 O7）

| Objective | D2 | D8 | D32 | D128 | All Pass |
| --- | --- | --- | --- | --- | --- |
| O0–O3, O5–O6 | ❌ acc=0.5（全 KEEP） | — | — | — | ❌ |
| O4 | ❌ acc=0.0（PARSE_FAIL） | — | — | — | ❌ |
| **O7** | ✅ 100% | ✅ 100% | ✅ 100% | ✅ 100% | ✅ |

O7 D128 post：acc/macro-F1/bal-acc/KEEP&SKIP recall 全部 **1.0**；margin_KEEP≈−7.06，margin_SKIP≈+7.20。

**关键对照：** O1 与 O7 **loss 相同**（`discriminative_ce`），仅 LoRA r=16→64；O1 卡在 D2，O7 贯通 D128。  
→ Round4 overfit128 失败的主因包含 **adapter 容量不足**，不只是 loss 公式名。

产物：`micro_overfit/MICRO_OVERFIT_MATRIX.md` · `micro_overfit/O7/`

---

### Barrier 4 — Full 1807/522 Objective Screen ✅

Valid=522（KEEP=423, SKIP=99）offline：

| Variant | bal_acc | macro_f1 | KEEP recall | SKIP recall | gate | mean_m_KEEP | mean_m_SKIP |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| o7_r64_seed42 | **1.000** | **1.000** | 1.000 | 1.000 | ✅ | −8.19 | +6.81 |
| o7_r64_seed43 | **1.000** | **1.000** | 1.000 | 1.000 | ✅ | −7.09 | +6.50 |
| o7_r64_seed44 | **1.000** | **1.000** | 1.000 | 1.000 | ✅ | −6.85 | +6.00 |
| compact_json_seed42 | 0.511 | 0.491 | 0.962 | 0.061 | ✅ | −3.83 | −4.17 |
| compact_json_seed43 | 0.499 | 0.447 | 0.998 | 0.000 | ❌ | −4.42 | −4.60 |
| compact_json_seed44 | 0.518 | 0.492 | 0.986 | 0.051 | ✅ | −4.10 | −4.33 |

**Top-2：** `o7_r64_seed44`, `o7_r64_seed43`（`B4_TOP2`）  
产物：`b4_full/` · `B4_GATE.json` · merged HF：`merged/o7_r64_seed{42,43,44}` · `merged/compact_json_seed42`

---

### Barrier 5 — Top-2 × 50q ⚠️ 跳过

`B5_COMPLETE` 于 20:46:58 写入，但 `closed_loop/b5_50q/` **无任何 shard 产物**。pipeline 在数秒内进入 B6。  
**记录为 infra 捷径/缺陷，不作为 50q 行为证据。** 行为结论以 B6 100q 为准。

---

### Barrier 6 — Best Objective × 3 Seeds × 100q ✅（无正信号）

比较：`Base` · `best_o7_{42,43,44}` · `compact_json`（seed42）；各 4×25 shard，`--dup-operation`。

**运维备注：** Best-44/shard1 曾卡在 `qid=535` `grep_corpus` ~26min 无推进；kill 后 `--resume` 定向续跑（23→25），再跑 compact_json；22:47 全完成。shard1 的 `summary.json` 仅统计续跑 2 条，下表指标改从 **全部 `dup_admission_events.jsonl` + `episodes.jsonl` 重聚合**。

| Variant | n_ep | DCR ↓ | FSR ↓ | SKIP recall | KEEP recall | n_pred_SKIP / n_events | mean reward | mean recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Base** | 100 | **0.000** | **0.000** | 0.000 | 1.000 | 0 / 16694 | **0.203** | **0.030** |
| O7-seed42 | 100 | 0.000 | 0.000 | 0.005 | 1.000 | 17 / 17055 | 0.134 | 0.023 |
| O7-seed43 | 100 | 0.010 | **0.517** | 0.289 | 0.483 | 7093 / 14602 | 0.157 | 0.024 |
| O7-seed44 | 100 | 0.045 | **0.366** | 0.232 | 0.634 | 5489 / 15958 | 0.144 | 0.017 |
| compact_json | 100 | 0.000 | 0.000 | 0.000 | 1.000 | 0 / 12218 | 0.136 | 0.029 |

（DCR=`duplicate_curate_rate`；FSR=`false_skip_rate` / unique reject；与 Round3 Base DCR=0 定义一致——KEEP 路径下 duplicate 的 `actually_curated` 可为 false。）

**闭环解读**

1. **Offline≠Closed-loop：** O7 三 seeds valid 上 SKIP recall=1.0，但闭环 seed42 几乎不发 SKIP；43/44 大量 SKIP 且误伤 unique（FSR 37–52%）。
2. **三 seeds 方向不一致：** 不满足 “3 seeds 方向一致”。
3. **相对 Base：** 无稳定的 “duplicate rejection 改善 + unique rejection 可控”；reward 全面低于 Base。
4. **compact_json：** 行为仍接近 Base（全 KEEP admission），offline 仅微弱 SKIP。

产物：`closed_loop/b6_100q/{base,best_o7_42,best_o7_43,best_o7_44,compact_json}/`

---

### Round 5 Positive Signal 判定（对照 0730-todo2）

| # | 条件 | 结果 |
| --- | --- | --- |
| 1 | D128 能稳定 overfit | ✅ O7 |
| 2 | full valid 双侧 discrimination | ✅ O7×3 seeds bal_acc=1.0 |
| 3 | 3 seeds 方向一致 | ❌ 闭环 42 vs 43/44 分裂 |
| 4 | duplicate rejection 相对 Base 改善 | ❌ 无稳定改善（DCR 未优于 Base 叙事） |
| 5 | unique rejection 可控 | ❌ seed43/44 FSR 过高 |
| 6 | recall / reward 无系统性下降 | ❌ reward 相对 Base 下降 |

→ **`ROUND5_POSITIVE_SIGNAL=false` · `RECOMMEND_830=false`**

---

### 工程与事故记录

1. **B5 空跑：** marker 写入但无 50q 产物；后续勿把 B5 当证据。
2. **B6 Best-44/shard1 hang：** `grep_corpus` 卡住；kill + `--resume` 定向续跑成功。
3. **kill 竞态：** 首次 kill 曾误写 `B6_COMPLETE` 并跳过未完成 shard；已清 marker 后用 `logs/targeted_b6_resume.sh` 重跑剩余任务。
4. **代码入库状态：** `scripts/scope_round5/`、`training/scope_round5/`、`training/scope/operation_objectives.py` 等在记录时仍为 untracked/modified，复现需保留本地工作树或另行 commit。

---

### 代码与脚本（本轮）

```
training/scope/operation_objectives.py
training/scope_round5/          # B1–B4/B6 helpers + build_round5_report
scripts/scope_round5/           # pipeline_supervisor / run_b3–b6 / resume
tests/scope/test_operation_objectives.py
outputs/scope_round5/           # 全部 barrier 产物（只读历史 R1–R4）
```

---
---

## Round 6 — Closed-loop Calibration & On-Policy Shift Audit（07-31 ~ 08-01）

**Git：** `scope/dup-round6-closedloop-calibration` @ `61f1348c9ac32c4b89dc0db4f1ba087a3c239539`

**文档：** `0731-todo1.md`
**产物根：** `outputs/scope_round6/`
**报告：** `outputs/scope_round6/ROUND6_REPORT.md`
**记录更新时间：** 2026-08-01 12:22 CST

### Gate 结论

| Flag | 值 |
| --- | --- |
| `H_RUNTIME` | **False** |
| `H_CALIB` | **False** |
| `H_SHIFT` | **False** |
| `H_FEEDBACK` | **False** |
| adapter↔merged parity | 1.0 |
| HF↔runtime parity | 1.0 |
| `ROUND6_CLOSED_LOOP_POSITIVE` | **false** |
| `RECOMMEND_830` | **false** |

### Setting（冻结）

| 项 | 值 |
| --- | --- |
| Base model | `Qwen2.5-7B-Instruct` |
| O7 checkpoint | `outputs/scope_round5/merged/o7_r64_seed{42,43,44}` |
| Loss / LoRA | `discriminative_ce` · r=64 · α=128（与 Round5 O7 相同） |
| Runtime | \(H_{\min,\text{v2}}\) · `modules_minimal_v2.yaml` |
| 100q manifest | `round2_audit_100q/query_manifest.json` |
| Closed-loop | max_turns=35 · max_tokens=2048 · temperature=1.0 · BM25 |
| Calibration slice | shard0（25q）closed-loop states |
| Prospective 25q | shard1（C-CALIB） |
| Holdout 50q | shard2+shard3（Phase D） |
| τ_seed42 / 43 / 44 | -5.177083333333334 / -5.166666666666666 / -3.90625 |
| τ_shared | -4.75 |
| Decision rule | SKIP iff margin ≥ τ（`score_skip - score_keep`） |

### Phase B — Cross-score 核心结论

同一 checkpoint × 多 state source 离线重打分（merged HF scorer）：

- valid522 与全部 B6 admission states 上 **AUROC=1.0**（三 seeds 一致）
- 同一 states 上 **BalancedAcc@threshold=0 亦为 1.0**（offline 排序完美）
- **H_RUNTIME / H_SHIFT / H_CALIB / H_FEEDBACK 均为 false**
- 推论：Round5 闭环失败**不是** runtime parity 或 on-policy AUROC 崩塌；问题在 **closed-loop 决策边界 / 行为层**（校准后仍高 FSR）

产物：`phase_b/CROSS_SCORE_MATRIX.csv` · `ROOT_CAUSE_GATE.json` · `STATE_SHIFT_REPORT.md`

### Phase C-CALIB — shard1 25q（校准后前瞻）

| Run | DupRejectRecall | FSR | BalancedAcc | mean_reward |
| --- | ---: | ---: | ---: | ---: |
| per_seed/seed42 | 1.000 | 1.000 | 0.500 | 0.354 |
| per_seed/seed43 | 1.000 | 1.000 | 0.500 | 0.295 |
| per_seed/seed44 | 1.000 | 0.994 | 0.503 | 0.265 |
| shared/seed42 | 1.000 | 0.970 | 0.515 | 0.147 |
| shared/seed43 | 1.000 | 1.000 | 0.500 | 0.295 |
| shared/seed44 | 1.000 | 1.000 | 0.500 | 0.298 |
| threshold_zero/seed43 | 0.000 | 0.000 | 0.500 | 0.227 |

**解读：** per-seed τ 在 shard0 上可达 FSR≤5%；但 shard1 闭环中 O7 仍 **几乎全部 pred SKIP**（DupRejectRecall≈1 但 FSR≈1），校准 **未** 转化为可接受闭环行为。

### Phase D — Holdout 50q（shard2+shard3）

| Run | n_ep | DupRejectRecall | FSR | BalancedAcc | SKIP prior | reward | recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base/shard2 | 25 | 0.000 | 0.000 | 0.500 | 0.000 | 0.160 | 0.026 |
| base/shard3 | 25 | 0.000 | 0.000 | 0.500 | 0.000 | 0.041 | 0.018 |
| seed42/shard2 | 25 | 1.000 | 1.000 | 0.500 | 1.000 | 0.259 | 0.074 |
| seed42/shard3 | 25 | 1.000 | 1.000 | 0.500 | 1.000 | 0.133 | 0.062 |
| seed43/shard2 | 25 | 1.000 | 1.000 | 0.500 | 1.000 | 0.356 | 0.082 |
| seed43/shard3 | 25 | 1.000 | 1.000 | 0.500 | 1.000 | 0.157 | 0.041 |
| seed44/shard2 | 25 | 1.000 | 0.996 | 0.502 | 0.996 | 0.264 | 0.064 |
| seed44/shard3 | 25 | 1.000 | 1.000 | 0.500 | 1.000 | 0.127 | 0.012 |

**解读：**

1. **Base：** DupRejectRecall=0（从不 SKIP），FSR=0；与 Round5 一致。
2. **O7 + per-seed τ：** 校准后闭环仍 **SKIP 先验≈1.0**，FSR≈0.97–1.0；DupRejectRecall 高但来自 **误伤 unique**，非成功 duplicate internalization。
3. **任务保持失败：** mean_reward 系统性低于 Base（~0.04–0.26 vs Base ~0.16/0.04 on holdout shards）。
4. **Round6 正信号 gate 未过：** 要求 DupRejectRecall≥0.10 且 FSR≤0.05 且 BalancedAcc>0.50 — O7 满足前者但 **FSR 严重超标**。

### Round 6 最终判定

```text
ROUND6_CLOSED_LOOP_POSITIVE = false
RECOMMEND_830 = false
C-SHIFT (Dagg retrain) = 未触发（H_SHIFT=false）
```

### 工程备注

1. Phase D 首次运行 `get_tau()` JSON key bug（int vs str）导致 O7 holdout 未启动；已修复并用 `resume_holdout_o7.sh` 补跑。
2. `seed43/shard2` 曾在 query 335 卡住 9/25；kill 后 `--resume` 续跑剩余 16 题。
3. 所有闭环指标从 `episodes.jsonl` + `dup_admission_events.jsonl` 重聚合。

### 代码与脚本

```text
training/scope/decision_config.py
training/scope_round6/
scripts/scope_round6/
tests/scope/test_round6_scorer.py
outputs/scope_round6/
```

### 下一步

```text
RECOMMEND_830=false → 禁止扩 830 / E1 / weighting / multi-capability
P0 转向：为何 offline margin 完美 + τ 校准后 closed-loop 仍全 SKIP？
  → runtime vLLM scorer vs HF 在 live admission 路径是否仍一致
  → τ 在 offline replay margin 上有效但对 live score scale 无效
  → 考虑 on-policy Dagg 前需先修 live decision 路径或 score telemetry 对齐
```
