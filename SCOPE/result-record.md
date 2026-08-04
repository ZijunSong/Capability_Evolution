## 当前结论（2026-08-04）

- **已成立：** same-state shadow / measurement+scorer · Round5 O7 offline · Round6/7 live contract 修复 · **Round7 τ=0 100q positive** · **`RECOMMEND_830=true`** · **Round 8 Phase 1 Gate 1A/1B/1C 全部通过**。
- **Round 8 Phase 2（✅ 完成）：** 8 variant rollback SDI 训练 + merge；主方法 O7×3 valid operation acc **0.75–0.76**；hint distill **0.81**。
- **Round 8 Offline Gate：** `offline_gate_pass=false`（checkpoint acc ~8.5% 未达 0.70）；`phase3_eligible=true`（operation acc >0.70）→ Phase 3 按 eligibility 启动。
- **Round 8 Phase 3（✅ 完成）：** 8 variant × 100q closed-loop **800/800 episode**；`hard_capability_positive_signal=false` · `main_seeds_pass=false`。
- **Round 8 总判定：** Dup retention 成功 · rollback **离线 operation 可学** · **闭环硬控制能力未成立**（offline→closed-loop 严重断裂）。
- **Round 6 判定（历史）：** `H_RUNTIME/H_SHIFT/H_CALIB/H_FEEDBACK` 均为 false；`ROUND6_CLOSED_LOOP_POSITIVE=false`。
- **Round 7 判定：** `ROUND7_TAU0_CLOSED_LOOP_POSITIVE=true`；`RECOMMEND_830=true`。
- **当前 P0：** 诊断 rollback offline/closed-loop 断裂根因；**禁止**在未通过 Hard-capability Gate 前扩 multi-capability / weighting / DAgger。

---

## Qwen rollout 八组实验汇总（2026-08-01）

**范围：** SCOPE 仓库下 Qwen3-1.7B / Qwen3-30B，在 HotpotQA 与 BrowseComp+ 上分别运行 bare rollout 与 full harness rollout，共 8 组。当前结论是 **7/8 已完成，唯一未完成项为 Qwen3-30B BrowseComp+ harness rollout**。

### 统一 setting

| 维度 | HotpotQA | BrowseComp+ |
| --- | --- | --- |
| 数据 | `external/hotpotqa_subset_queries.json` 或 `HotpotQA_raw_data_20260730.tar.gz::HotpotQA/hotpot_dev_fullwiki_v1.json` | `external/BrowseComp-Plus/` full 830 queries，BM25 index=`external/BrowseComp-Plus/indexes/bm25` |
| bare rollout | vLLM backend，temperature=1.0，max_new_tokens=2048，max_model_len=8192 | vLLM backend，temperature=1.0，max_new_tokens=2048，max_model_len=8192，split=`all` |
| harness rollout | `hotpotqa_local_context` retrieval，max_turns=35，max_tokens=2048，temperature=1.0 | `modules_full_v2.yaml`，BM25 retrieval，max_turns=35，max_tokens=2048，temperature=1.0，reranker=`none` |
| Qwen3-1.7B | `/mnt/songzijun/models/Qwen3-1.7B` | `/mnt/songzijun/models/Qwen3-1.7B` |
| Qwen3-30B | `/mnt/songzijun/models/Qwen3-30B-A3B-Instruct-2507` | `/mnt/songzijun/models/Qwen3-30B-A3B-Instruct-2507` |

### 完成状态与结果

| 模型 | 数据集 | 模式 | 状态 | 输出目录 | records / target | errors | recall | trajectory_recall | final_answer_recall | reward | 备注 |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen3-1.7B | HotpotQA | bare | ✅ completed | `outputs/bare_rollout_hotpotqa_qwen3_1p7b_4gpu/` | 7405 / 7405 | 0 | n/a | n/a | n/a | n/a | 7405 unique query_ids，bad_json=0 |
| Qwen3-1.7B | HotpotQA | harness | ✅ completed | `outputs/harness_rollout_hotpotqa_qwen3_1p7b_4gpu/` | 14580 / 14580 | 43 | 0.050274 | 0.064952 | 0.050274 | 0.181716 | 7405 unique query_ids，parallel=64，bad_json=0 |
| Qwen3-1.7B | BrowseComp+ | bare | ✅ completed | `outputs/bare_rollout_browsecomp_qwen3_1_7b_4gpu/` | 830 / 830 | 0 | n/a | n/a | n/a | n/a | split=`all`，bad_json=0 |
| Qwen3-1.7B | BrowseComp+ | harness | ✅ completed | `outputs/harness_rollout_browsecomp_qwen3_1_7b_8gpu_parallel32/` | 830 / 830 | 0 | 0.028161 | 0.202500 | 0.038333 | 0.160796 | manifest parallel=64，max_model_len=32768，bad_json=0 |
| Qwen3-30B | HotpotQA | bare | ✅ completed | `outputs/bare_rollout_hotpotqa_qwen3_30b_8gpu_20260730/` | 7405 / 7405 | 0 | n/a | n/a | n/a | n/a | bad_json=0 |
| Qwen3-30B | HotpotQA | harness | ✅ completed | `outputs/harness_rollout_hotpotqa_qwen3_30b_8gpu_parallel32_20260731_151339/` | 7405 / 7405 | 0 | 0.063336 | 0.070763 | 0.063336 | 0.212855 | parallel=64，bad_json=0 |
| Qwen3-30B | BrowseComp+ | bare | ✅ completed | `outputs/bare_rollout_browsecomp_qwen3_30b_8gpu_20260730/` | 830 / 830 | 0 | n/a | n/a | n/a | n/a | split=`all`，bad_json=0 |
| Qwen3-30B | BrowseComp+ | harness | ❌ incomplete | `outputs/harness_rollout_browsecomp_qwen3_30b_8gpu_parallel64_20260801_125717/` | 0 / 830 | n/a | n/a | n/a | n/a | n/a | no `harness_rollouts.jsonl` / no manifest；pid not alive |

### 当前结论

1. HotpotQA 上，Qwen3-30B harness 的 recall / reward 高于 Qwen3-1.7B harness：recall 0.063336 vs 0.050274，reward 0.212855 vs 0.181716。
2. BrowseComp+ 上，Qwen3-1.7B harness 已完整跑完 830 题，recall=0.028161，trajectory_recall=0.202500；Qwen3-30B 只有 bare 完整结果，harness 尚无可用结果文件，不能做完整横向比较。
3. bare rollout 均只记录生成轨迹，不含 recall/reward 类指标；可用于后续训练/审计数据，不应和 harness metric 直接比较。
4. Qwen3-30B BrowseComp+ harness 的最近尝试已完成 vLLM ready、BM25 preflight 和 830 pending episodes 初始化，但没有写出 `harness_rollouts.jsonl` / manifest，且 `vllm_server.pid=16495` 已不存活；判定为未完成而非完成失败可评分。

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
| Round 7 Live Decision Contract | Round 7         | `scope/dup-round7-live-decision-contract` | `a3a7c1ee`       | ❌ 未确认 push                     | `outputs/scope_round7/`                                      |
| Round 8 AgentCore + Rollback   | Round 8         | `scope/round8-agentcore-hardcontrol`      | `a3a7c1ee`       | ❌ 未 push                         | `outputs/scope_round8/` · `artifacts/datasets/scope_round8/` |

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

---

## Round 7 — Live Decision Contract Audit（08-01 ~ 08-02）

**Git：** `scope/dup-round7-live-decision-contract` @ `a3a7c1ee0019031edd0def187600797db90d8002`（自 Round 6 `61f1348` 分叉）

**文档：** `0801-todo1.md`  
**产物根：** `outputs/scope_round7/`  
**报告：** `outputs/scope_round7/ROUND7_REPORT.md` · `HOLDOUT_TAU0_SUMMARY.md` · `ROOT_CAUSE_GATE.json`  
**记录更新时间：** 2026-08-02 12:21 CST

**目标：** 证明并修复 Round 5/6 矛盾——offline AUROC=1.0、adapter parity=1.0，但 live closed-loop 行为与 replay 不一致（Round6 负 τ 下 FSR≈1.0；τ=0 下 Base 全 KEEP）。本轮对 **同一 live admission event** 审计 DecisionState、prompt、score、margin、threshold、operation、ActionRealizer 是否与 HF/vLLM replay 完全一致。

---

### Gate 结论

| Flag | 值 |
| --- | --- |
| `ROUND7_TRACE_VALID` (Gate A) | **true** |
| `ROUND7_LIVE_HF_PARITY` (Gate B) | **true** |
| `ROUND7_LIVE_VLLM_PARITY` (Gate B) | **true** |
| `ROUND7_THRESHOLD_INVARIANT_VALID` (Gate C) | **true** |
| `ROUND7_ACTION_REALIZER_VALID` | **true** |
| `ROUND7_TAU0_CLOSED_LOOP_POSITIVE` (Gate D) | **true** |
| `ROUND7_DAGGER_NEEDED` | **false** |
| `RECOMMEND_830` | **true** |
| 根因分类 | **R7-H6**（contract 全对；threshold=0 行为可解释，非 calibration overfit） |

Gate B 以 **operation parity=1.0** 为主判据（vLLM score 数值 parity ~95%，不影响 operation 一致）。

---

### Setting（冻结）

| 项 | 值 |
| --- | --- |
| Base model | `Qwen2.5-7B-Instruct` |
| O7 checkpoint | `outputs/scope_round5/merged/o7_r64_seed{42,43,44}` |
| Loss / LoRA | `discriminative_ce` · r=64 · α=128（与 Round5 O7 相同） |
| Runtime | \(H_{\min,\text{v2}}\) · `modules_minimal_v2.yaml` |
| 100q manifest | `round2_audit_100q/query_manifest.json` |
| Closed-loop | max_turns=35 · max_tokens=2048 · temperature=1.0 · BM25 |
| **Decision threshold** | **τ=0**（`SKIP iff margin ≥ 0`；**不使用** Round6 per-seed 负 τ） |
| Contract audit shard | shard1（25q）→ live trace + HF/vLLM replay parity |
| Holdout | shard2+shard3（50q）· 仅 Gate A–C 通过后运行 |
| GPU | 8×H20 144G · TP=1 · 最多 4 路 harness 并发 · 错峰 75s |
| Parallel | shard1 rerun `PARALLEL=64`（修复后）；seed43 重跑曾用 `PARALLEL=8` |
| Trace | `live_dup_decision_trace.jsonl` + `prompt_sidecar` + `decide_dup_operation()` 统一路径 |

**禁止：** 830q retention · E0/E1 · weighting · multi-capability · DAgger · 新 objective tournament（本轮未做）。

---

### 基础设施与 Parity 修复

**新增核心代码：**

```text
training/scope/decide_dup_operation.py          # 统一决策函数
training/scope/live_dup_decision_trace.py       # event-level trace
training/scope_round7/                          # replay / compare / gate / report
tests/scope/test_round7_contract.py             # 8 项 contract 测试
scripts/scope_round7/                           # 8-GPU queue / monitor / finish_seed43
```

**第一轮实验（parity 修复前，`contract_trace/live/`）：** 4 变体 shard1 live 完成，但 **Contract Gate 全失败**（live scorer 缺 `candidate_id`、共享 runtime 致 admission 重复、HF replay 双重 format、vLLM logprob 用 word count 等）。

**修复后重跑（`contract_trace/live_rerun/`）：**

| 修复项 | 说明 |
| --- | --- |
| `candidate_id` 传入 scorer | live prompt 与 replay 对齐 |
| `fork_for_query()` | `PARALLEL=64` 下隔离 telemetry，消除 admission 重复 |
| `score_rendered_prompt()` | HF replay 避免双重 `format_operation_prompt` |
| tokenizer token 边界 | vLLM logprob 对齐 |
| Gate B 主判据 | operation parity（非 score 数值 parity @1e-5） |

---

### Phase A — Shard1 Contract Audit（25q × 4 变体，`live_rerun`）

| Variant | trace | admission | Gate A | Gate B | vLLM op parity |
| --- | ---: | ---: | --- | --- | ---: |
| base | 341 | 341 | pass | pass | 1.000 |
| seed42 | 309 | 309 | pass | pass | 1.000 |
| seed43 | 308 | 308 | pass | pass | 1.000 |
| seed44 | 366 | 366 | pass | pass | 1.000 |

**Sentinel（Gate C）：** `+inf`→全 KEEP · `-inf`→全 SKIP · `τ=0`→argmax — 全部 pass。

**独立 archived-state audit（GPU4–6）：** 对第一轮 `live/` trace 做 HF/vLLM 独立 replay（`comparisons/*_indep`），辅助定位 parity 根因。

产物：`contract_trace/live_rerun/` · `contract_trace/comparisons/*_rerun/` · `LIVE_REPLAY_PARITY.csv` · `preflight/LIVE_DECISION_CALL_GRAPH.md`

---

### Phase B — Holdout τ=0（50q × 4 变体，`holdout_tau0_rerun`）

Gate D 阈值：DupRejectRecall≥0.10 · FSR≤0.05 · BalancedAcc>0.50 · 三 seeds 方向一致。

#### Pooled 50q（shard2+shard3）

| Variant | DupRejectRecall | FSR | BalancedAcc | SKIP prior | mean_reward | mean_recall | Gate D |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| base | 0.000 | 0.000 | 0.500 | 0.000 | 0.121 | 0.011 | baseline |
| seed42 | 1.000 | 0.000 | 1.000 | 0.202 | 0.175 | 0.026 | **PASS** |
| seed43 | 1.000 | 0.000 | 1.000 | 0.189 | 0.078 | 0.015 | **PASS** |
| seed44 | 1.000 | 0.002 | 0.999 | 0.202 | 0.104 | 0.033 | **PASS** |

#### Per-shard

| Run | n_ep | DupRejectRecall | FSR | BalancedAcc | reward | recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base/shard2 | 25 | 0.000 | 0.000 | 0.500 | 0.219 | 0.023 |
| base/shard3 | 25 | 0.000 | 0.000 | 0.500 | 0.023 | 0.000 |
| seed42/shard2 | 25 | 1.000 | 0.000 | 1.000 | 0.221 | 0.048 |
| seed42/shard3 | 25 | 1.000 | 0.000 | 1.000 | 0.129 | 0.004 |
| seed43/shard2 | 25 | 1.000 | 0.000 | 1.000 | 0.112 | 0.019 |
| seed43/shard3 | 25 | 1.000 | 0.000 | 1.000 | 0.045 | 0.010 |
| seed44/shard2 | 25 | 1.000 | 0.004 | 0.998 | 0.164 | 0.041 |
| seed44/shard3 | 25 | 1.000 | 0.000 | 1.000 | 0.045 | 0.024 |

#### Paired vs Base（bootstrap 95% CI，n=50）

| Seed | Δrecall | 95% CI | Δreward | 95% CI | W/L/T |
| --- | ---: | --- | ---: | --- | ---: |
| 42 | +0.0147 | [-0.0040, +0.0363] | +0.0547 | [-0.0909, +0.1949] | 6/2/42 |
| 43 | +0.0034 | [-0.0120, +0.0189] | -0.0424 | [-0.1522, +0.0316] | 3/2/45 |
| 44 | +0.0216 | [-0.0015, +0.0496] | -0.0162 | [-0.1291, +0.0841] | 6/2/42 |

**解读（对比 Round 6）：**

1. **Base @ τ=0：** 仍全 KEEP（DupRejectRecall=0，FSR=0），与 Round5/6 一致。
2. **O7 @ τ=0：** 三 seeds **双侧行为恢复**——DupRejectRecall≈1.0、FSR≈0、BalancedAcc≈1.0；SKIP prior≈0.19–0.20（**非** Round6 的 ≈1.0 全 SKIP）。
3. **与 Round6 负 τ holdout 对比：** Round6 O7 FSR≈0.97–1.0（误伤 unique）；Round7 contract 修复 + τ=0 后 FSR≈0，说明此前闭环失败主因是 **live decision contract drift**，而非 offline 排序能力不足。
4. **任务保持：** recall delta 95% CI 均包含 0（≤1pp 要求满足）；reward delta CI 较宽但未出现三 seeds 一致显著下降。
5. **Gate D 通过：** 三 seeds 方向一致且均满足 DupRejectRecall/FSR/BalancedAcc 阈值。

---

### Round 7 最终判定

```text
ROUND7_TRACE_VALID = true
ROUND7_LIVE_HF_PARITY = true
ROUND7_LIVE_VLLM_PARITY = true
ROUND7_THRESHOLD_INVARIANT_VALID = true
ROUND7_TAU0_CLOSED_LOOP_POSITIVE = true
ROUND7_DAGGER_NEEDED = false
RECOMMEND_830 = true
ROOT_CAUSE_CLASS = R7-H6
NEXT_ACTION = 扩 830 retention 验证（trained-vs-Base，不代表直接启动 multi-capability）
```

**根因结论（R7-H6）：** 排除 R7-H1~H5（state/prompt/score/threshold/realizer drift）；offline 与 live replay 在 contract 对齐后一致；**τ=0 为可解释自然决策边界**，无需 per-seed 负 τ 或 DAgger。

---

### 工程备注

1. **第一轮 Gate 全失败：** parity 修复前 `contract_trace/live/` 结果仅作 archived audit 参考；**正式结论以 `live_rerun/` 为准**。
2. **seed43 卡住：** GPU2 vLLM OOM + `--resume` 导致 trace 重复（442 events / 11 duplicate event_id）；09:05 起 `finish_seed43.sh` 干净重跑（`--no-resume`）后 Gate A/B 通过。
3. **monitor 脚本 bug：** `pgrep` 误匹配 nohup 父进程，seed43 replay 延迟 8h；已修复为 `python.*hmin_v2_dup_rollout.py` 精确匹配。
4. **GPU2 僵尸 vLLM：** Round6 残留进程曾占用 GPU；kill 后恢复。
5. **报告脚本：** 初版 `build_round7_report.py` 读 `live/` 导致报告失真；已改为读 `live_rerun/` + `holdout_tau0_rerun/`。

---

### 代码与脚本

```text
training/scope/decide_dup_operation.py
training/scope/live_dup_decision_trace.py
training/scope_round7/build_round7_report.py
training/scope_round7/compare_live_replay.py
training/scope_round7/contract_gate.py
training/scope_round7/replay_live_trace_{hf,vllm}.py
scripts/scope_round7/launch_all_8gpu.sh
scripts/scope_round7/finish_seed43.sh
scripts/scope_round7/monitor_rerun.sh
tests/scope/test_round7_contract.py
outputs/scope_round7/
```

---

### 下一步

```text
RECOMMEND_830=true → 允许下一轮 BrowseComp+ 830q SCOPE retention（trained O7 vs Base）
禁止：在未完成 830 retention 前启动 multi-capability / weighting / DAgger
主方法：threshold=0（不再使用 Round6 per-seed 负 τ 作为主决策边界）
可选：GPU7 续跑 Qwen3-30B BrowseComp+ harness 830（与 P0 audit 不争用 I/O）
```

---

## Round 8 — AgentCore 基线重构 + Rollback 硬控制能力（08-02 ~ 08-04）

**Git：** `scope/round8-agentcore-hardcontrol` @ `a3a7c1ee0019031edd0def187600797db90d8002`（自 Round 7 同 commit 分叉）

**文档：** `0802-todo1.md`  
**产物根：** `outputs/scope_round8/` · `artifacts/datasets/scope_round8/`  
**Gate 文件：** `HARD_CAPABILITY_GATE.json` · `OFFLINE_GATE.json` · `HARD_CAPABILITY_GATE_PHASE3.json`  
**记录更新时间：** 2026-08-04 10:34 CST

**目标：** （1）完成 Round 7 批准的 BrowseComp+ **830q matched Dup retention**；（2）将 bare vs harness rollout 重构为共享 **SearchAgentCore** 的严格对照；（3）新增并验证 **rollback_decision** 硬状态控制能力（typed `CONTINUE` / `REPLAN` / `ROLLBACK_TO`），数据收集 + SDI 训练 + closed-loop 评测。

**禁止（Hard-capability Gate 通过前）：** capability weighting · multi-capability 联合训练 · Recovery 全模块 · RL · DAgger · E0 无差别扩 830。

---

### Gate 结论（Phase 1 Barrier）

| Gate | 判据 | 结果 | 备注 |
| --- | --- | --- | --- |
| **1A** Dup 830 retention | Base + O7×3 各 830/830 · DupRejectRecall / FSR / BalancedAcc | **PASS** | 三 seeds DupRejectRecall=1.0 · FSR≈0 · BalancedAcc≈1.0 |
| **1B** AgentCore config diff | 仅允许 evidence_state / context_budget / verification / retrieval 模块差异 | **PASS** | runtime budget / tools / prompt 无漂移 |
| **1C** Rollback 数据集 | train≥1500 · valid≥400 · rollback 25–60% · healthy≥25% | **PASS** | 1980 events · 60/40 · train=1578 · valid=402 |
| `all_gates_pass` | 1A+1B+1C | **true** | 2026-08-03 修复 Gate 检查与数据集平衡后通过 |

**1A  pooled Dup telemetry（τ=0，来自 `HARD_CAPABILITY_GATE.json`）：**

| Variant | DupRejectRecall | FSR | BalancedAcc | SKIP prior |
| --- | ---: | ---: | ---: | ---: |
| base | 0.000 | 0.000 | 0.500 | 0.000 |
| seed42 | 1.000 | 0.000126 | 0.9999 | 0.209 |
| seed43 | 1.000 | 0.00311 | 0.9984 | 0.208 |
| seed44 | 1.000 | 0.000465 | 0.9998 | 0.195 |

**1B config diff：** `agent_core.yaml` vs `agent_core_full_harness.yaml` — 变更模块 `context_budget` · `evidence_state` · `retrieval`（rerank）· `verification`；`changed_budget_fields=[]` · `gate_1b_pass=true`。

**1C 数据集（`artifacts/datasets/scope_round8/rollback_sdi/`）：**

| 指标 | 值 |
| --- | ---: |
| raw events（去重前） | 3580 |
| balanced total | 1980 |
| train / valid | 1578 / 402 |
| rollback / healthy | 60% / 40% |
| visibility / schema / hash violations | 0 |

---

### Phase 1 — Setting（冻结）

| 项 | 值 |
| --- | --- |
| Base model | `Qwen2.5-7B-Instruct` |
| Dup O7 checkpoint | `outputs/scope_round5/merged/o7_r64_seed{42,43,44}` |
| Dup runtime | \(H_{\min,\text{v2}}\) · `modules_minimal_v2.yaml` |
| Dup threshold | **τ=0**（与 Round 7 一致） |
| Retriever | BM25 · BrowseComp+ 830q |
| Closed-loop | max_turns=35 · max_tokens=2048 · temperature=1.0 |
| 830 manifest | `artifacts/datasets/scope_round8/query_manifest_830.json` |
| 100q diagnostic manifest | `artifacts/datasets/round2_audit_100q/query_manifest.json` |
| GPU | 8×H20 · TP=1 · Phase 1 启动 2026-08-02 14:50 · 完成 ~22:20（~7.5h） |
| Preflight | `pytest tests/scope/` 142 项通过 |

**GPU 分配（Phase 1）：**

| GPU | 任务 A | 任务 B |
| --- | --- | --- |
| GPU0–3 | shard0–3：Base → O7-seed42 → O7-seed43 → O7-seed44（830q retention） | — |
| GPU4 | Qwen2.5 AgentCore 100q（shard0，25q） | natural rollback collection shard0 |
| GPU5 | Qwen2.5 AgentCore+FullHarness 100q（shard1，25q） | natural rollback collection shard1 |
| GPU6 | Qwen3-1.7B AgentCore → FullHarness（shard2，25q） | injected rollback collection shard2 |
| GPU7 | Qwen3-30B AgentCore → FullHarness（shard3，25q） | injected rollback collection shard3 |

**产物目录：**

```text
outputs/scope_round8/dup_retention_830/{base,seed42,seed43,seed44}/shard{0..3}/
outputs/scope_round8/agent_core_diagnostic/{agent_core,full_harness,qwen3_*}/shard*/
outputs/scope_round8/rollback_collection/{natural,injected}/shard*/
artifacts/datasets/scope_round8/rollback_sdi/{train,valid}.jsonl
```

---

### Phase 1 — 结果

#### 6.1 Dup 830q matched retention

| Variant | episodes 完成 | DupRejectRecall | FSR | BalancedAcc | SKIP prior |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | **830/830** | 0.000 | 0.000 | 0.500 | 0.000 |
| seed42 | **830/830** | 1.000 | 0.000126 | 0.9999 | 0.209 |
| seed43 | **830/830** | 1.000 | 0.00311 | 0.9984 | 0.208 |
| seed44 | **830/830** | 1.000 | 0.000465 | 0.9998 | 0.195 |

**结论：** 830q 规模下 O7 三 seeds **双侧 Dup 行为与 Round 7 holdout 一致**（DupRejectRecall≈1 · FSR≈0）；Base 仍全 KEEP。Retention 实验 **达标**，支持 Round 7 `RECOMMEND_830=true` 在更大样本上的延续。

#### 6.2 AgentCore 100q diagnostic（6 组 × 25q = 150 episodes）

| Config | Model | shard | episodes | mean_recall |
| --- | --- | --- | ---: | ---: |
| AgentCore | Qwen2.5-7B | shard0 | 25 | 0.0317 |
| AgentCore+FullHarness | Qwen2.5-7B | shard1 | 25 | 0.0600 |
| AgentCore | Qwen3-1.7B | shard2 | 25 | 0.0800 |
| AgentCore+FullHarness | Qwen3-1.7B | shard2 | 25 | 0.0310 |
| AgentCore | Qwen3-30B | shard3 | 25 | 0.0000 |
| AgentCore+FullHarness | Qwen3-30B | shard3 | 25 | 0.0293 |

**对照：** FullHarness vs AgentCore 在 Qwen2.5 上 recall 0.060 vs 0.032（+~2.8pp）；Qwen3-1.7B 上 AgentCore 反而略高（0.080 vs 0.031）；Qwen3-30B 两组均低（~0–0.03）。config diff 符合 Gate 1B（仅四模块开关差异）。

#### 6.3 Rollback 状态收集

| 模式 | shard | events |
| --- | --- | ---: |
| natural | shard0–3 | 500 + 500 + 479 + 482 = **1961** |
| injected | shard0–3 | 489 + 469 + 451 + 210 = **1669** |
| **raw 合计** | | **3580** |

经 `build_rollback_sdi_dataset.py` 去重 + 平衡（ROLLBACK 下采样至 60% · healthy upsample · query-level split）→ **1980** 条 SDI 训练集。

**工程备注：** injected 收集原始 rollback 比例 ~80%（injected 逻辑在 `len(checkpoints)≥2` 时倾向注入 ROLLBACK）；Gate 1C 通过数据集后处理修复，非重跑收集。

---

### Phase 2 — Setting（冻结 / 实际执行）

**启动条件：** Gate 1A/1B/1C 全部通过（2026-08-03 07:34 正式启动；OOM 修复后重启；**12:26 全部训练 + merge 完成**）。

| 项 | 规格（0802-todo1） | 实际执行 |
| --- | --- | --- |
| Base model | `Qwen2.5-7B-Instruct` | 同左 |
| Loss | discriminative CE over typed operations（O7） | 同左 |
| LoRA | r=64 · α=128 | 同左 |
| lr / epochs | 2e-5 · 3 | 同左 |
| max_length | 4096 | 4096 + **trainer 内 token 截断**（`max_length-96`） |
| batch / accum | 未明确 | **batch_size=1 · grad_accum=16**（OOM 修复后） |
| 优化 | — | gradient checkpointing · 逐样本梯度累积 |
| 数据 | `rollback_sdi/train.jsonl` · `valid.jsonl` | 同左 |
| query-level split | 是 | train 1578 · valid 402 |

**GPU 分配：**

| GPU | Variant | route 过滤 | n_train | 状态 |
| --- | --- | --- | ---: | --- |
| GPU0 | `rollback_o7_seed42` | 全量 | 1578 | ✅ DONE |
| GPU1 | `rollback_o7_seed43` | 全量 | 1578 | ✅ DONE |
| GPU2 | `rollback_o7_seed44` | 全量 | 1578 | ✅ DONE |
| GPU3 | `rollback_endorse_only` | ENDORSE | 614 | ✅ DONE |
| GPU4 | `rollback_prompt_hint_distill` | 全量 + hint | 1578 | ✅ DONE |
| GPU5 | `rollback_trajectory_imitation` | 全量 + trajectory | 1578 | ✅ DONE |
| GPU6 | `rollback_correct_only` | CORRECT | 964 | ✅ DONE |
| GPU7 | `rollback_soft_replan_only` | 全量（训练时跳过 ROLLBACK_TO） | 614 | ✅ DONE |

注：规格中 GPU3 原计划 `stop_o7_seed42`；Stop 双侧 Gate 未过，改跑 **`rollback_endorse_only`**（与 todo §7.1 fallback 一致）。

**产物：** `outputs/scope_round8/phase2_training/{variant}/` · merge → `outputs/scope_round8/merged/{variant}/`（8/8）

---

### Phase 2 — 结果（2026-08-03 12:26 完成）

| Variant | wall_clock | n_train | valid n | operation_accuracy | balanced_accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| `rollback_o7_seed42` | **286.1 min** | 1578 | 402 | **0.764** | **0.764** |
| `rollback_o7_seed43` | **289.9 min** | 1578 | 402 | **0.751** | **0.751** |
| `rollback_o7_seed44` | **288.6 min** | 1578 | 402 | **0.764** | **0.764** |
| `rollback_prompt_hint_distill` | **288.7 min** | 1578 | 1578* | **0.808** | **0.808** |
| `rollback_trajectory_imitation` | **289.4 min** | 1578 | 1578* | **0.749** | **0.749** |
| `rollback_correct_only` | **197.8 min** | 964 | 964* | **0.557** | **0.557** |
| `rollback_endorse_only` | **98.7 min** | 614 | 614* | 0.443 | 0.443 |
| `rollback_soft_replan_only` | **98.5 min** | 614 | 614* | 0.443 | 0.443 |

\* 过滤 route 的 variant 在 `train_report.json` 中 valid 集与 train 同规模（未单独 holdout split）。

**Offline Gate（`OFFLINE_GATE.json`，2026-08-03 16:36）：**

| 判据 | 主方法 O7×3 | 结果 |
| --- | --- | --- |
| operation balanced accuracy >0.70 | 0.751–0.764 | **PASS** |
| target checkpoint accuracy >0.70 | **0.085** | **FAIL** |
| invalid checkpoint <1% | 0% | PASS |
| 三 seed 方向一致 | yes | PASS |
| HF/merged/vLLM parity | 1.0 | PASS |
| `offline_gate_pass` | — | **false** |
| `phase3_eligible` | operation acc 达标 | **true** |

**解读：** 离线可学好 **operation type**（CONTINUE/REPLAN/ROLLBACK_TO），但 **checkpoint ID 选择几乎未学**（~8.5%）；与闭环所需能力不对齐，是后续 Phase 3 失败的重要前兆。

---

### Phase 2 — 工程备注

1. **首次 Phase 2 启动 OOM：** `student_state_text` 平均 ~9785 字符未截断 + 多样本 loss 同图 backward；修复后稳定运行。
2. **首次 launcher `wait` bug：** `launch … &` 导致未等待训练进程即打印 "Phase 2 complete"；已改为收集 PID 后 `wait`。
3. **Gate 1B 误判（已修）：** `compare_agent_configs` 曾将 harness module flags 记入 budget diff。
4. **Gate 1C 未过（已修）：** injected 收集 rollback 偏斜 ~80%；`build_rollback_sdi_dataset.py` 增加 balance + split slack。
5. **Phase 1 首次 Gate 失败：** 2026-08-02 22:20 自动流水线因 1B/1C 失败未启 Phase 2；修复后 2026-08-03 手动重建数据集并启动。

---

### Phase 3 — Setting（冻结 / 实际执行）

**启动条件：** `phase3_eligible=true`（2026-08-03 16:36 Offline Gate 评估后启动；**非** `offline_gate_pass`）。

| 项 | 值 |
| --- | --- |
| Manifest | `artifacts/datasets/round2_audit_100q/query_manifest.json`（100q） |
| Runtime | `agent_core_recovery.yaml` · AgentCore + RollbackRuntime |
| Scorer | vLLM merged adapter · `VllmRollbackScorer` |
| Parallel | `PARALLEL_PHASE3=16`（续跑降至 8 防 hang） |
| Sharding | 4 shard × 25q = 100 episode / variant |
| Dup threshold | τ=0 |
| 统一 budget | max_turns=35 · max_tokens=2048 · temperature=1.0 |

**GPU 分配（`launch_phase3_8gpu.sh`）：**

| GPU | Variant | merged 路径 | vLLM port 基址 |
| --- | --- | --- | ---: |
| GPU0 | `base_agent_core` | base model | 9400 |
| GPU1 | `rollback_o7_seed42` | merged/o7_seed42 | 9410 |
| GPU2 | `rollback_o7_seed43` | merged/o7_seed43 | 9420 |
| GPU3 | `rollback_o7_seed44` | merged/o7_seed44 | 9430 |
| GPU4 | `rollback_prompt_hint_distill` | merged/hint | 9440 |
| GPU5 | `rollback_trajectory_imitation` | merged/trajectory | 9450 |
| GPU6 | `rollback_correct_only` | merged/correct_only | 9460 |
| GPU7 | `rollback_soft_replan_only` | merged/soft_replan | 9470 |

每 GPU 内 shard0→shard3 **顺序**执行（每 shard 独占 port）；8 GPU **并行**。

**产物：** `outputs/scope_round8/phase3_closed_loop/{variant}/shard*/` · 汇总 `HARD_CAPABILITY_GATE_PHASE3.json`

**完成时间：** 2026-08-04 10:30（`rollback_correct_only` shard2/shard3 续跑 + aggregate）

---

### Phase 3 — 结果（100q closed-loop，800/800 episode）

#### 3.1 完成状态

| Variant | episodes | 状态 |
| --- | ---: | --- |
| `base_agent_core` | 100/100 | ✅ |
| `rollback_o7_seed42` | 100/100 | ✅ |
| `rollback_o7_seed43` | 100/100 | ✅ |
| `rollback_o7_seed44` | 100/100 | ✅ |
| `rollback_prompt_hint_distill` | 100/100 | ✅ |
| `rollback_trajectory_imitation` | 100/100 | ✅ |
| `rollback_soft_replan_only` | 100/100 | ✅ |
| `rollback_correct_only` | 100/100 | ✅（shard1 曾僵死 · shard2 vLLM OOM 后续跑） |

#### 3.2 闭环主指标（`HARD_CAPABILITY_GATE_PHASE3.json`）

| Variant | mean_recall | RollbackRecall | RollbackPrec | FalseRollback | ContinueRecall | ckpt_acc | op_bal_acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `base_agent_core` | 0.034 | 0.220 | 0.55 | 0.040 | 0.189 | 0.220 | 0.196 |
| `rollback_o7_seed42` | 0.012 | 0.253 | 0.70 | 0.029 | 0.013 | 0.253 | **0.075** |
| `rollback_o7_seed43` | 0.031 | 0.249 | 0.66 | 0.029 | 0.005 | 0.249 | **0.062** |
| `rollback_o7_seed44` | 0.026 | 0.280 | 0.58 | 0.037 | 0.012 | 0.280 | **0.062** |
| `rollback_prompt_hint_distill` | 0.039 | 0.127 | 0.27 | 0.067 | **0.671** | 0.127 | 0.563 |
| `rollback_trajectory_imitation` | 0.024 | 0.271 | 0.62 | 0.034 | 0.001 | 0.271 | **0.057** |
| `rollback_correct_only` | 0.026 | 0.229 | 0.58 | 0.037 | 0.000 | 0.229 | **0.052** |
| `rollback_soft_replan_only` | 0.032 | **0.000** | 0.00 | 0.000 | **1.000** | 0.000 | **0.796** |

**Hard-capability Gate 判据（0802-todo1 §8.2，主方法三 seed）：**

| 判据 | 要求 | O7×3 实际 | 结果 |
| --- | --- | --- | --- |
| RollbackRecall | ≥0.30 | 0.25–0.28 | **近达标但未稳过** |
| FalseRollbackRate | ≤0.05 | 0.029–0.037 | PASS |
| target checkpoint accuracy | ≥0.70 | **0.25–0.28** | **FAIL** |
| state hash restore rate | =1.0 | 聚合器报告异常* | **FAIL** |
| budget violations | =0 | 聚合器报告异常* | **FAIL** |
| 优于 prompt-hint / trajectory | 显著 | op_acc 远低于 hint（0.56 vs 0.08） | **FAIL** |

\* `aggregate_phase3_gate.py` 中 `state_hash_restore_rate` / `budget_violations` 计数逻辑待审计（当前数值与 event 量级不成比例）；但 **operation_balanced_accuracy 与 checkpoint_accuracy 的低值可信**。

#### 3.3 Phase 3 Gate 总判定

| 字段 | 值 |
| --- | --- |
| `main_seeds_pass` | **false** |
| `recovery_better_than_base` | **true**（RollbackRecall 略高于 base） |
| `hard_capability_positive_signal` | **false** |

---

### Phase 3 — 工程备注

1. **`RecoveryBudget.used_rollbacks` 属性错误** → 改为 `budget.remaining()`。
2. **`rollback budget exhausted` 崩溃整条 query** → `rollback_action_realizer` + rollout hook 降级为 REPLAN；通用 Exception 捕获。
3. **`rollback_correct_only` shard1 僵死**（17:24 停更 · GPU6 0% util）→ kill + `resume_correct_only_phase3.sh` 续跑完成。
4. **shard2 vLLM EngineCore 初始化失败**（shard1 残留 vLLM 占满 GPU6 ~134GB）→ kill EngineCore + `resume_correct_only_shard23` 续跑 shard2/shard3。
5. **launcher 改进：** `resume_correct_only_phase3.sh` 增加 `fuser -k` 清理陈旧 vLLM port。

---

### Round 8 总判定（2026-08-04）

```text
ROUND8_PHASE1_COMPLETE           = true
ROUND8_GATE_1A_PASS              = true   # Dup 830 retention
ROUND8_GATE_1B_PASS              = true   # AgentCore config diff
ROUND8_GATE_1C_PASS              = true   # rollback SDI dataset
ROUND8_PHASE2_COMPLETE           = true   # 8/8 variant train + merge
ROUND8_OFFLINE_GATE_PASS         = false  # checkpoint acc ~8.5% << 0.70
ROUND8_PHASE3_ELIGIBLE           = true   # operation acc >0.70
ROUND8_PHASE3_COMPLETE           = true   # 800/800 episodes
ROUND8_MAIN_SEEDS_PASS           = false
ROUND8_HARD_CAPABILITY_POSITIVE  = false
NEXT_ACTION                      = 诊断 offline→closed-loop 断裂；禁止扩 multi-capability
```

**结论摘要：**

1. **Dup 830 retention 成功：** Round 7 τ=0 结论在 830q 上复现；Gate 1A 通过；`RECOMMEND_830` 延续成立。
2. **AgentCore 公平基线已建立：** 100q diagnostic 完成；Gate 1B 确认仅四模块差异可解释 recall 漂移。
3. **Rollback SDI 离线 operation 可学：** O7×3 valid operation acc **0.75–0.76**；hint distill 最高 **0.81**；但 checkpoint ID acc 仅 **~8.5%**。
4. **闭环硬控制能力未成立：** 训练后闭环 operation_balanced_accuracy 仅 **~6–8%**（远低于离线 75%）；ContinueRecall≈0；模型在闭环中几乎不输出 CONTINUE，行为接近「全 ROLLBACK/REPLAN」。
5. **与 Dup 能力对比：** Dup O7 在 830q 上 DupRejectRecall≈1 · 闭环稳定；Rollback 呈现 **典型的 offline 可学、closed-loop 不迁移** 模式，与 Round 6/7 Dup 路径的成功形成对照。
6. **baseline 对照：** `soft_replan_only` 闭环 operation acc 最高（~0.80）但 RollbackRecall=0（训练时禁用 ROLLBACK_TO）；`prompt_hint_distill` ContinueRecall 最高（~0.67）但 rollback 质量差；**无一 variant 同时满足 Hard-capability Gate**。
7. **不可声称：** rollback 硬状态控制已内化 · 模型替代 rollback executor · Recovery Harness 已被吸收。

---

### 代码与脚本

```text
harness/recovery/*                     # RollbackRuntime contract
harness/configs/agent_core*.yaml
harness/configs/agent_core_recovery.yaml
training/scope/rollback_sdi_trainer.py
training/scope/rollback_operation_runtime.py
training/scope/vllm_rollback_scorer.py
training/scope_round8/build_rollback_sdi_dataset.py
training/scope_round8/check_phase1_gates.py
training/scope_round8/check_offline_gate.py
training/scope_round8/run_phase2_train.py
training/scope_round8/rollback_closed_loop_rollout.py
training/scope_round8/aggregate_phase3_gate.py
scripts/scope_round8/launch_phase1_8gpu.sh
scripts/scope_round8/launch_phase2_8gpu.sh
scripts/scope_round8/launch_phase3_8gpu.sh
scripts/scope_round8/run_offline_gate_and_phase3.sh
scripts/scope_round8/resume_correct_only_phase3.sh
scripts/scope_round8/wait_phase1_and_launch_phase2.sh
scripts/scope_round8/status.sh
tests/scope/test_agent_core_parity.py
tests/scope/test_rollback_contract.py
```

---

### 下一步

```text
1. 诊断 offline valid acc 75% → closed-loop op_acc 6% 的断裂（state 表示 / vLLM scorer / distribution shift / CONTINUE 类不平衡）
2. 审计 aggregate_phase3_gate.py 中 state_hash_restore_rate / budget_violations 计数
3. 若需 Round 9：优先修 measurement+closed-loop parity，而非扩 multi-capability
禁止：Hard-capability Gate 未过前启动 weighting · multi-capability · DAgger · Recovery 全模块
```
