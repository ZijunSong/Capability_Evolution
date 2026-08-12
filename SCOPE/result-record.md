# SCOPE result-record（精简汇总）

> 完整原文备份：`result-record.md.bak-20260810`（4150 行）。本文件保留各轮 **Setting / 分支 / 输出根 / 关键数值 / 判定**，合并重复实验与冗长叙述。  
> 仓库默认：`/data/ppnm/Capability_Evolution/SCOPE`（H100 机器路径常写作 `/mnt/songzijun/Capability_Evolution/SCOPE`）。

---

## 当前结论速览

| 线 | 状态 |
|---|---|
| **Round14 Capability Portfolio（进行中）** | 新 `ModuleRetirementGate` 已用 Dup FRESH100 校准通过（gate_c=true）；rollback_lite local Gate B FAIL；stop/verify/evidence/budget/external-verify Gate A FAIL→`UNRESOLVED`；Dup 830 confirm **T_OFF 未齐** |
| **Dup（R7–R8 / R14 校准）** | τ=0 closed-loop positive；R8 830 retention 成功；R14 新 gate 在 FRESH100 上仍判 `PROVEN` 方向（T_OFF bal≈1.0） |
| **Rollback（R8–R13 / R14-lite）** | 完整 rollback 闭环未成立（R13）；R14 `RECOVER/CONTINUE` local bal≈0.62、RECOVER recall≈0.31–0.33 → Gate B FAIL → hard negative |
| **H100 BrowseComp+ T0** | 1.7B 以 **0807 32768 版**为准；30B harness gain `+0.0315`；canonical textual final-answer 仍弱 |
| **H100 Finalization** | 瓶颈=evidence coverage + synthesis；严格 exact-budget 后 **query-selection / structure 无稳定因果增益** |
| **H100 HotpotQA T0** | bare scale gain 成立；T=0 harness collapse；1.7B↔30B 干预方向相反，**无 universal variant** |

---

## H20 Round14（0810-todo2）Capability Portfolio + Module-Retirement Gate（2026-08-10 → 08-11，**进行中**）

**Setting**

- 分支：`scope/round14-capability-portfolio`（自 `scope/round10-rollback-live-parity` @ `719c613257f6333c17e0c6c02af5db241832d0b5`）
- 指令：`0810-todo2.md` · 输出：`outputs/scope_round14/` · 数据：`artifacts/datasets/scope_round14/`
- 机器：8×H20 · env `bishop` · base `/data/ppnm/models/Qwen2.5-7B-Instruct`
- 方法主线：typed local decision（O7 discriminative）+ **ModuleRetirementGate**（Gate A/B/C），取代单一 `P_m`
- 冻结 manifest（seed=1414）：`R14_FRESH100`（已修为单 shard 100q）/ `R14_SMOKE20` / `R14_TRAIN_POOL` / `R14_HOLD_830`（8 shard）
- Dup 正锚：复用 R5 O7 merged `outputs/scope_round5/merged/o7_r64_seed{42,43,44}/`；闭环 T=0
- 对比矩阵（Dup）：`B_OFF` / `B_ON` / `T_OFF`（module OFF + O7 `--dup-operation`）

**分项进度**

| 项 | 状态 | 关键数值 / 说明 |
|---|---|---|
| 共享代码 + unit tests | **DONE** | adapters C0–C6、`gates.py`、retirement/eval、22–25 pytest pass；`BASELINE_AUDIT.md` / `RUN_MANIFEST.json` |
| GPU0 Dup FRESH100 校准 | **DONE** | 100q×{B_OFF,B_ON,T_OFF×3}；`DUP_RETIREMENT_GATE.json`：**gate_c_pass=true**，best_seed=42 |
| GPU0 Dup 830 confirm | **进行中** | B_OFF **830/830** · B_ON **830/830** · T_OFF **610/830**（缺 shard0/4/6；shard4 卡死待续跑） |
| GPU1 stop_decision | **DONE（Gate A FAIL）** | 事件单侧（全 CONTINUE）→ `UNRESOLVED`；未训练 |
| GPU2 verification_routing | **DONE（Gate A FAIL）** | 单侧 NO_VERIFY → `UNRESOLVED` |
| GPU3 evidence_admission | **DONE（Gate A FAIL）** | 单侧 ADMIT → `UNRESOLVED` |
| GPU4 context_budget_routing | **DONE（Gate A FAIL）** | 单侧 KEEP_CONTEXT → `UNRESOLVED` |
| GPU5 external_verification_routing | **DONE（Gate A FAIL）** | 单侧 DO_NOT → `UNRESOLVED`；卡曾借用跑 rollback seed43 |
| GPU6 rollback_lite | **DONE（Gate B FAIL）** | Gate A pass（R13 remap 6075/2957）；seed42/43 hard_boundary：bal≈**0.615/0.620**，CR≈0.92/0.91，**RR≈0.31/0.33** → local Gate B FAIL；未扩 closed-loop/830 |
| GPU7 method ablation | **部分 DONE** | B（typed local O7）复用 GPU0；A=full-trace **proxy only**；C info-safe 计数为 0；待 830 齐后补报告 |
| 最终汇总文件 | **未齐** | 缺 `ROUND14_REPORT.md` / `MODULE_RETIREMENT_SUMMARY.*` / `ROOT_CAUSE_DECISION.json` / `SHA256SUMS`；`CAPABILITY_PORTFOLIO.*` 仍为占位 |

**Dup FRESH100 关键数值（T=0）**

| 条件 | n | bal_acc | DupRejectRate | FSR | task recall |
|---|---:|---:|---:|---:|---:|
| B_OFF（base, module OFF, 无 DupRuntime） | 100 | —（无 telemetry） | — | — | 0.021 |
| B_ON（base, module ON） | 100 | — | — | — | 0.014 |
| T_OFF seed42 | 100 | **1.0** | 0.925 | 0.0 | 0.018 |
| T_OFF seed43 | 100 | **1.0** | 0.916 | 0.0 | 0.008 |
| T_OFF seed44 | 100 | 0.999 | 0.873 | 0.002 | 0.009 |

三 seed capability 改善方向一致（`direction_positive=[true,true,true]`）→ 新 gate **能识别已知正例**。

**rollback_lite 关键数值**

| seed | bal | CONTINUE recall | RECOVER recall | Gate B |
|---:|---:|---:|---:|---|
| 42 | 0.615 | 0.918 | **0.312** | FAIL |
| 43 | 0.620 | 0.914 | **0.327** | FAIL |

（未跑 seed44：seed42 已明显低于 bal≥0.75 / RR≥0.70；符合 early-stop）

**判定（截至 2026-08-11 上午）**

- `ModuleRetirementGate` 校准：**通过**（Dup FRESH100）
- Dup taxonomy 方向：维持 **PROVEN_INTERNALIZED** 候选；830 T_OFF 补齐前不作最终 830 确认结论
- rollback_lite：**CURRENTLY_NOT_INTERNALIZED**（local Gate B FAIL；完整 rollback 继续作 hard negative / capability boundary）
- stop / verify-routing / evidence-admission / budget-routing / external-verify-routing：**UNRESOLVED**（Gate A：自然/挖掘事件单侧，需 targeted bilateral 采集后才能训）
- 下一步 Go/No-Go：先杀并续跑 T_OFF 830 缺失 shard → 写齐 ROUND14 汇总；**不要**在 Gate A 未过前扩新 capability 830；勿再无限救完整 rollback checkpoint selector

**产物根：** `outputs/scope_round14/` · `artifacts/datasets/scope_round14/` · `scripts/scope_round14/` · `training/scope_round14/`  
**状态文件：** `outputs/scope_round14/STATUS_LIVE.md`

---

## H20 Round13（0810-todo1）Fresh On-Policy Rollback Distillation（2026-08-10，DONE · STOP_AFTER_STAGE1_VALID）

**Setting**

- 分支：`scope/round10-rollback-live-parity` @ `719c613257f6333c17e0c6c02af5db241832d0b5`
- 指令：`0810-todo1.md` · 输出：`outputs/scope_round13/`
- 方法：ONE-PASS ON-POLICY SAME-STATE SHADOW DISTILLATION（非 DAgger）
- 冻结 split（seed=1309，与 audit100 query-disjoint）：`R13_TRAIN200 / VALID100 / TEST100 / SMOKE20 / FINAL100`
- 旧 holdout（R8/9 offline_valid、R9–12 base_live、`round2_audit_100q`）仅 historical diagnostic

**结果**

| 项 | 值 |
|---|---|
| operation SDI | train/valid/test = 6075 / 2957 / 2980；conflict_rate=0 |
| hist→fresh domain AUC | ≈0.910 |
| Stage1（querynorm×3） | bal≈0.64 · CR≈0.85 · **RR≈0.40–0.44** → FAIL（gate RR≥0.70） |
| hard / query-norm | RR 0.400 ≫ nohard 0.283；略优于 event-uniform 0.344 |
| Stage2 pointer×3 | top1≈0.70–0.71 · MRR≈0.77 → FAIL（gate 0.75/0.88）；vs R11 listwise 0.627/0.808 |
| Stage2 退化？ | **否**（H0 latest≈0.13，entropy≈4 bits） |

**判定：** `STAGE1_VALID_GATE_PASS=false` · `STOP_AFTER_STAGE1_VALID=true` · 未跑 sealed TEST/Smoke/FINAL · operation/checkpoint internalization = **NO** · 完整 rollback hard = **NO**  
**产物：** `outputs/scope_round13/` · `artifacts/datasets/scope_round13/` · `ROUND13_REPORT.md` · `ROOT_CAUSE_DECISION.json`

---

## H20 Round12（H20-0809-todo1）Checkpoint Provenance → Operation Boundary（2026-08-09，DONE · STOP_AFTER_OPERATION_BOUNDARY）

**Setting：** 同分支 `@719c613` · 指令 `H20-0809-todo1.md` · 输出 `outputs/scope_round12/`

| 项 | 结果 |
|---|---|
| C9/C10 heuristic ckpt | top1/MRR=**1.0**（R9「0.892」=oracle_op 未 re-pick 的假象） |
| C11 listwise / pairwise | 0.627/0.808 · 0.608/0.732 → 未达 0.70/0.85 |
| M0×A0 tau=0 | live_bal=0.705 · CR=0.638 · RR=0.772（最平衡） |
| scalar/dual-view 选参 | live RR 崩到 0.17–0.22 → FAIL |

**判定：** `STOP_AFTER_OPERATION_BOUNDARY=true` · 未启动 Phase C / Stage2 重训 / 20q/100q · rollback hard = **NO**

---

## H20 Round11（0808-todo2）Operation/Checkpoint Decoupling（2026-08-08，DONE · STOP_AFTER_PHASE_B）

**Setting：** `@719c613` · `0808-todo2.md` · `outputs/scope_round11/` · Barrier0：offline_valid=402 · base_live=3347

| 阶段 | 要点 |
|---|---|
| Phase A | 最小过门视图 **A1 state-only**（bal/CR/RR=0.681/0.706/0.656）；checkpoint 语义推向 ROLLBACK |
| Phase B main×3（A1） | live CR≈0.95 但 **RR≈0.26–0.30**（CONTINUE 过激） |
| full_stage1（A0） | live_bal=**0.705** · CR=0.638 · RR=**0.772**（更平衡） |
| Stage2 listwise | ck_top1=**0.627** · MRR=0.808（仍 <0.70/0.85） |

**判定：** `FROZEN_LIVE_GATE.pass=false` · `STOP_AFTER_PHASE_B=true`

---

## H20 Round10 Followup（0808-todo1）Parity → CONTINUE Boundary（2026-08-08，DONE · STOP_AFTER_PHASE_B）

**Setting：** `@719c613` · `0808-todo1.md` · `outputs/scope_round10_followup/` · 复用 R9 P0 checkpoints

| 项 | 结果 |
|---|---|
| Canonical backend | agreement=1.0 · `CANONICAL_BACKEND_GATE=true`；R10-P8=**B. cross-backend numerical difference** |
| main_noweight×3 | live_bal≈0.65–0.66 · CR≈0.55–0.58 · ck_top1≈0.53 → Gate FAIL；span=0.009 |
| vs P0 | live CR 0.554 > P0 0.419；仍未过 0.70 |
| stage1_state_only | CR=0.996 但 RR=0.223 |

**判定：** `PHASE_B_GATE.pass=false` · `STOP_AFTER_PHASE_B=true` · rollback hard = **NOT ESTABLISHED**

---

## H100-1 BrowseComp+ Deterministic（主结论：0807 32768 版）

**Setting**

- 分支：`main` @ `0cf2d9eecea795354a1c6cc29d133606c14aa44f`
- 模型：Qwen3-1.7B · BrowseComp+ 830q · BM25 · `modules_full_v2.yaml`
- Decode：`temperature=0` · `seed=42` · **`max_model_len=32768`** · harness `max_turns=35`
- 输出：`outputs/h100_1_0807_{preflight20,full830}_qwen3_1p7b_browsecomp_deterministic/`
- ⚠️ `outputs/deterministic_main_qwen3_1p7b_browsecomp_t0/`（0805）为 **4096 ctx / seed 未接入** 污染结果，不得混写

| 条件 | n | 主指标 | paired harness−bare |
|---|---:|---|---|
| 0807 full830 bare | 830 | answer_match_acc **0.0108** | — |
| 0807 full830 harness | 830 | recall **0.0314** · far=0.037 · reward=0.153 · turns≈25.5 | **106/8/716** |

**判定：** matched T0 成立；harness 有正 paired signal，绝对分仍低。

---

## H100-2 BrowseComp+ Deterministic + Evidence-to-Answer（2026-08-05）

**Setting：** `main @ 0cf2d9e` · Qwen3-30B-A3B-Instruct-2507 · T0 · 32768 · `outputs/deterministic_main_qwen3_30b_browsecomp_t0/` · audit `outputs/h100_2_30b_evidence_to_answer_audit/`

| 模型 | bare | harness | Δ | W/L/T |
|---|---:|---:|---:|---|
| 30B | 0.0205 | **0.0520** | **+0.0315** | 76/17/737 |
| 1.7B（污染 T0，仅对照） | 0.000 | 0.0078 | +0.0078 | 11/0/819 |

30B harness：traj_recall=0.195 · reward=0.182 · mean turns≈35。  
C0–C7 readout/stop（unbiased200）：**无一超过 C0 natural 0.115**；C3 最好但仍 −0.080；`RECOMMEND_FULL830=false`。

**判定：** scale gain + harness gain 成立；剩余瓶颈在 **answer emission / finalization contract**，非 naive readout。

---

## H100-2 Finalization 线（合并 08-06 ~ 08-10）

**共享：** `outputs/h100_2_native_finalization_contract/manifests/finalization100.json` + F0 evidence · T0 · seed=42 · max_model_len=32768 · 不训练 / 默认不扩 full830

### 子实验索引

| 子实验 | 输出根 | 模型 |
|---|---|---|
| Native contract（08-06） | `outputs/h100_2_native_finalization_contract/` | 30B |
| Model-scale F0/F3（08-07） | `outputs/h100_2_model_scale_control_qwen3_1p7b/` | 1.7B vs 30B |
| Evidence-sufficiency（08-07） | `outputs/h100_2_finalizer_evidence_sufficiency_audit/` | 1.7B+30B |
| Retrieval-supply R0/R1（08-08） | `.../runs/retrieval_supply_paired/` | 30B |
| Query-conditioned compression（08-07） | `outputs/h100_4_query_conditioned_compression_finalization/` | 1.7B+30B |
| Structured S0–S4（08-08） | `outputs/h100_2_structured_evidence_mechanism_ablation/` | 1.7B+30B |
| **Exact-budget N/Q/QS（08-10，主结论）** | `outputs/h100_2_exact_budget_factorial/` | 1.7B+30B |

### 关键结果

| 实验 | 要点数值 | Flags / 判定 |
|---|---|---|
| Native F0/F2/F3 | F0 can=0 · F2/F3 can=**0.02**（parse/nonempty=100） | serialization bug + emitter works + **capability limit** · `RECOMMEND_FULL830=false` |
| 1.7B F0/F3 | can 均 0.02 · F3 vs F0 1/1/98 · vs 30B F3 0/0/100 | 无 scale×finalizer 正信号 |
| Evidence-sufficiency | native 0.02；oracle answer-present 0.06–0.08；gold-cert **0.94–0.98** | coverage + synthesis bottleneck；非 parser |
| Retrieval R0→R1 | answer-pos 0.61→0.64；canonical 仍 0.00；paired 6/5/89 | 小增益；不改 finalizer prompt |
| Compression S0→S1 | 1.7B 0.02→**0.07**；30B 0.01→**0.05** | 正信号但未触发停止 |
| Structured S0–S4 | 1.7B S4=0.05；30B S2=0.05（**未 token-matched**） | 被 08-10 部分推翻 |
| **Exact-budget** | 旧 S1–S4 mean tokens 1444→2890…未匹配；严格匹配后 1.7B/30B 在 B256/512/1024 上 Q/QS 均无稳定跨预算增益 | **`QUERY_SELECTION_CAUSAL_SIGNAL=false` · `STRUCTURE_ADDS_VALUE=false`** |

**合并结论：** 以 **08-10 exact-budget** 为准——预算受控 construction 或有小幅收益，但 **不支持** query-conditioned selection / claim-source 结构的稳定因果增益；普通 prompt/readout tournament 应停止。

---

## H100-1 Fresh Selection Replication（2026-08-10，DONE · fixed-budget verified）

**Setting**

- 指令：`H100-1-0808-todo2.md`（文件标题为 fresh held-out query-conditioned selection replication）
- 输出根：`outputs/h100_1_fresh_selection_replication/`
- Manifest：`manifests/fresh200.json`，BrowseComp+ fresh held-out 200q，stable hashing seed=43，排除 `outputs/h100_2_native_finalization_contract/manifests/finalization100.json`
- 验证：`fresh200_n=200`，`finalization100_overlap_n=0`
- 模型：Qwen3-1.7B、Qwen3-30B-A3B-Instruct-2507
- Decode：`temperature=0` · `do_sample=false` · `top_p=1` · `seed=42` · `max_model_len=32768`
- Retrieval：只用当前 R0 轨迹采集；Stage B readout 只复用 frozen trajectory evidence；**无 R1 / 无训练 / 无 full830**
- 最终采用修正后的 fixed-budget 输出：`qwen3_1p7b_fixed_budget/`、`qwen3_30b_fixed_budget/`
- Token contract：F1/F2 rendered evidence 每条均在 `min(raw evidence token count, 1024)` 的 **±5%** 内；最终校验两模型均 `budget_violations=0`

**结果**

| model / condition | n | canonical acc | correct | parser | supported | answer-positive n | mean evidence toks | mean prompt toks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.7B F0 raw | 200 | 0.000 | 0 | 1.000 | 0.565 | 30 | 534.2 | 713.5 |
| 1.7B F1 naive matched | 200 | **0.015** | 3 | 1.000 | 0.410 | 30 | 534.2 | 718.5 |
| 1.7B F2 query-conditioned flat | 200 | 0.010 | 2 | 1.000 | **0.705** | 30 | 534.2 | 713.5 |
| 30B F0 raw | 200 | 0.000 | 0 | 1.000 | 0.110 | 12 | 288.5 | 467.8 |
| 30B F1 naive matched | 200 | 0.005 | 1 | 1.000 | 0.030 | 12 | 299.4 | 472.8 |
| 30B F2 query-conditioned flat | 200 | 0.005 | 1 | 1.000 | **0.185** | 12 | 316.1 | 467.8 |

| comparison | W/L/T | Δacc | gate |
|---|---:|---:|---|
| 1.7B F2 vs F1 | **1/2/197** | −0.005 | FAIL |
| 1.7B F2 vs F0 | 2/0/198 | +0.010 | partial |
| 30B F2 vs F1 | **0/0/200** | +0.000 | FAIL |
| 30B F2 vs F0 | 1/0/199 | +0.005 | partial |

**判定 / 结论**

- Required replication gate：`F2 vs F1 wins > losses AND F2 vs F0 wins > losses`；fixed-budget 结果下 **两模型均未通过 F2 vs F1**。
- Final flag：`QUERY_CONDITIONED_SELECTION_IS_MANIFEST_SENSITIVE=true`
- `QUERY_CONDITIONED_SELECTION_GENERALIZES=false`
- 解释：F2 在 support rate 上明显提高，但 canonical final-answer accuracy 未稳定优于 budget-matched naive control；因此 fresh200 不支持继续沿 query-conditioned flat selection 方向自动扩 full830。
- 产物：`FINAL_REPORT.md`、`ROOT_CAUSE_DECISION.json`、`RUN_MANIFEST.json`、`per_condition_metrics.json`、`paired_comparisons.json`、`SHA256SUMS`

---

## H100-3 HotpotQA Deterministic 2×2（2026-08-05，DONE）

**Setting：** `main @ 0cf2d9e` · HotpotQA fullwiki **7405q** · T0 · 32768 · `outputs/deterministic_main_hotpotqa_2model_2mode_t0/`

| 模型 | bare acc | harness recall | traj | reward | turns | harness−bare W/L/T · Δ |
|---|---:|---:|---:|---:|---:|---|
| 1.7B | **0.152** | 0.047 | 0.063 | 0.174 | 13.3 | 457/1091/5857 · **−0.105** |
| 30B | **0.283** | 0.060 | 0.069 | 0.205 | **34.4** | 481/2039/4885 · **−0.223** |

bare 30B−1.7B Δ=+0.131；2×2 interaction=**−0.118**。

**判定：** model-scale（bare）成立；**T=0 harness gain 为负**；30B greedy long-loop；不做 T=0 harness 盲目扩跑。

---

## H100-3 HotpotQA 诊断链（合并 decomposition → controller 2×2）

**共享：** frozen `unbiased200` from `outputs/h100_3_hotpotqa_harness_decomposition_t0/` · T0 · seed=42 · 不扩 full7405

| 子实验 | 输出根 | 关键结果 | 判定 |
|---|---|---|---|
| Decomposition C0–C3 | 同上 | 1.7B：C2 minimal **0.290** ≫ C0 0.150；30B：C3 forced **0.285** ≫ C0 0.015 | 1.7B=module interference；30B=finalization + long-loop |
| Turn-cut | `.../turn_cut_curve_t0/` | 1.7B best@35=0.235；30B best@**25**=0.290 | 30B 过长无益 |
| Late-loop S×E | `.../late_loop_factorization_t0/` | 1.7B best S35+E25=0.340（late evidence −0.02）；30B 全高≈0.42–0.435 | scale-dependent；后经 readout audit 下调为 contract-sensitive |
| Readout contract | `.../readout_contract_audit/` | Barrier A **800/800** mismatch（`state_text_hash`）；replay parity=true；canonical E25/E35 1.7B 0.230/0.235 · 30B 0.455/0.455 | `READOUT_CONTRACT_DRIFT=true` |
| Evidence compaction | `.../evidence_compaction_t0/` | E2 compact vs E1：1.7B 12/12/176 · 30B 4/4/192；acc 持平 | **compaction 不可恢复 dilution** |
| **Controller×Finalization** | `.../controller_finalization_factorial/` | 见下表 | **无 universal 组合** · fresh200 未启动 |

| model | A full+native | B full+forced | C min+native | D min+forced | interaction |
|---|---:|---:|---:|---:|---:|
| 1.7B | **0.355** | 0.230 | 0.295 | 0.275 | +0.105 |
| 30B | 0.765 | 0.450 | **0.965** | 0.485 | −0.162 |

**Flags：** `UNIVERSAL_MINIMAL_PLUS_FORCED_CANDIDATE=false` · `HARNESS_INTERVENTION_IS_SCALE_DEPENDENT=true` · `RECOMMEND_FULL7405=false`  
**叙事：** small model 偏向减 controller；large model 偏向保留 search + 显式 finalization——**不能合成单一 universal variant**。

---

## Qwen T=1.0 Rollout 八组（2026-08-04，8/8 DONE）

**Setting：** `main @ 0cf2d9e`（领先 origin 1 commit；工作树 dirty；manifest 无 git commit）· **temperature=1.0** · bare max_model_len=8192（BrowseComp harness 32768）

| 模型 | 数据 | 模式 | 输出目录 | n | recall | traj | reward |
|---|---|---|---|---:|---:|---:|---:|
| 1.7B | HotpotQA | bare | `outputs/bare_rollout_hotpotqa_qwen3_1p7b_4gpu/` | 7405 | — | — | — |
| 1.7B | HotpotQA | harness | `outputs/harness_rollout_hotpotqa_qwen3_1p7b_4gpu/` | 7405 | 0.050 | 0.065 | 0.182 |
| 1.7B | BrowseComp+ | bare | `outputs/bare_rollout_browsecomp_qwen3_1_7b_4gpu/` | 830 | — | — | — |
| 1.7B | BrowseComp+ | harness | `outputs/harness_rollout_browsecomp_qwen3_1_7b_8gpu_parallel32/` | 830 | 0.028 | 0.203 | 0.161 |
| 30B | HotpotQA | bare | `outputs/bare_rollout_hotpotqa_qwen3_30b_8gpu_20260730/` | 7405 | — | — | — |
| 30B | HotpotQA | harness | `outputs/harness_rollout_hotpotqa_qwen3_30b_8gpu_parallel32_20260731_151339/` | 7405 | 0.063 | 0.071 | 0.213 |
| 30B | BrowseComp+ | bare | `outputs/bare_rollout_browsecomp_qwen3_30b_8gpu_20260730/` | 830 | — | — | — |
| 30B | BrowseComp+ | harness | `outputs/harness_rollout_browsecomp_qwen3_30b_8gpu_parallel64_20260801_125717/` | 830 | 0.045 | 0.229 | 0.169 |

> bare 无 recall/reward，不与 harness 直接比；T=1.0 仅作历史基线，不作 T=0 paired 结论。

---

## Dup / Rollback 主线 Round 2–9（合并精简）

### 分支谱系

| 阶段 | 分支 | commit | 产物根 |
|---|---|---|---|
| E0 Distillability | `main` + 本地 | `3e95fad` | `outputs/scope_e0_distillability/` |
| Round 2 Behavioral | `scope/dup-round2-behavioral`（未 push） | 代码 `ad072b9` | `outputs/scope_round2/` |
| Round 3 Bilateral | `scope/dup-round3-bilateral` | **`ad072b9`** | `outputs/scope_round3/` |
| Round 4 Objective | `scope/dup-round4-objective-repair` | `6b4e88b` | `outputs/scope_round4/` |
| Round 5 Learnability | `scope/dup-round5-learnability` | `6b4e88b`+本地 | `outputs/scope_round5/` · O7 merged |
| Round 6 Closed-loop | `scope/dup-round6-closedloop-calibration` | `61f1348c9ac32c4b89dc0db4f1ba087a3c239539` | `outputs/scope_round6/` |
| Round 7 Live Contract | `scope/dup-round7-live-decision-contract` | `a3a7c1ee0019031edd0def187600797db90d8002` | `outputs/scope_round7/` |
| Round 8 AgentCore+Rollback | `scope/round8-agentcore-hardcontrol` | `a3a7c1e…` | `outputs/scope_round8/` |
| Round 9 / P0 | `scope/round9-rollback-parity` | **`719c613…`** | `outputs/scope_round9/` · `wave_b_p0/` |
| Round 10–13 | `scope/round10-rollback-live-parity` | **`719c613…`** | `outputs/scope_round{10_followup,11,12,13}/` |
| Round 14 | `scope/round14-capability-portfolio` | **`719c613…`**+本地 | `outputs/scope_round14/`（见上文 Round14 节） |

早期 Phase0/v3/R1 在 `main` 本地未提交代码跑通；产物目录与 git 分支解耦。

### E0 Distillability（100q）

BrowseComp+ audit100 · Qwen2.5-7B · OFF/PROC/FULL。Dup Δ^proc=+0.0088（HYBRID-CANDIDATE/LOW-CONF）；其余 capability PROC 弱/无效。→ P0 转 `duplicate_evidence`；**不扩 E0 830**。

### Round 2–4（Dup 测量与 objective）

| Round | Setting 要点 | 关键结果 | 判定 |
|---|---|---|---|
| R2 | minimal_v2 · LoRA r=16 · 8 路消融 | Base recall 2.29%；数据集 KEEP/SKIP=0/287（单侧） | `POSITIVE_SIGNAL=NOT_ESTABLISHED` · `RECOMMEND_830=false` |
| R3 | 双侧 KEEP=1784/SKIP=545 · operation_ce | offline macro-F1=0.5、**SKIP recall=0**；闭环 DCR+0.13 但 recall↓ | `POSITIVE_SIGNAL=false` |
| R4 | measurement + overfit128 | measurement/scorer **VALID**；SKIP recall 0.25≪0.90 | objective **INVALID** → R5 |

### Round 5–7（O7 → τ=0 闭环）

| Round | Setting | 关键结果 | 判定 |
|---|---|---|---|
| R5 | O7=`discriminative_ce` · LoRA r=64 · 8-obj tournament | offline **bal_acc=1.0**；闭环 reward 低于 Base，seed 分裂 | offline VALID · closed-loop **false** |
| R6 | O7+负 τ 校准 | AUROC=1.0；holdout FSR≈0.97–1.0（全 SKIP） | `CLOSED_LOOP_POSITIVE=false` |
| **R7** | **τ=0** · contract + HF/vLLM parity | holdout DupRejectRecall=**1.0** · FSR≈0 · bal≈1.0 | Gate A–D **PASS** · **`RECOMMEND_830=true`** · 根因 R7-H6 contract drift |

### Round 8–9（Rollback hard；含 P0）

**R8 Setting：** `@a3a7c1e` · Dup O7 τ=0 + Rollback O7 · `outputs/scope_round8/`

| Phase | 结果 |
|---|---|
| 1A Dup 830q | O7×3 DupRejectRecall=1.0 · FSR≈0 · Gate 1A/1B/1C **PASS** |
| Phase2 offline rollback | op_bal **0.75–0.76** · ckpt_acc≈0.085 · offline_gate **false** |
| Phase3 closed-loop 100q | op_bal **≈0.06–0.08** · ContinueRecall≈0 |

→ Dup retention 成功；rollback **offline 可学、闭环不迁移** · `HARD_CAPABILITY_POSITIVE=false`

**R9 + P0 Setting：** `@719c613` · hier Stage1/2 · `outputs/scope_round9/`

| 阶段 | 结果 |
|---|---|
| Wave A | 主路径 HF↔vLLM agreement=1.0（6/8 PASS） |
| Oracle | learned_op+ckpt bal=0.53；**oracle_op+learned_ckpt=1.0** |
| Wave B hier×3 | offline bal≈0.72 · Cont≈0.49 · Replan=0；holdout Cont≈0.18 · Gate **FAIL** |
| **P0**（禁 REPLAN + CONTINUE 上采样 75%） | offline bal≈0.77–0.80 · holdout Cont≈0.45–0.49 · holdout parity≈**0.75** · Gate **仍 FAIL** |

→ Wave C / rollback 830 / DAgger / RL **禁止**，直至 Offline Gate 通过（后续 R10–13 亦未打通）。

---

## 附录：方法与历史待办（极简）

| 节（原文档） | 类型 | 摘要 |
|---|---|---|
| 训练主循环 | 方法论 | method Steps 1–9；Dup Step1–5/8 已通；Weighting/Recovery 暂缓 |
| 实验设计消融 §4 | 方法论 | E0–E6 计划；仅 E0/Phase0 有结果；E6 Phase0 Full v2 recall 3.80% vs Minimal 2.45% |
| 全局待办 / 进度总览 | 状态快照 | 原则：先测对→学得到→再扩规模；**以本文「当前结论速览」与 R10–R13 为准**（旧「进度总览」止于 R5，已过时） |

---

*精简原则：保留 Setting/分支/输出根/决策数字；合并重复 H100 节与 R9+P0；删除长篇必答复述、产物树枚举与过程性 debug 叙述。需要原文细节时查 `result-record.md.bak-20260810`。*

---

## H100-3 Fresh Controller Confirmation（2026-08-10，DONE）

**Setting**

- 目标：按 `H100-3-0810-todo2.md` 验证 HotpotQA controller effect 是否能在 fresh split 上复现；本轮先去掉 forced-readout，只比较 native finalization。
- Split：从 7405q 中按 deterministic hash/seed=42 冻结 `fresh200`，与既有 `unbiased200` query-disjoint；manifest 位于 `outputs/h100_3_hotpotqa_fresh_controller_confirmation/manifests/fresh200.json`。
- Decode/runtime：`temperature=0`，`seed=42`，`max_model_len=32768`，`max_turns=35`，`max_tokens=2048`，HotpotQA local context/retrieval 不变。
- Models：Qwen3-1.7B 与 Qwen3-30B-A3B-Instruct-2507。
- Conditions：`full-controller + native` vs `minimal-controller + native`；无 forced-readout、无 training、无 retrieval/index 改动。
- Scoring：使用 `scripts/h100_3_hotpotqa_readout_contract.py` 中的 `evaluate_hotpotqa_answer` 对 persisted `terminal_action_text` 计 answer accuracy；不以 recall/context coverage 替代最终答案准确率。
- 运行与监控：30B 初始两路 4-GPU 并行启动后出现 post-load API 不 ready，已停止 orphaned vLLM workers，并改用单路 8-GPU TP=8 顺序续跑 30B full/minimal；1.7B 用 GPU0/GPU1 双路并行完成。所有 run 均 200/200、errors=0。
- Output：`outputs/h100_3_hotpotqa_fresh_controller_confirmation/FRESH_CONTROLLER_CONFIRMATION.md` 与 `.json`，并生成 `RUN_MANIFEST.json` / `SHA256SUMS`。

**Results**

| model | controller | answer acc | paired W/L/T（minimal vs full） | mean turns | max-turn rate | tool calls | mean n_pool | mean n_curated | errors |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1p7b | full | 0.310 (62/200) | — | 22.820 | 0.245 | 22.820 | 271.565 | 23.855 | 0 |
| 1p7b | minimal | 0.250 (50/200) | 32/44/124 | 11.970 | 0.095 | 11.970 | 132.625 | 15.540 | 0 |
| 30b | full | 0.800 (160/200) | — | 31.745 | 0.800 | 31.745 | 106.665 | 14.760 | 0 |
| 30b | minimal | 0.955 (191/200) | 37/6/157 | 33.785 | 0.945 | 33.785 | 116.570 | 9.010 | 0 |

**Interaction / scale effect**

- 1.7B controller effect（minimal-full acc）：`-0.060`。
- 30B controller effect（minimal-full acc）：`+0.155`。
- Scale effect under full controller（30B-1.7B acc）：`+0.490`。
- Scale effect under minimal controller（30B-1.7B acc）：`+0.705`。
- Diff-in-diff interaction：`+0.215`。

**Conclusions**

- 30B 上 minimal-controller 的 answer accuracy 在 fresh200 上明确复现优于 full-controller：0.955 vs 0.800，paired W/L/T=37/6/157。
- 但 30B 的 turns/tool-calls 未与 accuracy 同方向改善：full→minimal 为 31.745→33.785，max-turn rate 0.800→0.945，因此不满足 todo 中“30B minimal 稳定优于 full，且 paired W/L/T、turns/tool-calls 同方向”这一 full7405 扩跑 gate。
- 1.7B 上 minimal-controller 不复现优势：0.250 vs full 0.310，paired W/L/T=32/44/124；它虽然降低 turns/tool-calls，但 accuracy 下降。

---

## H100-1 Capability Opportunity × Utility Audit（2026-08-10，DONE）

**Setting**

- 指令：`H100-1-0810-todo2.md` · 输出：`outputs/h100_1_capability_opportunity_utility/`。
- 目标：停止 query-conditioned selection 线，转为 `module × event support × closed-loop utility` 审计；不训练，不继续 F0/F1/F2 selection 扫描。
- 基线：Qwen3-1.7B · BrowseComp+ · `modules_full_v2.yaml` · `temperature=0` · `top_p=1` · `seed=42` · `max_model_len=32768`。
- 事件支持统计来源：0807 full830 harness trajectories：`outputs/h100_1_0807_full830_qwen3_1p7b_browsecomp_deterministic/harness/harness_rollouts.jsonl`。
- Utility：同一 frozen fresh200，对 `full`、`minus_context_budget`、`minus_evidence_state`、`minus_verification`、`minus_retrieval_rerank` 各 200q deterministic rollout；5/5 条件均完成，真实错误 0。

**Event support**

| module | event status | decisions | triggered queries | minority action | 判定 |
|---|---|---:|---:|---:|---|
| retrieval_rerank | MEASURABLE | 16759 | 830 | 6524 | 事件支持充分 |
| context_budget | MEASURABLE | 22020 | 830 | 830 | 事件支持充分 |
| evidence_state | ONE_SIDED | 3869 | 824 | 1 | 真实触发但几乎单侧 |
| verification | ONE_SIDED | 71 | 54 | 0 | 触发少且单侧 |
| recovery | NO_EVENT | 0 | 0 | 0 | BrowseComp harness 中未启用 |

**Fresh200 module-off utility**

| module | condition | utility | task Δ full-off | W/L/T（full vs off） | traj Δ | reward Δ | curated Δ |
|---|---|---|---:|---|---:|---:|---:|
| context_budget | minus_context_budget | TASK_NEUTRAL | -0.0114 | 15/13/172 | +0.0262 | -0.0154 | -0.75 |
| evidence_state | minus_evidence_state | TASK_NEUTRAL | -0.0089 | 19/16/165 | +0.0238 | +0.0132 | -1.30 |
| verification | minus_verification | UTILITY_POSITIVE | +0.0051 | 23/14/163 | +0.0242 | +0.0311 | -2.33 |
| retrieval_rerank | minus_retrieval_rerank | BEHAVIOR_ONLY | -0.0046 | 20/16/164 | +0.0169 | +0.0144 | -2.32 |

**H20 candidate priority / conclusion**

- `retrieval_rerank`：MEASURABLE + BEHAVIOR_ONLY → **Priority B**，可映射到 H20 `external_verification_routing/retrieval_routing`；有行为效用但 BrowseComp+ end-task metric 不敏感。
- `verification`：fresh200 utility 为 positive，但自然事件支持 ONE_SIDED / low-count，因此 **Do not train from this natural dataset**；若 H20 要做需 targeted event enrichment。
- `context_budget`：MEASURABLE 但 TASK_NEUTRAL，本轮不送 H20 internalization。
- `evidence_state`：ONE_SIDED + TASK_NEUTRAL，本轮不送 H20 internalization。
- `recovery`：NO_EVENT，本轮不训练。
- Final flags：`STOP_QUERY_CONDITIONED_SELECTION_LINE=true`，`DO_NOT_EXPAND_F2_FULL830=true`；本轮不扩 full830 module confirmation。

---

## H100-2 BrowseComp+ Matched Module Utility Ablation（2026-08-10，DONE · STOP_ALL_MODULE_ROUTES）

**Setting**

- 指令：`H100-2-0810-todo2.md`
- 输出根：`outputs/h100_2_module_utility/`
- 报告：`MODULE_UTILITY_REPORT.md/json`
- Manifest：`outputs/h100_2_module_utility/manifests/frozen_fresh200_queries.json`，fresh200，与 finalization100 query-disjoint
- 模型/解码：Qwen3-30B-A3B-Instruct-2507 · `temperature=0` · `top_p=1` · `max_model_len=32768` · `max_turns=35`
- 条件：`full`、`minus_context_budget`、`minus_evidence_state`、`minus_verification`、`minus_retrieval_rerank`
- 运行说明：native vLLM tool-call server 不可用，2-query smoke 后改用 text-tool JSON adapter；5 个 fresh200 条件均 `errors=0`

**Fresh200 条件结果**

| condition | n | canonical acc | traj recall | final recall | final-answer recall | turns | curated | pool | verify count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 200 | 0.0000 | 0.1877 | 0.0506 | 0.0519 | 23.75 | 23.61 | 197.09 | 0.00 |
| minus_context_budget | 200 | 0.0000 | 0.1884 | 0.0474 | 0.0486 | 24.51 | 22.49 | 198.89 | 0.00 |
| minus_evidence_state | 200 | 0.0000 | 0.1856 | 0.0533 | 0.0537 | 23.93 | 23.01 | 190.62 | 0.00 |
| minus_verification | 200 | 0.0000 | 0.1709 | 0.0472 | 0.0414 | 23.42 | 23.84 | 192.06 | 0.00 |
| minus_retrieval_rerank | 200 | 0.0000 | 0.1949 | 0.0590 | 0.0575 | 24.92 | 22.91 | 196.18 | 0.00 |

**Paired module decision**

| module | ablated condition | full vs ablated W/L/T | Δacc | event support | decision |
|---|---|---:|---:|---:|---|
| context_budget | minus_context_budget | 0/0/200 | 0.0000 | true | `STOP_MODULE_ROUTE` |
| evidence_state | minus_evidence_state | 0/0/200 | 0.0000 | true | `STOP_MODULE_ROUTE` |
| verification | minus_verification | 0/0/200 | 0.0000 | false | `STOP_MODULE_ROUTE` |
| retrieval_rerank | minus_retrieval_rerank | 0/0/200 | 0.0000 | false | `STOP_MODULE_ROUTE` |

**判定 / 结论**

- Required gate：只有“关闭模块后 canonical accuracy 稳定下降（full wins > losses 且 Δacc>0）并且真实 trigger/intervention support 非零”的模块才标记 `UTILITY_POSITIVE`。
- 本轮 4 个目标模块均未出现 canonical accuracy paired drop；`verification` 与 `retrieval_rerank` 还缺少 event support。
- `UTILITY_POSITIVE_MODULES=[]`；按 todo 规则 **不启动 full830 paired expansion**。
- 结论：`STOP_ALL_MODULE_ROUTES=true`；不继续扩多模块组合，不做训练，不继续 finalizer/compression prompt tournament。
