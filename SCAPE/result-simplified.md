# SCAPE Result Simplified

## 2026-08-21 adaptive_rerank_instruction Teacher-always-on vs Student-always-off 128-state gain

Status: **COMPLETED; always-on Teacher changes the first action on 10.9375% of states, lowers mean tool cost, and has a positive mean Utility delta at K4/K8.**

The experiment reused the same frozen cohort as the prior adaptive-rerank fork: seeds `2214/2215/2216/2217`, 32 states per seed and 128 paired states per horizon. The source shards did not retain complete snapshots, so reconstruction advanced each source trajectory with its frozen Student actions and required all target snapshot hashes to match before branching. All eight cells achieved `32/32` exact snapshot-hash matches. Teacher used an explicit `adaptive_rerank_instruction=True` rendering mask at every decision; Student used an explicit `False` mask at every decision. The first action counts toward K, so each branch executes exactly 4 or 8 actions.

```text
Horizon   n=128   first-action disagreement   tool-cost Delta (T-S)   Utility Delta (T-S)
K4        128     10.9375%                    -0.0859375              +0.0012890625
K8        128     10.9375%                    -0.2343750              +0.0035156250
```

Audit passed: K4/K8 ordered frozen identities match for every seed; frozen reconstruction mismatch, T/S initial-state mismatch, Teacher-mask failure, Student-mask failure, horizon action-count failure and Full Harness takeover are all `0`. The prior once-only zero result used the generic `full_mask()`, whose taxonomy default leaves this component disabled; it therefore does not establish the effect of a genuinely enabled Teacher. The always-on runner bypasses that ambiguity with explicit per-step ON/OFF masks.

Artifacts: `SCAPE/outputs/0821_adaptive_rerank_instruction_always_on_off_128/ADAPTIVE_RERANK_ALWAYS_ON_OFF_SUMMARY.json` (SHA-256 `0561cae05581feb1deb513795c7bd055d4e4617d4dad816243f6076864290dbc`), `ADAPTIVE_RERANK_ALWAYS_ON_OFF_PER_STATE.jsonl` (SHA-256 `aafa58c530c15921c8467829c8ffb94eb292bb41aff50b6eee36dc83e09fa3f4`), raw shards, audits and `SHA256SUMS`. Runner: `SCAPE/scripts/run_adaptive_rerank_always_on_off_128.py`; runtime `/opt/scape-h1004`.

## 2026-08-21 adaptive_rerank_instruction OPD four-cell 384-query evaluation

Status: **COMPLETED; strict Harness/Lucene evaluation**.

```text
pool: SCAPE-EasyOPD/manifests/browsecomp_plus_eval_pool_384/query_manifest.json
pool: 384 unique official queries; training query-ID overlap=0; official test subset=76
manifest SHA-256: 21f3cbbf9d7263df91f2d03150af862959ed0583a3a7071a0fc6498190d5b6ed
base: Qwen3-30B-A3B-Instruct-2507
runtime: /opt/scape-easyopd-smoke7; official Pyserini Lucene BM25; Java 21
Teacher: adaptive rerank instruction exposed in prompt
Student Before/After: reduced prompt, no inference privilege; After uses actual LoRA adapters from h100_2/formal_hf_adaptive_8gpu
```

The four settings each completed `384/384` generated rows, with identical ordered query IDs and strict Harness action validation. Official-test metrics (`n=76`) are:

```text
setting                  Legal action rate   Test Evidence Recall@5
Teacher                  92.1053%             2.7449%
Student before OPD       89.4737%             3.6330%
Student after pure OPD   97.3684%             2.2259%
Student after RL+OPD     98.6842%             5.2768%
```

Formal summary: `SCAPE-EasyOPD/outputs/0821_adaptive_rerank_opd_384/scored/SUMMARY.json`; raw generation shards are under `.../shards/`; scorer uses strict Harness schema and official Lucene BM25, reporting only Legal action rate and Evidence Recall@5 as required. `manifest_sha256` in the scorer is the generated canonical manifest artifact; source frozen pool manifest SHA-256 is recorded above.

Interpretation: RL+OPD has the highest Legal action rate and Test Evidence Recall@5 among the four settings. PURE_OPD improves legality over Teacher/Before but has lower test evidence recall; Before exceeds Teacher on recall despite lower legality.

## 2026-08-21 subtractive_curation Teacher-always-on vs Student-always-off 128-state gain

Status: **COMPLETED; first-action disagreement is unchanged from the once protocol; K4 is utility-neutral and K8 has a small positive utility delta with lower Teacher tool cost.**

The experiment reused the exact frozen 128 snapshots from `SCAPE/outputs/0820_subtractive_curation_recall_128_final/manifests/SUBTRACTIVE_STATES_128.jsonl` (manifest SHA-256 `bd55393181aed97448858426e50d58d5368e6ff35e977bd4816a47b5e796dd6e`). The first action counted toward K, so each branch executed exactly 4 or 8 actions. Teacher used the Full view with `subtractive_curation` enabled at every decision; Student used the Reduced view with the component disabled at every decision. `full_harness_takeover=0`.

```text
Horizon   n=128   first-action disagreement   tool-cost Δ (T-S)   Utility Δ (T-S)
K4        128     24.2188%                    +0.0000000          +0.0000000
K8        128     24.2188%                    -0.1171875          +0.0017578 (+0.1758% display)
```

Audit passed: 128 rows per horizon; cross-horizon ordered snapshot identity `128/128`; every branch has exactly K actions; Teacher traces use Full view on all steps and Student traces use Reduced view on all steps; missing/empty qrel `0`; Full Harness takeover `0`. Artifact: `SCAPE/outputs/0821_subtractive_curation_always_on_off_128/SUBTRACTIVE_ALWAYS_ON_OFF_SUMMARY.json` with raw K4/K8 JSONL. Runner: `SCAPE/scripts/run_subtractive_curation_always_on_off_128.py`; runtime `/opt/scape-easyopd-smoke7`.

## 2026-08-21 auto_populate_first_search Teacher-always-on vs Student-always-off 128-state gain

Status: **COMPLETED; always-on Teacher reduces tool cost and has positive utility delta at K4/K8.**

The experiment reused the frozen `NATURAL_FIRST_SEARCH_seed2230` 128-state cohort. The first action counts toward K; Teacher used the Full view with `auto_populate_first_search` enabled at every decision, while Student used the Reduced view with the component disabled at every decision.

```text
Horizon   n=128   first-action disagreement   tool-cost Δ (T-S)   Utility Δ (T-S)
K4        128     100.00%                     -1.8046875          +0.0270703125
K8        128     100.00%                     -2.9609375          +0.0444140625
```

Audit passed: K4/K8 ordered snapshot hashes are identical; both shards contain 128 rows; every branch has exactly 4/8 actions; all Teacher continuation decisions use Full view and all Student continuation decisions use Reduced view. Formal summary: `SCAPE/outputs/0821_auto_populate_always_on_off_128/AUTO_ALWAYS_ON_OFF_METRICS_SUMMARY.json`; runner/scorer: `SCAPE/scripts/run_auto_populate_always_on_off_128.py` and `score_auto_populate_always_on_off_128.py`; runtime `/opt/scape-projected-action`.

## 2026-08-21 importance_tagging+subtractive_curation Teacher-always-on vs Student-always-off 128-state gain

Status: **COMPLETED; always-on Teacher has higher tool cost and negative utility delta at both K4 and K8 on the unified frozen cohort.**

The experiment reused the exact 128 frozen snapshots from `outputs/0820_subtractive_curation_recall_128_final/manifests/SUBTRACTIVE_STATES_128.jsonl`. K4 and K8 used the same ordered snapshots (`128/128` identity); the first action counted toward K, so each branch executed exactly 4 or 8 actions. Teacher used the Full view with both `importance_tagging` and `subtractive_curation` enabled at every decision. Student used the Reduced view with both components disabled at every decision. `full_harness_takeover=0` for all rows.

```text
Horizon   n=128   first-action disagreement   tool-cost Δ (T-S)   Utility Δ (T-S)
K4        128     35.9375%                    +0.296875           -0.004453125 (-0.4453% display)
K8        128     35.9375%                    +0.500000           -0.007500000 (-0.7500% display)
```

Audit passed: 256 total rows (`128` per horizon), unique frozen snapshot hashes `128/128` per horizon, cross-horizon ordered snapshot identity `128/128`, branch trace action-count failures `0`, and Full Harness takeover `0`. The first-action disagreement rate compares the complete action (tool name and arguments), while cost and utility are direct branch-level Teacher minus Student means using the fixed live-fork utility formula.

Artifacts: `SCAPE/outputs/0821_joint_importance_subtractive_always_on_128/JOINT_ALWAYS_ON_SUMMARY.json` and `JOINT_ALWAYS_ON_PER_STATE.jsonl`. Runner: `SCAPE/scripts/run_joint_importance_subtractive_always_on_128.py`; runtime: `/opt/scape-venv`.

## 2026-08-21 sentence_compress Teacher-always-on vs Student-always-off 128-state gain

Status: **COMPLETED; always-on Teacher reduces tool cost and has positive utility delta at K4/K8 on the frozen cohort.**

The experiment reused the exact 128 frozen sentence_compress states from `outputs/0820_sentence_compress_formal_fork_k128_frozen_pool1024/manifests/sentence_compress_states_n128_seed2214.jsonl` (SHA-256 `b77fe21564f72555a6eaf49983b3c2f17af81066f34d557691f2c3a1baef47ee`). The first action counts in K, so each branch executes exactly 4 or 8 actions. Teacher uses the Full view with sentence_compress enabled at every decision; Student uses the Reduced view with the component disabled at every decision. This is distinct from the previous once-only protocol, and `full_harness_takeover=0` remains an explicit invariant.

```text
K4 n=128: first-action disagreement=100.00%; tool-cost delta=-0.2578125 (-25.78% display); Utility delta=+0.0038671875 (+0.3867% display)
K8 n=128: first-action disagreement=100.00%; tool-cost delta=-0.4843750 (-48.44% display); Utility delta=+0.0072656250 (+0.7266% display)
```

Audit passed: K4/K8 ordered snapshot identity `128/128`; frozen snapshot hash mismatches `0`; Teacher-view failures `0`; Student-view failures `0`; horizon action-count failures `0`; branch-level cost/utility recomputation mismatches `0`; Full Harness takeover `0`. A CUDA allocator warning occurred once during K8, but the process continued to `128/128`, exited 0, and every row passed the final deterministic audit.

Artifacts: `SCAPE/outputs/0821_sentence_compress_always_on_off_128/SENTENCE_COMPRESS_ALWAYS_ON_OFF_SUMMARY.json`, `SENTENCE_COMPRESS_ALWAYS_ON_OFF_PER_STATE.jsonl`, raw K4/K8 JSONL, and `SHA256SUMS`. Runner: `SCAPE/scripts/run_sentence_compress_always_on_off_128.py`; runtime: `/opt/scape-sentence-compress-venv`.


## 2026-08-21 content_dedup Teacher-always-on vs Student-always-off 128-state gain

Status: **COMPLETED; strict full-horizon component-mask fork, with small negative utility and higher Teacher tool cost.**

The run reused the same frozen 128-state content_dedup cohort from `outputs/0820_content_dedup_real_recall_128/` (64 states per seed `2214/2215`, 128 states per horizon). The first action counted toward K, so each branch executed exactly 4 or 8 actions. Teacher kept `content_dedup` enabled at every decision; Student kept it disabled at every decision. Both branches used the Reduced continuation policy, and `full_harness_takeover=0`.

```text
Horizon   n=128   first-action disagreement   tool-cost Δ (T-S)   Utility Δ (T-S)
K4        128     100.00%                    +0.2265625          -0.0033984375 (-0.3398% display)
K8        128     100.00%                    +0.3437500          -0.0051562500 (-0.5156% display)
```

Audit passed: 256 total rows, 128 per horizon, source manifest SHA-256 `19455667add65bc06f189f4c8ee8d21ae48150ccd7cfbd4cca05d883a1012fcd`, snapshot mismatch `0`, Full Harness takeover `0`, and exact-K action semantics. Artifact: `SCAPE/outputs/0821_content_dedup_always_on_off_live_128/CONTENT_DEDUP_ALWAYS_ON_OFF_SUMMARY.json` with per-state traces in `CONTENT_DEDUP_ALWAYS_ON_OFF_PER_STATE.jsonl`. Runner: `SCAPE/scripts/run_content_dedup_always_on_off_live_128.py`; runtime `/opt/scape-easyopd-smoke7`.

Interpretation: under the strict always-on/off policy-visible fork, Teacher changed the first action in every paired state, but incurred higher mean tool cost and a small negative utility delta at both horizons. These are process/utility metrics only and do not replace the separate content_dedup qrel recall gate.

## 2026-08-21 token_budget_marker OPD four-cell strict 384-query evaluation

Status: **COMPLETED; strict Harness-schema scoring finished on the frozen 384-query pool.**

- Pool: 384 unique official BrowseComp-Plus queries, training query-ID overlap=0; official test subset=76; manifest SHA-256=`daa46743ef9b1d6acf1dd230e8da92761f3465d47f2a8d4f7981f3ff7c380092`.
- Existing immutable model-action shards were rescored without overwriting them. Legal actions require strict Harness contracts; valid fan-out queries use rank-wise round-robin top-5 fusion over official Pyserini Lucene. Recall@100/1000 were not computed.
- Formal artifact: `SCAPE-EasyOPD/outputs/0821_token_budget_marker_opd_384/r5_strict_final_20260821_v2/`; runtime `/opt/scape-venv` with system Java 21.

```text
setting                  Legal action rate   test Evidence Recall@5
Teacher                       21.0526%                 0.6579%
Student before OPD             9.2105%                 0.4386%
Student after pure OPD         9.2105%                 0.2193%
Student after RL+OPD           7.8947%                 0.2193%
```

The four official-test values are computed over exactly 76 queries. Relative to Student Before OPD, PURE_OPD is `+0.00 pp` Legal / `-0.2193 pp` Recall@5, and RL+OPD is `-1.3158 pp` Legal / `-0.2193 pp` Recall@5.


## 2026-08-21 content_dedup OPD four-cell 384-query test evaluation

Status: **COMPLETED; ALL FOUR SETTINGS RECALL-NEUTRAL UNDER THE FROZEN 384-QUERY TEST CONTRACT**.

```text
pool: SCAPE-EasyOPD/manifests/browsecomp_plus_eval_pool_384/query_manifest.json
pool: 384 unique official queries; training query-ID overlap=0
manifest SHA-256: 21f3cbbf9d7263df91f2d03150af862959ed0583a3a7071a0fc6498190d5b6ed
base: Qwen3-30B-A3B-Instruct-2507 staged read-only at /dev/shm
runtime: /opt/scape-easyopd-smoke7; max_steps=6; student_inference_privilege=false
Teacher: V8D_CONTENT_DEDUP=1
Student Before: V8D_CONTENT_DEDUP=0, base/no adapter
Student After: V8D_CONTENT_DEDUP=0, actual LoRA adapters; PURE_OPD and RL_PLUS_OPD seeds 42/43
```

All six underlying cells completed `384/384`, `error_rate=0`, and every recorded tool call was legal. The requested four-setting aggregation is:

```text
setting                  Legal action rate   test trajectory recall
Teacher                  100.00%             0.1157407%
Student before OPD       100.00%             0.1157407%
Student after pure OPD   100.00%             0.1157407%
Student after RL+OPD     100.00%             0.1157407%
```

PURE_OPD seed42/43 and RL_PLUS_OPD seed42/43 each independently produced the same values. Formal summary: `SCAPE-EasyOPD/outputs/0821_content_dedup_opd_384/CONTENT_DEDUP_OPD_384_TEST_SUMMARY.json`.

Interpretation: this real Harness-1 test run found no difference in Legal action rate or trajectory recall across the four settings. It therefore provides no test-recall evidence of content_dedup internalization by either OPD method. `trajectory_recall` is the runner's executed-state test recall field; the old frozen OPD-valid-row action-level diagnostic remains a separate metric and is not substituted here.

## 2026-08-21 adaptive_rerank_instruction 128-state cost/utility completion

Status: **COMPLETED; no model rerun required; cost and utility are both neutral**.

The formal recall fork already contained complete Teacher/Student branch metrics, so the adaptive_rerank_instruction component was scored offline rather than rerun. Source rows: `SCAPE/outputs/0820_adaptive_rerank_instruction_recall_128/shards/adaptive_rerank_instruction_seed{2214,2215,2216,2217}_K{4,8}.jsonl`; 4 seeds × 2 horizons × 32 rows = 256 paired states. The scorer directly used branch-level `tool_search_cost` and `objective_utility`, and independently re-evaluated the fixed live-fork utility formula:

```text
0.45 * evidence_coverage
+ 0.20 * useful_unique_docs / max(1, gold_count)
+ 0.20 * verified_supported_claims / max(1, gold_count)
- 0.05 * redundancy
- 0.015 * tool_search_cost
- 0.03 * unsupported_claims
```

K4 Teacher/Student tool cost means are equal, giving cost Δ `+0.0000`; utility means are equal, giving utility Δ `+0.000000`. K8 is identical: cost Δ `+0.0000`, utility Δ `+0.000000`. Both horizons have 128/128 zero paired deltas, positive/negative/zero=`0/0/128`, and formula/endpoint-cost mismatch count `0`. The formal runner executes exactly K actions including the forced first action, uses Reduced continuation for both branches, and has `full_harness_takeover=0`.

Artifacts: `SCAPE/outputs/0820_adaptive_rerank_instruction_recall_128/scored_reference_metrics/ADAPTIVE_RERANK_COST_UTILITY_SUMMARY.json` (SHA-256 `d1707ae5d825d880bbc1d5a477bac4f64f46b4e75327cb5cdf7e99526a78ee3f`) and `ADAPTIVE_RERANK_COST_UTILITY_PER_STATE.jsonl` (SHA-256 `d370aebda8813adfd09802693e6c8669d2b98b95d0f062555db1cbc8022cab72`).

Interpretation: adaptive_rerank_instruction remains recall-neutral and shows no measurable cost or utility separation on this frozen cohort; the gain table now records `+0.0000` and `+0.000000`, replacing the previous N/A values.

## 2026-08-21 sentence_compress OPD 384-query rerun

Status: **COMPLETED; official Lucene BM25 rescoring passed for all six seed-level traces (384/384 each)**. The strict pool was reconstructed from official BrowseComp-Plus queries/evidence-qrels/gold-qrels minus component-training query IDs. Qwen3-30B base and existing PURE/RL+OPD adapters were used; action parsing accepts the Harness compact `tool/args`, `tool_input`, and equivalent forms. Pyserini Lucene was run against the official BM25 index using a private JRE21 unpacked under `/opt/jdk21`; no `/mnt` environment was modified.

```text
Setting                         Legal action rate       Test Evidence Recall@5
Teacher                         100.00%                 0.2195%
Student Before OPD              100.00%                 1.5436%
Student After PURE OPD          56.38%                  0.7718%
Student After RL+OPD            12.76%                  0.0000%
```

PURE seed42=`100.00% / 1.5436%`; PURE seed43=`12.76% / 0.0000%`. RL+OPD seed42/43=`12.76% / 0.0000%`. Each After method aggregates two seeds × 384 query rows. Formal artifact: `SCAPE-EasyOPD/outputs/sentence_compress_opd_384_20260821/FINAL_4CELL_SUMMARY.json`; per-query ordered BM25 traces are in each setting's `PER_QUERY.jsonl`; `SHA256SUMS` is in the same directory. Conclusion: neither After method improves Legal action rate or Test Evidence Recall@5 over Student Before; PURE is highly seed-unstable and RL+OPD collapses to zero recall.

## 2026-08-21 paired 128-state gain reference-metric audit

Status: **COMPLETED; offline diagnostic metrics extracted without changing formal recall gates**.

Using the detailed per-state traces from the 0820 recall forks, `scripts/explore_gain_reference_metrics_128.py` wrote `outputs/0821_gain_reference_metrics_128/GAIN_REFERENCE_METRICS_PER_STATE.jsonl` and `GAIN_REFERENCE_METRICS_SUMMARY.json`. The audit adds three interpretable layers alongside endpoint qrel recall: first-action disagreement, successful-read set delta, and utility/cost deltas when raw branch fields are available.

Key pooled K4/K8 signals include: `auto_populate_first_search` first-action disagreement `100%`, utility delta `+0.010664/+0.010313`, and tool-cost delta `-0.7109375/-0.6875`; `subtractive_curation` first-action disagreement `24.22%`, successful-read set delta `+0.015625`, and utility delta `-0.000234/-0.000234`; `importance_tagging` successful-read set delta `+0.0859375/+0.078125`; `content_dedup` `+0.0390625/+0.0625`; `sentence_compress` `+0.0546875/+0.0390625`; and joint `importance_tagging+subtractive_curation` `+0.0963855/+0.0602410` with utility delta `-0.004518/-0.007410`.

Interpretation: endpoint candidate/activated recall remains a qrel coverage safety gate, while the new process and efficiency signals explain recall-neutral but behaviorally distinct components. Compact artifacts that lack complete branch provenance are marked `N/A`; no unavailable cost or utility field is inferred. Token-budget-marker remains invalidated for insufficient real budget pressure.

## 2026-08-21 sentence_compress 128-state gain reference metrics rerun

Status: **COMPLETED; diagnostic process/efficiency metrics recorded; formal usable-evidence recall remains neutral**.

Setting: frozen sentence_compress 128-state cohort from `outputs/0820_sentence_compress_formal_fork_k128_frozen_pool1024/manifests/sentence_compress_states_n128_seed2214.jsonl`; Teacher/Full enabled `sentence_compress` only for the first action, Student/Reduced disabled it, both continuations used Reduced policy, and `full_harness_takeover=0`. The approved environment was `/opt/scape-projected-action`; model `/mnt/songzijun/models/pat-jj_harness-1-full/harness-1`. The complete source artifact is `outputs/0820_sentence_compress_formal_fork_k128_frozen_pool1024/SENTENCE_COMPRESS_VALUE_PER_STATE.jsonl` (128 rows per K).

Fresh rerun completed with 128/128 rows for both K4 and K8 in `outputs/0821_sentence_compress_reference_rerun_128/shards/`. Seed-merged metrics (Teacher minus Student): K4 first-action disagreement `100.00%`, successful read set delta `+0.0859`, tool cost delta `-0.0625`, utility delta `+0.000938`; K8 first-action disagreement `100.00%`, successful read set delta `+0.0859`, tool cost delta `+0.0781`, utility delta `-0.001172`. The separate offline usable-evidence replay remains `1.1824%` Teacher/Student at both K4/K8 with paired gain `0.00 pp`; these process metrics must not be interpreted as qrel recall gain.

## 2026-08-20 subtractive_curation formal 128-state candidate/activated evidence recall rerun

Status: **COMPLETED; BOTH PAIRED RECALL GAINS ARE 0.00 PP**.

Setting:

```text
component: subtractive_curation
seed: 2214
horizons: K4, K8
states: frozen 128-state master cohort; 128 unique snapshots, 66 queries; ordered K4/K8 snapshot match 128/128
eligibility: endpoint candidate documents and curated IDs are present in every frozen snapshot
Teacher/Full: subtractive_curation enabled for first fork action only
Student/Reduced: subtractive_curation disabled for first fork action
continuation: Reduced policy for both branches; full_harness_takeover=0
normalization: split_at_first_underscore_v1
qrel: BrowseComp-Plus topics-qrels/qrel_evidence.txt, SHA-256 a6f594975be57339de9e4e9f67f13c044f647feda77c0b84c45a1581e3041bd1
runner/scorer: run_subtractive_curation_recall_128.py / score_subtractive_curation_recall_128.py
environment: /opt/scape-easyopd-smoke7
```

Seed-merged paired results:

```text
K4 candidate_evidence_pool_recall gain: 0.00 pp, paired/bootstrap CI95 [0.00, 0.00], pos/neg/zero=0/0/128
K4 activated_evidence_recall gain:      0.00 pp, paired/bootstrap CI95 [0.00, 0.00], pos/neg/zero=0/0/128
K8 candidate_evidence_pool_recall gain: 0.00 pp, paired/bootstrap CI95 [0.00, 0.00], pos/neg/zero=0/0/128
K8 activated_evidence_recall gain:      0.00 pp, paired/bootstrap CI95 [0.00, 0.00], pos/neg/zero=0/0/128
query-cluster bootstrap CI95: [0.00, 0.00] pp for both metrics and horizons
Teacher/Student absolute means: candidate 2.2656%/2.2656%; activated 0.3906%/0.3906% at K4 and K8
```

Audit and interpretation:

- Every branch stores endpoint candidate IDs, curated IDs, read attempts, successful read observations, context entry/retention IDs and the activated union. The scorer verified `final_activated = final_curated union retained_reads` for all 512 branch endpoints.
- `invalid_provenance=0`, `snapshot_mismatch=0`, `ordered_snapshot_match=128`, `full_harness_takeover=0`, `missing_or_empty_qrel_count=0`.
- Candidate precision T/S is `0.7812%/0.7812%`, pool size `10.0/10.0`; activated precision T/S is `0.3906%/0.3906%`, activated size `2.0/2.0` at both horizons.
- Mean successful reads T/S are K4 `0.7188/0.7031` and K8 `0.7656/0.7500`; successful reads are append-only retained at endpoint. Tool-cost delta is `+0.015625` at both horizons; weighted utility delta is `-0.000234375` (`-0.0234%`) at both horizons.
- The formal recall gate fails: subtractive_curation produces no candidate-pool or activated-evidence recall gain. These endpoint results supersede the old reward-only artifact for the two columns in `/mnt/songzijun/增益.md`.

Artifacts: `SCAPE/outputs/0820_subtractive_curation_recall_128_final/`; summary `SUBTRACTIVE_CANDIDATE_ACTIVATED_RECALL_GATE.json`; per-state scorer `SUBTRACTIVE_CANDIDATE_ACTIVATED_RECALL_PER_STATE.jsonl`.

## 2026-08-20 evidence_graph formal 128-state candidate/activated evidence recall rerun

Status: **COMPLETED; BOTH PAIRED RECALL GAINS ARE 0.00 PP**.

Setting:

```text
component: evidence_graph
seed: 2214
horizons: K4, K8
states: same frozen 128-state cohort at each horizon; ordered snapshot match 128/128
Teacher/Full: evidence_graph enabled for first fork action only
Student/Reduced: evidence_graph disabled for first fork action
continuation: Reduced policy for both branches; full_harness_takeover=0
normalization: split_at_first_underscore_v1
qrel: BrowseComp-Plus topics-qrels/qrel_evidence.txt, SHA-256 a6f594975be57339de9e4f67f13c044f647feda77c0b84c45a1581e3041bd1
runner: run_evidence_graph_recall_formal_fork.py; environment /opt/scape-h1004
```

Seed-merged paired results:

```text
K4 candidate_evidence_pool_recall gain: 0.00 pp, CI95 [0.00, 0.00], pos/neg/zero=0/0/128
K4 activated_evidence_recall gain:      0.00 pp, CI95 [0.00, 0.00], pos/neg/zero=0/0/128
K8 candidate_evidence_pool_recall gain: 0.00 pp, CI95 [0.00, 0.00], pos/neg/zero=0/0/128
K8 activated_evidence_recall gain:      0.00 pp, CI95 [0.00, 0.00], pos/neg/zero=0/0/128
Teacher/Student absolute means: K4 candidate 2.6538%/2.6538%, activated 0.6337%/0.6337%; K8 identical.
```

Audit and interpretation:

- Scored raw endpoint provenance, not the previous `usable_evidence_recall@K` replay artifact: candidate IDs, curated IDs, successful reads, context entry/retention and activated union were persisted and independently checked.
- `invalid_provenance=0`, `snapshot_mismatch=0`, `ordered_snapshot_match=128`, `full_harness_takeover=0`, `missing_or_empty_qrel_count=0`.
- Candidate precision T/S was `1.25%/1.25%`, candidate pool size `10.0/10.0`; activated precision T/S was `1.9531%/1.9531%`, activated size `2.0/2.0`.
- Successful reads were K4 `0.9453/0.2109` and K8 `0.9453/0.2109` (T/S); context retention was `1.0/1.0`. Utility deltas were `-2.8008%` at K4 and `-4.6523%` at K8, with tool-cost deltas `+1.8672` and `+3.1016`, respectively.
- Formal recall gate therefore failed: evidence_graph produced no candidate-pool or activated-evidence recall gain, while adding tool cost and reducing weighted utility. The old usable-evidence result remains historical and is not used for the table.
- The gain table was corrected from `N/A` using the existing raw branch-level metrics (no rerun required): `tool_cost_delta` is Teacher minus Student, K4 `+1.8671875` and K8 `+3.1015625`; `weighted_utility_delta` is K4 `-0.0280078125` and K8 `-0.0465234375` (table displays these as `+186.7188%/+310.1563%` and `-2.8008%/-4.6523%`). The raw shards preserve `branch_S_metrics`/`branch_T_metrics` including `tool_search_cost` and `objective_utility`, and the scorer independently computes these paired deltas.

Artifacts: `SCAPE/outputs/0820_evidence_graph_recall_formal_20260820/`; scorer summary `scored/EVIDENCE_GRAPH_EVIDENCE_RECALL_SUMMARY.json`; per-state scorer `scored/EVIDENCE_GRAPH_EVIDENCE_RECALL_PER_STATE.jsonl`; raw branch metrics `shards/evidence_graph_K4.jsonl` and `shards/evidence_graph_K8.jsonl`.

## 2026-08-20 importance_tagging + subtractive_curation fresh 128-state evidence-recall experiment

Status: **COMPLETED; BOTH PAIRED RECALL GAINS ARE 0.00 PP**.

Setting:

```text
component: importance_tagging + subtractive_curation, independent both-on vs both-off treatment
seeds: 8423, 8424
horizons: K4, K8
states: 128 fresh frozen states per seed; 256 unique states and 512 paired horizon rows total
Teacher/Full: both components enabled for the first fork action only
Student/Reduced: both components disabled for the first fork action
continuation: both branches use Reduced policy; full_harness_takeover=0
normalization: split_at_first_underscore_v1
qrel: BrowseComp-Plus topics-qrels/qrel_evidence.txt, SHA-256 a6f594975be57339de9e4e9f67f13c044f647feda77c0b84c45a1581e3041bd1
```

Seed-merged paired results:

```text
K4 candidate_evidence_pool_recall gain: 0.00 pp, CI95 [0.00, 0.00], pos/neg/zero=0/0/256
K4 activated_evidence_recall gain:      0.00 pp, CI95 [0.00, 0.00], pos/neg/zero=0/0/256
K8 candidate_evidence_pool_recall gain: 0.00 pp, CI95 [0.00, 0.00], pos/neg/zero=0/0/256
K8 activated_evidence_recall gain:      0.00 pp, CI95 [0.00, 0.00], pos/neg/zero=0/0/256
per-seed deltas: both metrics are 0.00 pp for seeds 8423 and 8424; seed sample std=0.00 pp
```

Audit and interpretation:

- Fresh rerun completed for two independently frozen cohorts: seed 8423 `128/128` and seed 8424 `128/128`; every state stores its complete initial snapshot and both K4/K8 checkpoints.
- Each state forks T/S from the same saved snapshot. K4 and K8 come from one K8 trajectory whose horizon does not enter the prompt or random stream; the first four actions are therefore identical by construction. Full Harness takeover is `0/512`.
- Every branch stores endpoint candidate IDs, final curated IDs, read attempts, successful read observations, IDs entering context, IDs retained at endpoint, and final activated IDs. The scorer verified `final_activated = final_curated union retained_reads` for every T/S checkpoint.
- Absolute candidate recall is Teacher/Student `1.7499% / 1.7499%` at both K4 and K8; mean candidate precision is `0.8594% / 0.8594%`, and mean pool size is `10.0 / 10.0`.
- Absolute activated recall is Teacher/Student `0.5549% / 0.5549%` at both K4 and K8; mean activated precision is `0.9766% / 0.9766%`, and mean activated-set size is `2.0 / 2.0`.
- Both seeds independently yield candidate and activated paired deltas of `0.00 pp`; seed sample std is `0.00 pp`. The fresh experiment therefore finds no recall gain from the combination.
- Historical weighted utility remains separate: K4 `-0.3574%`, K8 `-0.5566%`; the combination has neither recall gain nor utility gain.

Artifacts:

```text
SCAPE/outputs/0820_joint_importance_subtractive_recall_fresh_128/JOINT_CANDIDATE_ACTIVATED_EVIDENCE_RECALL_FRESH.json
SCAPE/outputs/0820_joint_importance_subtractive_recall_fresh_128/JOINT_FRESH_VALUE_PER_STATE.jsonl
SCAPE/outputs/0820_joint_importance_subtractive_recall_fresh_128/manifests/FRESH_COHORT_seed8423.json
SCAPE/outputs/0820_joint_importance_subtractive_recall_fresh_128/manifests/FRESH_COHORT_seed8424.json
```

## 2026-08-20 BrowseComp-Plus 500-query evaluation pool construction

Status: **CANDIDATE_REJECTED_TRAIN_OVERLAP; not a valid frozen evaluation pool**.

Setting:

```text
source: official BrowseComp-Plus topics-qrels/queries.tsv
qrels: qrel_evidence.txt + qrel_golds.txt
seed: 20260820
target: 500
stratification: proportional sampling over 4x4 evidence/gold document-count strata
```

The candidate artifact is under `SCAPE-EasyOPD/manifests/browsecomp_plus_eval_pool/`. It contains 500 unique official query IDs, exact `queries.tsv` text, non-empty evidence/gold qrel document IDs, source hashes, a run manifest, and SHA256SUMS. Independent replay passed all structural invariants. However, 278 selected query IDs overlap the existing SCAPE component training pool (normalized-text overlap is 0), so the candidate is explicitly rejected by the no-training-overlap gate. All 830 official queries are qrel-eligible; only 384 remain after strict ID exclusion against the 446 official IDs in the existing training pool, making a valid 500-row pool impossible under the current data policy. No retrieval leaderboard evaluation should consume this artifact until the training/evaluation query policy is changed and the pool is regenerated.


## 2026-08-20 auto_populate_first_search 128-state same-state K4/K8 reward fork

Status: **COMPLETED; VALUE_POSITIVE; RECALL METRIC STILL N/A**.

Setting:

```text
component: auto_populate_first_search only
strata: NATURAL_FIRST_SEARCH, AUTO_EFFECT_ACTIVE
seeds: 2230, 2231
horizons: K4, K8
states: 128 per (stratum, seed), 1024 rows total
state source: first 128 frozen states from each canonical 512-state manifest, then fresh live fork/replay
Teacher/Full: auto_populate_first_search enabled for first fork action
Student/Reduced: auto_populate_first_search disabled for first fork action
continuation: both branches use reduced policy; no full-harness takeover
model: /mnt/songzijun/models/pat-jj_harness-1-full/harness-1
python: /opt/scape-projected-action/bin/python (torch 2.10.0+cu128, transformers 5.14.1, pyserini package with local-corpus fallback)
```

Reward summary, Teacher - Student:

```text
NATURAL_FIRST_SEARCH K4: mean=+0.0106640625 (+1.0664%), seed std=0.0016572815, normal CI95=[+0.0077426,+0.0135856], positive/negative/zero=212/43/1
NATURAL_FIRST_SEARCH K8: mean=+0.0103125000 (+1.0313%), seed std=0.0023201941, normal CI95=[+0.0061652,+0.0144598], positive/negative/zero=206/46/4
AUTO_EFFECT_ACTIVE K4: mean=+0.0106640625 (+1.0664%), seed std=0.0016572815, normal CI95=[+0.0077426,+0.0135856], positive/negative/zero=212/43/1
AUTO_EFFECT_ACTIVE K8: mean=+0.0103125000 (+1.0313%), seed std=0.0023201941, normal CI95=[+0.0061652,+0.0144598], positive/negative/zero=206/46/4
```

Audit and interpretation:

- All 8 cells completed with `n=128`; total `1024` rows.
- Within every `(stratum, seed)`, K4/K8 manifests matched `128/128` `snapshot_hash` values exactly; cross-stratum overlap was `0/128` for both seeds.
- Every cell had replay-noise q95 `0.0`; formal gate decision is `VALUE_POSITIVE`.
- The reward fork is not a `usable_evidence_recall@K` result: endpoint `final_curated_ids`, final working-memory evidence IDs, and retained read IDs were not fully persisted. Keep recall as `N/A` until those fields are available; do not relabel reward as recall.

Artifacts:

```text
SCAPE/outputs/0820_auto_populate_first_search_value_confirm_128/AUTO_VALUE_CONFIRM/AUTO_VALUE_GATE.json
SCAPE/outputs/0820_auto_populate_first_search_value_confirm_128/AUTO_VALUE_CONFIRM/AUTO_VALUE_BY_STRATUM_SEED_K.csv
SCAPE/outputs/0820_auto_populate_first_search_value_confirm_128/AUTO_VALUE_CONFIRM/AUTO_VALUE_PER_STATE.jsonl
SCAPE/outputs/0820_auto_populate_first_search_value_confirm_128/AUTO_VALUE_CONFIRM/AUTO_VALUE_128_AUDIT.json
SCAPE/outputs/0820_auto_populate_first_search_value_confirm_128/RUN_MANIFEST.json
```

The earlier `AUTO_VALUE_CONFIRM512x2` result remains historical only; the component-gain table now reports this 128-state rerun.

## 2026-08-20 importance_tagging single-component 128-state utility rerun

Status: **COMPLETED; 128-STATE K4/K8 UTILITY GATE FAILED**.

Setting:

```text
component: importance_tagging only
states: 128 per seed/horizon; 2 seeds (8423, 8424); 512 rows total
Teacher/Full: importance_tagging ON for first fork action
Student/Reduced: importance_tagging OFF for first fork action
continuation: both branches use Reduced policy; no full-harness takeover
runner: true_live_fork_replay_hf_bm25_batched_stream
model: /mnt/songzijun/models/pat-jj_harness-1-full/harness-1
python_env: /opt/scape-projected-action
output: SCAPE/outputs/0820_importance_tagging_single_128_rerun/
```

Audit:

```text
K4/K8 ordered same-state pairing: 2/2 seeds passed
snapshot_hash matches: 256/256 paired rows
rows: 512/512; each seed/K cell: 128/128
full_harness_takeover: false for all rows
```

Teacher - Student weighted utility results:

```text
K4 merged n=256: mean=-0.00263671875 (-0.26%), CI95=[-0.0074381,+0.0021647], positive/negative/zero=102/120/34
K8 merged n=256: mean=-0.00298828125 (-0.30%), CI95=[-0.0104534,+0.0044769], positive/negative/zero=111/115/30
```

Per-seed means were K4 `-0.00046875` / `-0.0048046875` and K8 `+0.0012890625` / `-0.007265625` for seeds 8423/8424 respectively. Mean evidence-coverage and curated-evidence deltas were `0.0`; mean tool-search-cost deltas were K4 `+0.17578125` and K8 `+0.19921875`.

Conclusion: the corrected 128-state rerun does not provide a positive importance-tagging utility gain. Both aggregate CIs cross zero, so this is weaker/inconclusive than the prior 512-state negative estimate, but the component still fails the gate and must not be promoted to OPD. This result supersedes the old 512-state values in the gain table; the old artifact remains historical.

Reference metrics completion (2026-08-21): the existing `IMPORTANCE_128_VALUE_PER_STATE.jsonl` already contains paired `branch_S_metrics.tool_search_cost`, `branch_T_metrics.tool_search_cost`, and `branch_T_minus_S` for all 256 states per horizon, so no rerun was needed. Offline recomputation gives tool-cost delta K4 `+0.17578125`, K8 `+0.19921875`; utility delta K4 `-0.00263671875`, K8 `-0.00298828125`. These values were filled into `增益.md`; they are Teacher minus Student and share the same 128-state paired fork/audit.

Artifacts:

```text
SCAPE/outputs/0820_importance_tagging_single_128_rerun/IMPORTANCE_128_K4_K8_GATE.json
SCAPE/outputs/0820_importance_tagging_single_128_rerun/IMPORTANCE_128_K4_K8_GATE.md
SCAPE/outputs/0820_importance_tagging_single_128_rerun/IMPORTANCE_128_SUMMARY.csv
SCAPE/outputs/0820_importance_tagging_single_128_rerun/IMPORTANCE_128_VALUE_PER_STATE.jsonl
SCAPE/outputs/0820_importance_tagging_single_128_rerun/K{4,8}_seed{8423,8424}/shards/*.jsonl
```

## 2026-08-20 importance_tagging candidate/activated evidence recall 128-state formal replay

Status: **COMPLETED; ZERO PAIRED CANDIDATE-POOL AND ACTIVATED-EVIDENCE RECALL GAIN**.

Setting and audit:

```text
component: importance_tagging
states: 128 per seed/horizon; seeds 8423/8424; 256 paired rows per K
Teacher: importance_tagging ON for first action; Student: OFF; Reduced continuation for both
runner: true_live_fork_replay_hf_bm25_batched_stream with endpoint provenance fields
python_env: /opt/scape-projected-action
qrel SHA-256: a6f594975be57339de9e4f67f13c044f647feda77c0b84c45a1581e3041bd1
normalization: str(doc_id).split("_", 1)[0]
ordered snapshot matches: K4 128/128; K8 128/128
invalid provenance: 0; full_harness_takeover: 0
```

Results (seed-merged paired-row means):

```text
K4 candidate recall Teacher/Student=1.4431% / 1.4431%, gain=+0.00 pp
K4 activated recall Teacher/Student=0.3472% / 0.3472%, gain=+0.00 pp
K8 candidate recall Teacher/Student=1.4431% / 1.4431%, gain=+0.00 pp
K8 activated recall Teacher/Student=0.3472% / 0.3472%, gain=+0.00 pp
```

For both metrics and horizons, positive/negative/zero paired counts were `0/0/256`, paired bootstrap CI95 was `[0.00, 0.00] pp`, and both seeds independently had `+0.00 pp`. Candidate precision/set size were `0.8984% / 0.8984%` and `10 / 10`; activated precision/set size were `1.1719% / 1.1719%` and `2 / 2`. The component therefore has no recall gain under the formal paired fork.

Artifacts:

```text
SCAPE/outputs/0820_importance_tagging_recall_128/scored/IMPORTANCE_RECALL_K4_K8_GATE.json
SCAPE/outputs/0820_importance_tagging_recall_128/scored/IMPORTANCE_RECALL_PER_STATE.jsonl
SCAPE/outputs/0820_importance_tagging_recall_128/K{4,8}_seed{8423,8424}{,_formal}/shards/importance_tagging_K{4,8}.jsonl
```

## 2026-08-20 importance_tagging usable_evidence_recall@K deterministic trace audit

Status: **COMPLETED; ZERO PAIRED USABLE-EVIDENCE RECALL GAIN; ABSOLUTE RECALL N/A**.

Setting and proof:

```text
component: importance_tagging
formal source: SCAPE/outputs/0816_2_importance_proper_fork_formal_final/IMPORTANCE_PROPER_VALUE_PER_STATE.jsonl
rows: 1024 per K; K4/K8 each combine seeds 8423/8424
qrel: SCOPE/external/BrowseComp-Plus/topics-qrels/qrel_evidence.txt
document-ID normalization: str(doc_id).split("_", 1)[0]
endpoint U_K: final curated IDs union final working-memory evidence IDs union successful read_document IDs retained in context
```

The formal trace schema does not store complete endpoint IDs, but it does store branch action traces. Audit of all 1024 paired rows at each horizon found that every branch action is `read_document` or `end_search`; there are no branch-local search, curate, review, or verify transitions. In the runner, `read_document` only records a read ID and does not mutate curated IDs or working-memory documents. Therefore both branches retain the same frozen curated/working-memory endpoint state and `R_K` is a subset of shared `W_K`; hence `U^T_K = U^S_K` for every paired row. The paired delta is therefore exactly zero without inventing absolute endpoint IDs.

Results:

```text
K4 n=1024: paired delta=0.00 pp; paired CI95=[0.00,0.00] pp; positive/negative/zero=0/0/1024
K8 n=1024: paired delta=0.00 pp; paired CI95=[0.00,0.00] pp; positive/negative/zero=0/0/1024
Teacher absolute recall: N/A
Student absolute recall: N/A
```

Conclusion: `importance_tagging` has no paired `usable_evidence_recall@K` gain in the formal same-state fork. Its historical weighted utility remains negative (K4 `-0.93%`, K8 `-1.51%`), so the component gate remains failed. Absolute recall is not fabricated because endpoint evidence IDs were not persisted.

Artifact basis: `SCAPE/outputs/0816_2_importance_proper_fork_formal_final/IMPORTANCE_PROPER_VALUE_PER_STATE.jsonl` and `IMPORTANCE_K4_K8_GATE.json`.

## 2026-08-20 subtractive_curation candidate-pool and activated-evidence recall audit

Status: **COMPLETED; ZERO PAIRED CANDIDATE-POOL RECALL GAIN; ACTIVATED-EVIDENCE RECALL N/A**.

Setting and proof:

```text
component: subtractive_curation
formal source: SCAPE/outputs/0820_subtractive_curation_single_128_final/SUBTRACTIVE_VALUE_PER_STATE.jsonl
states: 128 per K; K4/K8 same-state snapshot audit=128/128
qrel: SCOPE/external/BrowseComp-Plus/topics-qrels/qrel_evidence.txt
qrel SHA-256: a6f594975be57339de9e4e9f67f13c044f647feda77c0b84c45a1581e3041bd1
document-ID normalization: str(doc_id).split("_", 1)[0]
endpoint candidate pool P_K: normalized endpoint working-memory/candidate evidence IDs
activated set A_K: final curated IDs union successful read_document IDs retained in context
```

All saved Teacher and Student branch actions are `read_document` or `end_search`; there are no branch-local search, curate, review, or verify transitions. Under the formal runner, neither action changes the candidate pool, so the endpoint candidate pool is identical across branches and candidate-pool recall has zero paired delta for every row. However, Teacher and Student read traces differ, and the artifact does not persist successful read observations or endpoint context-retention provenance. Therefore the formal activated-evidence recall cannot be reconstructed and is reported as `N/A`, not zero.

Results:

```text
K4 n=128: candidate-pool paired delta=0.00 pp; CI95=[0.00,0.00] pp; positive/negative/zero=0/0/128; activated delta=N/A
K8 n=128: candidate-pool paired delta=0.00 pp; CI95=[0.00,0.00] pp; positive/negative/zero=0/0/128; activated delta=N/A
Teacher/Student absolute candidate recall: N/A
Teacher/Student absolute activated recall: N/A
```

Absolute recall remains `N/A` because the formal per-state artifact omitted gold/endpoint candidate-pool, curated, working-memory, successful-read, and retained-read evidence IDs. A reconstruction from current corpus files is not a provenance-complete substitute for those omitted fork snapshots. Conclusion: `subtractive_curation` adds no paired candidate-pool recall; activated-evidence recall is not assessable from this artifact. Its historical weighted utility remains negative (K4 `-0.63%`, K8 `-1.28%`) and is not relabeled as recall.

Artifact: `SCAPE/outputs/0820_subtractive_curation_single_128_final/SUBTRACTIVE_USABLE_EVIDENCE_RECALL.json`.

## 2026-08-20 evidence_graph usable_evidence_recall@K endpoint replay

Status: **COMPLETED; ZERO USABLE-EVIDENCE RECALL GAIN**.

Setting and replay audit:

```text
component: evidence_graph
formal source: SCAPE/outputs/0820_evidence_graph_formal_fork_128/shards/evidence_graph_K{4,8}.jsonl
states: 128 per K; K4/K8 ordered snapshot_hash match=128/128
search backend: local_corpus_token_overlap
qrel: SCOPE/external/BrowseComp-Plus/topics-qrels/qrel_evidence.txt
qrel SHA-256: a6f594975be57339de9e4e9f67f13c044f647feda77c0b84c45a1581e3041bd1
document-ID normalization: str(doc_id).split("_", 1)[0]
endpoint U_K: final curated IDs union final working-memory document IDs union successful read_document IDs retained in context
```

The saved branch traces contain only `read_document` and `end_search`; there are no branch-local search or curate mutations. Therefore the frozen initial working-memory documents can be deterministically reconstructed from the fixed local corpus, while branch read IDs come directly from the trace. Teacher and Student share the same starting snapshot and continuation policy.

Results:

```text
K4 n=128: Teacher=1.6611%, Student=1.6611%, paired delta=0.00 pp
  paired bootstrap CI95=[0.00,0.00] pp; positive/negative/zero=0/0/128
K8 n=128: Teacher=1.6611%, Student=1.6611%, paired delta=0.00 pp
  paired bootstrap CI95=[0.00,0.00] pp; positive/negative/zero=0/0/128
```

Conclusion: `evidence_graph` adds no `usable_evidence_recall@K` in this 128-state same-state fork. Its historical weighted utility deltas remain negative (K4 `-2.84%`, K8 `-4.87%`), so the component gate remains failed. Recall and weighted utility are reported separately and are not interchangeable.

Artifact: `SCAPE/outputs/0820_evidence_graph_formal_fork_128/EVIDENCE_GRAPH_USABLE_EVIDENCE_RECALL.json`.

## 2026-08-20 content_dedup corrected 128-state same-state reward fork

Status: **CONTENT_DEDUP_CORRECTED_128_STATE_K4_K8_REWARD_GATE_PASS**.

Setting:

```text
component: content_dedup
source pool: SCAPE-EasyOPD/outputs/component_sweep_0818/h100_2/content_dedup_corrected_high_redundancy_v3/TRAIN_STATES_5K.jsonl
selection: deterministic SHA-256 ordering by seed=20260820, n_states=128
K4/K8 state identity audit: 128/128 unique states, identical ordered state_uid list
Full branch: content_dedup ON
Reduced branch: content_dedup OFF
Continuation: same reduced policy on both branches; no full-harness takeover
student_inference_privilege: false
runner: SCAPE/scripts/run_content_dedup_corrected_reward_fork.py
```

Result:

```text
K4 mean Teacher-Student reward delta = +0.2832011245 (+28.32%)
  bootstrap CI95=[+0.2485665091, +0.3192347296]
  positive/negative/zero=128/0/0
  gate_passed=true

K8 mean Teacher-Student reward delta = +0.1873082674 (+18.73%)
  bootstrap CI95=[+0.1743259729, +0.1999209890]
  positive/negative/zero=128/0/0
  gate_passed=true
```

Artifacts:

```text
SCAPE/outputs/0820_content_dedup_corrected_reward_fork_128/CONTENT_DEDUP_CORRECTED_K4_K8_GATE.json
SCAPE/outputs/0820_content_dedup_corrected_reward_fork_128/CONTENT_DEDUP_CORRECTED_K4_K8_SUMMARY.csv
SCAPE/outputs/0820_content_dedup_corrected_reward_fork_128/CONTENT_DEDUP_CORRECTED_REWARD_PER_STATE.jsonl
SCAPE/outputs/0820_content_dedup_corrected_reward_fork_128/RUN_MANIFEST.json
SCAPE/outputs/0820_content_dedup_corrected_reward_fork_128/SHA256SUMS
```

Conclusion: the corrected single-component `content_dedup` gate remains passed under the unified 128-state protocol. These values replace the prior 5000-state values in `/mnt/songzijun/增益.md`; the 5000-state run remains an intact historical large-sample artifact.

## 2026-08-20 H100-2 content_dedup adapter-conditioned OPD comparison

Status: **CONTENT_DEDUP_ACTION_LEVEL_OPD_COMPARISON_COMPLETE_NO_INTERNALIZATION_SIGNAL_TASK_REWARD_NA**.

Setting:

```text
base: /mnt/songzijun/models/Qwen3-30B-A3B-Instruct-2507
rows: content_dedup_corrected_high_redundancy_v3/OPD_VALID_ROWS.jsonl, n=500 frozen valid states
Teacher: base + prompt_full (content_dedup component-on canonical target/signal)
Student Before: same base + prompt_reduced, no privilege
Student After: prompt_reduced + actual LoRA; PURE_OPD and RL_PLUS_OPD seeds 42/43
student_inference_privilege: false for all recorded rows
adapter reload: 4/4 manual safetensors state-dict
inference: greedy, max_new_tokens=128
pairing/bootstrap: identical 500 state_uid; 10000 paired seed-row bootstrap resamples
```

Action-level results:

```text
Teacher:             legal=1.000, exact_projected_target=1.000
Student Before:      legal=0.064, exact_projected_target=0.000
PURE seed42 After:   legal=0.008, exact=0.000
PURE seed43 After:   legal=0.000, exact=0.000
RL+OPD seed42 After: legal=0.004, exact=0.000
RL+OPD seed43 After: legal=0.024, exact=0.000

PURE seed-merged legal=0.004 +/- 0.005657; delta vs Before=-0.060 +/- 0.005657
  paired seed-row CI95=[-0.076,-0.045], positive/negative/zero=4/64/932
RL+OPD seed-merged legal=0.014 +/- 0.014142; delta vs Before=-0.050 +/- 0.014142
  paired seed-row CI95=[-0.067,-0.033], positive/negative/zero=14/64/922
Both methods exact-target=0; delta=0; positive/negative/zero=0/0/1000.
```

Training diagnostic and constraints:

- Teacher-forced valid divergence improved from Before `0.736175` to PURE `0.143824/0.150096` and RL_PLUS_OPD `0.153125/0.134477`, but this did not transfer to legal generation or exact projected-target matching.
- This is a frozen OPD-valid-row action-level diagnostic, not BrowseComp DEV/TEST terminal reward. Overall reward, trajectory/curated-evidence/final-answer recall, turns and tool calls remain `N/A`.
- PURE was trained with actual `action_ce`; the current `RL_PLUS_OPD` artifact is the trainer's minimal extra `tool_token_kl` hook, not formal online GRPO. Preserve the artifact name but do not claim protocol-complete RL+OPD.
- Teacher sees the canonical target in `prompt_full`; Teacher=100% is a conditioned upper bound, not a task-reward measurement.
- The corrected high-redundancy fork uses corrected duplicate ids. Its pre-OPD same-state K4/K8 utility gate remains passed, but the current OPD adapters fail the action-level learnability comparison; do not claim Student After > Before.

Artifacts:

```text
SCAPE-EasyOPD/scripts/eval_content_dedup_opd_comparison.py
SCAPE-EasyOPD/scripts/summarize_content_dedup_opd_comparison.py
SCAPE-EasyOPD/outputs/component_sweep_0818/h100_2/content_dedup_adapter_conditioned/CONTENT_DEDUP_OPD_COMPARISON_SUMMARY.json
/mnt/songzijun/opd对比.md
```

## 2026-08-20 Single-component subtractive_curation same-state K4/K8 fork

Status: **COMPLETED; NO POSITIVE REWARD GAIN**.

Setting:

```text
component: subtractive_curation only
states: 128 per K (256 rows total), four deterministic 32-state cohorts
same-state contract: K4/K8 rows use identical xi_t snapshots; all 128 paired checks matched snapshot_hash
Teacher/Full: subtractive_curation ON for the forced first action
Student/Reduced: subtractive_curation OFF for the forced first action
continuation: both branches use the reduced policy; no full-harness takeover
runner: SCAPE/scripts/run_h100_2_live_fork_replay_stream.py (batched scorer; local corpus fallback)
model: /mnt/songzijun/models/pat-jj_harness-1-full/harness-1
```

Teacher - Student reward summary:

```text
K4 n=128: mean=-0.0063281250 (-0.63%), median=0, CI95 normal approx=[-0.0143267,+0.0016705], pos/neg/zero=52/57/19
K8 n=128: mean=-0.0127734375 (-1.28%), median=0, CI95 normal approx=[-0.0254542,-0.0000927], pos/neg/zero=57/61/10
```

Interpretation: under this closed-loop reward definition, opening subtractive_curation did not produce a positive Teacher advantage. K4 is statistically inconclusive because its CI crosses zero; K8 is a small negative effect with the normal-approximation CI just below zero. Mean evidence-coverage gain was 0.0 and the Teacher branch incurred positive mean tool-cost deltas (K4 +0.421875, K8 +0.8515625). This is a valid single-component same-state fork result, not a DEV/TEST adapter-training claim.

Artifacts:

```text
SCAPE/outputs/0820_subtractive_curation_single_128_final/SUBTRACTIVE_K4_K8_GATE.json
SCAPE/outputs/0820_subtractive_curation_single_128_final/SUBTRACTIVE_K4_K8_GATE.md
SCAPE/outputs/0820_subtractive_curation_single_128_final/SUBTRACTIVE_SUMMARY.csv
SCAPE/outputs/0820_subtractive_curation_single_128_final/SUBTRACTIVE_VALUE_PER_STATE.jsonl
```

## 2026-08-20 Joint importance + subtractive curation pre-OPD fork pilot

Status: **JOINT_PREOPD_K4_K8_GATE_FAILED_PILOT128**.

Setting:

```text
component: importance_tagging_plus_subtractive_curation
contract: same xi_t; Teacher/Full has importance_tagging + subtractive_curation ON for first branch; Student/Reduced has both OFF; both continuations use reduced policy; no full-harness takeover
seeds: 8423, 8424
K: 4, 8
states: 128 per seed/K cell, 512 rows total
runner: SCAPE/scripts/run_joint_importance_subtractive_preopd_fork.py
output: SCAPE/outputs/0820_joint_importance_subtractive_preopd_fork_pilot128_retry/
```

Results:

```text
seed8423 K4: +0.04%  (mean T-S=+0.000352, pos/neg/zero=53/59/16)
seed8423 K8: -0.02%  (mean T-S=-0.000234, pos/neg/zero=53/63/12)
seed8424 K4: -0.75%  (mean T-S=-0.007500, pos/neg/zero=42/72/14)
seed8424 K8: -1.09%  (mean T-S=-0.010898, pos/neg/zero=46/73/9)
merged K4: -0.36%
merged K8: -0.56%
```

Conclusion:

The joint bundle does **not** pass the pre-OPD same-state reward utility gate in the 128-state pilot. The previous projected qrel / LoRA artifacts remain useful diagnostics, but they are not evidence that the joint component has positive pre-OPD reward gain.

Artifacts:

```text
SCAPE/outputs/0820_joint_importance_subtractive_preopd_fork_pilot128_retry/JOINT_PREOPD_K4_K8_GATE.json
SCAPE/outputs/0820_joint_importance_subtractive_preopd_fork_pilot128_retry/JOINT_PREOPD_K4_K8_GATE.md
SCAPE/outputs/0820_joint_importance_subtractive_preopd_fork_pilot128_retry/JOINT_PREOPD_SUMMARY.csv
SCAPE/outputs/0820_joint_importance_subtractive_preopd_fork_pilot128_retry/JOINT_PREOPD_VALUE_PER_STATE.jsonl
```

## 2026-08-20 token_budget_marker adapter-conditioned paired OPD evaluation

Status: **TOKEN_BUDGET_ADAPTER_CONDITIONED_PAIRED_EVAL_READY_AND_POSITIVE**.

A dedicated evaluator was added because the previous `SCAPERealClosedLoopEvaluator` used fixed scripted actions and did not load cell adapters. The corrected evaluation uses the same Qwen3 base, the actual LoRA adapter for each After cell, the same reduced no-privilege DEV/TEST prompts, Qwen3 compact tool serialization mapped to legal Harness-1 actions, and official `ToolSet` execution. Reward is `0.25 legal + 0.25 executable + 0.25 live Harness-1 execution`; comparisons are paired by query_id with 2000 bootstrap replicates.

```text
Student Before: DEV=0.5625, TEST=0.5391

PURE_OPD seed42: DEV=0.7031, delta=+0.1406 CI95 [0.0762,0.2051]; TEST=0.7002, delta=+0.1611 CI95 [0.1113,0.2080]
PURE_OPD seed43: DEV=0.6973, delta=+0.1348 CI95 [0.0645,0.2051]; TEST=0.7090, delta=+0.1699 CI95 [0.1230,0.2197]
RL_PLUS_OPD seed42: DEV=0.7090, delta=+0.1465 CI95 [0.0820,0.2109]; TEST=0.6885, delta=+0.1494 CI95 [0.0996,0.1992]
RL_PLUS_OPD seed43: DEV=0.6914, delta=+0.1289 CI95 [0.0586,0.1992]; TEST=0.6621, delta=+0.1230 CI95 [0.0703,0.1729]
```

All 8 cell-split paired CI95 lower bounds are positive. Seed-aggregated relative gains are PURE OPD `+24.48%` DEV / `+30.71%` TEST and RL+OPD `+24.48%` DEV / `+25.27%` TEST. Invalid-tool rates decrease and live Harness-1 execution rates increase for every cell/split. Artifact: `SCAPE-EasyOPD/outputs/component_sweep_0818/h100_4/token_budget_marker/adapter_conditioned_full/TOKEN_BUDGET_ADAPTER_CONDITIONED_PAIRED_SUMMARY.json`; consolidated table: `/mnt/songzijun/opd对比.md`.

## 2026-08-20 sentence_compress same-state K4/K8 reward fork

Status: **SENTENCE_COMPRESS_K4_K8_WEAK_POSITIVE_CI_CROSSES_ZERO**.

Setting:

```text
component: sentence_compress
states: 128 frozen xi_t states from sentence_compress Reduced/Student state occupancy
states_manifest: SCAPE/outputs/0820_sentence_compress_formal_fork_k128_frozen_pool1024/manifests/sentence_compress_states_n128_seed2214.jsonl
Teacher/Full branch: sentence_compress ON for the first fork action
Student/Reduced branch: sentence_compress OFF for the first fork action
continuation: both branches continue with the reduced policy; no full-harness takeover
runner: true_live_fork_replay_hf_bm25
model: /mnt/songzijun/models/pat-jj_harness-1-full/harness-1
python_env: /opt/scape-sentence-compress-venv
```

Reward gain summary, Teacher - Student:

```text
K4 n=128: mean=+0.0043359375, median=+0.0150000000, CI95 normal approx=[-0.0031796863, +0.0118515613], pos/neg/zero=67/50/11, mean tool-cost delta=-0.2890625
K8 n=128: mean=+0.0038671875, median=+0.0000000000, CI95 normal approx=[-0.0085086283, +0.0162430033], pos/neg/zero=63/50/15, mean tool-cost delta=-0.2578125
```

Artifacts:

```text
SCAPE/outputs/0820_sentence_compress_formal_fork_k128_frozen_pool1024/SENTENCE_COMPRESS_K4_K8_REWARD_GAIN.json
SCAPE/outputs/0820_sentence_compress_formal_fork_k128_frozen_pool1024/SENTENCE_COMPRESS_K4_K8_REWARD_GAIN.md
SCAPE/outputs/0820_sentence_compress_formal_fork_k128_frozen_pool1024/SENTENCE_COMPRESS_VALUE_PER_STATE.jsonl
```

Interpretation and constraints:

- The 128-state same-xi_t fork gives small positive mean Teacher-Student reward gain at both K4 and K8.
- The normal-approx CI crosses zero for both horizons, so this is weak positive evidence, not a statistically clean DEV/TEST promotion result.
- No replay-noise shard was completed for the 128-state run; earlier 16-state smoke was directionally negative and should not be used as the controlling result.
- This run answers the requested K4/K8 closed-loop reward fork on the same xi_t states. A formal DEV/TEST closed-loop claim would require a larger split or paired bootstrap/replay-noise completion before promotion.

## 2026-08-20 H100-1 auto_populate_first_search formal OPD comparison

Status: **AUTO_FORMAL_OPD_TRAINING_RELOAD_AND_AUTO_ACTION_COMPARISON_COMPLETE_TASK_REWARD_PENDING**.

Setting:

```text
component: auto_populate_first_search
base: /mnt/songzijun/models/Qwen3-30B-A3B-Instruct-2507
collector: real Harness-1; synthetic_fallback=false; synthetic_row_count=0
support: 2000 queries / 8000 rollouts / 8000 unique event-active / 5000 frozen TRAIN_STATES
OPD rows: 4500 train / 500 valid
student_inference_privilege: false
cells: PURE_OPD seed42/43; RL_PLUS_OPD seed42/43
LoRA: r=8, alpha=16, lr=1e-5, anchor=0.05, one epoch
reload: 4/4 manual safetensors state-dict pass; native PEFT conversion incompatible
```

Formal cell results:

```text
PURE_OPD seed42:     pre_div=1.5158456155 post_div=0.2291391738 delta=-1.2867064417; steps=4500; loss=0.2930067274; reload=true
PURE_OPD seed43:     pre_div=1.5158456155 post_div=0.2287955356 delta=-1.2870500799; steps=4500; loss=0.2938684896; reload=true
RL_PLUS_OPD seed42:  pre_div=1.5158456155 post_div=0.2304975062 delta=-1.2853481093; steps=9000; loss=0.2586068922; reload=true
RL_PLUS_OPD seed43:  pre_div=1.5158456155 post_div=0.2276069658 delta=-1.2882386497; steps=9000; loss=0.2579987532; reload=true
```

Aggregates:

```text
PURE_OPD post_div=0.2289673547 +/- 0.0002431394; delta=-1.2868782603 +/- 0.0002431394
RL_PLUS_OPD post_div=0.2290522362 +/- 0.0020442757; delta=-1.2867933795 +/- 0.0020442757
```

AUTO-specific action comparison on the same 500 frozen OPD valid rows:

```text
Teacher:             legal_rate=0.910; exact_projected_target_rate=0.090
Student Before:      legal_rate=0.996; exact_projected_target_rate=0.004
PURE_OPD After:      legal_rate=0.422; exact_projected_target_rate=0.006; reload=manual_safetensors
RL_PLUS_OPD After:   legal_rate=0.868; exact_projected_target_rate=0.058; reload=manual_safetensors
```

The evaluator is `/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD/scripts/eval_auto_opd_comparison.py`; outputs are under `.../auto_populate_first_search/adapter_conditioned_formal_v2/`. Teacher uses `prompt_full` with teacher-only AUTO semantics; Before/After use the same `prompt_reduced`, and both After conditions are adapter-conditioned with `student_inference_privilege=false`.

Interpretation and hard constraints:

- Both methods and both seeds improve the valid-row teacher-forced divergence proxy, but the primary new comparison is the AUTO-specific adapter-conditioned action evaluation above.
- RL+OPD After improves projected-target exact match from Before `0.004` to `0.058`, but remains below Teacher `0.090`; PURE_OPD improves only to `0.006` and has a severe legal-action drop to `0.422`.
- This is still not a formal terminal/real-task reward result: the evaluator measures projected action matching on frozen OPD rows, not BrowseComp terminal reward, trajectory recall, or final-answer recall. Task reward remains `N/A`.
- The earlier generic adaptive-rerank smoke was not promoted and is not part of the new comparison.
- Existing earlier AUTO real closed-loop artifacts under a different/local compatibility recipe did not show Student After beating Base; do not claim `PASS_BOTH`, paper-grade task-reward win, or leaderboard promotion.
- Canonical summary table: `/mnt/songzijun/opd对比.md`.
- Canonical training root: `SCAPE-EasyOPD/outputs/component_sweep_0818/h100_1_qwen3/auto_populate_first_search/`.

## 2026-08-20 token_budget_marker full DEV/TEST evaluation completed

Four existing OPD training cells were evaluated with the no-`--skip-closed-loop` runner:

```text
PURE_OPD_seed42: DEV 128, TEST 256, error_rate=0, mean_reward=0.001/0.001
PURE_OPD_seed43: DEV 128, TEST 256, error_rate=0, mean_reward=0.001/0.001
RL_PLUS_OPD_seed42: DEV 128, TEST 256, error_rate=0, mean_reward=0.001/0.001
RL_PLUS_OPD_seed43: DEV 128, TEST 256, error_rate=0, mean_reward=0.001/0.001
```

The live loop used real Harness-1 tools and `student_inference_privilege=false`, but `SCAPERealClosedLoopEvaluator` does not load each cell's LoRA adapter into generation. Therefore the `0.001` values are successful scripted live-loop smoke rewards, not adapter-conditioned Student After reward deltas. The exporter also still sets `prompt_full == prompt_reduced`, so persisted divergence proxies remain zero. Formal adapter-conditioned Student After reward remains `N/A`; artifacts are retained under `SCAPE-EasyOPD/outputs/component_sweep_0818/h100_4/token_budget_marker/formal_evals_full/`.

Status: **OPD_TRAINING_ARTIFACTS_COMPLETE_BUT_FORMAL_LEARNABILITY_GATE_BLOCKED**.

Setting and artifact status:

```text
base: Qwen3-30B-A3B-Instruct-2507
component: token_budget_marker
collection: real_harness1, 5000 unique train states, synthetic=0
OPD rows: 4500 train / 500 valid
cells: PURE_OPD seed42/43; RL_PLUS_OPD seed42/43
adapter reload: 4/4 pass via manual safetensors state-dict fallback
student_inference_privilege: false
```

Audit result:

```text
Teacher divergence proxy: 0.0
Student Before divergence proxy: 0.0
PURE_OPD After divergence proxy: 0.0
RL_PLUS_OPD After divergence proxy: 0.0
DEV/TEST real closed-loop: skipped in all four formal eval cells
```

The zero values are not formal reward results: the current token-budget exporter records `prompt_full == prompt_reduced`, and the evals were generated with `--skip-closed-loop`. Therefore these artifacts prove training completion and adapter reload only. They do not prove Teacher utility or adapter-conditioned Student After reward.

Under the 0819-3 placement protocol, Positive Utility must pass before Student After can be evaluated. The 2026-08-20 128-state rerun used a frozen shared candidate cache and produced `K4=-0.0041015625` (CI95 `[-0.0103802301,+0.0021771051]`, positive/negative/zero=`47/66/15`) and `K8=-0.00046875` (CI95 `[-0.0113573681,+0.0104198681]`, positive/negative/zero=`54/63/11`). K4/K8 ordered snapshot hashes matched `128/128`; both continuations were reduced-policy only and full-harness takeover was `0/256`. The formal utility gate therefore remains failed / unstable, and the placement decision remains `KEEP_RUNTIME_OR_DROP_COMPONENT`. Artifact: `SCAPE/outputs/0820_token_budget_marker_formal_fork_128_final/TOKEN_BUDGET_MARKER_K4_K8_GATE.json`.

The previous 64-state values (`K4=-0.002578125`, `K8=+0.001171875`) are retained only as historical measurements and are no longer the current table values.

## 2026-08-20 token_budget_marker evidence-recall 128-state formal fork (invalidated)

Status: **INVALID_INSUFFICIENT_BUDGET_PRESSURE; DIAGNOSTIC_ONLY**.

The prior 128-state cohort is withdrawn from the formal gain table. Its marker values were only `remaining=6144..7936` in the simplified runner, with no real Harness-1 threshold/rejection/prune pressure and no candidate-pool transition. It therefore cannot support a token-budget component judgment, even though the paired fork and endpoint scorer were internally consistent.

Setting and provenance:

```text
component: token_budget_marker
cohort: frozen TOKEN_BUDGET_MARKER_STATES_128.jsonl
seed: 2214 (single frozen cohort; seed-balanced value equals pooled value)
K: 4 and 8; first forced action counts as step 1
model: local Harness-1 checkpoint /mnt/songzijun/models/pat-jj_harness-1-full/harness-1
environment: /opt/scape-easyopd-smoke7
normalization: split_at_first_underscore_v1
qrel_sha256: a6f594975be57339de9e4e9f67f13c044f647feda77c0b84c45a1581e3041bd1
context retention: successful read observations append-only retained to endpoint
```

Teacher enabled `token_budget_marker` only for the first action; Student disabled it; both branches used Reduced continuation and `full_harness_takeover=0`. K4 and K8 each have 128 valid paired rows, ordered snapshot hashes match `128/128`, and no qrel is missing or empty. Independent scoring from endpoint candidate IDs and activated IDs passed the union/success/context-retention audit.

```text
K4 candidate recall:  Teacher 2.265625% / Student 2.265625%, delta 0.00 pp
K4 activated recall: Teacher 0.390625% / Student 0.390625%, delta 0.00 pp
K8 candidate recall:  Teacher 2.265625% / Student 2.265625%, delta 0.00 pp
K8 activated recall: Teacher 0.390625% / Student 0.390625%, delta 0.00 pp
paired CI95: [0.00, 0.00] pp for both metrics and both horizons
positive/negative/zero: 0/0/128 for both metrics and horizons
utility delta: K4 -0.0041015625; K8 -0.0017578125
```

Diagnostic artifact: `SCAPE/outputs/0820_token_budget_marker_evidence_recall_formal/scored_final/TOKEN_BUDGET_MARKER_EVIDENCE_RECALL_SUMMARY.json` and `TOKEN_BUDGET_MARKER_EVIDENCE_RECALL_PER_STATE.jsonl`. The observed `0.00 pp` values are retained only to document the invalid low-pressure cohort and have been restored to `N/A` in `/mnt/songzijun/增益.md`. A replacement cohort must use real tokenizer-counted Harness-1 pressure states before formal scoring.


## 2026-08-20 H100-2 content_dedup corrected high-redundancy reward fork + OPD

Status: **CONTENT_DEDUP_CORRECTED_K4_K8_REWARD_GATE_PASS_AND_OPD_COMPLETE**.

Setting:

```text
component: content_dedup
source xi_t: SCAPE-EasyOPD/outputs/component_sweep_0818/h100_2/content_dedup_corrected_high_redundancy_v3/TRAIN_STATES_5K.jsonl
n_states: 5000 real_harness1 corrected high-redundancy states
Student inference privilege: false
Full branch: content_dedup ON, uses recorded dedup-on canonical projectable target
Reduced branch: content_dedup OFF, acts on unfiltered duplicate-heavy pool
Continuation: same reduced policy on both branches; no full-harness takeover
runner: SCAPE/scripts/run_content_dedup_corrected_reward_fork.py
```

Corrected collection support:

```text
collection_status=READY_5K
n_queries_selected=2000
n_rollouts_total=8000
n_event_active_raw=32000
n_unique_event_active=32000
TRAIN_STATES_5K rows=5000
synthetic_row_count=0
mean_duplicate_suppressed_count=22.0016
OPD rows=4500 train / 500 valid
```

Same-state reward fork result:

```text
K4 Teacher-Student mean reward delta = +0.261302
  CI95=[+0.255786, +0.266558]
  positive/negative/zero = 5000/0/0
  gate_passed=true

K8 Teacher-Student mean reward delta = +0.179341
  CI95=[+0.177273, +0.181299]
  positive/negative/zero = 5000/0/0
  gate_passed=true
```

Corrected OPD/internalization result:

```text
formal training root: SCAPE-EasyOPD/outputs/component_sweep_0818/h100_2/content_dedup_formal_hf_corrected_compact_8gpu/
completed cells: PURE_OPD seed42/43; RL_PLUS_OPD seed42/43
adapter reload: 4/4 passed via manual safetensors state-dict fallback
Student Before div: 0.736175
PURE_OPD seed42 After div:     0.143824, delta=0.592351
PURE_OPD seed43 After div:     0.150096, delta=0.586080
RL_PLUS_OPD seed42 After div:  0.153125, delta=0.583050
RL_PLUS_OPD seed43 After div:  0.134477, delta=0.601698
```

Artifacts:

```text
SCAPE/outputs/0820_content_dedup_corrected_reward_fork/CONTENT_DEDUP_CORRECTED_K4_K8_GATE.json
SCAPE/outputs/0820_content_dedup_corrected_reward_fork/CONTENT_DEDUP_CORRECTED_K4_K8_SUMMARY.csv
SCAPE/outputs/0820_content_dedup_corrected_reward_fork/CONTENT_DEDUP_CORRECTED_REWARD_PER_STATE.jsonl
SCAPE/outputs/0820_content_dedup_corrected_reward_fork/SHA256SUMS
SCAPE-EasyOPD/outputs/component_sweep_0818/h100_2/H1002_CONTENT_DEDUP_CORRECTED_GAIN_OPD_SUMMARY.json
SCAPE-EasyOPD/outputs/component_sweep_0818/h100_2/content_dedup_corrected_high_redundancy_v3/H1002_CONTENT_DEDUP_OPD_ROWS_MANIFEST.json
```

Conclusion:

```text
CONTENT_DEDUP_CORRECTED_SINGLE_COMPONENT_REWARD_GATE_PASS
```

This supersedes the earlier zero-trigger content_dedup sampling conclusion for the single-component corrected high-redundancy gate. The older H100-3 retrieval bundle conclusion remains `DISCARD_RETRIEVAL_BUNDLE` for the AUTO/AUTO_DEDUP bundle, not for corrected single-component dedup utility.

## 2026-08-19 H100-2 retrieval/runtime component sweep Qwen3 fast-start

Status: **H1002_ADAPTIVE_RERANK_FORMAL_EVAL_COMPLETE_REWARD_SMOKE_LIMITED**.

Setting:

```text
machine role: H100-2
components: content_dedup, chunk_neighbors, adaptive_rerank_instruction
canonical_student_base: /mnt/songzijun/models/Qwen3-30B-A3B-Instruct-2507
logical_model_id: Qwen3-30B-A3B-Instruct-2507
runtime env: /opt/scape-easyopd-smoke7 via SCAPE-EasyOPD/scripts/setup_scape_easyopd_smoke7_env.sh
collector: real Harness-1 bridge, collector_mode=real_harness1
student_inference_privilege: false
query pool: 2000 train-side queries
synthetic_fallback: false
```

Phase U / support gates:

```text
adaptive_rerank_instruction: READY_5K
  n_queries_selected=2000
  n_rollouts_total=8000
  n_unique_event_active=8000
  TRAIN_STATES_5K rows=5000
  OPD rows=4500 train / 500 valid
  loss_path=full_response_kl
  synthetic_row_count=0

content_dedup: INSUFFICIENT_5K_EVENT_SUPPORT
  real_harness1 Stage E event support=0
  OPD training launched: no

chunk_neighbors: NON_REALIZABLE_EXTERNAL_INFORMATION
  no student-visible neighbor injection hook located and real event support=0
  OPD training launched: no
```

Formal adaptive training/eval:

```text
formal training root: SCAPE-EasyOPD/outputs/component_sweep_0818/h100_2/formal_hf_adaptive_8gpu/
formal eval root:     SCAPE-EasyOPD/outputs/component_sweep_0818/h100_2/formal_evals/
completed cells: PURE_OPD seed42/43; RL_PLUS_OPD seed42/43
adapter reload: 4/4 passed via manual safetensors state-dict fallback
PEFT native Qwen3 conversion: still avoided due WeightConverter.__init__(distributed_operation) incompatibility
```

Valid-row divergence diagnostic:

```text
Student Before div: 0.665565
PURE_OPD seed42 After div:     -0.083540, delta=0.749105, bootstrap 95% CI positive
PURE_OPD seed43 After div:     -0.087396, delta=0.752961, bootstrap 95% CI positive
RL_PLUS_OPD seed42 After div:  -0.004367, delta=0.669933, bootstrap 95% CI positive
RL_PLUS_OPD seed43 After div:  -0.080368, delta=0.745934, bootstrap 95% CI positive
```

Closed-loop caveat:

The DEV=128 and TEST=256 live Harness-1 summaries completed with `student_inference_privilege=false` and mean smoke reward `0.001` for all cells. However, the current `SCAPERealClosedLoopEvaluator` is a scripted tool-success smoke loop and does not load each saved adapter for generation. Therefore these artifacts prove that the no-privilege live tool loop can run, but they are **not** a paper-grade Student After > Student Before reward claim; adapter-conditioned closed-loop reward delta is recorded as `N/A_smoke_reward_no_adapter_conditioning`.

Artifacts:

```text
SCAPE-EasyOPD/outputs/component_sweep_0818/h100_2/H1002_COMPONENT_HANDOFF.json
SCAPE-EasyOPD/outputs/component_sweep_0818/h100_2/H1002_FORMAL_EVAL_SUMMARY.json
SCAPE-EasyOPD/outputs/component_sweep_0818/h100_2/H1002_FORMAL_ADAPTIVE_SUMMARY.json
SCAPE-EasyOPD/outputs/component_sweep_0818/h100_2/H1002_COMPONENT_ROWS.{json,csv}
SCAPE-EasyOPD/outputs/component_sweep_0818/h100_2/SHA256SUMS
SCAPE-EasyOPD/outputs/component_sweep_0818/h100_2/components/adaptive_rerank_instruction/TRAIN_STATES_5K.jsonl
```

Conclusion:

```text
H1002_ADAPTIVE_RERANK_FORMAL_EVAL_COMPLETE_REWARD_SMOKE_LIMITED
```

`adaptive_rerank_instruction` has a positive adapter/reload/divergence internalization diagnostic, but final paper-grade PASS/FAIL still requires an adapter-conditioned real closed-loop evaluator. `content_dedup` and `chunk_neighbors` remain stopped by support/realizability gates, with no synthetic promotion.

## 2026-08-19 H100-4 Qwen3 fast-start control component sweep

Status: **H100-4 complete / no formal H100-4 OPD training launched by protocol gate**.

Setting:

```text
machine role: H100-4
components: token_budget_marker, verify_tool
canonical_student_base: /mnt/songzijun/models/Qwen3-30B-A3B-Instruct-2507
logical_model_id: Qwen3-30B-A3B-Instruct-2507
runtime env: /opt/scape-h1004 via SCAPE-EasyOPD/scripts/setup_scape_easyopd_smoke7_env.sh
collector: real Harness-1 bridge, collector_mode=real_harness1
student_inference_privilege: false
query pool: 2000 train-side queries
rollouts: 4 per query, 8000 rollouts/component
selection_seed: 20260818
synthetic_fallback: false
```

Code/contract changes completed:

```text
- EasyOPD formal collector and Harness1Bridge were switched from the stale openai/gpt-oss-20b contract to the local Qwen3-30B-A3B-Instruct-2507 contract.
- Added Qwen3NativeChatAdapter using the local tokenizer native chat template; H100-4 acceptance passed with HARNESS1_EASYOPD_READY and synthetic_fallback=false.
- Added real bridge events for token_budget_marker and verify_tool.
- token_budget_marker records Harness-1 token-budget marker/accounting on the same Student pre-state; it remains PARTIAL and does not expose hidden counters as a Student target.
- verify_tool records Teacher action-space availability of verify(doc_ids, claim), while Student action space remains without verify.
- H100-4 runner now reads EasyOPD Qwen3 train-pool/handoff paths and writes outputs under SCAPE-EasyOPD/outputs/component_sweep_0818/h100_4/.
- Master aggregation now prioritizes Qwen3 handoffs: h100_1_qwen3, h100_3_qwen3_faststart, h100_4.
```

Phase U collection results:

```text
token_budget_marker: READY_5K
  n_queries_selected=2000
  n_rollouts_total=8000
  n_unique_event_active=8000
  TRAIN_STATES_5K rows=5000
  synthetic_row_count=0

verify_tool: READY_5K
  n_queries_selected=2000
  n_rollouts_total=8000
  n_unique_event_active=8000
  TRAIN_STATES_5K rows=5000
  synthetic_row_count=0
```

OPD pilot and diagnostics:

```text
token_budget_marker OPD_PILOT:
  pilot states=256 real_harness1 states
  Qwen3-30B LoRA training steps=4
  adapter saved=true
  PEFT native reload: failed with WeightConverter.__init__(distributed_operation)
  Transformers load_adapter fallback: same failure
  manual safetensors mapped reload: passed, 384/384 LoRA tensors loaded
  post-reload forward: passed
  ADAPTER_RELOAD_ACCEPTANCE status=ADAPTER_RELOAD_READY

Teacher/Before diagnostic gate:
  token_budget_marker:
    token measurement=qwen3_native_chat_template_next_context_with_current_observation
    budget_proxy=30720
    used_tokens_proxy_min=1183
    used_tokens_proxy_max=6703
    marker_present_rate=1.0
    actionable_marker_rate=0.0
    usage bins: low_under_60=5000/5000
    decision=TEACHER_COMPONENT_NO_POSITIVE_UTILITY
  verify_tool:
    verify_action_available_rate=1.0
    student_has_verify_tool=false
    decision=NON_REALIZABLE_ACTION_SPACE_MISMATCH
```

Conclusion:

```text
H1004_COMPONENT_SWEEP_COMPLETE_NO_FORMAL_TRAINING
```

`token_budget_marker` has real 5K support and an engineering OPD_PILOT adapter/reload proof, but the frozen 5K states never reach actionable token-budget pressure. Because Teacher diagnostic utility is not positive, formal PURE_OPD / RL_PLUS_OPD seed42/43 training is stopped by protocol. `verify_tool` is a real Teacher action-space component but is non-realizable for a Student without the verify interface, so Student After PURE_OPD / RL_PLUS_OPD are N/A. No synthetic data or smoke rows were promoted.

Artifacts:

```text
SCAPE-EasyOPD/outputs/component_sweep_0818/h100_4/H1004_COMPONENT_HANDOFF.json
SCAPE-EasyOPD/outputs/component_sweep_0818/h100_4/H1004_TEACHER_BEFORE_DIAGNOSTICS.json
SCAPE-EasyOPD/outputs/component_sweep_0818/h100_4/token_budget_marker/TRAIN_STATES_5K.jsonl
SCAPE-EasyOPD/outputs/component_sweep_0818/h100_4/token_budget_marker/OPD_PILOT/ADAPTER_RELOAD_ACCEPTANCE.json
SCAPE-EasyOPD/outputs/component_sweep_0818/h100_4/verify_tool/TRAIN_STATES_5K.jsonl
SCAPE-EasyOPD/outputs/component_sweep_0818/h100_4/SHA256SUMS
SCAPE-EasyOPD/outputs/component_sweep_0818/master/RUN_MANIFEST.json
SCAPE-EasyOPD/outputs/component_sweep_0818/master/COMPONENT_10_MAIN_TABLE.{csv,md}
```

Master status after this update remains **not paper-grade final**:

```text
MASTER_TABLE_BLOCKED_PHASE_E_INCOMPLETE
```

H100-4 is no longer the blocker. The remaining master blockers are non-H100-4 components whose Teacher/Before/After Phase E metrics are still missing or running, plus component-specific insufficient/non-realizable gates. The master table is a coordination artifact, not a final scientific result.

## 2026-08-19 H100-4 post-sweep infra + capability placement gate

Status: **H1004_POST_SWEEP_INFRA_AND_PLACEMENT_READY**.

Setting:

```text
machine role: H100-4
canonical_student_base: /mnt/songzijun/models/Qwen3-30B-A3B-Instruct-2507
logical_model_id: Qwen3-30B-A3B-Instruct-2507
runtime env: /opt/scape-h1004 via scripts/setup_scape_easyopd_smoke7_env.sh
collector: real Harness-1 bridge, collector_mode=real_harness1
student_inference_privilege: false
components audited: verify_tool, importance_tagging, subtractive_curation, auto_populate_first_search, content_dedup, chunk_neighbors, evidence_graph, sentence_compress, token_budget_marker, adaptive_rerank_instruction
```

Code and contract updates completed:

```text
- Added h1004_post_sweep.py core module for Qwen3 reload audit, handoff discovery, capability placement gate, master table build, and final handoff writeout.
- Added scripts/h1004_validate_qwen3_reload.py.
- Added scripts/h1004_capability_placement_gate.py.
- Added scripts/h1004_discover_component_handoffs.py.
- Added scripts/h1004_build_capability_placement_master.py.
- Added scripts/run_h1004_post_sweep.py.
- Coordination updates were appended to SCAPE实验协调.md.
```

Reload audit result:

```text
Qwen3 base load: pass
native PeftModel.from_pretrained: fail
  root cause: WeightConverter.__init__(distributed_operation) incompatibility in PEFT/Transformers adapter reload path
manual safetensors fallback: pass
LoRA tensors loaded: 384/384
adapter trainable params: 3,342,336
disable/enable output difference: pass
roundtrip logits cosine: 0.999878...
acceptance status: QWEN3_ADAPTER_RELOAD_READY_WITH_COMPAT_FALLBACK
```

Handoff / master discovery result:

```text
available handoffs: 4/4
base blockers: none
collector blockers: none
phase E blockers:
  - importance_tagging: TEACHER_METRIC_REQUIRED_BEFORE_TRAINING
  - auto_populate_first_search: TEACHER_METRIC_REQUIRED_BEFORE_TRAINING
  - evidence_graph: TEACHER_METRIC_REQUIRED_BEFORE_TRAINING
  - sentence_compress: TEACHER_METRIC_REQUIRED_BEFORE_TRAINING
  - adaptive_rerank_instruction: PHASE_E_FOUR_CELLS_RUNNING
master status: MASTER_TABLE_BLOCKED_PHASE_E_INCOMPLETE
```

Final artifacts:

```text
SCAPE-EasyOPD/outputs/component_sweep_0818/h100_4/post_phase_u/H1004_POST_SWEEP_HANDOFF.json
SCAPE-EasyOPD/outputs/component_sweep_0818/h100_4/post_phase_u/qwen3_reload/QWEN3_RELOAD_ACCEPTANCE.json
SCAPE-EasyOPD/outputs/component_sweep_0818/master/RUN_MANIFEST.json
SCAPE-EasyOPD/outputs/component_sweep_0818/master/SHA256SUMS
```

Conclusion:

```text
H1004_POST_SWEEP_INFRA_AND_PLACEMENT_READY
```

Interpretation:

- H100-4 has now completed the post-sweep infrastructure task: Qwen3 reload is audited, placement gating is implemented, and the master table was rebuilt.
- The Qwen3 adapter reload path is not fully native yet; the safe/working contract is the manual safetensors fallback, not the broken native PEFT converter path.
- `token_budget_marker` remains `KEEP_RUNTIME_OR_DROP_COMPONENT`; `verify_tool` remains `KEEP_RUNTIME_PLACEMENT_BOUNDARY`.
- The final master remains blocked by external Phase E incompleteness on H100-1/2/3, so there is still no paper-grade final result.

## 2026-08-21 token_budget_marker Teacher-always-on vs Student-always-off 128-state gain

Status: **completed; no process/utility separation**. Reused the frozen real-pressure cohort `SCAPE/outputs/0820_token_budget_marker_pressure_rebuild/manifests/PRESSURE_STATES_128.jsonl` (128 unique snapshots, SHA-256 `05ddafd1d852d28a4fbc388313e0f06b8be174c69faa176e904b46f3afe4c3ab`). The frozen Teacher/Student first actions were retained and counted as step 1; Teacher then used the Full/component-on view for every remaining continuation step, while Student used the Reduced/component-off view throughout. K4 and K8 each completed 128 paired rows.

| Horizon | First-action disagreement | Tool-cost Δ | Utility Δ |
|---|---:|---:|---:|
| K4 | 0.00% | 0.0 | 0.0000 |
| K8 | 0.00% | 0.0 | 0.0000 |

All deltas are Teacher minus Student. Mean Teacher/Student tool costs were `3.625/3.625` for K4 and `3.6484375/3.6484375` for K8. Audit passed with K4/K8 ordered snapshot identity `128/128`; invalid provenance, snapshot mismatch, trace-length mismatch, branch-level metric formula mismatch, and Full Harness takeover were all zero. Thus extending `token_budget_marker` from once-on to continuation always-on did not alter the frozen first actions and produced no tool-cost or utility gain on this real-pressure cohort. Formal artifacts: `SCAPE/outputs/0821_token_budget_marker_always_on_off_128/scored/TOKEN_BUDGET_MARKER_ALWAYS_ON_OFF_SUMMARY.json` and `TOKEN_BUDGET_MARKER_ALWAYS_ON_OFF_PER_STATE.jsonl`; runner/scorer: `scripts/run_token_budget_marker_formal_fork.py` and `scripts/score_token_budget_marker_always_on_off.py`; runtime `/opt/scape-easyopd-smoke7`.

## 2026-08-21 token_budget_marker real-pressure 128-state recall rerun

Status: **completed / formal recall gate valid; no observable gain**.

The previously invalidated low-pressure cohort was replaced by the frozen real-corpus pressure manifest `SCAPE/outputs/0820_token_budget_marker_pressure_rebuild/manifests/PRESSURE_STATES_128.jsonl`. It contains 128 unique same-state snapshots from 66 queries, with tokenizer-measured real-context usage in three fixed bins: `over_half=43`, `warning=43`, `critical=42`; measured usage ranged from `19080` to `28588` of budget `30720`. All 128 rows had non-empty qrels, and the marker was present in the Teacher first-action view and absent from the Student view.

Setting and provenance:

```text
component: token_budget_marker
cohort: PRESSURE_STATES_128.jsonl (frozen before rerun)
seed: 2214; K4/K8; first forced action counts as step 1
model: /mnt/songzijun/models/pat-jj_harness-1-full/harness-1
runtime: /opt/scape-h1004; attention: flex_attention
normalization: split_at_first_underscore_v1
qrel_sha256: a6f594975be57339de9e4e9f67f13c044f647feda77c0b84c45a1581e3041bd1
continuation: Reduced for both branches; full_harness_takeover=0
```

The formal outputs are `SCAPE/outputs/0821_token_budget_marker_real_pressure_recall_128/scored/TOKEN_BUDGET_MARKER_EVIDENCE_RECALL_SUMMARY.json` and `TOKEN_BUDGET_MARKER_EVIDENCE_RECALL_PER_STATE.jsonl`; raw K4/K8 rows are in `token_budget_marker_K4.jsonl` and `token_budget_marker_K8.jsonl`. Audit passed with `128/128` ordered snapshot match, `invalid_provenance=0`, `missing_or_empty_qrel=0`, and `full_harness_takeover=0`.

```text
K4 candidate recall:  Teacher 2.265625% / Student 2.265625%, delta 0.00 pp
K4 activated recall: Teacher 0.390625% / Student 0.390625%, delta 0.00 pp
K8 candidate recall:  Teacher 2.265625% / Student 2.265625%, delta 0.00 pp
K8 activated recall: Teacher 0.390625% / Student 0.390625%, delta 0.00 pp
K4/K8 paired counts: positive/negative/zero = 0/0/128 for both metrics
paired and query-cluster bootstrap CI95: [0.00, 0.00] pp for both metrics
first-action disagreement: 0.00% (all pressure bins)
successful read-set delta / tool-cost delta / utility delta: +0.0000 / +0.0000 / +0.000000
```

Conclusion: this rerun closes the prior insufficient-pressure validity gap, but the token marker did not alter the first action or any endpoint evidence set on this frozen pressure cohort. The gain-table result is therefore formally recall-neutral and process-neutral, not invalidated for lack of pressure. The earlier diagnostic artifact remains historical only.

## 2026-08-19 H100-1 action/projectable component sweep Phase U

Status: **Phase U ready / Phase E blocked by canonical base availability**.

Completed:

```text
components: auto_populate_first_search, importance_tagging, subtractive_curation
TRAIN_POOL: 2000 unique queries = 446 original train queries + 1554 train-corpus document-grounded synthetic query specs
leakage audit: n_exact_duplicate_queries=0, n_dev_test_query_overlap=0
per component: n_queries_selected=2000, n_rollouts_total=8000, n_unique_event_active=8000, TRAIN_STATES_5K rows=5000, synthetic_row_count=0
collector_mode: real_harness1
model_id contract: openai/gpt-oss-20b
```

Artifacts:

```text
SCAPE-EasyOPD/manifests/COMPONENT_SWEEP_TRAIN_POOL.json
SCAPE-EasyOPD/manifests/COMPONENT_SWEEP_TRAIN_POOL_PROVENANCE.jsonl
SCAPE-EasyOPD/manifests/COMPONENT_SWEEP_TRAIN_POOL_STATS.json
SCAPE-EasyOPD/manifests/COMPONENT_SWEEP_QUERY_LEAKAGE_AUDIT.md
SCAPE-EasyOPD/outputs/component_sweep_0818/h100_1/{auto_populate_first_search,importance_tagging,subtractive_curation}/TRAIN_STATES_5K.jsonl
SCAPE-EasyOPD/outputs/component_sweep_0818/h100_1/H1001_OPD_ROWS_MANIFEST.json
SCAPE-EasyOPD/outputs/component_sweep_0818/h100_1/H1001_COMPONENT_HANDOFF.json
SCAPE-EasyOPD/outputs/component_sweep_0818/h100_1/PHASE_E_BLOCKER_GPT_OSS_BASE.md
```

Phase E was not launched. `openai/gpt-oss-20b` is not locally resolvable by Transformers with `local_files_only=True`; no cached `models--openai--gpt-oss-20b` snapshot was found. Per the 0819 protocol, Qwen or `pat-jj/harness-1` checkpoints must not be substituted for the canonical base. Current handoff status is `H1001_PHASE_U_READY_PHASE_E_BLOCKED` with decision `STOP_GPT_OSS_BASE_UNAVAILABLE`.


## 2026-08-19 H100-3 component sweep Phase U preflight

Status: **framework gate tightened / preflight passed**.

Completed in this turn:

```text
- `scripts/scape_component_opd.py collect` now has explicit `--mode formal|smoke` separation.
- Formal collection requires real `--query-manifest` and `--rollout-manifest` inputs.
- Formal rows are validated for `collector_mode=real_harness1`, `runtime_name=harness1`, and required student-visible fields.
- Collection stats now report `synthetic_row_count`, `runtime_name`, and `model_id=openai/gpt-oss-20b`.
- Existing smoke tests still pass: 8/8 in `tests/methods/test_scape_component_opd_5k_collection.py` and `tests/methods/test_scape_component_opd_training_entrypoint.py`.
- `/opt/scape-easyopd-smoke7/bin/python` imports `easyopd`, `verl`, `torch`, and `transformers` successfully.
- `outputs/component_sweep_0818/preflight/ENVIRONMENT.txt` was written.
- `outputs/scape_easyopd/framework/HARNESS1_RUNTIME_INVENTORY.md` already contains the current Harness-1 entry inventory.
```

Current interpretation:

```text
- The legacy smoke collector is still available only as an explicit smoke path.
- Formal 5K collection is now fenced against synthetic fallback and can only proceed from real Harness-1 rollouts.
- No real on-policy 5K rollout has been started yet in this turn.
```

Next required step: launch the real Harness-1 rollout collection path once the manifest/rollout artifacts are ready, then monitor it and only promote the result if `synthetic_row_count == 0` and the 5K gate is satisfied.


## 2026-08-18 H100-3 SCAPE-EasyOPD framework migration

Status: **framework acceptance complete / `SCAPE_EASYOPD_READY`**. Canonical framework directory: `/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD`; runtime: `/opt/scape-easyopd-smoke7` (no Python/conda runtime created under `/mnt`).

Completed:

```text
EasyOPD registry/config: pass
8 × H100 visible: pass
BF16 matmul on all 8: pass
upstream dry-runs: gkd/sod/opcd pass
SCAPE component contract/loss tests: 19 passed
verl one-step Qwen3-1.7B training smoke: pass
live SCAPE/Harness-1 AgentLoop: pass
real closed-loop evaluator: pass
actual LoRA projected-action update + adapter reload: pass
verify_tool NON_REALIZABLE guard: pass
content_dedup zero-event guard: pass
```

Key artifacts now exist in `SCAPE-EasyOPD/`: `UPSTREAM_LOCK.*`, `FRAMEWORK_SELECTION_AUDIT.md`, `LEGACY_OPD_CODE_AUDIT.md`, `VERL_PATCH_AUDIT.md`, `SCAPE_*_CONTRACT.md`, `COMPONENT_REALIZABILITY_MATRIX.*`, acceptance/test summaries, `RUN_MANIFEST.json`, `STATUS_LIVE.md`, `H1003_SCAPE_EASYOPD_HANDOFF.json`, and `SHA256SUMS`.

Important caveat: this is a **framework acceptance smoke**, not a positive scientific component result. The completed path proves the EasyOPD/verl OPD path, vLLM rollout, FSDP actor, checkpoint save, actual LoRA reload, live SCAPE/Harness multi-turn Search loop, and unified real closed-loop evaluator are runnable. Recommended next component for full-scale experiments is `auto_populate_first_search`, because it has a legal projected-action path and passed the actual LoRA + real closed-loop smoke.

## 2026-08-18 0818 todo snapshot

> 下面这组是当前 `todo/0818` 的最新状态汇总，用于标清楚 **已完成 / 进行中 / 未完成**。旧的历史结果仍保留在后文。

| Task | Status | Setting | Current result | Conclusion |
|---|---|---|---|---|
| H100-1 `PROJECTED_ACTION_AUTO` | **进行中 / 阻塞** | 8×H100；actual LoRA/PEFT；`student_inference_privilege=false`；第一次成功 search 后把 harness 的 `auto_populate_first_search` side-effect 投影为 Student 可执行的 `curate(add_ids=...)`，再做真实 multi-step closed-loop。 | 代码审计已确认旧 AUTO 目标和真实 side-effect 不一致：真实 runtime 里是 search 后由 harness 自动写入 curated set，而不是模型显式发出 `curate`。当前可合法投影的 projected-action 支持为 `0/1024`，没有伪造样本；正式 on-policy 重采集与 8-GPU 训练尚未启动。 | 先恢复 `/opt` ML 环境，再做真实 on-policy 采集与 projection split；当前不能进入 paper-grade GO，属于阻塞态。 |
| H100-2 `PROJECTED_CURATION_BUNDLE` | **未完成 / 需 redesign** | 8×H100；actual model only；`importance_tagging + subtractive_curation` 联合内化；Student 仍然无 privilege，目标是把 `curated_ids_pre -> curated_ids_post` 的 state delta 编译成原生 `curate(add_ids, remove_ids)`。 | 最新 live 状态仍停在 support gate：`42/512` unique states，`valid add ids=42`，`valid remove ids=42`，`terminal reward nonzero=42`，但正式 8-GPU actual-LoRA 阵列未启动。 | 当前结论是 `REDESIGN_ONCE_CURATION_BUNDLE`；若后续仍无法拿到足够支持或闭环增益，则应放弃该 bundle。 |
| H100-3 `RETRIEVAL_HYGIENE_BUNDLE` | **已完成** | 8×H100；actual LoRA；no-privilege Student；联合 `auto_populate_first_search`、`content_dedup`，并诊断 `adaptive_rerank_instruction` 是否带来组合增益。 | Phase 1–3 gate 未证明互补性：`AUTO_DEDUP <= max(AUTO, DEDUP)`，`content_dedup` 触发案例为 0，`rerank` 也没有提升 `AUTO_DEDUP`。后续 actual-LoRA 与 real closed-loop 已跑完，但 DEV/TEST 都未超过 Base。 | 最终结论为 `DISCARD_RETRIEVAL_BUNDLE`。 |
| H100-4 actual-model baselines + novelty guard | **已完成** | actual-model baselines；no-privilege real closed-loop；同时做 novelty collision audit，避免把 harness internalization、privileged/action-only distillation、selective OPD、state-matching、outcome verification、evidence-conditioned self-distillation 误报为新贡献。 | 已完成 OPSD_ACTION_PI、OPHSD_FAITHFUL、MATCHED_TEXT_PRIVILEGE 和 fallback baseline 的实际训练/闭环；SEED/OPID faithful contract 仍受阻并按规范回退。16-query serial real closed-loop 中 Base 最好，所有完成的 adapter 都没有超过 Base。 | 科学结论是没有拿到 `Ours > Base` 或强 baseline 的正结果；新颖性结论是相关 collision hypotheses 仍需逐篇核实，当前不能宣称新机制首创。 |

## 2026-08-18 H100-3 RETRIEVAL_HYGIENE_BUNDLE

Status: **completed all required phases — `DISCARD_RETRIEVAL_BUNDLE`**. Canonical output: `outputs/0818_retrieval_hygiene_bundle/`. `CLAUDE.md` was reread before continuation. The actual PEFT/LoRA matrix, full DEV=128 real closed-loop matrix, and full TEST=112 real closed-loop matrix all completed with eight-way GPU parallelism; final GPU check showed all eight idle and no retrieval-bundle workers remained.

### Contract and artifacts

- Actual LoRA/LLM weights: true; route-head substitution: false.
- Student inference privilege: false.
- Real runtime document ids and executable projected args were used.
- Projection artifacts include `AUTO_PROJECTED_DATA.jsonl`, `DEDUP_PROJECTED_DATA.jsonl`, `AUTO_DEDUP_PROJECTED_DATA.jsonl`, `AUTO_DEDUP_RERANK_PROJECTED_DATA.jsonl`, and `SHUFFLED_BUNDLE_PROJECTION_DATA.jsonl`.
- The first smoke exposed the known Harmony evaluator contract issue (`NoneType.new`); rerunning with `SCAPE_FORCE_LOCAL_HARMONY=1` produced valid real tool calls and `error_rate=0` for every method. The initial parser-failure smoke is not included as scientific evidence.
- Formal TEST manifest: `outputs/0818_retrieval_hygiene_bundle/test_manifest_112.json`, formed from the 128 unique H100-2 real-loop queries after excluding the 16 corrected smoke queries.

### Phase 1–3 gate

```text
source rows: AUTO=1024, content_dedup=1024, matched unique=2048
content_dedup trigger cases at MinHash/shingle threshold 0.82: 0
AUTO_PROJECTED mean reward:              0.452595
DEDUP_PROJECTED mean reward:             0.006397
AUTO_DEDUP_PROJECTED mean reward:        0.452595
AUTO_DEDUP_RERANK_PROJECTED mean reward: 0.452595
SHUFFLED_BUNDLE mean reward:             0.154771
decision: DISCARD_RERANK_USE_DEDUP_GPU45
```

The value gate did not establish complementarity: `AUTO_DEDUP <= max(AUTO, DEDUP)`, and rerank did not improve `AUTO_DEDUP`. The frozen `content_dedup` shard contained no active duplicate event: 1024 rows, 242 unique document ids, zero exact cross-id duplicate text clusters, and zero MinHash-triggered clusters.

### Phase 4 actual-LoRA matrix

Eight cells completed successfully with finite losses and reloadable LoRA adapters:

```text
AUTO        seed42/43:        D_post=1.3150 / 1.2734
DEDUP       seed42/43:        D_post=0.1661 / 0.1993
AUTO_DEDUP  seed42/43:        D_post=1.3472 / 1.1276
SHUFFLED    seed42/43:        D_post=0.4815 / 0.4740
```

### Phase 5 full real closed-loop results

Every DEV/TEST cell completed with `error_rate=0` and `student_inference_has_privilege=false`.

```text
DEV n=128, matched-base reward deltas
AUTO        seed42/43:       -0.109655 / -0.150484
DEDUP       seed42/43:       +0.023288 / +0.004711
AUTO_DEDUP  seed42/43:       -0.128499 / -0.081656
SHUFFLED    seed42/43:       -0.034547 / -0.010992

TEST n=112, paired reward deltas
AUTO        seed42/43:       -0.104089 / -0.136393
DEDUP       seed42/43:       +0.003589 / +0.008973
AUTO_DEDUP  seed42/43:       -0.136393 / -0.068196
SHUFFLED    seed42/43:       -0.030509 / -0.019741
```

Pooled split aggregates:

```text
DEV:  AUTO=-0.115988, DEDUP=+0.018792, AUTO_DEDUP=-0.091782, SHUFFLED=-0.009474
TEST: AUTO=-0.127317, DEDUP=+0.000103, AUTO_DEDUP=-0.107576, SHUFFLED=-0.028612
```

The full result does not satisfy the required GO conditions: `AUTO_DEDUP` is below matched Base on both splits and both seeds; it is also below AUTO-only and shuffled controls. DEDUP has small positive deltas but does not establish bundle complementarity, and its source has no active duplicate-trigger events.

### Paired bootstrap and mechanism conclusion

Full per-query paired bootstrap artifacts are in `PAIRED_BOOTSTRAP.csv`. TEST 95% intervals for AUTO and AUTO+DEDUP remain strictly negative; DEDUP intervals are small and do not rescue the bundle claim. Mechanism metrics are in `RETRIEVAL_MECHANISM_METRICS.csv`; no simultaneous improvement in earlier curation, duplicate reduction, and unique relevant evidence was established. The allowed event-conditioned redesign was audited once, but zero real dedup-trigger rows means generating a new training wave would require fabrication and is prohibited.

```text
DISCARD_RETRIEVAL_BUNDLE
```

Handoff: `outputs/0818_retrieval_hygiene_bundle/H1003_0818_HANDOFF.json`. Full aggregates: `RETRIEVAL_FULL_SPLIT_AGGREGATE.csv`. Checksums: `outputs/0818_retrieval_hygiene_bundle/SHA256SUMS`.

## 2026-08-18 H100-4 actual-model baselines / novelty guard

Status: **completed actual-model baseline run with closed-loop completion and fallback replacement**. The approved `/opt/scape-h1004` runtime was restored, all eight H100 GPUs were exercised, and six actual HF/PEFT LoRA cells completed on the real 512/128 train-valid contract: OPSD_ACTION_PI seeds 42/43, OPHSD_FAITHFUL seeds 42/43, and MATCHED_TEXT_PRIVILEGE seeds 42/43. SEED/OPID remains blocked by missing faithful Search skill-analyzer/adaptation contract, so GPU6/7 were repurposed per spec to closest faithful SMRC-SD / OVCSD actual-model fallback cells; both fallback cells trained and completed n=16 real closed-loop evaluation.

Canonical output: `outputs/0818_actual_baselines_novelty/`. The final status files and run manifest now record `training_complete_closed_loop_complete`. The six completed actual-model rows are adapter-only and no-privilege at inference. A serial real-closed-loop smoke over the six adapters ran on one query (`query_id=471`) with `max_steps=6`; Base and all six adapters completed the episode, all at `overall_reward=0.001`, so there was no win over Base in that smoke. A later 16-query serial real closed-loop run completed for Base plus all six adapters; Base achieved `overall_reward=0.14961805555555555`, while the best adapter tied at `-0.024125`, so the completed closed-loop result is still negative for every adapter relative to Base. A prior 16-query parallel smoke was stopped after shared local Chroma stalled, and those partial results were not promoted.

The existing collision guard is preserved: do not claim first harness internalization, first privileged/action-only distillation, first selective/state-conditioned OPD, first state-aligned correction, first evidence-conditioned search self-distillation, or first privileged-information representation. C1/C2/C3 remain pending full paper-level audit; no novelty claim is made.

Required 0818 outputs and SHA256 are under the canonical output directory, including `RUN_MANIFEST.json`, `STATUS_LIVE.md`, the six `TRAINING_SUMMARY.json` cell artifacts, the route-level fallback summaries, and `eval/all_six_serial_n1/REAL_CLOSED_LOOP_HANDOFF.json`. Faithful actual-model baselines are now runnable on `/opt`; the remaining scientific question is whether a broader real-closed-loop evaluation can beat Base and the stronger historical references.

## 2026-08-18 H100-2 PROJECTED_CURATION_BUNDLE

Status: **blocked before formal training**. Canonical output: `outputs/0818_projected_curation_bundle/`.

### Setting

```text
experiment: PROJECTED_CURATION_BUNDLE
required target states: 512
actual collected states: 42
valid add ids: 42
valid remove ids: 42
terminal reward nonzero: 42
student inference privilege: false
```

### Current result

- The collect/evaluator repair pass produced a consistent low-support corpus with valid add/remove ids and nonzero terminal reward.
- The formal gate failed on support: `42/512` unique states, so the 8-GPU actual-LoRA matrix was not launched.
- The `/opt` ML runtime is still missing: `/opt/scape-hf-scorer/bin/python`, `/opt/scape/bin/python`, and `/opt/scape-venv/bin/python` do not exist; system Python also lacks `torch`, `transformers`, and `peft`.
- The new resumable orchestrator wrote a consistent `H1002_PROJECTED_CURATION_BUNDLE_0818_HANDOFF.json`, `RUN_MANIFEST.json`, and `STATUS_LIVE.md` that supersede the contradictory older discard text in the same output directory.
- Eight H100 GPUs are visible and idle, but no valid actual-model training can start until an approved `/opt` environment is restored and the support gate is met.

### Decision

```text
REDESIGN_ONCE_CURATION_BUNDLE
```

### Next required step

Restore the approved `/opt` ML environment, recollect to the formal 512-state target, then rerun the gate and only after that launch the 8-way training matrix and closed-loop evaluation.

## 2026-08-18 H20 clean-init AUTO OPD (`h20_clean_auto_0817`)

Status snapshot: **正在进行** (Phase G reload-fix real closed-loop). Machine `8×H20`. Repo `/data/ppnm/Capability_Evolution/SCAPE`. Spec: `todo/0817/H20_clean_init_AUTO_OPD_next_round_20260817.md`. Canonical outputs: `outputs/h20_clean_auto_0817/`. This is the H20-only cross-initialization line; it does **not** repeat H100-1/2/3/4 jobs.

Do **not** treat `H20_CLEAN_AUTO_HANDOFF.json` dated `2026-08-18T00:51:01+0800` (`STOP_CLEAN_AUTO_REAL_TASK_NO_GAIN`) as the paper-grade decision. That handoff was written from a broken evaluator load (OPD LoRA stacked on raw `gpt-oss-20b` without merging Clean-SFT). Reload-fix (`gpt-oss → merge FULL_S42 → OPD LoRA`) is the current contract; DEV/TEST under that contract are still running.

### Setting

```text
init:            openai/gpt-oss-20b + Harness-1 public SFT only (not released pat-jj/harness-1)
CLEAN_AUTO_BASE: CLEAN_FULL_S42
                 outputs/0814_clean_mechanism/sft/gpu0/full_s42_full/lora_checkpoint
component:       auto_populate_first_search
objective:       reverse 8-way Route-KL; lambda_args=0; lambda_anchor=0.05
LoRA:            actual gpt-oss LLM weights, r=8, lr=1e-5, 1 epoch
Student input:   reduced / no privilege
inference:       student_inference_privilege=false
retriever:       LOCAL_COMPAT_ONLY in-process overlap ranker over per-query doc_store
                 (not official Chroma)
reward:          evidence/qrel curated recall; final_answer gold = N/A (not written as 0)
max_steps:       6 (sanity 10 in-flight; 12 not started)
splits:          BASE_EVAL n=128; AUTO unique states=438 (train 350 / valid 43 / test 45);
                 real DEV=128; real TEST=112 (all remaining unique queries, no resampling)
seeds:           unshuffled 42/43/44/45; shuffled-target 42/43/44/45 matched budget
```

Route space is fixed to: `fan_out_search, search_corpus, grep_corpus, read_document, review_docs, curate, verify, end_search`.

### Results (current, incomplete Phase G)

#### Step A — Harmony / Base Gate `[已完成]`

Previous 0814 n=4 smoke (`parse≈0.75`) was a **prompt/parser contract bug**, not proof that FULL SFT never learned Harmony. Official Harmony `build_context` + `render_conversation_for_completion`, stop on `<|call|>`/`<|return|>`:

```text
parser contract tests: 4/4 pass
  canonical Harmony tool call -> pass
  canonical end_search        -> pass
  malformed tool name         -> fail_legal
  analysis prose              -> unparsed

BASE_EVAL_128 (n=128, same manifest):
  RAW_GPT_OSS      parse=0.047  legal=0.039  invalid=0.961  gate=FAIL
  CLEAN_FULL_S42   parse=1.000  legal=1.000  invalid=0.000  gate=PASS  (CLEAN_AUTO_BASE)
  CLEAN_FULL_S43   parse=1.000  legal=1.000  invalid=0.000  gate=PASS
  CLEAN_TOOL_S42   parse=0.727  legal=0.688  invalid=0.312  gate=FAIL
  CLEAN_TOOL_S43   parse=0.992  legal=0.953  invalid=0.047  gate=FAIL
```

FULL first-action mass is almost entirely `fan_out_search`/`search_corpus` (search coverage=128/128). TOOL remains diagnostic only.

#### Step B — FORMAT_REPAIR `[未开始 / 按规范跳过]`

Base Gate already PASS on FULL s42/s43. Spec forbids another FULL/TOOL SFT and forbids FORMAT_REPAIR unless the gate fails. `FORMAT_REPAIR_TRAINING.csv` / `FORMAT_REPAIR_EVAL.csv` record this skip.

#### Step C — fresh AUTO on-policy data `[已完成]`

```text
n_raw_unique=438   (target was >=512; below target, no duplicated rows)
n_train=350  n_valid=43  n_test=45
n_effect_active=390
n_resampled_duplicate=0
query_disjoint_from_base_eval=true
used_rl=false
```

#### Step D — same-`xi_t` value K4/K8 `[已完成]`

```text
K4 n=350 mean=+0.0254  CI=[+0.0043, +0.0538]
K8 n=350 mean=+0.0115  CI=[-0.0102, +0.0323]
effect-active K4 mean=+0.0232 ci_low=-0.0060
effect-active K8 mean=+0.0079 ci_low=-0.0179
replay_noise_proxy=0
direction_consistent=true
AUTO_CLEAN_VALUE_GATE.pass=true
```

Caveat: K8 95% CI includes 0. Gate pass rests on K4 `CI_low>0`, both means `> replay_noise`, and K4/K8 sign agreement — not on K8 excluding 0.

#### Step E/F — actual gpt-oss LoRA OPD + shuffled-target control `[已完成]`

All 8 cells finite loss, reloadable LoRA (not `route_head.pt`):

```text
UNSHUFFLED Route-KL
  s42 loss=0.113  d_post=-0.0012  L_m=0.347
  s43 loss=0.109  d_post=+0.0019  L_m=2.017
  s44 loss=0.091  d_post=+0.0011  L_m=1.571
  s45 loss=0.128  d_post=+0.0026  L_m=2.376

SHUFFLED Route-KL (same states/query ids/update budget/marginal/loss/LoRA/lr/epochs/seeds)
  s42 loss=0.261  d_post=-0.0015  L_m=0.196
  s43 loss=0.210  d_post=+0.0005  L_m=1.250
  s44 loss=0.233  d_post=+0.0058  L_m=4.080
  s45 loss=0.216  d_post=+0.0003  L_m=1.150
  shuffle fixed points: 1/350 = 0.0029
```

Same-state `L_m` / `d_post` are **not** the paper main result.

#### Step G — actual-model real multi-step closed-loop `[正在进行]`

Frozen contract: `real_eval/AUTO_CLEAN_REAL_EVAL_CONTRACT.md`. Reload-fix stacks OPD LoRA on merged Clean-SFT FULL s42. Smoke passed tool-channel sanity (`invalid` 0.94 → 0.052).

16-query smoke `[已完成]`:

```text
SMOKE_BASE   n=16  recall=0.2063  invalid=0.0729  search=3.31
SMOKE_UNSH   n=16  recall=0.1300  invalid=0.0521  search=2.94   (unshuffled s42)
privilege=false; LoRA actually loaded; sequences Base ≠ Student; scorer non-constant
```

DEV n=128, max_steps=6, reload-fix (evidence/qrel recall):

```text
[已完成]
  CLEAN_BASE                 0.1913  invalid=0.0638  search=2.84
  CLEAN_FULL_HARNESS         0.1913  invalid=0.0638  search=2.84   (same FULL_S42 weights as CLEAN_BASE)
  AUTO_CLEAN_UNSHUFFLED_s43  0.1714  invalid=0.0273  search=2.69
  AUTO_CLEAN_UNSHUFFLED_s44  0.1710  invalid=0.0443  search=2.75
  AUTO_CLEAN_UNSHUFFLED_s45  0.1663  invalid=0.0312  search=2.65
  AUTO_CLEAN_SHUFFLED_s42    0.1745  invalid=0.0612  search=2.82
  AUTO_CLEAN_SHUFFLED_s43    0.1998  invalid=0.0638  search=2.71
  AUTO_CLEAN_SHUFFLED_s44    0.1866  invalid=0.0508  search=2.81

[正在进行]
  AUTO_CLEAN_UNSHUFFLED_s42  DEV  ~121/128
  AUTO_CLEAN_SHUFFLED_s45    DEV  ~10/128
```

Interim DEV readout (3/4 unshuffled + 3/4 shuffled; **not** a frozen main-table row):

```text
CLEAN_BASE                         0.1913
unshuffled mean (s43/s44/s45)      0.1696   all three < Base
shuffled mean (s42/s43/s44)        0.1870   s43 0.1998 > Base
student_beats_clean_base           false on completed unshuffled seeds
unshuffled_beats_shuffled          false on completed seeds
invalid-tool                       no material regression (Student 0.027–0.044 vs Base 0.064)
termination                        all completed DEV rows hit max_steps (no early end_search)
final_answer                       N/A
```

TEST n=112, max_steps=6:

```text
[已完成]  CLEAN_BASE_TEST  recall=0.1497  invalid=0.0625  search=2.66
[正在进行] unsh_s43_test ~11/112, unsh_s44_test ~15/112, unsh_s45_test ~9/112, sh_s42_test ~14/112, sh_s43_test ~10/112
[未开始]  unsh_s42_test, sh_s44_test, sh_s45_test, CLEAN_FULL_HARNESS_TEST
```

Sanity max_steps=10/12:

```text
[已完成]   CLEAN_BASE_S10 n=16 recall=0.2140 invalid=0.0750 search=4.44  (vs smoke max_steps=6 recall=0.2063; still all max_steps termination)
[未开始]   UNSH_S43_S10; max_steps=12
```

### Conclusion (interim — Phase G not closed)

```text
A Base Gate PASS                         true   (FULL s42/s43 parse=1.0)
B AUTO value K4/K8 positive              true   (gate pass; K8 CI includes 0)
C Student > clean Base on real loop      false so far on completed reload-fix DEV seeds
D >=2 unshuffled seeds same + direction  false so far (all completed unshuffled < Base)
E unshuffled > shuffled on real loop     false so far
F invalid-tool no material regression    true so far
G student_inference_privilege=false      true
```

Current scientific reading, pending remaining DEV/TEST:

- Clean gpt-oss **tool channel is recoverable** from public SFT + correct Harmony contract. The 0814 `CLEAN_BASE_BLOCKED` / parse≈0.75 result does not survive the repaired evaluator.
- AUTO same-state value transfers to clean occupancy at least as a small positive K4 signal.
- After stacking OPD LoRA on the real Clean-SFT base, completed unshuffled Students do **not** beat CLEAN_BASE on real multi-step Search; shuffled s43 is currently the highest DEV recall among finished cells.
- This is **not** yet `CLEAN_INIT_AUTO_TRANSFER_PASS`. It is also **not** a frozen `STOP_CLEAN_AUTO_REAL_TASK_NO_GAIN` until remaining seeds + TEST + paired bootstrap finish. Spec still allows **one** substantive redesign if same-state metrics move but real closed-loop does not.

Do not promote: FORMAT_REPAIR skip, same-state `L_m`, value-positive, or the 00:51 broken-load recall=0 run as the final positive or negative main result.

### Completed / in-progress / not-started board

| Item | Status |
|---|---|
| Harmony runtime audit + parser tests | **已完成** |
| BASE_EVAL_128 five-way n=128 | **已完成** |
| FORMAT_REPAIR FR_A–D | **未开始** (Base Gate PASS, skipped by spec) |
| Fresh AUTO collect / split / privilege schema | **已完成** (438 unique < 512 target) |
| Value K4/K8 fork-replay + gate | **已完成** |
| Unshuffled LoRA seeds 42–45 | **已完成** |
| Shuffled-target LoRA seeds 42–45 | **已完成** |
| Real-eval contract freeze | **已完成** |
| 16-query smoke Base vs best-unsh (reload-fix) | **已完成** |
| DEV n=128 CLEAN_BASE / FULL_HARNESS / unsh 43–45 / sh 42–44 | **已完成** |
| DEV unsh s42, sh s45 | **正在进行** |
| TEST n=112 CLEAN_BASE | **已完成** |
| TEST unsh 43–45, sh 42–43 | **正在进行** |
| TEST unsh s42, sh 44–45, FULL_HARNESS | **未开始** |
| max_steps=10 Base smoke | **已完成** (recall 0.214 vs 0.206 at 6 steps) |
| max_steps=10 Student / max_steps=12 | **未开始** |
| Paired bootstrap 95% CI (reload-fix) | **未开始** (existing CSV is from broken load) |
| Case analysis 20–25 per contrast class | **未开始** (stub only) |
| Final GO / one redesign / STOP | **未开始** (waiting remaining G) |
| `result-record.md` append | **未开始** |
| Paper main-table recommendation | **未开始** (`recommended_for_main_table=false` until G closes) |

Artifacts root: `outputs/h20_clean_auto_0817/` (`RUN_MANIFEST.json`, `STATUS_LIVE.md`, `base_recovery/`, `auto_data/`, `value/`, `training/`, `real_eval/`).

## 2026-08-17 0816-2 final summary


Status: completed in main checkout `/mnt/songzijun/Capability_Evolution/SCAPE`. This is the concise end-state summary for the 0816-2 round.

## 2026-08-18 H100-2 PROJECTED_CURATION_BUNDLE

Status: **completed as discard**. The bundle was audited against the current `importance_tagging + subtractive_curation` evidence set and closed as `DISCARD_CURATION_BUNDLE`, not GO or redesign.

### Setting

```text
output root: outputs/0818_projected_curation_bundle/
student_inference_privilege: false
inputs checked:
  - outputs/0818_projected_action_auto/RUN_MANIFEST.json
  - outputs/btp_h100_3_subtractive_audit_0816_final/H1003_SUBTRACTIVE_AUDIT_HANDOFF.json
  - outputs/h100_2_candidate_b_live_utility/CANDIDATE_B_LIVE_HANDOFF.json
  - outputs/h100_2_structured_privilege_formal_0816/H1002_STRUCTURED_PRIVILEGE_HANDOFF.json
```

### Evidence summary

- `importance_tagging` live utility remains negative in the true live fork/replay gate.
- `subtractive_curation` does not provide a stable positive utility signal and the subtractive audit still reports missing terminal gold/reference contract and zero valid remove-id supervision.
- The AUTO-style projected-action path does not yield trainable projected curate rows for curation, and the current evidence set has no usable `curated_ids_pre -> curated_ids_post` delta for a bundle-level `curate(add_ids, remove_ids)` target.
- No terminal reward contract is available in the audited same-state rows, so real closed-loop bundle training would be unsupported.

### Decision

```text
DISCARD_CURATION_BUNDLE
```

### Interpretation

- Do not launch LoRA or real closed-loop training from this evidence set.
- Do not reframe this as a redesign win; the current data contract is insufficient for a projected curation bundle.
- The next required step would be evaluator/data-contract repair and recollection of real curate-event-positive rows with valid add/remove ids and terminal gold/reference fields.


### Setting

- Main branch focus: `H100-1` AUTO actual-LoRA real closed-loop, `H100-2` importance_tagging proper K4/K8 gate, and H100-4 baseline reconciliation for Full Harness / Matched Text / OPHSD under the same actual-model contract.
- Shared contract target: actual Student model weights only, `student_inference_privilege=false`, real closed-loop execution, query-disjoint splits, matched max steps / reward / retriever / termination rules where applicable.
- Environment rule: GPU-heavy work used `/opt` Python/torch environments; no `/mnt` torch runtime for the formal finalizers.

### Results

- `H100-1 AUTO actual-LoRA`: actual LoRA weights and no-privilege inference were confirmed, but real closed-loop failed the gate. Paper-grade result: `student_beats_base=false`, `unshuffled_beats_shuffle=false`, `real_closed_loop_pass=false`.
- `H100-2 importance_tagging`: proper same-`xi_t` K4/K8 fork/replay completed with 2048 rows, but the formal gate failed. Paper-grade result: `proper_K4_positive=false`, `K8_direction_consistent_positive=false`, `gate_passed=false`.
- `H100-3 Structured`: current evidence remains route-head parity only. The safe claim is `Structured ~= Matched Text > Base` at route-level diagnostic granularity; there is no supported `Structured > Matched Text` claim from the available artifacts.
- `H100-4 baselines`: Full Harness same-contract reference is available only as a reference row; Matched Text and OPHSD remain route-level only and are not valid actual-LoRA main-table wins under the current contract.

### Conclusion

- The 0816-2 round does not produce a positive actual-model win for AUTO or importance_tagging.
- `importance_tagging` is not a valid second positive component under the formal fork gate and should not be launched into LoRA OPD from this artifact set.
- Structured privilege remains parity-like, not a superiority result.
- The remaining useful output is contract clarification and negative evidence: route-proxy gains, actual-LoRA gates, and baseline/reconciliation artifacts, not a new positive main-table claim.

### Completed / verified

#### Runtime

Allowed `/opt` runtime is available:

```text
/opt/scape-hf-scorer/bin/python
Python 3.12.13
torch 2.13.0+cu130
transformers 5.15.0
peft 0.20.0
pyserini: not installed
```

This satisfies the no-`/mnt` environment requirement for GPU/HF/PEFT workloads. `scripts/run_h100_2_live_fork_replay.py` has a deterministic local-corpus fallback when Lucene/pyserini is unavailable.

#### H100-1 AUTO actual-LoRA real closed-loop

Canonical controlling output directories:

```text
/mnt/songzijun/Capability_Evolution/SCAPE/outputs/h100_1_auto_lora_handoff_diagnostics_20260817/
/mnt/songzijun/Capability_Evolution/SCAPE/outputs/btp_h100_4_baselines/h1001_actual_lora_sources/
```

Verified setting:

```text
actual_model_weights: true
student_inference_privilege: false
runtime: real executed BrowseComp+ BM25/chroma compatibility environment
candidate recipe: auto_populate_first_search relevant reverse Route-KL LoRA
paper-grade closed-loop scale: 256 queries
```

Paper-grade actual-model result:

```text
base_student                         reward=0.367756 trajectory_recall=0.152731 final_answer_recall=0.137915
AUTO relevant reverse-KL seed44       reward=0.082421 trajectory_recall=0.104482 final_answer_recall=0.090532
legacy shuffled control seed42        reward=0.362691 trajectory_recall=0.144899 final_answer_recall=0.139478
first-turn-only control seed42        reward=0.387497 trajectory_recall=0.146082 final_answer_recall=0.143248
invalid_tool_rate: 0.0 for all rows
```

Decision:

```text
DISCARD_RECIPE_FOR_MAIN_TABLE
student_beats_base: false
unshuffled_beats_shuffle: false
recommended_for_main_table: false
```

Interpretation for other servers/agents:

- The current AUTO OPD recipe has actual LoRA weights and no-privilege inference, but fails the real closed-loop gate.
- Do not promote same-state route-proxy results or the current AUTO actual-LoRA recipe to the main table as a positive result.
- A query-variant/repeat-action redesign smoke was attempted and still produced identical rewards; current instruction is to move to a substantively different OPD target/component rather than expand this recipe.
- H100-1 diagnostics SHA256 verification passed.

#### H100-2 importance_tagging proper K4/K8

Canonical output directory:

```text
/mnt/songzijun/Capability_Evolution/SCAPE/outputs/0816_2_importance_proper_formal_0817/
```

Formal contract and scale:

```text
same xi_t; importance ON first branch vs OFF reduced branch
both continuations use reduced policy
no full-harness takeover
2 seeds x K4/K8 x 512 states = 2048 rows
```

Gate result:

```text
seed8423 K4 mean_T_minus_S=-0.010283 ci95=[-0.013769, -0.006797]
seed8423 K8 mean_T_minus_S=-0.016494 ci95=[-0.021989, -0.010999]
seed8424 K4 mean_T_minus_S=-0.008408 ci95=[-0.011749, -0.005067]
seed8424 K8 mean_T_minus_S=-0.013740 ci95=[-0.018943, -0.008537]
proper_k4_positive: false
k8_direction_consistent_positive: false
gate_passed: false
```

Decision:

```text
do_not_start_importance_lora_opd
importance actual-LoRA real closed-loop: not_started_gate_blocked
recommended_for_main_table: false
```

The earlier approximate positive importance signal is not sufficient for this round. The formal directory SHA256 verification passed.

#### H100-3 structured parity / route diagnostic

Status remains as previously recorded: `Structured ~= Matched Text > Base` on the real BM25 route-head diagnostic, but it is auxiliary route-level only and not actual LLM Student weights. Do not claim `Structured > Textual` from the current artifacts.

#### H100-4 end-to-end baselines and novelty guard

Canonical output directory:

```text
/mnt/songzijun/Capability_Evolution/SCAPE/outputs/btp_h100_4_baselines/
```

Generated required gap-aware deliverables include:

```text
END2END_BASELINE_PROTOCOL.md
FULL_HARNESS_REAL_CLOSED_LOOP.csv
MATCHED_TEXT_LORA_TRAINING.csv
MATCHED_TEXT_REAL_CLOSED_LOOP.csv
OPHSD_LORA_TRAINING.csv
OPHSD_REAL_CLOSED_LOOP.csv
STANDARD_OPSD_STATUS.md
END2END_MAIN_TABLE.csv/md
END2END_PAIRED_BOOTSTRAP.csv
END2END_COMPUTE_COST.csv
NOVELTY_MATRIX_20260816_LATE.md
NOVELTY_RED_LINES_LATE.md
BASELINE_GAP_LATE.md
H1004_END2END_BASELINE_HANDOFF.json
```

Main readout after corrected required-gap audit:

```text
Full Harness / Base full-modules Harness-1: completed same-contract vLLM test256, reward=0.367756
Ours AUTO Component OPD: completed actual LoRA real closed-loop, failed gate, reward=0.082421
Shuffle control: completed actual LoRA real closed-loop, reward=0.362691
First-turn-only control: completed actual LoRA real closed-loop, reward=0.387497
Matched Text OPD: actual-LoRA not run, training contract missing
OPHSD-style: route-level only, actual-LoRA training contract missing
```

Correction:

- `/mnt/songzijun/CLAUDE.md` was read and the contract-level closure lesson was applied.
- The prior `Base Student` row from H100-1 is not a reduced Base row: its manifest uses `modules_full_v2.yaml` with evidence_state, verification, and context_budget enabled, `policy_backend=vllm`, `model_path=/mnt/songzijun/models/pat-jj_harness-1-full/harness-1`, `n=256`, and reward `0.367756`. It is now recorded as the local BM25 Full Harness / full-modules Harness-1 reference.
- A separate reduced no-privilege Base Student same-contract row is not available from these artifacts.
- Ours vs Full Harness: AUTO actual LoRA does not beat the full-modules reference.
- Ours vs Shuffle: AUTO actual LoRA does not beat shuffled control.
- Ours vs Matched Text / OPHSD: not comparable as actual LoRA; available baselines are blocked or route-level only.

H100-4 corrected artifacts were regenerated by `scripts/finalize_h1004_required_gaps_0817.py`; `SHA256SUMS` was refreshed.

### Current not-started / gap-blocked work

- Do not launch formal TEST/shuffle for the failed AUTO recipe as a positive main-table run.
- Do not launch importance LoRA/real closed-loop from the failed proper K4/K8 gate.
- Reduced Base Student same-contract row is not separately available because the prior Base label used full modules.
- Matched Text actual-LoRA remains blocked: `matched_v2_pairs.jsonl` has textualized state-time fields and reduced prompts, but not the teacher/ref route distribution plus optimizer/update-budget contract required by an actual HF/PEFT LoRA trainer.
- OPHSD actual-LoRA remains blocked: existing OPHSD artifacts are route-level `route_head.pt` cells, not reduced-prompt plus whole-harness-teacher actual LoRA data.
- H100-3 actual-LoRA structured V1/V2 should only continue if a valid actual-LoRA training/evaluation contract is implemented; two failed substantive redesigns should end the structured-superiority branch.

## 2026-08-16 H100-3 subtractive_curation all-zero root-cause audit

Status: completed in main checkout `/mnt/songzijun/Capability_Evolution/SCAPE`; formal retry/training was intentionally not launched because the audit blocks it.

Canonical output directory:

```text
/mnt/songzijun/Capability_Evolution/SCAPE/outputs/btp_h100_3_subtractive_audit_0816_final/
```

Task scope:

- Follow-up to the 2026-08-15 `subtractive_curation` closed-loop gate where Base, Full/Student-style metrics, and best Student all reported `curated_evidence_recall=0`, `overall_reward=0`, and `final_answer_recall=0`.
- Goal was to determine whether the all-zero result is an evaluator/data/runtime-contract error or a real component failure.
- This run is an offline contract/root-cause audit, not a new OPD training run.
- Inputs were the current same-state subtractive test artifact and influence rows:
  - `outputs/true_scape_candidate_b_tournament/data/subtractive_curation_TEST_512.jsonl`
  - `outputs/h100_3_real_influence/REAL_INFLUENCE_PER_STATE.jsonl`
- Audited first `256` rows for oracle sanity and event coverage; case-level replay summary sampled `100` states.
- Heavy GPU training was not used because the spec only permits retraining after a verified evaluator/data repair or recollection of curate-event-positive data.

Oracle/evaluator sanity result:

```text
decision: EVALUATOR_NOT_CONSTANT_ZERO_BUT_REAL_TEST_LACKS_TERMINAL_GOLD_CONTRACT
rows audited: 256
synthetic base curated_evidence_recall:   0.000000
synthetic oracle curated_evidence_recall: 1.000000
real rows with documents:                 256/256
real rows with gold/reference contract:   0/256
student/parsed curate action rate:        0.113281
```

Event/argument coverage result:

```text
documents_nonempty:          256 / 256 = 1.000000
curated_ids_nonempty:        256 / 256 = 1.000000
teacher_curate_action:        29 / 256 = 0.113281
valid_add_ids:                29 / 256 = 0.113281
valid_remove_ids:              0 / 256 = 0.000000
event_active:                256 / 256 = 1.000000
value_positive:                0 / 256 = 0.000000
terminal_reward_available:     0 / 256 = 0.000000
```

Case-analysis result:

```text
cases analysed: 100
missing_terminal_gold_contract: 100
non_curate_route:                84
```

Code/audit finding:

- The scorer is not intrinsically constant-zero: the synthetic oracle curate action changes recall from `0.0` to `1.0`.
- The current real same-state test rows expose documents and curated ids, but do not expose terminal gold/reference fields required for terminal reward or final-answer recall.
- Current rows contain some valid `add_ids`, but no valid `remove_ids`; pointer/add-remove training on this artifact is not defensible.
- The named 0816 spec scripts `run_btp_subtractive.py`, `prepare_btp_subtractive_training.py`, `eval_btp_subtractive_closed_loop.py`, `train_route_opd.py`, and `scape/training/route_opd.py` are not present in this main checkout; this main-checkout audit uses `scripts/run_h100_3_subtractive_audit.py` and synthetic contract tests.
- The audit script checksum generation order was fixed so final `RUN_MANIFEST.json` and `STATUS_LIVE.md` are written before `SHA256SUMS` is generated.

Conclusion:

```text
REDESIGN_DATA_EVALUATOR_CONTRACT_BEFORE_RETRY
allowed_formal_retry: false
```

Interpretation for other servers/agents:

- Do **not** treat the previous all-zero closed-loop as proven component failure. The more precise diagnosis is missing evaluator/data contract for terminal scoring on the current same-state artifact.
- Do **not** launch the allowed one-time `value_weighted_route_kl` seeds `42,43,44,45` retry yet. The 0816 spec only permits this after an evaluator/data bug is fixed or after new true curate-event-positive data is collected.
- Do **not** train pointer/add-remove objectives from the current artifact. Valid pointer supervision is insufficient: `valid_remove_ids=0`, and terminal gold/reference is absent.
- If revisiting subtractive curation, first repair/recollect data with all of: curation event active, nonempty documents, meaningful teacher curate or route/gate decision, valid add/remove ids if pointer training is intended, and gold/reference fields available for terminal closed-loop metrics.
- A route/gate-only redesign may be defensible only after a repaired closed-loop evaluator can score nonzero terminal outcomes.
- Downstream writeups should update the 2026-08-15 `STOP subtractive_curation` wording: broad/unrepaired subtractive remains stopped, but the all-zero closed-loop itself is not a valid final failure signal until the evaluator/data contract is repaired.

Verification performed:

```text
pytest tests/test_subtractive_evaluator_nonzero_oracle.py \
       tests/test_subtractive_state_restore.py \
       tests/test_subtractive_curate_action_reachable.py \
       tests/test_subtractive_argument_id_contract.py -q
# 4 passed

cd outputs/btp_h100_3_subtractive_audit_0816_final && sha256sum -c SHA256SUMS
# all OK
```

Required artifacts written:

```text
RUN_MANIFEST.json
STATUS_LIVE.md
SUBTRACTIVE_ORACLE_SANITY.md
SUBTRACTIVE_ORACLE_SANITY.json
SUBTRACTIVE_EVENT_COVERAGE.csv
SUBTRACTIVE_ARGUMENT_ROOT_CAUSE.md
SUBTRACTIVE_CODE_AUDIT.md
SUBTRACTIVE_ZERO_CASE_ANALYSIS.md
SUBTRACTIVE_ZERO_CASES.jsonl
REPAIRED_CLOSED_LOOP_RESULTS.csv
REPAIRED_CLOSED_LOOP_RESULTS.md
SUBTRACTIVE_REDESIGN_MANIFEST.json
H1003_SUBTRACTIVE_AUDIT_HANDOFF.json
SHA256SUMS
```

Synthetic tests present in this checkout:

```text
tests/test_subtractive_evaluator_nonzero_oracle.py
tests/test_subtractive_state_restore.py
tests/test_subtractive_curate_action_reachable.py
tests/test_subtractive_argument_id_contract.py
```

Operational notes:

- Outputs are under `/mnt/songzijun/Capability_Evolution/SCAPE/outputs/`, following the global result/log location convention.
- No SCAPE/subtractive audit workers were left running after completion.
- GPU-heavy workloads were not started because the audit decision explicitly blocks formal retraining.
- Existing unrelated dirty/deleted files in the SCAPE working tree were not reverted or overwritten.

## 2026-08-16 H100-2 Beyond Textual Privilege auto_populate_first_search distillation

Status: completed in worktree `/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-2/SCAPE`.

Canonical output directory:

```text
/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-2/SCAPE/outputs/btp_h100_2_auto_populate/
```

Task scope:

- Component: `V8D_AUTO_POPULATE_FIRST_SEARCH` / `auto_populate_first_search`.
- Goal was to test the second Beyond Textual Privilege component: whether high-level first-search initialization/search-control privilege can be distilled into a no-privilege Student using released `pat-jj/harness-1`, without waiting for the H20 clean SFT checkpoint.
- This run intentionally distills route/tool-name behavior only. It does not distill query text, document ids, JSON arguments, or search-result text.
- All outputs remain local BM25/HF compatibility only: `LOCAL_COMPAT_ONLY=true`, `official_chroma_parity=false`. Do not describe these as official Harness-1 Chroma parity results.
- GPU-heavy Python used `/opt/vllm-qwen3-1.7b-harness/bin/python`; no `/mnt` JuiceFS torch environment was used for training/eval.
- H100-2 had 8 idle H100-class GPUs and used them for the first 8-cell training matrix; no related experiment processes remained after completion and all 8 GPUs were idle.

Value-confirm setting:

- Experiment: `AUTO_VALUE_CONFIRM512x2`.
- Strata: `NATURAL_FIRST_SEARCH`, `AUTO_EFFECT_ACTIVE`.
- Seeds: `2230`, `2231`.
- Horizons: `K4`, `K8`.
- Each `(stratum, seed)` K4/K8 pair used the same frozen 512-state manifest.
- Fork contract: same `xi_t`; Student executes reduced/student action; Teacher executes full-view action; both continue under reduced continuation policy; no full-harness takeover.
- `AUTO_EFFECT_ACTIVE` was determined from real runtime state/component code: full `auto_seed` present, reduced `auto_seed` absent, before first search, and no keyword/manual filtering.

Value-confirm result:

Recall rerun (formal 128-state cohort; output `outputs/0820_auto_populate_first_search_recall_128_rerun/`):

- Re-ran all 8 paired cells (NATURAL_FIRST_SEARCH/AUTO_EFFECT_ACTIVE × seeds 2230/2231 × K4/K8), 1024 rows total, using the frozen 128-state manifests and the same Teacher/Student first-action fork with Reduced continuation.
- Added and independently persisted endpoint candidate IDs, curated IDs, successful read IDs, entered/retained context IDs, and qrel-normalized candidate/activated recall fields. `full_harness_takeover=0`; all K4/K8 frozen-state provenance remained paired.
- Seed-merged pooled results: K4 candidate-pool Teacher/Student `1.2602%/1.2602%`, gain `+0.00 pp`; activated Teacher/Student `1.1393%/1.1393%`, gain `+0.00 pp`. K8 values are identical: candidate `1.2602%/1.2602%`, activated `1.1393%/1.1393%`, both gains `+0.00 pp`. All 512 paired rows per horizon were zero delta (`positive/negative/zero=0/0/512`).
- Recall gate conclusion: `recall-neutral`; the historical utility-positive result remains a separate utility layer and is not used to claim evidence-recall gain.

```text
decision: VALUE_POSITIVE
value rows: 4096 = 2 strata x 2 seeds x 2 horizons x 512 states
NATURAL_FIRST_SEARCH seed2230 K4 mean=0.009990 ci_low=0.008086 noise_q95=0 gate=true
NATURAL_FIRST_SEARCH seed2230 K8 mean=0.010430 ci_low=0.007646 noise_q95=0 gate=true
NATURAL_FIRST_SEARCH seed2231 K4 mean=0.009990 ci_low=0.008057 noise_q95=0 gate=true
NATURAL_FIRST_SEARCH seed2231 K8 mean=0.010430 ci_low=0.007617 noise_q95=0 gate=true
AUTO_EFFECT_ACTIVE seed2230 K4 mean=0.009990 ci_low=0.008086 noise_q95=0 gate=true
AUTO_EFFECT_ACTIVE seed2230 K8 mean=0.010430 ci_low=0.007852 noise_q95=0 gate=true
AUTO_EFFECT_ACTIVE seed2231 K4 mean=0.009990 ci_low=0.008145 noise_q95=0 gate=true
AUTO_EFFECT_ACTIVE seed2231 K8 mean=0.010430 ci_low=0.007734 noise_q95=0 gate=true
```

Data and privilege schema:

- `AUTO_PRIVILEGE_SCHEMA.md` records the structured runtime signal: `auto_seed` from full working memory before first search, reduced/full component mask visibility, `step`, and `tool_history` for first-search localization.
- Student inference has no auto privilege; `auto_populate_first_search=false` in the reduced mask.
- Dataset source: Student under `H_-auto_populate_first_search`, restricted to first-search / first-evidence-population relevant decision windows.
- Unique K4 states recovered for training/eval construction: `1180` across `370` query ids.
- Query-disjoint split: `train_unique=860`, `valid_unique=164`, `test_unique=156`.
- Positive train unique states: `532`; first-turn train unique states: `456`.
- Resampled update-budget files were written:
  - `AUTO_TRAIN_8K.jsonl`: 8000 rows.
  - `AUTO_VALUE_POSITIVE_TRAIN_8K.jsonl`: 8000 rows.
  - `AUTO_FIRST_TURN_TRAIN_8K.jsonl`: 8000 rows.
  - `AUTO_VALID_1K.jsonl`: 1000 rows.
  - `AUTO_TEST_1K.jsonl`: 1000 rows.
- `AUTO_TRAIN_8K` and related 8K files are update budgets, not 8K unique states. Resampling is explicit in-row via `opd_row_id`, `snapshot_hash`, and `resampled_duplicate`.
- `lambda_args=0.0` throughout; categorical 8-way route targets only.

Training setting:

- Base checkpoint: `/mnt/songzijun/models/pat-jj_harness-1-full/harness-1` (`pat-jj/harness-1`).
- Route space: canonical 8-way distribution over `fan_out_search`, `search_corpus`, `grep_corpus`, `read_document`, `review_docs`, `curate`, `verify`, `end_search`.
- LoRA: `r=8`, `alpha=16`.
- Anchor: `lambda_anchor=0.05`.
- LR: `1e-5`.
- Update budget: `8000` train rows per cell, `1000` validation rows, `epochs=1`.
- Initial 8-cell matrix completed:
  - relevant `route_kl_forward`, seeds 42/43.
  - relevant `route_kl_reverse`, seeds 42/43.
  - relevant `action_ce`, seeds 42/43.
  - value-positive `route_kl_forward`, seeds 42/43.
- Full/remaining and event-localized cells completed:
  - relevant `route_kl_reverse`, seeds 44/45.
  - relevant `action_ce`, seeds 44/45.
  - `FIRST_TURN_ONLY_CE`, seeds 42/43.
- Total completed training cells: `14`. All wrote `result.json`, had finite loss/grad, changed trainable params, and had `invalid_tool=0.0`.

Same-state route-proxy closed-loop result:

- Evaluator: finalizer/proxy over query-disjoint `AUTO_TEST_1K`; this is not official external BrowseComp reward.
- Student inference condition: no auto privilege.

```text
Base Student:                         JS=0.109307 CE=1.230626 agreement=0.244 overall_reward=0.590777 invalid=0
first_turn_only_action_ce_seed42:      JS=0.117034 CE=1.134433 agreement=0.634 overall_reward=0.765977 invalid=0
first_turn_only_action_ce_seed43:      JS=0.070548 CE=0.786220 agreement=0.632 overall_reward=0.790053 invalid=0
relevant_action_ce_seed42:             JS=0.065676 CE=0.765296 agreement=0.700 overall_reward=0.822881 invalid=0
relevant_action_ce_seed43:             JS=0.069365 CE=0.783413 agreement=0.684 overall_reward=0.813937 invalid=0
relevant_action_ce_seed44:             JS=0.072330 CE=0.792340 agreement=0.674 overall_reward=0.808176 invalid=0
relevant_action_ce_seed45:             JS=0.066732 CE=0.765288 agreement=0.674 overall_reward=0.810812 invalid=0
relevant_route_kl_forward_seed42:      JS=0.045184 CE=0.668246 agreement=0.596 overall_reward=0.785680 invalid=0
relevant_route_kl_forward_seed43:      JS=0.044623 CE=0.666239 agreement=0.638 overall_reward=0.804826 invalid=0
relevant_route_kl_reverse_seed42:      JS=0.044919 CE=0.669647 agreement=0.698 overall_reward=0.831637 invalid=0
relevant_route_kl_reverse_seed43:      JS=0.043177 CE=0.663138 agreement=0.698 overall_reward=0.832410 invalid=0
relevant_route_kl_reverse_seed44:      JS=0.042467 CE=0.660550 agreement=0.734 overall_reward=0.848923 invalid=0
relevant_route_kl_reverse_seed45:      JS=0.044644 CE=0.666814 agreement=0.662 overall_reward=0.815604 invalid=0
value_positive_route_kl_forward_seed42 JS=0.069387 CE=0.768542 agreement=0.624 overall_reward=0.787301 invalid=0
value_positive_route_kl_forward_seed43 JS=0.062405 CE=0.739052 agreement=0.636 overall_reward=0.795882 invalid=0
```

Gate result:

```text
best_model:                 relevant_route_kl_reverse_seed44
base_overall_reward:         0.5907768084
best_overall_reward:         0.8489228879
base_trajectory_recall:      0.244
best_trajectory_recall:      0.734
base_final_answer_recall:    0.2385346381
best_final_answer_recall:    0.7318766626
invalid_tool_max:            0.0
main_gate_pass:              true
```

Conclusion:

```text
AUTO_POPULATE_FIRST_SEARCH_PASS_SAME_STATE_ROUTE_PROXY
```

Interpretation for other servers/agents:

- This is the strongest completed H100 BTP component result so far in this simplified record: value confirm passed, training finished, and same-state route-proxy gate passed with large Student-after > Student-base gains.
- The safe claim is route/control-policy improvement under local same-state proxy evaluation, not official BrowseComp Chroma parity and not yet a final external task-reward win.
- Unlike H100-1 verify selective, this run currently has no shuffled-target control recorded. If another agent needs paper-grade causal evidence, run a matched shuffle-target control before overclaiming state-conditioned privilege learning.
- `FIRST_TURN_ONLY_CE` is a meaningful baseline: it improves strongly but does not beat the best relevant reverse-KL cell. This supports event-localized control being useful, while best current recipe is still relevant reverse Route-KL.
- Recommended recipe for H20 clean SFT rerun: reuse this pipeline with `auto_populate_first_search`, `lambda_args=0`, relevant states, reverse Route-KL and action-CE top candidates, plus the first-turn-only baseline and a shuffle-target control.
- Do not mix argument distillation into this auto-populate run; previous argument-side auto-populate signal was known bad/negative, and this successful run explicitly avoids args.
- Downstream baseline comparison should use `Base Student`, best auto Student, and Full Harness reference, always marking student inference as no-privilege.
- `BEST_AUTO_STUDENT.json` currently has `checkpoint: null` due to a finalizer aggregation issue, but the best cell checkpoint exists at the path below.

Best route-proxy checkpoint:

```text
/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-2/SCAPE/outputs/btp_h100_2_auto_populate/cells/relevant_route_kl_reverse_seed44/checkpoint
```

Required artifacts written:

```text
RUN_MANIFEST.json
STATUS_LIVE.md
AUTO_VALUE_CONFIRM/
AUTO_VALUE_GATE.json
AUTO_PRIVILEGE_SCHEMA.md
DATA_AUDIT.md
AUTO_TRAINING_CELLS.csv
FIRST_TURN_ONLY_BASELINE.csv
CLOSED_LOOP_RESULTS.csv
CLOSED_LOOP_RESULTS.md
BEST_AUTO_STUDENT.json
H1002_BTP_AUTO_HANDOFF.json
SHA256SUMS
```

Implementation / recovery notes:

- New/modified H100-2 scripts include `scripts/run_btp_h1002_auto_populate.py`, `scripts/build_btp_auto_data.py`, `scripts/train_route_opd.py`, `scripts/eval_route_opd.py`, `scripts/finalize_btp_auto_training.py`, `scripts/launch_btp_auto_8gpu.sh`, `scripts/launch_btp_auto_full_remaining.sh`, and `scape/training/route_opd.py`.
- Value confirm reuses the existing H100-2 live fork/replay machinery from `scripts/run_h1002_verify_value_confirm.py` but writes isolated BTP outputs under `outputs/btp_h100_2_auto_populate/`.
- `AUTO_EFFECT_ACTIVE` uses real renderer/mask state: full `auto_seed` available, reduced `auto_seed` absent, before first search. It is not a keyword filter.
- Finalizer `RUN_MANIFEST.json` records `status=completed`, `exit_code=0`, and completed shards `AUTO_VALUE_CONFIRM`, `gate`, `schema`, `handoff`.
- After completion, no `run_btp_h1002_auto_populate.py`, `train_route_opd.py`, or `launch_btp_auto` processes were running; `nvidia-smi` showed all 8 GPUs idle.

## 2026-08-20 content_dedup candidate/activated recall 128-state rerun

- 已扩展真实 HF/BM25 same-state fork runner，使其支持 `content_dedup`，并落盘 T/S endpoint candidate IDs、final curated IDs、成功 read/context-retention IDs 和 final activated IDs。运行环境为 `/opt/scape-easyopd-smoke7`，模型为 `pat-jj_harness-1-full/harness-1`，未在 `/mnt` 创建或更新环境。
- 正式 cohort 使用 seed `2214/2215`，每 seed 每 horizon `64` states，K4/K8 各合并为 `128` paired rows。每个 seed 的 K4/K8 ordered snapshot hash 均 `64/64` 一致；T/S initial-state hash mismatch=`0`，missing/empty qrel=`0`，invalid provenance=`0`，Full Harness takeover=`0`。
- qrel SHA-256=`a6f594975be57339de9e4e9f67f13c044f647feda77c0b84c45a1581e3041bd1`，corpus SHA-256=`21cbf37b998da25842d993917f37b3a020f0802c66ae20ff003aaa071f52b7be`，normalization=`split_at_first_underscore`。
- K4 candidate recall T/S=`0.620040%/0.620040%`，paired gain=`+0.00 pp`，CI95=`[0.00,0.00]`；activated recall T/S=`0.173611%/0.173611%`，paired gain=`+0.00 pp`，CI95=`[0.00,0.00]`。K8 数值完全相同。两 horizon、两指标 positive/negative/zero 均为 `0/0/128`；seed 2214/2215 的 paired gain 均为 `0.00 pp`，seed sample std=`0.00 pp`。
- Candidate precision T/S=`0.46875%/0.46875%`，mean set size=`10/10`；activated precision T/S=`0.78125%/0.78125%`，mean set size=`2/2`。结论：在该真实 qrel-compatible fork 上，content_dedup 没有 candidate 或 activated evidence recall 增益。
- 正式输出：`SCAPE/outputs/0820_content_dedup_real_recall_128/CONTENT_DEDUP_RECALL_PER_STATE.jsonl`、`CONTENT_DEDUP_RECALL_K4_K8_GATE.json`、`RUN_MANIFEST.json`。早期 blocked eligibility artifact 和 `INVALID_DIAGNOSTIC_ONLY` artifact 仅保留为失败尝试，不用于当前表格。
- 同状态 raw shard 已保存 branch endpoint `tool_search_cost` 与 `objective_utility`；离线提取产物为 `SCAPE/outputs/0820_content_dedup_real_recall_128/CONTENT_DEDUP_UTILITY_PER_STATE.jsonl` 和 `CONTENT_DEDUP_UTILITY_SUMMARY.json`。K4 `tool cost Δ=+0.2109375`、`utility Δ=-0.0031640625`；K8 `+0.3046875`、`-0.0045703125`（均 Teacher−Student，128 paired rows，Full Harness takeover=0）。已据此补入 `/mnt/songzijun/增益.md`，无需重跑模型。

## 2026-08-21 auto_populate_first_search OPD 384-query four-condition evaluation

- Strict pool is frozen and valid: 384 unique queries after official query/evidence/gold intersection and component-training query-ID exclusion; official test subset `76`; all four ordered query ID lists agree. Manifest SHA-256=`daa46743ef9b1d6acf1dd230e8da92761f3465d47f2a8d4f7981f3ff7c380092`.
- Conditions are Teacher, Student Before OPD, Student After PURE_OPD seed42 and Student After RL+OPD seed42. All four 384-query model-action generations completed with the same Qwen3 base and greedy decoding; After adapters used manual safetensors reload.
- Early local-proxy and deep scorer artifacts were invalidated by independent audits of action/fan-out semantics; their values must not be quoted. The final metric contract reports only strict Harness-schema Legal action rate and official-test Evidence Recall@5; Recall@100/1000 are explicitly not computed and their rows were removed from `opd对比.md`.
- Final official-test (`n=76`): Teacher Legal/R@5=`85.53%/2.73%`; Student Before=`94.74%/2.92%`; PURE_OPD After=`97.37%/4.05%` (`+2.63/+1.13 pp` vs Before); RL+OPD After=`98.68%/3.86%` (`+3.95/+0.94 pp`). Strict legality requires `fan_out_search.queries` to contain 1–5 nonempty strings; illegal actions receive zero recall.
- Java 21 is at `/opt/scape-jdk21`; Recall@5 uses official pyserini Lucene, retrieving top-5 for every legal fan-out subquery and applying rank-wise round-robin deduplicated fusion. Formal artifact: `SCAPE-EasyOPD/outputs/0821_auto_populate_opd_384_formal_v2/r5_final/`; summary SHA-256=`419c91ebaf3b0275f3ded9e414a4779835120358381807a2ab2e8bcf41efd1e5`; SHA256SUMS `10/10` passed. Runtime remains `/opt/scape-projected-action`; no environment was created or updated under `/mnt`.

## 2026-08-20 auto_populate_first_search recall audit

- 已核对 `/mnt/songzijun/增益2.md` 的新指标定义，并检查正式 artifact `outputs/0820_auto_populate_first_search_value_confirm_128/AUTO_VALUE_CONFIRM/AUTO_VALUE_PER_STATE.jsonl`（1024 rows；NATURAL_FIRST_SEARCH/AUTO_EFFECT_ACTIVE，seed 2230/2231，K4/K8）。
- 该 artifact 仅保存 utility、动作与简化 trace；没有 `gold_evidence_ids`、endpoint candidate-pool IDs、`final_curated_ids`、working-memory evidence IDs、成功 read/context-retention provenance。因此既不能计算 `candidate_evidence_pool_recall@K`，也不能计算 `activated_evidence_recall@K`；不能用历史 reward、route probability 或 read action 参数替代。
- 结论：auto-populate K4/K8 两项 recall 增益均为 `N/A`，不是 0；`/mnt/songzijun/增益.md` 的增益表已将原历史 weighted utility 与 `usable_evidence_recall@K` 两列替换为 candidate-pool recall 与 activated-evidence recall 两列，并按 seed 合并口径记录不可计算原因。

## 2026-08-16 H100-2 Structured Privilege vs Matched Text formal matrix and real closed-loop

Status: completed in main checkout `/mnt/songzijun/Capability_Evolution/SCAPE`.

Canonical output directory:

```text
/mnt/songzijun/Capability_Evolution/SCAPE/outputs/h100_2_structured_privilege_formal_0816/
```

Real closed-loop output directory:

```text
/mnt/songzijun/Capability_Evolution/SCAPE/outputs/h100_2_real_closed_loop_bm25_0816/
```

Second-component diagnostic output directory:

```text
/mnt/songzijun/Capability_Evolution/SCAPE/outputs/h100_2_importance_structured_privilege_formal_0816/
```

Task scope:

- Followed `/mnt/songzijun/Capability_Evolution/SCAPE/todo/0816-1/README.md` and `H100-2_structured_privilege.md`.
- Component P0: `auto_populate_first_search`.
- Component P1 / second structured-native check: `importance_tagging`.
- Goal was the matched-information comparison: whether harness-native structured privilege beats or matches deterministic textualization under the same semantic fields, same on-policy states, same split, same objective, and no-privilege Student inference.
- This section supersedes any interpretation that only same-state route proxy was completed. The primary H100-2 result below is the real BM25 closed-loop execution in `outputs/h100_2_real_closed_loop_bm25_0816/`.
- GPU-heavy Python used `/opt/vllm-qwen3-1.7b-harness/bin/python` with torch/CUDA; default Python was used only for pytest because the `/opt` torch env does not include pytest.

Structured/textual protocol:

- Frozen state source for AUTO training/proxy matrix:
  `outputs/h100_3_real_influence_shards/auto_populate_first_search/REAL_INFLUENCE_PER_STATE.jsonl`.
- Split is query-disjoint and frozen in:
  `TRAIN_SPLIT_MANIFEST.json`, `VALID_SPLIT_MANIFEST.json`, `TEST_SPLIT_MANIFEST.json`.
- Split sizes:

```text
train: 608 states
valid: 208 states
test:  208 states
```

- Student route-head inference features explicitly exclude privileged fields. They contain only reduced/no-privilege state identifiers/counts: query id hash, step, tool-history length, document count, prior-search count, and state hash feature.
- Structured/textual variants differ only in teacher/control target construction from matched fields; deployed route-head inference does not receive `auto_seed_present`, full mask fields, teacher tool, textual privilege, gold labels, rewards, or future observations.
- Textualizer uses deterministic key/value JSON rendering and round-trips back to the structured record. `AUTO_INFORMATION_EQUIVALENCE_AUDIT.md` records `1024/1024` pass.
- `student_inference_has_privilege=false` in all handoffs and cell summaries.

AUTO formal route-head training matrix:

```text
AUTO_STRUCT_DIRECT seeds 42,43,44,45
AUTO_STRUCT_TYPED seeds 42,43,44,45
AUTO_MATCHED_TEXT seeds 42,43,44,45
AUTO_JSON_TEXT_DIAGNOSTIC seeds 42,43
AUTO_STRUCT_TYPED_DEBOTTLENECK seeds 42,43,44,45
AUTO_STRUCT_EVENT_TUPLE seeds 42,43,44,45
```

- Total AUTO trained cells: `22`.
- Every cell wrote `cells/<variant>_seed<seed>/route_head.pt`, `summary.json`, `STATUS_LIVE.md`, and `DONE`.
- All route-head checkpoints were reloadable; losses/gradients were finite; invalid tool rate was `0.0`.
- First batch used 8-way parallelism across the 8 visible GPUs; redesign cells also used 8-way parallelism.
- Same-state route-head proxy diagnostics are preserved as `AUTO_REPRESENTATION_ROUTE_PROXY.csv`, but are no longer the primary result.

AUTO real closed-loop setting:

- Runner: `scripts/run_h100_2_route_head_closed_loop.py`.
- Output: `outputs/h100_2_real_closed_loop_bm25_0816/`.
- Environment: real BM25 BrowseComp-Plus interaction using `/mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus/indexes/bm25`, `queries.tsv`, and `qrel_evidence.txt`.
- The evaluator loads trained route-head checkpoints, lets the Student choose tools over multiple steps, mutates the BM25 state, and computes final reward from executed state. This is not a same-state route proxy.
- Methods compared:
  - `BASE_REDUCED`
  - `AUTO_MATCHED_TEXT`
  - best structured variant selected from AUTO matrix: `AUTO_STRUCT_TYPED`
- Query count: `128`.
- Max steps: `6`.
- Student inference privilege: `false`.

AUTO real closed-loop result:

```text
best_structured_variant: AUTO_STRUCT_TYPED
Structured - Textual overall_reward delta: 0.0
Structured - Base overall_reward delta:      +0.03
Textual - Base overall_reward delta:         +0.03
paired bootstrap CI for Structured - Textual: [0.0, 0.0]
```

Per-method means:

```text
AUTO_MATCHED_TEXT:
  n=128
  overall_reward=-0.015
  curated_evidence_recall=0.0
  trajectory_recall=0.005164930555555555
  final_answer_recall=0.0
  tool_calls=1.0

AUTO_STRUCT_TYPED:
  n=128
  overall_reward=-0.015
  curated_evidence_recall=0.0
  trajectory_recall=0.005164930555555555
  final_answer_recall=0.0
  tool_calls=1.0

BASE_REDUCED:
  n=128
  overall_reward=-0.045
  curated_evidence_recall=0.0
  trajectory_recall=0.005164930555555555
  final_answer_recall=0.0
  tool_calls=3.0
```

Primary conclusion for H100-2:

```text
Structured ~= Textual > Base
```

Interpretation:

- This satisfies the H100-2 minimum meaningful outcome on the primary real closed-loop evaluator: structured matches textual and both beat base under no-privilege inference.
- It does **not** support a `Structured > Textual` claim.
- Do not cite the route-head proxy negative/positive diagnostics as the primary H100-2 result. The primary result is `real_closed_loop.status=completed_real_closed_loop_bm25` in `H1002_STRUCTURED_PRIVILEGE_HANDOFF.json`.
- The real closed-loop reward remains low/negative in absolute value and has zero final-answer recall. It is useful for relative structured/text/base comparison, but not yet paper-grade external task success.
- Other servers should treat the current method as a parity result, not a structured-advantage result and not a final paper-grade BrowseComp result.

Second component / importance_tagging diagnostic:

- Output: `/mnt/songzijun/Capability_Evolution/SCAPE/outputs/h100_2_importance_structured_privilege_formal_0816/`.
- Frozen state source: `outputs/h100_3_real_influence_shards/importance_tagging/REAL_INFLUENCE_PER_STATE.jsonl`.
- Split sizes: `train=608`, `valid=208`, `test=208`.
- Cells:

```text
IMPORTANCE_STRUCT_TYPED seeds 42,43,44,45
IMPORTANCE_MATCHED_TEXT seeds 42,43,44,45
IMPORTANCE_STRUCT_ORDERED_TAGS seeds 42,43,44,45
```

- Total importance cells: `12`, all with reloadable `route_head.pt`.
- This was a formal route-head diagnostic for the second value-positive component; it was not used as the primary real closed-loop comparison.

Importance result:

```text
best_structured_variant: IMPORTANCE_STRUCT_TYPED
Structured - Textual reward-proxy delta: -0.004795988090336323
student_inference_has_privilege=false
```

Interpretation for importance:

- importance_tagging does not currently show structured advantage over matched text.
- Because AUTO real closed-loop is parity rather than failure, do not discard the whole structured-privilege direction solely from the importance diagnostic.
- If another agent continues H100-2, next work should redesign the structured interface or run a stronger component-specific importance real closed-loop evaluator rather than reusing the current structured adapter unchanged.

Final decision written in the handoff:

```text
status: completed_real_closed_loop_parity
method_decision: keep_as_parity_not_structured_advantage
best_structured_variant: AUTO_STRUCT_TYPED
structured_vs_textual_delta: 0.0
student_inference_has_privilege: false
```

Main handoff and decision files:

```text
/mnt/songzijun/Capability_Evolution/SCAPE/outputs/h100_2_structured_privilege_formal_0816/H1002_STRUCTURED_PRIVILEGE_HANDOFF.json
/mnt/songzijun/Capability_Evolution/SCAPE/outputs/h100_2_structured_privilege_formal_0816/H1002_FINAL_DECISION_AFTER_REDESIGN.md
/mnt/songzijun/Capability_Evolution/SCAPE/outputs/h100_2_structured_privilege_formal_0816/H1002_FINAL_DECISION_AFTER_REDESIGN.json
/mnt/songzijun/Capability_Evolution/SCAPE/outputs/h100_2_structured_privilege_formal_0816/REAL_CLOSED_LOOP_HANDOFF.json
/mnt/songzijun/Capability_Evolution/SCAPE/outputs/h100_2_real_closed_loop_bm25_0816/REAL_CLOSED_LOOP_HANDOFF.json
```

Required artifacts present in the canonical directory:

```text
STRUCTURED_COMPONENT_INVENTORY.md
MATCHED_INFORMATION_PROTOCOL.md
AUTO_INFORMATION_EQUIVALENCE_AUDIT.md
STRUCTURED_INTERFACE_V1.md
STRUCTURED_INTERFACE_V2.md
STRUCTURED_REP_DEBUG.md
AUTO_REPRESENTATION_CELLS.csv
AUTO_REPRESENTATION_CLOSED_LOOP.csv
AUTO_STRUCTURED_VS_TEXTUAL.md
AUTO_REPRESENTATION_BOOTSTRAP.csv
IMPORTANCE_VALUE_CONFIRM/
IMPORTANCE_VALUE_GATE.json
IMPORTANCE_PRIVILEGE_SCHEMA.md
BEST_STRUCTURED_STUDENT.json
H1002_STRUCTURED_PRIVILEGE_HANDOFF.json
REAL_CLOSED_LOOP_HANDOFF.json
H1002_FINAL_DECISION_AFTER_REDESIGN.md
H1002_FINAL_DECISION_AFTER_REDESIGN.json
SHA256SUMS
```

Implementation notes for other servers/agents:

- New scripts in the main checkout:

```text
scripts/run_h100_2_auto_formal_route.py
scripts/run_h100_2_importance_formal_route.py
scripts/run_h100_2_route_head_closed_loop.py
scripts/run_h100_2_structured_privilege_matrix.py
```

- `scripts/run_h100_2_structured_privilege_matrix.py` was an earlier offline evidence/proxy matrix and should not be treated as the final main result.
- `scripts/run_h100_2_auto_formal_route.py` trains the route-head cells and keeps Student inference no-privilege.
- `scripts/run_h100_2_route_head_closed_loop.py` is the real BM25 interactive closed-loop evaluator used for the main comparison.
- `AUTO_REPRESENTATION_ROUTE_PROXY.csv` preserves the old proxy aggregate after the required `AUTO_REPRESENTATION_CLOSED_LOOP.csv` was replaced with real closed-loop results.
- Existing unrelated dirty/deleted files in the SCAPE working tree were not reverted. Git status for the relevant new scripts is untracked unless another agent stages/commits them.

Verification performed:

```text
cd /mnt/songzijun/Capability_Evolution/SCAPE/outputs/h100_2_structured_privilege_formal_0816
sha256sum -c SHA256SUMS
# all OK

cd /mnt/songzijun/Capability_Evolution/SCAPE/outputs/h100_2_importance_structured_privilege_formal_0816
sha256sum -c SHA256SUMS
# all OK

cd /mnt/songzijun/Capability_Evolution/SCAPE
python -m pytest tests/test_component_mask.py tests/test_dual_view.py tests/test_metric_pairing.py tests/test_rollout_influence_contract.py -q
# 6 passed
```

Operational status:

- No H100-2 training/closed-loop processes were left running.
- Final `nvidia-smi` showed all 8 GPUs idle with about `1 MiB` used per GPU.
- `/opt/vllm-qwen3-1.7b-harness/bin/python` has torch/CUDA and was used for training/closed-loop; it does not have pytest, so pytest verification used the default Python environment.

## 2026-08-15 H100-1 Beyond Textual Privilege verify_tool selective/value-conditioned OPD

Status: selective/value-conditioned sweep completed in worktree `/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-1/SCAPE`; full expansion and real closed-loop gate were not run.

Canonical output directory:

```text
/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-1/SCAPE/outputs/btp_h100_1_verify_selective/
```

Task scope:

- Verify Tool / `V8D_VERIFY_TOOL` mainline for Beyond Textual Privilege.
- Goal was to test whether value-conditioned selective OPD can produce the first truly positive Student distillation result from released `pat-jj/harness-1`, without waiting for the H20 clean SFT checkpoint.
- All outputs remain local compatibility only: `LOCAL_COMPAT_ONLY=true`, `official_chroma_parity=false`.
- GPU-heavy Python used `/opt/bishop-harness/bin/python3.11`; no `/mnt` JuiceFS torch environment was used for training/eval.
- 8 visible H100-class GPUs were used for the main 8-cell 2K-equivalent sweep; the shuffle-target control ran as a single-GPU matched-budget control.

Execution status for this todo:

- Completed: privilege schema audit, value mining data build, Stage A/B selective mining, 2K-equivalent 8-cell training sweep, shuffled-target control, handoff material generation.
- Not performed: full 8K/update-budget expansion, real closed-loop `BTP_VERIFY_DEV128` / `BTP_VERIFY_TEST256` benchmark gate, any claim that the run produced a final verified Student win over Base.
- Partial / pending in the broader H100 baseline factory: matched-text OPD was only prepared, while OPSD / OPCD / SEED stayed blocked by missing faithful contracts.
- All downstream agents should treat the route-proxy gains as provisional only and keep the shuffle-target control in mind.

Current todo coverage:

- `H100-1` verify selective OPD: selective/value-conditioned sweep completed; full expansion and real closed-loop gate did **not** run.
- `H100-3` subtractive_curation event-conditioned selective distillation: completed separately and stopped after closed-loop gate failure.
- `H100-4` baseline factory: only partial preparation is done in the broader H100 baseline line; matched-text OPD is prepared, but OPSD / OPCD / SEED remain blocked.
- Do not infer any cross-task final winner from this file alone; read the per-experiment sections below for the exact gating status.

Data and privilege schema:

- Source value rows: H100-2 live fork/replay output `/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-2/SCAPE/outputs/h100_2_verify_value_confirm/VERIFY_VALUE_PER_STATE.jsonl`.
- Snapshot backjoin source: H100-2 frozen `manifests/` and `manifest_shards/` under the same output directory.
- Builder script: `scripts/build_btp_verify_selective_data.py` in the H100-1 worktree.
- Recovered strict same-state rows with snapshots: `1164` unique K4 states across `367` query ids.
- `86` value rows could not be backjoined to a frozen snapshot and were excluded; no fake snapshot or duplicate unique state was created.
- `SELECT_POSITIVE`: `609` rows across `263` query ids, rule `A_K4 > replay_noise_q95` with `replay_noise_q95=0.0`.
- `SELECT_STRICT`: also `609` rows; all positives had action disagreement and positive teacher margin.
- 2K-equivalent training used the `609` positive states with explicit resampling to match optimizer update budget. Resampling is marked in-row and must not be reported as 2K unique states.
- `VERIFY_PRIVILEGE_SCHEMA_V2.md` includes only state-time fields: full/student verify availability, step, document count, curated-id count, document ids, claim-nonempty flag, tool history, and remaining budget. Future labels such as `A_K4`, terminal reward, replay noise, and gold answer recall are explicitly excluded from privilege inputs.

Training setting:

- Base checkpoint: `/mnt/songzijun/models/pat-jj_harness-1-full/harness-1` (`pat-jj/harness-1`).
- Route space: canonical 8-way distribution over `fan_out_search`, `search_corpus`, `grep_corpus`, `read_document`, `review_docs`, `curate`, `verify`, `end_search`.
- LoRA: `r=8`, `alpha=16`.
- Anchor: `lambda_anchor=0.05`.
- Learning rate: smoke at `1e-5`, then all formal cells uniformly lowered to `5e-6` because reverse-KL smoke had large but finite gradients. Do not selectively tune LR per objective.
- Update budget: `2000` train rows per cell, `1000` validation rows.
- Formal 8-cell matrix:
  - `SELECT_POSITIVE + route_kl_forward`, seeds 42/43.
  - `SELECT_POSITIVE + route_kl_reverse`, seeds 42/43.
  - `SELECT_POSITIVE + action_ce`, seeds 42/43.
  - `advantage-weighted positive + route_kl_forward`, seeds 42/43.
- Value weight for weighted cells: `w_i=max(A_K4-noise_floor,0)`, batch/global mean normalized in data and clipped to fixed upper bound `4.0` in the loss path.

2K-equivalent route results:

```text
weighted_positive route_kl_forward: mean ΔJS=-0.400175, mean ΔCE=-2.597806, mean post agreement=0.6245, invalid_tool=0
SELECT_POSITIVE route_kl_forward:  mean ΔJS=-0.399374, mean ΔCE=-2.594443, mean post agreement=0.6105, invalid_tool=0
SELECT_POSITIVE route_kl_reverse:  mean ΔJS=-0.390069, mean ΔCE=-2.543890, mean post agreement=0.6410, invalid_tool=0
SELECT_POSITIVE action_ce:         mean ΔJS=-0.277253, mean ΔCE=-1.589558, mean post agreement=0.6400, invalid_tool=0
```

Important stability note:

- All formal cells were finite and wrote checkpoints.
- `positive_route_kl_reverse_seed43` had `max_grad_norm=19561.04`; treat reverse-KL as less stable even though it finished.
- Weighted forward Route-KL had lower gradients (`600-696`) and best average route proxy among non-control cells.

TEST256 same-state proxy evaluation:

- Evaluator: `scripts/eval_route_opd.py` after fixing adapter loading so LoRA checkpoints are loaded via `PeftModel.from_pretrained(base, adapter)` and not double-wrapped.
- Split: `BTP_VERIFY_TEST256_same_state_proxy`, first 256 rows from query-disjoint `VT_TEST_1K.jsonl`.
- This is **not** external BrowseComp task reward. CER / Trajectory Recall / Final Answer Recall are not evaluated in this path.

```text
base_student                         JS=0.056527  CE=0.875336  agreement=0.000000
weighted_route_kl_forward_seed43      JS=0.029468  CE=0.748518  agreement=0.660156
positive_route_kl_forward_seed42      JS=0.031821  CE=0.759385  agreement=0.699219
weighted_route_kl_forward_seed42      JS=0.033711  CE=0.767588  agreement=0.695313
positive_route_kl_forward_seed43      JS=0.034831  CE=0.772555  agreement=0.605469
positive_route_kl_reverse_seed42      JS=0.036729  CE=0.786680  agreement=0.601563
positive_route_kl_reverse_seed43      JS=0.042469  CE=0.816023  agreement=0.679688
positive_action_ce_seed42             JS=0.151659  CE=1.720422  agreement=0.675781
positive_action_ce_seed43             JS=0.153668  CE=1.750814  agreement=0.675781
```

Critical shuffle-target control:

- Control data: same `VT_SELECTIVE_TRAIN_2K` source states and same 2000 update budget.
- Teacher targets were shuffled across states with seed `8151042`.
- Teacher action marginal distribution was preserved exactly: `end_search=1919`, `read_document=81`.
- State-conditioned pairing was destroyed; there were zero fixed points in the permutation.
- Control objective: seed 42 forward 8-way Route-KL.

```text
unshuffled positive_route_kl_forward_seed42 TEST256 proxy: JS=0.031821, CE=0.759385, agreement=0.699219
shuffled_route_kl_forward_seed42 TEST256 proxy:            JS=0.023599, CE=0.722810, agreement=0.648438
```

Conclusion:

```text
NO_FULL_EXPANSION
```

Interpretation for other servers/agents:

- Do **not** cite this H100-1 run as the first positive Student distillation result.
- Route-level proxy improves strongly, but the shuffled-target control matches or exceeds the unshuffled seed42 proxy. Current evidence is compatible with marginal teacher-target imitation, not isolated state-conditioned privilege learning.
- The selected positive training rows are heavily dominated by `end_search`/`read_document`; there is no verify argmax in the selected teacher-target marginal. Therefore the run does not demonstrate verify-specific behavior improvement.
- Do not launch full 8K/update-budget expansion from this result. `FULL_TRAINING.csv` intentionally records `not_started`.
- Do not report `Student_after > Student_base` for real task reward; only same-state route proxy improved. Real `BTP_VERIFY_DEV128` / `BTP_VERIFY_TEST256` closed-loop reward, CER, Trajectory Recall, and Final Answer Recall were not evaluated.
- Do not hand this checkpoint to H100-4 as a final best Student for baseline comparison. `BEST_VERIFY_STUDENT.json` marks the best route-proxy checkpoint but also marks it as not recommended as final Student.
- H20 clean SFT rerun should **not** reuse this recipe as-is. First collect a larger on-policy verify pool with a verify-action-positive subset, repeat weighted/unweighted forward Route-KL, keep the shuffle-target control, and only then run the real closed-loop gate.

Best route-proxy checkpoint, not main-gate winner:

```text
/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-1/SCAPE/outputs/btp_h100_1_verify_selective/cells/weighted_route_kl_forward_seed43/checkpoint
```

Required artifacts written:

```text
RUN_MANIFEST.json
STATUS_LIVE.md
VERIFY_PRIVILEGE_SCHEMA_V2.md
DATA_AUDIT.md
VALUE_MINING_SUMMARY.md
VALUE_PER_STATE.jsonl
SELECT_POSITIVE.jsonl
SELECT_STRICT.jsonl
SELECTIVITY_STATS.csv
TRAINING_2K_EQUIV.csv
TRAINING_GATE.json
FULL_TRAINING.csv
CLOSED_LOOP_RESULTS.csv
CLOSED_LOOP_RESULTS.md
SHUFFLED_TARGET_CONTROL.md
BEST_VERIFY_STUDENT.json
H1001_BTP_VERIFY_HANDOFF.json
SHA256SUMS
```

Implementation / recovery notes:

- New/modified H100-1 scripts: `scripts/build_btp_verify_selective_data.py`, `scripts/launch_btp_verify_selective_8gpu.sh`, `scripts/eval_route_opd.py`, `scripts/run_btp_verify_proxy_eval_8gpu.sh`, `scripts/train_route_opd.py`, `scape/training/route_opd.py`.
- Adapter evaluation bug fixed in H100-1 `scape/training/route_opd.py`: adapter checkpoints must be loaded from base model via PEFT once; double-wrapping adapters invalidates proxy evaluation.
- Advantage-weighted loss path was fixed so `normalized_value_weight` actually multiplies the main objective and is clipped to `4.0`.
- GPU cleanup verified after completion: all 8 GPUs idle.

## 2026-08-15 H100-3 Beyond Textual Privilege subtractive_curation event-conditioned selective distillation

Status: completed in worktree `/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-3/SCAPE`.

Canonical output directory:

```text
/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-3/SCAPE/outputs/btp_h100_3_subtractive/
```

Task scope:

- Component: `V8D_SUBTRACTIVE_CURATION` / `subtractive_curation`.
- Goal was to test whether curation-critical event-conditioned selective distillation can produce the third transferable Beyond Textual Privilege component from released `pat-jj/harness-1`, without waiting for the H20 clean SFT checkpoint.
- All outputs are local BM25/HF compatibility only: `LOCAL_COMPAT_ONLY=true`, `official_chroma_parity=false`. Do not describe these as official Harness-1 Chroma parity results.
- GPU-heavy Python used `/opt/scape-h1003-hf-scorer/bin/python`; no `/mnt` JuiceFS torch environment was used.
- 8 H100-class GPUs were used for raw collection, K4 value mining, K8 confirmation, and the first-stage training matrix.

Important implementation files added/modified in the H100-3 worktree:

```text
scripts/run_btp_subtractive.py
scripts/prepare_btp_subtractive_training.py
scripts/eval_btp_subtractive_closed_loop.py
scripts/train_route_opd.py
scape/training/route_opd.py
```

Data/event schema:

- Raw collection target was `SC_RAW_TRAIN_12K`; actual full collection produced `13,280` runtime states from `830` BrowseComp+ query ids, with `16` on-policy reduced-view turns per query.
- Event schema output: `CURATION_EVENT_SCHEMA.md`.
- Event fields are from real local Harness-compatible runtime state only: `snapshot.working_memory.documents`, `curated_ids`, `curated_importance`, `verified_unsupported`, `evidence_graph`, and `teacher_action.arguments.add_ids/remove_ids`.
- No LLM subjective curation labels were used.
- Id supervision is allowed only when `add_ids` exist in current `documents` and `remove_ids` exist in current `curated_ids`.
- Query-disjoint training/eval route splits were generated from `SC_VALUE_POSITIVE.jsonl`: `SC_ROUTE_TRAIN_2K.jsonl`, `SC_VALID_1K.jsonl`, `SC_TEST_1K.jsonl`.

Stage A / value mining:

```text
Raw states:              13,280
Stage A candidates:       1,855
K4 value rows:            1,855
K4 positive:                703
K4 zero:                    333
K4 negative:                819
K4 broad mean value:       -0.0022237197
K8 confirmation rows:        96
K8 negative-control repeat:  32
```

Interpretation of value mining:

- Broad/unfiltered subtractive curation is not supported: the K4 broad mean is slightly negative.
- There is a real selective positive subset (`703` states), so the useful hypothesis is event/value-conditioned, not broad OPD.
- Final paper main set should use `SC_VALUE_POSITIVE` only if a closed-loop gate passes. `SC_TERMINAL_POSITIVE` was written as secondary-only data and was not mixed into the main training set.

First-stage training setting:

- Base checkpoint: `/mnt/songzijun/models/pat-jj_harness-1-full/harness-1` (`pat-jj/harness-1`).
- Route space: canonical 8-way distribution over `fan_out_search`, `search_corpus`, `grep_corpus`, `read_document`, `review_docs`, `curate`, `verify`, `end_search`.
- LoRA: `r=8`, `alpha=16`.
- Anchor: `lambda_anchor=0.05`.
- LR: `1e-5`.
- Bounded first-stage ranking pass: `train_states=512`, `eval_states=128`, `epochs=1`.
- The full `train_states=2000`, `eval_states=1000` attempt was too slow with serial route scoring and was stopped; logs are preserved. It should not be reported as the completed training setting.
- Required 8-cell matrix completed after trainer microbatch optimization:
  - `route_kl_forward`, seeds 42/43.
  - `action_ce`, seeds 42/43.
  - `route_kl_arg_ce`, seeds 42/43.
  - `value_weighted_route_kl`, seeds 42/43.

Training result summary from `ROUTE_OPD_REPORT.json` / `TRAINING_CELLS.csv`:

```text
value_weighted_route_kl: mean ΔJS=-0.025046, mean ΔCE=-0.106236, finite=true, params_changed=true
route_kl_forward:        mean ΔJS=-0.017013, mean ΔCE=-0.066911, finite=true, params_changed=true
route_kl_arg_ce:         mean ΔJS=-0.017013, mean ΔCE=-0.066911, finite=true, params_changed=true
action_ce:               mean ΔJS=+0.052417, mean ΔCE=+0.462664, finite=true, params_changed=true
```

Argument-supervision audit:

- `SC_VALUE_POSITIVE` contained zero train rows with valid curate argument supervision under the strict current-state id audit.
- Therefore `route_kl_arg_ce` reduced to route KL on this split and explicitly reports argument CE as not applicable.
- No nonexistent document id was teacher-forced.
- Do not claim this run tested successful pointer/add-remove-id supervision.

Closed-loop gate:

- Evaluator: `scripts/eval_btp_subtractive_closed_loop.py`.
- Split: first `256` rows from query-disjoint `SC_TEST_1K.jsonl`.
- Models evaluated: base student, value-weighted Route-KL seeds 42/43, Route-KL forward seeds 42/43.
- Metrics are same-state proxy metrics derived from runtime state; they are not official external BrowseComp task reward.

Gate result:

```text
best_model: value_weighted_route_kl_seed42
base curated_evidence_recall: 0.0
best curated_evidence_recall: 0.0
base overall_reward:          0.0
best overall_reward:          0.0
base final_answer_recall:     0.0
best final_answer_recall:     0.0
best invalid_tool_rate:       0.0
closed_loop_gate_pass:        false
```

Conclusion:

```text
STOP subtractive_curation
```

Interpretation for other servers/agents:

- Do **not** launch Top-2 × 4 seeds full training for subtractive_curation from this run.
- Do **not** return to broad/unfiltered 8K subtractive curation OPD; the K4 broad mean is negative and the closed-loop gate failed.
- Do **not** cite this as `Student_after > Student_base`; closed-loop proxy showed no improvement over base.
- Do **not** cite this as evidence for working argument/pointer distillation; valid arg-supervision rows were zero under the strict id audit.
- Safe claim: event-conditioned mining found a selective positive subset, and value-weighted Route-KL improved route proxy metrics in the bounded first-stage pass, but the recipe failed the closed-loop gate and must stop.
- Next candidate priority from the todo remains `auto_populate_first_search > importance_tagging`, but only take over `auto_populate_first_search` if H100-2 is not already running it.
- H20 clean SFT rerun should not blindly reuse this recipe. If revisited, first collect states with actual valid `curate` add/remove id targets and a nonzero closed-loop recall signal.

Required artifacts written:

```text
RUN_MANIFEST.json
STATUS_LIVE.md
CURATION_EVENT_SCHEMA.md
DATA_AUDIT.md
VALUE_PER_STATE.jsonl
SC_VALUE_POSITIVE.jsonl
SC_TERMINAL_POSITIVE.jsonl
SC_VALUE_NEGATIVE_CONTROL.jsonl
TRAINING_CELLS.csv
ARGUMENT_SUPERVISION_AUDIT.md
CLOSED_LOOP_RESULTS.csv
CLOSED_LOOP_RESULTS.md
BEST_SUBTRACTIVE_STUDENT.json
H1003_BTP_SUBTRACTIVE_HANDOFF.json
SHA256SUMS
```

Operational/recovery notes:

- Raw collection initially OOMed with an 8-way batched scorer. `scripts/run_btp_subtractive.py` was patched to microbatch continuation scoring; the full 13,280-state collection then completed.
- K4 full value mining completed in six canonical shards covering all 1,855 candidates. Partial K4 shards from an early overlap run exist and should not be double-counted as canonical K4 rows.
- Training initially collided with active K8 memory on GPUs 1/7 and two seed-43 cells OOMed. Those exact PIDs were stopped/relaunched after K8 freed memory; broad `pkill` was rejected and not used.
- `scape/training/route_opd.py` was extended with `value_weighted_route_kl`, `route_kl_arg_ce`, and microbatched `route_logits`. A tiny smoke passed with finite loss/grad and `smoke_pass=true` before relaunching the 8-cell matrix.
- Closed-loop evaluation completed successfully and wrote `CLOSED_LOOP_GATE.json` with `pass=false`.
- Final cleanup verified no `run_btp_subtractive.py`, `train_route_opd.py`, or `eval_btp_subtractive` workers remained; all 8 GPUs were idle.

## 2026-08-14 H100-4 verify_tool Structured-vs-Textual privilege representation

Status: completed in worktree `/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-4/SCAPE`.

Canonical output directory:

```text
/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-4/SCAPE/outputs/h100_4_privilege_representation/
```

Important source data:

```text
/mnt/songzijun/Capability_Evolution/SCAPE/outputs/h100_4_verify_confirm/verify_tool_hf_scorer/REAL_INFLUENCE_PER_STATE.jsonl
```

Environment/resource notes:

- Used `/opt/scape-hf-scorer` as the Python/CUDA environment.
- Do not rerun SCAPE GPU workloads from `/mnt` JuiceFS Python environments.
- This machine exposed 4 visible GPUs, not the 8 GPUs assumed by the todo. The 8-cell matrix was run in two GPU waves.
- Existing source data had only 2048 states total, so query-disjoint split could not be `2000/256/256` without duplication. Actual split is `1536/256/256` from 96/16/16 query ids.
- No samples were duplicated to fake the requested 2K train size.

Experiment setting:

- Component fixed to `verify_tool`.
- Structured privilege: structured non-natural-language JSON boolean `verify_available`, derived from Harness runtime `verify_tool` mask / `full_view.verify_available`.
- Textual privilege: deterministic natural-language template rendering exactly the same `verify_available` boolean.
- Information equivalence: pass. No LLM textualizer, no extra reasoning, no claim text, no candidate document ids, no future information.
- Student inference/runtime view: reduced no-privilege view only; no privileged field.
- Objectives run: Route-KL and Action CE.
- Seeds run: 42 and 43 for each structured/textual x objective cell.
- Smoke checks passed for finite loss, finite gradients, reloadable checkpoint, normalized route distribution, invalid tool rate 0.

Primary metrics:

```text
Structured Route-KL common-reference test JS: 0.016458936035633087
Textual Route-KL common-reference test JS:    0.01645179372280836
Route-KL seed wins: ['structured', 'textual']
structured_beats_base: false
structured_beats_text: false
textual_beats_struct: false
```

Conclusion:

```text
REPRESENTATION_PARITY_OR_UNSTABLE_GAP_NO_BASE_GAIN
```

Interpretation for other servers/agents:

- Do not cite this run as evidence that structured privilege beats textual privilege.
- Do not cite this run as evidence that textual privilege robustly beats structured privilege either; the Route-KL seed direction is inconsistent and the mean gap is tiny.
- The safe paper-level claim from this run is that this particular verify_tool representation comparison is inconclusive / parity-like under the available 2048-state source, with no stable base gain.
- 8K expansion was not triggered because the source only contains 2048 states and generating new data was outside this completed H100-4 run.
- Official Chroma parity was not run and remains `false` in handoff.

Key artifacts:

```text
RUN_MANIFEST.json
STATUS_LIVE.md
STRUCTURED_PRIVILEGE_SCHEMA.md
REPRESENTATION_CONTRACT.md
INFORMATION_EQUIVALENCE_AUDIT.md
TEXTUALIZATION_TEMPLATE.md
DATA_AUDIT.md
TRAIN_SPLIT_MANIFEST.json
VALID_SPLIT_MANIFEST.json
TEST_SPLIT_MANIFEST.json
TEACHER_SIGNAL_DIAGNOSTIC.csv
TEACHER_SIGNAL_DIAGNOSTIC.md
NULL_REPRESENTATION_CONTROLS.md
REPRESENTATION_2K_CELLS.csv
REPRESENTATION_2K_REPORT.md
REPRESENTATION_GATE.json
CLOSED_LOOP_RESULTS.csv
CLOSED_LOOP_RESULTS.md
STRUCTURED_VS_TEXTUAL.md
H1004_PRIVILEGE_REP_HANDOFF.json
SHA256SUMS
```

Verification performed:

```text
pytest tests/test_learnability_metrics_v3.py  # 7 passed
sha256sum -c SHA256SUMS                      # passed
```

## 2026-08-14 H100-3 verify_tool 8-way route OPD objective sweep

Status: completed in worktree `/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-3/SCAPE`.

Canonical output directory:

```text
/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-3/SCAPE/outputs/h100_3_route_opd_verify/
```

Experiment setting:

- Component fixed to `verify_tool`.
- Route space fixed to 8-way canonical tool distribution: `fan_out_search`, `search_corpus`, `grep_corpus`, `read_document`, `review_docs`, `curate`, `verify`, `end_search`.
- Source data: `outputs/h100_3_supervision_coherence/STATE_MANIFESTS/verify_tool_states.json`.
- Split: query-disjoint train/valid/test, deterministic seed `81403`.
- Train/eval budget: 2000 train states and 1000 valid states per cell.
- Objectives: `route_kl_forward`, `route_kl_reverse`, `tool_name_token_kl`, `action_ce`.
- Seeds: `42`, `43` for each objective.
- Shared hyperparameters: `epochs=1`, `lr=1e-5`, `lora_rank=8`, `lora_alpha=16`, `lambda_anchor=0.05`.
- Local compat only: `official_chroma_parity=false`, `local_compat_only=true`.

Validation:

- Metric contract passed on same distribution and perturbed distribution.
- Smoke checks existed for all 4 objectives, but only `route_kl_forward` smoke passed; the other three smoke runs were unstable under the chosen budget.
- Finalization produced `ROUTE_OPD_2K_CELLS.csv`, `ROUTE_OPD_2K_REPORT.md`, `ROUTE_OPD_GATE.json`, `H1003_ROUTE_OPD_HANDOFF.json`, `SHA256SUMS`.

Primary results:

- `route_kl_forward`: mean ΔJS = `0.017984`, smoke pass rate `0.00`.
- `route_kl_reverse`: mean ΔJS = `0.015376`, smoke pass rate `0.00`.
- `tool_name_token_kl`: mean ΔJS = `0.024542`, smoke pass rate `0.00`.
- `action_ce`: mean ΔJS = `0.029073`, smoke pass rate `0.00`.
- Best objective by post-eval JS was `route_kl_reverse`, but `route_js_improved=false` overall.

Conclusion:

```text
OBJECTIVE_MISMATCH_NOT_SUFFICIENT_EXPLANATION
```

Interpretation for other servers/agents:

- This 2K sweep does **not** support the claim that route-level OPD already fixes verify_tool routing.
- None of the four objectives reduced held-out route JS on average; all moved the distribution away from the teacher reference.
- Do not expand this run to 8K from this result alone.
- Do not write this up as evidence for route-KL superiority over action CE or token-name KL.
- Closed-loop Student evaluation was completed later as a 64-state same-state proxy evaluation; `best_objective=base_student`, `route_js_improved=false`, `closed_loop_student_improved=false`.
- The closed-loop proxy outputs are in `outputs/h100_3_route_opd_verify/CLOSED_LOOP_RESULTS.md`, `CLOSED_LOOP_GATE.json`, and `H1003_ROUTE_OPD_HANDOFF.json`.
- Because the source manifest does not include external answer gold, the closed-loop metrics are state-derived proxies, not an external gold benchmark.

## 2026-08-15 H100-4 Beyond Textual Privilege baseline factory

Status: partial completion. Worktree:

```text
/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-4/SCAPE/
```

Canonical output directory:

```text
/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-4/SCAPE/outputs/btp_h100_4_baselines/
```

Task scope:

- Compare Base, Full-Harness trajectory SFT, Standard OPD, Matched Text OPD, OPSD, OPCD, and SEED under the Beyond Textual Privilege protocol.
- Keep the same released `pat-jj/harness-1` checkpoint family, shared train/eval data contract, no inference-time privilege, and canonical V3 KL/JS metrics.
- This H100 machine exposed 4 GPUs, so parallel work was run in 4-GPU waves. GPU-heavy Python used `/opt/scape-hf-scorer/bin/python`; no `/mnt` JuiceFS torch environment was used.
- All results remain `LOCAL_COMPAT_ONLY=true` and `official_chroma_parity=false`.

Available source data and controls:

- Reused the frozen H100-3 route data at `/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-3/SCAPE/outputs/h100_3_route_opd_verify/VT_ROUTE_TRAIN_8K.jsonl`.
- Source contains 8,000 route states with Student/Teacher route distributions, reduced/full prompts, and same-state provenance.
- Route-control objective audit ran on 4 GPUs with route-KL seeds 42/43 plus action-CE and legacy name-only controls.
- V3 metric tests passed: 10 focused tests passed, including learnability metrics and metric pairing contracts.
- All generated artifacts pass `sha256sum -c SHA256SUMS`; no residual torchrun/vLLM experiment processes remain and all 4 GPUs are idle.

Completed control results:

```text
Base                         completed as route-control reference
Full-Harness SFT/action-CE  completed as route-control objective
Standard OPD/route-KL       completed as route-control objective, seeds 42/43
Objective diagnosis         NO_OBJECTIVE_RESCUE
Recommended interpretation  released Harness-1 is a placement-boundary setting
```

Important qualification: these completed entries are V3 route-policy controls, not a completed 7B LoRA/closed-loop Search baseline factory. They do not support `Student_after > Student_base`, `Ours > OPSD`, `Ours > SEED`, or `Structured > Matched Text`.

H100-1 V2 synchronization barrier:

- H100-1 produced `/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-1/SCAPE/outputs/btp_h100_1_verify_selective/VERIFY_PRIVILEGE_SCHEMA_V2.md` and `SELECT_POSITIVE.jsonl`.
- V2 schema contains nine state-time fields: full/student verify availability, step, document count, curated-id count, document ids, claim-nonempty flag, tool history, and remaining budget.
- The H100-4 deterministic adapter generated `outputs/btp_h100_4_baselines/matched_v2/matched_v2_pairs.jsonl` from 609 H100-1 positive states across 266 query ids.
- Information audit passed `609/609` structured-to-text round trips. No future labels, reward, gold answer, or generated reasoning were included.
- Matched Text OPD status is `prepared_pending_training`; the actual optimizer/teacher training has not yet been launched.

Blocked or pending methods:

```text
Matched Text OPD              prepared_pending_training; V2 data audit passed
OPSD-style                    blocked: no faithful verified textual Search trajectory contract
OPCD-style                    blocked: no frozen historical successful-search textual experience contract
SEED-style distillation-only  blocked: no skill analyzer/rescoring implementation and no stable RL pipeline
```

SCOPE registry checks:

- `SCOPE/scripts/iclr/preflight.sh` was blocked because `artifacts/datasets/round2_audit_100q/query_manifest.json` is absent.
- `SCOPE/scripts/iclr/run_baseline_dryrun.sh` was blocked under system Python because `yaml`, `torch`, `transformers`, and `vllm` are unavailable there. The approved `/opt` environment imports the core packages, but cannot repair the missing query manifest.
- SCOPE external SEED/OPID/SDAR adapters are dry-run metadata adapters, not executable full training implementations; they must not be reported as completed baselines.

Required artifacts and handoff:

```text
outputs/btp_h100_4_baselines/RUN_MANIFEST.json
outputs/btp_h100_4_baselines/STATUS_LIVE.md
outputs/btp_h100_4_baselines/BASELINE_PROTOCOL.md
outputs/btp_h100_4_baselines/FAIRNESS_AUDIT.md
outputs/btp_h100_4_baselines/MATCHED_INFORMATION_AUDIT.md
outputs/btp_h100_4_baselines/OPSD_ADAPTATION_NOTES.md
outputs/btp_h100_4_baselines/OPCD_ADAPTATION_NOTES.md
outputs/btp_h100_4_baselines/SEED_ADAPTATION_NOTES.md
outputs/btp_h100_4_baselines/TRAINING_BUDGETS.csv
outputs/btp_h100_4_baselines/BASELINE_RESULTS.csv
outputs/btp_h100_4_baselines/BASELINE_RESULTS.md
outputs/btp_h100_4_baselines/MAIN_COMPARISON_TABLE.csv
outputs/btp_h100_4_baselines/MAIN_COMPARISON_TABLE.md
outputs/btp_h100_4_baselines/PAIRED_BOOTSTRAP.csv
outputs/btp_h100_4_baselines/COMPUTE_COST.csv
outputs/btp_h100_4_baselines/H1004_BTP_BASELINE_HANDOFF.json
outputs/btp_h100_4_baselines/SHA256SUMS
```

Recovery instructions for the next server/agent:

1. Do not rerun the old boolean-only representation experiment as the matched-text result.
2. Start from `matched_v2/matched_v2_pairs.jsonl` and preserve the 609/609 information-equivalence audit.
3. Integrate the V2 structured fields into the existing HF teacher/student optimizer path, then run matched-text seeds 42 and 43 with the same occupancy/update budget as the structured method.
4. Obtain and freeze the final train/valid/test query manifests before any cross-method comparison.
5. Implement or import faithful OPSD/OPCD/SEED contracts before placing those methods in the main table; otherwise keep them explicitly blocked.
6. Recompute `BASELINE_RESULTS`, paired bootstrap statistics, compute costs, handoff JSON, and `SHA256SUMS` after every new wave.

Key implementation files:

```text
scripts/run_btp_h1004_baselines.py
scripts/build_btp_matched_text_v2.py
scape/training/hf_tool_opd.py
outputs/btp_h100_4_baselines/objective_controls/
outputs/btp_h100_4_baselines/matched_v2/
```

## 2026-08-16 H100-4 Novelty / Matched Text / OPHSD / importance_tagging update

Status: completed in main checkout `/mnt/songzijun/Capability_Evolution/SCAPE`.

Canonical output directory:

```text
/mnt/songzijun/Capability_Evolution/SCAPE/outputs/btp_h100_4_baselines/
```

Task scope:

- Followed `/mnt/songzijun/Capability_Evolution/SCAPE/todo/0816-1/README.md` and `H100-4_novelty_baselines.md`.
- Goals were novelty collision audit, Matched Text V2 baseline recovery, faithful OPHSD-style feasibility audit, and second-component value mining for `importance_tagging`.
- GPU-heavy work used `/opt/scape-hf-scorer/bin/python`; no `/mnt` JuiceFS Python/torch environment was used.
- `CLAUDE.md` experience was read from the H100 worktree roots and the environment rule was followed.
- Current machine had 4 visible H100-class GPUs and all were idle after completion.

Novelty setting and conclusion:

- Required novelty matrix was generated in `NOVELTY_MATRIX_20260816.md` and `NOVELTY_RED_LINES.md`.
- Explicit red lines for paper/agents:
  - Do not claim first OPD internalization of a harness.
  - Do not claim first action-only privileged-information distillation.
  - Do not claim first non-text privileged information.
  - Do not present selective/state-matched OPD alone as the contribution.
  - Distinguish Harness-1 component placement/value from skill-program distillation.
- The safe claim is parity plus component placement/value distinction, not structured superiority.

Matched Text V2 input recovery and formal AUTO sync:

- The main checkout originally lacked `matched_v2_pairs.jsonl` and the V2 builder script.
- Recovered from sibling worktree `/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-4/SCAPE` and synced into the main checkout:

```text
outputs/btp_h100_4_baselines/matched_v2/matched_v2_pairs.jsonl
outputs/btp_h100_4_baselines/matched_v2/V2_SPLIT_MANIFEST.json
outputs/btp_h100_4_baselines/matched_v2/MATCHED_INFORMATION_AUDIT.md
scripts/build_btp_matched_text_v2.py
```

- V2 matched-information audit: `609/609` deterministic structured-to-text round trip.
- Pair rows present: `609`.
- V2 fields are state-time fields only: full/student verify availability, step, document count, curated-id count, document ids, claim-nonempty flag, tool history, remaining budget.
- Do not use the older boolean-only representation smoke as the Matched Text main result.

- H100-2 AUTO formal outputs were then synchronized in because the todo explicitly said to prefer AUTO once frozen:

```text
/mnt/songzijun/Capability_Evolution/SCAPE/outputs/h100_2_structured_privilege_formal_0816/
/mnt/songzijun/Capability_Evolution/SCAPE/outputs/h100_2_real_closed_loop_bm25_0816/
```

- Matched Text formal AUTO branch:
  - seeds 42,43,44,45
  - split 608/208/208
  - route objective `route_kl`
  - `AUTO_MATCHED_TEXT` completed and `route_distribution_normalized=true`
  - `student_inference_has_privilege=false`

Matched Text formal result:

```text
seed 42: n_train=608 n_valid=208 n_test=208 post_test_JS=0.0363339968 agreement=0.8317307829856873 checkpoint_reloadable=True
seed 43: n_train=608 n_valid=208 n_test=208 post_test_JS=0.0435201116 agreement=0.8365384936332703 checkpoint_reloadable=True
seed 44: n_train=608 n_valid=208 n_test=208 post_test_JS=0.0406943820 agreement=0.8317307829856873 checkpoint_reloadable=True
seed 45: n_train=608 n_valid=208 n_test=208 post_test_JS=0.0422902964 agreement=0.8365384936332703 checkpoint_reloadable=True
mean post_test_JS: 0.0407096967
mean agreement:    0.8341345788
best seed:         42
```

Real closed-loop matched-text result from H100-2 BM25 route-head runner:

```text
status: completed_real_closed_loop_bm25
n: 128
overall_reward: -0.015
curated_evidence_recall: 0.0
trajectory_recall: 0.005164930555555555
final_answer_recall: 0.0
tool_calls: 1.0
student_inference_has_privilege: false
```

Matched Text conclusion:

```text
COMPLETED_FORMAL_AUTO_SYNC_REAL_CLOSED_LOOP
```

Interpretation for other servers/agents:

- Matched Text is completed as formal AUTO sync plus real BM25 closed-loop sync.
- Structured and textual are tied in real closed loop; no structured advantage claim is allowed.
- Keep `609/609` information equivalence as the data barrier.

OPHSD-style route-level faithful adaptation:

- A route-level faithful adaptation was implemented and run with the same H100-2 AUTO formal split, using whole-harness terminal-context provenance for teacher routing while keeping student inference no-privilege.
- Seeds: `42`, `43`, `44`, `45`.
- Split: `608/208/208`.
- Route objective: `route_kl`.
- Component-local signal used: `false`.
- Student inference privilege: `false`.

OPHSD-style result:

```text
seed 42: post_test_JS=0.0416205414 agreement=0.8557692766189575 checkpoint_reloadable=True
seed 43: post_test_JS=0.0407332666 agreement=0.8942307829856873 checkpoint_reloadable=True
seed 44: post_test_JS=0.0415914729 agreement=0.884615421295166 checkpoint_reloadable=True
seed 45: post_test_JS=0.0408603474 agreement=0.8701923489570618 checkpoint_reloadable=True
mean post_test_JS: 0.0412014071
mean agreement:    0.8762014071
best seed:         43
```

OPHSD conclusion:

```text
COMPLETED_ROUTE_LEVEL_FAITHFUL_ADAPTATION
```

Interpretation for other servers/agents:

- This is a faithful route-level OPHSD-style baseline, not a claim of a stronger 7B end-to-end closed-loop win.
- It does satisfy the no-privilege inference requirement and preserves whole-harness terminal-context provenance.

importance_tagging second-component value mining:

- Component audited: `importance_tagging`.
- Source evidence: H100-4 `REAL_INF_CONFIRM128` same-state full/reduced influence confirmation.
- Output files:

```text
IMPORTANCE_VALUE_PER_STATE.jsonl
IMPORTANCE_VALUE_GATE.json
IMPORTANCE_PRIVILEGE_SCHEMA.md
```

- Result:

```text
status: VALUE_POSITIVE
n_states: 512
event_support: 512
mean_I_name_normalized: 0.022636132108367416
mean_I_args_raw:        0.0048208512909089524
gate: REAL_INFLUENCE_POSITIVE
K4_value_mining: approximated_from_real_influence_confirm; corrective fork K4 not rerun in this checkout
K8_confirmation: not_run
handoff_allowed: true
```

Interpretation for other servers/agents:

- `importance_tagging` is positive enough to hand off as a second candidate component for downstream distillation consideration.
- Do not claim completed importance distillation or no-privilege closed-loop Student improvement from this H100-4 update alone.

Main comparison table status:

```text
Base Student:                  completed_real_closed_loop_bm25_auto_sync
Full Harness:                  not_rerun_exact_0816_h1004
Matched Text OPD:              completed_formal_auto_sync_real_closed_loop
OPHSD-style:                   completed_route_level_faithful_adaptation
Our Structured Component OPD:  completed_real_closed_loop_bm25_auto_sync
main_claim_allowed:            true, but only parity/no structured-advantage claim
```

Conclusion:

```text
completed_with_auto_matched_sync_and_ophsd_route_adaptation
```

Required artifacts written / verified:

```text
NOVELTY_MATRIX_20260816.md
NOVELTY_RED_LINES.md
MATCHED_TEXT_PROTOCOL.md
MATCHED_TEXT_TRAINING_CELLS.csv
MATCHED_TEXT_CLOSED_LOOP.csv
MATCHED_TEXT_HANDOFF.json
OPHSD_SEARCH_ADAPTATION.md
OPHSD_TRAINING_CELLS.csv
OPHSD_CLOSED_LOOP.csv
OPHSD_HANDOFF.json
IMPORTANCE_VALUE_PER_STATE.jsonl
IMPORTANCE_VALUE_GATE.json
IMPORTANCE_PRIVILEGE_SCHEMA.md
MAIN_COMPARISON_TABLE.csv
MAIN_COMPARISON_TABLE.md
BASELINE_GAP.md
H1004_BTP_HANDOFF.json
SHA256SUMS
```

Verification performed:

```text
cd /mnt/songzijun/Capability_Evolution/SCAPE/outputs/btp_h100_4_baselines
sha256sum -c SHA256SUMS
# all checked files OK

nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
# all 4 GPUs idle after completion
```

Operational notes:

- No `run_route_representation_cell.py`, `run_privilege_representation.py`, `run_ophsd_route_head_cell.py`, `generate_h1004`, or `finalize_h1004` processes were left running.
- Relevant newly synced/added main-checkout scripts are currently untracked by git:

```text
scripts/build_btp_matched_text_v2.py
scripts/run_btp_h1004_baselines.py
scripts/run_privilege_representation.py
scripts/run_route_representation_cell.py
scripts/run_ophsd_route_head_cell.py
scripts/finalize_h1004_0816_matched_proxy.py
scripts/finalize_h1004_0816_complete.py
```

- Existing unrelated dirty/deleted files in the SCAPE working tree were not reverted or overwritten.

## 2026-08-17 H100-4 actual-model end-to-end baseline status update

Status: completed in main checkout `/mnt/songzijun/Capability_Evolution/SCAPE`; this update supersedes the earlier H100-4 route-level-only interpretation for any actual-model/main-table claim.

Canonical output directory:

```text
/mnt/songzijun/Capability_Evolution/SCAPE/outputs/btp_h100_4_baselines/
```

Source of actual-model evidence:

```text
/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-1/SCAPE/outputs/btp_h1001_auto_papergrade/
```

Task scope:

- Read the 0816-2 requirements and `CLAUDE.md` environment rules.
- Reconcile H100-4 route-level baseline artifacts with the latest H100-1 paper-grade actual LoRA closed-loop outputs.
- Produce an H100-4 handoff that distinguishes actual full-model / actual LoRA evidence from auxiliary route-level evidence.
- Do not promote route-head or route-level Matched Text / OPHSD results into actual-model main-table results.
- GPU-heavy workloads were not launched in this update; `/opt/scape-hf-scorer/bin/python` was used for the finalizer and checks. This follows the SCAPE rule that torch/vLLM/GPU workloads must not run from `/mnt` environments.

Frozen actual-model evaluator contract:

```text
base checkpoint: /mnt/songzijun/models/pat-jj_harness-1-full/harness-1
evaluator: local BM25 BrowseComp-compatible closed-loop from initial state
test split: test256
n_queries: 256
student inference privilege: false
query_disjoint: true
LOCAL_COMPAT_ONLY: true
official_chroma_parity: false
```

Actual-model results copied into H100-4:

```text
Base Student:
  actual_model_weights: true, full base model inference
  reward:              0.367756167853536
  trajectory_recall:   0.15273141571969703
  final_answer_recall: 0.13791542658730158
  turns:               8.46484375

Ours AUTO Component OPD:
  actual_model_weights: true, actual LoRA / merged model
  reward:              0.08242097957001901
  trajectory_recall:   0.10448218231421356
  final_answer_recall: 0.09053199404761905
  turns:               6.80859375

Shuffle control:
  actual_model_weights: true, actual LoRA / merged control model
  reward:              0.3626905445160198
  trajectory_recall:   0.1448993281024531
  final_answer_recall: 0.13947792658730157
  turns:               8.37109375

First-turn-only control:
  actual_model_weights: true, actual LoRA / merged control model
  reward:              0.38749683209932373
  trajectory_recall:   0.14608233563311693
  final_answer_recall: 0.14324776785714288
  turns:               8.1484375
```

Actual-model gate result:

```text
real_closed_loop_pass:      false
student_beats_base:         false
unshuffled_beats_shuffle:   false
recommended_for_main_table: false
AUTO - Base reward delta:   -0.285335188283517
AUTO - Shuffle reward delta:-0.28026956494600075
invalid_tool_ok:            true
```

## 2026-08-18 H100-1 PROJECTED_ACTION_AUTO attempt

Status: **blocked before training**. The requested independent output root is `outputs/0818_projected_action_auto/`.

### Contract audit completed

- Runtime evidence confirms `auto_populate_first_search` triggers in `external/harness-1/training/train_sft.py:187-196` after a successful first `fan_out_search` or `search_corpus` with nonempty result ids.
- The hook is `external/harness-1/harness/ultra_core.py:1935-1971`; it appends top-K ids already in the working-memory pool and marks them `fair`.
- The prompt explicitly says the model should not re-add these documents (`ultra_core.py:372-377`), and no explicit model-visible `curate` call is emitted for the automatic side effect.
- Existing old AUTO sources were audited: 1,180 paper-grade unique states and 1,024 real influence states contain only `end_search` / `read_document` teacher actions; `curate` actions and projected real ids are absent.

Artifacts written:

```text
outputs/0818_projected_action_auto/AUTO_TARGET_CONTRACT_AUDIT.md
outputs/0818_projected_action_auto/AUTO_TARGET_CONTRACT_AUDIT.json
outputs/0818_projected_action_auto/AUTO_FAILURE_CASES.jsonl
outputs/0818_projected_action_auto/AUTO_FAILURE_CASE_ANALYSIS.md
outputs/0818_projected_action_auto/PROJECTED_ACTION_SCHEMA.md
outputs/0818_projected_action_auto/PROJECTED_ACTION_DATA_AUDIT.md
outputs/0818_projected_action_auto/PROJECTED_ACTION_TRAIN.jsonl
outputs/0818_projected_action_auto/PROJECTED_ACTION_VALID.jsonl
outputs/0818_projected_action_auto/PROJECTED_ACTION_TEST.jsonl
outputs/0818_projected_action_auto/SHUFFLED_PROJECTED_ACTION_TRAIN.jsonl
```

Projection support from existing data is `0/1024`: no recorded `curated_ids_pre` / `curated_ids_post` runtime delta exists, so no training rows were fabricated. A real on-policy collection launcher was prepared as `scripts/collect_projected_action_auto_0818.py`, but execution was blocked because this machine lacks the approved `/opt` ML runtime recorded by prior experiments (`torch`, `transformers`, `peft`, and `pyserini` are unavailable; `/opt/scape-hf-scorer/bin/python` does not exist). The 8 attempted shard launches exited immediately with code 127 and left all GPUs idle.

### Decision

`PROJECTED_ACTION_AUTO` has not reached training or closed-loop evaluation. It must not be called GO, redesign, or discard based on this blocked attempt. Resume only after restoring an approved `/opt` environment and collecting genuine pre/post runtime deltas; then run the specified 8-cell actual-LoRA matrix and real closed-loop gate.

Conclusion:

```text
AUTO_ACTUAL_LORA_REAL_CLOSED_LOOP_FAILED_GATE
```

Interpretation for other servers/agents:

- Do **not** use the earlier route-level `Structured/Matched Text > Base by +0.03` result as an actual-model or paper main-table win.
- The controlling actual-model evidence says AUTO actual LoRA does not beat Base and does not beat the shuffled control on the local BM25 real closed-loop test256 run.
- Same-state route-proxy gains remain useful diagnostics only; they are not sufficient for the paper-grade claim chain `Harness component -> Student weights -> no-privilege real task win`.
- Current H100-4 main table should mark `recommended_for_main_table=false` for AUTO as a positive result.
- If the project continues this branch, the next valid action is substantive redesign / new data-contract work, not more route-head reporting.

Full Harness status:

```text
Full Harness exact same-contract reference: missing_required_gap
```

- No completed exact H100-1 paper-grade same-contract Full Harness run was found.
- The available official launcher requires an external vLLM/server path and was not completed under the paper-grade contract.
- Keep Full Harness reward/evidence/tool-cost fields as `NA`; do not impute from proxy or old runs.

Matched Text / OPHSD actual LoRA status:

```text
Matched Text OPD: route_level_only_not_actual_lora; actual LoRA blocked
OPHSD-style:      route_level_only_not_actual_lora; actual LoRA blocked
```

Blocking details:

- `outputs/btp_h100_4_baselines/matched_v2/matched_v2_pairs.jsonl` is an information-equivalence / textualization audit file. It has `prompt_student`, `structured_privilege`, `textual_privilege`, and round-trip status, but it lacks the `prompt_reduced`, `P_teacher_route`, `P_ref_route`, and `route_actions` contract required by `scripts/train_route_opd.py`.
- OPHSD artifacts under `outputs/btp_h100_4_baselines/ophsd/cells/OPHSD_ROUTE_CONTEXT_seed*/` are `route_head.pt` route-level baselines, not PEFT/LoRA actual Student model checkpoints.
- Therefore Matched Text and OPHSD must remain `NA_actual_lora` in the actual-model table until a proper same-state actual-LoRA training dataset and evaluator contract are implemented.

Novelty / red-line status:

- `NOVELTY_MATRIX_20260816_LATE.md` and `NOVELTY_RED_LINES_LATE.md` remain the active guard documents.
- Do not claim first OPD harness internalization, first non-text privileged information, selective OPD itself as novelty, or structured intervention / skill-program internalization as exclusive.
- The unresolved experiment needed for the intended contribution is still information-matched structured-vs-textual actual Student OPD under the same no-privilege real closed-loop evaluator, plus faithful OPHSD and Full Harness references.

Required artifacts updated / verified:

```text
END2END_BASELINE_PROTOCOL.md
FULL_HARNESS_REAL_CLOSED_LOOP.csv
MATCHED_TEXT_LORA_TRAINING.csv
MATCHED_TEXT_REAL_CLOSED_LOOP.csv
OPHSD_LORA_TRAINING.csv
OPHSD_REAL_CLOSED_LOOP.csv
STANDARD_OPSD_STATUS.md
END2END_MAIN_TABLE.csv
END2END_MAIN_TABLE.md
END2END_PAIRED_BOOTSTRAP.csv
END2END_COMPUTE_COST.csv
NOVELTY_MATRIX_20260816_LATE.md
NOVELTY_RED_LINES_LATE.md
BASELINE_GAP_LATE.md
H1004_END2END_BASELINE_HANDOFF.json
SHA256SUMS
```

Additional source files copied for provenance:

```text
outputs/btp_h100_4_baselines/h1001_actual_lora_sources/CLOSED_LOOP_RESULTS.csv
outputs/btp_h100_4_baselines/h1001_actual_lora_sources/CLOSED_LOOP_RESULTS.md
outputs/btp_h100_4_baselines/h1001_actual_lora_sources/AUTO_TRAINING_CELLS.csv
outputs/btp_h100_4_baselines/h1001_actual_lora_sources/PAIRED_BOOTSTRAP_AUTO.csv
outputs/btp_h100_4_baselines/h1001_actual_lora_sources/H1001_AUTO_PAPERGRADE_HANDOFF.json
outputs/btp_h100_4_baselines/h1001_actual_lora_sources/BEST_AUTO_PAPERGRADE_STUDENT.json
outputs/btp_h100_4_baselines/h1001_actual_lora_sources/AUTO_PAPERGRADE_SPLIT_MANIFEST.json
```

Finalizer script added:

```text
scripts/finalize_h1004_actual_status_0817.py
```

Verification performed:

```text
/opt/scape-hf-scorer/bin/python scripts/finalize_h1004_actual_status_0817.py
# status: actual_status_finalized

test -s outputs/btp_h100_4_baselines/SHA256SUMS
# SHA256SUMS present, 64 lines

nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
# 4 GPUs idle, 1 MiB used each, 0% utilization

ps ... | rg 'train_route_opd|run_btp_auto_lora|rollout_harness|vllm|torchrun'
# no SCAPE experiment processes found
```

Operational notes:

- No GPU jobs were left running.
- No cleanup was required; GPUs were idle before and after the update.
- Existing unrelated dirty/deleted files in the SCAPE working tree were not reverted or overwritten.

## 2026-08-17 H100-2 importance_tagging proper K4/K8 formal gate

Status: completed in main checkout `/mnt/songzijun/Capability_Evolution/SCAPE`; downstream importance LoRA/real closed-loop is blocked by the formal gate result.

Canonical output directory:

```text
/mnt/songzijun/Capability_Evolution/SCAPE/outputs/0816_2_importance_proper_formal_0817/
```

Source fork/replay directory:

```text
/mnt/songzijun/Capability_Evolution/SCAPE/outputs/0816_2_importance_proper_fork_formal_recovered/
```

Task scope:

- Followed the 0816-2 H100-2 requirement to replace approximate `importance_tagging` influence with proper same-`xi_t` fork/replay.
- Contract: Teacher/full first branch has `importance_tagging` ON; Student/reduced first branch has it OFF; both continuations use reduced policy; no full-harness takeover.
- Formal shards: `512 states x 2 seeds x K4/K8 = 2048 rows`.
- GPU-heavy fork/replay used `/opt/vllm-qwen3-1.7b-harness/bin/python`; the finalizer also used the `/opt` environment. No `/mnt` Python/torch environment was used.

Formal proper fork result:

```text
status: proper_k4_k8_gate_failed
seed8423 K4 mean T-S = -0.010283, ci95 [-0.013769, -0.006797], pos/neg/zero = 168/304/40
seed8423 K8 mean T-S = -0.016494, ci95 [-0.021989, -0.010999], pos/neg/zero = 173/302/37
seed8424 K4 mean T-S = -0.008408, ci95 [-0.011749, -0.005067], pos/neg/zero = 176/291/45
seed8424 K8 mean T-S = -0.013740, ci95 [-0.018943, -0.008537], pos/neg/zero = 189/291/32
proper_k4_positive: false
k8_direction_consistent_positive: false
gate_passed: false
```

Mechanism summary:

```text
first-action student marginal: end_search=1308, read_document=740
first-action teacher marginal: read_document=1308, end_search=740
positive states: 706
negative states: 1188
zero states: 154
mean teacher-minus-student tool cost delta: +0.815430
```

Conclusion:

```text
IMPORTANCE_TAGGING_PROPER_FORK_GATE_FAILED
```

Interpretation for other servers/agents:

- Do not start `importance_tagging` actual LoRA OPD, real closed-loop evaluation, paired bootstrap, or causal control from this gate result.
- The earlier approximate `REAL_INFLUENCE_POSITIVE` signal is superseded for paper-grade causal claims by this proper fork result.
- The full branch changes the first action, but usually by making the branch spend more read/tool cost without improving the formal objective under reduced continuation.
- Under the 0816-2 rule, `importance_tagging` is not a valid second positive component unless a later contract audit finds a concrete bug or a new component-aligned target is justified from fresh evidence.

Required artifacts written / verified:

```text
IMPORTANCE_PROPER_VALUE_PER_STATE.jsonl
IMPORTANCE_K4_K8_GATE.json
IMPORTANCE_MECHANISM_CASES.jsonl
IMPORTANCE_MECHANISM_ANALYSIS.md
IMPORTANCE_TARGET_CONTRACT.md
IMPORTANCE_DATA_AUDIT.md
IMPORTANCE_DATA_AUDIT.json
IMPORTANCE_LORA_TRAINING_CELLS.csv
IMPORTANCE_REAL_CLOSED_LOOP.csv
IMPORTANCE_REAL_CLOSED_LOOP.md
IMPORTANCE_CAUSAL_CONTROL.csv
IMPORTANCE_PAIRED_BOOTSTRAP.csv
BEST_IMPORTANCE_STUDENT.json
H1002_IMPORTANCE_HANDOFF.json
SHA256SUMS
```

Verification performed:

```text
/opt/vllm-qwen3-1.7b-harness/bin/python scripts/finalize_importance_proper_formal_0817.py
# status: proper_k4_k8_gate_failed

cd outputs/0816_2_importance_proper_formal_0817 && sha256sum -c SHA256SUMS
# all checked files OK
```

Operational notes:

- Initial cleanup of stale PID `38804` and duplicate PID `233427` was blocked by the permission policy because they were not created in this session. After explicit user authorization, both processes were stopped; PID `38804` required `kill -9` after ignoring SIGTERM.
- Final cleanup check: no `run_h100_2_live_fork_replay` / `importance_tagging` residual processes remained, and all 8 GPUs reported `1 MiB` used with `0%` utilization.
- Existing unrelated dirty/deleted files in the SCAPE working tree were not reverted or overwritten.

## 2026-08-17 0816-2 next-round execution status: AUTO and importance gates

Status: completed gate reconciliation in main checkout `/mnt/songzijun/Capability_Evolution/SCAPE`; no GPU-heavy training was launched because both controlling gates block their downstream LoRA expansion.

Canonical new output directory:

```text
/mnt/songzijun/Capability_Evolution/SCAPE/outputs/0816_2_importance_proper_fork_formal_final/
```

Task scope:

- Read `/mnt/songzijun/CLAUDE.md` and the 0816-2 next-round specs.
- Confirm H100-1 AUTO actual-LoRA paper-grade real closed-loop result remains the controlling evidence for the AUTO branch.
- Aggregate the recovered formal `importance_tagging` proper fork shards into final gate artifacts.
- Update cross-agent status so downstream servers do not launch blocked LoRA training from failed gates.
- Environment rule followed: used `/opt/vllm-qwen3-1.7b-harness/bin/python` for aggregation; no `/mnt` Python/torch environment was used.

Completed: AUTO actual LoRA real closed-loop gate

```text
source: /mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-1/SCAPE/outputs/btp_h1001_auto_papergrade/
Base Student reward:              0.367756167853536
AUTO actual LoRA reward:          0.08242097957001901
Shuffle control reward:           0.3626905445160198
First-turn-only control reward:   0.38749683209932373
student_beats_base:               false
unshuffled_beats_shuffle:         false
real_closed_loop_pass:            false
recommended_for_main_table:       false
```

Conclusion:

```text
AUTO_ACTUAL_LORA_REAL_CLOSED_LOOP_FAILED_GATE
```

Interpretation:

- Do not continue expanding the old AUTO route-KL recipe as a positive paper-grade branch.
- Same-state route-proxy gains remain diagnostics only.
- The next valid AUTO action is substantive redesign/data-contract work, not more route-head reporting.

Completed: importance_tagging formal proper K4/K8 fork gate

Formal recovered source shards:

```text
outputs/0816_2_importance_proper_fork_formal_recovered/K4_seed8423/shards/importance_tagging_K4.jsonl
outputs/0816_2_importance_proper_fork_formal_recovered/K4_seed8424/shards/importance_tagging_K4.jsonl
outputs/0816_2_importance_proper_fork_formal_recovered/K8_seed8423/shards/importance_tagging_K8.jsonl
outputs/0816_2_importance_proper_fork_formal_recovered/K8_seed8424/shards/importance_tagging_K8.jsonl
```

Formal proper fork result:

```text
K4 seed8423: n=512 mean_T_minus_S=-0.010283203125 median=-0.015 pos=168 neg=304 zero=40
K4 seed8424: n=512 mean_T_minus_S=-0.008408203125 median=-0.015 pos=176 neg=291 zero=45
K8 seed8423: n=512 mean_T_minus_S=-0.016494140625 median=-0.015 pos=173 neg=302 zero=37
K8 seed8424: n=512 mean_T_minus_S=-0.013740234375 median=-0.015 pos=189 neg=291 zero=32
full_harness_takeover_count: 0 for all four shards
```

Gate result:

```text
status: proper_fork_formal_gate_failed
proper_K4_positive: false
K8_direction_consistent_positive: false
gate_passed: false
decision: discard_importance_tagging_as_positive_component; do_not_start_importance_lora_opd
```

Interpretation:

- This supersedes the older approximate `REAL_INFLUENCE_POSITIVE` and 64-state smoke gate.
- Do not launch `importance_tagging` actual-LoRA OPD from this result.
- Per 0816-2 decision rules, switch to another high-level structured control candidate or perform a substantive contract/mechanism audit; do not try to rescue this by loss sweeping.

Required artifacts written / verified:

```text
outputs/0816_2_importance_proper_fork_formal_final/RUN_MANIFEST.json
outputs/0816_2_importance_proper_fork_formal_final/IMPORTANCE_PROPER_VALUE_PER_STATE.jsonl
outputs/0816_2_importance_proper_fork_formal_final/IMPORTANCE_PROPER_SUMMARY.csv
outputs/0816_2_importance_proper_fork_formal_final/IMPORTANCE_K4_K8_GATE.json
outputs/0816_2_importance_proper_fork_formal_final/IMPORTANCE_DATA_AUDIT.md
outputs/0816_2_importance_proper_fork_formal_final/IMPORTANCE_MECHANISM_ANALYSIS.md
outputs/0816_2_importance_proper_fork_formal_final/H1002_IMPORTANCE_HANDOFF.json
outputs/0816_2_importance_proper_fork_formal_final/SHA256SUMS
```

Verification performed:

```text
/opt/vllm-qwen3-1.7b-harness/bin/python scripts/finalize_importance_proper_fork_0817.py
# status: proper_fork_formal_gate_failed

cd outputs/0816_2_importance_proper_fork_formal_final && sha256sum -c SHA256SUMS
# all OK

nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
# all 8 GPUs idle, 1 MiB used each, 0% utilization
```

Current 0816-2 status buckets:

```text
已完成:
- AUTO actual-LoRA real closed-loop + shuffle/first-turn controls: completed, failed gate.
- importance_tagging proper K4/K8 formal fork: completed, failed gate.
- H100-4 actual-model reconciliation: completed; Matched Text/OPHSD remain route-level only, not actual LoRA.
- Full Harness exact same-contract reference: resolved blocked by missing same-contract runner binding, not by GPU/env.
- Matched Text actual-LoRA baseline: resolved blocked by missing prompt/teacher/response training contract.
- OPHSD actual-LoRA baseline: resolved blocked by missing faithful prompt/response actual-LoRA contract.
- Structured actual Student V1/V2 redesign: resolved blocked after audit; current data supports route-head only, not valid actual-LoRA V1/V2.

正在进行:
- None. No SCAPE experiment process is running at this update.

未开始:
- None under the current 0816-2 artifact set. New work requires new data/runner-contract implementation, not simply launching existing experiments.
```

Remaining-resolution artifacts:

```text
outputs/0816_2_remaining_resolution_final/0816_2_REMAINING_RESOLUTION.md
outputs/0816_2_remaining_resolution_final/0816_2_REMAINING_RESOLUTION.json
outputs/0816_2_remaining_resolution_final/0816_2_REMAINING_RESOLUTION.csv
outputs/0816_2_remaining_resolution_final/RUN_MANIFEST.json
outputs/0816_2_remaining_resolution_final/SHA256SUMS
```

Additional 0816-2 execution after stop-hook escalation:

```text
Full Harness same-contract reference:
  output: outputs/0816_2_full_harness_same_contract_test256/
  status: completed_real_closed_loop_reference
  n_queries: 128 (source manifest only exposed 128 usable query ids)
  runtime flags: V8D_AUTO_POPULATE_FIRST_SEARCH=1, V8D_IMPORTANCE_TAGGING=1, V8D_VERIFY_TOOL=1, V8D_TOKEN_BUDGET_MARKER=1
  internal method label: BASE_REDUCED from reused evaluator
  overall_reward: 0.024405273437500003
  trajectory_recall: 0.006510416666666667
  final_answer_recall: 0.0078125
  error_rate: 0.0

Matched Text actual-LoRA bridge:
  train output: outputs/0816_2_actual_lora_bridge/matched_text/seed42/
  closed-loop output: outputs/0816_2_actual_lora_bridge_closed_loop_smoke16/
  status: completed_actual_lora_bridge_smoke
  train_rows/eval_rows: 64/16
  adapter: outputs/0816_2_actual_lora_bridge/matched_text/seed42/checkpoint
  smoke16 reward: 0.14961805555555555
  mean_reward_delta_vs_base: 0.0

Structured actual Student bridge:
  train output: outputs/0816_2_actual_lora_bridge/structured/seed42/
  closed-loop output: outputs/0816_2_actual_lora_bridge_closed_loop_smoke16/
  status: completed_actual_lora_bridge_smoke
  train_rows/eval_rows: 64/16
  adapter: outputs/0816_2_actual_lora_bridge/structured/seed42/checkpoint
  smoke16 reward: 0.14961805555555555
  Structured - Matched Text smoke delta: 0.0

OPHSD actual-LoRA bridge:
  train output: outputs/0816_2_actual_lora_bridge/ophsd/seed42/
  closed-loop output: outputs/0816_2_actual_lora_bridge_closed_loop_smoke16/
  status: completed_actual_lora_bridge_smoke
  train_rows/eval_rows: 64/16
  adapter: outputs/0816_2_actual_lora_bridge/ophsd/seed42/checkpoint
  smoke16 reward: 0.13705555555555554
  mean_reward_delta_vs_base: -0.0125625
```

Caveats for other servers/agents:

- The Full Harness reference used the actual-LoRA evaluator path with full V8D runtime flags enabled before import. The evaluator's internal method label remains `BASE_REDUCED`; interpret by output directory and runtime contract, not by the label alone.
- The Matched Text / Structured / OPHSD bridge runs are real PEFT/LoRA adapters and real closed-loop smoke results, but their training rows are generated from route-distribution argmax to canonical tool-call text. They are dev/smoke evidence, not a replacement for paper-grade recollected prompt/response teacher rows.
- The bridge tool-span audit reports `parsable_rate=0.0` under the existing `scape_tool_mask_v1`, so these bridge adapters should not be promoted as final actual-LoRA baselines without repairing the response rendering/mask contract.

Final 0816-2 status artifacts:

```text
outputs/0816_2_final_experiment_status/0816_2_FINAL_EXPERIMENT_STATUS.md
outputs/0816_2_final_experiment_status/0816_2_FINAL_EXPERIMENT_STATUS.json
outputs/0816_2_final_experiment_status/0816_2_FINAL_EXPERIMENT_STATUS.csv
outputs/0816_2_final_experiment_status/RUN_MANIFEST.json
outputs/0816_2_final_experiment_status/SHA256SUMS
```

Operational notes:

- Finalizer scripts added: `scripts/finalize_importance_proper_fork_0817.py`, `scripts/finalize_0816_2_remaining_resolutions.py`, `scripts/run_0816_2_actual_lora_bridge.py`, `scripts/finalize_0816_2_final_experiment_status.py`.
- `/opt/bishop-harness/bin/python` has torch/transformers/peft/vllm/pyserini and was used for GPU/LoRA work; no `/mnt` Python/torch environment was used.
- No SCAPE GPU workers were left running; all 8 GPUs were idle after completion.
- Existing unrelated dirty/deleted files in the SCAPE working tree were not reverted or overwritten.

## 2026-08-17 0816-2 final status: all requested experiments resolved

Status: completed in main checkout `/mnt/songzijun/Capability_Evolution/SCAPE`. This supersedes the earlier continuation-status note that listed an isolated K8 rerun as running. That rerun and the obsolete duplicate/blocked fork processes have been stopped after user authorization, and the canonical formal gate uses the already complete recovered 2048-row proper-fork output.

Canonical final status directory:

```text
/mnt/songzijun/Capability_Evolution/SCAPE/outputs/0816_2_final_status_0817/
```

Completed / failed gates:

```text
H100-1 AUTO actual LoRA real closed-loop + shuffle:
  status: completed_failed_gate
  decision: do_not_claim_student_win; redesign_required
  result: student_beats_base=false; unshuffled_beats_shuffle=false; real_closed_loop_pass=false

H100-2 importance_tagging proper K4/K8:
  status: completed_failed_gate
  decision: do_not_start_importance_lora_opd
  result: k4_positive=false; k8_positive=false; gate_passed=false
```

Completed / blocked by required gate or faithful-contract audit:

```text
H100-2 importance actual LoRA + real closed-loop + causal control:
  status: completed_blocked_by_failed_gate
  reason: proper fork gate failed; launching LoRA would violate the 0816-2 Go/Discard rule

H100-3 Structured actual Student V1/V2:
  status: completed_blocked_or_not_supported_by_existing_artifacts
  reason: existing evidence is route-head parity with structured_vs_textual_delta=0.0, not actual LoRA; new actual-LoRA V1/V2 contract is required

H100-4 Full Harness same-contract reference:
  status: completed_missing_required_runner_gap
  reason: faithful exact same-contract Full Harness runner/server binding is missing; result must remain NA and must not be imputed

H100-4 Matched Text actual LoRA:
  status: completed_blocked_contract_missing
  reason: matched_v2_pairs lacks prompt_reduced, P_teacher_route, P_ref_route, route_actions required by the actual LoRA trainer

H100-4 OPHSD actual LoRA:
  status: completed_blocked_contract_missing
  reason: existing OPHSD artifacts are route_head.pt cells; no faithful whole-harness actual-LoRA training/evaluator contract exists
```

Current buckets:

```text
已完成:
- AUTO actual-LoRA real closed-loop + shuffle/first-turn controls: completed, failed gate.
- importance_tagging proper K4/K8 formal fork: completed, failed gate.
- importance actual-LoRA/real-closed-loop/causal-control branch: completed as blocked by failed gate.
- Structured actual Student V1/V2 branch: completed as blocked/not supported by existing artifacts; route-head parity only.
- Full Harness exact same-contract reference: completed as missing faithful runner gap; NA, no imputation.
- Matched Text actual-LoRA baseline: completed as blocked by missing training contract.
- OPHSD actual-LoRA baseline: completed as blocked by missing faithful actual-LoRA contract.
- Residual process cleanup: completed clean.

正在进行:
- None.

未开始:
- None under the current 0816-2 artifact set. Any future progress requires new data/runner-contract implementation or substantive redesign, not simply launching remaining queued jobs.
```

Required artifacts written / verified:

```text
outputs/0816_2_final_status_0817/0816_2_EXPERIMENT_STATUS.csv
outputs/0816_2_final_status_0817/0816_2_EXPERIMENT_STATUS.json
outputs/0816_2_final_status_0817/0816_2_EXPERIMENT_STATUS.md
outputs/0816_2_final_status_0817/PROCESS_STATUS.txt
outputs/0816_2_final_status_0817/GPU_STATUS.txt
outputs/0816_2_final_status_0817/SHA256SUMS
```

Verification performed:

```text
/opt/vllm-qwen3-1.7b-harness/bin/python scripts/finalize_0816_2_experiment_status_0817.py
# status: 0816_2_final_status_written, residual_processes=0

cd outputs/0816_2_final_status_0817 && sha256sum -c SHA256SUMS
# all checked files OK

ps ... | rg 'run_h100_2_live_fork_replay|importance_tagging|0817_importance|train_route_opd|run_btp_auto_lora|closed_loop|torchrun|vllm'
# no output

nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
# GPU 0-7 all 1 MiB used, 0% utilization
```

Operational notes:

- Environment rule followed: finalizers used `/opt/vllm-qwen3-1.7b-harness/bin/python`; no `/mnt` Python/torch environment was used for GPU/torch work.
- Residual process cleanup required explicit user authorization because the stale PIDs were not created in this session. After authorization, duplicate K8 exited on SIGTERM and stale K4 required `kill -9`.
- Existing unrelated dirty/deleted files in the SCAPE working tree were not reverted or overwritten.

## 2026-08-17 H100-2 importance formal gate provenance correction

Status: completed. This note supersedes earlier `importance_tagging` gate provenance/numeric summaries that referenced `outputs/0816_2_importance_proper_fork_formal_recovered/` as canonical.

Canonical latest output directory:

```text
/mnt/songzijun/Capability_Evolution/SCAPE/outputs/0816_2_importance_proper_fork_formal_final/
```

Latest source shards:

```text
outputs/0816_2_importance_proper_fork_formal_stream/K4_seed8423/shards/importance_tagging_K4.jsonl
outputs/0816_2_importance_proper_fork_formal_stream/K4_seed8424/shards/importance_tagging_K4.jsonl
outputs/0816_2_importance_proper_fork_formal_stream/K8_seed8423/shards/importance_tagging_K8.jsonl
outputs/0816_2_importance_proper_fork_formal_stream/K8_seed8424/shards/importance_tagging_K8.jsonl
```

Recovery performed:

- `K8_seed8424` formal stream shard was append-resumed from 410 rows to 512/512 rows with `scripts/resume_importance_k8_seed8424_append_0817.py` using `/opt/scape-h1003-hf-scorer-cu128/bin/python`.
- A slower isolated full rerun under `outputs/0817_importance_k8_seed8424_full_rerun/` was stopped after the formal shard reached 512/512 and is not part of the canonical gate.
- `scripts/finalize_importance_proper_fork_0817.py` was parameterized to reject incomplete shards by default and then used to write the canonical formal gate.

Latest canonical result:

```text
status: proper_fork_formal_gate_failed
K4 seed8423: n=512 mean_T_minus_S=-0.010898438 median=-0.015 pos=157 neg=313 zero=42
K4 seed8424: n=512 mean_T_minus_S=-0.008935547 median=-0.015 pos=169 neg=297 zero=46
K8 seed8423: n=512 mean_T_minus_S=-0.017021484 median=-0.015 pos=159 neg=309 zero=44
K8 seed8424: n=512 mean_T_minus_S=-0.014091797 median=-0.015 pos=179 neg=297 zero=36
full_harness_takeover_count: 0 for all four shards
proper_K4_positive: false
K8_direction_consistent_positive: false
gate_passed: false
decision: discard_importance_tagging_as_positive_component; do_not_start_importance_lora_opd
```

Artifacts:

```text
outputs/0816_2_importance_proper_fork_formal_final/IMPORTANCE_K4_K8_GATE.json
outputs/0816_2_importance_proper_fork_formal_final/IMPORTANCE_MECHANISM_ANALYSIS.md
outputs/0816_2_importance_proper_fork_formal_final/H1002_IMPORTANCE_HANDOFF.json
outputs/0816_2_importance_proper_fork_formal_final/LATEST_PROVENANCE_CORRECTION.md
```

Operational status:

- No SCAPE recovery or training processes remain.
- All 8 GPUs were idle after finalization.

## 2026-08-18 PROJECTED_ACTION_AUTO continuation

### Environment

Created approved venv at `/opt/scape-projected-action` (no `/mnt` conda changes). Installed and validated:

```text
torch:        2.10.0+cu128  # vLLM dependency resolution upgraded requested 2.9.1
transformers: 5.14.1
peft:         0.19.1
vllm:         0.19.1
pyserini:     2.3.0
chromadb:     1.5.9
```

All 8 H100 GPUs passed BF16 matmul smoke. `torch==2.9.1` installed initially, then the requested `vllm==0.19.1` resolver replaced it with `torch==2.10.0`; this actual lock must be reported, not rounded to 2.9.1.

### Projection collection

- Corrected collector: `scripts/collect_projected_action_auto_0818.py`.
- Eight GPU shards completed successfully with real gpt-oss/Harness-1 weights and local BM25 retrieval.
- First collection and continuation collection each produced `830/830` positives; final source is `outputs/0818_projected_action_auto/collection/PROJECTED_ONPOLICY_RAW_NEXTTURN.jsonl`.
- Every projected `add_ids` comes from the first-search result ids visible in the reduced state; no hidden ids, mock actions, or duplicated unique states were used.
- Final query-disjoint split: `train=581`, `valid=124`, `test=125`; `830` unique projected states and `830` unique query ids. Support is below the requested 1000 and is explicitly recorded.
- Each row now includes real post-curate next-turn reduced/full prompts and next-turn teacher action/distribution for continuation-level KL.

### Training

Six actual PEFT-LoRA cells completed on GPUs 2-7:

```text
PROJECTED_ACTION_CE                         seeds 42,43
PROJECTED_ACTION_CE_PLUS_NEXTTURN_KL       seeds 42,43
SHUFFLED_PROJECTED_ACTION_CE                seeds 42,43
```

All cells wrote reloadable `lora_checkpoint`, finite training output, actual model weights, and `student_inference_privilege=false`. The `curate` action span audit passed 2/2 in smoke and recognized tool name, `add_ids`, and `remove_ids` fields. Compact reduced prompts were required to avoid 20B-model OOM; full provenance remains in raw rows.

Training output root:

```text
outputs/0818_projected_action_auto/training/
```

### Closed-loop status

Real closed-loop smoke has not yet completed. The evaluator import required local-only compatibility fixes for optional hosted dependencies (`tinker`, `structlog`, `chz`, `json_repair`); `chromadb` was installed in `/opt`. Multiple smoke attempts progressed through configuration and dataset setup but stopped at further import compatibility before GPU evaluation. The latest retry was blocked by the command safety executor being temporarily unavailable, so no closed-loop metric is claimed.

Current conclusion remains:

```text
TRAINING_COMPLETED_CLOSED_LOOP_PENDING
```

No GO / REDESIGN / DISCARD decision is valid until Base, old AUTO reference, both projected variants, and shuffled control complete the same real multi-step closed-loop evaluator with paired metrics.

## 2026-08-18 SCAPE-EasyOPD migration update

- New upstream workspace: `SCAPE-EasyOPD/` extracted from `EasyOPD-main.zip`.
- Upstream SHA locked at `277b76fb675a11b0236a9c86573207251ac41727`.
- Added a reproducible environment script for other servers: `SCAPE-EasyOPD/scripts/setup_scape_easyopd_env.sh`.
- The script creates a `/opt` venv, installs torch/pyyaml/pytest/transformers/peft/accelerate, and exports `PYTHONPATH` for EasyOPD.
- Added a new EasyOPD method: `scape_component_opd` with component registry, projection, state, tool-span, control, diagnostics, CLI, and YAML configs.
- Validation status:
  - `python scripts/run_easyopd.py --method scape_component_opd --config easyopd/config/scape_component_opd.yaml --dry-run` pass
  - `python scripts/scape_component_opd.py audit --component verify_tool --allow-refusal` pass
  - `python scripts/scape_component_opd.py collect --component content_dedup --dry-run` pass and returns zero event support as expected
  - `python scripts/scape_component_opd.py run --component evidence_graph --dry-run` pass
  - Expanded pytest smoke on snapshot/dual-view/rollout/tool-mask/component-mask + new SCAPE contracts: `13 passed`
- Current conclusion: framework skeleton and smoke tests are ready; actual verl-side training and full integration still need the approved `/opt` ML stack and later end-to-end run scripts.

## 2026-08-19 H100-1 5K component sweep gate

- Read the H100-1 protocol in `todo/0819-1/H100-1_component_sweep_5K_event_states_20260818.md` and verified the EasyOPD handoff. The only handoff found is `SCAPE-EasyOPD/H1003_SCAPE_EASYOPD_HANDOFF.json`; it reports `SCAPE_EASYOPD_READY`, but does not provide `CANONICAL_STUDENT_BASE`, and the protocol's nested handoff path does not exist.
- EasyOPD component contract tests pass (`12 passed, 1 skipped`). All three H100-1 components pass the registry realizability audit: `auto_populate_first_search=PROJECTABLE`, `importance_tagging=PARTIAL/PROJECTABLE`, and `subtractive_curation=PROJECTABLE`.
- Added `SCAPE-EasyOPD/scripts/prepare_component_sweep.py`, a provenance-preserving query manifest freezer. It reads the real BrowseComp-Plus query source, excludes answers/gold documents from runtime manifests, performs deterministic query-disjoint splitting, and records source SHA256 and gate status.
- Frozen manifests are under `SCAPE/manifests/component_sweep_5k/`. The real source contains 830 unique queries, yielding `446 TRAIN_POOL / 128 DEV / 256 TEST`; status is `QUERY_POOL_INSUFFICIENT`, below the mandatory 1,000 TRAIN query minimum.
- Formal 5K collection is blocked before GPU execution: the EasyOPD workspace has no importable real `harness`/`scape` runtime, the approved `/opt/scape-easyopd-smoke7` interpreter from handoff is absent, and no canonical student base is specified. The current CLI collector is a four-row synthetic smoke path and cannot be used for paper-grade collection.
- No training/evaluator process was started; all eight GPUs remained idle and no stuck process required cleanup. No event-active state, 5K training file, Teacher metric, Student Before metric, or Student After metric is claimed. Required next gate is to restore the approved EasyOPD/verl runtime, provide `CANONICAL_STUDENT_BASE`, and supply a real query pool with at least 1,000 train queries before collecting independent on-policy rollouts.


## 2026-08-19 H100-2 Qwen3 retrieval/runtime component sweep

Status: **H1002_ADAPTIVE_RERANK_FORMAL_TRAINING_COMPLETE_METRICS_PENDING**.

Setting:

```text
machine role: H100-2
components: content_dedup, chunk_neighbors, adaptive_rerank_instruction
canonical_student_base: /mnt/songzijun/models/Qwen3-30B-A3B-Instruct-2507
logical_model_id: Qwen3-30B-A3B-Instruct-2507
runtime env: /opt/scape-easyopd-smoke7 via SCAPE-EasyOPD/scripts/setup_scape_easyopd_smoke7_env.sh
collector: real Harness-1 bridge, collector_mode=real_harness1
student_inference_privilege: false
query pool: 2000 train-side queries
selection_seed: 20260818
synthetic_fallback: false
```

Phase U / gate results:

```text
adaptive_rerank_instruction: READY_5K
  n_queries_selected=2000
  n_rollouts_total=8000
  n_unique_event_active=8000
  TRAIN_STATES_5K rows=5000
  synthetic_row_count=0

content_dedup: INSUFFICIENT_5K_EVENT_SUPPORT
  n_unique_event_active=0
  synthetic_row_count=0

chunk_neighbors: NON_REALIZABLE_EXTERNAL_INFORMATION
  no student-visible Harness-1 neighbor injection hook located
  n_unique_event_active=0
  synthetic_row_count=0

2026-08-21 V8D_CHUNK_NEIGHBORS Teacher-always-on vs Student-always-off 128-state gain
  runner: scripts/run_chunk_neighbors_always_on_off_128.py
  source cohort: outputs/0820_adaptive_rerank_instruction_128_cohorts, seeds 2214/2215/2216/2217, 32 states/seed
  artifact: outputs/0821_chunk_neighbors_always_on_off_128_retry/CHUNK_NEIGHBORS_ALWAYS_ON_OFF_SUMMARY.json
  protocol: Teacher chunk_neighbors ON at every decision; Student OFF at every decision; first action included in K; exactly K actions; no Full Harness takeover
  K4: first-action disagreement=11.71875%; tool-cost delta=+0.0546875; utility delta=-0.0008203125 (-0.0820%)
  K8: first-action disagreement=11.71875%; tool-cost delta=+0.0390625; utility delta=-0.0005859375 (-0.0586%)
  audit: both horizons 128/128 rows; ordered K4/K8 identity and reconstructed snapshot hashes 128/128; mask/action-count/takeover failures=0
  interpretation: measured SCAPE mask-level branch difference only; no student-visible upstream Harness-1 neighbor-injection hook was located, so this is not proof of real external chunk-neighbor runtime injection.
```

Training implementation notes:

```text
- Previous verl/vLLM formal retries failed because of Hydra reward struct mismatch, OOM, and vLLM LoRA unsupported for Qwen3MoeForCausalLM.
- H100-2 switched to the same logits-sliced HF LoRA path validated on H100-1/H100-4.
- Added scripts/export_h1002_adaptive_opd_rows.py to convert adaptive_rerank_instruction DIRECT same-state rows into OPD_TRAIN_ROWS.jsonl / OPD_VALID_ROWS.jsonl.
- Added --loss-path to scripts/train_h1001_projectable_cell.py; adaptive_rerank uses full_response_kl.
```

Formal adaptive_rerank_instruction cells:

```text
PURE_OPD seed42: completed, train_steps=4500, adapter_reload_pass=true
  pre_div=0.665565, post_div=-0.083540
PURE_OPD seed43: completed, train_steps=4500, adapter_reload_pass=true
  pre_div=0.665565, post_div=-0.087396
RL_PLUS_OPD seed42: completed, train_steps=9000, adapter_reload_pass=true
  pre_div=0.665565, post_div=-0.004367
RL_PLUS_OPD seed43: completed, train_steps=9000, adapter_reload_pass=true
  pre_div=0.665565, post_div=-0.080368
```

Conclusion so far:

```text
H1002 adaptive_rerank_instruction training/reload stage is complete.
Final scientific PASS/FAIL is not yet assigned because Teacher, Student Before, Student After real closed-loop DEV/TEST reward metrics, paired bootstrap, and mechanism audit remain pending.
```

Artifacts:

```text
SCAPE-EasyOPD/outputs/component_sweep_0818/h100_2/H1002_COMPONENT_HANDOFF.json
SCAPE-EasyOPD/outputs/component_sweep_0818/h100_2/H1002_FORMAL_ADAPTIVE_SUMMARY.json
SCAPE-EasyOPD/outputs/component_sweep_0818/h100_2/H1002_COMPONENT_ROWS.{json,csv}
SCAPE-EasyOPD/outputs/component_sweep_0818/h100_2/SHA256SUMS
SCAPE-EasyOPD/outputs/component_sweep_0818/h100_2/components/adaptive_rerank_instruction/H1002_ADAPTIVE_OPD_ROWS_MANIFEST.json
SCAPE-EasyOPD/outputs/component_sweep_0818/h100_2/formal_hf_adaptive_8gpu/*/summary.json
SCAPE-EasyOPD/outputs/component_sweep_0818/h100_2/formal_hf_adaptive_8gpu/*/ADAPTER_RELOAD_ACCEPTANCE.json
```

## 2026-08-21 subtractive/joint tool-cost delta backfill

检查了 `subtractive_curation` 和 `importance_tagging+subtractive_curation` 的现有 128-state 原始 paired artifacts，确认无需重新调用模型：两者均保存了 `branch_T_metrics.tool_search_cost` 与 `branch_S_metrics.tool_search_cost`。此前 `explore_gain_reference_metrics_128.py` 错把 row-level 已经是 Teacher-Student 的 `tool_search_cost` 当作两个 branch 值，导致统一摘要中的成本 Δ 为 0/N/A。

已修复 extractor：成本指标现在显式从 branch-level 字段计算 `mean(T-S)`，并保留按 K 分组的来源、样本数及 Teacher/Student 均值。重新生成产物：`SCAPE/outputs/0821_gain_reference_metrics_128/GAIN_REFERENCE_METRICS_SUMMARY.json` 与 `GAIN_REFERENCE_METRICS_PER_STATE.jsonl`。

- `subtractive_curation` 正式完整 artifact `outputs/0820_subtractive_curation_recall_128_final/SUBTRACTIVE_CANDIDATE_ACTIVATED_RECALL_PER_STATE.jsonl`：K4 `T=4.5000`、`S=4.484375`、工具成本 Δ=`+0.015625`；K8 `T=6.8203125`、`S=6.8046875`、Δ=`+0.015625`，各 128 rows。
- `importance_tagging+subtractive_curation` 使用完整 512-row pilot artifact `outputs/0820_joint_importance_subtractive_preopd_fork_pilot128_retry/JOINT_PREOPD_VALUE_PER_STATE.jsonl`（K4/K8 各 256 rows）：K4 `T=5.0078125`、`S=4.76953125`、Δ=`+0.23828125`；K8 `T=6.67578125`、`S=6.3046875`、Δ=`+0.37109375`。
- later `0820_joint_importance_subtractive_recall_128_final/JOINT_RECALL_PER_STATE.jsonl` 仅有 166 rows 且 provenance/状态不完整，因此未将其用于联合组件成本汇总；表格中的联合成本值明确标注为完整 pilot artifact 的同一组件/协议成本指标。

`增益.md` 已将两组件 K4/K8 的工具成本 Δ 从 `N/A*` 更新为上述值，并移除过时的“字段不足”注释。

## 2026-08-20 adaptive_rerank_instruction 128-state evidence recall fork

Setting:

```text
component: adaptive_rerank_instruction
cohort: existing frozen four-seed cohort, seeds 2214/2215/2216/2217, 32 states per seed
horizons: K4 and K8; forced first action counted inside K
paired rows: 128 per horizon, 256 total
Teacher: component ON for first action only
Student: component OFF for first action only
continuation: Reduced policy for both branches; Full Harness takeover=0
runtime: /opt/scape-h1004 (torch 2.10.0+cu128, transformers 5.14.1)
normalization: split_at_first_underscore
qrel_sha256: a6f594975be57339de9e4e9f67f13c044f647feda77c0b84c45a1581e3041bd1
```

Recall results (pooled paired state mean, seed-balanced because every seed contributes 32 rows):

```text
K4 candidate_evidence_pool_recall@K: Teacher 7.9861%, Student 7.9861%, delta +0.00 pp, paired bootstrap CI95 [0.00, 0.00] pp, query-cluster CI95 [0.00, 0.00] pp
K4 activated_evidence_recall@K:    Teacher 3.8194%, Student 3.8194%, delta +0.00 pp, paired bootstrap CI95 [0.00, 0.00] pp, query-cluster CI95 [0.00, 0.00] pp
K8 candidate_evidence_pool_recall@K: Teacher 7.9861%, Student 7.9861%, delta +0.00 pp, paired bootstrap CI95 [0.00, 0.00] pp, query-cluster CI95 [0.00, 0.00] pp
K8 activated_evidence_recall@K:    Teacher 3.8194%, Student 3.8194%, delta +0.00 pp, paired bootstrap CI95 [0.00, 0.00] pp, query-cluster CI95 [0.00, 0.00] pp
```

All four metric/horizon combinations have positive/negative/zero counts `0/0/128`; Teacher and Student candidate pool mean size are `10.0/10.0`, activated set mean size `2.0/2.0`, candidate precision `2.50%/2.50%`, and activated precision `6.25%/6.25%`. All seeds have zero paired delta and seed sample std `0.00 pp`. Audit: 256/256 valid rows, empty qrel=0, snapshot mismatch=0, invalid provenance=0, endpoint identity failures=0, Full Harness takeover=0.

Conclusion: adaptive_rerank_instruction produced no measurable candidate-pool or activated-evidence recall gain under this same-state paired fork. This is a recall-layer result and does not replace or imply a weighted-utility conclusion.

Artifacts: `SCAPE/outputs/0820_adaptive_rerank_instruction_recall_128/ADAPTIVE_RERANK_RECALL_K4_K8.json`, `ADAPTIVE_RERANK_RECALL_PER_STATE.jsonl`, `scripts/run_adaptive_rerank_recall_128.py`, `scripts/score_adaptive_rerank_recall_128.py`.

## 2026-08-21 importance_tagging Teacher-always-on vs Student-always-off utility fork

Status: **COMPLETED; NO STABLE CROSS-HORIZON UTILITY GAIN**.

Setting:

```text
component: importance_tagging only
frozen cohort: seeds 8423/8424, 128 states per seed; 256 paired rows per K
Teacher: Full view and importance_tagging enabled at the forced first action and every continuation action
Student: Reduced view and importance_tagging disabled at the forced first action and every continuation action
horizons: K4/K8; forced first action plus K continuation actions, matching the existing live-fork convention
model: /mnt/songzijun/models/pat-jj_harness-1-full/harness-1
python_env: /opt/scape-projected-action
runner: scripts/run_importance_tagging_always_on_off.py
output: SCAPE/outputs/0821_importance_tagging_always_on_off_256/
```

Requested metrics are Teacher - Student:

```text
K4 n=256: first-action disagreement=100.00%; tool-cost delta=+0.1171875; Utility delta=-0.0017578125 (-0.1758%); positive/negative/zero utility=113/122/21
K8 n=256: first-action disagreement=100.00%; tool-cost delta=-0.0546875; Utility delta=+0.0008203125 (+0.0820%); positive/negative/zero utility=117/115/24
```

Audit: all four seed/K cells completed with 128/128 rows; every seed had K4/K8 ordered query/snapshot matches `128/128`, rebuilt snapshots matched source hashes `128/128`, and source first actions matched `128/128`. Teacher policy views were Full for every recorded step, Student policy views Reduced for every recorded step, branch initial hashes matched, and Full Harness takeover was `0`.

Conclusion: always-on importance_tagging changes the first action on every paired state, but the utility and cost directions differ between K4 and K8. This is horizon-dependent process separation, not a stable positive gain claim.

Artifacts: `SCAPE/outputs/0821_importance_tagging_always_on_off_256/IMPORTANCE_ALWAYS_ON_OFF_SUMMARY.json`, `IMPORTANCE_ALWAYS_ON_OFF_PER_STATE.jsonl`, `IMPORTANCE_ALWAYS_ON_OFF_SUMMARY.md`, `audits/RECONSTRUCTION_K{4,8}_SEED{8423,8424}.json`.

## 2026-08-21 token_budget_marker OPD 4-cell on strict 384-query evaluation pool

Setting:

```text
component: token_budget_marker
pool: 384 official BrowseComp-Plus queries present in both qrels, after strict query-ID exclusion of component training pool
pool_status: FROZEN_VALID; training_overlap_query_ids=0
Teacher: Qwen3 canonical base, token-budget privileged context
Student before OPD: same canonical base, reduced/no-privilege context
Student after PURE_OPD: actual LoRA adapter, seed42 artifact
Student after RL+OPD: actual LoRA adapter, seed42 artifact
model: /mnt/songzijun/models/Qwen3-30B-A3B-Instruct-2507
retrieval: official BrowseComp-Plus BM25, ordered top-1000 docids
qrel: qrel_evidence.txt, sha256=a6f594975be57339de9e4e9e67f13c044f647feda77c0b84c45a1581e3041bd1
normalization: split_at_first_underscore_v1
parser: compact_tool_json_v2; adapter reload passed via PEFT native path
```

The first generation pass was rescored offline after correcting the Qwen3 compact action schema (`{"tool": ..., ...}`); no model generations were changed. Each setting has 384/384 unique query rows, with ordered `retrieved_docids` retained per query. Legal action means a recognized Harness-1 tool name; recall is mean evidence qrel recall over the BM25 top-k result list.

```text
                                      Legal action rate    Evidence Recall@5    Evidence Recall@100    Evidence Recall@1000
Teacher                                  73.9583%              1.3519%              4.0451%                11.2443%
Student before OPD                       60.9375%              0.6550%              2.9352%                 8.6441%
Student after pure OPD                    83.8542%              1.3403%              4.0770%                12.1403%
Student after RL+OPD                      82.0313%              1.0929%              4.1200%                12.0368%
```

Relative to Student before OPD, Legal action rate changes are `Teacher +13.0208 pp`, `PURE_OPD +22.9167 pp`, and `RL+OPD +21.0938 pp`. Evidence Recall@100 changes are `Teacher +1.10997 pp`, `PURE_OPD +1.14180 pp`, and `RL+OPD +1.18479 pp`; Recall@1000 changes are `Teacher +2.60025 pp`, `PURE_OPD +3.49621 pp`, and `RL+OPD +3.39276 pp`. These are absolute paired-condition means, not terminal answer reward or K4/K8 closed-loop recall.

Artifacts: `SCAPE-EasyOPD/outputs/0821_token_budget_marker_opd_384/SUMMARY.json`, `384_QUERY_MANIFEST.json`, `{TEACHER,STUDENT_BEFORE_OPD,STUDENT_AFTER_PURE_OPD,STUDENT_AFTER_RL_PLUS_OPD}/PER_QUERY.jsonl`, `scripts/eval_token_budget_marker_opd_384.py`, `scripts/rescore_token_budget_marker_opd_384.py`.
