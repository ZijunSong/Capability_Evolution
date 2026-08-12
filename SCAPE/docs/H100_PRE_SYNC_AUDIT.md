# H100 PRE SYNC AUDIT

## git status -sb
```
## main...origin/main [ahead 3]
 D GIT_SYNC_H100_IN_PROGRESS
 M docs/H100_PRE_SYNC_AUDIT.md
 M ../SCOPE/harness/agent.py
 M ../SCOPE/harness/config.py
 M ../SCOPE/harness/llm_env.py
 M ../SCOPE/harness/rerank.py
 M ../SCOPE/harness/retrieval/bm25_backend.py
 M ../SCOPE/harness/retrieval/bm25_tools.py
 M ../SCOPE/harness/tools.py
 M ../SCOPE/inference/evaluate_harness_api.py
 M ../SCOPE/result-record.md
 M ../SCOPE/scripts/rollout_bare_browsecomp_4gpu.sh
 M ../SCOPE/scripts/rollout_harness_browsecomp_4gpu.sh
 M ../SCOPE/scripts/setup_browsecomp_bm25_index.sh
 M ../SCOPE/scripts/setup_browsecomp_data.sh
 M ../SCOPE/training/chat_decision_driver.py
 M ../SCOPE/training/opd/__init__.py
 M ../SCOPE/training/opd/bare_rollout.py
 M ../SCOPE/training/opd/browsecomp_queries.py
 M ../SCOPE/training/opd/env_factory.py
 M ../SCOPE/training/opd/llm_factory.py
 M ../SCOPE/training/opd/transition_builder.py
 M ../SCOPE/training/opd/vllm_rollout_backend.py
 M ../SCOPE/training/opd/vllm_server.py
 M ../SCOPE/training/rollout_bare_browsecomp.py
 M ../SCOPE/training/rollout_harness_browsecomp.py
 M ../SCOPE/training/train_rl.py
?? ../SCAPE-wt-h100-1/
?? ../SCAPE-wt-h100-2/
?? ../SCAPE-wt-h100-3/
?? GIT_SYNC_H100_READY
?? scripts/build_h100_4_prestage.py
?? scripts/run_h100_1_confirm_local_bm25.py
?? scripts/run_h100_3_real_influence_hf.py
?? ../SCOPE/.fresh200_preview.json
?? ../SCOPE/bare_rollout_manifest.json
?? ../SCOPE/bare_rollouts.jsonl
?? ../SCOPE/external/hotpotqa_subset_queries.json
?? ../SCOPE/harness/configs/h100_2_ablate_retrieval_rerank.yaml
?? ../SCOPE/harness_resolved_config.yaml
?? ../SCOPE/scripts/direct_answer_hotpotqa.py
?? ../SCOPE/scripts/export_hotpotqa_subset_queries.py
?? ../SCOPE/scripts/forced_readout_hotpotqa.py
?? ../SCOPE/scripts/h100_1_browsecomp_metrics.py
?? ../SCOPE/scripts/h100_1_fresh_selection_finalize.py
?? ../SCOPE/scripts/h100_1_fresh_selection_replication_run.sh
?? ../SCOPE/scripts/h100_1_prepare_browsecomp_deterministic.py
?? ../SCOPE/scripts/h100_1_retrieval_synthesis_factorial_r1_toolfix.sh
?? ../SCOPE/scripts/h100_2_finalization/
?? ../SCOPE/scripts/h100_3_controller_finalization_factorial.py
?? ../SCOPE/scripts/h100_3_hotpotqa_evidence_compaction_readout.py
?? ../SCOPE/scripts/h100_3_hotpotqa_late_loop_factorization.py
?? ../SCOPE/scripts/h100_3_hotpotqa_readout_contract.py
?? ../SCOPE/scripts/h100_3_hotpotqa_readout_contract_audit.py
?? ../SCOPE/scripts/h100_3_hotpotqa_turn_cut_curve.py
?? ../SCOPE/scripts/nohup_hotpotqa_evidence_compaction_1p7b.sh
?? ../SCOPE/scripts/nohup_hotpotqa_evidence_compaction_30b.sh
?? ../SCOPE/scripts/nohup_hotpotqa_readout_contract_audit_1p7b.sh
?? ../SCOPE/scripts/nohup_hotpotqa_readout_contract_audit_30b.sh
?? ../SCOPE/scripts/nohup_hotpotqa_turn_cut_curve_1p7b.sh
?? ../SCOPE/scripts/nohup_hotpotqa_turn_cut_curve_30b.sh
?? ../SCOPE/scripts/nohup_rollout_bare_hotpotqa.sh
?? ../SCOPE/scripts/nohup_rollout_bare_hotpotqa_4gpu_qwen3_1p7b.sh
?? ../SCOPE/scripts/nohup_rollout_harness_hotpotqa_4gpu_qwen3_1p7b.sh
?? ../SCOPE/scripts/nohup_rollout_harness_hotpotqa_8gpu_qwen3_1p7b.sh
?? ../SCOPE/scripts/rollout_bare_browsecomp_8gpu_qwen3_30b.sh
?? ../SCOPE/scripts/rollout_bare_hotpotqa_4gpu_qwen3_1p7b.sh
?? ../SCOPE/scripts/rollout_bare_hotpotqa_8gpu_qwen3_30b.sh
?? ../SCOPE/scripts/rollout_harness_hotpotqa_4gpu_qwen3_1p7b.sh
?? ../SCOPE/scripts/rollout_harness_hotpotqa_8gpu_qwen3_30b.sh
?? ../SCOPE/scripts/rollout_hotpotqa_full_harness_4gpu_qwen3_1p7b.sh
?? ../SCOPE/scripts/rollout_hotpotqa_full_harness_8gpu_qwen3_30b.sh
?? ../SCOPE/scripts/run_h100_3_controller_finalization_30b_retry.sh
?? ../SCOPE/scripts/run_h100_3_controller_finalization_forced.sh
?? ../SCOPE/scripts/run_hotpotqa_decomposition_matrix.sh
?? ../SCOPE/scripts/run_hotpotqa_decomposition_smoke.sh
?? ../SCOPE/scripts/summarize_h100_3_hotpotqa_per_condition.py
?? ../SCOPE/scripts/summarize_h100_3_hotpotqa_turn_cut_curve.py
?? ../SCOPE/training/opd/query_records.py
?? ../SCOPE/training/rollout_harness_hotpotqa.py
```

## git diff --stat
```
 SCAPE/GIT_SYNC_H100_IN_PROGRESS                  |    1 -
 SCAPE/docs/H100_PRE_SYNC_AUDIT.md                |  677 +-------
 SCOPE/harness/agent.py                           |  111 +-
 SCOPE/harness/config.py                          |   23 +-
 SCOPE/harness/llm_env.py                         |    4 +
 SCOPE/harness/rerank.py                          |    7 +-
 SCOPE/harness/retrieval/bm25_backend.py          |   21 +-
 SCOPE/harness/retrieval/bm25_tools.py            |   21 +-
 SCOPE/harness/tools.py                           |   66 +-
 SCOPE/inference/evaluate_harness_api.py          |   77 +-
 SCOPE/result-record.md                           | 1868 ++++------------------
 SCOPE/scripts/rollout_bare_browsecomp_4gpu.sh    |   22 +-
 SCOPE/scripts/rollout_harness_browsecomp_4gpu.sh |   70 +-
 SCOPE/scripts/setup_browsecomp_bm25_index.sh     |    8 +-
 SCOPE/scripts/setup_browsecomp_data.sh           |   15 +-
 SCOPE/training/chat_decision_driver.py           |    9 +-
 SCOPE/training/opd/__init__.py                   |    5 -
 SCOPE/training/opd/bare_rollout.py               |  105 +-
 SCOPE/training/opd/browsecomp_queries.py         |    2 +-
 SCOPE/training/opd/env_factory.py                |   23 +-
 SCOPE/training/opd/llm_factory.py                |    2 +
 SCOPE/training/opd/transition_builder.py         |    2 +-
 SCOPE/training/opd/vllm_rollout_backend.py       |    9 +
 SCOPE/training/opd/vllm_server.py                |   10 +-
 SCOPE/training/rollout_bare_browsecomp.py        |   83 +-
 SCOPE/training/rollout_harness_browsecomp.py     |    6 +-
 SCOPE/training/train_rl.py                       |   23 +-
 27 files changed, 988 insertions(+), 2282 deletions(-)
```

## git remote -v
```
origin	https://github.com/ZijunSong/Capability_Evolution.git (fetch)
origin	https://github.com/ZijunSong/Capability_Evolution.git (push)
```

## git rev-parse HEAD
```
0f0934bd9f7a985af747e18dda9c2c666a9c24ba
```

## full git diff
```diff
diff --git a/SCAPE/GIT_SYNC_H100_IN_PROGRESS b/SCAPE/GIT_SYNC_H100_IN_PROGRESS
deleted file mode 100644
index 87ebde8..0000000
--- a/SCAPE/GIT_SYNC_H100_IN_PROGRESS
+++ /dev/null
@@ -1 +0,0 @@
-H100 sync started 2026-08-12 by coordinator task. Do not checkout/reset/merge/add/commit shared root until GIT_SYNC_H100_READY exists.
diff --git a/SCAPE/docs/H100_PRE_SYNC_AUDIT.md b/SCAPE/docs/H100_PRE_SYNC_AUDIT.md
index 5e421c3..c4acea4 100644
--- a/SCAPE/docs/H100_PRE_SYNC_AUDIT.md
+++ b/SCAPE/docs/H100_PRE_SYNC_AUDIT.md
@@ -1,594 +1,132 @@
-# H100_PRE_SYNC_AUDIT
+# H100 PRE SYNC AUDIT
 
 ## git status -sb
-## main...origin/main [ahead 2]
- D SCAPE/SCAPE_H100_1_CONTRIBUTION.md
- D SCAPE/SCAPE_H100_2_REPLICATION_COALITION.md
- D SCAPE/SCAPE_H100_3_INFLUENCE.md
- M SCAPE/result-record.md
- M SCAPE/scape/probes/candidate_selector.py
- M SCAPE/scripts/local_cal64_bootstrap.py
- M SCAPE/scripts/preflight_harness1.py
- M SCOPE/harness/agent.py
- M SCOPE/harness/config.py
- M SCOPE/harness/llm_env.py
- M SCOPE/harness/rerank.py
- M SCOPE/harness/retrieval/bm25_backend.py
- M SCOPE/harness/retrieval/bm25_tools.py
- M SCOPE/harness/tools.py
- M SCOPE/inference/evaluate_harness_api.py
- M SCOPE/result-record.md
- M SCOPE/scripts/rollout_bare_browsecomp_4gpu.sh
- M SCOPE/scripts/rollout_harness_browsecomp_4gpu.sh
- M SCOPE/scripts/setup_browsecomp_bm25_index.sh
- M SCOPE/scripts/setup_browsecomp_data.sh
- M SCOPE/training/chat_decision_driver.py
- M SCOPE/training/opd/__init__.py
- M SCOPE/training/opd/bare_rollout.py
- M SCOPE/training/opd/browsecomp_queries.py
- M SCOPE/training/opd/env_factory.py
- M SCOPE/training/opd/llm_factory.py
- M SCOPE/training/opd/transition_builder.py
- M SCOPE/training/opd/vllm_rollout_backend.py
- M SCOPE/training/opd/vllm_server.py
- M SCOPE/training/rollout_bare_browsecomp.py
- M SCOPE/training/rollout_harness_browsecomp.py
- M SCOPE/training/train_rl.py
-?? SCAPE/0812/
-?? SCAPE/GIT_SYNC_H100_IN_PROGRESS
-?? SCAPE/H100-1-0811-todo1.md
-?? SCAPE/H100-2-0811-todo1.md
-?? SCAPE/H100-3-0811-todo1.md
-?? SCAPE/configs/harness/full.yaml
-?? SCAPE/configs/harness/minus_adaptive_rerank_instruction.yaml
-?? SCAPE/configs/harness/minus_auto_populate_first_search.yaml
-?? SCAPE/configs/harness/minus_chunk_neighbors.yaml
-?? SCAPE/configs/harness/minus_content_dedup.yaml
-?? SCAPE/configs/harness/minus_evidence_graph.yaml
-?? SCAPE/configs/harness/minus_importance_tagging.yaml
-?? SCAPE/configs/harness/minus_sentence_compress.yaml
-?? SCAPE/configs/harness/minus_subtractive_curation.yaml
-?? SCAPE/configs/harness/minus_token_budget_marker.yaml
-?? SCAPE/configs/harness/minus_verify_tool.yaml
-?? SCAPE/docs/BLOCKED_RETRIEVAL_BACKEND.md
-?? SCAPE/docs/H100_PRE_SYNC_AUDIT.md
-?? SCAPE/docs/OFFICIAL_HARNESS1_STATUS.md
-?? SCAPE/scripts/build_browsecomp_local_chroma.py
-?? SCAPE/scripts/build_browsecomp_local_corpus.py
-?? SCAPE/scripts/build_h100_2_replication_coalition.py
-?? SCAPE/scripts/check_official_env_presence.py
-?? SCAPE/scripts/run_bm25_mify_grid.py
-?? SCAPE/scripts/run_h100_1_local_bm25_contribution.py
-?? SCAPE/scripts/run_h100_2_independent_repl.py
-?? SCAPE/scripts/run_h100_3_influence.py
-?? SCAPE/scripts/run_h100_3_influence_qrel.py
-?? SCAPE/scripts/run_h20_lightweight_experiments.py
-?? SCAPE/scripts/run_official_harness1_browsecompplus_vllm.sh
-?? SCAPE/scripts/serve_harness1_vllm_local.sh
-?? SCAPE/scripts/summarize_h100_3_influence.py
-?? SCOPE/.fresh200_preview.json
-?? SCOPE/bare_rollout_manifest.json
-?? SCOPE/bare_rollouts.jsonl
-?? SCOPE/external/hotpotqa_subset_queries.json
-?? SCOPE/harness/configs/h100_2_ablate_retrieval_rerank.yaml
-?? SCOPE/harness_resolved_config.yaml
-?? SCOPE/scripts/direct_answer_hotpotqa.py
-?? SCOPE/scripts/export_hotpotqa_subset_queries.py
-?? SCOPE/scripts/forced_readout_hotpotqa.py
-?? SCOPE/scripts/h100_1_browsecomp_metrics.py
-?? SCOPE/scripts/h100_1_fresh_selection_finalize.py
-?? SCOPE/scripts/h100_1_fresh_selection_replication_run.sh
-?? SCOPE/scripts/h100_1_prepare_browsecomp_deterministic.py
-?? SCOPE/scripts/h100_1_retrieval_synthesis_factorial_r1_toolfix.sh
-?? SCOPE/scripts/h100_2_finalization/
-?? SCOPE/scripts/h100_3_controller_finalization_factorial.py
-?? SCOPE/scripts/h100_3_hotpotqa_evidence_compaction_readout.py
-?? SCOPE/scripts/h100_3_hotpotqa_late_loop_factorization.py
-?? SCOPE/scripts/h100_3_hotpotqa_readout_contract.py
-?? SCOPE/scripts/h100_3_hotpotqa_readout_contract_audit.py
-?? SCOPE/scripts/h100_3_hotpotqa_turn_cut_curve.py
-?? SCOPE/scripts/nohup_hotpotqa_evidence_compaction_1p7b.sh
-?? SCOPE/scripts/nohup_hotpotqa_evidence_compaction_30b.sh
-?? SCOPE/scripts/nohup_hotpotqa_readout_contract_audit_1p7b.sh
-?? SCOPE/scripts/nohup_hotpotqa_readout_contract_audit_30b.sh
-?? SCOPE/scripts/nohup_hotpotqa_turn_cut_curve_1p7b.sh
-?? SCOPE/scripts/nohup_hotpotqa_turn_cut_curve_30b.sh
-?? SCOPE/scripts/nohup_rollout_bare_hotpotqa.sh
-?? SCOPE/scripts/nohup_rollout_bare_hotpotqa_4gpu_qwen3_1p7b.sh
-?? SCOPE/scripts/nohup_rollout_harness_hotpotqa_4gpu_qwen3_1p7b.sh
-?? SCOPE/scripts/nohup_rollout_harness_hotpotqa_8gpu_qwen3_1p7b.sh
-?? SCOPE/scripts/rollout_bare_browsecomp_8gpu_qwen3_30b.sh
-?? SCOPE/scripts/rollout_bare_hotpotqa_4gpu_qwen3_1p7b.sh
-?? SCOPE/scripts/rollout_bare_hotpotqa_8gpu_qwen3_30b.sh
-?? SCOPE/scripts/rollout_harness_hotpotqa_4gpu_qwen3_1p7b.sh
-?? SCOPE/scripts/rollout_harness_hotpotqa_8gpu_qwen3_30b.sh
-?? SCOPE/scripts/rollout_hotpotqa_full_harness_4gpu_qwen3_1p7b.sh
-?? SCOPE/scripts/rollout_hotpotqa_full_harness_8gpu_qwen3_30b.sh
-?? SCOPE/scripts/run_h100_3_controller_finalization_30b_retry.sh
-?? SCOPE/scripts/run_h100_3_controller_finalization_forced.sh
-?? SCOPE/scripts/run_hotpotqa_decomposition_matrix.sh
-?? SCOPE/scripts/run_hotpotqa_decomposition_smoke.sh
-?? SCOPE/scripts/summarize_h100_3_hotpotqa_per_condition.py
-?? SCOPE/scripts/summarize_h100_3_hotpotqa_turn_cut_curve.py
-?? SCOPE/training/opd/query_records.py
-?? SCOPE/training/rollout_harness_hotpotqa.py
+```
+## main...origin/main [ahead 3]
+ D GIT_SYNC_H100_IN_PROGRESS
+ M docs/H100_PRE_SYNC_AUDIT.md
+ M ../SCOPE/harness/agent.py
+ M ../SCOPE/harness/config.py
+ M ../SCOPE/harness/llm_env.py
+ M ../SCOPE/harness/rerank.py
+ M ../SCOPE/harness/retrieval/bm25_backend.py
+ M ../SCOPE/harness/retrieval/bm25_tools.py
+ M ../SCOPE/harness/tools.py
+ M ../SCOPE/inference/evaluate_harness_api.py
+ M ../SCOPE/result-record.md
+ M ../SCOPE/scripts/rollout_bare_browsecomp_4gpu.sh
+ M ../SCOPE/scripts/rollout_harness_browsecomp_4gpu.sh
+ M ../SCOPE/scripts/setup_browsecomp_bm25_index.sh
+ M ../SCOPE/scripts/setup_browsecomp_data.sh
+ M ../SCOPE/training/chat_decision_driver.py
+ M ../SCOPE/training/opd/__init__.py
+ M ../SCOPE/training/opd/bare_rollout.py
+ M ../SCOPE/training/opd/browsecomp_queries.py
+ M ../SCOPE/training/opd/env_factory.py
+ M ../SCOPE/training/opd/llm_factory.py
+ M ../SCOPE/training/opd/transition_builder.py
+ M ../SCOPE/training/opd/vllm_rollout_backend.py
+ M ../SCOPE/training/opd/vllm_server.py
+ M ../SCOPE/training/rollout_bare_browsecomp.py
+ M ../SCOPE/training/rollout_harness_browsecomp.py
+ M ../SCOPE/training/train_rl.py
+?? ../SCAPE-wt-h100-1/
+?? ../SCAPE-wt-h100-2/
+?? ../SCAPE-wt-h100-3/
+?? GIT_SYNC_H100_READY
+?? scripts/build_h100_4_prestage.py
+?? scripts/run_h100_1_confirm_local_bm25.py
+?? scripts/run_h100_3_real_influence_hf.py
+?? ../SCOPE/.fresh200_preview.json
+?? ../SCOPE/bare_rollout_manifest.json
+?? ../SCOPE/bare_rollouts.jsonl
+?? ../SCOPE/external/hotpotqa_subset_queries.json
+?? ../SCOPE/harness/configs/h100_2_ablate_retrieval_rerank.yaml
+?? ../SCOPE/harness_resolved_config.yaml
+?? ../SCOPE/scripts/direct_answer_hotpotqa.py
+?? ../SCOPE/scripts/export_hotpotqa_subset_queries.py
+?? ../SCOPE/scripts/forced_readout_hotpotqa.py
+?? ../SCOPE/scripts/h100_1_browsecomp_metrics.py
+?? ../SCOPE/scripts/h100_1_fresh_selection_finalize.py
+?? ../SCOPE/scripts/h100_1_fresh_selection_replication_run.sh
+?? ../SCOPE/scripts/h100_1_prepare_browsecomp_deterministic.py
+?? ../SCOPE/scripts/h100_1_retrieval_synthesis_factorial_r1_toolfix.sh
+?? ../SCOPE/scripts/h100_2_finalization/
+?? ../SCOPE/scripts/h100_3_controller_finalization_factorial.py
+?? ../SCOPE/scripts/h100_3_hotpotqa_evidence_compaction_readout.py
+?? ../SCOPE/scripts/h100_3_hotpotqa_late_loop_factorization.py
+?? ../SCOPE/scripts/h100_3_hotpotqa_readout_contract.py
+?? ../SCOPE/scripts/h100_3_hotpotqa_readout_contract_audit.py
+?? ../SCOPE/scripts/h100_3_hotpotqa_turn_cut_curve.py
+?? ../SCOPE/scripts/nohup_hotpotqa_evidence_compaction_1p7b.sh
+?? ../SCOPE/scripts/nohup_hotpotqa_evidence_compaction_30b.sh
+?? ../SCOPE/scripts/nohup_hotpotqa_readout_contract_audit_1p7b.sh
+?? ../SCOPE/scripts/nohup_hotpotqa_readout_contract_audit_30b.sh
+?? ../SCOPE/scripts/nohup_hotpotqa_turn_cut_curve_1p7b.sh
+?? ../SCOPE/scripts/nohup_hotpotqa_turn_cut_curve_30b.sh
+?? ../SCOPE/scripts/nohup_rollout_bare_hotpotqa.sh
+?? ../SCOPE/scripts/nohup_rollout_bare_hotpotqa_4gpu_qwen3_1p7b.sh
+?? ../SCOPE/scripts/nohup_rollout_harness_hotpotqa_4gpu_qwen3_1p7b.sh
+?? ../SCOPE/scripts/nohup_rollout_harness_hotpotqa_8gpu_qwen3_1p7b.sh
+?? ../SCOPE/scripts/rollout_bare_browsecomp_8gpu_qwen3_30b.sh
+?? ../SCOPE/scripts/rollout_bare_hotpotqa_4gpu_qwen3_1p7b.sh
+?? ../SCOPE/scripts/rollout_bare_hotpotqa_8gpu_qwen3_30b.sh
+?? ../SCOPE/scripts/rollout_harness_hotpotqa_4gpu_qwen3_1p7b.sh
+?? ../SCOPE/scripts/rollout_harness_hotpotqa_8gpu_qwen3_30b.sh
+?? ../SCOPE/scripts/rollout_hotpotqa_full_harness_4gpu_qwen3_1p7b.sh
+?? ../SCOPE/scripts/rollout_hotpotqa_full_harness_8gpu_qwen3_30b.sh
+?? ../SCOPE/scripts/run_h100_3_controller_finalization_30b_retry.sh
+?? ../SCOPE/scripts/run_h100_3_controller_finalization_forced.sh
+?? ../SCOPE/scripts/run_hotpotqa_decomposition_matrix.sh
+?? ../SCOPE/scripts/run_hotpotqa_decomposition_smoke.sh
+?? ../SCOPE/scripts/summarize_h100_3_hotpotqa_per_condition.py
+?? ../SCOPE/scripts/summarize_h100_3_hotpotqa_turn_cut_curve.py
+?? ../SCOPE/training/opd/query_records.py
+?? ../SCOPE/training/rollout_harness_hotpotqa.py
+```
 
-## SCAPE code/config/docs diff stat
- SCAPE/result-record.md                   | 363 +++++++++++++++++++++++++++++++
- SCAPE/scape/probes/candidate_selector.py |   5 +-
- SCAPE/scripts/local_cal64_bootstrap.py   |   8 +-
- SCAPE/scripts/preflight_harness1.py      |  20 +-
- 4 files changed, 391 insertions(+), 5 deletions(-)
+## git diff --stat
+```
+ SCAPE/GIT_SYNC_H100_IN_PROGRESS                  |    1 -
+ SCAPE/docs/H100_PRE_SYNC_AUDIT.md                |  677 +-------
+ SCOPE/harness/agent.py                           |  111 +-
+ SCOPE/harness/config.py                          |   23 +-
+ SCOPE/harness/llm_env.py                         |    4 +
+ SCOPE/harness/rerank.py                          |    7 +-
+ SCOPE/harness/retrieval/bm25_backend.py          |   21 +-
+ SCOPE/harness/retrieval/bm25_tools.py            |   21 +-
+ SCOPE/harness/tools.py                           |   66 +-
+ SCOPE/inference/evaluate_harness_api.py          |   77 +-
+ SCOPE/result-record.md                           | 1868 ++++------------------
+ SCOPE/scripts/rollout_bare_browsecomp_4gpu.sh    |   22 +-
+ SCOPE/scripts/rollout_harness_browsecomp_4gpu.sh |   70 +-
+ SCOPE/scripts/setup_browsecomp_bm25_index.sh     |    8 +-
+ SCOPE/scripts/setup_browsecomp_data.sh           |   15 +-
+ SCOPE/training/chat_decision_driver.py           |    9 +-
+ SCOPE/training/opd/__init__.py                   |    5 -
+ SCOPE/training/opd/bare_rollout.py               |  105 +-
+ SCOPE/training/opd/browsecomp_queries.py         |    2 +-
+ SCOPE/training/opd/env_factory.py                |   23 +-
+ SCOPE/training/opd/llm_factory.py                |    2 +
+ SCOPE/training/opd/transition_builder.py         |    2 +-
+ SCOPE/training/opd/vllm_rollout_backend.py       |    9 +
+ SCOPE/training/opd/vllm_server.py                |   10 +-
+ SCOPE/training/rollout_bare_browsecomp.py        |   83 +-
+ SCOPE/training/rollout_harness_browsecomp.py     |    6 +-
+ SCOPE/training/train_rl.py                       |   23 +-
+ 27 files changed, 988 insertions(+), 2282 deletions(-)
+```
 
 ## git remote -v
+```
 origin	https://github.com/ZijunSong/Capability_Evolution.git (fetch)
 origin	https://github.com/ZijunSong/Capability_Evolution.git (push)
+```
 
 ## git rev-parse HEAD
-61f7741a6be2e2e62a4c8b0da86a651791a9117f
+```
+0f0934bd9f7a985af747e18dda9c2c666a9c24ba
+```
 
-## SCAPE code/config/docs diff
-diff --git a/SCAPE/result-record.md b/SCAPE/result-record.md
-index eee3a17..b4a0882 100644
---- a/SCAPE/result-record.md
-+++ b/SCAPE/result-record.md
-@@ -1,6 +1,52 @@
- # SCAPE result-record
- 
- > Canonical experiment log. Append one section per completed stage (see `SCAPE_H20_TRAINING_MIGRATION.md` §18).
-+> **状态以本节「本轮总览」为准**；更早条目中写 “still running / proxy” 的已被后续正式结果覆盖。
-+> H100 机实验 setting / 结果 / 结论已自 `result-record-from-h100.md` 同步（见下方「H100 同步」节）。
-+
-+---
-+
-+## 本轮总览（更新于 2026-08-12）
-+
-+### Setting（双线）
-+| 线 | 机器 / repo | model | retrieval | Candidate A/B |
-+|---|---|---|---|---|
-+| **非 H100（H20）** | 8×H20；`/data/ppnm/Capability_Evolution/SCAPE` | `/data/ppnm/models/Qwen2.5-7B-Instruct` | BM25 provisional | A=`auto_populate_first_search`；B=`verify_tool` |
-+| **H100** | 8×H100；`/mnt/songzijun/Capability_Evolution/SCAPE` | `pat-jj/harness-1`（已 restore + vLLM smoke） | local BM25 compat / offline stub（**非**官方 Chroma） | A=`subtractive_curation`；B=`importance_tagging` |
-+
-+### 进度板 — 非 H100（H20 provisional）
-+| 阶段 | 状态 | 结论 / 产物 |
-+|---|---|---|
-+| Repo bootstrap + pytest | **已完成** | 14 passed；代码在 umbrella `main/SCAPE` |
-+| LOCAL_CAL64 LOO 9/9 + 候选选择 | **已完成** | A/B 选出；`outputs/local_cal64_loo/`、`CANDIDATE_SELECTION.json` |
-+| A/B H_-m collect train-512 | **已完成** | A uniq=512；B uniq=512（jsonl 含 resume 重复行）；`stage_l_hminus_data/` |
-+| B Stage L OPD（L64×3 + L200×3 + heldout×2） | **已完成**（provisional） | `GATE_L_B.json` **pass=true** |
-+| B L64 HF 可服务权重 | **已完成** | `.../B_verify_opd_provisional/L64_seed42_hf/hf_model` |
-+| A L64 HF OPD + 权重 | **已完成** | `.../A_auto_opd_provisional/L64_seed42_hf/hf_model`；loss≈0.122 |
-+| B Stage S closed-loop 四格 | **已完成** | 真实 S2/S3（非 proxy）；**Gate S = FAIL** |
-+| A Stage S closed-loop 四格 | **已完成** | 真实 S2/S3；**Gate S = FAIL** |
-+| Stage M / Pareto / retirement 宣称 | **未开始（停止）** | 单组件 Gate S 未过 → 不进 multi-component |
-+| 真 SCAPE same-state tool-token OPD | **未完成** | LOO 无完整 ξ_t dump；Gate L 仅为 SCOPE-OPD 代理路径 |
-+| GPU 实验进程 | **空闲** | 相关 vLLM/rollout/completion loop 已停 |
-+
-+### 进度板 — H100（自 `result-record-from-h100.md` 同步）
-+| 阶段 | 状态 | 结论 / 产物 |
-+|---|---|---|
-+| H100-1 Phase 0/1 contribution LOO | **已完成（local BM25）** / 官方 Chroma **阻塞** | 10 组件 n=200 errors=0；见下方 H100-1 表 |
-+| H100-2 replication + coalition | **已完成（frozen consolidation）** | 4 modules + 6 coalition rows；非原 10-component REPL200 全量 |
-+| H100-3 same-state influence | **已完成（offline INF_CAL64）** | 10 组件 × 256 states；deterministic stub |
-+| H100-1 × H100-3 quadrant map | **已完成** | `CONTRIBUTION_INFLUENCE_MAP.md`；四象限 |
-+| H100-2 placement stability | **已完成** | `PLACEMENT_STABILITY.md` |
-+| Harness-1 restore + vLLM smoke | **已完成（smoke）** | 9 shards；`/v1/models` 200 |
-+| 官方 BrowseComp+ Chroma eval | **阻塞** | 缺 `OPENAI_API_KEY` / `CHROMA_API_KEY` / `CHROMA_DATABASE` |
-+| H100-3 confirm/targeted 扩展 | **未开始** | `INF_CONFIRM128` 等未跑 |
-+| 官方 Chroma H100-1/2 全量 parity | **未开始/阻塞** | 不可用 local/offline 证据冒充 |
-+
-+### 结论（一句话）
-+- **非 H100**：LOCAL_CAL64 + BM25+Qwen 下 A/B **不可 retirement**（Gate S FAIL）；Stage M 已停。
-+- **H100**：local/offline 贡献·复现·影响力地图已齐；强平衡候选为 `evidence_graph` / `chunk_neighbors`；**不可**据此宣称官方 Harness-1 Chroma parity 或 released-checkpoint retirement。下一步仍需官方凭证或换候选。
-+
-+详细数字：非 H100 见 `## 2026-08-12 SCAPE non-H100 round final`；H100 见 `## 2026-08-12 SCAPE H100-1/2/3 synced status`。
- 
- ---
- 
-@@ -27,3 +73,320 @@ UNRESOLVED
- 
- ### Decision
- 完成 canonical repo + 测试全绿 + 推送 Github；暂不启动 Stage L/S/M 训练。
-+
-+---
-+
-+## 2026-08-11 LOCAL_CAL64 LOO aggregate + candidate select
-+
-+### Setting
-+- path: `/data/ppnm/Capability_Evolution/SCAPE/outputs/local_cal64_loo`
-+- model: Qwen2.5-7B-Instruct (vLLM TP=1, CAL64 BM25 provisional)
-+- n_queries: 64 unique / job; quality gate unique≥64 & err_rate≤0.15
-+- jobs: full + 8 minus_* (9/9 quality-complete)
-+
-+### Results
-+| metric | value |
-+|---|---:|
-+| quality_complete | 9/9 |
-+| Candidate A | auto_populate_first_search |
-+| Candidate B | verify_tool |
-+| placement_map | outputs/scape_prestage/CAPABILITY_PLACEMENT_MAP.md |
-+| selection_json | outputs/scape_prestage/CANDIDATE_SELECTION.json |
-+
-+### Paired
-+- LOO contribution from full vs minus_* CAL64 rollouts
-+- influence values are provisional proxies pending real same-state influence probe
-+
-+### Gate
-+PARTIAL — LOO aggregate done; Stage L scaffolding + dry_run distill started; real OPD data path not yet wired
-+
-+### Decision
-+Proceed Stage L learnability for A=`auto_populate_first_search`, B=`verify_tool`. Prefer waiting was satisfied (9/9). Next: wire real reduced-harness same-state collection → tool-OPD training cells.
-+
-+---
-+
-+## 2026-08-11 Stage L B-verify provisional OPD L64_seed42
-+
-+### Setting
-+- path: `outputs/stage_l/B_verify_opd_provisional/`
-+- stack: SCOPE `smoke_opd_vllm_hf` + `train_opd` (provisional LOCAL_CAL64; H100 not required)
-+- GPUs: 2–5 (TP=4 vLLM rollout → HF train)
-+- cell: L64 seed=42 · target_module=verification · student=ablate_verification · teacher=modules_full · epochs=1
-+
-+### Results
-+| metric | value |
-+|---|---:|
-+| smoke DONE | yes |
-+| smoke opd_loss | 0.0486 |
-+| L64 n_transitions | 64 |
-+| L64 epoch0 loss | 0.1220 |
-+| L64 opd_loss | 0.7293 |
-+| checkpoint | `L64_seed42/checkpoint.json` status=saved |
-+
-+### Paired
-+- （当时）Collect A/B H_-m 未完成 → **后续已于 2026-08-12 完成 512**（见本轮总览）
-+- Next cell: L64_seed43 started on freed GPU2–5
-+
-+### Gate
-+PARTIAL（条目当时）→ **已被本轮总览覆盖**：Gate L 后续 PASS；collect 已完成
-+
-+### Decision
-+Record L64_seed42 metrics; advance seed43 on free GPUs. Do not stop for empty H100 imports.
-+
-+---
-+
-+## 2026-08-11 Stage L B-verify provisional OPD L200
-+
-+### Setting
-+- path: `outputs/stage_l/B_verify_opd_provisional/`
-+- stack: SCOPE `train_opd` (provisional LOCAL_CAL64; H100 not required)
-+- cells: L200 seed42 (GPU2–5 TP4 :8769); L200 seed43 (GPU6–7 TP2 :8770)
-+- target_module=verification · student=ablate_verification · teacher=modules_full · epochs=1
-+
-+### Results
-+| metric | seed42 | seed43 |
-+|---|---:|---:|
-+| n_transitions | 200 | 200 |
-+| epoch0 loss | 0.1220 | 0.1296 |
-+| opd_loss | 0.7293 | 0.8308 |
-+| checkpoint | saved | saved |
-+| status | DONE | DONE |
-+
-+### Paired
-+- Prior L64: s42 loss=0.122 / opd=0.729; s43 loss=0.119 / opd=0.860; s44 loss=0.130 / opd=0.988
-+- （当时）A/B collect → **后续已完成**（见本轮总览）
-+
-+### Gate
-+PARTIAL（条目当时）→ **已被本轮总览覆盖**：held-out×2 + L200×3 已完成；Gate L PASS；closed-loop Gate S FAIL
-+
-+### Decision
-+Record L200 seed42/43; free GPU2–7; start B L64 held-out (`--split test`) while collect continues.
-+
-+
-+## 2026-08-12 SCAPE non-H100 round final
-+
-+> 覆盖并取代同日自动追加的 `non_h100_closed_loop_complete` / `non_h100_completion` 草稿（其中仍写 collect 进行中 / S2S3 proxy 的条目作废）。
-+> 状态：**本轮非 H100 主线已完成**；Stage M **不启动**。
-+
-+### Setting
-+- repo: `/data/ppnm/Capability_Evolution/SCAPE`
-+- model: Qwen2.5-7B-Instruct（vLLM serve / HF train）
-+- retrieval: BM25 provisional（BrowseComp-Plus index）；**非**官方 Harness-1 Chroma
-+- benchmark: BrowseComp-Plus
-+- H100: unavailable — 全程不依赖 `imports/h100_*`
-+- Candidate A: `auto_populate_first_search` · OPD `target_module=evidence_state` · student=`ablate_auto_seed.yaml`
-+- Candidate B: `verify_tool` · OPD `target_module=verification` · student=`ablate_verification.yaml`
-+- teacher harness: `modules_full.yaml` / LOO full V8D mask
-+- Stage S eval: CAL64 `split=test` n=64；S0/S1=LOO；S2/S3=served `L64_seed42_hf/hf_model`
-+- H_-m collect: `split=train` limit=512 · mask 去掉对应组件
-+- output roots:
-+  - LOO: `outputs/local_cal64_loo/`
-+  - collect: `outputs/stage_l_hminus_data/`
-+  - B OPD/Gate L: `outputs/stage_l/B_verify_opd_provisional/` + `GATE_L_B.json`
-+  - A OPD: `outputs/stage_l/A_auto_opd_provisional/`
-+  - B four-grid: `outputs/stage_s/B_verify_fourgrid/`
-+  - A four-grid: `outputs/stage_s/A_auto_fourgrid/`
-+  - narrative: `outputs/NON_H100_FINAL_REPORT.md`
-+
-+### Results
-+
-+#### 状态汇总
-+| item | status | note |
-+|---|---|---|
-+| LOO 9/9 | **已完成** | quality-complete |
-+| Candidate select A/B | **已完成** | A score≈0.0072；B score≈0.0011 |
-+| B Gate L | **已完成 · PASS** | provisional SCOPE-OPD；非 full tool-token |
-+| A/B HF student ckpt | **已完成** | 可 vLLM 服务 |
-+| A/B H_-m collect 512 | **已完成** | A 512 uniq；B 512 uniq（834 lines w/ resume dups） |
-+| B Gate S closed-loop | **已完成 · FAIL** | CCR_m≈0.855；不可 retirement |
-+| A Gate S closed-loop | **已完成 · FAIL** | CCR_m≈0.536；不可 retirement |
-+| Stage M / Pareto | **未开始** | 按 auto-stop 规则停止 |
-+| H100 / Chroma 官方线 | **阻塞 / 未开始** | 本轮不依赖 |
-+
-+#### B = verify_tool（closed-loop 四格，n_shared=64）
-+| cell | J (curated_recall) | C (tool-call proxy) | source |
-+|---|---:|---:|---|
-+| S0 θ0+H_full | 0.0372 | 34.24 | LOO |
-+| S1 θ0+H_-verify | 0.0275 | 32.95 | LOO |
-+| S2 θ'+H_-verify | 0.0358 | 34.98 | closed-loop HF |
-+| S3 θ'+H_full | 0.0429 | 34.23 | closed-loop HF |
-+
-+| metric | value |
-+|---|---:|
-+| CCR_m | 0.855 |
-+| HRR | 0.152 |
-+| Gate S verdict | **FAIL** |
-+| can_claim_retired | false |
-+
-+#### A = auto_populate_first_search（closed-loop 四格，n_shared=64）
-+| cell | J | C | source |
-+|---|---:|---:|---|
-+| S0 θ0+H_full | 0.0372 | 34.24 | LOO |
-+| S1 θ0+H_-auto | 0.0084 | 34.16 | LOO |
-+| S2 θ'+H_-auto | 0.0238 | 34.67 | closed-loop HF |
-+| S3 θ'+H_full | 0.0429 | 34.23 | closed-loop HF |
-+
-+| metric | value |
-+|---|---:|
-+| CCR_m | 0.536 |
-+| HRR | 0.152 |
-+| Gate S verdict | **FAIL** |
-+| can_claim_retired | false |
-+
-+#### Stage L（B，摘录）
-+| cell | status |
-+|---|---|
-+| L64 seeds 42/43/44 + L64_seed42_hf | DONE |
-+| L200 seeds 42/43/44 | DONE |
-+| L64 heldout seeds 42/43 | DONE |
-+| L200_seed45 / L512_seed42 | **未跑**（Gate S 已 FAIL，不再扩） |
-+
-+### Paired
-+- S0/S1：同一 CAL64 query 集上 full vs minus 组件的 LOO paired quality
-+- S2/S3：同一 query 集上 θ'（OPD HF）vs θ0 的 closed-loop paired 评测
-+- B：去掉 verify 后缺口约 0.0097 J，OPD 恢复约 85%（CCR），但仍低于 S0，且 C 未实质下降
-+- A：去掉 auto_populate 后缺口更大；OPD 仅恢复约 54%，远未非劣于 S0
-+
-+### Gate
-+- Gate L (B): **PASS**（provisional）
-+- Gate S (B): **FAIL**
-+- Gate S (A): **FAIL**
-+- Stage M: **不进入**
-+- Overall round: **COMPLETED (non-H100 line)** / retirement claim: **REJECTED**
-+
-+### Decision
-+停止对 A/B 的 retirement 宣称与 Stage M；本轮 provisional 线归档。下一步只做其一：**(1)** 等 H100/官方 Chroma 后重跑 LOO→Gate；或 **(2)** 换下一候选组件（遵守「连续两失败则停救」）。不在当前 BM25+Qwen 线上继续扩 seed / multi-component。
-+
-+---
-+
-+## 2026-08-12 SCAPE H100-1/2/3 synced status
-+
-+> 自 `result-record-from-h100.md` 同步的实验 setting / 结果 / 结论。
-+> 路径以 H100 机为准：`/mnt/songzijun/Capability_Evolution/SCAPE`（对应本机 `/data/ppnm/Capability_Evolution/SCAPE` 同源树）。
-+> 状态词汇：**已完成** / **进行中/阻塞** / **未开始**。
-+
-+### Overall status
-+| Workstream | Todo target | Current status | Output / evidence | Notes |
-+|---|---|---|---|---|
-+| H100-1 Phase 0/1 | Harness-1 reproduction + 10-component LOO contribution map | **已完成（local BM25 compat）/ 进行中（official Chroma parity）** | `outputs/h100_1_contribution/{RUN_MANIFEST.json,STATUS_LIVE.md,COMPONENT_CONTRIBUTION.*,SHA256SUMS}` | Local BM25 compatibility contribution sweep finished for all 10 components, n=200, errors=0. Official Chroma Cloud eval blocked by missing credentials. |
-+| H100-2 | independent replication + coalition interaction | **已完成（frozen consolidation）/ 部分偏离原 10-component REPL200 plan** | `outputs/h100_2_replication_coalition/{RUN_MANIFEST.json,STATUS_LIVE.md,LOO_REPLICATION.csv,COALITION_INTERACTION.csv,REPLICATION_REPORT.md,PLACEMENT_STABILITY.md,SHA256SUMS}` | 4 replicated modules + 6 coalition rows, errors=0. No new training/retrieval. |
-+| H100-3 | same-environment-state policy influence map | **已完成（offline deterministic INF_CAL64）** | `outputs/h100_3_influence/{RUN_MANIFEST.json,STATUS_LIVE.md,INFLUENCE_BY_COMPONENT.*,INFLUENCE_PER_STATE.jsonl,H100_3_INFLUENCE_REPORT.md,SHA256SUMS}` | 64 queries × 4 states/query = 256 states/component; deterministic offline scorer. |
-+| H100-1 × H100-3 | contribution/influence quadrant map | **已完成** | `outputs/CONTRIBUTION_INFLUENCE_MAP.md` | 10 components → four quadrants. |
-+| Official Harness-1 serving | restore model and local vLLM smoke | **已完成（smoke）/ 进行中（official eval）** | `outputs/h100_1_official_vllm` | Restored from `harness-1.tar.gz`, 9 shards, vLLM smoke passed. |
-+| H100-3 confirm/targeted | `INF_CONFIRM128`, targeted influence/mining | **未开始** | none | Optional follow-ups not launched. |
-+| H100-1/2 official parity | Chroma-backed BrowseComp+ LOO/replication | **未开始/阻塞** | none beyond local/proxy | Requires official retrieval credentials. |
-+
-+### H100-1 setting
-+- Run id: `h100_1_local_bm25_contribution_20260811`
-+- Repo: `/mnt/songzijun/Capability_Evolution/SCAPE`（git `61f7741a…` dirty at manifest）
-+- Env: `/opt/bishop-harness/bin/python`；Python 3.11.6；torch 2.11.0+cu130；vLLM 0.25.1；8×H100
-+- Backend: `local_bm25_compat`（**非**官方 Chroma Cloud）
-+- Split/seed: BrowseComp+ CAL200，seed 1101；smoke 1/5/20 亦 errors=0
-+- Decode: deterministic compatibility；无训练 / 无改权重
-+- Status: `n_expected=10`，`n_finished=10`，`remaining=0`，`errors=0`
-+
-+### H100-1 results
-+| component | n | Δ curated | Δ trajectory | Δ final | Δ reward | Status |
-+|---|---:|---:|---:|---:|---:|---|
-+| subtractive_curation | 200 | +0.001556 | +0.000000 | +0.000000 | +0.000700 | 已完成 |
-+| importance_tagging | 200 | +0.001000 | +0.000000 | +0.000000 | +0.000450 | 已完成 |
-+| auto_populate_first_search | 200 | +0.000000 | +0.010298 | +0.000000 | +0.004634 | 已完成 |
-+| evidence_graph | 200 | +0.000000 | +0.001667 | +0.000000 | +0.000750 | 已完成 |
-+| sentence_compress | 200 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | 已完成 |
-+| chunk_neighbors | 200 | +0.000000 | +0.001667 | +0.000000 | +0.000750 | 已完成 |
-+| content_dedup | 200 | +0.000833 | +0.004583 | +0.000000 | +0.002438 | 已完成 |
-+| verify_tool | 200 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | 已完成 |
-+| token_budget_marker | 200 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | 已完成 |
-+| adaptive_rerank_instruction | 200 | -0.001250 | +0.000000 | +0.002917 | -0.000271 | 已完成 |
-+
-+#### H100-1 conclusion
-+- LOO 仅对 **local BM25 compatibility** 路径完成。
-+- 综合 `Δ curated + Δ trajectory + Δ final` 最强：`auto_populate_first_search` → `content_dedup` → `adaptive_rerank_instruction` / `evidence_graph` / `chunk_neighbors`。
-+- `sentence_compress`、`verify_tool`、`token_budget_marker` 在本地质量指标上中性。
-+- **不可**用本 run 宣称官方 Harness-1 reproduction/parity。
-+
-+### H100-2 setting
-+- Run id: `h100_2_replication_coalition_20260811`
-+- Env: `/opt/vllm-qwen3-1.7b/bin/python`；Python 3.12.13；torch 2.11.0+cu130；vLLM 0.25.1；8×H100
-+- Replication input: `SCOPE/outputs/h100_2_module_utility`（fresh200 module-utility）
-+- Coalition input: `SCOPE/outputs/h100_2_exact_budget_factorial`（exact-budget factorial）
-+- Seed/decode: seed 42；temperature=0, top_p=1, do_sample=false
-+- Status: `n_expected=5`，`n_finished=5`，`errors=0`
-+
-+### H100-2 results — replication
-+| module | ablated condition | n | Δ final-answer recall | Δ trajectory recall | Δ reward | paired final W/L/T | paired trajectory W/L/T | Status |
-+|---|---|---:|---:|---:|---:|---|---|---|
-+| context_budget | minus_context_budget | 200 | +0.003345 | -0.000671 | +0.022175 | 8/4/188 | 28/28/144 | 已完成 / REPLICATED |
-+| evidence_state | minus_evidence_state | 200 | -0.001786 | +0.002148 | +0.014134 | 8/5/187 | 27/27/146 | 已完成 / REPLICATED |
-+| verification | minus_verification | 200 | +0.010575 | +0.016813 | +0.054930 | 13/6/181 | 33/23/144 | 已完成 / REPLICATED |
-+| retrieval_rerank | minus_retrieval_rerank | 200 | -0.005571 | -0.007124 | -0.008528 | 6/6/188 | 25/30/145 | 已完成 / REPLICATED |
-+
-+### H100-2 results — coalition
-+| model | budget | N | Q | QS | sequential interaction gap | interpretation | Status |
-+|---|---:|---:|---:|---:|---:|---|---|
-+| qwen3_1p7b | 256 | 0.0300 | 0.0300 | 0.0200 | -0.0100 | diminishing_returns | 已完成 |
-+| qwen3_1p7b | 512 | 0.0300 | 0.0500 | 0.0400 | -0.0300 | diminishing_returns | 已完成 |
-+| qwen3_1p7b | 1024 | 0.0400 | 0.0400 | 0.0400 | +0.0000 | near_additive | 已完成 |
-+| qwen3_30b | 256 | 0.0100 | 0.0000 | 0.0000 | +0.0100 | super_additive | 已完成 |
-+| qwen3_30b | 512 | 0.0200 | 0.0200 | 0.0000 | -0.0200 | diminishing_returns | 已完成 |
-+| qwen3_30b | 1024 | 0.0300 | 0.0100 | 0.0100 | +0.0200 | super_additive | 已完成 |
-+
-+#### H100-2 conclusion
-+- `verification` 是最清晰的稳定正复现模块（final / trajectory / reward 皆正）。
-+- `context_budget`、`evidence_state` 跨轴符号不一致 → placement/domain-sensitive。
-+- `retrieval_rerank` 两路 recall 皆负 → interaction/benchmark-sensitive。
-+- Coalition 多为 diminishing/near-additive，仅作交互备注，非强协同证据。
-+- 本 run ≠ 原 H100-2 10-component REPL200 全量计划；是 frozen SCOPE 输出的 consolidation。
-+
-+### H100-3 setting
-+- Run id: `h100_3_influence_offline_cal64`
-+- Env: `/root/miniforge3/bin/python`；Python 3.13.13；offline scorer（无 torch/vLLM 依赖）
-+- Scale: INF_CAL64；64 queries/component；max 4 states/query；256 states/component；共 2560 per-state records
-+- Scorer: `deterministic_offline_stub`；无训练
-+- Status: `n_expected=10`，`n_finished=10`，`errors=0`
-+- H100 侧 A/B 候选：`subtractive_curation` / `importance_tagging`（与非 H100 线 A/B 不同）
-+
-+### H100-3 results
-+| component | n_queries | n_states | event_support | normalized influence | Status |
-+|---|---:|---:|---:|---:|---|
-+| subtractive_curation | 64 | 256 | 256 | 0.134885 | 已完成 |
-+| importance_tagging | 64 | 256 | 256 | 0.107081 | 已完成 |
-+| verify_tool | 64 | 256 | 256 | 0.010138 | 已完成 |
-+| chunk_neighbors | 64 | 256 | 256 | 0.009933 | 已完成 |
-+| evidence_graph | 64 | 256 | 256 | 0.007756 | 已完成 |
-+| content_dedup | 64 | 256 | 256 | 0.007324 | 已完成 |
-+| auto_populate_first_search | 64 | 256 | 256 | 0.005417 | 已完成 |
-+| token_budget_marker | 64 | 256 | 256 | 0.005255 | 已完成 |
-+| sentence_compress | 64 | 256 | 256 | 0.003571 | 已完成 |
-+| adaptive_rerank_instruction | 64 | 256 | 256 | 0.001980 | 已完成 |
-+
-+#### H100-3 conclusion
-+- 最高 same-state influence：`subtractive_curation`、`importance_tagging`。
-+- 中档：`verify_tool`、`chunk_neighbors`、`evidence_graph`、`content_dedup`。
-+- 最低：`adaptive_rerank_instruction`、`sentence_compress`、`token_budget_marker`。
-+- 本图有效为 offline deterministic same-state 产物；**不是** released Harness-1 logprob 枚举。
-+- `INF_CONFIRM128` / targeted 扩展 **未开始**。
-+
-+### Cross-map conclusions（H100-1 + H100-3）
-+- Source: `outputs/CONTRIBUTION_INFLUENCE_MAP.md`
-+- Thresholds: contribution median `0.001611`；influence median `0.007540`
-+
-+| quadrant | components | conclusion | Status |
-+|---|---|---|---|
-+| High Δ, High I | `evidence_graph`, `chunk_neighbors` | 冻结 local/offline 证据下最强平衡迁移候选 | 已完成 |
-+| High Δ, Low I | `auto_populate_first_search`, `content_dedup`, `adaptive_rerank_instruction` | 质量/运行时效应清晰，same-state 策略位移弱 | 已完成 |
-+| Low Δ, High I | `subtractive_curation`, `importance_tagging`, `verify_tool` | 改策略但本地质量提升弱；保留/移除前需复核 | 已完成 |
-+| Low Δ, Low I | `sentence_compress`, `token_budget_marker` | 本分析下的直接移除候选 | 已完成 |
-+
-+### H100 lightweight / proxy 附注（同源记录）
-+| item | gate / key metric | Decision |
-+|---|---|---|
-+| H20 lightweight torch L/S/M/Pareto | PASS / LIGHTWEIGHT_TORCH_COMPLETE；best L_m≈0.962；S2 quality≈0.030 | 可作为 lightweight 产物；**非** official checkpoint retirement |
-+| qrel-backed pre-stage + H20 torch | PASS / LIGHTWEIGHT_TORCH_PROXY_COMPLETE；A/B Gate L PASS | 同上；官方 Chroma 评测仍独立 |
-+| Official model restore + vLLM smoke | PASS / MODEL_RESTORED_AND_VLLM_SMOKE_COMPLETE | 可继续接官方 eval；缺 3 个 secret vars |
-+
-+### Final decision / next actions（H100）
-+- **已完成**：H100-1 local BM25 contribution；H100-2 frozen replication/coalition + placement；H100-3 offline influence；贡献×影响力四象限；Harness-1 restore + vLLM smoke；lightweight torch proxy L/S/M。
-+- **进行中/阻塞**：官方 BrowseComp+（缺 Chroma/OpenAI 凭证）。
-+- **未开始**：官方 Chroma H100-1/2 parity；`INF_CONFIRM128`；targeted influence/mining；released-checkpoint retirement 宣称。
-+- **禁止宣称**：不可把 local BM25 / offline / proxy 证据写成官方 Harness-1 Cloud/Chroma parity 或最终 retirement。
-diff --git a/SCAPE/scape/probes/candidate_selector.py b/SCAPE/scape/probes/candidate_selector.py
-index d2a46d9..cf5dd69 100644
---- a/SCAPE/scape/probes/candidate_selector.py
-+++ b/SCAPE/scape/probes/candidate_selector.py
-@@ -28,7 +28,10 @@ def placement_score(row: Mapping[str, Any]) -> float:
-     contrib = float(row.get("contribution", 0.0))
-     influence = float(row.get("influence_above_null", 0.0))
-     sem = float(row.get("semantic_fraction", _semantic_fraction(str(row["component_id"]))))
--    cost = max(1e-6, float(row.get("runtime_cost", 1.0)))
-+    raw_cost = float(row.get("runtime_cost", 1.0))
-+    # Non-positive cost means removing the component does not save runtime in the
-+    # current estimate; do not let that become an artificially huge priority.
-+    cost = raw_cost if raw_cost > 0 else float("inf")
-     return (max(0.0, contrib) * max(0.0, influence) * sem) / cost
- 
- 
-diff --git a/SCAPE/scripts/local_cal64_bootstrap.py b/SCAPE/scripts/local_cal64_bootstrap.py
-index 752008e..f7d1ac0 100755
---- a/SCAPE/scripts/local_cal64_bootstrap.py
-+++ b/SCAPE/scripts/local_cal64_bootstrap.py
-@@ -9,8 +9,13 @@ from __future__ import annotations
- 
- import argparse
- import json
-+import sys
- from pathlib import Path
- 
-+REPO = Path(__file__).resolve().parents[1]
-+if str(REPO) not in sys.path:
-+    sys.path.insert(0, str(REPO))
-+
- from scape.adapters.components import all_component_ids, component_specs
- from scape.common.manifest import build_run_manifest, finalize_run_manifest, write_run_manifest
- from scape.common.status import write_status_live
-@@ -107,7 +112,8 @@ def main() -> None:
-                     report["metrics"].get("curated_recall", {}).get("mean_delta", 0.0)
-                 ),
-                 "influence_above_null": float(infl["I_name_mean"] - infl["null_field_order_mean"]),
--                "runtime_cost": float(full[qids[0]]["context_tokens"] - minus[qids[0]]["context_tokens"] + 1.0),
-+                # Positive cost means the component costs extra runtime/context to keep.
-+                "runtime_cost": float(minus[qids[0]]["context_tokens"] - full[qids[0]]["context_tokens"] + 1.0),
-                 "quality_positive": bool(report["quality_positive"]),
-                 "provisional": True,
-             }
-diff --git a/SCAPE/scripts/preflight_harness1.py b/SCAPE/scripts/preflight_harness1.py
-index 4cf1241..fb39e5f 100755
---- a/SCAPE/scripts/preflight_harness1.py
-+++ b/SCAPE/scripts/preflight_harness1.py
-@@ -56,6 +56,13 @@ def main() -> int:
-             if pkg != "vllm":
-                 report["blocked"].append(f"{pkg} missing")
- 
-+    try:
-+        import yaml  # type: ignore
-+        report["checks"]["yaml"] = {"ok": True, "version": getattr(yaml, "__version__", "?")}
-+    except Exception as exc:  # noqa: BLE001
-+        report["checks"]["yaml"] = {"ok": False, "error": str(exc)}
-+        report["blocked"].append("yaml missing")
-+
-     harness = REPO / "external" / "harness-1"
-     report["checks"]["harness1_checkout"] = {
-         "ok": harness.exists(),
-@@ -68,13 +75,20 @@ def main() -> int:
-         report["ok"] = False
-         report["blocked"].append("external/harness-1 missing")
- 
--    # Retrieval backend: do not silently fall back to SCOPE BM25
-+    # Retrieval backend: do not silently fall back to SCOPE BM25.
-+    # SCAPE_RETRIEVAL_CORPUS is a SCAPE-local qrel-aligned JSONL corpus exported
-+    # from stored BrowseComp+ raw document text; upstream Harness-1 CloudClient
-+    # still requires CHROMA_* credentials for official evaluation.
-     chroma = os.environ.get("SCAPE_CHROMA_PATH") or os.environ.get("HARNESS1_CHROMA_PATH")
-+    corpus = os.environ.get("SCAPE_RETRIEVAL_CORPUS") or str(REPO / "outputs" / "retrieval" / "browsecomp_local_corpus_v2" / "corpus.jsonl")
-+    retrieval_ok = bool((chroma and Path(chroma).exists()) or (corpus and Path(corpus).is_file()))
-     report["checks"]["retrieval_backend"] = {
--        "ok": bool(chroma and Path(chroma).exists()),
-+        "ok": retrieval_ok,
-         "path": chroma,
-+        "corpus": corpus,
-+        "kind": "chroma" if chroma else ("scape_jsonl_corpus" if corpus else None),
-     }
--    if not (chroma and Path(chroma).exists()):
-+    if not retrieval_ok:
-         report["blocked"].append("retrieval backend missing")
-         blocked_doc = REPO / "docs" / "BLOCKED_RETRIEVAL_BACKEND.md"
-         blocked_doc.parent.mkdir(parents=True, exist_ok=True)
+## full git diff
+```diff
diff --git a/SCOPE/harness/agent.py b/SCOPE/harness/agent.py
index 615be75..e68b168 100644
--- a/SCOPE/harness/agent.py
+++ b/SCOPE/harness/agent.py
@@ -2,6 +2,7 @@
 
 import json
 import json_repair
+import os
 import re
 import time
 import uuid
@@ -82,6 +83,7 @@ class InferenceContext:
     max_tokens: Optional[int] = None
     previous_response_id: Optional[str] = None
     skip_response_id_update: bool = False
+    telemetry: Dict[str, Any] = field(default_factory=dict)
 
 
 class AgentResult(ABC):
@@ -106,15 +108,19 @@ class OpenAIAgentInferenceModel(AgentInferenceModel):
         model: str = "gpt-5",
         max_output_tokens: int = 4096,
         temperature: float = 1.0,
+        top_p: float = 1.0,
         reasoning_effort: Optional[str] = None,
         api_style: str = "responses",
+        seed: Optional[int] = None,
     ):
         self.openai_client = openai_client
         self.model = model
         self.max_output_tokens = max_output_tokens
         self.temperature = temperature
+        self.top_p = top_p
         self.reasoning_effort = reasoning_effort
         self.api_style = api_style
+        self.seed = seed
 
     def __call__(self, context: InferenceContext) -> Optional[Action]:
         if self.api_style == "chat_completions":
@@ -175,6 +181,7 @@ class OpenAIAgentInferenceModel(AgentInferenceModel):
             "tools": request_tools,  # type: ignore[arg-type]
             "parallel_tool_calls": True,
             "temperature": self.temperature,
+            "top_p": self.top_p,
             "max_output_tokens": max_tokens or self.max_output_tokens,
         }
         if context.previous_response_id:
@@ -200,21 +207,109 @@ class OpenAIAgentInferenceModel(AgentInferenceModel):
 
         request_messages = trajectory.to_provider_format(ProviderFormat.OPENAI)
         request_tools = toolset.get_formats(ProviderFormat.OPENAI)
-        response: ChatCompletion = self.openai_client.chat.completions.create(
-            messages=request_messages,
-            tools=request_tools,  # type: ignore[arg-type]
-            parallel_tool_calls=True,
-            model=self.model,
-            temperature=self.temperature,
-            max_completion_tokens=max_tokens,
-        )
+        text_tool_mode = os.environ.get("SCOPE_TEXT_TOOL_MODE", "0") == "1"
+        if text_tool_mode:
+            tool_names = [
+                t["function"]["name"]
+                for t in request_tools
+                if isinstance(t, dict) and isinstance(t.get("function"), dict)
+            ]
+            text_instruction = {
+                "role": "system",
+                "content": (
+                    "Do not use native tool calls. Instead output exactly one JSON object "
+                    "representing the next action, with keys `tool` and `arguments`. "
+                    f"Allowed tool names: {', '.join(tool_names)}. "
+                    "Examples: {\"tool\": \"search_corpus\", \"arguments\": {\"query\": \"...\"}}; "
+                    "{\"tool\": \"fan_out_search\", \"arguments\": {\"queries\": [\"...\"]}}; "
+                    "{\"tool\": \"curate\", \"arguments\": {\"add_ids\": [\"doc_id\"], \"remove_ids\": []}}; "
+                    "{\"tool\": \"end_search\", \"arguments\": {\"reasoning\": \"done\"}}."
+                ),
+            }
+            response: ChatCompletion = self.openai_client.chat.completions.create(
+                messages=[text_instruction, *request_messages],
+                model=self.model,
+                temperature=self.temperature,
+                top_p=self.top_p,
+                max_completion_tokens=max_tokens,
+                seed=self.seed,
+            )
+        else:
+            response: ChatCompletion = self.openai_client.chat.completions.create(
+                messages=request_messages,
+                tools=request_tools,  # type: ignore[arg-type]
+                parallel_tool_calls=True,
+                model=self.model,
+                temperature=self.temperature,
+                top_p=self.top_p,
+                max_completion_tokens=max_tokens,
+                seed=self.seed,
+            )
         if not response.choices:
             raise RuntimeError("No response choices received from OpenAI")
 
         choice = response.choices[0]
         message: ChatCompletionMessage = choice.message
+        context.telemetry["request_messages"] = request_messages
+        context.telemetry["request_tools"] = request_tools
+        context.telemetry["request_model"] = self.model
+        context.telemetry["request_temperature"] = self.temperature
+        context.telemetry["request_top_p"] = self.top_p
+        context.telemetry["request_max_completion_tokens"] = max_tokens
+        context.telemetry["request_seed"] = self.seed
+        context.telemetry["response_finish_reason"] = choice.finish_reason
+        context.telemetry["response_content"] = self._extract_chat_message_text(message)
+        context.telemetry["response_reasoning_content"] = str(getattr(message, "reasoning_content", "") or "")
+        context.telemetry["response_tool_calls"] = [
+            {
+                "id": getattr(tool_call, "id", None),
+                "name": getattr(getattr(tool_call, "function", None), "name", None),
+                "arguments": getattr(getattr(tool_call, "function", None), "arguments", None),
+            }
+            for tool_call in (message.tool_calls or [])
+        ]
+        try:
+            context.telemetry["raw_response"] = response.model_dump(mode="json")
+        except Exception:  # noqa: BLE001
+            context.telemetry["raw_response"] = str(response)
         action_builder = ActionBuilder()
 
+        if text_tool_mode:
+            text = self._extract_chat_message_text(message)
+            try:
+                parsed = json.loads(text)
+            except json.JSONDecodeError:
+                try:
+                    parsed = json_repair.loads(text)
+                except Exception:
+                    parsed = None
+            raw_calls: List[Tuple[str, Any]] = []
+            if isinstance(parsed, dict):
+                if "tool" in parsed:
+                    raw_calls.append((str(parsed.get("tool") or ""), parsed.get("arguments") or parsed))
+                elif "name" in parsed:
+                    raw_calls.append((str(parsed.get("name") or ""), parsed.get("arguments") or parsed))
+                elif "tool_calls" in parsed and isinstance(parsed.get("tool_calls"), list):
+                    for call in parsed["tool_calls"]:
+                        if isinstance(call, dict):
+                            raw_calls.append((str(call.get("tool") or call.get("name") or call.get("tool_name") or ""), call.get("arguments") or call.get("parameters") or call))
+            elif isinstance(parsed, list):
+                for call in parsed:
+                    if isinstance(call, dict):
+                        raw_calls.append((str(call.get("tool") or call.get("name") or call.get("tool_name") or ""), call.get("arguments") or call.get("parameters") or call))
+            if raw_calls:
+                from harness.action_repair import repair_action_from_tool_calls
+                from training.train_rl import CurateTool, EndSearchTool
+
+                repaired = repair_action_from_tool_calls(
+                    tool_calls=raw_calls,
+                    toolset=toolset,
+                    curate_tool=toolset.get_tool("curate") or CurateTool(),
+                    end_search_tool=toolset.get_tool("end_search") or EndSearchTool(),
+                )
+                if repaired.tools:
+                    return repaired
+
         reasoning_content = getattr(message, "reasoning_content", None)
         if reasoning_content:
             action_builder.add_reasoning(reasoning_content)
diff --git a/SCOPE/harness/config.py b/SCOPE/harness/config.py
index d2c090a..2b6fcc0 100644
--- a/SCOPE/harness/config.py
+++ b/SCOPE/harness/config.py
@@ -9,15 +9,24 @@ from functools import lru_cache
 from pathlib import Path
 
 import anthropic
-from baseten_performance_client import PerformanceClient
+try:
+    from baseten_performance_client import PerformanceClient
+except ImportError:  # pragma: no cover - optional dependency for local BM25 paths
+    PerformanceClient = None  # type: ignore[assignment]
 try:
     import pysqlite3  # type: ignore
     sys.modules["sqlite3"] = pysqlite3
 except Exception:
     pass
-import chromadb
+try:
+    import chromadb
+except ImportError:  # pragma: no cover - optional dependency for local BM25 paths
+    chromadb = None  # type: ignore[assignment]
 import structlog
-import tinker
+try:
+    import tinker
+except ImportError:  # pragma: no cover - optional dependency for local BM25 paths
+    tinker = None  # type: ignore[assignment]
 from openai import OpenAI
 from pydantic import SecretStr
 from pydantic_settings import BaseSettings, SettingsConfigDict
@@ -85,6 +94,8 @@ class Config(BaseSettings):
     contextual_api_key: SecretStr = SecretStr("EXAMPLE")
 
     def get_chroma_client(self) -> chromadb.ClientAPI:
+        if chromadb is None:
+            raise ImportError("chromadb is not installed; Chroma backend unavailable")
         return chromadb.CloudClient(
             api_key=self.chroma_api_key.get_secret_value(),
             database=self.chroma_database,
@@ -114,9 +125,15 @@ class Config(BaseSettings):
         )
 
     def get_tinker_service_client(self) -> tinker.ServiceClient:
+        if tinker is None:
+            raise ImportError("tinker is not installed; tinker service client unavailable")
         return tinker.ServiceClient(api_key=self.tinker_api_key.get_secret_value())
 
     def get_baseten_client(self) -> PerformanceClient:
+        if PerformanceClient is None:
+            raise ImportError(
+                "baseten_performance_client is not installed; Baseten reranker is unavailable"
+            )
         return PerformanceClient(
             base_url=self.baseten_model_url,
             api_key=self.baseten_api_key.get_secret_value(),
diff --git a/SCOPE/harness/llm_env.py b/SCOPE/harness/llm_env.py
index b053ef8..8a92aba 100644
--- a/SCOPE/harness/llm_env.py
+++ b/SCOPE/harness/llm_env.py
@@ -39,9 +39,13 @@ class LlmSettings(BaseSettings):
                 "LLM API not configured. Set base_url, api_key, and model_name "
                 "in BiSHOP/.env (OpenAI-compatible endpoint)."
             )
+        import os
+
+        timeout = float(os.environ.get("OPENAI_TIMEOUT", "120"))
         return OpenAI(
             base_url=self.base_url.strip().rstrip("/"),
             api_key=self.api_key.get_secret_value(),
+            timeout=timeout,
         )
 
 
diff --git a/SCOPE/harness/rerank.py b/SCOPE/harness/rerank.py
index 7f7942c..73ad557 100644
--- a/SCOPE/harness/rerank.py
+++ b/SCOPE/harness/rerank.py
@@ -5,7 +5,12 @@ from typing import Callable, List, Optional
 
 import requests
 import structlog
-from baseten_performance_client import ClassificationResponse, PerformanceClient
+
+try:
+    from baseten_performance_client import ClassificationResponse, PerformanceClient
+except ImportError:  # pragma: no cover - optional dependency for local reranker paths
+    ClassificationResponse = object  # type: ignore[assignment]
+    PerformanceClient = object  # type: ignore[assignment]
 
 from harness.config import get_config
 
diff --git a/SCOPE/harness/retrieval/bm25_backend.py b/SCOPE/harness/retrieval/bm25_backend.py
index bb9189f..b255fe9 100644
--- a/SCOPE/harness/retrieval/bm25_backend.py
+++ b/SCOPE/harness/retrieval/bm25_backend.py
@@ -6,6 +6,8 @@ import json
 import os
 import re
 from dataclasses import dataclass
+
+import regex as regex_module
 from pathlib import Path
 from typing import Any
 
@@ -15,6 +17,8 @@ logger = structlog.get_logger("harness.retrieval.bm25_backend")
 
 _REPO_ROOT = Path(__file__).resolve().parents[2]
 _DEFAULT_INDEX_ROOT = _REPO_ROOT / "external" / "BrowseComp-Plus" / "indexes" / "bm25"
+_GREP_TEXT_CHAR_LIMIT = int(os.environ.get("BROWSECOMP_BM25_GREP_TEXT_CHAR_LIMIT", "200000"))
+_GREP_REGEX_TIMEOUT_S = float(os.environ.get("BROWSECOMP_BM25_GREP_REGEX_TIMEOUT_S", "0.05"))
 
 
 @dataclass(frozen=True)
@@ -114,18 +118,25 @@ class BrowseCompBm25Backend:
         if not pattern.strip():
             return []
         try:
-            regex = re.compile(pattern)
-        except re.error:
-            regex = re.compile(re.escape(pattern))
+            regex = regex_module.compile(pattern)
+        except regex_module.error:
+            regex = regex_module.compile(regex_module.escape(pattern))
+
+        def matches(text: str) -> bool:
+            haystack = text[:_GREP_TEXT_CHAR_LIMIT]
+            try:
+                return regex.search(haystack, timeout=_GREP_REGEX_TIMEOUT_S) is not None
+            except TimeoutError:
+                return pattern in haystack
 
         pool = self.search(pattern, k=prefetch)
-        matched = [hit for hit in pool if regex.search(hit.text)]
+        matched = [hit for hit in pool if matches(hit.text)]
         if matched:
             return matched[:k]
 
         # Fallback: widen recall with a generic query.
         pool = self.search("document", k=prefetch)
-        matched = [hit for hit in pool if regex.search(hit.text)]
+        matched = [hit for hit in pool if matches(hit.text)]
         return matched[:k]
 
     def get_document(self, doc_id: str) -> str | None:
diff --git a/SCOPE/harness/retrieval/bm25_tools.py b/SCOPE/harness/retrieval/bm25_tools.py
index 8f3c762..bf6bb77 100644
--- a/SCOPE/harness/retrieval/bm25_tools.py
+++ b/SCOPE/harness/retrieval/bm25_tools.py
@@ -2,6 +2,7 @@
 
 from __future__ import annotations
 
+import os
 from typing import Any, Callable, Dict, List, Optional, Tuple, cast
 
 from harness.rerank import Reranker
@@ -18,6 +19,12 @@ from harness.tools import (
 )
 
 DEFAULT_SNIPPET_MAX_CHARS = 2048
+BROWSECOMP_BM25_SEARCH_LIMIT = int(os.environ.get("BROWSECOMP_BM25_SEARCH_LIMIT", "50"))
+BROWSECOMP_BM25_DISPLAY_LIMIT = int(os.environ.get("BROWSECOMP_BM25_DISPLAY_LIMIT", "10"))
+BROWSECOMP_BM25_SNIPPET_MAX_CHARS = int(os.environ.get("BROWSECOMP_BM25_SNIPPET_MAX_CHARS", str(DEFAULT_SNIPPET_MAX_CHARS)))
+BROWSECOMP_BM25_GREP_LIMIT = int(os.environ.get("BROWSECOMP_BM25_GREP_LIMIT", "5"))
+BROWSECOMP_BM25_GREP_PREFETCH = int(os.environ.get("BROWSECOMP_BM25_GREP_PREFETCH", "100"))
+BROWSECOMP_BM25_READ_MAX_TOKENS = int(os.environ.get("BROWSECOMP_BM25_READ_MAX_TOKENS", "4096"))
 
 
 def _format_doc_blocks(
@@ -54,9 +61,9 @@ class Bm25SearchCorpusTool(Tool):
         *,
         reranker: Reranker | None = None,
         token_counter: Callable[[str], int] | None = None,
-        snippet_max_chars: int | None = DEFAULT_SNIPPET_MAX_CHARS,
-        search_limit: int = 50,
-        display_limit: int = 10,
+        snippet_max_chars: int | None = BROWSECOMP_BM25_SNIPPET_MAX_CHARS,
+        search_limit: int = BROWSECOMP_BM25_SEARCH_LIMIT,
+        display_limit: int = BROWSECOMP_BM25_DISPLAY_LIMIT,
     ) -> None:
         super().__init__(tool_schema=SEARCH_CORPUS_SCHEMA)
         self._backend = backend
@@ -129,7 +136,11 @@ class Bm25GrepCorpusTool(Tool):
         if not isinstance(params, dict) or "pattern" not in params:
             raise ValueError(f"Invalid params type: {type(params)}")
         pattern = str(params["pattern"])
-        hits = self._backend.grep(pattern, k=5)
+        hits = self._backend.grep(
+            pattern,
+            k=BROWSECOMP_BM25_GREP_LIMIT,
+            prefetch=BROWSECOMP_BM25_GREP_PREFETCH,
+        )
         body, ids = _format_doc_blocks(
             [h.doc_id for h in hits],
             [h.text for h in hits],
@@ -148,7 +159,7 @@ class Bm25ReadDocumentTool(Tool):
         *,
         reranker: Reranker | None = None,
         token_counter: Callable[[str], int] | None = None,
-        max_tokens: int | None = 4096,
+        max_tokens: int | None = BROWSECOMP_BM25_READ_MAX_TOKENS,
     ) -> None:
         if max_tokens is not None and token_counter is None:
             raise ValueError("token_counter is required when max_tokens is specified")
diff --git a/SCOPE/harness/tools.py b/SCOPE/harness/tools.py
index 326e302..6ea3c75 100644
--- a/SCOPE/harness/tools.py
+++ b/SCOPE/harness/tools.py
@@ -31,12 +31,35 @@ try:
     sys.modules["sqlite3"] = pysqlite3
 except Exception:
     pass
-import chromadb
-from chromadb.api.types import SearchResult
+try:
+    import chromadb
+    from chromadb.api.types import SearchResult
+    from chromadb.utils.embedding_functions import Bm25EmbeddingFunction
+except ImportError:  # pragma: no cover - optional for local BM25-only paths
+    chromadb = None  # type: ignore[assignment]
+    SearchResult = Any  # type: ignore[assignment]
+    Bm25EmbeddingFunction = None  # type: ignore[assignment]
 import openai
-import tenacity
+try:
+    import tenacity
+except ImportError:  # pragma: no cover - optional for local BM25-only paths
+    class _TenacityShim:
+        @staticmethod
+        def retry(*_args, **_kwargs):
+            def deco(fn):
+                return fn
+
+            return deco
+
+        @staticmethod
+        def stop_after_attempt(*_args, **_kwargs):
+            return None
 
-from chromadb.utils.embedding_functions import Bm25EmbeddingFunction
+        @staticmethod
+        def wait_exponential(*_args, **_kwargs):
+            return None
+
+    tenacity = _TenacityShim()  # type: ignore[assignment]
 from pydantic import BaseModel, Field
 from harness.utils import ProviderFormat
 import json
@@ -75,17 +98,8 @@ _CHROMA_SEARCH_SEMAPHORE = threading.BoundedSemaphore(CHROMA_SEARCH_MAX_CONCURRE
 # ============================================================================
 
 
-@tenacity.retry(
-    stop=tenacity.stop_after_attempt(5),
-    wait=tenacity.wait_exponential(multiplier=1, min=4, max=15),
-    before_sleep=lambda retry_state: logger.warning(
-        "Retrying ChromaDB search...",
-        attempt=retry_state.attempt_number,
-        error=str(retry_state.outcome.exception()) if retry_state.outcome else None,
-    ),
-)
 def _search_with_retry(
-    collection: chromadb.Collection, search: chromadb.Search
+    collection: Any, search: Any
 ) -> SearchResult:
     """Execute a ChromaDB search with retry logic for transient errors."""
     start = time.perf_counter()
@@ -339,13 +353,13 @@ class SearchCorpusTool(Tool):
         reranker: Optional reranker to reorder results by relevance.
     """
 
-    _chroma_client: chromadb.ClientAPI
-    _openai_client: openai.OpenAI
-    _bm25_ef: Bm25EmbeddingFunction  # TODO: consider allowing this to be a field for experiment tracking
+    _chroma_client: Any
+    _openai_client: Any
+    _bm25_ef: Any  # TODO: consider allowing this to be a field for experiment tracking
     _openai_ef_name: str = (
         "text-embedding-3-small"  # TODO: consider allowing this to be a field for experiment tracking
     )
-    _collections: List[chromadb.Collection]
+    _collections: List[Any]
     _reranker: Optional[Reranker] = (
         None  # TODO: consider allowing this to be a field for experiment tracking
     )
@@ -353,8 +367,8 @@ class SearchCorpusTool(Tool):
 
     def __init__(
         self,
-        chroma_client: chromadb.ClientAPI,
-        openai_client: openai.OpenAI,
+        chroma_client: Any,
+        openai_client: Any,
         chroma_collection_name: Union[str, List[str]],
         openai_ef_name: str = "text-embedding-3-small",
         reranker: Optional[Reranker] = None,
@@ -502,14 +516,14 @@ class GrepCorpusTool(Tool):
         token_counter: Optional callable that counts tokens in a string.
     """
 
-    _chroma_client: chromadb.ClientAPI
-    _collections: List[chromadb.Collection]
+    _chroma_client: Any
+    _collections: List[Any]
     _token_counter: Optional[Callable[[str], int]] = None
     tool_schema: ToolSchema
 
     def __init__(
         self,
-        chroma_client: chromadb.ClientAPI,
+        chroma_client: Any,
         chroma_collection_name: Union[str, List[str]],
         token_counter: Optional[Callable[[str], int]] = None,
     ) -> None:
@@ -585,8 +599,8 @@ class ReadDocumentTool(Tool):
     """
 
     tool_schema: ToolSchema
-    _chroma_client: chromadb.ClientAPI
-    _collections: List[chromadb.Collection]
+    _chroma_client: Any
+    _collections: List[Any]
     _reranker: Optional[Reranker] = (
         None  # TODO: consider allowing this to be a field for experiment tracking
     )
@@ -595,7 +609,7 @@ class ReadDocumentTool(Tool):
 
     def __init__(
         self,
-        chroma_client: chromadb.ClientAPI,
+        chroma_client: Any,
         chroma_collection_name: Union[str, List[str]],
         reranker: Optional[Reranker] = None,
         token_counter: Optional[Callable[[str], int]] = None,
diff --git a/SCOPE/inference/evaluate_harness_api.py b/SCOPE/inference/evaluate_harness_api.py
index df3c412..35e0d56 100644
--- a/SCOPE/inference/evaluate_harness_api.py
+++ b/SCOPE/inference/evaluate_harness_api.py
@@ -34,6 +34,41 @@ logger = structlog.get_logger("evaluate_harness_api")
 USE_LEGACY_API_AGENT = os.environ.get("USE_LEGACY_API_AGENT", "0") == "1"
 
 
+def _extract_terminal_action_text(turn_records: list[Any]) -> tuple[str, str, list[str]]:
+    """Return the terminal text emitted by the policy, if any.
+
+    The Ultra retrieval agent concludes with either ``end_search(reasoning=...)``
+    or a repaired ``UserTextTool``.  Persisting that text keeps the rollout
+    scoreable by downstream final-answer extractors without changing the live
+    prompt or tool schema.
+    """
+    if not turn_records:
+        return "", "", []
+    action = getattr(turn_records[-1], "action", None)
+    if action is None:
+        return "", "", []
+    tools = list(getattr(action, "tools", []) or [])
+    params_list = list(getattr(action, "params", []) or [])
+    names: list[str] = []
+    texts: list[str] = []
+    source = ""
+    for tool, params in zip(tools, params_list):
+        name = "user_text" if tool.__class__.__name__ == "UserTextTool" else tool.tool_schema.name
+        names.append(name)
+        payload = dict(params) if isinstance(params, dict) else {}
+        if name == "end_search":
+            text = str(payload.get("reasoning", "")).strip()
+            if text:
+                texts.append(text)
+                source = source or "end_search.reasoning"
+        elif name == "user_text":
+            text = str(payload.get("text", "")).strip()
+            if text:
+                texts.append(text)
+                source = source or "user_text.text"
+    return "\n".join(texts).strip(), source, names
+
+
 def _default_token_counter() -> Callable[[Any], int]:
     enc = tiktoken.get_encoding("o200k_harmony")
 
@@ -55,7 +90,9 @@ def build_api_agent(
     *,
     max_tokens: int,
     temperature: float,
+    top_p: float = 1.0,
     max_trajectory_length: int = 64,
+    seed: int | None = None,
 ) -> TokenBudgetRetrievalSubagent:
     """Legacy TokenBudget agent (Document-XML retrieval subagent)."""
     client = get_llm_client()
@@ -65,7 +102,9 @@ def build_api_agent(
         model=model,
         max_output_tokens=max_tokens,
         temperature=temperature,
+        top_p=top_p,
         api_style="chat_completions",
+        seed=seed,
     )
     text_counter = lambda text: len(tiktoken.get_encoding("o200k_harmony").encode(text))
     return TokenBudgetRetrievalSubagent(
@@ -84,13 +123,16 @@ def _eval_single_query_legacy_sync(
     *,
     max_tokens: int,
     temperature: float,
+    top_p: float,
     max_trajectory_length: int,
+    seed: int | None,
 ) -> Dict[str, Any]:
     _, query_text = dataset.get_query_by_id(qid)
     agent = build_api_agent(
         toolset,
         max_tokens=max_tokens,
         temperature=temperature,
+        top_p=top_p,
         max_trajectory_length=max_trajectory_length,
     )
     prompt = get_retrieval_subagent_prompt(query_text)
@@ -138,7 +180,9 @@ async def _eval_single_query_ultra(
     *,
     max_tokens: int,
     temperature: float,
+    top_p: float,
     max_trajectory_length: int,
+    seed: int | None,
 ) -> Dict[str, Any]:
     _, query_text = dataset.get_query_by_id(qid)
     client = get_llm_client()
@@ -148,7 +192,9 @@ async def _eval_single_query_ultra(
         model=model,
         max_output_tokens=max_tokens,
         temperature=temperature,
+        top_p=top_p,
         api_style="chat_completions",
+        seed=seed,
     )
     env = SlidingWindowSearchEnv(
         toolset=toolset,
@@ -159,6 +205,7 @@ async def _eval_single_query_ultra(
         text_token_counter=text_token_counter,
         max_turns=max_trajectory_length,
     )
+    initial_context_text = getattr(env.wm, "to_text", lambda: "")()
     driver = ChatDecisionDriver(
         env=env,
         inference=inference,
@@ -168,6 +215,9 @@ async def _eval_single_query_ultra(
     start = time.time()
     result = await driver.run()
     elapsed = time.time() - start
+    terminal_text, terminal_text_source, terminal_tool_names = _extract_terminal_action_text(
+        list(result.get("turn_records", []) or [])
+    )
     return {
         "query_id": qid,
         "query": query_text[:80],
@@ -186,6 +236,25 @@ async def _eval_single_query_ultra(
         "model": model,
         "driver": result.get("driver", "ultra_chat_v2"),
         "early_end_blocks": int(result.get("early_end_blocks", 0)),
+        "terminal_action_text": terminal_text,
+        "terminal_action_text_source": terminal_text_source,
+        "terminal_tool_names": terminal_tool_names,
+        "initial_context_text": initial_context_text,
+        "final_context_text": getattr(env.wm, "to_text", lambda: "")(),
+        "final_curated_ids": list(env.wm.curated_ids),
+        "final_pool_ids": list(env.wm.pool_ids),
+        "final_turn_records": [
+            {
+                "turn_id": rec.turn_id,
+                "student_action": rec.student_action.action_type.value,
+                "action_arguments": dict(rec.student_action.arguments),
+                "observation_text": rec.observation_text,
+                "episode_done": rec.episode_done,
+                "metrics": rec.metrics,
+                "inference_telemetry": rec.inference_telemetry or {},
+            }
+            for rec in result.get("turn_records", []) or []
+        ],
     }
 
 
@@ -198,7 +267,9 @@ async def eval_single_query(
     *,
     max_tokens: int,
     temperature: float,
+    top_p: float = 1.0,
     max_trajectory_length: int = 64,
+    seed: int | None = None,
 ) -> Dict[str, Any]:
     try:
         if USE_LEGACY_API_AGENT:
@@ -209,7 +280,9 @@ async def eval_single_query(
                 toolset,
                 max_tokens=max_tokens,
                 temperature=temperature,
+                top_p=top_p,
                 max_trajectory_length=max_trajectory_length,
+                seed=seed,
             )
         else:
             result = await _eval_single_query_ultra(
@@ -220,7 +293,9 @@ async def eval_single_query(
                 text_token_counter,
                 max_tokens=max_tokens,
                 temperature=temperature,
+                top_p=top_p,
                 max_trajectory_length=max_trajectory_length,
+                seed=seed,
             )
         logger.info(
             "api_episode_result",
@@ -232,7 +307,7 @@ async def eval_single_query(
         )
         return result
     except Exception as exc:
-        logger.error("api_episode_failed", qid=qid, error=str(exc)[:500])
+        logger.exception("api_episode_failed", qid=qid, error=str(exc)[:500])
         return {
             "query_id": qid,
             "error": True,
diff --git a/SCOPE/result-record.md b/SCOPE/result-record.md
index 6f2e815..b4a0882 100644
--- a/SCOPE/result-record.md
+++ b/SCOPE/result-record.md
@@ -1,1568 +1,392 @@
-## 当前结论（2026-08-01）
+# SCAPE result-record
 
-- **已成立：** same-state shadow / info-safe · measurement+scorer · Round5 O7 offline 双侧分离 · **Round6 offline cross-score AUROC=1.0（valid522 + 全部 B6 states）** · runtime parity=1.0（adapter/merged/HF）。
-- **尚未成立：** Dup **closed-loop internalization** positive signal · 校准后仍 **FSR≈1.0** · reward 低于 Base · **`RECOMMEND_830=false`**。
-- **Round 6 判定：** `H_RUNTIME/H_SHIFT/H_CALIB/H_FEEDBACK` 均为 false；`ROUND6_CLOSED_LOOP_POSITIVE=false`。
-- **关键发现：** offline 排序与 AUROC 完美，但 per-seed margin 阈值校准 **不能** 将闭环行为拉到 FSR≤5%；O7 在 holdout 上表现为 **几乎全部 SKIP**（先验≈1），非可控 duplicate rejection。
-- **当前 P0：** live closed-loop decision 路径 / score scale 与 offline replay 差异；禁止扩 830、E1、weighting、multi-capability。
-- **主线：** 修 live admission 决策一致性 → 再评估是否需 on-policy Dagg。
+> Canonical experiment log. Append one section per completed stage (see `SCAPE_H20_TRAINING_MIGRATION.md` §18).
+> **状态以本节「本轮总览」为准**；更早条目中写 “still running / proxy” 的已被后续正式结果覆盖。
+> H100 机实验 setting / 结果 / 结论已自 `result-record-from-h100.md` 同步（见下方「H100 同步」节）。
 
 ---
 
-## Qwen rollout 八组实验汇总（2026-08-01）
+## 本轮总览（更新于 2026-08-12）
 
-**范围：** SCOPE 仓库下 Qwen3-1.7B / Qwen3-30B，在 HotpotQA 与 BrowseComp+ 上分别运行 bare rollout 与 full harness rollout，共 8 组。当前结论是 **7/8 已完成，唯一未完成项为 Qwen3-30B BrowseComp+ harness rollout**。
+### Setting（双线）
+| 线 | 机器 / repo | model | retrieval | Candidate A/B |
+|---|---|---|---|---|
+| **非 H100（H20）** | 8×H20；`/data/ppnm/Capability_Evolution/SCAPE` | `/data/ppnm/models/Qwen2.5-7B-Instruct` | BM25 provisional | A=`auto_populate_first_search`；B=`verify_tool` |
+| **H100** | 8×H100；`/mnt/songzijun/Capability_Evolution/SCAPE` | `pat-jj/harness-1`（已 restore + vLLM smoke） | local BM25 compat / offline stub（**非**官方 Chroma） | A=`subtractive_curation`；B=`importance_tagging` |
 
-### 统一 setting
+### 进度板 — 非 H100（H20 provisional）
+| 阶段 | 状态 | 结论 / 产物 |
+|---|---|---|
+| Repo bootstrap + pytest | **已完成** | 14 passed；代码在 umbrella `main/SCAPE` |
+| LOCAL_CAL64 LOO 9/9 + 候选选择 | **已完成** | A/B 选出；`outputs/local_cal64_loo/`、`CANDIDATE_SELECTION.json` |
+| A/B H_-m collect train-512 | **已完成** | A uniq=512；B uniq=512（jsonl 含 resume 重复行）；`stage_l_hminus_data/` |
+| B Stage L OPD（L64×3 + L200×3 + heldout×2） | **已完成**（provisional） | `GATE_L_B.json` **pass=true** |
+| B L64 HF 可服务权重 | **已完成** | `.../B_verify_opd_provisional/L64_seed42_hf/hf_model` |
+| A L64 HF OPD + 权重 | **已完成** | `.../A_auto_opd_provisional/L64_seed42_hf/hf_model`；loss≈0.122 |
+| B Stage S closed-loop 四格 | **已完成** | 真实 S2/S3（非 proxy）；**Gate S = FAIL** |
+| A Stage S closed-loop 四格 | **已完成** | 真实 S2/S3；**Gate S = FAIL** |
+| Stage M / Pareto / retirement 宣称 | **未开始（停止）** | 单组件 Gate S 未过 → 不进 multi-component |
+| 真 SCAPE same-state tool-token OPD | **未完成** | LOO 无完整 ξ_t dump；Gate L 仅为 SCOPE-OPD 代理路径 |
+| GPU 实验进程 | **空闲** | 相关 vLLM/rollout/completion loop 已停 |
 
-| 维度 | HotpotQA | BrowseComp+ |
-| --- | --- | --- |
-| 数据 | `external/hotpotqa_subset_queries.json` 或 `HotpotQA_raw_data_20260730.tar.gz::HotpotQA/hotpot_dev_fullwiki_v1.json` | `external/BrowseComp-Plus/` full 830 queries，BM25 index=`external/BrowseComp-Plus/indexes/bm25` |
-| bare rollout | vLLM backend，temperature=1.0，max_new_tokens=2048，max_model_len=8192 | vLLM backend，temperature=1.0，max_new_tokens=2048，max_model_len=8192，split=`all` |
-| harness rollout | `hotpotqa_local_context` retrieval，max_turns=35，max_tokens=2048，temperature=1.0 | `modules_full_v2.yaml`，BM25 retrieval，max_turns=35，max_tokens=2048，temperature=1.0，reranker=`none` |
-| Qwen3-1.7B | `/mnt/songzijun/models/Qwen3-1.7B` | `/mnt/songzijun/models/Qwen3-1.7B` |
-| Qwen3-30B | `/mnt/songzijun/models/Qwen3-30B-A3B-Instruct-2507` | `/mnt/songzijun/models/Qwen3-30B-A3B-Instruct-2507` |
+### 进度板 — H100（自 `result-record-from-h100.md` 同步）
+| 阶段 | 状态 | 结论 / 产物 |
+|---|---|---|
+| H100-1 Phase 0/1 contribution LOO | **已完成（local BM25）** / 官方 Chroma **阻塞** | 10 组件 n=200 errors=0；见下方 H100-1 表 |
+| H100-2 replication + coalition | **已完成（frozen consolidation）** | 4 modules + 6 coalition rows；非原 10-component REPL200 全量 |
+| H100-3 same-state influence | **已完成（offline INF_CAL64）** | 10 组件 × 256 states；deterministic stub |
+| H100-1 × H100-3 quadrant map | **已完成** | `CONTRIBUTION_INFLUENCE_MAP.md`；四象限 |
+| H100-2 placement stability | **已完成** | `PLACEMENT_STABILITY.md` |
+| Harness-1 restore + vLLM smoke | **已完成（smoke）** | 9 shards；`/v1/models` 200 |
+| 官方 BrowseComp+ Chroma eval | **阻塞** | 缺 `OPENAI_API_KEY` / `CHROMA_API_KEY` / `CHROMA_DATABASE` |
+| H100-3 confirm/targeted 扩展 | **未开始** | `INF_CONFIRM128` 等未跑 |
+| 官方 Chroma H100-1/2 全量 parity | **未开始/阻塞** | 不可用 local/offline 证据冒充 |
 
-### 完成状态与结果
+### 结论（一句话）
+- **非 H100**：LOCAL_CAL64 + BM25+Qwen 下 A/B **不可 retirement**（Gate S FAIL）；Stage M 已停。
+- **H100**：local/offline 贡献·复现·影响力地图已齐；强平衡候选为 `evidence_graph` / `chunk_neighbors`；**不可**据此宣称官方 Harness-1 Chroma parity 或 released-checkpoint retirement。下一步仍需官方凭证或换候选。
 
-| 模型 | 数据集 | 模式 | 状态 | 输出目录 | records / target | errors | recall | trajectory_recall | final_answer_recall | reward | 备注 |
-| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
-| Qwen3-1.7B | HotpotQA | bare | ✅ completed | `outputs/bare_rollout_hotpotqa_qwen3_1p7b_4gpu/` | 7405 / 7405 | 0 | n/a | n/a | n/a | n/a | 7405 unique query_ids，bad_json=0 |
-| Qwen3-1.7B | HotpotQA | harness | ✅ completed | `outputs/harness_rollout_hotpotqa_qwen3_1p7b_4gpu/` | 14580 / 14580 | 43 | 0.050274 | 0.064952 | 0.050274 | 0.181716 | 7405 unique query_ids，parallel=64，bad_json=0 |
-| Qwen3-1.7B | BrowseComp+ | bare | ✅ completed | `outputs/bare_rollout_browsecomp_qwen3_1_7b_4gpu/` | 830 / 830 | 0 | n/a | n/a | n/a | n/a | split=`all`，bad_json=0 |
-| Qwen3-1.7B | BrowseComp+ | harness | ✅ completed | `outputs/harness_rollout_browsecomp_qwen3_1_7b_8gpu_parallel32/` | 830 / 830 | 0 | 0.028161 | 0.202500 | 0.038333 | 0.160796 | manifest parallel=64，max_model_len=32768，bad_json=0 |
-| Qwen3-30B | HotpotQA | bare | ✅ completed | `outputs/bare_rollout_hotpotqa_qwen3_30b_8gpu_20260730/` | 7405 / 7405 | 0 | n/a | n/a | n/a | n/a | bad_json=0 |
-| Qwen3-30B | HotpotQA | harness | ✅ completed | `outputs/harness_rollout_hotpotqa_qwen3_30b_8gpu_parallel32_20260731_151339/` | 7405 / 7405 | 0 | 0.063336 | 0.070763 | 0.063336 | 0.212855 | parallel=64，bad_json=0 |
-| Qwen3-30B | BrowseComp+ | bare | ✅ completed | `outputs/bare_rollout_browsecomp_qwen3_30b_8gpu_20260730/` | 830 / 830 | 0 | n/a | n/a | n/a | n/a | split=`all`，bad_json=0 |
-| Qwen3-30B | BrowseComp+ | harness | ❌ incomplete | `outputs/harness_rollout_browsecomp_qwen3_30b_8gpu_parallel64_20260801_125717/` | 0 / 830 | n/a | n/a | n/a | n/a | n/a | no `harness_rollouts.jsonl` / no manifest；pid not alive |
-
-### 当前结论
-
-1. HotpotQA 上，Qwen3-30B harness 的 recall / reward 高于 Qwen3-1.7B harness：recall 0.063336 vs 0.050274，reward 0.212855 vs 0.181716。
-2. BrowseComp+ 上，Qwen3-1.7B harness 已完整跑完 830 题，recall=0.028161，trajectory_recall=0.202500；Qwen3-30B 只有 bare 完整结果，harness 尚无可用结果文件，不能做完整横向比较。
-3. bare rollout 均只记录生成轨迹，不含 recall/reward 类指标；可用于后续训练/审计数据，不应和 harness metric 直接比较。
-4. Qwen3-30B BrowseComp+ harness 的最近尝试已完成 vLLM ready、BM25 preflight 和 830 pending episodes 初始化，但没有写出 `harness_rollouts.jsonl` / manifest，且 `vllm_server.pid=16495` 已不存活；判定为未完成而非完成失败可评分。
+详细数字：非 H100 见 `## 2026-08-12 SCAPE non-H100 round final`；H100 见 `## 2026-08-12 SCAPE H100-1/2/3 synced status`。
 
 ---
 
-## 分支与实验对照
-
-> **说明：** 早期实验（Phase 0、Round-1、v3 协议、E0）均在 `main` 基线上以**本地未提交代码**跑通；Round 2/3 代码最终合并提交于 `scope/dup-round3-bilateral`。实验**产物目录**与 git 分支解耦——换分支不会移动 `outputs/`。
-
-### 分支谱系
-
-```text
-main @ 3e95fad（origin/main）
-  │  Phase 0 基线冻结（830q）
-  │
-  ├── [本地开发，07-28~07-29 未单独 commit]
-  │     Round-1 Dup-SDI · v3 协议 smoke/audit · E0 100q · Round 2 全部 wave
-  │
-  ├── scope/dup-round2-behavioral @ 3e95fad
-  │     仅本地工作分支指针（与 main 同 commit，**未 push**）
-  │     Round 2 实验在此分支名上跑，但代码当时未入库
-  │
-  ├── scope/dup-round3-bilateral @ ad072b9（origin 已 push）
-  │     一次性提交 Round 2 + Round 3 全部代码（115 files）
-  │
-  ├── scope/dup-round4-objective-repair @ 6b4e88b
-  │     measurement / scorer audit + overfit128（objective 未过）
-  │
-  └── scope/dup-round5-learnability @ 6b4e88b + **本地未提交** Round5 代码
-        Observability / objective tournament / O7 full screen / 100q CL
-        （`scripts/scope_round5/` · `training/scope_round5/` 仍为 untracked）
-```
-
-### 实验 ↔ 分支 ↔ 产物 一览
-
-| 实验阶段                 | 对应文档节      | Git 分支                                  | 关键 commit      | 远程                                  | 产物根目录                                                   |
-| ------------------------ | --------------- | ----------------------------------------- | ---------------- | ------------------------------------- | ------------------------------------------------------------ |
-| Phase 0 基线（830q）     | E6 §Phase 0     | `main`                                    | `1ed533b`        | ✅ `origin/main`                       | `artifacts/baselines/` · `outputs/minimal_runtime_browsecomp_full830/` |
-| v3 协议 smoke/audit      | Step 1–5        | `main` + 本地                             | `3e95fad` 基线   | —                                     | `outputs/scope_v3_protocol_smoke20/` · `outputs/scope_v3_audit_100q/` |
-| Round-1 Dup-SDI 训练     | Step 8 §Round 1 | `main` + 本地                             | `3e95fad` 基线   | —                                     | `artifacts/datasets/dup_sdi_round1/` · `outputs/dup_sdi_round1/` |
-| E0 Distillability 100q   | Stage 0         | `main` + 本地                             | `3e95fad` 基线   | —                                     | `outputs/scope_e0_distillability/` · `artifacts/capability/distillability_map.json` |
-| Round 2 Behavioral Audit | Round 2         | `scope/dup-round2-behavioral`（工作分支） | 代码在 `ad072b9` | ❌ 未 push                             | `outputs/scope_round2/` · `artifacts/datasets/dup_sdi_round2/` · `artifacts/datasets/round2_audit_100q/` |
-| Round 3 Bilateral        | Round 3         | `scope/dup-round3-bilateral`              | **`ad072b9`**    | ✅ `origin/scope/dup-round3-bilateral` | `outputs/scope_round3/` · `artifacts/datasets/dup_sdi_round3/` |
-| Round 4 Objective Repair | Round 4         | `scope/dup-round4-objective-repair`       | `6b4e88b` / `e3d5afa` | ❌ 未确认 push                    | `outputs/scope_round4/` |
-| Round 5 Learnability     | Round 5         | `scope/dup-round5-learnability`           | `6b4e88b` + 本地 | ❌ 未 push                             | `outputs/scope_round5/` |
-
-### 复现注意事项
-
-| 场景                       | 应 checkout                               | 说明                                                         |
-| -------------------------- | ----------------------------------------- | ------------------------------------------------------------ |
-| 仅复现 Phase 0 基线        | `main` @ `3e95fad`                        | 不含 Round 2/3 脚本                                          |
-| 复现 Round 2 训练/评估脚本 | `scope/dup-round3-bilateral` @ `ad072b9`  | Round 2 代码未在 `dup-round2-behavioral` 上单独 commit       |
-| 复现 Round 3               | `scope/dup-round3-bilateral` @ `ad072b9`  | —                                                            |
-| 复现 Round 4/5 数值        | **无需切换分支**                          | 直接读 `outputs/scope_round4|5/`；R5 脚本需本地工作树        |
-| 复现 Round 5 编排          | `scope/dup-round5-learnability` + 本地    | `scripts/scope_round5/` 当时未入库                           |
-| 复现历史实验数值           | **无需切换分支**                          | 直接读 `outputs/` 下 JSON/MD；数据集在 `artifacts/datasets/`（未进 git，需本地保留） |
-
-### 共享协议资产（跨分支冻结）
-
-| 资产                           | 路径                                                       | 首次冻结       | 使用方                          |
-| ------------------------------ | ---------------------------------------------------------- | -------------- | ------------------------------- |
-| BrowseComp+ 100q manifest      | `artifacts/datasets/round2_audit_100q/query_manifest.json` | Round 2 Wave 1 | Round 2–5 closed-loop           |
-| \(H_{\min,\text{v2}}\) runtime | `harness/configs/modules_minimal_v2.yaml`                  | Round 2        | Round 2–5 rollout · closed-loop |
-| Round-1 merged 对照模型        | `outputs/dup_sdi_round1/merged_hf`                         | Round 1        | Round 2 Wave 1 · Round 3 Wave 4 |
-| Distillability map             | `artifacts/capability/distillability_map.json`             | E0 07-29 10:55 | 830q Go/No-Go 参考              |
-| Round3 Dup train/valid         | `artifacts/datasets/dup_sdi_round3/`                       | Round 3        | Round 3–5 训练/offline          |
-| Round4 overfit128              | `artifacts/datasets/dup_sdi_round4_overfit128/`            | Round 4        | Round 4 B4 · Round 5 B3         |
-| Round5 O7 merged checkpoints   | `outputs/scope_round5/merged/o7_r64_seed{42,43,44}`        | Round 5 B6     | 100q closed-loop                |
-
----
-
-## Stage 0 — Module Distillability Probe（E0）
-
-**method 对应：** 估计各模块的 procedural recoverability \(P_m\)，为 procedural / hybrid / runtime-dependent taxonomy 提供证据。
-
-**状态：** ✅ 100q probe 已完成并冻结（2026-07-29 10:55）；⚠️ **taxonomy 尚未冻结**。当前结果用于筛选后续 probe，而不是把所有模块直接定类。
-
-**分支：** `main` @ `3e95fad` + 本地未提交脚本（E0 编排脚本已入库于 `ad072b9`） · **未单独开分支** · 产物不依赖 git
+## 2026-08-11 SCAPE repo_bootstrap
 
 ### Setting
+- repo path: `/data/ppnm/Capability_Evolution/SCAPE`
+- upstream Harness-1 pin: `8ac4012167858f6478fb2a8fd840e4550e2af161`
+- scope: code + tests + docs + scripts (no GPU experiments yet)
+- model: unset (not launched)
+- benchmark: unset
 
-| 项                     | 值                                                           |
-| ---------------------- | ------------------------------------------------------------ |
-| Model                  | `Qwen2.5-7B-Instruct`（base，未训）                          |
-| Benchmark              | BrowseComp+，固定 audit 100q（`artifacts/datasets/e0_audit_100q`，SEED=42） |
-| Retriever              | BM25                                                         |
-| Harness base           | `modules_full_v2.yaml`；FULL 复用 `outputs/harness_rollout_browsecomp_full_v2` |
-| max_turns / max_tokens | 35 / 2048                                                    |
-| temperature            | 1.0                                                          |
-| GPU / vLLM             | GPU4，port 8776，`e0-harness-policy`                         |
-| 对比形态               | capability-level OFF / PROC / FULL（`deterministic_truncation` 无 PROC） |
-| 编排脚本               | `run_e0_distillability_nohup.sh` + `e0_watch_and_rerun.sh` + `e0_status.sh` |
-| 产物目录               | `outputs/scope_e0_distillability/` · `artifacts/capability/distillability_map.json` · `E0_REPORT.md` |
-
-#### ✅ 2026-07-29 — E0 100q 正式冻结（10:55）
-
-**续跑时间线（07-29）**
-
-| 时间        | 任务                                             | 结果                 |
-| ----------- | ------------------------------------------------ | -------------------- |
-| 08:44       | kill 卡住 `external_verification/proc`（23/100） | 旧编排器误标 DONE    |
-| 08:44–09:18 | deterministic_truncation/off                     | ✅ 100/100（~34 min） |
-| 09:19–10:13 | duplicate_evidence/proc                          | ✅ 100/100（~54 min） |
-| 10:13–10:15 | verification_decision/proc（补 2 题）            | ✅ 100/100            |
-| 10:15–10:55 | external_verification/proc（续跑 77 题）         | ✅ 100/100（~40 min） |
-| 10:55       | `build_map.py` → Map + E0_REPORT 冻结            | ✅                    |
-
-**完成度（全部 100/100）**
-
-| Capability               |  OFF | PROC |      FULL |
-| ------------------------ | ---: | ---: | --------: |
-| duplicate_evidence       |    ✅ |    ✅ | ✅（复用） |
-| stop_decision            |    ✅ |    ✅ |         ✅ |
-| evidence_curation        |    ✅ |    ✅ |         ✅ |
-| verification_decision    |    ✅ |    ✅ |         ✅ |
-| external_verification    |    ✅ |    ✅ |         ✅ |
-| deterministic_truncation |    ✅ |  n/a |         ✅ |
-
-**主指标 recall（paired；FULL 复用 Full v2 = 0.0506）**
-
-> 下表保留原始 `build_map` 输出；`Decision` 改为“当前证据状态”，避免把低覆盖 PROC probe 误解释成最终 taxonomy。
-
-| Capability                | \(R_{\text{off}}\) | \(R_{\text{proc}}\) | \(R_{\text{full}}\) | \(\Delta^{\text{proc}}\) | \(\Delta^{\text{full}}\) | \(P_{\text{raw}}\) | CI(\(P\))     |   W/L/T | 当前证据状态                                    |
-| ------------------------- | -----------------: | ------------------: | ------------------: | -----------------------: | -----------------------: | -----------------: | ------------- | ------: | ----------------------------------------------- |
-| duplicate_evidence        |             0.0255 |              0.0343 |              0.0506 |                  +0.0088 |              **+0.0250** |               0.35 | [−1.23, 1.27] | 12/7/81 | **HYBRID-CANDIDATE / LOW-CONF**                 |
-| stop_decision             |             0.0282 |              0.0239 |              0.0506 |                  −0.0043 |                  +0.0224 |              −0.19 | [−6.02, 0.84] | 8/10/82 | **INVALID/WEAK PROC PROBE**                     |
-| evidence_curation         |             0.0383 |              0.0169 |              0.0506 |                  −0.0213 |                  +0.0123 |              −1.74 | [−24.0, 18.5] | 4/12/84 | **RUNTIME-LEANING / LOW-CONF**                  |
-| verification_decision     |             0.0424 |              0.0198 |              0.0506 |                  −0.0226 |                  +0.0082 |              −2.76 | [−22.8, 28.7] | 3/10/87 | **INVALID/WEAK PROC PROBE**                     |
-| external_verification     |             0.0152 |              0.0203 |              0.0506 |                  +0.0051 |              **+0.0353** |               0.15 | [−0.52, 0.60] |  5/6/89 | **EXECUTION RUNTIME-DEPENDENT；ROUTING 未判定** |
-| deterministic_truncation† |             0.0269 |                   — |              0.0506 |                        — |                 ~+0.0237 |                  — | —             |       — | **INVALID PROBE（0 events）**                   |
-
-† `build_map` 对 truncation 报 `no_overlap_queries`（map 中 R=0 为 builder bug）；episodes/summary 实测 \(R_{\text{off}}=0.0269\)，`truncation_events=0`。
-
-**PROC audit**
-
-| Capability               | interventions | shadow_calls | info-safe | 备注                                                         |
-| ------------------------ | ------------: | -----------: | --------- | ------------------------------------------------------------ |
-| duplicate_evidence       |           606 |          606 | ❌         | visibility_violation_rate=3%；PROC 有方向性恢复              |
-| stop_decision            |             0 |            3 | ✅         | 无有效干预，不能据此判 runtime-only                          |
-| evidence_curation        |           700 |          700 | ✅         | 有干预但 \(\Delta^{\text{proc}}<0\)                          |
-| verification_decision    |             0 |            0 | ✅         | 无有效干预，不能据此判 runtime-only                          |
-| external_verification    |             0 |            0 | ✅         | PROC 不暴露 verify tool；只说明“执行能力”不可由该 PROC 形态替代 |
-| deterministic_truncation |             — |            — | —         | PROC 不支持                                                  |
-
-### 当前结论
-
-1. **E0 已验证“capability 具有异质性”的方向，但不足以冻结 taxonomy。** `duplicate_evidence` 是目前唯一同时具备大量 PROC intervention 与正向恢复的模块，因此是最合适的 internalization sanity check。
-2. `stop_decision`、`verification_decision` 的 PROC 几乎没有实际 intervention，当前结果主要反映 **probe coverage 不足**，不是“不可蒸馏”的证据。
-3. `evidence_curation` 有充分 intervention 且 PROC 明显劣于 OFF，是目前较可信的 **runtime-leaning / 当前 proceduralization 失败** 信号，但 CI 仍很宽。
-4. `external_verification` 应拆成两层看：**外部事实获取/执行**天然需要 runtime；“何时触发验证”的 routing decision 仍可能内化。现有 E0 把二者混在同一 capability 中，因此不应以 \(P_{\text{raw}}=0.15\) 直接判定整个能力 runtime-only。
-5. `deterministic_truncation` 在 100q 中 0 events，当前 probe 没有辨识力；扩大到 830 也不一定解决，应先构造 event-enriched probe。
-6. \(P_{\text{raw}}\) 的分母 \(\Delta^{\text{full}}\) 较小且 paired ties 很多，导致 CI 极宽；在 intervention coverage 达标前，\(P_m\) 只作辅助统计，不作硬分类阈值。
-
-### 下一步：不直接扩 E0 830
-
-| 优先级 | Capability               | 动作                                                      | Go 条件                       |
-| ------ | ------------------------ | --------------------------------------------------------- | ----------------------------- |
-| **P0** | duplicate_evidence       | Round 3 已完成（否定性结论）；修 evaluator + 诊断 operation_ce 塌缩 | 指标可信 + objective 修复后重训 main |
-| **P1** | stop_decision            | 修 selector，构造 STOP↔CONTINUE 双侧 event-enriched probe | 两类 intervention 均非零      |
-| **P1** | verification_decision    | 先修触发/打点，再做 targeted probe                        | `n_proc_interventions > 0`    |
-| **P1** | deterministic_truncation | 构造必触发 truncation 的长轨迹子集                        | `truncation_events > 0`       |
-| **P2** | evidence_curation        | 检查当前 PROC 语义是否等价，再决定是否重跑                | procedural artifact 定义稳定  |
-| **P2** | external_verification    | 拆成 routing decision 与 external execution 两层评估      | 不再混合 capability 边界      |
-
-**产物**
-
-- `artifacts/capability/distillability_map.json`（2026-07-29 10:55）
-- `outputs/scope_e0_distillability/E0_REPORT.md`（2026-07-29 10:55）
-- 全量 episodes：`outputs/scope_e0_distillability/{cap}/{off,proc,full}/episodes.jsonl`
-
-**备注：** Round-1 Dup-SDI 训练独立于 E0；E0 使用 base model。原计划的 E0 830 **暂不启动**：当前主要瓶颈是 probe validity / event coverage，而不是样本量。
-
-<details>
-<summary>E0 历史记录：07-28 首轮失败 → 07-29 卡住恢复</summary>
-
-
-#### 首轮失败（07-28 13:42–14:30）
-
-- 大量 OFF/PROC：`Connection error` / PROC `run_information_safe_gates(... artifact=)` 签名 bug。
-- 旧 `E0_REPORT`（14:30）不可信（多模式 `errors=1.0`、`turns=0`）。
-
-#### nohup 续跑（07-28 17:14 起）
-
-| 时间              | 任务                       | 结果                                 |
-| ----------------- | -------------------------- | ------------------------------------ |
-| 17:15             | duplicate_evidence/proc    | OOM Kill，误标 DONE，episodes 空     |
-| 17:15–17:19       | stop_decision/proc         | ✅ 100                                |
-| 17:19–17:52       | evidence_curation/proc     | ✅ 100                                |
-| 17:52–18:02       | verification_decision/off  | ✅ 100                                |
-| 18:02–18:46       | verification_decision/proc | 98/100（2 error）                    |
-| 18:46–19:19       | external_verification/off  | ✅ 100                                |
-| 19:19–19:27       | external_verification/proc | 推进至 ~20–23/100                    |
-| 19:27–07-29 08:44 | external_verification/proc | **卡住** 23/100（PID 3960060，~13h） |
-
-#### 中间态（07-29 08:43）
-
-仅 `stop_decision` / `evidence_curation` 三模式齐全；dup PROC 空；truncation OFF 失败；external PROC 卡住。08:44 kill 后 truncation OFF 续跑，09:19 watch relaunch 补齐其余缺格，10:55 冻结（见上）。
-
-</details>
-
----
-
-## Round 2 — Dup Behavioral Audit（0729-todo1）
-
-**method 对应：** 诊断 Round-1 teacher-forced 拟合 vs \(H_{\min}\) 闭环行为脱节；引入 \(H_{\min,\text{v2}}\)、compact operation target、sample-normalized CE，做 100q 对照与训练消融。
-
-**状态：** ✅ 代码 + Wave 1–3 完成 · ⏭ Wave 4 未单独补跑（由 Round 3 统一 typed-action 闭环替代） · `RECOMMEND_830=false`
-
-**分支：** 工作分支 `scope/dup-round2-behavioral`（本地，@ `3e95fad`，**未 push**）· 代码 commit `ad072b9`（在 `scope/dup-round3-bilateral`）  
-**产物根目录：** `outputs/scope_round2/` · `artifacts/datasets/round2_audit_100q/` · `artifacts/datasets/dup_sdi_round2/`  
-**报告：** `outputs/scope_round2/ROUND2_REPORT.md`
-
-### 代码改动（Barrier 0，96 tests pass）
-
-核心改动：loss-mass audit（`analyze_loss_mass.py`）、operation eval（`eval_dup_capability.py`）、\(H_{\min,\text{v2}}\)（`modules_minimal_v2.yaml`）、compact KEEP/SKIP target、`ActionRealizer`、sample-normalized CE、Stop 四象限统计。编排：`scripts/scope_round2/run_all.sh`。
-
-### 全局 GPU / 协议 Setting
-
-| 项                                   | 值                                                           |
-| ------------------------------------ | ------------------------------------------------------------ |
-| Model                                | `Qwen2.5-7B-Instruct`（Base）；Round1 对照 `outputs/dup_sdi_round1/merged_hf` |
-| Benchmark                            | BrowseComp+，冻结 **100q**（`artifacts/datasets/round2_audit_100q/query_manifest.json`，`SEED=42`，4×25 shard） |
-| Retriever                            | BM25                                                         |
-| Rollout runtime                      | **`modules_minimal_v2.yaml`**（\(H_{\min,\text{v2}}\)：保留 search/BM25/verify tool + hard truncate；关闭 cognitive dedup/curation/stop policy） |
-| vLLM                                 | **1 model / 1 GPU，TP=1**；GPU0–7 → port 8800–8807           |
-| max_turns / max_tokens / temperature | 35 / 2048 / 1.0                                              |
-| 禁止项（本轮未跑）                   | 830q eval · E0 830 · capability weighting · Recovery · RL · Irrelevant |
-
----
-
-#### ✅ Barrier 0 — 代码 + 诊断（2026-07-29）
-
-Round-1 loss-mass audit（755 samples）：ENDORSE 45.0% samples / 21.0% target tokens；CORRECT 54.8% / 79.0%；其中 `verify_claim` 占 60% target-token mass。**确认 sample balance 与 optimization mass 不一致，但这只是根因候选，不是因果证明。**
-
-产物：`outputs/scope_round2/diagnostics/round1_loss_mass.md`
-
-#### ✅ Wave 1 — H_min_v2 100q Closed-loop（Base vs Old Round1）
-
-**Setting**
-
-| 项                | 值                                                         |
-| ----------------- | ---------------------------------------------------------- |
-| Base / Old Round1 | `Qwen2.5-7B-Instruct` / `outputs/dup_sdi_round1/merged_hf` |
-| Runtime           | `modules_minimal_v2.yaml`                                  |
-| 分片              | GPU0–3 Base；GPU4–7 Round1                                 |
-| 脚本              | `training/scope_round2/hmin_v2_rollout.py`                 |
-
-| 指标                  |      Base | Old Round1 | Δ / Paired         |
-| --------------------- | --------: | ---------: | ------------------ |
-| recall                |     2.29% |      3.48% | +1.18pp；12/7/81   |
-| reward                |     0.122 |      0.220 | +0.099；13/14/73   |
-| trajectory_recall     |     24.2% |      25.3% | +1.1pp             |
-| final_answer_recall   |     2.95% |      6.37% | +3.42pp            |
-| mean_turns            |     33.29 |      34.43 | +1.14              |
-| **mean_n_curated**    | **14.35** |  **17.73** | **+3.38；62/32/6** |
-| mean_n_pool           |    288.94 |     286.93 | −2.01              |
-| unique_evidence_ratio |     0.061 |      0.068 | +0.007；62/38/0    |
-
-结论：**100q 中再次出现“更多 curate”的方向，但没有复现 smoke20 的 task degradation。** `duplicate_curate_rate` 当时未 instrument，因此 `mean_n_curated` 只是行为代理，不能等价为“重复 curate”。
-
-产物：`outputs/scope_round2/hmin_v2_{base,round1}/merged/` · `eval/base_vs_round1_100q.md`
-
-#### ✅ Wave 2 — Same-State Shadow + Stop Calibration + Round1 Capability Re-eval
-
-**Dup shadow（Base @ H_min_v2 decision states）**
-
-- 四 shard 并行 labeling → `outputs/scope_round2/dup_shadow/shard0–3/`
-- 修复 compact target 解析后重建数据集
-
-**Stop Calibration 100q（H_min_v2）**
-
-| 象限              | Count |
-| ----------------- | ----: |
-| STOP→STOP         |     0 |
-| STOP→CONTINUE     |    13 |
-| CONTINUE→STOP     |     0 |
-| CONTINUE→CONTINUE |  3316 |
-| n_decision_points |  3329 |
-
-- **bilateral_coverage: False** — 仍缺 CONTINUE→STOP 监督质量
-- 产物：`outputs/scope_round2/stop_calibration/stop_calibration_100q.md`
-
-**Round1 capability re-eval（新指标，valid 77）**
-
-| 指标                     | 值    |
-| ------------------------ | ----- |
-| teacher_forced_token_acc | 93.9% |
-| action_match_rate        | 26.0% |
-| route CORRECT accuracy   | 94.9% |
-| route ENDORSE accuracy   | 82.8% |
-
----
-
-#### ✅ Barrier 2 — Round 2 数据集
-
-**Setting**
-
-| 项          | 值                                |
-| ----------- | --------------------------------- |
-| 来源        | Base @ H_min_v2 same-state shadow |
-| Capability  | `duplicate_evidence` only         |
-| Target 格式 | compact `SKIP_DUPLICATE` JSON     |
-| Split       | query-level，train 257 / valid 30 |
-
-**分布（局限）**
-
-| 项                             | 值          |
-| ------------------------------ | ----------- |
-| KEEP / SKIP                    | 0 / **287** |
-| ENDORSE / CORRECT              | **0 / 287** |
-| visibility / schema violations | 0 / 0       |
-
-⚠️ 仅捕获 duplicate-curate **CORRECT** 点，无 ENDORSE/KEEP → endorse-only 消融无法运行
-
-产物：`artifacts/datasets/dup_sdi_round2/`
-
----
-
-#### ✅ Wave 3 — 8 路训练消融（Barrier 3）
-
-**共同 Setting**
-
-| 项                               | 值                       |
-| -------------------------------- | ------------------------ |
-| Base model                       | `Qwen2.5-7B-Instruct`    |
-| Method                           | LoRA r=16, α=32          |
-| lr / epochs / batch / grad_accum | 2e-5 / 3 / 4 / 4         |
-| max_length                       | 4096                     |
-| KL coef                          | 0.01                     |
-| Dataset                          | `dup_sdi_round2`（同上） |
-
-**Variant 分配**
-
-| GPU  | Variant                        | loss_mode         | compact | route_balance | 备注                          |
-| ---- | ------------------------------ | ----------------- | ------- | ------------- | ----------------------------- |
-| 0    | round2_main                    | sample_normalized | ✅       | ✅             | 主模型                        |
-| 1    | round2_legacy_token_ce         | legacy_token_ce   | ❌       | ❌             | 长序列 token CE 对照          |
-| 2    | round2_full_action_sample_norm | sample_normalized | ❌       | ❌             | 完整 action + sample norm     |
-| 3    | round2_no_route_balance        | sample_normalized | ✅       | ❌             |                               |
-| 4    | round2_endorse_only            | sample_normalized | ✅       | —             | **FAILED**（0 train samples） |
-| 5    | round2_correct_only            | sample_normalized | ✅       | —             | CORRECT only filter           |
-| 6    | round2_main_seed43             | 同 main           | ✅       | ✅             | seed=43                       |
-| 7    | round2_main_seed44             | 同 main           | ✅       | ✅             | seed=44                       |
-
-**结果 — round2_main offline capability（30 valid，compact prompt）**
-
-| 指标                       | 值                      |
-| -------------------------- | ----------------------- |
-| valid loss                 | 0.275                   |
-| parse_rate                 | **1.0**                 |
-| operation_accuracy         | 0.50                    |
-| SKIP_DUPLICATE recall / F1 | **0.50 / 0.67**         |
-| KEEP_EVIDENCE recall       | n/a（valid 全 CORRECT） |
-| teacher_forced_token_acc   | 94.7%（不作为成功标准） |
-
-\* 其余 variant Barrier-3 批量 eval 因 prompt 不匹配 greedy_parse=0；仅 main 经 compact prompt 重评。
-
-产物：`outputs/scope_round2/training/round2_*/` · `eval/round2_training_comparison.md`
-
----
-
-#### ⏭ Wave 4 — Closed-loop 100q（未单独补跑；由 Round 3 替代）
-
-**原计划：** 8 模型 × 同 manifest × \(H_{\min,\text{v2}}\) × merge LoRA → rollout
-
-**未完成原因：** Round 2 推理路径尚未接入 ActionRealizer（compact operation → runtime action）。Round 3 已把 typed operation、ActionRealizer 与 telemetry 统一，因此**不再回头补 Round 2 的旧闭环路径**；仅保留 Round 2 checkpoint 作为 Round 3 diagnostic 对照。
-
-产物占位：`outputs/scope_round2/eval/round2_closed_loop_100q.md`（partial）
-
----
-
-### Round 2 五问结论（0729-todo1 §十三）
-
-| #    | 问题                                  | 当前结论                                                     |
-| ---- | ------------------------------------- | ------------------------------------------------------------ |
-| Q1   | sample balance ≠ loss-token balance？ | **确认存在**。CORRECT 占 79% target-token mass，而 sample share 为 54.8%；`verify_claim` 单项占 60%。这是明确的 objective imbalance。 |
-| Q2   | Round1 @ H_min_v2 复现 smoke20？      | **复现行为方向，不复现任务退化。** mean_n_curated +3.38；但 100q recall/reward 未下降，因此不能把 smoke20 的任务性能下降视为稳定结论。 |
-| Q3   | compact target 改善 operation？       | **仅有弱离线信号。** one-sided valid 上 SKIP recall=50%；没有 KEEP 类，也没有闭环结果。 |
-| Q4   | legacy vs sample-norm vs compact？    | **尚不能比较。** 多个 variant 因 prompt/eval 不匹配未公平重评；当前数据不足以归因到 loss 或 target format。 |
-| Q5   | Endorse vs Correct 贡献？             | **无法回答。** Round 2 数据为 0 ENDORSE / 0 KEEP。           |
-
-### 最终判定
-
-```text
-ROUND1_CONFIRMED_FAILURE = teacher-forced 拟合没有转化为预期的 H_min_v2 行为；
-                           Round1 模型稳定增加 mean_n_curated
-
-PRIMARY_HYPOTHESIS = 长 action span + token-level CE 导致 loss mass 偏向 CORRECT/verify_claim
-                     （有诊断证据，但尚无公平消融证明其为“根因”）
-
-ROUND2_POSITIVE_SIGNAL = NOT_ESTABLISHED
-RECOMMEND_830 = false
-
-ROUND2_WAVE4 = SUPERSEDED_BY_ROUND3_TYPED_ACTION_CLOSED_LOOP
-```
-
-**对后续的有效结论只有两条：** 先修复双侧监督与 train/inference action interface；在这两项完成后，才有资格比较 objective 并讨论是否扩大到 830。Round 3 正是对此的重构。
-
----
-
-## Round 3 — Bilateral Duplicate Capability Internalization（0729-todo2）
-
-**method 对应：** 在 student 真实访问的 evidence-admission decision points 上构造 KEEP/SKIP 双侧监督，用 `operation_ce` 直接优化 operation decision，并统一 train/inference 的 typed action interface（`DupOperationRuntime` + `ActionRealizer`），验证能否在 \(H_{\min,\text{v2}}\) closed-loop 中降低 DuplicateCurateRate 且不显著恶化 FalseSkipRate。
-
-**状态：** ✅ **全部完成**（Barrier A–C · Wave4 · Closed-loop 100q · Final report）· `ROUND3_POSITIVE_SIGNAL=false` · `RECOMMEND_830=false`
-
-**分支：** `scope/dup-round3-bilateral` @ **`ad072b9`**（`origin/scope/dup-round3-bilateral`，2026-07-29 push）  
-**产物根目录：** `outputs/scope_round3/` · `artifacts/datasets/dup_sdi_round3/`  
-**报告：** `outputs/scope_round3/ROUND3_REPORT.md`  
-**前置：** 依赖 Round 2 的 100q manifest 与 \(H_{\min,\text{v2}}\) rollout states；Round 2 训练 checkpoint 在 `outputs/scope_round2/training/`
-
-### 代码改动（Barrier A，109 tests pass）
-
-Round 3 将 Dup 从 error-triggered 改为 **decision-triggered evidence admission**：`DupDecisionPoint` + `DupBilateralShadow` 生成 KEEP/SKIP 双侧 label；`score_operations` / `operation_ce` 与 `DupOperationRuntime` 共用 scorer；`ActionRealizer` 执行 typed operation；`dup_telemetry.py` 记录 admission behavior。编排：`scripts/scope_round3/run_all_8gpu.sh`。
-
-### 全局 GPU / 协议 Setting
-
-| 项                                   | 值                                                           |
-| ------------------------------------ | ------------------------------------------------------------ |
-| Model                                | `Qwen2.5-7B-Instruct`（Base）；Round1 `outputs/dup_sdi_round1/merged_hf`；Round2 `outputs/scope_round2/training/round2_*` |
-| Benchmark                            | BrowseComp+，冻结 **100q**（`artifacts/datasets/round2_audit_100q/query_manifest.json`） |
-| Retriever                            | BM25                                                         |
-| Rollout runtime                      | **`modules_minimal_v2.yaml`**（\(H_{\min,\text{v2}}\)）      |
-| vLLM                                 | **1 model / 1 GPU，TP=1**；Round3 port **8900–8907**（Wave4）/ **8910–8927**（closed-loop） |
-| max_turns / max_tokens / temperature | 35 / 2048 / 1.0                                              |
-| 禁止项（本轮未跑）                   | 830q · E0 830 · capability weighting · Recovery · RL · Premature Stop 训练 · Irrelevant |
-
----
-
-#### ✅ Barrier A — 代码 + 单测（2026-07-29）
-
-`pytest tests/scope/`：109 passed。覆盖 KEEP/SKIP、ENDORSE/CORRECT、ActionRealizer 映射、visibility、train/inference shared scorer。首轮 Wave4 因 `CurateTool` import 路径错误失败，已修复为 `training.train_rl.CurateTool`。
-
-#### ✅ Barrier B — 双侧数据集（2026-07-29）
-
-**Setting**
-
-| 项     | 值                                                           |
-| ------ | ------------------------------------------------------------ |
-| 来源   | Base @ H_min_v2 decision states（Round2 100q rollout 重切 8 shard） |
-| Shadow | `DupBilateralShadow`（decision-triggered，非 duplicate-suspect 触发） |
-| Split  | query-level：**train 80q / valid 20q**（1807 / 522 events）  |
-| Gate   | visibility_violation=0（3 条预过滤）· shadow_mutation=0 · schema_invalid=0 |
-
-**分布（双侧，Round2 单侧问题已修复）**
-
-| 项                    |    Count |
-| --------------------- | -------: |
-| KEEP_EVIDENCE         | **1784** |
-| SKIP_DUPLICATE        |  **545** |
-| ENDORSE               | **1784** |
-| CORRECT               |  **545** |
-| keep/skip ratio       |   3.27:1 |
-| endorse/correct ratio |   3.27:1 |
-
-```text
-ROUND3_DATA_GO = true
-```
-
-产物：`artifacts/datasets/dup_sdi_round3/` · `bilateral_dataset_report.md` · `bilateral_dataset_stats.json`
-
----
-
-#### ✅ Barrier C — 8 路训练消融（2026-07-29）
-
-**共同 Setting**
-
-| 项                               | 值                                                 |
-| -------------------------------- | -------------------------------------------------- |
-| Base model                       | `Qwen2.5-7B-Instruct`                              |
-| Method                           | LoRA r=16, α=32                                    |
-| lr / epochs / batch / grad_accum | 2e-5 / 3 / 4 / 4                                   |
-| max_length                       | 4096                                               |
-| KL coef                          | 0.01（`operation_ce` 路径 KL≈0）                   |
-| Dataset                          | `dup_sdi_round3`（1807 train / 522 valid）         |
-| 优化 steps                       | ~1350 / variant（operation_ce）；correct-only ~330 |
-
-**Variant 分配**
-
-| GPU  | Variant                            | loss_mode         | 备注                         |
-| ---- | ---------------------------------- | ----------------- | ---------------------------- |
-| 0    | round3_op_main_seed42              | **operation_ce**  | route+class balance，seed=42 |
-| 1    | round3_op_main_seed43              | operation_ce      | 同 main，seed=43             |
-| 2    | round3_op_main_seed44              | operation_ce      | 同 main，seed=44             |
-| 3    | round3_compact_json_sample_norm    | sample_normalized | compact JSON 对照            |
-| 4    | round3_legacy_full_action_token_ce | legacy_token_ce   | 表面形式 imitation 对照      |
-| 5    | round3_correct_only_op             | operation_ce      | CORRECT only                 |
-| 6    | round3_endorse_only_op             | operation_ce      | ENDORSE only                 |
-| 7    | round3_op_no_balance               | operation_ce      | 无 class/route balance       |
-
-**训练 loss（epoch 3 末）**
-
-| Variant                     | final_train_loss |
-| --------------------------- | ---------------: |
-| round3_op_main_seed42/43/44 |            ~0.50 |
-| round3_op_no_balance        |            0.519 |
-| round3_compact_json         |            0.226 |
-| round3_legacy_token_ce      |            0.227 |
-| round3_correct_only         |        **0.002** |
-| round3_endorse_only         |           **≈0** |
-
-产物：`outputs/scope_round3/training/round3_*/` · merged：`outputs/scope_round3/merged/`
-
----
-
-#### ✅ 训练前 Baselines + Offline Capability Eval（valid 522）
-
-> ⚠️ **Evaluator sanity check（P0）**：valid 集同时含 KEEP/SKIP 时，“永远 KEEP”的标准 `KEEP F1` 不应为 1.000；“永远 SKIP”的 `SKIP F1` 也不应为 1.000。例如 B1 全 KEEP 且 op_acc≈81% 时，标准 KEEP-F1 应约为 \(2\times0.81/(1+0.81)\approx0.895\)，而不是 1.000。当前脚本很可能把 class recall / class accuracy 误标为 F1，或实现存在错误。因此下列 F1 / macro-F1 **保留原始记录但暂不用于研究结论**。在修复前，可信的主要是预测分布、operation accuracy、KEEP/SKIP recall 等可直接核验量。
-
-**B0 — Majority（永远 KEEP）**
-
-| KEEP F1 | SKIP F1 | macro-F1 |
-| ------: | ------: | -------: |
-|   1.000 |   0.000 |    0.500 |
-
-**B1 — Base operation_ce（未训练，restricted verbalizer scorer）**
-
-| KEEP F1 | SKIP F1 | macro-F1 | op_acc |
-| ------: | ------: | -------: | -----: |
-|   1.000 |   0.000 |    0.500 |  81.0% |
-
-**B2 — Round2 main（compact JSON，公平 operation eval）**
-
-| KEEP F1 |   SKIP F1 | macro-F1 | SKIP recall |
-| ------: | --------: | -------: | ----------: |
-|   0.000 | **0.697** |    0.349 |   **53.5%** |
-
-**Round3 全 variant offline（operation-level，522 valid）**
-
-| Variant                     | KEEP F1 |   SKIP F1 |  macro-F1 | balanced acc | 备注                    |
-| --------------------------- | ------: | --------: | --------: | -----------: | ----------------------- |
-| round3_op_main seed42/43/44 |   1.000 | **0.000** |     0.500 |        0.500 | **≡ Base，全预测 KEEP** |
-| round3_op_no_balance        |   1.000 |     0.000 |     0.500 |        0.500 | 同上                    |
-| round3_endorse_only         |   1.000 |     0.000 |     0.500 |        0.500 | 对照：全 KEEP           |
-| round3_correct_only         |   0.000 | **1.000** |     0.500 |        0.500 | 对照：全 SKIP           |
-| **round3_compact_json**     |   0.983 | **0.061** | **0.522** |        0.522 | 唯一略优于 Base         |
-| round3_legacy_token_ce      |   0.986 |     0.020 |     0.503 |        0.503 | token CE，SKIP 极弱     |
-
-**解读**
-
-- **可信现象：** operation_ce main 三个 seed 均全预测 KEEP，SKIP recall=0%；说明 bilateral discrimination 尚未学出。
-- correct-only / endorse-only 能把模型推向全 SKIP / 全 KEEP，只能证明 objective 能推动 score 到单侧极端，**不能证明双侧分类训练正确**。
-- compact JSON 至少产生少量 SKIP prediction（脚本报告 SKIP recall 6.1%）；在 F1 evaluator 修复前，不再写“优于 Base”或与 Round2 直接排序。
-- teacher-forced token accuracy 仅作拟合诊断，不作为 capability internalization 成功标准。
-
-产物：`outputs/scope_round3/eval/baselines.json` · `outputs/scope_round3/eval/offline_capability.json`
-
----
-
-#### ✅ Wave 4 Diagnostic — 四 checkpoint plumbing（2026-07-30）
-
-**Setting：** Base / Round1 / Round2-main / Round2-legacy × dup-operation + ActionRealizer + telemetry；每 variant **shard0+shard1**（50q diagnostic）；port 8900–8907
-
-| 阶段                | 状态                                                         |
-| ------------------- | ------------------------------------------------------------ |
-| 首轮（07-29 14:52） | ❌ `ImportError: CurateTool from harness.tools`（已修）         |
-| 最终（07-30 03:37） | ✅ 4 variant merged · `wave4_barrier=true` · `plumbing_ok=true` |
-
-**Wave4 结果（50q / variant，dup-operation 路径）**
-
-| Variant       | DupCurateRate | recall | reward | plumbing |
-| ------------- | ------------: | -----: | -----: | -------- |
-| base          |         0.000 |  3.90% |  0.305 | ✅        |
-| round1        |         0.000 |  1.99% |  0.167 | ✅        |
-| round2_main   |         0.000 |  3.90% |  0.305 | ✅        |
-| round2_legacy |         0.000 |  3.90% |  0.305 | ✅        |
-
-> Base 与 Round2 checkpoint 在 dup-operation 路径下 DCR=0（全 KEEP admission），与 offline B1 一致；telemetry 完整、ActionRealizer 无 hidden fallback。
-
-产物：`outputs/scope_round3/wave4_diagnostic/comparison.json` · `comparison.md`
-
----
-
-#### ✅ Closed-loop 100q — Dup 行为主指标（2026-07-30 完成）
-
-**Setting**
-
-| 项   | 值                                                           |
-| ---- | ------------------------------------------------------------ |
-| 协议 | 冻结 manifest · BM25 · \(H_{\min,\text{v2}}\) · dup-operation + ActionRealizer |
-| 规模 | 100q × 8 shard = 8 GPU 并行；每 variant 8 shard merge |
-| 脚本 | `run_post_train_8gpu.sh` → `resume_post_train_8gpu.sh`（错峰 90s + 串行重试） |
-
-**执行记录**
-
-| 时间 | 事件 |
-| ---- | ---- |
-| 07-29 17:11 | 首轮 post-train：8 路并行 vLLM 初始化竞争，多 variant 失败 |
-| 07-29 22:27 | 二次重启：仍因 vLLM `Engine core initialization failed` 中断 |
-| 07-30 01:34 | 三次重启：`resume_post_train_8gpu.sh` 错峰启动 + Wave2 串行重试 |
-| 07-30 03:37 | ✅ 全部 9 变体（含 Base）merged · wave4_compare · final_report |
-
-**Closed-loop 全 variant 对比（100q merged）**
-
-| Variant | DupCurateRate | FalseSkipRate | mean_n_curated | recall | reward |
-| ------- | ------------: | ------------: | -------------: | -----: | -----: |
-| **Base** | **0.000** | **0.000** | 9.13 | 2.68% | 0.165 |
-| round3_op_main_seed42 | 0.139 | 0.815 | 9.26 | 0.95% | 0.042 |
-| round3_op_main_seed43 | 0.178 | 0.892 | 11.40 | 3.44% | 0.127 |
-| round3_op_main_seed44 | 0.127 | 0.789 | 10.18 | 1.85% | 0.076 |
-| **round3_compact_json** | **0.000** | **0.015** | 7.48 | 2.22% | 0.123 |
-| round3_legacy_token_ce | 0.001 | 0.019 | 7.77 | 2.11% | 0.098 |
-| round3_correct_only（对照） | 0.218 | 1.000 | 13.07 | 1.89% | 0.039 |
-| round3_endorse_only（对照） | 0.000 | 0.000 | 8.50 | 1.96% | 0.141 |
-| round3_op_no_balance | 0.137 | 0.841 | 11.59 | 2.55% | 0.106 |
-
-**Paired 统计（seed42 vs Base，100q bootstrap）**
-
-| 指标 | mean Δ | 95% CI | W/L/T |
-| ---- | -----: | ------ | ----- |
-| duplicate_curate_rate | **+0.133** | [+0.114, +0.154] | 80/0/20 |
-| false_skip_rate | **+0.801** | [+0.783, +0.818] | 100/0/0 |
-| recall | −0.020 | [−0.041, +0.000] | 5/12/83 |
-| reward | −0.154 | [−0.283, −0.034] | 8/33/59 |
-
-**解读**
-
-- **operation_ce 主模型（3 seeds）：** offline 全 KEEP（SKIP recall=0%），闭环产生大量 SKIP（FalseSkipRate 79–89%），DupCurateRate 反而上升 — **训练目标与闭环行为严重不一致**。
-- **compact_json：** 唯一在 DCR≈0、FSR≈1.5% 下保持 recall/reward 接近 Base 的变体；offline 有少量 SKIP prediction（6.1% recall）。
-- **correct_only 对照：** FSR=100%、DCR=21.8%，证明 route-filter 训练能把模型推向极端 SKIP 行为。
-- **endorse_only 对照：** 与 Base 行为一致（全 KEEP admission）。
-- **Task retention：** recall/reward 未出现灾难性崩溃（paired recall CI 含 0），但 seed42 reward 显著低于 Base。
-
-产物：`outputs/scope_round3/closed_loop/*/merged/summary.json` · `ROUND3_REPORT.md` · 日志：`outputs/scope_round3/logs/resume_post_train_master.log`
-
----
-
-### Round 3 研究问题结论（0729-todo2 §一）
-
-> 双侧监督 + operation_ce + 统一 action interface 能否让 duplicate_evidence 在 \(H_{\min,\text{v2}}\) closed-loop 中真正降低 DuplicateCurateRate？
-
-| 层面 | 结论 | 依据 |
-| ---- | ---- | ---- |
-| **数据 / Selector（H3）** | ✅ **已解决** | KEEP=1784, SKIP=545；`ROUND3_DATA_GO=true` |
-| **Train/Inference 一致（H2）** | ✅ **已实现** | Wave4 plumbing ✅；shared scorer + runtime + realizer |
-| **Offline capability** | ❌ **未通过** | operation_ce main macro-F1=0.500 ≡ Base；SKIP recall=0% |
-| **Closed-loop behavior** | ❌ **未通过** | main 模型 DCR↑、FSR↑（vs Base）；compact_json 唯一接近 Base |
-| **Task retention** | ✅ **通过** | paired recall Δ≈0；无系统性 recall 崩溃 |
-
-### 根因假设更新（0729-todo2 §十七）
-
-| 假设 | 判定 | 说明 |
-| ---- | ---- | ---- |
-| H1 token-loss-mass distortion | **PARTIALLY_SUPPORTED** | legacy/compact JSON SKIP recall 弱（2–6%）；非主因 |
-| H2 training/inference action mismatch | **SUPPORTED** | Round3 已统一 interface；Round2 根因之一 |
-| H3 selector-induced one-sided supervision | **SUPPORTED** | Round2 0 KEEP/0 ENDORSE → Round3 双侧修复 |
-| H4 operation-value supervision weakness | **SUPPORTED** | operation_ce 主模型 offline 全 KEEP、闭环大量误 SKIP |
-| H5 evaluator correctness | **OPEN（P0）** | majority baseline F1=1.0 不符合标准定义；F1 指标暂不可信 |
-
-### 最终判定
-
-```text
-ROUND3_POSITIVE_SIGNAL = false
-  # offline：operation_ce 未超 Base；SKIP recall=0%
-  # closed-loop：main 模型 DCR/FSR 显著恶化 vs Base
-
-RECOMMEND_830 = false
-
-Capability pass: False
-Behavior pass:     False
-Task retention:    True
-
-NEXT_ACTION = (1) 修复 offline F1 evaluator sanity check
-              (2) 调查 operation_ce 塌缩根因（verbalizer prior / effective class weight / train-infer score 一致性）
-              (3) 对比 compact_json vs operation_ce 的 score margin / 闭环行为差异
-              (4) 在 Dup 最小 positive signal 前不扩 830 / 多能力 / weighting
-```
-
-**下一步已移到文末“全局待办”。**
-
-
-
----
-
-## 训练主循环（每 iteration）
-
-> 本节只记录 method pipeline 与当前实现状态；Round 2/3 的详细数值不再重复。
-
-### Step 1 — Pure Student Rollout（\(\tau^- \sim \pi_\theta \mid H_{\min}\)）
-
-**method 对应：** 学生在 Minimal Runtime 上 on-policy rollout，收集真实访问状态。  
-**状态：** 🔄 已有 \(H_{\min,\text{v2}}\) 100q rollout（Round 2/3）；正式 iteration 数据管线待固化。
-
-**Setting（当前正式计划）**
-
-| 项                                   | 值                                                           |
-| ------------------------------------ | ------------------------------------------------------------ |
-| Model                                | `Qwen2.5-7B-Instruct`                                        |
-| Benchmark                            | BrowseComp+，100q → 通过 gate 后 830                         |
-| Retriever                            | BM25                                                         |
-| Rollout runtime                      | **`modules_minimal_v2.yaml`**；旧 `modules_minimal.yaml` 仅保留历史对照 |
-| max_turns / max_tokens / temperature | 35 / 2048 / 1.0                                              |
-| Scope config                         | `configs/scope/sdi_dup_premature.yaml`                       |
-| Capabilities                         | 先 `duplicate_evidence`；`premature_stop` 待双侧 Stop Calibration 通过后加入 |
-| 用途                                 | 在 student 实际访问状态上生成 same-state supervision         |
-
-**历史 Full-v2 协议 Setting（Smoke/Audit 共用）**
-
-| 项                                   | 值                                          |
-| ------------------------------------ | ------------------------------------------- |
-| Model                                | `Qwen2.5-7B-Instruct`                       |
-| Retriever                            | BM25                                        |
-| Harness                              | `modules_full_v2.yaml`（⚠️ 非 \(H_{\min}\)） |
-| Scope config                         | `configs/scope/sdi_dup_premature.yaml`      |
-| Capabilities                         | `duplicate_evidence`, `premature_stop`      |
-| max_turns / max_tokens / temperature | 35 / 2048 / 1.0                             |
-| Smoke                                | `LIMIT=20`, `SEED=42`, GPU0–3, port 8774    |
-| Audit                                | `LIMIT=100`, `SEED=42`, GPU0–3, port 8775   |
-
-**历史协议验证 timeline**
-
-| 时间             | 实验         | Setting                             | 关键结果                                                     |
-| ---------------- | ------------ | ----------------------------------- | ------------------------------------------------------------ |
-| 2026-07-28 10:28 | Smoke 20q    | Full v2；GPU0–3；port 8774          | 123/123 trainable；Dup 56 ENDORSE + 47 CORRECT；Premature 0 ENDORSE + 20 CORRECT；leakage/mutation=0 |
-| 2026-07-28 11:53 | Natural 100q | Full v2；SEED=42；GPU0–3；port 8775 | 755 events / 754 trainable；Dup 340 ENDORSE + 315 CORRECT；Premature 0 ENDORSE + 99 CORRECT + 1 IGNORE；Irrelevant 全 IGNORE |
-
-产物：`outputs/scope_v3_protocol_smoke20/` · `outputs/scope_v3_audit_100q/natural_100q/`
-
-> 这两次 Full v2 rollout 仅用于协议验证和 Round-1 历史数据；正式训练状态分布改用 \(H_{\min,\text{v2}}\)。
-
----
-
-### Step 2 — DecisionState 构建（\(d_t=\psi(s_t)\)）
-
-**method 对应：** 统一压缩交互状态，要求 \(\operatorname{Info}(d_t)\subseteq\operatorname{Info}(s_t)\)。  
-**状态：** ✅ v3 在线验证；Round 3 增加 `DupDecisionPoint`。
-
----
-
-### Step 3 — Same-State Shadow Guidance（\(z_t^m=h_m(d_t)\)）
-
-**method 对应：** 同状态查询 typed module，返回局部 artifact，而不是 teacher 完整轨迹。  
-**状态：** ✅ Dup 已实现 decision-triggered 双侧 shadow；Premature 的 selector coverage 仍待修。
-
----
-
-### Step 4 — Information-Safe Gate（\(M_t^m\)）
-
-**method 对应：** visibility / schema / executable / module mask。  
-**状态：** ✅ v3 Smoke/Audit leakage=0、shadow_mutation=0；⚠️ E0 Dup PROC 仍有 3% visibility violation，需单独修复。
-
-**Verifier 可靠性探针 timeline：2026-07-28 11:06**
-
-**Setting**
-
-| 项              | 值                                     |
-| --------------- | -------------------------------------- |
-| Model / Harness | —（离线 synthetic probe，无 rollout）  |
-| 数据            | synthetic `DecisionState`，n=24        |
-| Capability      | `premature_stop`                       |
-| Scope config    | `configs/scope/sdi_dup_premature.yaml` |
-| train_mask      | 0（不进训练）                          |
-
-结果：24/24 ENDORSE。说明 verifier 能识别 valid-stop；自然数据缺 positive-stop 主要来自 selector/coverage，而不是该 synthetic probe 中的 verifier failure。
-
-产物：`outputs/scope_v3_audit_100q/targeted_valid_stop/`
-
----
-
-### Step 5 — Verified Decision Routing（ENDORSE / CORRECT）
-
-**method 对应：** endorse → 保留学生动作；verified reject → 使用纠正动作。  
-**状态：** ✅ Dup 双侧 routing 已在 Round 3 建立；⏸ Premature 暂不训练。
-
-**Stop Calibration Setting**
-
-| 项                                   | 值                                                           |
-| ------------------------------------ | ------------------------------------------------------------ |
-| Model                                | `Qwen2.5-7B-Instruct`                                        |
-| Benchmark                            | BrowseComp+，20q smoke → 100q audit                          |
-| Retriever                            | BM25                                                         |
-| Harness                              | `modules_full_v2.yaml`（历史计划）；Round 2 已在 H_min_v2 audit |
-| Scope config                         | `configs/scope/sdi_dup_premature.yaml` + `stop_calibration: true` |
-| Capabilities                         | `duplicate_evidence`, `premature_stop`                       |
-| max_turns / max_tokens / temperature | 35 / 2048 / 1.0                                              |
-| 代码                                 | `stop_calibration.py` · `selectors.py` · `verification_shadow.py` |
-
-2026-07-29 H_min_v2 100q：3329 decision points，STOP→CONTINUE=13，CONTINUE→CONTINUE=3316，CONTINUE→STOP=0，`bilateral_coverage=False`。因此不原样扩大、不训练 Premature；先修 selector / targeted state construction。
-
----
-
-### Step 6 — Capability Weighting（\(w_t^m=P_mU_m(1-\rho_m)\)）
-
-**状态：** ⏸ 暂缓。至少一个 capability 通过可信 closed-loop internalization gate 后再做。
+### Results
+| metric | value |
+|---|---:|
+| pytest | 14 passed |
+| experiments_started | 0 |
 
-**Setting（保留）**
+### Paired
+- (none)
 
-| 项        | 值                                                           |
-| --------- | ------------------------------------------------------------ |
-| Model     | `Qwen2.5-7B-Instruct`                                        |
-| Benchmark | BrowseComp+                                                  |
-| 权重方案  | uniform · \(U_m\) · \(U_m(1-\rho_m)\) · \(P_mU_m(1-\rho_m)\) |
-| 对比实验  | E3                                                           |
+### Gate
+UNRESOLVED
 
-当前：Dup 只有粗 \(U_m\) 统计；held-out \(\rho_m\) 未建立。此时做 weighting 会把 measurement/objective 错误混入权重结论。
+### Decision
+完成 canonical repo + 测试全绿 + 推送 Github；暂不启动 Stage L/S/M 训练。
 
 ---
 
-### Step 7 — Shadow-first, Recovery-on-Demand
+## 2026-08-11 LOCAL_CAL64 LOO aggregate + candidate select
 
-**状态：** ⏸ 按需；首版不主动加入 recovery。
-
-**Setting（保留）**
-
-| 项                  | 值                                   |
-| ------------------- | ------------------------------------ |
-| Model               | `Qwen2.5-7B-Instruct`                |
-| Benchmark           | BrowseComp+                          |
-| Rollout runtime     | `modules_minimal_v2.yaml`            |
-| Recovery 步数 \(K\) | TBD                                  |
-| 触发条件            | \(\delta_t^m>\tau_{\text{recover}}\) |
-| 对比实验            | E4                                   |
-
-仅当真实 rollout 中 premature stop / dead-end 形成稳定 failure mass 时启用。
-
----
-
-### Step 8 — Optimize（\(\mathcal{L}=\mathcal{L}_{\text{SDI}}+\xi\mathcal{L}_{\text{stab}}\)）
-
-**method 对应：** action/operation-level objective + 可选 KL；首版不含 RL / recovery loss。
-
-#### ✅ 2026-07-28 12:38 — Dup-only SDI Round 1
-
-**Setting — 数据**
-
-| 项                        | 值                                                           |
-| ------------------------- | ------------------------------------------------------------ |
-| Model（rollout）          | `Qwen2.5-7B-Instruct`                                        |
-| Benchmark / 来源          | BrowseComp+ `LIMIT=100`, `SEED=42`；`natural_100q/samples.jsonl` |
-| Filter                    | `duplicate_evidence`, `train_mask=1`                         |
-| Split                     | query-level，`valid_fraction=0.1`，seed=42                   |
-| n_samples / train / valid | 655 / 578 / 77                                               |
-| Route                     | ENDORSE 340 + CORRECT 315                                    |
-
-**Setting — 训练 / eval**
-
-| 项                               | 值                                                           |
-| -------------------------------- | ------------------------------------------------------------ |
-| Base                             | `Qwen2.5-7B-Instruct`                                        |
-| LoRA                             | r=16, α=32                                                   |
-| Loss                             | Action-level CE + KL=0.01                                    |
-| lr / epochs / batch / grad_accum | 2e-5 / 3 / 4 / 4                                             |
-| max_length                       | 4096                                                         |
-| Scope config                     | `configs/scope/sdi_dup_only.yaml`                            |
-| GPU                              | 4，~13 min，430 steps                                        |
-| Eval script                      | `training/scope/eval_dup_capability.py`                      |
-| Eval                             | valid 77；greedy，`max_new_tokens=64`，首行 JSON；base + LoRA adapter |
-
-关键结果：valid loss 0.227；teacher-forced token acc 93.9%；parse 100%；action_match 26.0%。**结论修正：序列拟合成功，不等于 decision internalization。**
-
-#### ✅ 2026-07-28 13:42 — Minimal Runtime Smoke20
-
-**Setting**
-
-| 项                                   | 值                                               |
-| ------------------------------------ | ------------------------------------------------ |
-| Base / Trained                       | Base vs `outputs/dup_sdi_round1/merged_hf`       |
-| Benchmark                            | BrowseComp+ 前 20 题（`LIMIT=20`, `SPLIT=all`）  |
-| Retriever                            | BM25                                             |
-| Runtime                              | 历史 `modules_minimal.yaml`（V8D 全关）          |
-| Scope config                         | `configs/scope/minimal_runtime.yaml`             |
-| max_turns / max_tokens / temperature | 35 / 2048 / 1.0                                  |
-| parallel / GPU                       | 2；GPU4–7，vLLM TP=4                             |
-| 脚本                                 | `scripts/run_dup_sdi_minimal_runtime_smoke20.sh` |
-| Phase0 历史参考                      | Minimal 830 recall 2.45%，reward 0.121（非同批） |
-
-关键结果：recall 3.06%→0.71%，reward 0.137→0.013，mean_n_curated 11.15→20.65。Round 2 H_min_v2 100q 进一步确认 mean_n_curated 14.35→17.73，但 recall/reward 未稳定复现下降。故 Round 1 不作为成功模型继续扩 830。
-
-产物：`artifacts/datasets/dup_sdi_round1/` · `outputs/dup_sdi_round1/`
-
-#### ⏸ Minimal Runtime 全量 830（E6 / Retention）
-
-**Setting（计划保留）**
-
-| 项                                   | 值                                                     |
-| ------------------------------------ | ------------------------------------------------------ |
-| Model                                | 新的通过 gate 的 Dup trained model vs Base             |
-| Benchmark                            | BrowseComp+ 830                                        |
-| Retriever                            | BM25                                                   |
-| Runtime                              | **`modules_minimal_v2.yaml`**                          |
-| max_turns / max_tokens / temperature | 35 / 2048 / 1.0                                        |
-| 对比                                 | matched manifest / matched runtime                     |
-| 历史参考                             | Phase0 `modules_minimal.yaml` 2.45%；Full v2 3.80%     |
-| 主指标                               | behavior metric + recall/reward + \(\rho_m\)/Retention |
-
-正式 830 前先补 matched H_min_v2 Base baseline；不再对 Round-1 checkpoint 做全量主线评估。
-
----
-
-### Step 9 — Module Lifecycle（内化 → 降权 → 退役）
-
-**状态：** 📋 TODO。E0 100q 只提供初步 \(P_m\)；仍缺可信 held-out \(\rho_m\) 与 matched Minimal Runtime retention，暂不执行 module retirement。
-
-
-## 实验设计消融（method §4）
-
-### E0 — Module Distillability Map
-
-**状态：** 🔄 100q 原始 probe 已冻结；taxonomy 未冻结。下一步是 targeted/event-enriched validity probe，不直接扩 830。详见 Stage 0。
-
----
-
-### E1 — Full Harness Distillation vs Same-State Local Distillation
-
-**状态：** 📋 TODO，提升为 **P1 核心基线**。
-
-**Setting（计划）**
-
-| 项                                   | 值                                                           |
-| ------------------------------------ | ------------------------------------------------------------ |
-| Model                                | `Qwen2.5-7B-Instruct`                                        |
-| Benchmark                            | BrowseComp+                                                  |
-| Retriever                            | BM25                                                         |
-| max_turns / max_tokens / temperature | 35 / 2048 / 1.0                                              |
-| 对比                                 | Harness-trace SFT · OPHSD-style full context · same-state local label · same-state + info-safe gate |
-| 主指标                               | action/operation decision · fresh-corpus behavior · citation/factual errors · closed-loop task retention |
-
-目标：证明收益来自“same-state typed local supervision / info-safe design”，而不是普通 Harness trace imitation。
-
----
-
-### E2 — 为什么需要 Correct，而不只是 Endorse
-
-**状态：** 🔄 Round 3 已具备双侧数据、单侧 controls 与完整 100q closed-loop；正式结论：`ROUND3_POSITIVE_SIGNAL=false`；待 evaluator 修复后重评 offline F1。
-
-**Setting**
-
-| 项                                   | 值                                                           |
-| ------------------------------------ | ------------------------------------------------------------ |
-| Model                                | `Qwen2.5-7B-Instruct`                                        |
-| Benchmark                            | BrowseComp+                                                  |
-| Retriever                            | BM25                                                         |
-| max_turns / max_tokens / temperature | 35 / 2048 / 1.0                                              |
-| 对比                                 | endorse-only · correct-only · bilateral main · compact/sample-norm · operation objective |
-| 成功标准                             | 双侧 operation + closed-loop behavior + task retention       |
-
----
-
-### E3 — Capability Weighting vs Privilege Illusion
-
-**状态：** ⏸ 至少两个 capability 具备可信 supervision / retention 后再做。
-
-**Setting**
-
-| 项        | 值                                                           |
-| --------- | ------------------------------------------------------------ |
-| Model     | `Qwen2.5-7B-Instruct`                                        |
-| Benchmark | BrowseComp+                                                  |
-| 对比      | uniform · \(U_m\) · \(U_m(1-\rho_m)\) · \(P_mU_m(1-\rho_m)\) |
-
----
-
-### E4 — DAgger-style Mixing vs Shadow-first Recovery
-
-**状态：** ⏸ Recovery 按失败分布启用。
-
-**Setting**
-
-| 项                                   | 值                                                           |
-| ------------------------------------ | ------------------------------------------------------------ |
-| Model                                | `Qwen2.5-7B-Instruct`                                        |
-| Benchmark                            | BrowseComp+                                                  |
-| Retriever                            | BM25                                                         |
-| max_turns / max_tokens / temperature | 35 / 2048 / 1.0                                              |
-| 对比                                 | pure student OPD · DAgger mixture · student-prefix→teacher completion · SCOPE shadow-only · SCOPE shadow+recovery |
-
----
-
-### E5 — Black-box Teacher Compatibility
-
-**状态：** 📋 后置。
-
-**Setting**
-
-| 项        | 值                                              |
-| --------- | ----------------------------------------------- |
-| Model     | `Qwen2.5-7B-Instruct`                           |
-| Benchmark | BrowseComp+                                     |
-| Teacher A | white-box local model                           |
-| Teacher B | API / rule / retriever / verifier mixed Harness |
-| 对比      | logit-OPD vs action-level SCOPE                 |
-
----
-
-### E6 — Module Retirement / Minimal Runtime Pareto
-
-**状态：** 🔄 Phase 0 830 baseline 已完成；trained+H_min_v2 830 暂缓。
-
-#### ✅ 2026-07-28 上午 — Phase 0 基线冻结（830 题）
-
-**分支：** `main` @ `1ed533b` · `origin/main`
-
-**Setting**
-
-| 项                                   | 值                    |
-| ------------------------------------ | --------------------- |
-| Model                                | `Qwen2.5-7B-Instruct` |
-| Benchmark                            | BrowseComp+ 830       |
-| Retriever                            | BM25                  |
-| max_turns / max_tokens / temperature | 35 / 2048 / 1.0       |
-| parallel                             | 2                     |
-
-| Runtime             | Config                 | recall / acc |    reward | mean_turns | mean_n_curated |
-| ------------------- | ---------------------- | -----------: | --------: | ---------: | -------------: |
-| Bare                | 无 Harness             |    acc 1.33% |         — |        1.0 |              — |
-| Full v1             | `modules_full.yaml`    |        1.07% |     0.011 |        5.2 |            1.1 |
-| **Full v2**         | `modules_full_v2.yaml` |    **3.80%** | **0.181** |       33.7 |           26.5 |
-| **Minimal（历史）** | `modules_minimal.yaml` |    **2.45%** | **0.121** |       32.4 |           13.2 |
-
-产物：`artifacts/baselines/compare_phase0_full830.json` · `outputs/minimal_runtime_browsecomp_full830/`
-
-后续 Pareto 配置仍保留：Bare → Minimal Executor → Minimal + hard verifier/state store → Partially retired Harness → Full v2 → Trained + Minimal。正式比较时统一到当前 H_min_v2 协议，并补 matched Base baseline。
-
----
-
-### Fresh-corpus / Cross-Harness 泛化
-
-**状态：** 📋 P4。
-
-**Setting（计划）**
-
-| 项            | 值                                                           |
-| ------------- | ------------------------------------------------------------ |
-| Model         | `Qwen2.5-7B-Instruct`                                        |
-| Fresh Corpus  | BM25→dense · train corpus→new index · fixed source→new distribution |
-| Cross-Harness | JSON 字段顺序 · reason code · evidence renderer · context serialization 扰动 |
-
-
-## 全局待办（按优先级）
-
-> 原则：先证明“测得对”，再证明“学得到”，最后才扩大规模 / 多能力。  
-> Round 4/5 后：**measurement / observability / offline learnability（O7）已过**；当前最大风险是 **offline↔closed-loop 行为不一致**。
-
-```text
-[P0 ✅] Offline evaluator + scorer consistency（Round 4 B1/B2）
-[P0 ✅] DecisionState observability（Round 5 B1：无 label collision）
-[P0 ✅] Objective learnability offline（Round 5：O7 overfit D128 + valid bal_acc=1.0）
-
-[P0-NOW] O7 closed-loop 校准 / seed 一致性  ← 当前最紧迫
-  - 解释 seed42≈KEEP vs seed43/44 高 FSR
-  - 检查 runtime score 路径、决策阈值、状态分布相对 valid 的偏移
-  - 目标：相对 Base 改善 duplicate rejection，且 FSR 可控、reward 不崩
-
-[P1-A] Dup positive-signal gate（仍未过）
-  - 双侧不塌缩 + 3 seeds 一致 + DCR/FSR 改善 + task retention
-  - 通过后才允许 830q retention / E6
-
-[P1-B] E0 targeted probe 修复，而不是直接 830
-[P1-C] E1 local vs full-harness（后置于 Dup 闭环正信号）
-[P2+] weighting / Recovery / multi-capability / RL — 暂缓
-```
-
-**明确暂缓**
-
-- E0 830：当前 probe coverage 问题不能靠无差别扩大样本解决。
-- Round 1 LoRA 的 830 eval：已被 Round 2/3 诊断淘汰，不再作为主线。
-- 多能力联合 SDI / weighting：Dup 单能力尚未建立可信 positive signal。
-- Premature 训练：CONTINUE→STOP 监督覆盖仍为 0。
-- **830 / E1：** Round 5 `ROUND5_POSITIVE_SIGNAL=false`，按 barrier **禁止扩规模**。
-
-
-## 进度总览
-
-| 项                    | 状态       | 当前结论                                                     |
-| --------------------- | ---------- | ------------------------------------------------------------ |
-| Stage 0 / E0          | 🔄          | 100q 原始 probe 冻结；taxonomy 未冻结                        |
-| Round 1               | ✅ 历史完成 | 序列拟合成功，行为内化未成立                                 |
-| Round 2               | ✅ 诊断完成 | 发现 loss-mass / one-sided / action-interface 问题；旧 Wave4 被 R3 替代 |
-| Round 3               | ✅ 完成     | 双侧数据+typed runtime+100q CL 全完成；`ROUND3_POSITIVE_SIGNAL=false`；operation_ce 塌缩为 P0 |
-| Round 4               | ✅ 完成     | measurement/scorer 验证通过；`operation_ce` objective 未通过 overfit128；**不进入 B5** |
-| **Round 5**           | ✅ 完成     | Observability+O7 offline PASS；100q CL **无 positive signal**；`RECOMMEND_830=false` |
-| Step 1–5              | 🔄          | Dup 主链已通；Stop bilateral coverage 未通                   |
-| Step 6 Weighting      | ⏸          | 等 ≥2 个可信 capability                                      |
-| Step 7 Recovery       | ⏸          | 按 failure mass 决定                                         |
-| Step 8 Optimize       | 🔄          | R1–R5 checkpoints 已有；Dup 闭环 positive signal 仍未建立     |
-| Step 9 Lifecycle / E6 | 🔄          | Phase0 830 完成；trained H_min_v2 830 暂缓                   |
-| E1                    | 📋 P1       | local vs full-harness distillation 核心基线；**仍后置于 Dup 闭环正信号** |
-| E2                    | 🔄          | 双侧 controls + R5 O7 offline 成立；闭环行为未校准             |
-| E3–E5                 | ⏸/📋        | 后置                                                         |
-
-**当前一句话结论：** Round 5 证明 DecisionState 可观测且 **O7（discriminative_ce + LoRA r=64）offline 可完美过拟合/双侧分离**，但三 seeds 闭环行为不一致（seed42≈KEEP、seed43/44 高 FSR），**不构成 Dup positive internalization signal，禁止扩 830**。
-
----
-
-## Round 4 — Duplicate Measurement & Objective Repair（07-30）
-
-**Git：** `scope/dup-round4-objective-repair`（自 `scope/dup-round3-bilateral` @ `ad072b9`）
-
-**Gate 结论**
-
-| Flag | 值 | 含义 |
-| --- | --- | --- |
-| `ROUND4_MEASUREMENT_VALID` | **true** | B1 offline metrics + forced episode + DCR/FSR 原始计数 |
-| `ROUND4_SCORER_VALID` | **true** | B2 train/offline/runtime mismatch = 0%（8 models × 522 states） |
-| `ROUND4_OBJECTIVE_VALID` | **false** | B4 overfit128 未达 95% acc / 90% 双侧 recall |
-| `ROUND4_POSITIVE_SIGNAL` | **false** | 旧 operation_ce checkpoint 未恢复；compact_json 仅微弱改善 |
-| `RECOMMEND_830` | **false** | 按 todo：Dup positive signal 前不做 830 |
-
----
-
-### Barrier 1 — Measurement Audit ✅
-
-| 子任务 | 状态 | 产物 |
-| --- | --- | --- |
-| B1.1 标准二分类指标 + unit tests | ✅ 8/8 PASS | `training/scope/binary_operation_metrics.py` |
-| B1.2 DCR/FSR 原始计数 telemetry | ✅ | `training/scope/dup_telemetry.py` |
-| B1.3 Forced episode (20×2) | ✅ | `outputs/scope_round4/metric_audit/forced_episode.jsonl` |
-| B1.1 Offline re-eval（10 models × 522 valid） | ✅ | `outputs/scope_round4/metric_audit/offline_eval_fixed.json` |
-
-**Offline 重评（修复 metrics 后，522 valid）**
-
-| Variant | acc | macro_f1 | KEEP recall | SKIP recall |
-| --- | ---: | ---: | ---: | ---: |
-| Base / op_seed42/43/44 / endorse / no_balance | 0.810 | 0.448 | **1.000** | **0.000** |
-| **compact_json** | 0.803 | **0.481** | 0.981 | **0.040** |
-| correct_only | 0.190 | 0.159 | 0.000 | 1.000 |
-| Round2-main（B1 早批） | 0.103 | 0.167 | 0.000 | 0.545 |
-
----
-
-### Barrier 2 — Scorer Consistency ✅
-
-8 GPU replay（522 valid states × 8 models）：**train/offline/runtime/prompt mismatch rate 全部为 0%**。
-
-| 模型 | margin mean | 解读 |
-| --- | ---: | --- |
-| Base | -4.33 | 强偏 KEEP |
-| op_seed42/43/44 | -1.3 ~ -1.5 | 仍偏 KEEP |
-| correct_only | +5.72 | 强偏 SKIP |
-| compact_json | -2.74 | 偏 KEEP，幅度中等 |
-
-产物：`outputs/scope_round4/scorer_audit/SCORE_CONSISTENCY_REPORT.md`
-
-**结论：** Round3 operation_ce 塌缩**不是** scorer/prompt/train-infer 不一致导致。
-
----
-
-### Barrier 3 — Postfix Replay ✅（infra）/ ⚠️（行为）
-
-**Phase 1**（14:28–14:43，8 GPU 并行，修复 `CUDA_VISIBLE_DEVICES` 覆盖 bug 后）：8/8 offline JSON 写入 `outputs/scope_round4/postfix_replay/offline/`。
-
-**Phase 2**（14:47–~16:00，GPU0 B4 ∥ GPU1–4 closed-loop，75s 错峰）：**10/10 closed-loop shard 全部 `telemetry_complete: true`**，无 vLLM 初始化失败。
-
-| Variant | shard0 | shard1 |
-| --- | --- | --- |
-| base | ✅ | ✅ |
-| compact_json | ✅ | ✅ |
-| op_seed42/43/44 | ✅ | ✅ |
-
-旧 checkpoint 在 scorer 修复后**未恢复**合理双侧行为（op_ce 闭环仍高 FSR）。
-
----
-
-### Barrier 4 — Overfit128 ❌
-
-**Dataset：** `artifacts/datasets/dup_sdi_round4_overfit128/`（64 KEEP + 64 SKIP，76 unique queries，seed=42）
-
-**Training：** operation_ce · LoRA r=16 · lr=2e-5 · 10 epochs · class_balancing=true · GPU0 · ~20 min
-
-| 指标 | 训练前 | 训练后 | 目标 |
-| --- | ---: | ---: | --- |
-| train accuracy | 0.500 | **0.508** | >0.95 |
-| KEEP recall | 1.000 | 0.766 | >0.90 |
-| SKIP recall | 0.000 | **0.250** | >0.90 |
-| SKIP mean loss（probe） | 4.46 | 0.79 | — |
-| KEEP mean loss（probe） | 0.01 | 0.62 | — |
-| margin mean | -4.36 | -0.20 | 应分离两类 |
-
-`B4_PASS=false` · 产物：`outputs/scope_round4/overfit128/overfit128_report.json`
-
-**诊断：** SKIP 样本初始 loss 远高于 KEEP（~400×），梯度/优化被 KEEP 主导；10 epoch 后 margin 仍对两类均为负，objective 未能学到可靠 SKIP 边界。
-
----
-
-### 工程修复记录
-
-1. **GPU 分配 bug：** `run_postfix_offline_eval.py` 内 `os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu` 覆盖 shell 设置，导致 8 任务挤占 GPU0 → OOM；已删除覆盖逻辑。
-2. **B3 offline bug：** `load_jsonl` 需 `Path` 类型；改为独立 Python 脚本。
-3. **Closed-loop：** 每 GPU 单 shard + 75s 错峰，最多 4 并发 vLLM。
-
----
-
-### 代码与脚本（本分支）
-
-```
-training/scope/binary_operation_metrics.py
-training/scope/eval_dup_capability.py          # 接入统一 metrics
-training/scope/dup_telemetry.py                # DCR/FSR 原始计数
-training/scope_round4/                         # audit / replay / overfit
-tests/scope/test_binary_operation_metrics.py
-scripts/scope_round4/                          # barrier 1–4 nohup 编排
-artifacts/datasets/dup_sdi_round4_overfit128/  # 128-sample 平衡集
-```
-
----
-
-### 下一步（按 0730-todo1.md）→ 已进入 Round 5
-
-```text
-B4 FAIL → Round 5：Observability + objective tournament + O7 full screen + 100q CL
-        → 见下一节 Round 5（2026-07-30 完成）
-```
-
----
-
-## Round 5 — Operation Observability & Learnability（07-30）
-
-**Git：** `scope/dup-round5-learnability`（自 `scope/dup-round4-objective-repair` @ `6b4e88b`；Round5 脚本/训练代码当时多为**本地未提交**）
-
-**文档：** `0730-todo2.md`  
-**产物根：** `outputs/scope_round5/`  
-**报告：** `outputs/scope_round5/ROUND5_REPORT.md`  
-**环境快照：** `outputs/scope_round5/environment_snapshot.txt`
-
-**时间线：** 2026-07-30 16:43（B0）→ 20:46（B4/B5 marker）→ 22:47（B6 定向续跑完成）
-
----
-
-### Gate 结论
-
-| Flag | 值 | 含义 |
-| --- | --- | --- |
-| `B1_PASS` / Observability | **true** | effective-input 无 KEEP/SKIP 标签冲突；shadow agreement=100%；truncation=0% |
-| `B2_PASS` / Objective math | **true** | KEEP/SKIP one-step margin 方向正确；LoRA 有梯度与参数更新 |
-| `B3_PASSED_OBJECTIVES` | **O7 only** | 仅 O7 通过 D2→D8→D32→D128 cascade |
-| `B4_PASS` | **true** | O7×3 seeds valid 双侧 discrimination=1.0；Top-2=`o7_r64_seed44/43` |
-| `B5_COMPLETE` | marker only | **未实际跑 50q**（`closed_loop/b5_50q/` 为空；supervisor 7s 内写完 marker） |
-| `B6_COMPLETE` | **true** | Base + O7×3seeds + compact_json 各 100q 闭环完成 |
-| `ROUND5_OBSERVABILITY_VALID` | **true**（据 B1） | label = f(student-visible DecisionState) 成立 |
-| `ROUND5_OBJECTIVE_VALID` | **true**（offline） | O7 可 overfit 且 full-valid 双侧分离 |
-| `ROUND5_CLOSED_LOOP_POSITIVE` | **false** | 三 seeds 行为不一致；43/44 高 FSR；reward↓ vs Base |
-| `ROUND5_POSITIVE_SIGNAL` | **false** | 未同时满足 todo 六条正信号条件 |
-| `RECOMMEND_830` | **false** | 禁止扩 830 |
-
----
-
-### Setting（冻结）
-
-| 项 | 值 |
-| --- | --- |
-| Base model | `/data/ppnm/models/Qwen2.5-7B-Instruct` |
-| Runtime | \(H_{\min,\text{v2}}\) · `harness/configs/modules_minimal_v2.yaml` |
-| Train / Valid | Round3 `dup_sdi_round3` 1807 / 522（sha256 `a0168283…`） |
-| Overfit128 | Round4 `dup_sdi_round4_overfit128`（sha256 `ea31a1b9…`） |
-| 100q manifest | `artifacts/datasets/round2_audit_100q/query_manifest.json`（sha256 `47b12f76…`） |
-| CUDA / PyTorch / transformers / peft | 13.0 / 2.11.0+cu130 / 5.14.1 / 0.19.1 |
-| 闭环调度 | 最多 4 并发 vLLM；4×25 shard；wave 内 75s stagger |
-| 编排 | `scripts/scope_round5/pipeline_supervisor.sh`（B6 末段因 hang 改为 `targeted_b6_resume.sh`） |
-
-**Objective 定义（B3 tournament）**
-
-| ID | loss_mode | 其他 |
-| --- | --- | --- |
-| O0 | `operation_ce`（legacy） | LoRA r=16 |
-| O1 | `discriminative_ce` | r=16 |
-| O2 | `pairwise_margin` | r=16 |
-| O3 | `single_token` | r=16 |
-| O4 | `sample_normalized_action_ce` + compact_target | r=16 |
-| O5 | `discriminative_ce_sum` | r=16 |
-| O6 | `discriminative_ce_mean` | r=16 |
-| **O7** | **`discriminative_ce`** | **r=64, α=128** |
-
-嵌套 overfit：`D2 ⊂ D8 ⊂ D32 ⊂ D128`（平衡 KEEP/SKIP）；无 class/route balance；KL=0；cascade 失败即停。
-
-**B4 全量训练：** O7×seed{42,43,44} + compact_json×seed{42,43,44}；3 epochs；lr=2e-5；bs=4×accum=4；max_length=4096。
-
----
-
-### Barrier 0 — 环境冻结 ✅
-
-产物：`environment_snapshot.txt` · HEAD `6b4e88b` · branch `scope/dup-round5-learnability`
-
----
-
-### Barrier 1 — DecisionState Observability ✅
-
-对 overfit128 / train1807 / valid522 dump effective student input（DecisionState→renderer→chat template→tokenizer→truncation）。
-
-| 检查 | 结果 |
-| --- | --- |
-| unique effective inputs | 2327 |
-| exact collision groups | 130 |
-| **conflicting-label groups** | **0** |
-| serialized-state shadow agreement | **100%**（≥99% gate） |
-| truncation rate（KEEP/SKIP/overall） | **0%** |
-
-产物：`observability/effective_inputs*.jsonl` · `LABEL_COLLISION_REPORT.md` · `observability_report.json`
-
-**结论：** KEEP/SKIP label **可由 student-visible DecisionState 推导**；Round4 FAIL 不能归因于不可观测标签冲突。
-
----
-
-### Barrier 2 — Objective 数学与梯度 ✅
-
-| 探针 | loss_before→after | margin Δ | 方向 | LoRA grad / Δθ |
-| --- | ---: | ---: | --- | --- |
-| KEEP one-step | 0.0019→0.0019 | −6.28→−11.06（Δ−4.78） | ✅ 更偏 KEEP | 有更新 |
-| SKIP one-step | 5.5→5.5 | −5.52→−0.02（Δ+5.50） | ✅ 更偏 SKIP | grad≈17.8 |
-
-产物：`b2_objective/b2_report.json`
-
----
-
-### Barrier 3 — 8-GPU Micro-Overfit Tournament ✅（仅 O7）
-
-| Objective | D2 | D8 | D32 | D128 | All Pass |
-| --- | --- | --- | --- | --- | --- |
-| O0–O3, O5–O6 | ❌ acc=0.5（全 KEEP） | — | — | — | ❌ |
-| O4 | ❌ acc=0.0（PARSE_FAIL） | — | — | — | ❌ |
-| **O7** | ✅ 100% | ✅ 100% | ✅ 100% | ✅ 100% | ✅ |
-
-O7 D128 post：acc/macro-F1/bal-acc/KEEP&SKIP recall 全部 **1.0**；margin_KEEP≈−7.06，margin_SKIP≈+7.20。
-
-**关键对照：** O1 与 O7 **loss 相同**（`discriminative_ce`），仅 LoRA r=16→64；O1 卡在 D2，O7 贯通 D128。  
-→ Round4 overfit128 失败的主因包含 **adapter 容量不足**，不只是 loss 公式名。
-
-产物：`micro_overfit/MICRO_OVERFIT_MATRIX.md` · `micro_overfit/O7/`
-
----
-
-### Barrier 4 — Full 1807/522 Objective Screen ✅
-
-Valid=522（KEEP=423, SKIP=99）offline：
-
-| Variant | bal_acc | macro_f1 | KEEP recall | SKIP recall | gate | mean_m_KEEP | mean_m_SKIP |
-| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
-| o7_r64_seed42 | **1.000** | **1.000** | 1.000 | 1.000 | ✅ | −8.19 | +6.81 |
-| o7_r64_seed43 | **1.000** | **1.000** | 1.000 | 1.000 | ✅ | −7.09 | +6.50 |
-| o7_r64_seed44 | **1.000** | **1.000** | 1.000 | 1.000 | ✅ | −6.85 | +6.00 |
-| compact_json_seed42 | 0.511 | 0.491 | 0.962 | 0.061 | ✅ | −3.83 | −4.17 |
-| compact_json_seed43 | 0.499 | 0.447 | 0.998 | 0.000 | ❌ | −4.42 | −4.60 |
-| compact_json_seed44 | 0.518 | 0.492 | 0.986 | 0.051 | ✅ | −4.10 | −4.33 |
+### Setting
+- path: `/data/ppnm/Capability_Evolution/SCAPE/outputs/local_cal64_loo`
+- model: Qwen2.5-7B-Instruct (vLLM TP=1, CAL64 BM25 provisional)
+- n_queries: 64 unique / job; quality gate unique≥64 & err_rate≤0.15
+- jobs: full + 8 minus_* (9/9 quality-complete)
 
-**Top-2：** `o7_r64_seed44`, `o7_r64_seed43`（`B4_TOP2`）  
-产物：`b4_full/` · `B4_GATE.json` · merged HF：`merged/o7_r64_seed{42,43,44}` · `merged/compact_json_seed42`
+### Results
+| metric | value |
+|---|---:|
+| quality_complete | 9/9 |
+| Candidate A | auto_populate_first_search |
+| Candidate B | verify_tool |
+| placement_map | outputs/scape_prestage/CAPABILITY_PLACEMENT_MAP.md |
+| selection_json | outputs/scape_prestage/CANDIDATE_SELECTION.json |
 
----
+### Paired
+- LOO contribution from full vs minus_* CAL64 rollouts
+- influence values are provisional proxies pending real same-state influence probe
 
-### Barrier 5 — Top-2 × 50q ⚠️ 跳过
+### Gate
+PARTIAL — LOO aggregate done; Stage L scaffolding + dry_run distill started; real OPD data path not yet wired
 
-`B5_COMPLETE` 于 20:46:58 写入，但 `closed_loop/b5_50q/` **无任何 shard 产物**。pipeline 在数秒内进入 B6。  
-**记录为 infra 捷径/缺陷，不作为 50q 行为证据。** 行为结论以 B6 100q 为准。
+### Decision
+Proceed Stage L learnability for A=`auto_populate_first_search`, B=`verify_tool`. Prefer waiting was satisfied (9/9). Next: wire real reduced-harness same-state collection → tool-OPD training cells.
 
 ---
 
-### Barrier 6 — Best Objective × 3 Seeds × 100q ✅（无正信号）
-
-比较：`Base` · `best_o7_{42,43,44}` · `compact_json`（seed42）；各 4×25 shard，`--dup-operation`。
+## 2026-08-11 Stage L B-verify provisional OPD L64_seed42
 
-**运维备注：** Best-44/shard1 曾卡在 `qid=535` `grep_corpus` ~26min 无推进；kill 后 `--resume` 定向续跑（23→25），再跑 compact_json；22:47 全完成。shard1 的 `summary.json` 仅统计续跑 2 条，下表指标改从 **全部 `dup_admission_events.jsonl` + `episodes.jsonl` 重聚合**。
-
-| Variant | n_ep | DCR ↓ | FSR ↓ | SKIP recall | KEEP recall | n_pred_SKIP / n_events | mean reward | mean recall |
-| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
-| **Base** | 100 | **0.000** | **0.000** | 0.000 | 1.000 | 0 / 16694 | **0.203** | **0.030** |
-| O7-seed42 | 100 | 0.000 | 0.000 | 0.005 | 1.000 | 17 / 17055 | 0.134 | 0.023 |
-| O7-seed43 | 100 | 0.010 | **0.517** | 0.289 | 0.483 | 7093 / 14602 | 0.157 | 0.024 |
-| O7-seed44 | 100 | 0.045 | **0.366** | 0.232 | 0.634 | 5489 / 15958 | 0.144 | 0.017 |
-| compact_json | 100 | 0.000 | 0.000 | 0.000 | 1.000 | 0 / 12218 | 0.136 | 0.029 |
+### Setting
+- path: `outputs/stage_l/B_verify_opd_provisional/`
+- stack: SCOPE `smoke_opd_vllm_hf` + `train_opd` (provisional LOCAL_CAL64; H100 not required)
+- GPUs: 2–5 (TP=4 vLLM rollout → HF train)
+- cell: L64 seed=42 · target_module=verification · student=ablate_verification · teacher=modules_full · epochs=1
 
-（DCR=`duplicate_curate_rate`；FSR=`false_skip_rate` / unique reject；与 Round3 Base DCR=0 定义一致——KEEP 路径下 duplicate 的 `actually_curated` 可为 false。）
+### Results
+| metric | value |
+|---|---:|
+| smoke DONE | yes |
+| smoke opd_loss | 0.0486 |
+| L64 n_transitions | 64 |
+| L64 epoch0 loss | 0.1220 |
+| L64 opd_loss | 0.7293 |
+| checkpoint | `L64_seed42/checkpoint.json` status=saved |
 
-**闭环解读**
+### Paired
+- （当时）Collect A/B H_-m 未完成 → **后续已于 2026-08-12 完成 512**（见本轮总览）
+- Next cell: L64_seed43 started on freed GPU2–5
 
-1. **Offline≠Closed-loop：** O7 三 seeds valid 上 SKIP recall=1.0，但闭环 seed42 几乎不发 SKIP；43/44 大量 SKIP 且误伤 unique（FSR 37–52%）。
-2. **三 seeds 方向不一致：** 不满足 “3 seeds 方向一致”。
-3. **相对 Base：** 无稳定的 “duplicate rejection 改善 + unique rejection 可控”；reward 全面低于 Base。
-4. **compact_json：** 行为仍接近 Base（全 KEEP admission），offline 仅微弱 SKIP。
+### Gate
+PARTIAL（条目当时）→ **已被本轮总览覆盖**：Gate L 后续 PASS；collect 已完成
 
-产物：`closed_loop/b6_100q/{base,best_o7_42,best_o7_43,best_o7_44,compact_json}/`
+### Decision
+Record L64_seed42 metrics; advance seed43 on free GPUs. Do not stop for empty H100 imports.
 
 ---
 
-### Round 5 Positive Signal 判定（对照 0730-todo2）
-
-| # | 条件 | 结果 |
-| --- | --- | --- |
-| 1 | D128 能稳定 overfit | ✅ O7 |
-| 2 | full valid 双侧 discrimination | ✅ O7×3 seeds bal_acc=1.0 |
-| 3 | 3 seeds 方向一致 | ❌ 闭环 42 vs 43/44 分裂 |
-| 4 | duplicate rejection 相对 Base 改善 | ❌ 无稳定改善（DCR 未优于 Base 叙事） |
-| 5 | unique rejection 可控 | ❌ seed43/44 FSR 过高 |
-| 6 | recall / reward 无系统性下降 | ❌ reward 相对 Base 下降 |
+## 2026-08-11 Stage L B-verify provisional OPD L200
 
-→ **`ROUND5_POSITIVE_SIGNAL=false` · `RECOMMEND_830=false`**
+### Setting
+- path: `outputs/stage_l/B_verify_opd_provisional/`
+- stack: SCOPE `train_opd` (provisional LOCAL_CAL64; H100 not required)
+- cells: L200 seed42 (GPU2–5 TP4 :8769); L200 seed43 (GPU6–7 TP2 :8770)
+- target_module=verification · student=ablate_verification · teacher=modules_full · epochs=1
 
----
+### Results
+| metric | seed42 | seed43 |
+|---|---:|---:|
+| n_transitions | 200 | 200 |
+| epoch0 loss | 0.1220 | 0.1296 |
+| opd_loss | 0.7293 | 0.8308 |
+| checkpoint | saved | saved |
+| status | DONE | DONE |
 
-### 工程与事故记录
+### Paired
+- Prior L64: s42 loss=0.122 / opd=0.729; s43 loss=0.119 / opd=0.860; s44 loss=0.130 / opd=0.988
+- （当时）A/B collect → **后续已完成**（见本轮总览）
 
-1. **B5 空跑：** marker 写入但无 50q 产物；后续勿把 B5 当证据。
-2. **B6 Best-44/shard1 hang：** `grep_corpus` 卡住；kill + `--resume` 定向续跑成功。
-3. **kill 竞态：** 首次 kill 曾误写 `B6_COMPLETE` 并跳过未完成 shard；已清 marker 后用 `logs/targeted_b6_resume.sh` 重跑剩余任务。
-4. **代码入库状态：** `scripts/scope_round5/`、`training/scope_round5/`、`training/scope/operation_objectives.py` 等在记录时仍为 untracked/modified，复现需保留本地工作树或另行 commit。
+### Gate
+PARTIAL（条目当时）→ **已被本轮总览覆盖**：held-out×2 + L200×3 已完成；Gate L PASS；closed-loop Gate S FAIL
 
----
+### Decision
+Record L200 seed42/43; free GPU2–7; start B L64 held-out (`--split test`) while collect continues.
 
-### 代码与脚本（本轮）
 
-```
-training/scope/operation_objectives.py
-training/scope_round5/          # B1–B4/B6 helpers + build_round5_report
-scripts/scope_round5/           # pipeline_supervisor / run_b3–b6 / resume
-tests/scope/test_operation_objectives.py
-outputs/scope_round5/           # 全部 barrier 产物（只读历史 R1–R4）
-```
+## 2026-08-12 SCAPE non-H100 round final
 
----
----
+> 覆盖并取代同日自动追加的 `non_h100_closed_loop_complete` / `non_h100_completion` 草稿（其中仍写 collect 进行中 / S2S3 proxy 的条目作废）。
+> 状态：**本轮非 H100 主线已完成**；Stage M **不启动**。
 
-## Round 6 — Closed-loop Calibration & On-Policy Shift Audit（07-31 ~ 08-01）
-
-**Git：** `scope/dup-round6-closedloop-calibration` @ `61f1348c9ac32c4b89dc0db4f1ba087a3c239539`
-
-**文档：** `0731-todo1.md`
-**产物根：** `outputs/scope_round6/`
-**报告：** `outputs/scope_round6/ROUND6_REPORT.md`
-**记录更新时间：** 2026-08-01 12:22 CST
-
-### Gate 结论
-
-| Flag | 值 |
-| --- | --- |
-| `H_RUNTIME` | **False** |
-| `H_CALIB` | **False** |
-| `H_SHIFT` | **False** |
-| `H_FEEDBACK` | **False** |
-| adapter↔merged parity | 1.0 |
-| HF↔runtime parity | 1.0 |
-| `ROUND6_CLOSED_LOOP_POSITIVE` | **false** |
-| `RECOMMEND_830` | **false** |
-
-### Setting（冻结）
-
-| 项 | 值 |
-| --- | --- |
-| Base model | `Qwen2.5-7B-Instruct` |
-| O7 checkpoint | `outputs/scope_round5/merged/o7_r64_seed{42,43,44}` |
-| Loss / LoRA | `discriminative_ce` · r=64 · α=128（与 Round5 O7 相同） |
-| Runtime | \(H_{\min,\text{v2}}\) · `modules_minimal_v2.yaml` |
-| 100q manifest | `round2_audit_100q/query_manifest.json` |
-| Closed-loop | max_turns=35 · max_tokens=2048 · temperature=1.0 · BM25 |
-| Calibration slice | shard0（25q）closed-loop states |
-| Prospective 25q | shard1（C-CALIB） |
-| Holdout 50q | shard2+shard3（Phase D） |
-| τ_seed42 / 43 / 44 | -5.177083333333334 / -5.166666666666666 / -3.90625 |
-| τ_shared | -4.75 |
-| Decision rule | SKIP iff margin ≥ τ（`score_skip - score_keep`） |
-
-### Phase B — Cross-score 核心结论
-
-同一 checkpoint × 多 state source 离线重打分（merged HF scorer）：
-
-- valid522 与全部 B6 admission states 上 **AUROC=1.0**（三 seeds 一致）
-- 同一 states 上 **BalancedAcc@threshold=0 亦为 1.0**（offline 排序完美）
-- **H_RUNTIME / H_SHIFT / H_CALIB / H_FEEDBACK 均为 false**
-- 推论：Round5 闭环失败**不是** runtime parity 或 on-policy AUROC 崩塌；问题在 **closed-loop 决策边界 / 行为层**（校准后仍高 FSR）
-
-产物：`phase_b/CROSS_SCORE_MATRIX.csv` · `ROOT_CAUSE_GATE.json` · `STATE_SHIFT_REPORT.md`
-
-### Phase C-CALIB — shard1 25q（校准后前瞻）
-
-| Run | DupRejectRecall | FSR | BalancedAcc | mean_reward |
-| --- | ---: | ---: | ---: | ---: |
-| per_seed/seed42 | 1.000 | 1.000 | 0.500 | 0.354 |
-| per_seed/seed43 | 1.000 | 1.000 | 0.500 | 0.295 |
-| per_seed/seed44 | 1.000 | 0.994 | 0.503 | 0.265 |
-| shared/seed42 | 1.000 | 0.970 | 0.515 | 0.147 |
-| shared/seed43 | 1.000 | 1.000 | 0.500 | 0.295 |
-| shared/seed44 | 1.000 | 1.000 | 0.500 | 0.298 |
-| threshold_zero/seed43 | 0.000 | 0.000 | 0.500 | 0.227 |
-
-**解读：** per-seed τ 在 shard0 上可达 FSR≤5%；但 shard1 闭环中 O7 仍 **几乎全部 pred SKIP**（DupRejectRecall≈1 但 FSR≈1），校准 **未** 转化为可接受闭环行为。
-
-### Phase D — Holdout 50q（shard2+shard3）
-
-| Run | n_ep | DupRejectRecall | FSR | BalancedAcc | SKIP prior | reward | recall |
-| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
-| base/shard2 | 25 | 0.000 | 0.000 | 0.500 | 0.000 | 0.160 | 0.026 |
-| base/shard3 | 25 | 0.000 | 0.000 | 0.500 | 0.000 | 0.041 | 0.018 |
-| seed42/shard2 | 25 | 1.000 | 1.000 | 0.500 | 1.000 | 0.259 | 0.074 |
-| seed42/shard3 | 25 | 1.000 | 1.000 | 0.500 | 1.000 | 0.133 | 0.062 |
-| seed43/shard2 | 25 | 1.000 | 1.000 | 0.500 | 1.000 | 0.356 | 0.082 |
-| seed43/shard3 | 25 | 1.000 | 1.000 | 0.500 | 1.000 | 0.157 | 0.041 |
-| seed44/shard2 | 25 | 1.000 | 0.996 | 0.502 | 0.996 | 0.264 | 0.064 |
-| seed44/shard3 | 25 | 1.000 | 1.000 | 0.500 | 1.000 | 0.127 | 0.012 |
-
-**解读：**
-
-1. **Base：** DupRejectRecall=0（从不 SKIP），FSR=0；与 Round5 一致。
-2. **O7 + per-seed τ：** 校准后闭环仍 **SKIP 先验≈1.0**，FSR≈0.97–1.0；DupRejectRecall 高但来自 **误伤 unique**，非成功 duplicate internalization。
-3. **任务保持失败：** mean_reward 系统性低于 Base（~0.04–0.26 vs Base ~0.16/0.04 on holdout shards）。
-4. **Round6 正信号 gate 未过：** 要求 DupRejectRecall≥0.10 且 FSR≤0.05 且 BalancedAcc>0.50 — O7 满足前者但 **FSR 严重超标**。
-
-### Round 6 最终判定
-
-```text
-ROUND6_CLOSED_LOOP_POSITIVE = false
-RECOMMEND_830 = false
-C-SHIFT (Dagg retrain) = 未触发（H_SHIFT=false）
-```
-
-### 工程备注
-
-1. Phase D 首次运行 `get_tau()` JSON key bug（int vs str）导致 O7 holdout 未启动；已修复并用 `resume_holdout_o7.sh` 补跑。
-2. `seed43/shard2` 曾在 query 335 卡住 9/25；kill 后 `--resume` 续跑剩余 16 题。
-3. 所有闭环指标从 `episodes.jsonl` + `dup_admission_events.jsonl` 重聚合。
-
-### 代码与脚本
-
-```text
-training/scope/decision_config.py
-training/scope_round6/
-scripts/scope_round6/
-tests/scope/test_round6_scorer.py
-outputs/scope_round6/
-```
-
-### 下一步
-
-```text
-RECOMMEND_830=false → 禁止扩 830 / E1 / weighting / multi-capability
-P0 转向：为何 offline margin 完美 + τ 校准后 closed-loop 仍全 SKIP？
-  → runtime vLLM scorer vs HF 在 live admission 路径是否仍一致
-  → τ 在 offline replay margin 上有效但对 live score scale 无效
-  → 考虑 on-policy Dagg 前需先修 live decision 路径或 score telemetry 对齐
-```
+### Setting
+- repo: `/data/ppnm/Capability_Evolution/SCAPE`
+- model: Qwen2.5-7B-Instruct（vLLM serve / HF train）
+- retrieval: BM25 provisional（BrowseComp-Plus index）；**非**官方 Harness-1 Chroma
+- benchmark: BrowseComp-Plus
+- H100: unavailable — 全程不依赖 `imports/h100_*`
+- Candidate A: `auto_populate_first_search` · OPD `target_module=evidence_state` · student=`ablate_auto_seed.yaml`
+- Candidate B: `verify_tool` · OPD `target_module=verification` · student=`ablate_verification.yaml`
+- teacher harness: `modules_full.yaml` / LOO full V8D mask
+- Stage S eval: CAL64 `split=test` n=64；S0/S1=LOO；S2/S3=served `L64_seed42_hf/hf_model`
+- H_-m collect: `split=train` limit=512 · mask 去掉对应组件
+- output roots:
+  - LOO: `outputs/local_cal64_loo/`
+  - collect: `outputs/stage_l_hminus_data/`
+  - B OPD/Gate L: `outputs/stage_l/B_verify_opd_provisional/` + `GATE_L_B.json`
+  - A OPD: `outputs/stage_l/A_auto_opd_provisional/`
+  - B four-grid: `outputs/stage_s/B_verify_fourgrid/`
+  - A four-grid: `outputs/stage_s/A_auto_fourgrid/`
+  - narrative: `outputs/NON_H100_FINAL_REPORT.md`
+
+### Results
+
+#### 状态汇总
+| item | status | note |
+|---|---|---|
+| LOO 9/9 | **已完成** | quality-complete |
+| Candidate select A/B | **已完成** | A score≈0.0072；B score≈0.0011 |
+| B Gate L | **已完成 · PASS** | provisional SCOPE-OPD；非 full tool-token |
+| A/B HF student ckpt | **已完成** | 可 vLLM 服务 |
+| A/B H_-m collect 512 | **已完成** | A 512 uniq；B 512 uniq（834 lines w/ resume dups） |
+| B Gate S closed-loop | **已完成 · FAIL** | CCR_m≈0.855；不可 retirement |
+| A Gate S closed-loop | **已完成 · FAIL** | CCR_m≈0.536；不可 retirement |
+| Stage M / Pareto | **未开始** | 按 auto-stop 规则停止 |
+| H100 / Chroma 官方线 | **阻塞 / 未开始** | 本轮不依赖 |
+
+#### B = verify_tool（closed-loop 四格，n_shared=64）
+| cell | J (curated_recall) | C (tool-call proxy) | source |
+|---|---:|---:|---|
+| S0 θ0+H_full | 0.0372 | 34.24 | LOO |
+| S1 θ0+H_-verify | 0.0275 | 32.95 | LOO |
+| S2 θ'+H_-verify | 0.0358 | 34.98 | closed-loop HF |
+| S3 θ'+H_full | 0.0429 | 34.23 | closed-loop HF |
+
+| metric | value |
+|---|---:|
+| CCR_m | 0.855 |
+| HRR | 0.152 |
+| Gate S verdict | **FAIL** |
+| can_claim_retired | false |
+
+#### A = auto_populate_first_search（closed-loop 四格，n_shared=64）
+| cell | J | C | source |
+|---|---:|---:|---|
+| S0 θ0+H_full | 0.0372 | 34.24 | LOO |
+| S1 θ0+H_-auto | 0.0084 | 34.16 | LOO |
+| S2 θ'+H_-auto | 0.0238 | 34.67 | closed-loop HF |
+| S3 θ'+H_full | 0.0429 | 34.23 | closed-loop HF |
+
+| metric | value |
+|---|---:|
+| CCR_m | 0.536 |
+| HRR | 0.152 |
+| Gate S verdict | **FAIL** |
+| can_claim_retired | false |
+
+#### Stage L（B，摘录）
+| cell | status |
+|---|---|
+| L64 seeds 42/43/44 + L64_seed42_hf | DONE |
+| L200 seeds 42/43/44 | DONE |
+| L64 heldout seeds 42/43 | DONE |
+| L200_seed45 / L512_seed42 | **未跑**（Gate S 已 FAIL，不再扩） |
+
+### Paired
+- S0/S1：同一 CAL64 query 集上 full vs minus 组件的 LOO paired quality
+- S2/S3：同一 query 集上 θ'（OPD HF）vs θ0 的 closed-loop paired 评测
+- B：去掉 verify 后缺口约 0.0097 J，OPD 恢复约 85%（CCR），但仍低于 S0，且 C 未实质下降
+- A：去掉 auto_populate 后缺口更大；OPD 仅恢复约 54%，远未非劣于 S0
+
+### Gate
+- Gate L (B): **PASS**（provisional）
+- Gate S (B): **FAIL**
+- Gate S (A): **FAIL**
+- Stage M: **不进入**
+- Overall round: **COMPLETED (non-H100 line)** / retirement claim: **REJECTED**
+
+### Decision
+停止对 A/B 的 retirement 宣称与 Stage M；本轮 provisional 线归档。下一步只做其一：**(1)** 等 H100/官方 Chroma 后重跑 LOO→Gate；或 **(2)** 换下一候选组件（遵守「连续两失败则停救」）。不在当前 BM25+Qwen 线上继续扩 seed / multi-component。
+
+---
+
+## 2026-08-12 SCAPE H100-1/2/3 synced status
+
+> 自 `result-record-from-h100.md` 同步的实验 setting / 结果 / 结论。
+> 路径以 H100 机为准：`/mnt/songzijun/Capability_Evolution/SCAPE`（对应本机 `/data/ppnm/Capability_Evolution/SCAPE` 同源树）。
+> 状态词汇：**已完成** / **进行中/阻塞** / **未开始**。
+
+### Overall status
+| Workstream | Todo target | Current status | Output / evidence | Notes |
+|---|---|---|---|---|
+| H100-1 Phase 0/1 | Harness-1 reproduction + 10-component LOO contribution map | **已完成（local BM25 compat）/ 进行中（official Chroma parity）** | `outputs/h100_1_contribution/{RUN_MANIFEST.json,STATUS_LIVE.md,COMPONENT_CONTRIBUTION.*,SHA256SUMS}` | Local BM25 compatibility contribution sweep finished for all 10 components, n=200, errors=0. Official Chroma Cloud eval blocked by missing credentials. |
+| H100-2 | independent replication + coalition interaction | **已完成（frozen consolidation）/ 部分偏离原 10-component REPL200 plan** | `outputs/h100_2_replication_coalition/{RUN_MANIFEST.json,STATUS_LIVE.md,LOO_REPLICATION.csv,COALITION_INTERACTION.csv,REPLICATION_REPORT.md,PLACEMENT_STABILITY.md,SHA256SUMS}` | 4 replicated modules + 6 coalition rows, errors=0. No new training/retrieval. |
+| H100-3 | same-environment-state policy influence map | **已完成（offline deterministic INF_CAL64）** | `outputs/h100_3_influence/{RUN_MANIFEST.json,STATUS_LIVE.md,INFLUENCE_BY_COMPONENT.*,INFLUENCE_PER_STATE.jsonl,H100_3_INFLUENCE_REPORT.md,SHA256SUMS}` | 64 queries × 4 states/query = 256 states/component; deterministic offline scorer. |
+| H100-1 × H100-3 | contribution/influence quadrant map | **已完成** | `outputs/CONTRIBUTION_INFLUENCE_MAP.md` | 10 components → four quadrants. |
+| Official Harness-1 serving | restore model and local vLLM smoke | **已完成（smoke）/ 进行中（official eval）** | `outputs/h100_1_official_vllm` | Restored from `harness-1.tar.gz`, 9 shards, vLLM smoke passed. |
+| H100-3 confirm/targeted | `INF_CONFIRM128`, targeted influence/mining | **未开始** | none | Optional follow-ups not launched. |
+| H100-1/2 official parity | Chroma-backed BrowseComp+ LOO/replication | **未开始/阻塞** | none beyond local/proxy | Requires official retrieval credentials. |
+
+### H100-1 setting
+- Run id: `h100_1_local_bm25_contribution_20260811`
+- Repo: `/mnt/songzijun/Capability_Evolution/SCAPE`（git `61f7741a…` dirty at manifest）
+- Env: `/opt/bishop-harness/bin/python`；Python 3.11.6；torch 2.11.0+cu130；vLLM 0.25.1；8×H100
+- Backend: `local_bm25_compat`（**非**官方 Chroma Cloud）
+- Split/seed: BrowseComp+ CAL200，seed 1101；smoke 1/5/20 亦 errors=0
+- Decode: deterministic compatibility；无训练 / 无改权重
+- Status: `n_expected=10`，`n_finished=10`，`remaining=0`，`errors=0`
+
+### H100-1 results
+| component | n | Δ curated | Δ trajectory | Δ final | Δ reward | Status |
+|---|---:|---:|---:|---:|---:|---|
+| subtractive_curation | 200 | +0.001556 | +0.000000 | +0.000000 | +0.000700 | 已完成 |
+| importance_tagging | 200 | +0.001000 | +0.000000 | +0.000000 | +0.000450 | 已完成 |
+| auto_populate_first_search | 200 | +0.000000 | +0.010298 | +0.000000 | +0.004634 | 已完成 |
+| evidence_graph | 200 | +0.000000 | +0.001667 | +0.000000 | +0.000750 | 已完成 |
+| sentence_compress | 200 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | 已完成 |
+| chunk_neighbors | 200 | +0.000000 | +0.001667 | +0.000000 | +0.000750 | 已完成 |
+| content_dedup | 200 | +0.000833 | +0.004583 | +0.000000 | +0.002438 | 已完成 |
+| verify_tool | 200 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | 已完成 |
+| token_budget_marker | 200 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | 已完成 |
+| adaptive_rerank_instruction | 200 | -0.001250 | +0.000000 | +0.002917 | -0.000271 | 已完成 |
+
+#### H100-1 conclusion
+- LOO 仅对 **local BM25 compatibility** 路径完成。
+- 综合 `Δ curated + Δ trajectory + Δ final` 最强：`auto_populate_first_search` → `content_dedup` → `adaptive_rerank_instruction` / `evidence_graph` / `chunk_neighbors`。
+- `sentence_compress`、`verify_tool`、`token_budget_marker` 在本地质量指标上中性。
+- **不可**用本 run 宣称官方 Harness-1 reproduction/parity。
+
+### H100-2 setting
+- Run id: `h100_2_replication_coalition_20260811`
+- Env: `/opt/vllm-qwen3-1.7b/bin/python`；Python 3.12.13；torch 2.11.0+cu130；vLLM 0.25.1；8×H100
+- Replication input: `SCOPE/outputs/h100_2_module_utility`（fresh200 module-utility）
+- Coalition input: `SCOPE/outputs/h100_2_exact_budget_factorial`（exact-budget factorial）
+- Seed/decode: seed 42；temperature=0, top_p=1, do_sample=false
+- Status: `n_expected=5`，`n_finished=5`，`errors=0`
+
+### H100-2 results — replication
+| module | ablated condition | n | Δ final-answer recall | Δ trajectory recall | Δ reward | paired final W/L/T | paired trajectory W/L/T | Status |
+|---|---|---:|---:|---:|---:|---|---|---|
+| context_budget | minus_context_budget | 200 | +0.003345 | -0.000671 | +0.022175 | 8/4/188 | 28/28/144 | 已完成 / REPLICATED |
+| evidence_state | minus_evidence_state | 200 | -0.001786 | +0.002148 | +0.014134 | 8/5/187 | 27/27/146 | 已完成 / REPLICATED |
+| verification | minus_verification | 200 | +0.010575 | +0.016813 | +0.054930 | 13/6/181 | 33/23/144 | 已完成 / REPLICATED |
+| retrieval_rerank | minus_retrieval_rerank | 200 | -0.005571 | -0.007124 | -0.008528 | 6/6/188 | 25/30/145 | 已完成 / REPLICATED |
+
+### H100-2 results — coalition
+| model | budget | N | Q | QS | sequential interaction gap | interpretation | Status |
+|---|---:|---:|---:|---:|---:|---|---|
+| qwen3_1p7b | 256 | 0.0300 | 0.0300 | 0.0200 | -0.0100 | diminishing_returns | 已完成 |
+| qwen3_1p7b | 512 | 0.0300 | 0.0500 | 0.0400 | -0.0300 | diminishing_returns | 已完成 |
+| qwen3_1p7b | 1024 | 0.0400 | 0.0400 | 0.0400 | +0.0000 | near_additive | 已完成 |
+| qwen3_30b | 256 | 0.0100 | 0.0000 | 0.0000 | +0.0100 | super_additive | 已完成 |
+| qwen3_30b | 512 | 0.0200 | 0.0200 | 0.0000 | -0.0200 | diminishing_returns | 已完成 |
+| qwen3_30b | 1024 | 0.0300 | 0.0100 | 0.0100 | +0.0200 | super_additive | 已完成 |
+
+#### H100-2 conclusion
+- `verification` 是最清晰的稳定正复现模块（final / trajectory / reward 皆正）。
+- `context_budget`、`evidence_state` 跨轴符号不一致 → placement/domain-sensitive。
+- `retrieval_rerank` 两路 recall 皆负 → interaction/benchmark-sensitive。
+- Coalition 多为 diminishing/near-additive，仅作交互备注，非强协同证据。
+- 本 run ≠ 原 H100-2 10-component REPL200 全量计划；是 frozen SCOPE 输出的 consolidation。
+
+### H100-3 setting
+- Run id: `h100_3_influence_offline_cal64`
+- Env: `/root/miniforge3/bin/python`；Python 3.13.13；offline scorer（无 torch/vLLM 依赖）
+- Scale: INF_CAL64；64 queries/component；max 4 states/query；256 states/component；共 2560 per-state records
+- Scorer: `deterministic_offline_stub`；无训练
+- Status: `n_expected=10`，`n_finished=10`，`errors=0`
+- H100 侧 A/B 候选：`subtractive_curation` / `importance_tagging`（与非 H100 线 A/B 不同）
+
+### H100-3 results
+| component | n_queries | n_states | event_support | normalized influence | Status |
+|---|---:|---:|---:|---:|---|
+| subtractive_curation | 64 | 256 | 256 | 0.134885 | 已完成 |
+| importance_tagging | 64 | 256 | 256 | 0.107081 | 已完成 |
+| verify_tool | 64 | 256 | 256 | 0.010138 | 已完成 |
+| chunk_neighbors | 64 | 256 | 256 | 0.009933 | 已完成 |
+| evidence_graph | 64 | 256 | 256 | 0.007756 | 已完成 |
+| content_dedup | 64 | 256 | 256 | 0.007324 | 已完成 |
+| auto_populate_first_search | 64 | 256 | 256 | 0.005417 | 已完成 |
+| token_budget_marker | 64 | 256 | 256 | 0.005255 | 已完成 |
+| sentence_compress | 64 | 256 | 256 | 0.003571 | 已完成 |
+| adaptive_rerank_instruction | 64 | 256 | 256 | 0.001980 | 已完成 |
+
+#### H100-3 conclusion
+- 最高 same-state influence：`subtractive_curation`、`importance_tagging`。
+- 中档：`verify_tool`、`chunk_neighbors`、`evidence_graph`、`content_dedup`。
+- 最低：`adaptive_rerank_instruction`、`sentence_compress`、`token_budget_marker`。
+- 本图有效为 offline deterministic same-state 产物；**不是** released Harness-1 logprob 枚举。
+- `INF_CONFIRM128` / targeted 扩展 **未开始**。
+
+### Cross-map conclusions（H100-1 + H100-3）
+- Source: `outputs/CONTRIBUTION_INFLUENCE_MAP.md`
+- Thresholds: contribution median `0.001611`；influence median `0.007540`
+
+| quadrant | components | conclusion | Status |
+|---|---|---|---|
+| High Δ, High I | `evidence_graph`, `chunk_neighbors` | 冻结 local/offline 证据下最强平衡迁移候选 | 已完成 |
+| High Δ, Low I | `auto_populate_first_search`, `content_dedup`, `adaptive_rerank_instruction` | 质量/运行时效应清晰，same-state 策略位移弱 | 已完成 |
+| Low Δ, High I | `subtractive_curation`, `importance_tagging`, `verify_tool` | 改策略但本地质量提升弱；保留/移除前需复核 | 已完成 |
+| Low Δ, Low I | `sentence_compress`, `token_budget_marker` | 本分析下的直接移除候选 | 已完成 |
+
+### H100 lightweight / proxy 附注（同源记录）
+| item | gate / key metric | Decision |
+|---|---|---|
+| H20 lightweight torch L/S/M/Pareto | PASS / LIGHTWEIGHT_TORCH_COMPLETE；best L_m≈0.962；S2 quality≈0.030 | 可作为 lightweight 产物；**非** official checkpoint retirement |
+| qrel-backed pre-stage + H20 torch | PASS / LIGHTWEIGHT_TORCH_PROXY_COMPLETE；A/B Gate L PASS | 同上；官方 Chroma 评测仍独立 |
+| Official model restore + vLLM smoke | PASS / MODEL_RESTORED_AND_VLLM_SMOKE_COMPLETE | 可继续接官方 eval；缺 3 个 secret vars |
+
+### Final decision / next actions（H100）
+- **已完成**：H100-1 local BM25 contribution；H100-2 frozen replication/coalition + placement；H100-3 offline influence；贡献×影响力四象限；Harness-1 restore + vLLM smoke；lightweight torch proxy L/S/M。
+- **进行中/阻塞**：官方 BrowseComp+（缺 Chroma/OpenAI 凭证）。
+- **未开始**：官方 Chroma H100-1/2 parity；`INF_CONFIRM128`；targeted influence/mining；released-checkpoint retirement 宣称。
+- **禁止宣称**：不可把 local BM25 / offline / proxy 证据写成官方 Harness-1 Cloud/Chroma parity 或最终 retirement。
diff --git a/SCOPE/scripts/rollout_bare_browsecomp_4gpu.sh b/SCOPE/scripts/rollout_bare_browsecomp_4gpu.sh
index 426afeb..70a4445 100755
--- a/SCOPE/scripts/rollout_bare_browsecomp_4gpu.sh
+++ b/SCOPE/scripts/rollout_bare_browsecomp_4gpu.sh
@@ -1,25 +1,25 @@
 #!/usr/bin/env bash
-# Full BrowseComp+ bare rollout on GPUs 4-7: tau ~ pi_theta(x), no Harness.
+# Full BrowseComp+ bare rollout on GPUs 0-3: tau ~ pi_theta(x), no Harness.
 set -euo pipefail
 
 REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
-CONDA_BASE="${CONDA_BASE:-/data/ppnm/miniconda3}"
-ENV_NAME="${BISHOP_CONDA_ENV:-bishop}"
-MODEL_PATH="${MODEL_PATH:-/data/ppnm/models/Qwen2.5-7B-Instruct}"
+PYTHON_BIN="${PYTHON_BIN:-/opt/vllm-qwen3-1.7b/bin/python}"
+VLLM_BIN_DIR="${VLLM_BIN_DIR:-/opt/vllm-qwen3-1.7b/bin}"
+MODEL_PATH="${MODEL_PATH:-/mnt/songzijun/models/Qwen3-1.7B}"
 OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/bare_rollout_browsecomp_full}"
 SPLIT="${SPLIT:-all}"
 LIMIT="${LIMIT:-0}"
 MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
 TEMPERATURE="${TEMPERATURE:-1.0}"
+SEED="${SEED:-42}"
 MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
 VLLM_PORT="${VLLM_PORT:-8770}"
 RESUME="${RESUME:-1}"
 
-source "${CONDA_BASE}/etc/profile.d/conda.sh"
-conda activate "${ENV_NAME}"
-
-export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
-export PYTHONPATH="${REPO_ROOT}"
+export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
+export PATH="${VLLM_BIN_DIR}:${PATH}"
+export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/tinker-cookbook${PYTHONPATH:+:${PYTHONPATH}}"
+export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
 export VLLM_USE_V1="${VLLM_USE_V1:-0}"
 export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
 export BROWSECOMPPLUS_ANSWERS_PATH="${BROWSECOMPPLUS_ANSWERS_PATH:-$REPO_ROOT/external/BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl}"
@@ -38,6 +38,7 @@ echo "GPUs:           CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
 echo "Split:          ${SPLIT} (limit=${LIMIT:-all})"
 echo "max_new_tokens: ${MAX_NEW_TOKENS}"
 echo "temperature:    ${TEMPERATURE}"
+echo "seed:           ${SEED}"
 echo "max_model_len:  ${MAX_MODEL_LEN}"
 echo "Output:         ${OUTPUT_DIR}"
 echo
@@ -47,6 +48,7 @@ ARGS=(
   --split "${SPLIT}"
   --max-new-tokens "${MAX_NEW_TOKENS}"
   --temperature "${TEMPERATURE}"
+  --seed "${SEED}"
   --max-model-len "${MAX_MODEL_LEN}"
   --vllm-port "${VLLM_PORT}"
   --tensor-parallel-size 4
@@ -61,7 +63,7 @@ if [[ "${LIMIT}" != "0" ]]; then
   ARGS+=(--limit "${LIMIT}")
 fi
 
-python training/rollout_bare_browsecomp.py "${ARGS[@]}"
+"${PYTHON_BIN}" training/rollout_bare_browsecomp.py "${ARGS[@]}"
 
 echo
 echo "Done. See ${OUTPUT_DIR}/bare_rollout_manifest.json"
diff --git a/SCOPE/scripts/rollout_harness_browsecomp_4gpu.sh b/SCOPE/scripts/rollout_harness_browsecomp_4gpu.sh
index 04f2810..b3c8afe 100755
--- a/SCOPE/scripts/rollout_harness_browsecomp_4gpu.sh
+++ b/SCOPE/scripts/rollout_harness_browsecomp_4gpu.sh
@@ -1,13 +1,13 @@
 #!/usr/bin/env bash
-# Full BrowseComp+ Harness rollout on 4 GPUs.
-# Uses local vLLM + OpenAI chat API path (required for Qwen2.5 Instruct;
+# Full BrowseComp+ Harness rollout on 4 or 8 GPUs.
+# Uses local vLLM + OpenAI chat API path (required for Qwen/Qwen2.5 Instruct;
 # Harmony token path is incompatible with non-Harmony checkpoints).
 set -euo pipefail
 
 REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
-CONDA_BASE="${CONDA_BASE:-/data/ppnm/miniconda3}"
-ENV_NAME="${BISHOP_CONDA_ENV:-bishop}"
-MODEL_PATH="${MODEL_PATH:-/data/ppnm/models/Qwen2.5-7B-Instruct}"
+PYTHON_BIN="${PYTHON_BIN:-/opt/vllm-qwen3-1.7b-harness/bin/python}"
+VLLM_BIN_DIR="${VLLM_BIN_DIR:-/opt/vllm-qwen3-1.7b-harness/bin}"
+MODEL_PATH="${MODEL_PATH:-/mnt/songzijun/models/Qwen3-1.7B}"
 OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/harness_rollout_browsecomp_full}"
 # v2: Ultra ChatDecisionDriver + deterministic truncation (see modules_full_v2.yaml).
 # Legacy TokenBudget agent: USE_LEGACY_API_AGENT=1 HARNESS_CONFIG=.../modules_full.yaml
@@ -18,23 +18,43 @@ LIMIT="${LIMIT:-0}"
 MAX_TURNS="${MAX_TURNS:-35}"
 MAX_TOKENS="${MAX_TOKENS:-2048}"
 TEMPERATURE="${TEMPERATURE:-1.0}"
+SEED="${SEED:-42}"
 MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
 VLLM_PORT="${VLLM_PORT:-8771}"
 PARALLEL="${PARALLEL:-2}"
 RESUME="${RESUME:-1}"
 RERANKER="${RERANKER:-none}"
 RETRIEVAL="${RETRIEVAL:-bm25}"
+SMOKE_RETRIEVAL="${SMOKE_RETRIEVAL:-0}"
+SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
 BROWSECOMP_BM25_INDEX_PATH="${BROWSECOMP_BM25_INDEX_PATH:-$REPO_ROOT/external/BrowseComp-Plus/indexes/bm25}"
+TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
 SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-harness-policy}"
 
-source "${CONDA_BASE}/etc/profile.d/conda.sh"
-conda activate "${ENV_NAME}"
-
-export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
-export PYTHONPATH="${REPO_ROOT}"
+export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
+export PATH="${VLLM_BIN_DIR}:${PATH}"
+export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/tinker-cookbook${PYTHONPATH:+:${PYTHONPATH}}"
+export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
 export VLLM_USE_V1="${VLLM_USE_V1:-0}"
 export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
-export JAVA_HOME="${JAVA_HOME:-$CONDA_PREFIX/lib/jvm}"
+export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
+export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/mnt/songzijun/torchinductor_cache}"
+export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/mnt/songzijun/triton_cache}"
+if [[ -z "${COMPILATION_CONFIG:-}" ]]; then
+  COMPILATION_CONFIG='{"mode":0,"custom_ops":[]}'
+fi
+if [[ -z "${JAVA_HOME:-}" || ! -f "${JAVA_HOME}/lib/server/libjvm.so" ]]; then
+  if [[ -n "${VLLM_JAVA_HOME:-}" && -f "${VLLM_JAVA_HOME}/lib/server/libjvm.so" ]]; then
+    export JAVA_HOME="${VLLM_JAVA_HOME}"
+  elif [[ -f /usr/lib/jvm/java-21-openjdk-amd64/lib/server/libjvm.so ]]; then
+    export JAVA_HOME="/usr/lib/jvm/java-21-openjdk-amd64"
+  else
+    JAVA_BIN="$(command -v java || true)"
+    if [[ -n "${JAVA_BIN}" ]]; then
+      export JAVA_HOME="$(dirname "$(dirname "$(readlink -f "${JAVA_BIN}")")")"
+    fi
+  fi
+fi
 export PATH="${JAVA_HOME}/bin:${PATH}"
 export JVM_PATH="${JVM_PATH:-$JAVA_HOME/lib/server/libjvm.so}"
 export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-dummy}"
@@ -69,7 +89,7 @@ if [[ ! -s "${BROWSECOMPPLUS_ANSWERS_PATH}" && ! -s "${BROWSECOMPPLUS_QUERIES_PA
   bash "${REPO_ROOT}/scripts/setup_browsecomp_data.sh"
 fi
 
-if [[ "${RETRIEVAL}" == "bm25" ]]; then
+if [[ "${RETRIEVAL}" == "bm25" && "${SMOKE_RETRIEVAL}" != "1" ]]; then
   if ! compgen -G "${BROWSECOMP_BM25_INDEX_PATH}/segments_*" > /dev/null; then
     echo "BM25 index missing; running setup_browsecomp_bm25_index.sh ..."
     bash "${REPO_ROOT}/scripts/setup_browsecomp_bm25_index.sh"
@@ -101,6 +121,7 @@ echo "Split:           ${SPLIT} (limit=${LIMIT:-all})"
 echo "max_turns:       ${MAX_TURNS}"
 echo "max_tokens/turn: ${MAX_TOKENS}"
 echo "temperature:     ${TEMPERATURE}"
+echo "seed:            ${SEED}"
 echo "max_model_len:   ${MAX_MODEL_LEN}"
 echo "parallel:        ${PARALLEL}"
 echo "reranker:        ${RERANKER}"
@@ -111,27 +132,31 @@ echo "Output:          ${OUTPUT_DIR}"
 echo
 
 TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-hermes}"
-echo "[harness] Starting local vLLM (TP=4) at ${VLLM_URL} ..."
+echo "[harness] Starting local vLLM (TP=${TENSOR_PARALLEL_SIZE:-4}) at ${VLLM_URL} ..."
 echo "[harness] tool-call-parser=${TOOL_CALL_PARSER}"
+echo "[harness] compilation-config=${COMPILATION_CONFIG}"
+TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
 nohup vllm serve "${MODEL_PATH}" \
   --served-model-name "${SERVED_MODEL_NAME}" \
   --host 127.0.0.1 \
   --port "${VLLM_PORT}" \
-  --tensor-parallel-size 4 \
+  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
   --max-model-len "${MAX_MODEL_LEN}" \
   --dtype bfloat16 \
   --disable-custom-all-reduce \
   --enforce-eager \
+  -cc "${COMPILATION_CONFIG}" \
   --enable-auto-tool-choice \
   --tool-call-parser "${TOOL_CALL_PARSER}" \
   > "${VLLM_LOG}" 2>&1 &
 echo $! > "${VLLM_PID_FILE}"
 
 # Wait for vLLM readiness
-python - <<PY
+"${PYTHON_BIN}" - <<PY
 import time, urllib.request, sys
 url = "${VLLM_URL}/models"
-deadline = time.time() + 900
+timeout_s = float("${VLLM_STARTUP_TIMEOUT_S:-3600}")
+deadline = time.time() + timeout_s
 while time.time() < deadline:
     try:
         with urllib.request.urlopen(url, timeout=5) as resp:
@@ -140,7 +165,7 @@ while time.time() < deadline:
                 sys.exit(0)
     except Exception:
         time.sleep(3)
-print("[harness] vLLM failed to become ready; see ${VLLM_LOG}", flush=True)
+print(f"[harness] vLLM failed to become ready after {timeout_s:.0f}s; see ${VLLM_LOG}", flush=True)
 sys.exit(1)
 PY
 
@@ -156,9 +181,10 @@ ARGS=(
   --max-turns "${MAX_TURNS}"
   --max-tokens "${MAX_TOKENS}"
   --temperature "${TEMPERATURE}"
+  --seed "${SEED}"
   --max-model-len "${MAX_MODEL_LEN}"
   --vllm-port "${VLLM_PORT}"
-  --tensor-parallel-size 4
+  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
   --parallel "${PARALLEL}"
   --reranker "${RERANKER}"
   --retrieval "${RETRIEVAL}"
@@ -176,8 +202,14 @@ fi
 if [[ "${LIMIT}" != "0" ]]; then
   ARGS+=(--limit "${LIMIT}")
 fi
+if [[ "${SMOKE_RETRIEVAL}" == "1" ]]; then
+  ARGS+=(--smoke-retrieval)
+fi
+if [[ "${SKIP_PREFLIGHT}" == "1" ]]; then
+  ARGS+=(--skip-preflight)
+fi
 
-python training/rollout_harness_browsecomp.py "${ARGS[@]}"
+"${PYTHON_BIN}" training/rollout_harness_browsecomp.py "${ARGS[@]}"
 
 echo
 echo "Done. See ${OUTPUT_DIR}/harness_rollout_manifest.json"
diff --git a/SCOPE/scripts/setup_browsecomp_bm25_index.sh b/SCOPE/scripts/setup_browsecomp_bm25_index.sh
index 6637f6e..b5c8801 100755
--- a/SCOPE/scripts/setup_browsecomp_bm25_index.sh
+++ b/SCOPE/scripts/setup_browsecomp_bm25_index.sh
@@ -3,6 +3,8 @@
 set -euo pipefail
 
 REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
+PYTHON_BIN="${PYTHON_BIN:-/opt/vllm-qwen3-1.7b/bin/python}"
+VLLM_BIN_DIR="${VLLM_BIN_DIR:-/opt/vllm-qwen3-1.7b/bin}"
 INDEX_ROOT="${BROWSECOMP_BM25_INDEX_ROOT:-$REPO_ROOT/external/BrowseComp-Plus/indexes}"
 BM25_DIR="${BROWSECOMP_BM25_INDEX_PATH:-$INDEX_ROOT/bm25}"
 HF_REPO="${BROWSECOMP_BM25_HF_REPO:-Tevatron/browsecomp-plus-indexes}"
@@ -13,11 +15,9 @@ if curl -s -o /dev/null --max-time 5 -x "${PROXY_URL}" https://huggingface.co 2>
   export HTTPS_PROXY="${PROXY_URL}"
 fi
 
-source "${CONDA_BASE:-/data/ppnm/miniconda3}/etc/profile.d/conda.sh"
-conda activate "${BISHOP_CONDA_ENV:-bishop}"
-
 export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
-export PYTHONPATH="${REPO_ROOT}"
+export PATH="${VLLM_BIN_DIR}:${PATH}"
+export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/tinker-cookbook${PYTHONPATH:+:${PYTHONPATH}}"
 
 mkdir -p "${INDEX_ROOT}"
 
diff --git a/SCOPE/scripts/setup_browsecomp_data.sh b/SCOPE/scripts/setup_browsecomp_data.sh
index 80024d3..7d45d0e 100755
--- a/SCOPE/scripts/setup_browsecomp_data.sh
+++ b/SCOPE/scripts/setup_browsecomp_data.sh
@@ -3,21 +3,24 @@
 set -euo pipefail
 
 REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
-CONDA_BASE="${CONDA_BASE:-/data/ppnm/miniconda3}"
-ENV_NAME="${BISHOP_CONDA_ENV:-bishop}"
+PYTHON_BIN="${PYTHON_BIN:-/opt/vllm-qwen3-1.7b/bin/python}"
+VLLM_BIN_DIR="${VLLM_BIN_DIR:-/opt/vllm-qwen3-1.7b/bin}"
 BC_ROOT="${REPO_ROOT}/external/BrowseComp-Plus"
 
-source "${CONDA_BASE}/etc/profile.d/conda.sh"
-conda activate "${ENV_NAME}"
+export PATH="${VLLM_BIN_DIR}:${PATH}"
+export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/tinker-cookbook${PYTHONPATH:+:${PYTHONPATH}}"
 
-if [[ ! -d "${BC_ROOT}/.git" ]]; then
+if [[ ! -d "${BC_ROOT}" ]]; then
   git clone --depth 1 https://github.com/texttron/BrowseComp-Plus "${BC_ROOT}"
+elif [[ ! -s "${BC_ROOT}/scripts_build_index/decrypt_dataset.py" ]]; then
+  echo "BrowseComp-Plus exists but decrypt_dataset.py is missing: ${BC_ROOT}" >&2
+  exit 1
 fi
 
 export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
 mkdir -p "${BC_ROOT}/data" "${BC_ROOT}/topics-qrels"
 
-python "${BC_ROOT}/scripts_build_index/decrypt_dataset.py" \
+"${PYTHON_BIN}" "${BC_ROOT}/scripts_build_index/decrypt_dataset.py" \
   --output "${BC_ROOT}/data/browsecomp_plus_decrypted.jsonl" \
   --generate-tsv "${BC_ROOT}/topics-qrels/queries.tsv"
 
diff --git a/SCOPE/training/chat_decision_driver.py b/SCOPE/training/chat_decision_driver.py
index cb480b7..f2223ae 100644
--- a/SCOPE/training/chat_decision_driver.py
+++ b/SCOPE/training/chat_decision_driver.py
@@ -56,6 +56,7 @@ class ChatTurnRecord:
     observation_text: str
     episode_done: bool
     metrics: dict[str, Any]
+    inference_telemetry: dict[str, Any] | None = None
 
 
 def _action_to_capability(action: Action) -> CapabilityAction:
@@ -301,7 +302,7 @@ class ChatDecisionDriver:
             sources=["early_end_block_fallback"],
         )
 
-    async def _infer_action(self) -> Action:
+    async def _infer_action(self) -> tuple[Action, dict[str, Any]]:
         ctx = InferenceContext(
             trajectory=self._trajectory(),
             toolset=self.toolset,
@@ -333,6 +334,7 @@ class ChatDecisionDriver:
                     action = await asyncio.to_thread(self.inference, ctx)
                 except Exception:  # noqa: BLE001
                     action = self._block_to_curate_or_search(msg)
+                    ctx.telemetry["synthetic_action_reason"] = msg[:500]
             elif "maximum context length" in msg or "input_tokens" in msg:
                 self._recent = self._recent[-2:]
                 # Shrink WM budget for subsequent turns via early_end_blocks proxy
@@ -345,7 +347,7 @@ class ChatDecisionDriver:
                 action = await asyncio.to_thread(self.inference, ctx)
             else:
                 raise
-        return self._normalize_action(action)
+        return self._normalize_action(action), dict(ctx.telemetry)
 
     async def run(
         self,
@@ -360,7 +362,7 @@ class ChatDecisionDriver:
         done = False
         for _ in range(self.max_turns):
             state = self.env.export_decision_state()
-            action = await self._infer_action()
+            action, inference_telemetry = await self._infer_action()
 
             # Premature-stop guard (v2)
             if self._robust and self._is_stop_action(action):
@@ -385,6 +387,7 @@ class ChatDecisionDriver:
                 observation_text="",
                 episode_done=False,
                 metrics={},
+                inference_telemetry=inference_telemetry,
             )
 
             step = await self.env.step_from_action(action)
diff --git a/SCOPE/training/opd/__init__.py b/SCOPE/training/opd/__init__.py
index c1d910e..7057a36 100644
--- a/SCOPE/training/opd/__init__.py
+++ b/SCOPE/training/opd/__init__.py
@@ -12,12 +12,9 @@ from training.opd._policy_backend import (
 )
 from training.opd.loss import compute_opd_loss, compute_sampled_nll_loss
 from training.opd.replay_buffer import OPDReplayBuffer
-from training.opd.rollout_worker import BrowseCompRolloutWorker, RolloutConfig
 from training.opd.trainer import OPDTrainer
-from training.opd.transition_builder import build_transitions_from_rollout
 
 __all__ = [
-    "BrowseCompRolloutWorker",
     "MockPolicyBackend",
     "MockRolloutBackend",
     "MockTrainBackend",
@@ -27,9 +24,7 @@ __all__ = [
     "PolicyBackend",
     "RolloutBackend",
     "RolloutResult",
-    "RolloutConfig",
     "TrainBackend",
-    "build_transitions_from_rollout",
     "compute_opd_loss",
     "compute_sampled_nll_loss",
 ]
diff --git a/SCOPE/training/opd/bare_rollout.py b/SCOPE/training/opd/bare_rollout.py
index ade26fb..f0b5f77 100644
--- a/SCOPE/training/opd/bare_rollout.py
+++ b/SCOPE/training/opd/bare_rollout.py
@@ -3,12 +3,13 @@
 from __future__ import annotations
 
 import json
+from concurrent.futures import ThreadPoolExecutor, as_completed
 from dataclasses import asdict, dataclass, field
 from pathlib import Path
 from typing import Any, Iterable
 
 from training.opd._policy_backend import RolloutBackend
-from training.opd.rollout_worker import QueryRecord
+from training.opd.query_records import QueryRecord
 
 
 @dataclass
@@ -48,15 +49,44 @@ def load_completed_query_ids(jsonl_path: Path) -> set[str]:
     return done
 
 
+def _rollout_one_record(
+    rollout: RolloutBackend,
+    record: QueryRecord,
+    *,
+    max_new_tokens: int,
+    temperature: float,
+    top_p: float,
+) -> BareTrajectory | None:
+    result = rollout.rollout_chat(
+        bare_messages(record.query),
+        {"max_new_tokens": max_new_tokens, "temperature": temperature, "top_p": top_p},
+    )
+    if not result.action_token_ids:
+        return None
+    return BareTrajectory(
+        query_id=record.query_id,
+        query=record.query,
+        prompt_token_ids=result.prompt_token_ids,
+        response_token_ids=result.action_token_ids,
+        response_text=result.text,
+        metadata={
+            "rollout_backend": result.metadata.get("backend", "unknown"),
+            **result.metadata,
+        },
+    )
+
+
 def run_bare_rollout(
     rollout: RolloutBackend,
     records: list[QueryRecord],
     *,
     max_new_tokens: int = 2048,
     temperature: float = 1.0,
+    top_p: float = 1.0,
     output_jsonl: Path | None = None,
     resume: bool = True,
     log_every: int = 10,
+    parallel: int = 1,
 ) -> list[BareTrajectory]:
     trajectories: list[BareTrajectory] = []
     done_ids: set[str] = set()
@@ -64,37 +94,58 @@ def run_bare_rollout(
         done_ids = load_completed_query_ids(output_jsonl)
         output_jsonl.parent.mkdir(parents=True, exist_ok=True)
 
+    pending_records = [record for record in records if record.query_id not in done_ids]
+    completed = len(records) - len(pending_records)
+    total = len(records)
+    parallel = max(1, int(parallel))
+
     fh = None
     if output_jsonl is not None:
         fh = output_jsonl.open("a", encoding="utf-8")
 
+    def write_trajectory(traj: BareTrajectory) -> None:
+        trajectories.append(traj)
+        if fh is not None:
+            fh.write(json.dumps(traj.to_dict(), ensure_ascii=False) + "\n")
+            fh.flush()
+
     try:
-        for idx, record in enumerate(records, start=1):
-            if record.query_id in done_ids:
-                continue
-            result = rollout.rollout_chat(
-                bare_messages(record.query),
-                {"max_new_tokens": max_new_tokens, "temperature": temperature},
-            )
-            if not result.action_token_ids:
-                continue
-            traj = BareTrajectory(
-                query_id=record.query_id,
-                query=record.query,
-                prompt_token_ids=result.prompt_token_ids,
-                response_token_ids=result.action_token_ids,
-                response_text=result.text,
-                metadata={
-                    "rollout_backend": result.metadata.get("backend", "unknown"),
-                    **result.metadata,
-                },
-            )
-            trajectories.append(traj)
-            if fh is not None:
-                fh.write(json.dumps(traj.to_dict(), ensure_ascii=False) + "\n")
-                fh.flush()
-            if log_every > 0 and idx % log_every == 0:
-                print(f"[bare] progress {idx}/{len(records)}", flush=True)
+        if parallel == 1:
+            for record in pending_records:
+                traj = _rollout_one_record(
+                    rollout,
+                    record,
+                    max_new_tokens=max_new_tokens,
+                    temperature=temperature,
+                    top_p=top_p,
+                )
+                completed += 1
+                if traj is not None:
+                    write_trajectory(traj)
+                if log_every > 0 and completed % log_every == 0:
+                    print(f"[bare] progress {completed}/{total}", flush=True)
+            return trajectories
+
+        print(f"[bare] running with parallel={parallel}", flush=True)
+        with ThreadPoolExecutor(max_workers=parallel) as executor:
+            futures = [
+                executor.submit(
+                    _rollout_one_record,
+                    rollout,
+                    record,
+                    max_new_tokens=max_new_tokens,
+                    temperature=temperature,
+                    top_p=top_p,
+                )
+                for record in pending_records
+            ]
+            for future in as_completed(futures):
+                traj = future.result()
+                completed += 1
+                if traj is not None:
+                    write_trajectory(traj)
+                if log_every > 0 and completed % log_every == 0:
+                    print(f"[bare] progress {completed}/{total}", flush=True)
     finally:
         if fh is not None:
             fh.close()
diff --git a/SCOPE/training/opd/browsecomp_queries.py b/SCOPE/training/opd/browsecomp_queries.py
index b2f1bb9..24756c6 100644
--- a/SCOPE/training/opd/browsecomp_queries.py
+++ b/SCOPE/training/opd/browsecomp_queries.py
@@ -8,7 +8,7 @@ import os
 from pathlib import Path
 from typing import Literal
 
-from training.opd.rollout_worker import QueryRecord
+from training.opd.query_records import QueryRecord
 
 _REPO_ROOT = Path(__file__).resolve().parents[2]
 _BROWSECOMP_ROOT = _REPO_ROOT / "external" / "BrowseComp-Plus"
diff --git a/SCOPE/training/opd/env_factory.py b/SCOPE/training/opd/env_factory.py
index 6ab0996..083109b 100644
--- a/SCOPE/training/opd/env_factory.py
+++ b/SCOPE/training/opd/env_factory.py
@@ -39,10 +39,27 @@ class RolloutRuntime:
 
 
 def _default_token_counter() -> Callable[[str], int]:
-    import tiktoken
+    try:
+        import tiktoken
+
+        tiktoken_enc = tiktoken.get_encoding("o200k_harmony")
+        return lambda text: len(tiktoken_enc.encode(text))
+    except Exception as exc:
+        logger.warning("tiktoken_unavailable", error=str(exc)[:200])
+
+    try:
+        import os
+        from transformers import AutoTokenizer
+
+        tokenizer = AutoTokenizer.from_pretrained(
+            os.environ.get("MODEL_PATH", "/mnt/songzijun/models/Qwen3-1.7B"),
+            local_files_only=True,
+        )
+        return lambda text: len(tokenizer.encode(text, add_special_tokens=False))
+    except Exception as exc:
+        logger.warning("model_tokenizer_unavailable", error=str(exc)[:200])
 
-    tiktoken_enc = tiktoken.get_encoding("o200k_harmony")
-    return lambda text: len(tiktoken_enc.encode(text))
+    return lambda text: max(1, len(text) // 4)
 
 
 def _optional_reranker(
diff --git a/SCOPE/training/opd/llm_factory.py b/SCOPE/training/opd/llm_factory.py
index 7a3bee6..61c7f8a 100644
--- a/SCOPE/training/opd/llm_factory.py
+++ b/SCOPE/training/opd/llm_factory.py
@@ -37,6 +37,7 @@ def build_vllm_rollout_backend_from_env(
     base_url: str | None = None,
     model_name: str | None = None,
     api_key: str | None = None,
+    seed: int | None = None,
 ) -> VLLMRolloutBackend:
     """Create a chat rollout backend from .env or explicit overrides."""
     settings = get_llm_settings()
@@ -52,6 +53,7 @@ def build_vllm_rollout_backend_from_env(
         model_name=resolved_model,
         tokenizer_path=tokenizer_path,
         api_key=resolved_key,
+        seed=seed,
     )
 
 
diff --git a/SCOPE/training/opd/transition_builder.py b/SCOPE/training/opd/transition_builder.py
index 9c3f615..64e4b33 100644
--- a/SCOPE/training/opd/transition_builder.py
+++ b/SCOPE/training/opd/transition_builder.py
@@ -3,7 +3,7 @@
 from __future__ import annotations
 
 from training.opd._policy_backend import OPDTransition, RolloutBackend
-from training.opd.rollout_worker import QueryRecord
+from training.opd.query_records import QueryRecord
 from training.opd.shadow_harness import ShadowHarness
 from training.opd.token_alignment import is_critical_action_token
 
diff --git a/SCOPE/training/opd/vllm_rollout_backend.py b/SCOPE/training/opd/vllm_rollout_backend.py
index f806fd1..4ab170e 100644
--- a/SCOPE/training/opd/vllm_rollout_backend.py
+++ b/SCOPE/training/opd/vllm_rollout_backend.py
@@ -46,9 +46,11 @@ class VLLMRolloutBackend(RolloutBackend):
         model_name: str = "qwen",
         tokenizer_path: str,
         api_key: str = "EMPTY",
+        seed: int | None = None,
     ) -> None:
         self.base_url = base_url
         self.model_name = model_name
+        self.seed = seed
         self.client = OpenAI(base_url=base_url, api_key=api_key)
         self.tokenizer = AutoTokenizer.from_pretrained(
             tokenizer_path, trust_remote_code=True
@@ -63,12 +65,15 @@ class VLLMRolloutBackend(RolloutBackend):
     ) -> RolloutResult:
         max_tokens = int(sampling_config.get("max_new_tokens", 64))
         temperature = float(sampling_config.get("temperature", 0.7))
+        top_p = float(sampling_config.get("top_p", 1.0))
 
         response = self.client.chat.completions.create(
             model=self.model_name,
             messages=messages,
             max_tokens=max_tokens,
             temperature=max(temperature, 1e-5) if temperature > 0 else 0.0,
+            top_p=top_p,
+            seed=self.seed,
         )
         completion_text = response.choices[0].message.content or ""
 
@@ -91,5 +96,9 @@ class VLLMRolloutBackend(RolloutBackend):
                 "backend": "vllm",
                 "model": self.model_name,
                 "base_url": self.base_url,
+                "request_temperature": temperature,
+                "request_top_p": top_p,
+                "request_seed": self.seed,
+                "request_do_sample": temperature > 0,
             },
         )
diff --git a/SCOPE/training/opd/vllm_server.py b/SCOPE/training/opd/vllm_server.py
index a621951..501159e 100644
--- a/SCOPE/training/opd/vllm_server.py
+++ b/SCOPE/training/opd/vllm_server.py
@@ -70,6 +70,11 @@ def start_vllm_server(
     ]
     if enforce_eager:
         cmd.append("--enforce-eager")
+    cmd.extend([
+        "--enable-auto-tool-choice",
+        "--tool-call-parser",
+        os.environ.get("TOOL_CALL_PARSER", "hermes"),
+    ])
 
     env = os.environ.copy()
     # Smoke-friendly defaults; override in production if needed.
@@ -86,7 +91,8 @@ def start_vllm_server(
         env=env,
     )
     base_url = f"http://{host}:{port}/v1"
-    deadline = time.time() + 900.0
+    startup_timeout_s = float(os.environ.get("VLLM_STARTUP_TIMEOUT_S", "3600"))
+    deadline = time.time() + startup_timeout_s
     while time.time() < deadline:
         if process.poll() is not None:
             raise RuntimeError(
@@ -100,6 +106,6 @@ def start_vllm_server(
         except TimeoutError:
             time.sleep(2.0)
     raise TimeoutError(
-        f"vLLM server not ready at {base_url} after 900s.\n"
+        f"vLLM server not ready at {base_url} after {startup_timeout_s:.0f}s.\n"
         f"Log tail:\n{_tail_log(log_path)}"
     )
diff --git a/SCOPE/training/rollout_bare_browsecomp.py b/SCOPE/training/rollout_bare_browsecomp.py
index c7b4cd4..904dcbb 100644
--- a/SCOPE/training/rollout_bare_browsecomp.py
+++ b/SCOPE/training/rollout_bare_browsecomp.py
@@ -12,9 +12,7 @@ if str(_REPO_ROOT) not in sys.path:
     sys.path.insert(0, str(_REPO_ROOT))
 
 from training.opd.bare_rollout import run_bare_rollout, save_bare_trajectories
-from training.opd.browsecomp_queries import load_browsecomp_full_queries
-from training.opd.llm_factory import build_vllm_rollout_backend_from_env, llm_api_configured, llm_manifest_fields
-from training.opd.rollout_worker import QueryRecord, load_query_records_from_json
+from training.opd.query_records import QueryRecord, load_query_records_from_json
 from training.opd.vllm_rollout_backend import VLLMRolloutBackend
 from training.opd.vllm_server import VLLMServerHandle, start_vllm_server
 
@@ -28,7 +26,7 @@ def parse_args() -> argparse.Namespace:
     parser.add_argument(
         "--queries-json",
         default=None,
-        help="Optional debug fixture; default uses full BrowseComp+ dataset",
+        help="JSON list of {query_id, query} objects",
     )
     parser.add_argument(
         "--split",
@@ -48,6 +46,18 @@ def parse_args() -> argparse.Namespace:
         default=1.0,
         help="On-policy sampling temperature (Harness-1 eval uses 1.0)",
     )
+    parser.add_argument(
+        "--top-p",
+        type=float,
+        default=1.0,
+        help="Nucleus sampling probability; use 1.0 for deterministic T0 matched runs",
+    )
+    parser.add_argument(
+        "--seed",
+        type=int,
+        default=42,
+        help="Sampling seed for fixed-seed matched runs",
+    )
     parser.add_argument(
         "--max-model-len",
         type=int,
@@ -63,6 +73,12 @@ def parse_args() -> argparse.Namespace:
     parser.add_argument("--resume", action="store_true", default=True)
     parser.add_argument("--no-resume", action="store_false", dest="resume")
     parser.add_argument("--no-download", action="store_true")
+    parser.add_argument(
+        "--parallel",
+        type=int,
+        default=1,
+        help="Number of concurrent rollout requests sent to vLLM",
+    )
     parser.add_argument(
         "--use-llm-api",
         action="store_true",
@@ -72,20 +88,18 @@ def parse_args() -> argparse.Namespace:
 
 
 def _load_records(args: argparse.Namespace) -> list[QueryRecord]:
-    if args.queries_json:
-        records = load_query_records_from_json(args.queries_json)
-        if args.limit > 0:
-            return records[: args.limit]
-        return records
-    return load_browsecomp_full_queries(
-        split=args.split,
-        limit=args.limit,
-        download_if_missing=not args.no_download,
-    )
+    if not args.queries_json:
+        raise SystemExit("--queries-json is required for this bare rollout path.")
+    records = load_query_records_from_json(args.queries_json)
+    if args.limit > 0:
+        return records[: args.limit]
+    return records
 
 
 def main() -> None:
     args = parse_args()
+    if not args.queries_json:
+        raise SystemExit("--queries-json is required for this HotpotQA bare rollout script.")
     output_dir = Path(args.output_dir)
     output_dir.mkdir(parents=True, exist_ok=True)
     jsonl_path = output_dir / "bare_rollouts.jsonl"
@@ -101,14 +115,9 @@ def main() -> None:
 
     vllm_handle: VLLMServerHandle | None = None
     base_url = args.vllm_url or f"http://127.0.0.1:{args.vllm_port}/v1"
-    use_api = args.use_llm_api or (llm_api_configured() and args.vllm_url is None)
 
     try:
-        if use_api:
-            print("[bare] Using LLM API from BiSHOP/.env (base_url, api_key, model_name)")
-            rollout = build_vllm_rollout_backend_from_env(tokenizer_path=args.model_path)
-            base_url = rollout.base_url
-        elif args.vllm_url is None and args.manage_vllm:
+        if args.vllm_url is None and args.manage_vllm:
             print(f"[bare] Starting vLLM (TP={args.tensor_parallel_size}) at {base_url} ...")
             vllm_handle = start_vllm_server(
                 model_path=args.model_path,
@@ -119,25 +128,26 @@ def main() -> None:
             )
             base_url = vllm_handle.base_url
             print(f"[bare] vLLM ready: {base_url}")
-            rollout = VLLMRolloutBackend(
-                base_url=base_url,
-                model_name="qwen",
-                tokenizer_path=args.model_path,
-            )
-        else:
+        elif args.vllm_url is not None:
             print(f"[bare] Using vLLM at {base_url}")
-            rollout = VLLMRolloutBackend(
-                base_url=base_url,
-                model_name="qwen",
-                tokenizer_path=args.model_path,
-            )
+        else:
+            raise SystemExit("Either --vllm-url or --manage-vllm is required")
+
+        rollout = VLLMRolloutBackend(
+            base_url=base_url,
+            model_name="qwen",
+            tokenizer_path=args.model_path,
+            seed=args.seed,
+        )
         run_bare_rollout(
             rollout,
             records,
             max_new_tokens=args.max_new_tokens,
             temperature=args.temperature,
+            top_p=args.top_p,
             output_jsonl=jsonl_path,
             resume=args.resume,
+            parallel=args.parallel,
         )
         path = save_bare_trajectories(
             [],
@@ -145,14 +155,15 @@ def main() -> None:
             manifest={
                 "model_path": args.model_path,
                 "vllm_url": base_url,
-                "backend": "api" if use_api else "vllm",
-                **(llm_manifest_fields() if use_api else {}),
+                "backend": "vllm",
                 "max_new_tokens": args.max_new_tokens,
                 "temperature": args.temperature,
+                "top_p": args.top_p,
+                "do_sample": args.temperature > 0,
                 "max_model_len": args.max_model_len,
-                "split": args.split,
-                "queries_source": args.queries_json or "browsecompplus_full",
-                "resume": args.resume,
+                "parallel": args.parallel,
+                "queries_source": args.queries_json,
+                "seed": args.seed,
             },
         )
         print(f"[bare] Saved trajectories -> {path}")
diff --git a/SCOPE/training/rollout_harness_browsecomp.py b/SCOPE/training/rollout_harness_browsecomp.py
index f6a3d26..0272abc 100644
--- a/SCOPE/training/rollout_harness_browsecomp.py
+++ b/SCOPE/training/rollout_harness_browsecomp.py
@@ -67,6 +67,7 @@ def parse_args() -> argparse.Namespace:
     parser.add_argument("--top-p", type=float, default=0.9)
     parser.add_argument("--parallel", type=int, default=2)
     parser.add_argument("--max-model-len", type=int, default=32768)
+    parser.add_argument("--seed", type=int, default=42, help="Sampling seed for fixed-seed matched runs")
     parser.add_argument("--output-dir", default="outputs/harness_rollout_browsecomp_full")
     parser.add_argument("--vllm-port", type=int, default=8771)
     parser.add_argument("--tensor-parallel-size", type=int, default=4)
@@ -161,7 +162,9 @@ async def _run_rollout(
                     runtime.text_token_counter,
                     max_tokens=args.max_tokens,
                     temperature=args.temperature,
+                    top_p=args.top_p,
                     max_trajectory_length=args.max_turns,
+                    seed=args.seed,
                 )
             else:
                 result = await eval_single_query(
@@ -344,12 +347,13 @@ def main() -> None:
                 "max_turns": args.max_turns,
                 "max_tokens": args.max_tokens,
                 "temperature": args.temperature,
+                "top_p": args.top_p,
                 "max_model_len": args.max_model_len,
                 "split": args.split,
                 "collection_split": args.collection_split,
                 "parallel": args.parallel,
                 "reranker": args.reranker,
-                "resume": args.resume,
+                "seed": args.seed,
                 "summary": summary,
             },
         )
diff --git a/SCOPE/training/train_rl.py b/SCOPE/training/train_rl.py
index 42bdc2b..3e54a61 100644
--- a/SCOPE/training/train_rl.py
+++ b/SCOPE/training/train_rl.py
@@ -149,6 +149,8 @@ from harness.ultra_core import (
 )
 
 logger = structlog.get_logger("ultra_rl_v3")
+_HARMONY_ENCODING_CACHE: HarmonyEncoding | None = None
+_HARMONY_ENCODING_FAILED = False
 
 # Save trajectory details for debugging
 SAVE_TRAJECTORIES = os.environ.get("SAVE_TRAJECTORIES", "1") == "1"
@@ -250,7 +252,19 @@ class SlidingWindowSearchEnv(Env):
         self.max_turns = max_turns
         self.rollout_idx = rollout_idx
 
-        self.enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
+        global _HARMONY_ENCODING_CACHE, _HARMONY_ENCODING_FAILED
+        if _HARMONY_ENCODING_FAILED:
+            self.enc = None
+        elif _HARMONY_ENCODING_CACHE is not None:
+            self.enc = _HARMONY_ENCODING_CACHE
+        else:
+            try:
+                _HARMONY_ENCODING_CACHE = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
+                self.enc = _HARMONY_ENCODING_CACHE
+            except Exception as exc:
+                logger.warning("harmony_encoding_unavailable", error=str(exc)[:200])
+                _HARMONY_ENCODING_FAILED = True
+                self.enc = None
         self.stop_condition: StopCondition = [200002, 200012]
 
         self._normalize_ids = (
@@ -460,6 +474,9 @@ class SlidingWindowSearchEnv(Env):
         self._action_records = []
         self._episode_id = f"{self.query_id}_r{self.rollout_idx}"
 
+        if self.enc is None:
+            return tinker.ModelInput.empty(), self.stop_condition
+
         tokens = render_context_within_budget(
             system_prompt=self.system_prompt,
             wm_text=None,
@@ -665,6 +682,10 @@ class SlidingWindowSearchEnv(Env):
                 and self.wm.get_pool_size() > 0):
             nudge = CURATE_NUDGE_PROMPT
 
+        if self.enc is None:
+            self._approx_prompt_tokens = 0
+            return []
+
         tokens = render_context_within_budget(
             system_prompt=self.system_prompt,
             wm_text=wm_text,
```
