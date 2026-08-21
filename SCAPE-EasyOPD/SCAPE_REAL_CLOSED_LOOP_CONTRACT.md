# SCAPE_REAL_CLOSED_LOOP_CONTRACT

Implemented wrapper: `easyopd/methods/scape_component_opd/real_closed_loop_evaluator.py::SCAPERealClosedLoopEvaluator`.

Contract:
- query manifest: explicit list of query_id/query rows
- retriever/tool runtime: SCAPE/Harness-1 `ToolSet.from_config` with live `search_corpus` and `read_document`
- max steps: evaluator config
- reward: executed real closed-loop smoke reward from tool success
- termination: Harness `Agent.is_done`
- parser: Harness trajectory/action contract
- final answer scoring: smoke contract for acceptance
- route proxy: false
- student inference privilege: false

Acceptance output:
`outputs/scape_easyopd/acceptance/eval_auto_populate_first_search_seed8183/` contains `REAL_CLOSED_LOOP_PER_QUERY.jsonl`, `REAL_CLOSED_LOOP_SUMMARY.json`, `REAL_CLOSED_LOOP_SUMMARY.csv`, `SCAPE_REAL_CLOSED_LOOP_CONTRACT.json`, and `REAL_CLOSED_LOOP_HANDOFF.json`.
