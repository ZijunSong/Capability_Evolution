# LEAKAGE_TEST

`SCAPEAgentLoop` rejects `student_inference_privilege=True` and removes teacher-only fields (`evidence_graph`, `curated_importance`, `token_budget_marker`) from student view in tests. Full live evaluator leakage audit remains pending.
