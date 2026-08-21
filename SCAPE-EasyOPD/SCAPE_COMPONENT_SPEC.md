# SCAPE_COMPONENT_SPEC

The core dataclass is `easyopd/methods/scape_component_opd/component_spec.py::ComponentSpec`.

Required contract fields implemented:
- `name`
- `effect_type`
- `realizability`
- teacher/student component toggles
- event detector
- teacher/student view builders
- state snapshot/restore
- effect/projected-action builders
- supervision builder
- default loss mode
- visibility/action/leakage validators
- mechanism metrics
- train refusal code

Registered components:
`verify_tool`, `importance_tagging`, `subtractive_curation`, `auto_populate_first_search`, `content_dedup`, `chunk_neighbors`, `evidence_graph`, `sentence_compress`, `token_budget_marker`, `adaptive_rerank_instruction`.
