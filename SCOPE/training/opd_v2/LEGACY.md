# Legacy OPD v2 baseline

`training/opd_v2/` and `OPDTransitionV2` are retained for ablation baselines only.

**Do not** use OPDTransitionV2 as the primary schema for new SCOPE supervision work.

New code path:

```text
training/scope/
  schema.py          → DecisionSupervisionSampleV3
  routing.py         → VerifiedDecisionRouting
  pipeline.py        → full DecisionState → sample chain
  dataset_builder.py → build from audit events
  validators.py      → InformationSafeGate façade
```

Entry points:

- `training/build_scope_dataset.py` (v3 JSONL)
- `training/scope/pipeline.py`

Legacy entry points (OPD v2):

- `training/train_scope.py` (still wires OPD for comparison runs)
