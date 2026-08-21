# INDEPENDENT_METRIC_IMPLEMENTATION

This audit independently rescored the lightweight `stage_l` checkpoints using the synthetic TinyToolPolicy setup already recorded in the repo.

- metric_contract_valid: false
- audited_rows: 16
- audited_ok_rows: 16
- missing_artifacts: 20

## Independence
- Does not import `scape.training.hf_tool_opd`.
- Rebuilds the dataset from component, seed, and n_samples.
- Uses a fresh TinyToolPolicy definition and fresh CE/KL scoring code.