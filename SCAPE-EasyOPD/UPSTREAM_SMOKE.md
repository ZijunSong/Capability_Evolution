# UPSTREAM_SMOKE

Environment: `/opt/scape-easyopd-smoke7`.

Passed:
- import `torch`, `transformers`, `peft`, `ray`, `easyopd`, `verl`
- 8 GPUs visible
- BF16 matmul on all 8 GPUs
- `scripts/run_easyopd.py --list-methods`
- `--method gkd --dry-run`
- `--method sod --dry-run`
- `--method opcd --dry-run`
- `--method scape_component_opd --config easyopd/config/scape_component_opd.yaml --dry-run component.name=auto_populate_first_search`

Note: dry-run was changed to avoid HuggingFace dataset download side effects.
