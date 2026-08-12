# H100 GitHub handoff

## Branch and snapshot

- branch: `sync/h100-20260812`
- current HEAD during audit: `6be4f1088f14033950a54566e762272af4d12e23`
- required H100 snapshot `0f0934bd9f7a985af747e18dda9c2c666a9c24ba` is an ancestor of current HEAD.

## GitHub sync status

- remote: `https://github.com/ZijunSong/Capability_Evolution.git`
- `gh auth status` could not be run on this host because `gh` is not installed.
- No push was attempted from this session.
- If a push is needed, use the existing HTTPS remote and the project-standard temporary `GIT_ASKPASS` token flow after sourcing `/mnt/songzijun/mify_api.env`; do not put tokens in remotes, command-line arguments, or logs.

## Diff summary prepared for PR/update

- Added `scripts/attribute_h100_3_influence.py` to aggregate the existing H100-3 real influence per-state JSONL into turn/tool/argument attribution reports.
- Generated, but intentionally did not stage for commit, output artifacts under `outputs/h100_3_influence_attribution/`:
  - `INFLUENCE_BY_TOOL.csv`
  - `INFLUENCE_BY_ARGUMENT_CLASS.csv`
  - `EVIDENCE_GRAPH_ATTRIBUTION.md`
  - `IMPORTANCE_TAGGING_ATTRIBUTION.md`
  - `VERIFY_TOOL_ATTRIBUTION.md`
  - `HIGH_INFLUENCE_ARCHETYPES.jsonl`
  - `H20_LOSS_RECOMMENDATION.md`
- GPU rescore was skipped because `outputs/h100_3_real_influence/REAL_INFLUENCE_PER_STATE.jsonl` already contains per-state tool distributions, influence metrics, and null statistics.

## Tests / validation

```bash
python -m py_compile scripts/attribute_h100_3_influence.py
python scripts/attribute_h100_3_influence.py
```

Both completed successfully in this session.

## H100-3 attribution headline

- `evidence_graph`: mean `I_name=0.038704`, mean `I_args=0.117327`; later ablation should emphasize args while retaining name loss.
- `importance_tagging`: mean `I_name=0.028771`, mean `I_args=0.016560`; later ablation can emphasize name loss with medium args loss.
- `verify_tool`: mean `I_name=0.019043`, mean `I_args=0.050669`; later ablation should emphasize args while retaining name loss.
- H20 V0 should still follow the coordination plan: uniform name+args tool-token KL first; attribution only informs subsequent ablations/stratification.
