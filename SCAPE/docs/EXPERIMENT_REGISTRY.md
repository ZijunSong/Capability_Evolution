# EXPERIMENT_REGISTRY

| Stage | ID pattern | Owner | Output root | Gate |
|---|---|---|---|---|
| Preflight | `preflight` | all | `outputs/preflight/` | env ready |
| Contribution | `h100_1_contribution` | H100-1 | `outputs/h100_1_contribution/` | map written |
| Replication/Coalition | `h100_2_replication` | H100-2 | `outputs/h100_2_replication/` | LOO/coalition CSV |
| Influence | `h100_3_influence` | H100-3 | `outputs/h100_3_influence/` | I_name/I_args |
| Local CAL64 | `local_cal64` | H20 | `outputs/local_cal64/` | provisional only |
| Candidate select | `scape_prestage` | H20 | `outputs/scape_prestage/` | top-2 A/B |
| Learnability | `stage_l/*` | H20 | `outputs/stage_l/` | Gate L |
| Single migration | `stage_s/*` | H20 | `outputs/stage_s/` | Gate S |
| Multi annealing | `stage_m/*` | H20 | `outputs/stage_m/` | Gate M |
| Pareto | `runtime_recomposition` | H20 | `outputs/pareto/` | final table |

Every run writes `RUN_MANIFEST.json`, `STATUS_LIVE.md`, and contributes to `SHA256SUMS` on finalize.
