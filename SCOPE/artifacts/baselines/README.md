# SCOPE Phase 0 baselines (frozen)

See **[PHASE0_FREEZE.md](./PHASE0_FREEZE.md)** — Full Harness v2 and Minimal Runtime
are **immutable** after tag `scope-phase0-freeze`.

## Frozen metrics (do not overwrite)

| File | Runtime |
|------|---------|
| `bare_metrics.json` | Bare single-shot |
| `full_harness_v1_metrics.json` | TokenBudget API harness (legacy) |
| `full_harness_v2_metrics.json` | Ultra ChatDecisionDriver full modules |
| `minimal_runtime_metrics.json` | Executor-only minimal modules |
| `compare_phase0_full830.json` | Joint 830-query comparison |

## Frozen configs

| File | Runtime |
|------|---------|
| `../../harness/configs/modules_full_v2.yaml` | Full Harness v2 |
| `../../harness/configs/modules_minimal.yaml` | Minimal Runtime |

Snapshots + checksums: `phase0_freeze/`.

```bash
python scripts/verify_phase0_freeze.py
```
