# Phase-0 freeze (IMMUTABLE)

**Tag:** `scope-phase0-freeze`

From this freeze forward:

- **Full Harness v2** (`harness/configs/modules_full_v2.yaml`) is an immutable baseline.
- **Minimal Runtime** (`harness/configs/modules_minimal.yaml`) is an immutable baseline.
- Do **not** optimize, retune, or overwrite these configs or the Phase-0 metrics below.
- For new experiments, copy to a new filename (e.g. `modules_full_v3_exp.yaml`).

## Protected files

| Path | Role |
|------|------|
| `harness/configs/modules_full_v2.yaml` | Full Harness v2 config |
| `harness/configs/modules_minimal.yaml` | Minimal Runtime config |
| `artifacts/baselines/bare_metrics.json` | Bare metrics |
| `artifacts/baselines/full_harness_v1_metrics.json` | Harness v1 metrics |
| `artifacts/baselines/full_harness_v2_metrics.json` | Harness v2 metrics |
| `artifacts/baselines/minimal_runtime_metrics.json` | Minimal Runtime metrics |
| `artifacts/baselines/compare_phase0_full830.json` | Four-way Phase-0 compare |

Byte-identical snapshots + SHA256 digests live in `phase0_freeze/`.

Verify anytime:

```bash
python SCOPE/scripts/verify_phase0_freeze.py
```

## Policy

SCOPE development after this tag must not regress or “improve” these baselines in place.
Training / ablation / retirement work should reference them as fixed controls.
