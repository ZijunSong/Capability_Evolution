# Capability Evolution
#
# Umbrella repo for SCOPE / HATCH / RECAST.
# Phase-0 SCOPE baselines are frozen at tag `scope-phase0-freeze`.
# Do not optimize or overwrite Full Harness v2 / Minimal Runtime configs
# or their Phase-0 metrics after that tag.

## Layout

```text
Capability_Evolution/
├── SCOPE/     # selective capability internalization (active)
├── HATCH/     # placeholder
├── RECAST/    # placeholder
└── .gitignore
```

## SCOPE Phase-0 freeze

Immutable baselines (see `SCOPE/artifacts/baselines/PHASE0_FREEZE.md`):

- `SCOPE/harness/configs/modules_full_v2.yaml`
- `SCOPE/harness/configs/modules_minimal.yaml`
- metrics under `SCOPE/artifacts/baselines/`

Tag: `scope-phase0-freeze`
