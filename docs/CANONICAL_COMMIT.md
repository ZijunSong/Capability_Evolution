# Canonical Commit

## Current H20 sync snapshot

| Field | Value |
|---|---|
| branch | `sync/h20-20260812` |
| commit | `f63e4b8c16f780c072678e492c7436c0c2c9e0ee` |
| date | 2026-08-12 |
| host | H20 `/data/ppnm/Capability_Evolution/SCAPE` |
| bundle | `artifacts/git/scape-h20-20260812.bundle` |

## Integration status

- GitHub push blocked (`Permission denied (publickey)`). See `GITHUB_SYNC_BLOCKED.md`.
- `origin/sync/h100-20260812` not available yet → `integration/scape-20260812` deferred.
- Until integration lands, treat `f63e4b8` as the H20 working canonical for true-SCAPE plumbing.

## Assertions for new experiments

- `legacy_scope_path_used = false`
- Canonical training modules under `scape/state|rendering|training|collection`
- Provisional Qwen+BM25 A/B Gate S line is archived diagnostics only
