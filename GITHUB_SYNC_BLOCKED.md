# GitHub Sync Blocked (H20, 2026-08-12)

## Symptom

```text
git@github.com: Permission denied (publickey).
fatal: Could not read from remote repository.
```

Neither `origin` (`ZijunSong/SCAPE`) nor `capability-evolution`
(`ZijunSong/Capability_Evolution`) accepts SSH from this host.

## Fallback artifacts

| Artifact | Path |
|---|---|
| H20 sync branch | `sync/h20-20260812` (local) |
| Git bundle | `artifacts/git/scape-h20-20260812.bundle` |
| Pre-sync audit | `docs/H20_PRE_SYNC_AUDIT.md` |

## How to restore elsewhere

```bash
git clone scape-h20-20260812.bundle scape-from-h20
cd scape-from-h20
git checkout sync/h20-20260812
# or fetch into an existing clone:
git fetch /path/to/scape-h20-20260812.bundle sync/h20-20260812:sync/h20-20260812
```

## Integration impact

- Cannot pull `origin/sync/h100-20260812` until auth or a H100 bundle arrives.
- H20 continues Part B/C (true SCAPE OPD plumbing) on the local sync branch.
- Part A2 integration will resume when H100 snapshot is deliverable via bundle or fixed SSH.
