# H20 Pre-Sync Audit (2026-08-12)

## Repo

- path: `/data/ppnm/Capability_Evolution/SCAPE`
- branch (at audit): `main`
- HEAD: `608068d56067419317f2cea120ebdc6fa099479b`
- remotes:
  - `origin` → `git@github.com:ZijunSong/SCAPE.git`
  - `capability-evolution` → `git@github.com:ZijunSong/Capability_Evolution.git`

## Dirty tree summary

Tracked modifications:

- `README.md`
- `result-record.md`

Untracked (code/docs/scripts relevant to sync; excludes `outputs/`):

- `0812/` (five-machine plan docs)
- `docs/BLOCKED_RETRIEVAL_BACKEND.md`
- `result-record-from-h100.md`
- provisional Stage L/S launch/aggregate scripts under `scripts/`

## GitHub reachability

```text
git@github.com: Permission denied (publickey).
```

`origin` fetch/push unavailable from this host at audit time.
Fallback: local branch `sync/h20-20260812` + `artifacts/git/scape-h20-20260812.bundle`
and `GITHUB_SYNC_BLOCKED.md`.

## H100 sync branch

`origin/sync/h100-20260812` could not be probed (same auth failure).
Integration (`integration/scape-20260812`) deferred until H100 bundle/branch is available.

## Host

- hostname: H20 node (`NF5688A7`)
- GPUs: 8× NVIDIA H20-3e (idle at audit)
- conda env: `bishop` (`/data/ppnm/miniconda3/envs/bishop`)
