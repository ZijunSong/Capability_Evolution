# SCAPE

**S**ystem-level **C**apability migration via public se**A**rch harness **P**olicy / post-training recomposition **E**xperiments.

Canonical research repo for moving from SCOPE’s capability-specific toy line to a Harness-1–centered system experiment:

```text
public real search harness
→ component intervention
→ same-environment-state dual view
→ unified tool-call OPD
→ post-training runtime recomposition
```

## Layout

```text
SCAPE/
├── external/harness-1/     # pinned upstream (submodule)
├── scape/                  # adapters, state, rendering, probes, training, eval
├── configs/
├── scripts/
├── tests/
├── docs/
├── imports/h100_{1,2,3}/   # synced pre-stage results
└── outputs/                # run artifacts (gitignored)
```

## GitHub

发布在 umbrella 仓库 `ZijunSong/Capability_Evolution`（**没有**独立的 `ZijunSong/SCAPE`）：

```text
https://github.com/ZijunSong/Capability_Evolution/tree/main/SCAPE
```

H20 当前同步分支：

```text
https://github.com/ZijunSong/Capability_Evolution/tree/sync/h20-20260812
```

本工作树 `origin` = `git@github.com:ZijunSong/Capability_Evolution.git`。

## Quickstart (code only)

```bash
cd /data/ppnm/Capability_Evolution/SCAPE
# recommended: bishop conda (Python 3.11+)
pip install -e ".[dev]"
pytest -q
```

## Machine roles

| Host | Role |
|---|---|
| H100-1 | System Contribution / LOO |
| H100-2 | Replication + coalition |
| H100-3 | Same-state Influence |
| H20 | Canonical repo + Learnability + Migration |

See `SCAPE_MASTER_4SERVER_PLAN.md` and `SCAPE_H20_TRAINING_MIGRATION.md`.

## Hard rules

1. Do not continue SCOPE rollback / KEEP·SKIP / P_m / O7 / Information-Safe Gate.
2. Do not substitute SCOPE BM25 when Chroma retrieval is missing — write `docs/BLOCKED_RETRIEVAL_BACKEND.md`.
3. Full-view teacher must not step the environment; reduced student rollout owns `xi_t`.
4. Runtime anchors are not full-internalization candidates.
