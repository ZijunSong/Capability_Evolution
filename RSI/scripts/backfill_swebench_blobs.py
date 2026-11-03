"""Backfill SWE-bench clone blobs so worktree adds are local (resumable).

git 2.34 has no `fetch --refetch`/`backfill`, so a blob:none promisor clone
cannot be bulk-materialized in place.  Two strategies instead:

- official 12 repos (hundreds of instances each): one FULL clone (no filter,
  --no-checkout) into ``clones_full/<repo>`` — every worktree add afterwards
  is purely local.
- panel-only repos (few missing instances each): per-base_commit
  ``git fetch --depth 1 origin <sha>`` into the existing clone.

Existing worktrees under repos/ stay valid (they keep pointing at clones/).
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path("/mnt/songzijun/datasets/swebench")
CLONES = ROOT / "clones"
CLONES_FULL = ROOT / "clones_full"

OFFICIAL_REPOS_FULL = [  # one full clone each
    "django__django", "sympy__sympy", "sphinx-doc__sphinx", "matplotlib__matplotlib",
    "scikit-learn__scikit-learn", "pytest-dev__pytest", "pydata__xarray",
    "astropy__astropy", "pylint-dev__pylint", "psf__requests",
    "mwaskom__seaborn", "pallets__flask",
]


def sh(cmd: list[str], cwd: Path | None = None, timeout: int = 14400) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stderr or p.stdout)[-300:]


def full_clone(repo: str) -> tuple[str, bool, str]:
    dest = CLONES_FULL / repo
    if (dest / "config").is_file():
        print(f"  {repo}: clones_full exists, skip", flush=True)
        return repo, True, "exists"
    t0 = time.time()
    CLONES_FULL.mkdir(parents=True, exist_ok=True)
    rc, err = sh(["git", "clone", "--no-checkout", f"https://github.com/{repo.replace('__', '/')}.git", str(dest)], timeout=14400)
    secs = round(time.time() - t0, 1)
    print(f"  {repo}: {'ok' if rc == 0 else 'FAILED'} ({secs}s) {err if rc else ''}", flush=True)
    return repo, rc == 0, err


def depth1_fetch(repo: str, sha: str) -> tuple[str, str, bool, str]:
    clone = CLONES / repo
    if not clone.is_dir():
        # clone missing (never cloned / clone step failed) — skip, don't crash the pool
        print(f"  {repo}@{sha[:10]}: SKIP (no clone dir)", flush=True)
        return repo, sha, False, "no clone dir"
    t0 = time.time()
    rc, err = sh(["git", "fetch", "--depth", "1", "origin", sha], cwd=clone, timeout=1800)
    secs = round(time.time() - t0, 1)
    if rc != 0:
        print(f"  {repo}@{sha[:10]}: FAILED ({secs}s) {err}", flush=True)
    return repo, sha, rc == 0, err


def missing_panel_commits() -> dict[str, list[str]]:
    """panel-scope instances with no worktree yet -> (repo -> [base_commits])."""
    sys.path.insert(0, "/mnt/songzijun/Capability_Evolution/RSI/src")
    import os as _os
    _os.environ.setdefault("RSI_SWEBENCH_ROOT", str(ROOT))
    from rsi.environments.task_providers import SWEBenchTaskProvider

    roles = {"work_train": 64, "harness_select": 32, "weight_dev": 32,
             "confirm_h": 16, "confirm_w": 16, "final_test": 32, "transfer_test": 16}
    p = SWEBenchTaskProvider()
    repos_dir = ROOT / "repos"
    have = {d.name for d in repos_dir.iterdir() if d.is_dir()} if repos_dir.is_dir() else set()
    need: dict[str, dict[str, str]] = {}
    for role in roles:
        for t in p.public_tasks(role):
            iid = str(t.metadata["instance_id"])
            if iid in have:
                continue
            repo = str(t.metadata["repo"]).replace("/", "__")
            need.setdefault(repo, {})[str(t.metadata["base_commit"])] = iid
    return {r: sorted(c) for r, c in need.items()}


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-full", action="store_true")
    ap.add_argument("--skip-depth1", action="store_true")
    ap.add_argument("--full-workers", type=int, default=3)
    # 2026-11-02: fetch-workers must be 1 — concurrent depth-1 fetches into the
    # same clone race on .git/shallow.lock ("shallow file has changed" /
    # "w.lock: File exists"); grouping by repo made that ~100% likely.
    ap.add_argument("--fetch-workers", type=int, default=1)
    a = ap.parse_args()

    failures: list[str] = []

    if not a.skip_full:
        print(f"== full clones x{len(OFFICIAL_REPOS_FULL)} -> clones_full/", flush=True)
        with ThreadPoolExecutor(max_workers=a.full_workers) as ex:
            for fut in as_completed([ex.submit(full_clone, r) for r in OFFICIAL_REPOS_FULL]):
                repo, ok, err = fut.result()
                if not ok:
                    failures.append(f"full:{repo}: {err}")

    if not a.skip_depth1:
        need = missing_panel_commits()
        # exclude repos already covered by clones_full
        need = {r: c for r, c in need.items() if r not in OFFICIAL_REPOS_FULL}
        jobs = [(r, sha) for r, commits in need.items() for sha in commits]
        print(f"== depth-1 fetches x{len(jobs)} across {len(need)} panel repos", flush=True)
        with ThreadPoolExecutor(max_workers=a.fetch_workers) as ex:
            for fut in as_completed([ex.submit(depth1_fetch, r, s) for r, s in jobs]):
                repo, sha, ok, err = fut.result()
                if not ok:
                    failures.append(f"depth1:{repo}@{sha[:10]}: {err}")

    print(f"DONE failures={len(failures)}", flush=True)
    for f in failures:
        print("  " + f[:200], flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
