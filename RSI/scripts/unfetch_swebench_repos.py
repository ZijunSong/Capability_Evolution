"""One-shot unshallow/unfilter of the blob:none partial clones (resumable).

2026-11-02: the per-worktree lazy blob fetch made ``prepare_swebench_workspaces.py``
~30 min per django instance (>150 h total).  A single full fetch per repo pulls
every blob once; afterwards every ``worktree add`` is purely local and instant.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

CLONES = Path("/mnt/songzijun/datasets/swebench/clones")

REPOS = [
    "django__django",
    "sympy__sympy",
    "sphinx-doc__sphinx",
    "matplotlib__matplotlib",
    "scikit-learn__scikit-learn",
    "pytest-dev__pytest",
    "pydata__xarray",
    "astropy__astropy",
    "pylint-dev__pylint",
    "psf__requests",
    "mwaskom__seaborn",
    "pallets__flask",
]


def sh(cmd: list[str], cwd: Path, timeout: int = 14400) -> int:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        print(f"  FAIL {' '.join(cmd)[:100]} :: {(proc.stderr or proc.stdout)[-200:]}", flush=True)
    return proc.returncode


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", nargs="*", default=None, help="subset of clone dir names")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    names = a.repos or REPOS

    def fetch_one(name: str) -> tuple[str, bool, float]:
        repo = CLONES / name
        if not repo.is_dir():
            return name, False, 0.0
        t0 = time.time()
        # fetch every ref's full tree+blob set; promisor is the partial-clone filter
        ok = (
            sh(["git", "fetch", "--unshallow", "--filter=", "origin", "+refs/heads/*:refs/remotes/origin/*"], repo) == 0
            or sh(["git", "fetch", "--filter=", "origin", "+refs/heads/*:refs/remotes/origin/*"], repo) == 0
            or sh(["git", "fetch", "origin"], repo) == 0
        )
        # belt & suspenders: force materialize all promisor objects
        if ok:
            sh(["git", "fetch", "--refetch", "--filter=", "origin"], repo)
        cfg = subprocess.run(["git", "config", "remote.origin.partialclonefilter"], cwd=str(repo), capture_output=True, text=True).stdout.strip()
        if cfg:
            # drop the partial-clone filter so future fetches stop being lazy
            subprocess.run(["git", "config", "--unset", "remote.origin.partialclonefilter"], cwd=str(repo))
            subprocess.run(["git", "config", "--unset", "remote.origin.promisor"], cwd=str(repo))
        secs = round(time.time() - t0, 1)
        print(f"{name}: {'ok' if ok else 'FAILED'} ({secs}s)", flush=True)
        return name, ok, secs

    import concurrent.futures as cf

    results = {}
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(fetch_one, n): n for n in names}
        for fut in cf.as_completed(futs):
            name, ok, secs = fut.result()
            results[name] = ok
    failed = [n for n, ok in results.items() if not ok]
    print(f"DONE ok={sum(results.values())}/{len(results)} failed={failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
