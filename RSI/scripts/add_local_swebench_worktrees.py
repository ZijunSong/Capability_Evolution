"""Materialize SWE-bench worktrees for every instance whose base_commit is
already local in ``clones/<repo>`` or ``clones_full/<repo>`` (instant, no
network).  Instances whose commit is still missing are reported for the
background depth-1 backfill to pick up later.  Resumable; safe to re-run
while a fetch is in flight for a *different* repo.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, "/mnt/songzijun/Capability_Evolution/RSI/src")
import os as _os

_os.environ.setdefault("RSI_SWEBENCH_ROOT", "/mnt/songzijun/datasets/swebench")

ROOT = Path("/mnt/songzijun/datasets/swebench")
REPOS = ROOT / "repos"
MANIFEST = ROOT / "workspaces_manifest.json"

ROLES = {"work_train": 64, "harness_select": 32, "weight_dev": 32,
         "confirm_h": 16, "confirm_w": 16, "final_test": 32, "transfer_test": 16}


def panel_and_official() -> dict[str, dict[str, str]]:
    """panel instances via provider + official universe via parquets."""
    import pandas as pd
    from rsi.environments.task_providers import SWEBenchTaskProvider

    out: dict[str, dict[str, str]] = {}
    p = SWEBenchTaskProvider()
    for role in ROLES:
        for t in p.public_tasks(role):
            out[str(t.metadata["instance_id"])] = {
                "repo": str(t.metadata["repo"]),
                "base_commit": str(t.metadata["base_commit"]),
            }
    te = pd.read_parquet(ROOT / "enriched" / "swe_bench_test.parquet")
    verified = set(pd.read_parquet(ROOT / "verified_test.parquet")["instance_id"])
    lite = set(pd.read_parquet(ROOT / "lite_test.parquet")["instance_id"])
    rows = te[te["instance_id"].isin(verified | lite)]
    for _, r in rows.iterrows():
        out.setdefault(str(r["instance_id"]), {"repo": str(r["repo"]), "base_commit": str(r["base_commit"])})
    return out


def has_commit(clone: Path, sha: str) -> bool:
    # a git repo dir: either a worktree-style clone with .git file, or a bare-ish
    # --no-checkout clone with .git/ subdir (clones_full).  Just ask git.
    if not (clone / ".git").exists():
        return False
    r = subprocess.run(["git", "-C", str(clone), "cat-file", "-t", sha],
                       capture_output=True, text=True, timeout=120)
    return r.returncode == 0 and "commit" in r.stdout


def add_worktree(iid: str, repo: str, sha: str) -> tuple[str, str]:
    # provider/parquet give 'owner/name'; clone dirs are 'owner__name'
    repo = repo.replace("/", "__")
    dest = REPOS / iid
    if dest.exists():
        try:
            if any(dest.iterdir()):
                return iid, "exists"
        except OSError:
            return iid, "exists"  # unreadable (juicefs hiccup) — don't rm it; report and move on
        # empty dir: git worktree add can create into an existing empty dir fine
    clone = None
    for cand in (ROOT / "clones_full" / repo, ROOT / "clones" / repo):
        if has_commit(cand, sha):
            clone = cand
            break
    if clone is None:
        return iid, "commit_not_local"
    t0 = time.time()
    r = subprocess.run(["git", "-C", str(clone), "worktree", "add", "--detach", str(dest), sha],
                       capture_output=True, text=True, timeout=1800)
    secs = round(time.time() - t0, 1)
    if r.returncode != 0:
        err = (r.stderr or r.stdout)[-150:].replace("\n", " ")
        return iid, f"failed: {err}"
    return iid, f"ok ({secs}s)"


def main() -> int:
    all_inst = panel_and_official()
    have = {d.name for d in REPOS.iterdir() if d.is_dir()} if REPOS.is_dir() else set()
    todo = {iid: rec for iid, rec in all_inst.items() if iid not in have}
    print(f"total universe {len(all_inst)} | have workspace {len(all_inst) - len(todo)} | todo {len(todo)}", flush=True)

    manifest: dict = json.loads(MANIFEST.read_text()) if MANIFEST.is_file() else {}
    results: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(add_worktree, iid, rec["repo"], rec["base_commit"]): iid
                for iid, rec in todo.items()}
        for fut in as_completed(futs):
            iid, status = fut.result()
            results[iid] = status
            if status == "exists" or status.startswith("ok"):
                rec = todo[iid]
                manifest[iid] = {"repo": rec["repo"], "commit": rec["base_commit"],
                                 "path": str(REPOS / iid),
                                 "status": "ok", "seconds": 0.0}
    MANIFEST.write_text(json.dumps(manifest, indent=2))

    ok = sum(1 for s in results.values() if s == "exists" or s.startswith("ok"))
    not_local = [i for i, s in results.items() if s == "commit_not_local"]
    print(f"DONE worktree ok/exists={ok} | commit_not_local={len(not_local)} | failed={len(results) - ok - len(not_local)}", flush=True)
    by_repo: dict[str, int] = {}
    for iid in not_local:
        by_repo[todo[iid]["repo"]] = by_repo.get(todo[iid]["repo"], 0) + 1
    for r, c in sorted(by_repo.items(), key=lambda x: -x[1]):
        print(f"  awaiting fetch {r}: {c}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
