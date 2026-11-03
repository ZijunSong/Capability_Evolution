"""Prepare SWE-bench repo workspaces for the frozen panels (resumable).

Scoped to the panel instances (development pool + final/transfer panels).
One partial clone (``--filter=blob:none``) per unique repo is shared; each
``(repo, base_commit)`` becomes a worktree under ``repos/<instance_id>``.
Gated follow-up: official scoring still needs the pinned eval images/docker.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path("/mnt/songzijun/datasets/swebench")
REPOS = ROOT / "repos"
CLONES = ROOT / "clones"
MANIFEST = ROOT / "workspaces_manifest.json"

PANEL_ROLES = {  # must mirror the provider's frozen panel slicing
    "work_train": 64,
    "harness_select": 32,
    "weight_dev": 32,
    "confirm_h": 16,
    "confirm_w": 16,
    "final_test": 32,
    "transfer_test": 16,
}


def sh(cmd: list[str], cwd: Path | None = None, timeout: int = 3600) -> int:
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        print(f"  FAIL: {' '.join(cmd)[:110]} :: {(proc.stderr or proc.stdout)[-200:]}", flush=True)
    return proc.returncode


def panel_instances() -> dict[str, dict[str, Any]]:
    """The exact panel the task provider freezes at runtime.

    2026-11-02 fix: the old standalone re-implementation hashed the bare
    ``instance_id`` while ``SWEBenchTaskProvider`` hashes ``"swebench:<iid>"``
    (the task_id) — different hash inputs produced a different 208-instance
    selection (only 1/208 overlapped the frozen split), so prepared workspaces
    were useless at runtime.  Select through the provider itself instead of
    re-implementing the slice, so prep and runtime can never diverge again.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from rsi.environments.task_providers import SWEBenchTaskProvider

    provider = SWEBenchTaskProvider()
    selected: dict[str, dict[str, Any]] = {}
    for role in PANEL_ROLES:
        for task in provider.public_tasks(role):
            iid = str(task.metadata["instance_id"])
            selected[iid] = {
                "repo": str(task.metadata["repo"]),
                "base_commit": str(task.metadata["base_commit"]),
            }
    return selected


def official_instances() -> dict[str, dict[str, Any]]:
    """Verified ∪ Lite — the officially scorable universe (specs + prebuilt images).

    2026-11-02 scope fix: the dev-pool roles draw from princeton *train*, which has
    no official eval specs or prebuilt images.  The eval scope is the official
    test-family sets instead (see decision in the SWE-bench eval plan).
    """
    import pandas as pd

    te = pd.read_parquet(ROOT / "enriched" / "swe_bench_test.parquet")
    verified = set(pd.read_parquet(ROOT / "verified_test.parquet")["instance_id"])
    lite = set(pd.read_parquet(ROOT / "lite_test.parquet")["instance_id"])
    ids = verified | lite
    rows = te[te["instance_id"].isin(ids)]
    return {
        str(r["instance_id"]): {"repo": str(r["repo"]), "base_commit": str(r["base_commit"])}
        for _, r in rows.iterrows()
    }


def prep_repo(repo: str, iids: list[str], selected: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    clone = CLONES / repo.replace("/", "__")
    if not clone.exists():
        CLONES.mkdir(parents=True, exist_ok=True)
        print(f"cloning {repo} (blob:none)...", flush=True)
        if sh(
            ["git", "clone", "--filter=blob:none", "--no-checkout", f"https://github.com/{repo}.git", str(clone)],
            timeout=7200,
        ) != 0:
            return {i: {"status": "clone_failed"} for i in iids}
    for iid in sorted(iids):
        commit = str(selected[iid]["base_commit"])
        dest = REPOS / iid
        if dest.exists():
            out[iid] = {"repo": repo, "commit": commit, "path": str(dest), "status": "exists"}
            continue
        t0 = time.time()
        ok = sh(["git", "-C", str(clone), "worktree", "add", "--detach", str(dest), commit]) == 0
        out[iid] = {
            "repo": repo,
            "commit": commit,
            "path": str(dest),
            "status": "ok" if ok else "failed",
            "seconds": round(time.time() - t0, 1),
        }
        print(f"  {iid}: {'ok' if ok else 'FAILED'} ({out[iid]['seconds']}s)", flush=True)
    return out


def main() -> int:
    import argparse
    import concurrent.futures as cf

    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=("panel", "official"), default="official",
                    help="official = Verified∪Lite (default; the eval universe); panel = provider role slices")
    ap.add_argument("--repo-workers", type=int, default=6)
    a = ap.parse_args()
    selected = official_instances() if a.scope == "official" else panel_instances()
    by_repo: dict[str, list[str]] = {}
    for iid, r in selected.items():
        by_repo.setdefault(str(r["repo"]), []).append(iid)
    print(f"scope={a.scope}: instances {len(selected)} | repos {len(by_repo)}", flush=True)
    manifest: dict[str, Any] = {}
    with cf.ThreadPoolExecutor(max_workers=a.repo_workers) as ex:
        futs = {ex.submit(prep_repo, repo, iids, selected): repo for repo, iids in by_repo.items()}
        for fut in cf.as_completed(futs):
            manifest.update(fut.result())
            MANIFEST.write_text(json.dumps(manifest, indent=2))
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    okc = sum(1 for v in manifest.values() if v["status"] in ("ok", "exists"))
    print(f"DONE {okc}/{len(selected)} workspaces -> {MANIFEST}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
