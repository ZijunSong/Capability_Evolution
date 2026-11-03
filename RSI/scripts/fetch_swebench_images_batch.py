#!/usr/bin/env python3
"""Batch-fetch swebench eval image tars (Verified ∪ Lite) from the Artifactory cache.

Reads instance->image from the enriched HF parquet, keeps the instances we can
score officially (verified_test + lite_test pools), and repacks each cached
image into a docker-save tar under /mnt/songzijun/datasets/swebench/images/.
Resumable: existing tars are skipped. Parallel via --workers.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path("/mnt/songzijun/datasets/swebench")
ENRICHED = ROOT / "enriched" / "swe_bench_test.parquet"
IMAGES = ROOT / "images"
FETCH = Path(__file__).resolve().parent / "fetch_swebench_image_from_cache.py"


def wanted_rows() -> list[dict]:
    te = pd.read_parquet(ENRICHED)
    verified = set(pd.read_parquet(ROOT / "verified_test.parquet")["instance_id"])
    lite = set(pd.read_parquet(ROOT / "lite_test.parquet")["instance_id"])
    ids = verified | lite
    rows = te[te["instance_id"].isin(ids)]
    out = []
    for _, r in rows.iterrows():
        img = r["image"]  # swebench/<folder>:<tag>
        folder = img.split("/")[-1].split(":")[0]
        out.append({"instance_id": r["instance_id"], "folder": folder, "repo_tag": img})
    return out


def one(row: dict) -> str:
    tar = IMAGES / f"{row['folder']}.tar"
    if tar.is_file() and tar.stat().st_size > 0:
        return f"skip {row['folder']}"
    last = ""
    for attempt in range(3):
        p = subprocess.run(
            [sys.executable, str(FETCH), "--image", row["folder"], "--repo-tag", row["repo_tag"],
             "--out-dir", str(IMAGES)],
            capture_output=True, text=True, timeout=3600,
        )
        if p.returncode == 0:
            return f"ok {row['folder']}"
        last = (p.stdout or p.stderr or f"rc={p.returncode} no-output")[-120:]
    return f"FAIL {row['folder']} :: {last}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    rows = wanted_rows()
    print(f"images to fetch: {len(rows)}", flush=True)
    ok = fail = 0
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        for line in ex.map(one, rows):
            print(line, flush=True)
            if line.startswith("FAIL"):
                fail += 1
            else:
                ok += 1
    print(f"DONE ok={ok} fail={fail}", flush=True)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
