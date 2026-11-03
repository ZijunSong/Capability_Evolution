"""Clone the remaining 14 panel repos (full for multi-instance, shallow for 1-2)."""

from __future__ import annotations

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path("/mnt/songzijun/datasets/swebench")
CLONES = ROOT / "clones"

FULL = [  # >= 2 instances each — one full clone is cheaper than N depth-1 fetches
    "pandas-dev__pandas", "mesonbuild__meson", "numpy__numpy",
    "wagtail__wagtail", "pantsbuild__pants", "pyca__cryptography",
    "pypa__pip", "ytdl-org__youtube-dl", "explosion__spaCy",
    "celery__celery", "pydicom__pydicom",
]
# 1-2 instances, huge repos — shallow fetch per commit instead
SHALLOW = ["open-mmlab__mmdetection", "tensorflow__models", "twisted__twisted"]


def clone_full(repo: str) -> tuple[str, bool]:
    dest = CLONES / repo
    if (dest / "config").is_file():
        return repo, True
    t0 = time.time()
    CLONES.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["git", "clone", "--no-checkout", f"https://github.com/{repo.replace('__', '/')}.git", str(dest)],
        capture_output=True, text=True, timeout=14400,
    )
    secs = round(time.time() - t0, 1)
    ok = r.returncode == 0
    print(f"{repo}: {'ok' if ok else 'FAILED'} ({secs}s) {(r.stderr or '')[-150:] if not ok else ''}", flush=True)
    return repo, ok


def main() -> int:
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(clone_full, r) for r in FULL]
        fails = [repo for repo, ok in (f.result() for f in as_completed(futs)) if not ok]
    print(f"DONE full clones ok={len(FULL)-len(fails)}/{len(FULL)} failed={fails}", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
