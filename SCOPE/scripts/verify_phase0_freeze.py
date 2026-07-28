#!/usr/bin/env python3
"""Verify Phase-0 frozen baseline files have not been overwritten.

Exit 0 if all protected paths match phase0_freeze/SHA256SUMS.
Exit 1 on mismatch or missing files.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

_REPO_SCOPE = Path(__file__).resolve().parents[1]
_FREEZE = _REPO_SCOPE / "artifacts" / "baselines" / "phase0_freeze"
_SUMS = _FREEZE / "SHA256SUMS"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if not _SUMS.exists():
        print(f"MISSING {_SUMS}", file=sys.stderr)
        return 1

    ok = True
    for line in _SUMS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, rel = line.split(maxsplit=1)
        # Paths in SUMS are relative to SCOPE/
        path = _REPO_SCOPE / rel
        if not path.exists():
            print(f"MISSING {rel}")
            ok = False
            continue
        got = sha256(path)
        if got != digest:
            print(f"CHANGED {rel}")
            print(f"  expected {digest}")
            print(f"  got      {got}")
            ok = False
        else:
            print(f"OK {rel}")

    if ok:
        print("Phase-0 freeze intact.")
        return 0
    print("Phase-0 freeze BROKEN — restore from tag scope-phase0-freeze / phase0_freeze/.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
