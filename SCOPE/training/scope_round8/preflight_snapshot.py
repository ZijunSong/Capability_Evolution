#!/usr/bin/env python3
"""Round 8 preflight environment snapshot."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
OUT = _REPO / "outputs/scope_round8/preflight"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    snap = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True
        ).strip(),
        "git_branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=_REPO, text=True
        ).strip(),
        "nvidia_smi": subprocess.check_output(["nvidia-smi", "-L"], text=True),
    }
    (OUT / "preflight_snapshot.json").write_text(json.dumps(snap, indent=2) + "\n")
    print(json.dumps(snap, indent=2))


if __name__ == "__main__":
    main()
