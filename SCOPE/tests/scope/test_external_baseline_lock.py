"""External baseline lock file tests."""

from __future__ import annotations

import pytest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
LOCK = _REPO / "experiments" / "baselines" / "BASELINE_LOCK.tsv"


def test_lock_file_exists_and_has_three_repos():
    assert LOCK.exists(), "BASELINE_LOCK.tsv missing — run clone step"
    lines = [
        ln.strip()
        for ln in LOCK.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    names = {ln.split("\t")[0] for ln in lines}
    assert {"SEED", "OPID", "SDAR"} <= names
    blocked = []
    for ln in lines:
        parts = ln.split("\t")
        assert len(parts) >= 3
        sha = parts[1]
        assert len(sha) >= 7
        if sha.startswith("BLOCKED"):
            blocked.append(parts[0])
    if blocked:
        pytest.skip(
            f"external baseline clone blocked on network for {blocked}; "
            "re-run scripts/iclr/clone_external_baselines.sh"
        )
