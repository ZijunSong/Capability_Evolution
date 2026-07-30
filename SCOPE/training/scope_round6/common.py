"""Shared utilities for SCOPE Round 6."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

OUT = _REPO / "outputs/scope_round6"
R5 = _REPO / "outputs/scope_round5"
BASE_MODEL = "/data/ppnm/models/Qwen2.5-7B-Instruct"
VALID522 = _REPO / "artifacts/datasets/dup_sdi_round3/valid.jsonl"
TRAIN1807 = _REPO / "artifacts/datasets/dup_sdi_round3/train.jsonl"
MANIFEST = _REPO / "artifacts/datasets/round2_audit_100q/query_manifest.json"
B6_ROOT = R5 / "closed_loop/b6_100q"
MERGED_ROOT = R5 / "merged"
ADAPTER_ROOT = R5 / "b4_full"

SEEDS = (42, 43, 44)
STATE_SOURCES = ("valid522", "base", "o7_42", "o7_43", "o7_44")
SCORER_TAGS = ("o7_42", "o7_43", "o7_44")


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_ids(ids: list[int]) -> str:
    return hashlib.sha256(",".join(str(x) for x in ids).encode()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def adapter_path(seed: int) -> Path:
    return ADAPTER_ROOT / f"o7_r64_seed{seed}" / "adapter"


def merged_path(seed: int) -> Path:
    return MERGED_ROOT / f"o7_r64_seed{seed}"


def b6_variant_dir(seed: int | None) -> Path:
    if seed is None:
        return B6_ROOT / "base"
    return B6_ROOT / f"best_o7_{seed}"


def seed_from_tag(tag: str) -> int:
    return int(tag.split("_")[1])


def scorer_tag(seed: int) -> str:
    return f"o7_{seed}"
