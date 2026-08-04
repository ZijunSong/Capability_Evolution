"""Shared utilities for SCOPE Round 7."""

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

OUT = _REPO / "outputs/scope_round7"
R5 = _REPO / "outputs/scope_round5"
R6 = _REPO / "outputs/scope_round6"
BASE_MODEL = "/data/ppnm/models/Qwen2.5-7B-Instruct"
VALID522 = _REPO / "artifacts/datasets/dup_sdi_round3/valid.jsonl"
MANIFEST = _REPO / "artifacts/datasets/round2_audit_100q/query_manifest.json"
MERGED_ROOT = R5 / "merged"
SEEDS = (42, 43, 44)
HF_TOL = 1e-5
VLLM_TOL = 1e-2  # vLLM concurrent live vs cold replay; ops must still match


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def merged_path(seed: int) -> Path:
    return MERGED_ROOT / f"o7_r64_seed{seed}"


def write_marker(
    path: Path,
    *,
    status: str,
    expected_episodes: int,
    actual_episodes: int,
    n_trace_events: int,
    n_errors: int,
    telemetry_complete: bool,
    artifacts: dict[str, str] | None = None,
) -> None:
    write_json(
        path,
        {
            "status": status,
            "expected_episodes": expected_episodes,
            "actual_episodes": actual_episodes,
            "n_trace_events": n_trace_events,
            "n_errors": n_errors,
            "telemetry_complete": telemetry_complete,
            "artifacts": artifacts or {},
        },
    )
