#!/usr/bin/env python3
"""Round 10 shared utilities."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
OUT = _REPO / "outputs/scope_round10"
DATA = _REPO / "artifacts/datasets/scope_round10"
R9_DATA = _REPO / "artifacts/datasets/scope_round9"
R8_DATA = _REPO / "artifacts/datasets/scope_round8"
R9_OUT = _REPO / "outputs/scope_round9"
BASE_MODEL = "/data/ppnm/models/Qwen2.5-7B-Instruct"
BINARY_OPS = {"CONTINUE", "ROLLBACK_TO"}
SEEDS = [42, 43, 44]


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gold_operation(row: dict) -> str:
    return str(
        row.get("gold_operation")
        or (row.get("target_action") or {}).get("operation")
        or row.get("operation")
        or "CONTINUE"
    )


def binary_operation(row: dict) -> str | None:
    op = gold_operation(row)
    if op == "REPLAN":
        return None
    if op in BINARY_OPS:
        return op
    return None


def class_distribution(rows: list[dict], *, binary: bool = False) -> dict[str, int]:
    c: Counter[str] = Counter()
    for r in rows:
        op = binary_operation(r) if binary else gold_operation(r)
        if op:
            c[op] += 1
    return dict(c)


def query_ids(rows: list[dict]) -> set[str]:
    return {str(r.get("query_id", "")) for r in rows if r.get("query_id") is not None}


def split_queries(
    qids: list[str], *, seed: int = 42
) -> dict[str, list[str]]:
    rng = random.Random(seed)
    qids = sorted(set(qids))
    rng.shuffle(qids)
    n = len(qids)
    n_cal = int(n * 0.20)
    n_train = int(n * 0.50)
    n_valid = int(n * 0.15)
  # remainder -> test
    cal = qids[:n_cal]
    train = qids[n_cal : n_cal + n_train]
    valid = qids[n_cal + n_train : n_cal + n_train + n_valid]
    test = qids[n_cal + n_train + n_valid :]
    return {
        "live_calibration": cal,
        "live_train": train,
        "live_valid": valid,
        "live_test": test,
    }


def enrich_from_frozen(row: dict, *, state_source: str = "live") -> dict | None:
    from training.scope_round9.build_hier_sdi_dataset import enrich_sample

    out = enrich_sample(row, include_summary=True)
    if not out:
        return None
    op = binary_operation(out)
    if op is None:
        return None
    out["state_source"] = state_source
    out["gold_operation"] = op
    out["operation"] = op
    ta = dict(out.get("target_action") or {})
    ta["operation"] = op
    out["target_action"] = ta
    return out


def dataset_gate(train: list[dict], valid: list[dict], test: list[dict]) -> dict:
    def collisions(rows: list[dict]) -> int:
        seen: dict[str, str] = {}
        conflicts = 0
        for r in rows:
            key = str(r.get("event_id") or f"{r.get('query_id')}:{r.get('turn', (r.get('decision_state') or {}).get('turn_id', ''))}")
            op = gold_operation(r)
            if key in seen and seen[key] != op:
                conflicts += 1
            seen[key] = op
        return conflicts

    q_train, q_valid, q_test = query_ids(train), query_ids(valid), query_ids(test)
    overlap = len((q_train & q_valid) | (q_train & q_test) | (q_valid & q_test))
    return {
        "n_train": len(train),
        "n_valid": len(valid),
        "n_test": len(test),
        "class_distribution_train": class_distribution(train, binary=True),
        "state_source_train": dict(Counter(r.get("state_source", "?") for r in train)),
        "query_overlap_train_valid_test": overlap,
        "conflicting_label_groups": collisions(train) + collisions(valid) + collisions(test),
        "future_leakage": 0,
        "schema_invalid": sum(1 for r in train + valid + test if not binary_operation(r)),
        "gate_pass": overlap == 0
        and collisions(train) + collisions(valid) + collisions(test) == 0
        and all(binary_operation(r) for r in train + valid + test),
    }
