#!/usr/bin/env python3
"""Barrier2.1: A0 effective-input conflict / truncation audit on fresh on-policy events."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from training.scope_round11.stage1_views import build_stage1_view

_REPO = Path(__file__).resolve().parents[2]
RAW = _REPO / "artifacts/datasets/scope_round13/onpolicy_raw"
OUT = _REPO / "outputs/scope_round13/phase_a_shift"
BASE = "/data/ppnm/models/Qwen2.5-7B-Instruct"
MAX_LEN = 1536


def load_events(root: Path) -> list[dict]:
    rows: list[dict] = []
    if not root.exists():
        return rows
    for p in sorted(root.glob("*/rollback_events.jsonl")):
        for line in p.open(encoding="utf-8"):
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    events = load_events(RAW / "train") + load_events(RAW / "valid")

    by_hash: dict[str, list[dict]] = defaultdict(list)
    trunc_by_class: dict[str, list[int]] = defaultdict(list)
    n = 0
    for ev in events:
        sample = {
            "decision_state": ev.get("decision_state") or {},
            "student_state_text": ev.get("student_state_text") or "",
            "target_action": ev.get("target_action")
            or {"operation": ev.get("gold_operation"), "checkpoint_id": ev.get("gold_checkpoint_id")},
            "gold_operation": ev.get("gold_operation"),
            "operation": ev.get("gold_operation"),
        }
        view = build_stage1_view(sample, tok, "A0", max_length=MAX_LEN)
        h = view.prompt_sha256
        gold = view.gold_operation
        by_hash[h].append({"event_id": ev.get("event_id"), "gold": gold, "truncated": view.truncated})
        trunc_by_class[gold].append(int(view.truncated))
        n += 1

    conflict_groups = 0
    conflict_events = 0
    collision_groups = 0
    for h, group in by_hash.items():
        if len(group) <= 1:
            continue
        collision_groups += 1
        labels = {g["gold"] for g in group}
        if len(labels) > 1:
            conflict_groups += 1
            conflict_events += len(group)

    rate = conflict_events / max(n, 1)
    pass_gate = rate <= 0.01
    report = {
        "n_events": n,
        "n_unique_effective_inputs": len(by_hash),
        "exact_collision_groups": collision_groups,
        "conflicting_label_collision_groups": conflict_groups,
        "conflicting_label_event_rate": rate,
        "truncation_rate_by_class": {
            k: (sum(v) / max(len(v), 1)) for k, v in trunc_by_class.items()
        },
        "OPERATION_OBSERVABILITY_PASS": pass_gate,
        "gate": "conflicting_label_event_rate <= 0.01",
    }
    (OUT / "OPERATION_OBSERVABILITY.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    md = [
        "# OPERATION_OBSERVABILITY\n",
        f"- n_events = {n}\n",
        f"- n_unique_effective_inputs = {len(by_hash)}\n",
        f"- conflicting_label_event_rate = {rate:.6f}\n",
        f"- OPERATION_OBSERVABILITY_PASS = {pass_gate}\n",
    ]
    (OUT / "OPERATION_OBSERVABILITY.md").write_text("".join(md), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not pass_gate:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
