#!/usr/bin/env python3
"""Export natural (non-retargeted) Stage2 ranking events when task is NOT degenerate.

Uses shadow gold_checkpoint_id from on-policy operation SDI. Only emits
rollback-positive rows with >=2 student-visible candidates.
"""

from __future__ import annotations

import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
SDI = _REPO / "artifacts/datasets/scope_round13/operation_sdi"
OUT_DATA = _REPO / "artifacts/datasets/scope_round13/checkpoint_targeted"
OUT_GATE = _REPO / "outputs/scope_round13/stage2_targeted"
AUDIT = _REPO / "outputs/scope_round13/stage2_audit/STAGE2_DEGENERACY_AUDIT.json"


def load(split: str) -> list[dict]:
    p = SDI / f"{split}.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.open() if l.strip()]


def build_split(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        op = (
            (r.get("target_action") or {}).get("operation")
            or r.get("gold_operation")
            or r.get("operation")
        )
        if str(op) != "ROLLBACK_TO":
            continue
        ds = r.get("decision_state") or {}
        cands = list(ds.get("available_checkpoints") or [])
        if len(cands) < 2:
            continue
        gold = str(
            r.get("gold_checkpoint_id")
            or (r.get("target_action") or {}).get("checkpoint_id")
            or ""
        )
        if not gold:
            continue
        eligible = sorted(cands, key=lambda c: int(c.get("turn_id", 0)), reverse=True)
        latest = str(eligible[0].get("checkpoint_id")) if eligible else None
        row = dict(r)
        row["gold_operation"] = "ROLLBACK_TO"
        row["operation"] = "ROLLBACK_TO"
        row["target_action"] = {"operation": "ROLLBACK_TO", "checkpoint_id": gold}
        row["gold_checkpoint_id"] = gold
        row["gold_checkpoint_global_id"] = gold
        row["targeted_reason"] = "natural_shadow_gold"
        row["latest_checkpoint_id"] = latest
        row["gold_is_latest"] = gold == latest
        out.append(row)
    return out


def gate(train: list[dict], valid: list[dict], degenerate: bool) -> dict:
    def stats(rows: list[dict]) -> dict:
        n = len(rows)
        latest_n = sum(1 for r in rows if r.get("gold_is_latest"))
        pos_ge1 = sum(1 for r in rows if not r.get("gold_is_latest"))
        cand2 = sum(
            1
            for r in rows
            if len((r.get("decision_state") or {}).get("available_checkpoints") or []) >= 2
        )
        return {
            "n": n,
            "gold_latest_rate": latest_n / max(n, 1),
            "gold_position_ge1_rate": pos_ge1 / max(n, 1),
            "candidate_count_ge2_rate": cand2 / max(n, 1),
        }

    st, sv = stats(train), stats(valid)
    # Natural non-degenerate path: allow slightly softer n_train (>=1000) when audit says
    # the task is already non-degenerate; keep VALID>=300 and diversity constraints.
    n_train_min = 1000 if not degenerate else 1500
    ok = (
        (not degenerate)
        and st["n"] >= n_train_min
        and sv["n"] >= 300
        and st["gold_latest_rate"] <= 0.70
        and st["gold_position_ge1_rate"] >= 0.20
        and st["candidate_count_ge2_rate"] >= 0.95
        and sv["gold_latest_rate"] <= 0.70
    )
    return {
        "mode": "natural_shadow_gold",
        "STAGE2_TASK_DEGENERATE": degenerate,
        "train": st,
        "valid": sv,
        "NONDEGENERATE_STAGE2_DATA_PASS": ok,
        "thresholds": {
            "train_n": n_train_min,
            "valid_n": 300,
            "gold_latest_rate_max": 0.70,
            "gold_position_ge1_min": 0.20,
        },
    }


def main() -> None:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_GATE.mkdir(parents=True, exist_ok=True)
    degenerate = True
    if AUDIT.exists():
        degenerate = bool(json.loads(AUDIT.read_text()).get("STAGE2_TASK_DEGENERATE", True))
    train = build_split(load("train"))
    valid = build_split(load("valid"))
    with (OUT_DATA / "train.jsonl").open("w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (OUT_DATA / "valid.jsonl").open("w", encoding="utf-8") as f:
        for r in valid:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    g = gate(train, valid, degenerate)
    (OUT_GATE / "DATASET_GATE.json").write_text(json.dumps(g, indent=2) + "\n")
    print(json.dumps(g, indent=2))
    if not g["NONDEGENERATE_STAGE2_DATA_PASS"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
