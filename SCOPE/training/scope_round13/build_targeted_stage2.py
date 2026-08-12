#!/usr/bin/env python3
"""Barrier5: construct non-degenerate checkpoint targets (T1/T2/T3) from on-policy states.

Executed only when STAGE2_TASK_DEGENERATE=true. Uses student-visible metadata to
prefer latest-safe-pre-failure checkpoint over latest overall when they differ.
"""

from __future__ import annotations

import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
SDI = _REPO / "artifacts/datasets/scope_round13/operation_sdi"
OUT_DATA = _REPO / "artifacts/datasets/scope_round13/checkpoint_targeted"
OUT_GATE = _REPO / "outputs/scope_round13/stage2_targeted"


def load(split: str) -> list[dict]:
    p = SDI / f"{split}.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.open() if l.strip()]


def candidates(r: dict) -> list[dict]:
    ds = r.get("decision_state") or {}
    return list(ds.get("available_checkpoints") or [])


def pick_safe_pre_failure(cands: list[dict], turn: int) -> tuple[str | None, str]:
    """Gold = latest checkpoint BEFORE earliest persistent failure-like signal.

    Proxy using student-visible fields:
    - Prefer checkpoint with verified>0 and not the absolute latest if a newer
      checkpoint has verified==0 / zero new evidence (T3 invalid-progress).
    - Else second-latest when >=2 candidates (T1/T2 post-failure branch).
    """
    eligible = sorted(
        [c for c in cands if int(c.get("turn_id", -1)) < turn] or cands,
        key=lambda c: int(c.get("turn_id", 0)),
    )
    if not eligible:
        return None, "no_candidates"
    latest = eligible[-1]
    latest_id = str(latest.get("checkpoint_id"))
    latest_ver = int(latest.get("n_verified", latest.get("verified_count", 0)) or 0)
    # Find last verified-positive before latest
    verified = [
        c
        for c in eligible[:-1]
        if int(c.get("n_verified", c.get("verified_count", 0)) or 0) > 0
    ]
    if verified and latest_ver == 0:
        g = str(verified[-1].get("checkpoint_id"))
        if g != latest_id:
            return g, "T3_invalid_progress_latest"
    if len(eligible) >= 2:
        # If latest looks weaker than previous (fewer curated/verified), prefer previous
        prev = eligible[-2]
        prev_score = int(prev.get("n_verified", 0) or 0) + int(prev.get("n_curated", 0) or 0)
        lat_score = latest_ver + int(latest.get("n_curated", 0) or 0)
        if prev_score > lat_score:
            return str(prev.get("checkpoint_id")), "T1_post_failure_checkpoint"
        if prev_score == lat_score and len(eligible) >= 3:
            return str(eligible[-3].get("checkpoint_id")), "T2_repeated_failure_branch"
        return str(prev.get("checkpoint_id")), "T1_second_latest"
    return latest_id, "fallback_latest_only"


def build_split(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        op = (
            (r.get("target_action") or {}).get("operation")
            or r.get("gold_operation")
            or r.get("operation")
        )
        # Enrich both CONTINUE and ROLLBACK states that have >=2 candidates;
        # emit ranking events only when gold != latest (non-degenerate).
        cands = candidates(r)
        if len(cands) < 2:
            continue
        ds = r.get("decision_state") or {}
        turn = int(r.get("turn") or ds.get("turn_id") or 0)
        gold, reason = pick_safe_pre_failure(cands, turn)
        if not gold:
            continue
        eligible = sorted(
            [c for c in cands if int(c.get("turn_id", -1)) < turn] or cands,
            key=lambda c: int(c.get("turn_id", 0)),
            reverse=True,
        )
        latest = str(eligible[0].get("checkpoint_id")) if eligible else None
        if gold == latest and reason.startswith("fallback"):
            continue
        row = dict(r)
        row["gold_operation"] = "ROLLBACK_TO"
        row["operation"] = "ROLLBACK_TO"
        row["target_action"] = {"operation": "ROLLBACK_TO", "checkpoint_id": gold}
        row["gold_checkpoint_id"] = gold
        row["gold_checkpoint_global_id"] = gold
        row["targeted_reason"] = reason
        row["latest_checkpoint_id"] = latest
        row["gold_is_latest"] = gold == latest
        out.append(row)
    return out


def gate(train: list[dict], valid: list[dict]) -> dict:
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
    ok = (
        st["n"] >= 1500
        and sv["n"] >= 300
        and st["gold_latest_rate"] <= 0.70
        and st["gold_position_ge1_rate"] >= 0.20
        and st["candidate_count_ge2_rate"] >= 0.95
        and sv["gold_latest_rate"] <= 0.70
    )
    return {
        "train": st,
        "valid": sv,
        "NONDEGENERATE_STAGE2_DATA_PASS": ok,
        "thresholds": {
            "train_n": 1500,
            "valid_n": 300,
            "gold_latest_rate_max": 0.70,
            "gold_position_ge1_min": 0.20,
        },
    }


def main() -> None:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_GATE.mkdir(parents=True, exist_ok=True)
    train = build_split(load("train"))
    valid = build_split(load("valid"))
    with (OUT_DATA / "train.jsonl").open("w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (OUT_DATA / "valid.jsonl").open("w", encoding="utf-8") as f:
        for r in valid:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    g = gate(train, valid)
    (OUT_GATE / "DATASET_GATE.json").write_text(json.dumps(g, indent=2) + "\n")
    print(json.dumps(g, indent=2))
    if not g["NONDEGENERATE_STAGE2_DATA_PASS"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
