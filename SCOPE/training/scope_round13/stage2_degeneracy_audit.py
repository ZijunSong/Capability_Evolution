#!/usr/bin/env python3
"""Barrier4: Stage2 checkpoint-target degeneracy audit on R13 rollback-positive events."""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
RAW = _REPO / "artifacts/datasets/scope_round13/onpolicy_raw"
OUT = _REPO / "outputs/scope_round13/stage2_audit"
SDI = _REPO / "artifacts/datasets/scope_round13/operation_sdi"


def load_rows() -> tuple[list[dict], list[dict]]:
    def _load(split: str) -> list[dict]:
        p = SDI / f"{split}.jsonl"
        if p.exists():
            return [json.loads(l) for l in p.open() if l.strip()]
        rows = []
        root = RAW / split
        if root.exists():
            for f in sorted(root.glob("*/rollback_events.jsonl")):
                for line in f.open():
                    if line.strip():
                        rows.append(json.loads(line))
        return rows

    return _load("train"), _load("valid")


def rollback_rows(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        op = (
            (r.get("target_action") or {}).get("operation")
            or r.get("gold_operation")
            or r.get("operation")
        )
        if op == "ROLLBACK_TO":
            out.append(r)
    return out


def candidates(r: dict) -> list[dict]:
    ds = r.get("decision_state") or {}
    return list(ds.get("available_checkpoints") or r.get("candidate_list") or [])


def gold_ck(r: dict) -> str | None:
    ta = r.get("target_action") or {}
    ck = ta.get("checkpoint_id") or r.get("gold_checkpoint_id") or r.get("gold_checkpoint_global_id")
    return str(ck) if ck else None


def h0_latest(cands: list[dict], turn: int) -> str | None:
    eligible = [c for c in cands if int(c.get("turn_id", -1)) < turn] or list(cands)
    if not eligible:
        return None
    best = max(eligible, key=lambda c: int(c.get("turn_id", 0)))
    return str(best.get("checkpoint_id")) if best else None


def h1_latest_successful(cands: list[dict], turn: int) -> str | None:
    eligible = [
        c
        for c in cands
        if int(c.get("turn_id", -1)) < turn
        and int(c.get("n_verified", c.get("verified_count", 0)) or 0) > 0
    ]
    if not eligible:
        return h0_latest(cands, turn)
    best = max(eligible, key=lambda c: int(c.get("turn_id", 0)))
    return str(best.get("checkpoint_id"))


def h2_nearest_before_failure(cands: list[dict], turn: int) -> str | None:
    # proxy: second-latest eligible
    eligible = sorted(
        [c for c in cands if int(c.get("turn_id", -1)) < turn],
        key=lambda c: int(c.get("turn_id", 0)),
        reverse=True,
    )
    if len(eligible) >= 2:
        return str(eligible[1].get("checkpoint_id"))
    return h0_latest(cands, turn)


def h3_oldest(cands: list[dict], turn: int) -> str | None:
    eligible = [c for c in cands if int(c.get("turn_id", -1)) < turn] or list(cands)
    if not eligible:
        return None
    best = min(eligible, key=lambda c: int(c.get("turn_id", 0)))
    return str(best.get("checkpoint_id"))


def eval_heuristic(rows: list[dict], picker) -> float:
    correct = 0
    n = 0
    for r in rows:
        cands = candidates(r)
        gold = gold_ck(r)
        if not cands or not gold:
            continue
        ds = r.get("decision_state") or {}
        turn = int(r.get("turn") or ds.get("turn_id") or 0)
        pred = picker(cands, turn)
        n += 1
        correct += int(pred == gold)
    return correct / max(n, 1)


def audit(rows: list[dict], name: str) -> dict:
    rb = rollback_rows(rows)
    pos_latest = 0
    positions = []
    meta = []
    for r in rb:
        cands = candidates(r)
        gold = gold_ck(r)
        ds = r.get("decision_state") or {}
        turn = int(r.get("turn") or ds.get("turn_id") or 0)
        ids = [str(c.get("checkpoint_id")) for c in cands]
        latest = h0_latest(cands, turn)
        if gold and latest and gold == latest:
            pos_latest += 1
        if gold in ids:
            # position from latest among eligible
            eligible = sorted(
                [c for c in cands if int(c.get("turn_id", -1)) < turn] or cands,
                key=lambda c: int(c.get("turn_id", 0)),
                reverse=True,
            )
            eids = [str(c.get("checkpoint_id")) for c in eligible]
            if gold in eids:
                positions.append(eids.index(gold))
        meta.append(
            {
                "event_id": r.get("event_id"),
                "candidate_count": len(cands),
                "gold": gold,
                "latest_is_gold": gold == latest,
            }
        )
    pos_counts = Counter(positions)
    ent = 0.0
    npos = sum(pos_counts.values()) or 1
    for c in pos_counts.values():
        p = c / npos
        ent -= p * math.log(p + 1e-12, 2)

    rng = random.Random(1309)

    def h4(cands, turn):
        if not cands:
            return None
        return str(rng.choice(cands).get("checkpoint_id"))

    heur = {
        "H0_latest": eval_heuristic(rb, h0_latest),
        "H1_latest_successful": eval_heuristic(rb, h1_latest_successful),
        "H2_nearest_before_last_failure": eval_heuristic(rb, h2_nearest_before_failure),
        "H3_oldest": eval_heuristic(rb, h3_oldest),
        "H4_random": eval_heuristic(rb, h4),
    }
    p_latest0 = pos_latest / max(len(rb), 1)
    degenerate = (
        heur["H0_latest"] >= 0.95
        or heur["H1_latest_successful"] >= 0.95
        or p_latest0 >= 0.95
        or ent < 0.5
    )
    return {
        "split": name,
        "n_rollback": len(rb),
        "heuristics": heur,
        "P_gold_position_from_latest_0": p_latest0,
        "target_position_entropy_bits": ent,
        "position_hist": dict(pos_counts),
        "STAGE2_TASK_DEGENERATE": degenerate,
        "n_meta_recorded": len(meta),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train, valid = load_rows()
    tr = audit(train, "R13_TRAIN200")
    va = audit(valid, "R13_VALID100")
    degenerate = bool(tr["STAGE2_TASK_DEGENERATE"] and va["STAGE2_TASK_DEGENERATE"])
    report = {
        "train": tr,
        "valid": va,
        "STAGE2_TASK_DEGENERATE": degenerate,
        "decision": (
            "do not train ordinary Stage2 ranker; build non-degenerate target"
            if degenerate
            else "Stage2 task not classified degenerate under criteria"
        ),
    }
    (OUT / "STAGE2_DEGENERACY_AUDIT.json").write_text(json.dumps(report, indent=2) + "\n")
    md = [
        "# STAGE2_DEGENERACY_AUDIT\n",
        f"- TRAIN H0={tr['heuristics']['H0_latest']:.3f} entropy={tr['target_position_entropy_bits']:.3f}\n",
        f"- VALID H0={va['heuristics']['H0_latest']:.3f} entropy={va['target_position_entropy_bits']:.3f}\n",
        f"- STAGE2_TASK_DEGENERATE = {degenerate}\n",
    ]
    (OUT / "STAGE2_DEGENERACY_AUDIT.md").write_text("".join(md), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
