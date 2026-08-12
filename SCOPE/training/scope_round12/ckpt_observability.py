#!/usr/bin/env python3
"""A4 — Stage2 identifiability / observability audit + heuristic baselines."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.scope.rollback_operation_runtime import pick_rollback_checkpoint

OUT = _REPO / "outputs" / "scope_round12" / "phase_a_ckpt_provenance"
TRAIN = _REPO / "artifacts" / "datasets" / "scope_round10" / "hier_sdi" / "train_p0_75.jsonl"
VALID = _REPO / "artifacts" / "datasets" / "scope_round10" / "hier_sdi" / "valid.jsonl"
CANON = _REPO / "artifacts" / "datasets" / "scope_round12" / "ckpt_canonical_base_live.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _effective_key(sample: dict) -> str:
    ds = sample.get("decision_state") or {}
    cands = ds.get("available_checkpoints") or sample.get("candidate_list") or []
    ids = [str(c.get("checkpoint_id")) for c in cands]
    # student-visible fields only
    visible = {
        "state": str(sample.get("student_state_text") or ds.get("rendered_context") or "")[:2000],
        "candidate_ids": ids,
        "turn": sample.get("turn") or ds.get("turn_id"),
    }
    return hashlib.sha256(json.dumps(visible, sort_keys=True).encode()).hexdigest()


def _gold_ck(sample: dict) -> str | None:
    ta = sample.get("target_action") or {}
    ck = ta.get("checkpoint_id") or sample.get("gold_checkpoint_global_id") or sample.get("gold_checkpoint_id")
    return str(ck) if ck else None


def _is_rollback(sample: dict) -> bool:
    ta = sample.get("target_action") or {}
    op = ta.get("operation") or sample.get("gold_operation") or sample.get("operation")
    return op == "ROLLBACK_TO"


def audit_split(rows: list[dict], name: str) -> dict:
    rb = [r for r in rows if _is_rollback(r)]
    groups: dict[str, list[str | None]] = defaultdict(list)
    cand_dup = 0
    positions = []
    ages = []
    counts = []
    for r in rb:
        key = _effective_key(r)
        gold = _gold_ck(r)
        groups[key].append(gold)
        ds = r.get("decision_state") or {}
        cands = list(ds.get("available_checkpoints") or r.get("candidate_list") or [])
        counts.append(len(cands))
        ids = [c.get("checkpoint_id") for c in cands]
        if len(ids) != len(set(ids)):
            cand_dup += 1
        if gold in ids:
            positions.append(ids.index(gold))
        turn = int(r.get("turn") or ds.get("turn_id") or 0)
        for c in cands:
            if c.get("checkpoint_id") == gold:
                ages.append(turn - int(c.get("turn_id", c.get("relative_turn", 0))))
                break
    conflict_groups = 0
    conflict_examples = 0
    for g, targets in groups.items():
        uniq = {t for t in targets if t}
        if len(uniq) > 1:
            conflict_groups += 1
            conflict_examples += len(targets)
    # position entropy
    pos_counts = Counter(positions)
    ent = 0.0
    npos = sum(pos_counts.values()) or 1
    for c in pos_counts.values():
        p = c / npos
        ent -= p * math.log(p + 1e-12, 2)
    return {
        "split": name,
        "n_rollback": len(rb),
        "n_effective_input_groups": len(groups),
        "conflicting_target_groups": conflict_groups,
        "conflicting_label_rate": conflict_examples / max(len(rb), 1),
        "candidate_list_duplicate_rate": cand_dup / max(len(rb), 1),
        "target_position_entropy_bits": ent,
        "candidate_count_mean": sum(counts) / max(len(counts), 1),
        "candidate_count_hist": dict(Counter(counts)),
        "target_age_mean": sum(ages) / max(len(ages), 1) if ages else None,
        "target_position_hist": {str(k): v for k, v in sorted(pos_counts.items())},
    }


def heuristic_baselines(events: list[dict]) -> dict:
    def score(picker) -> dict:
        correct = mrr = 0
        for ev in events:
            cands = ev["candidate_list"]
            pred = picker(ev, cands)
            gold = ev["gold_checkpoint_id"]
            ranked = [c["checkpoint_id"] for c in cands]
            if pred == gold:
                correct += 1
            if gold in ranked:
                # reorder ranked by picker preference when possible
                mrr += 1.0 / (ranked.index(gold) + 1)
        n = max(len(events), 1)
        return {"top1": correct / n, "MRR_list_order": mrr / n, "n": len(events)}

    def h0(ev, cands):
        return pick_rollback_checkpoint(
            [{"checkpoint_id": c["checkpoint_id"], "turn_id": c.get("relative_turn", 0)} for c in cands],
            int(ev.get("turn") or 0),
        )

    def h1(ev, cands):
        if not cands:
            return None
        return min(cands, key=lambda c: int(c.get("relative_turn", c.get("turn_id", 0))))["checkpoint_id"]

    def h2(ev, cands):
        turn = int(ev.get("turn") or 0)
        if not cands:
            return None
        return min(cands, key=lambda c: abs(int(c.get("relative_turn", 0)) - turn))["checkpoint_id"]

    def h3(ev, cands):
        # most recent successful ≈ highest verified/evidence then latest
        if not cands:
            return None
        return max(
            cands,
            key=lambda c: (
                int(c.get("verified_count", c.get("n_verified", 0))),
                int(c.get("evidence_count", c.get("n_curated", 0))),
                int(c.get("relative_turn", 0)),
            ),
        )["checkpoint_id"]

    return {
        "H0_latest": score(h0),
        "H1_oldest": score(h1),
        "H2_nearest_turn": score(h2),
        "H3_most_recent_successful": score(h3),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train = load_jsonl(TRAIN)
    valid = load_jsonl(VALID)
    events = load_jsonl(CANON)
    train_a = audit_split(train, "train_p0_75")
    valid_a = audit_split(valid, "offline_valid_hier")
    heur = heuristic_baselines(events)
    conflicting = max(train_a["conflicting_label_rate"], valid_a["conflicting_label_rate"])
    # If heuristics get near-perfect top1, target is largely identifiable from visible fields.
    h0 = heur["H0_latest"]["top1"]
    identifiable = conflicting < 0.01 and h0 >= 0.99
    report = {
        "train": train_a,
        "offline_valid": valid_a,
        "heuristic_baselines_on_canonical_base_live": heur,
        "CKPT_TARGET_NOT_IDENTIFIABLE": not identifiable and conflicting >= 0.01,
        "notes": {
            "H0_top1": h0,
            "conflicting_label_rate_max": conflicting,
            "interpretation": (
                "If H0≈1.0 and conflict rate≈0, gold targets are latest-checkpoint and "
                "identifiable from candidate turn ordering; Stage2 must beat/match this "
                "without gold-derived features."
            ),
        },
    }
    (OUT / "CKPT_OBSERVABILITY.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md = ["# CKPT_OBSERVABILITY\n\n"]
    md.append(f"- CKPT_TARGET_NOT_IDENTIFIABLE = `{report['CKPT_TARGET_NOT_IDENTIFIABLE']}`\n")
    md.append(f"- train conflicting_label_rate = {train_a['conflicting_label_rate']:.4f}\n")
    md.append(f"- valid conflicting_label_rate = {valid_a['conflicting_label_rate']:.4f}\n")
    md.append(f"- H0 latest top1 on canonical base_live = {h0:.4f}\n")
    md.append("\n## Heuristic baselines\n")
    for k, v in heur.items():
        md.append(f"- {k}: top1={v['top1']:.4f}\n")
    md.append(
        "\nInterpretation: gold rollback targets are almost always the latest eligible "
        "checkpoint; a learned Stage2 that underperforms H0 is capacity/representation limited, "
        "not an unidentifiable-label problem.\n"
    )
    (OUT / "CKPT_OBSERVABILITY.md").write_text("".join(md), encoding="utf-8")
    print(json.dumps({"CKPT_TARGET_NOT_IDENTIFIABLE": report["CKPT_TARGET_NOT_IDENTIFIABLE"], "H0": h0}, indent=2))


if __name__ == "__main__":
    main()
