"""Load state sources for Round 6 cross-score matrix."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from harness.capability.dup_operation import DupOperation
from harness.capability.state import DecisionState

from training.scope_round6.common import (
    B6_ROOT,
    VALID522,
    b6_variant_dir,
    load_jsonl,
    SEEDS,
)


def _shadow_label(ev: dict[str, Any]) -> str:
    shadow = str(ev.get("shadow_operation") or "").upper()
    if shadow in (DupOperation.KEEP_EVIDENCE.value, DupOperation.SKIP_DUPLICATE.value):
        return shadow
    if ev.get("candidate_is_duplicate"):
        return DupOperation.SKIP_DUPLICATE.value
    return DupOperation.KEEP_EVIDENCE.value


def load_valid522_samples() -> list[dict[str, Any]]:
    return load_jsonl(VALID522)


def load_b6_admission_states(variant: str) -> list[dict[str, Any]]:
    """Build admission-level samples from B6 closed-loop artifacts."""
    if variant == "base":
        root = b6_variant_dir(None)
        source_seed = None
    elif variant.startswith("o7_"):
        seed = int(variant.split("_")[1])
        root = b6_variant_dir(seed)
        source_seed = seed
    else:
        raise ValueError(f"unknown variant {variant}")

    samples: list[dict[str, Any]] = []
    for shard_dir in sorted(root.glob("shard*")):
        states_path = shard_dir / "decision_states.jsonl"
        events_path = shard_dir / "dup_admission_events.jsonl"
        if not states_path.exists() or not events_path.exists():
            continue
        states_by_key: dict[tuple[str, int], dict[str, Any]] = {}
        for row in load_jsonl(states_path):
            qid = str(row.get("query_id", ""))
            turn = int(row.get("turn_id", 0))
            states_by_key[(qid, turn)] = row

        for ev in load_jsonl(events_path):
            qid = str(ev.get("query_id", ""))
            turn = int(ev.get("turn_id", 0))
            st_row = states_by_key.get((qid, turn))
            if not st_row:
                continue
            ds = st_row.get("decision_state") or {}
            cid = str(ev.get("candidate_evidence_id", ""))
            label = _shadow_label(ev)
            samples.append(
                {
                    "sample_id": f"{variant}:{qid}:{turn}:{cid}",
                    "query_id": qid,
                    "turn_id": turn,
                    "decision_state": ds,
                    "student_state_text": ds.get("rendered_context") or ds.get("query") or "",
                    "target_action": {"operation": label},
                    "route": str(ev.get("route", "")).upper(),
                    "candidate_evidence_id": cid,
                    "state_source": variant,
                    "state_source_seed": source_seed,
                    "shadow_operation": label,
                    "student_operation": ev.get("student_operation"),
                    "candidate_is_duplicate": ev.get("candidate_is_duplicate"),
                }
            )
    return samples


def load_state_source(name: str) -> list[dict[str, Any]]:
    if name == "valid522":
        return load_valid522_samples()
    if name in ("base", "o7_42", "o7_43", "o7_44"):
        return load_b6_admission_states(name)
    raise ValueError(f"unknown state source {name}")


def decision_state_from_sample(sample: dict[str, Any]) -> DecisionState:
    ds = sample.get("decision_state") or {}
    return DecisionState.from_dict(ds)
