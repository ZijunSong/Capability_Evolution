#!/usr/bin/env python3
"""Bilateral Dup labeling from Base H_min_v2 decision states (Round 3)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from harness.artifacts.gates import run_information_safe_gates
from harness.capability.action_space import CapabilityAction
from harness.capability.dup_operation import DupOperation
from harness.capability.state import DecisionState
from harness.shadow.dup_bilateral_shadow import DupBilateralShadow
from training.scope.compact_target import apply_compact_target_to_sample
from training.scope.routing import route_decision
from training.scope.schema import Route


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def label_shard(states_path: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    shadow = DupBilateralShadow()
    samples_out = out_dir / "samples.jsonl"
    events_out = out_dir / "events.jsonl"
    n_keep = n_skip = n_endorse = n_correct = n_ignore = 0
    visibility_violations = 0
    shadow_mutation = 0
    schema_invalid = 0
    query_ids: set[str] = set()

    with samples_out.open("w", encoding="utf-8") as fout, events_out.open(
        "w", encoding="utf-8"
    ) as fevt:
        for row in load_jsonl(states_path):
            qid = str(row.get("query_id", ""))
            ds_raw = row.get("decision_state") or {}
            student_raw = row.get("student_action") or {}
            if not ds_raw or not student_raw:
                continue
            state = DecisionState.from_dict(ds_raw)
            student = CapabilityAction.from_dict(student_raw)
            if student.action_type.value != "curate_document":
                continue
            query_ids.add(qid)
            for artifact in shadow.analyze_all_candidates(state, student):
                gates = run_information_safe_gates(state, artifact)
                if not gates.visible:
                    visibility_violations += 1
                    continue
                if not gates.schema_valid:
                    schema_invalid += 1
                    continue
                if not gates.purity_ok:
                    shadow_mutation += 1
                    continue
                routed = route_decision(state, artifact, student)
                if routed.route == Route.IGNORE:
                    n_ignore += 1
                    continue
                sample = routed.sample.to_dict()
                meta = sample.get("metadata") or {}
                shadow_op = str(
                    (artifact.metadata or {}).get("shadow_operation", "")
                )
                op = (
                    DupOperation(shadow_op)
                    if shadow_op
                    else (
                        DupOperation.SKIP_DUPLICATE
                        if routed.route == Route.CORRECT
                        else DupOperation.KEEP_EVIDENCE
                    )
                )
                sample["target_action"] = {"operation": op.value}
                dp = (artifact.metadata or {}).get("decision_point") or {}
                meta["decision_point"] = dp
                meta["shadow_operation"] = op.value
                sample["metadata"] = meta
                sample = apply_compact_target_to_sample(sample)
                if routed.route == Route.ENDORSE:
                    n_endorse += 1
                else:
                    n_correct += 1
                if op == DupOperation.KEEP_EVIDENCE:
                    n_keep += 1
                else:
                    n_skip += 1
                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                fevt.write(
                    json.dumps(
                        {
                            "query_id": qid,
                            "turn_id": state.turn_id,
                            "candidate_evidence_id": dp.get("candidate_evidence_id"),
                            "shadow_operation": op.value,
                            "route": routed.route.value,
                            "candidate_is_duplicate": (artifact.metadata or {}).get(
                                "candidate_is_duplicate"
                            ),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    stats = {
        "n_keep": n_keep,
        "n_skip": n_skip,
        "KEEP_EVIDENCE": n_keep,
        "SKIP_DUPLICATE": n_skip,
        "ENDORSE": n_endorse,
        "CORRECT": n_correct,
        "IGNORE": n_ignore,
        "visibility_violation": visibility_violations,
        "visibility_violations": visibility_violations,
        "shadow_mutation": shadow_mutation,
        "schema_invalid": schema_invalid,
        "n_queries": len(query_ids),
        "n_samples": n_keep + n_skip,
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--states", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    stats = label_shard(args.states, args.output_dir)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
