#!/usr/bin/env python3
"""Dup same-state shadow labeling from Base H_min_v2 DecisionStates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from harness.capability.action_space import CapabilityAction
from harness.capability.state import DecisionState
from harness.shadow.evidence_shadow import EvidenceShadow
from harness.capability.dup_operation import DupOperation
from training.scope.compact_target import apply_compact_target_to_sample
from training.scope.pipeline import run_supervision_pipeline
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
    shadow = EvidenceShadow()
    samples_out = out_dir / "samples.jsonl"
    n_keep = n_skip = n_endorse = n_correct = n_ignore = 0
    violations = 0

    with samples_out.open("w", encoding="utf-8") as fout:
        for row in load_jsonl(states_path):
            ds_raw = row.get("decision_state") or {}
            student_raw = row.get("student_action") or {}
            if not ds_raw or not student_raw:
                continue
            state = DecisionState.from_dict(ds_raw)
            student = CapabilityAction.from_dict(student_raw)
            if student.action_type.value != "curate_document":
                continue
            artifact = shadow.analyze(state, student)
            if artifact.reason_code != "DUPLICATE_EVIDENCE":
                continue
            routed = route_decision(state, artifact, student)
            if routed.route == Route.IGNORE:
                n_ignore += 1
                continue
            sample = routed.sample.to_dict()
            op = (
                DupOperation.SKIP_DUPLICATE
                if routed.route == Route.CORRECT
                else DupOperation.KEEP_EVIDENCE
            )
            sample["target_action"] = {"operation": op.value}
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

    stats = {
        "n_keep": n_keep,
        "n_skip": n_skip,
        "ENDORSE": n_endorse,
        "CORRECT": n_correct,
        "IGNORE": n_ignore,
        "visibility_violations": violations,
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
