#!/usr/bin/env python3
"""Targeted valid-stop audit: synthetic evidence-sufficient stops through v3 pipeline.

NOT training data — verifies SCOPE ENDORSEs stop when it should.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import yaml

from harness.capability.action_space import CapabilityAction, CapabilityActionType
from harness.capability.state import ClaimState, DecisionState, VerificationRecordState
from harness.shadow.verification_shadow import VerificationShadow
from training.scope.audit_analytics import build_formal_audit_report, enrich_event_local_gt
from training.scope.pipeline import run_supervision_pipeline
from training.scope.schema import Route


def _make_valid_stop_state(
    *,
    idx: int,
    turn_id: int,
    n_curated: int,
    remaining_turns: int,
) -> DecisionState:
    docs = tuple(f"d{idx}_{j}" for j in range(n_curated))
    return DecisionState(
        episode_id=f"targeted_valid_stop_{idx}",
        task_id=f"targeted_{idx}",
        turn_id=turn_id,
        event_id=f"targeted:{idx}:{turn_id}",
        query=f"Synthetic valid-stop probe query {idx}?",
        goal=f"Synthetic valid-stop probe query {idx}?",
        rendered_context=f"Evidence for query {idx}: " + " ".join(docs),
        action_history=(),
        observation_ids=(f"obs_{idx}",),
        visible_document_ids=docs,
        pool_document_ids=docs + (f"pool_{idx}",),
        curated_document_ids=docs,
        evidence_claims=(
            ClaimState(
                claim_id=f"c{idx}",
                text=f"Answer for query {idx} is supported.",
                status="supported",
                supporting_document_ids=docs[:1],
            ),
        ),
        verification_records=(
            VerificationRecordState(
                turn_id=max(1, turn_id - 2),
                claim=f"Answer for query {idx} is supported.",
                document_ids=docs[:1],
                judgments={docs[0]: True},
            ),
        ),
        remaining_turns=remaining_turns,
        remaining_search_calls=10,
        token_budget_used=1000,
        token_budget_total=32768,
        last_action_type="verify_claim",
        repeated_query_score=0.0,
        wm_snapshot_hash=f"h{idx}",
    )


def _variants(n: int) -> list[dict[str, Any]]:
    specs = []
    for i in range(n):
        specs.append(
            {
                "idx": i,
                "turn_id": 8 + (i % 20),
                "n_curated": 1 + (i % 4),
                "remaining_turns": 3 + (i % 25),
            }
        )
    return specs


def run_targeted_audit(out_dir: Path, n: int = 24) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    shadow = VerificationShadow()
    events: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []

    for spec in _variants(n):
        state = _make_valid_stop_state(
            idx=spec["idx"],
            turn_id=spec["turn_id"],
            n_curated=spec["n_curated"],
            remaining_turns=spec["remaining_turns"],
        )
        student = CapabilityAction(
            action_type=CapabilityActionType.STOP_AND_ANSWER,
            arguments={"answer": f"synthetic answer {spec['idx']}", "targeted_probe": True},
        )
        artifact = shadow.analyze(state, student)
        result = run_supervision_pipeline(
            state,
            student,
            artifact=artifact,
            event_id=state.event_id,
            enforce_round1_capability_filter=True,
        )
        sample = result.sample
        sample_dict = sample.to_dict()
        sample_dict["train_mask"] = 0
        sample_dict["metadata"] = {
            **dict(sample_dict.get("metadata") or {}),
            "targeted_probe": True,
            "not_for_training": True,
            "probe_kind": "valid_stop",
        }
        ev = enrich_event_local_gt(
            {
                "event": "supervision_sample_emitted",
                "episode_id": state.episode_id,
                "turn_id": state.turn_id,
                "event_id": state.event_id,
                "task_id": state.task_id,
                "module_id": "verification",
                "capability_id": result.artifact.resolved_capability().value,
                "decision_state": state.to_dict(),
                "student_action_struct": student.to_dict(),
                "artifact": result.artifact.to_dict(),
                "gate_results": result.routing.gates.to_dict(),
                "route": result.routing.route.value,
                "train_mask": 0,
                "targeted_probe": True,
                "probe_spec": spec,
            }
        )
        events.append(ev)
        samples.append(sample_dict)

    events_path = out_dir / "events.jsonl"
    samples_path = out_dir / "samples.jsonl"
    events_path.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in events) + "\n",
        encoding="utf-8",
    )
    samples_path.write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in samples) + "\n",
        encoding="utf-8",
    )
    (out_dir / "errors.jsonl").write_text("", encoding="utf-8")

    n_endorse = sum(1 for e in events if e.get("route") == "ENDORSE")
    n_correct = sum(1 for e in events if e.get("route") == "CORRECT")
    base = {
        "mode": "targeted_valid_stop",
        "n_probes": n,
        "n_events": len(events),
        "n_endorse": n_endorse,
        "n_correct": n_correct,
        "n_ignore": len(events) - n_endorse - n_correct,
        "endorse_rate": n_endorse / max(1, len(events)),
        "n_trainable_samples": 0,
        "Dup": {"calls": 0, "endorse": 0, "correct": 0, "ignore": 0},
        "Premature": {
            "calls": sum(1 for e in events if e.get("capability_id") == "premature_stop"),
            "endorse": n_endorse,
            "correct": n_correct,
            "ignore": sum(1 for e in events if e.get("route") == "IGNORE"),
        },
    }
    summary = build_formal_audit_report(events, n_queries=0, base_summary=base)
    summary["targeted_valid_stop"] = {
        "n_probes": n,
        "n_endorse": n_endorse,
        "n_correct": n_correct,
        "pass": n_endorse >= max(5, n // 2),
        "note": "Synthetic evidence-sufficient stops; NOT for training.",
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(
            {
                "audit_mode": "targeted_valid_stop",
                "n_probes": n,
                "harness_config": "N/A (synthetic states)",
                "scope_config": "configs/scope/sdi_dup_premature.yaml",
                "train_mask": 0,
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--output-dir",
        default="outputs/scope_v3_audit_100q/targeted_valid_stop",
    )
    ap.add_argument("--n-probes", type=int, default=24)
    args = ap.parse_args()
    summary = run_targeted_audit(Path(args.output_dir), n=args.n_probes)
    print(json.dumps(summary.get("targeted_valid_stop", summary), indent=2))


if __name__ == "__main__":
    main()
