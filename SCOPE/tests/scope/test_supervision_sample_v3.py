"""DecisionSupervisionSampleV3 schema and pipeline tests."""

from __future__ import annotations

import json
from pathlib import Path

from harness.capability.action_space import CapabilityAction, CapabilityActionType
from harness.shadow.evidence_shadow import EvidenceShadow
from harness.telemetry.writer import ScopeTelemetryWriter
from training.scope.pipeline import run_supervision_pipeline
from training.scope.schema import DecisionSupervisionSampleV3, Route, SCHEMA_VERSION

from tests.scope.conftest import make_state


def test_supervision_sample_v3_schema_round_trip():
    state = make_state()
    student = CapabilityAction(
        action_type=CapabilityActionType.CURATE_DOCUMENT,
        arguments={"add_ids": ["d1"]},
    )
    art = EvidenceShadow().analyze(state, student)
    result = run_supervision_pipeline(
        state,
        student,
        artifact=art,
        event_id="ev-case1",
    )
    sample = result.sample
    assert sample.schema_version == SCHEMA_VERSION
    assert sample.capability_id == "duplicate_evidence"
    assert sample.route in {Route.CORRECT, Route.ENDORSE, Route.IGNORE}
    assert "decision_state" in sample.to_dict()
    restored = DecisionSupervisionSampleV3.from_dict(sample.to_dict())
    assert restored.sample_hash() == sample.sample_hash()


def test_pipeline_emits_telemetry(tmp_path: Path):
    state = make_state()
    student = CapabilityAction(
        action_type=CapabilityActionType.CURATE_DOCUMENT,
        arguments={"add_ids": ["d2"]},
    )
    events_path = tmp_path / "events.jsonl"
    writer = ScopeTelemetryWriter(events_path)
    run_supervision_pipeline(
        state,
        student,
        shadow=EvidenceShadow(),
        telemetry=writer,
        event_id="ev-telemetry",
    )
    lines = events_path.read_text().strip().splitlines()
    assert lines
    ev = json.loads(lines[-1])
    assert ev["event"] == "supervision_sample_emitted"
    payload = ev["payload"]
    assert payload["capability_id"]
    assert payload["gate_results"]
    assert "train_mask" in payload
