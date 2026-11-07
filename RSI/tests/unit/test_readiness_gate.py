"""PR-05 (guide §12 test_readiness_gate): the formal-training gate.

A READY artifact must be invalidated by ANY identity change (bank/H0/policy/
environment/metric), and training must refuse without a READY artifact when
``require_teacher_readiness: true``.
"""

from __future__ import annotations

import json

import pytest

from rsi.opd.readiness import (
    TeacherReadiness,
    check_readiness_for_training,
    load_readiness,
)


def _ready(**overrides) -> TeacherReadiness:
    base = dict(
        domain="alfworld",
        status="READY_FOR_PILOT",
        code_commit="30580de",
        bank_digest="b" * 16,
        policy_digest="p" * 16,
        runtime_contract_digest="r" * 16,
        environment_digest="e" * 16,
        metric_contract_digest="m" * 16,
        confirm_manifest_digest="c" * 16,
        confirm_result_ref="runs/x/confirm_h/confirm_h_r0.json",
        paired_delta=0.25,
        paired_interval=(0.05, 0.45),
    )
    base.update(overrides)
    return TeacherReadiness(**base)


def test_ready_passes_gate_with_matching_identities():
    problems = check_readiness_for_training(
        readiness=_ready(),
        bank_digest="b" * 16,
        policy_digest="p" * 16,
        runtime_contract_digest="r" * 16,
        environment_digest="e" * 16,
    )
    assert problems == []


def test_any_identity_change_invalidates():
    current = dict(
        bank_digest="b" * 16,
        policy_digest="p" * 16,
        runtime_contract_digest="r" * 16,
        environment_digest="e" * 16,
    )
    for field in current:
        changed = dict(current)
        changed[field] = "9" * 16
        problems = check_readiness_for_training(readiness=_ready(), **changed)
        assert problems and field in problems[0], field


def test_empty_bound_field_is_incomplete():
    problems = check_readiness_for_training(
        readiness=_ready(environment_digest=""),
        bank_digest="b" * 16,
        policy_digest="p" * 16,
        runtime_contract_digest="r" * 16,
        environment_digest="e" * 16,
    )
    assert any("EMPTY environment_digest" in p for p in problems)


def test_non_ready_status_blocks():
    for status in ("UNMEASURED", "GAIN_CONFIRMED_ALIGNMENT_PENDING", "PARTIAL_SIGNAL_ONLY",
                   "NO_CONFIRMED_GAIN", "RUNTIME_INVALID", "ALIGNMENT_UNSUPPORTED"):
        problems = check_readiness_for_training(
            readiness=_ready(status=status),
            bank_digest="b" * 16,
            policy_digest="p" * 16,
            runtime_contract_digest="r" * 16,
            environment_digest="e" * 16,
        )
        assert any("not a READY state" in p for p in problems), status


def test_missing_artifact_blocks():
    problems = check_readiness_for_training(
        readiness=None,
        bank_digest="b" * 16,
        policy_digest="p" * 16,
        runtime_contract_digest="r" * 16,
        environment_digest="e" * 16,
    )
    assert problems == ["no teacher_readiness artifact found (require_teacher_readiness=true)"]


def test_invalid_status_and_roundtrip(tmp_path):
    with pytest.raises(ValueError):
        _ready(status="READY")
    p = _ready().save(tmp_path / "readiness.json")
    loaded = load_readiness(p)
    assert loaded.status == "READY_FOR_PILOT"
    assert loaded.identity_digest() == _ready().identity_digest()
    raw = json.loads(p.read_text())
    assert raw["schema_version"] == "teacher_readiness_v1"
    # the guide example's EMPTY evidence fields must not round-trip as READY
    raw2 = dict(raw, status="READY_FOR_FORMAL", paired_delta=None)
    p2 = tmp_path / "r2.json"
    p2.write_text(json.dumps(raw2))
    loaded2 = load_readiness(p2)
    problems = check_readiness_for_training(
        readiness=loaded2,
        bank_digest="b" * 16,
        policy_digest="p" * 16,
        runtime_contract_digest="r" * 16,
        environment_digest="e" * 16,
    )
    assert any("paired_delta" in p for p in problems)


def test_partial_signal_only_is_not_ready_for_formal():
    """guide §5.2: partial-only signals stay exploratory."""
    r = _ready(status="PARTIAL_SIGNAL_ONLY")
    assert not r.ready
