"""DecisionStateV2 schema and information-safety tests."""

from __future__ import annotations

from harness.artifacts.provenance import ProvenanceKind
from harness.capability.decision_state import DecisionStateV2, SCHEMA_VERSION
from harness.capability.state import DecisionState

from tests.scope.conftest import make_state


def test_decision_state_v2_alias_and_schema():
    s = make_state(goal="Who invented X?", event_id="ev-1")
    assert isinstance(s, DecisionStateV2)
    assert s.schema_version == SCHEMA_VERSION
    assert s.goal == "Who invented X?"
    assert s.event_id == "ev-1"


def test_decision_state_v2_observed_ids_alias():
    s = make_state()
    assert s.observed_ids == s.observation_ids


def test_decision_state_v2_core_hash_stable():
    s = make_state(last_action_arguments={"add_ids": ["d2"]})
    d = s.to_dict()
    s2 = DecisionState.from_dict(d)
    assert s2.core_state_hash() == s.core_state_hash()


def test_decision_state_v2_field_provenance():
    s = make_state()
    prov = s.field_provenance()
    assert prov["query"] == ProvenanceKind.OBSERVED.value
    assert prov["remaining_turns"] == ProvenanceKind.RUNTIME.value
    assert prov.get("gold_answer", ProvenanceKind.PRIVILEGED_FORBIDDEN.value) == (
        ProvenanceKind.PRIVILEGED_FORBIDDEN.value
    )


def test_decision_state_v2_info_safety():
    s = make_state()
    ok, bad = s.check_info_safety()
    assert ok
    assert bad == ()
