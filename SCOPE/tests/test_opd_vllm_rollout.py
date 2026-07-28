"""Tests for vLLM rollout -> OPD transition builder (mock rollout)."""

from __future__ import annotations

from training.opd._policy_backend import MockRolloutBackend
from training.opd.rollout_worker import QueryRecord
from training.opd.shadow_harness import ShadowHarness
from training.opd.transition_builder import build_transitions_from_rollout
from harness.harness_config import config_path, load_harness_config


def test_build_transitions_from_mock_rollout():
    records = [QueryRecord(query_id="q1", query="Who discovered penicillin?")]
    shadow = ShadowHarness(load_harness_config(config_path("modules_full.yaml")), offline=True)
    transitions = build_transitions_from_rollout(
        MockRolloutBackend(),
        records,
        shadow,
        target_module="verification",
    )
    assert len(transitions) == 1
    assert transitions[0].query_id == "q1"
    assert transitions[0].action_ids == [10, 11, 12]
    assert transitions[0].metadata.get("rollout_backend") == "unknown"
