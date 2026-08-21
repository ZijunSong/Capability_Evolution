from __future__ import annotations

import json
from pathlib import Path

import pytest

from easyopd.methods.scape_component_opd.event_collection import collect_event_states, state_uid
from easyopd.methods.scape_component_opd.harness1_bridge import QWEN3_LOGICAL_MODEL_ID, QWEN3_STUDENT_BASE


def make_state(seed: int, *, event: bool = True) -> dict:
    return {
        "query_id": "q0",
        "rollout_id": f"r{seed}",
        "rollout_seed": seed,
        "step_id": 1,
        "event_active": event,
        "event_type": "search",
        "student_visible_prefix": "same prefix",
        "tool_history": [{"name": "search_corpus"}],
        "student_observable_env_state": {"docs": ["d1"]},
        "projectable_target": {"name": "curate", "arguments": {"add_ids": ["d2"]}},
    }


def test_state_uid_excludes_rollout_seed() -> None:
    assert state_uid(component="token_budget_marker", state=make_state(1)) == state_uid(component="token_budget_marker", state=make_state(2))


def test_collector_rejects_duplicate_query_manifest(tmp_path: Path) -> None:
    query = tmp_path / "queries.json"
    query.write_text(json.dumps({"query_ids": ["q0", "q0"]}))
    rollout = tmp_path / "rollouts.jsonl"
    rollout.write_text(json.dumps(make_state(1)) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        collect_event_states(component="token_budget_marker", query_manifest=query, rollout_manifest=rollout, output_dir=tmp_path / "out", query_min=1, query_max=1)


def test_collector_blocks_when_support_is_short(tmp_path: Path) -> None:
    query = tmp_path / "queries.json"
    query.write_text(json.dumps({"query_ids": ["q0"]}))
    rollout = tmp_path / "rollouts.jsonl"
    rollout.write_text("\n".join(json.dumps({"query_id": "q0", "rollout_id": f"r{i}", "rollout_seed": i, "states": [make_state(i)]}) for i in (1, 2)) + "\n")
    stats = collect_event_states(component="token_budget_marker", query_manifest=query, rollout_manifest=rollout, output_dir=tmp_path / "out", query_min=1, query_max=1, target_unique_states=5)
    assert stats["collection_status"] == "INSUFFICIENT_5K_EVENT_SUPPORT"
    assert (tmp_path / "out" / "TRAIN_STATES_5K.jsonl").read_text() == ""
    assert stats["n_unique_event_active"] == 1


def test_inactive_states_are_not_promoted(tmp_path: Path) -> None:
    query = tmp_path / "queries.json"
    query.write_text(json.dumps({"query_ids": ["q0"]}))
    rollout = tmp_path / "rollouts.jsonl"
    rows = [{"query_id": "q0", "rollout_id": f"r{i}", "rollout_seed": i, "states": [make_state(i, event=False)]} for i in (1, 2)]
    rollout.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    stats = collect_event_states(component="token_budget_marker", query_manifest=query, rollout_manifest=rollout, output_dir=tmp_path / "out", query_min=1, query_max=1, target_unique_states=1)
    assert stats["n_event_active_raw"] == 0
    assert stats["n_unique_event_active"] == 0
    assert stats["runtime_name"] == "harness1"
    assert stats["model_id"] == QWEN3_LOGICAL_MODEL_ID
    assert stats["logical_model_id"] == QWEN3_LOGICAL_MODEL_ID
    assert stats["resolved_model_path"] == QWEN3_STUDENT_BASE

