"""Shared fixtures for Probe-C acceptance tests (CPU-only, no GPU)."""

from __future__ import annotations

from typing import Any

from trim.probes.probe_c_data import PROBE_C_STATE_SCHEMA_VERSION, env_content_hash
from trim.state.snapshot import stable_hash


def make_state_row(
    *,
    query_id: str = "q1",
    split: str = "C_train",
    scenario_group_id: str = "",
    source: str = "real_h0_rollout",
    student_mask: dict[str, bool] | None = None,
    teacher_mask: dict[str, bool] | None = None,
    query_text: str = "what is the capital of france",
    docs: list[dict[str, str]] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    snapshot = {
        "query_id": query_id,
        "query_text": query_text,
        "step": 2,
        "created_at_step": 2,
        "harness_mask": dict(student_mask or {}),
        "working_memory": {
            "query": query_text,
            "documents": docs
            if docs is not None
            else [
                {"id": "d1", "text": "paris is the capital of france " * 3},
                {"id": "d2", "text": "lyon is a city in france " * 3},
            ],
            "curated_ids": ["d1"],
            "tool_history": [{"name": "search_corpus", "arguments": {"query": "capital"}}],
        },
        "tool_history": [{"name": "search_corpus", "arguments": {"query": "capital"}}],
        "observations": [],
        "metadata": {"owner": "student_reduced"},
    }
    row: dict[str, Any] = {
        "schema_version": PROBE_C_STATE_SCHEMA_VERSION,
        "source": source,
        "model_id": "Qwen/Qwen3.5-9B",
        "model_revision": "rev-9b-001",
        "policy_checksum": "pol-" + stable_hash({"q": query_id})[:8],
        "adapter_checksum": "",
        "query_id": query_id,
        "query_text": query_text,
        "scenario_group_id": scenario_group_id or f"grp-{query_id}",
        "split": split,
        "rollout_seed": 7,
        "turn_id": 2,
        "student_snapshot": snapshot,
        "env_content_hash": env_content_hash(snapshot),
        "snapshot_hash": stable_hash(snapshot),
        "env_restore_version": "env-v1",
        "raw_messages": [{"role": "user", "content": query_text}],
        "effective_student_prompt_ids": [1, 2, 3, 4, 5],
        "effective_tool_schema": {"tools": ["search_corpus", "read_document", "curate"]},
        "student_action_ids": [10, 11],
        "sampled_old_logprobs": [-0.1, -0.2],
        "original_action_parse_result": {"ok": True, "name": "search_corpus"},
        "student_mask": dict(student_mask or {}),
        "teacher_mask": dict(teacher_mask or {}),
        "component_revision": "comp-rev-001",
        "information_provenance": [{"source": "student_rollout", "owner": "student_reduced"}],
        "tokenizer_hash": "tok-1",
        "chat_template_hash": "ct-1",
        "enable_thinking": False,
        "truncation_policy_hash": "trunc-1",
    }
    row.update(overrides)
    if "student_snapshot" in overrides:
        snap = overrides["student_snapshot"]
        row["env_content_hash"] = env_content_hash(snap)
    return row
