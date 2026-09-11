"""Regression tests for recovery4 audit P0 fixes."""

from __future__ import annotations

from trim.adapters.components import minus_mask
from trim.state.snapshot import capture_snapshot
from trim.training.opd_events import harness_mutation
from trim.training.opd_prompt_encoding import encode_rollout_style_action, render_rollout_action_text
from trim.training.rl_opd_types import HybridRolloutGroup, StudentDecisionPoint
from trim.training.tinker_rl_opd_trainer import project_on_policy_decisions


class _FakeEnc:
    family = "qwen3"

    def encode(self, text: str) -> list[int]:
        return [1000 + (sum(map(ord, text)) % 500)]


def _snap():
    return capture_snapshot(
        query_id="q0",
        step=1,
        harness_mask=minus_mask("auto_populate_first_search"),
        working_memory={
            "curated_ids": ["d1"],
            "accessible_doc_ids": ["d1", "d2"],
            "pool": {"d2": {"id": "d2", "text": "visible"}},
        },
        metadata={"component_id": "auto_populate_first_search"},
    )


def _point(*, tokens: list[int], text: str, tools: list[str]) -> StudentDecisionPoint:
    snap = _snap()
    return StudentDecisionPoint(
        episode_id="e0",
        query_id="q0",
        rollout_idx=0,
        turn_id=1,
        policy_version="v0",
        pre_action_snapshot=snap,
        pre_action_snapshot_hash=snap.content_hash(),
        student_model_input="",
        student_action_tokens=tokens,
        student_action_text=text,
        action_tool_names=tools,
        structurally_valid=True,
        student_prompt_token_ids=[1, 2, 3],
        teacher_prompt_token_ids=[4, 5, 6],
    )


def _teacher(_point: StudentDecisionPoint):
    return [
        harness_mutation(
            "auto_populate_first_search",
            {"before_curated": ["d1"], "after_curated": ["d1", "d2"]},
        )
    ]


def test_projected_tokens_not_replaced_by_sampled():
    sampled = _point(tokens=[11, 12, 13], text="to=grep_corpus\n{}\n", tools=["grep_corpus"])
    steps, _audit, _extras = project_on_policy_decisions(
        [sampled],
        teacher_event_fn=_teacher,
        component_id="auto_populate_first_search",
    )
    assert steps
    meta = steps[0].metadata
    assert meta.get("sampled_action_token_ids") == [11, 12, 13]
    assert "target_token_ids" not in meta
    assert steps[0].target_action.get("name") != "grep_corpus"


def test_qwen_action_renderer_uses_tool_call_xml():
    action = {"name": "curate", "arguments": {"add_ids": ["d2"]}}
    text = render_rollout_action_text(_FakeEnc(), action)
    assert text.startswith("<tool_call>")
    assert "curate" in text
    ids, roundtrip = encode_rollout_style_action(_FakeEnc(), action)
    assert ids
    assert roundtrip == text
