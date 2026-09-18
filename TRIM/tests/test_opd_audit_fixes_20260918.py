"""CPU contracts from OPD_Code_Audit_and_Revision_Advice_20260918."""

from __future__ import annotations

import pytest
import torch

from trim.training.action_encoding import (
    encode_supervised_action,
    prompt_visible_doc_ids_from_prompt,
    visible_doc_ids_for_decision,
)
from trim.training.opd_dataset import ProjectedTrainingStep
from trim.training.opd_prompt_encoding import attach_prompt_context, render_rollout_action_text
from trim.training.rl_opd_types import collection_needs
from trim.training.tinker_opd_datum import (
    TinkerOPDDatum,
    build_projected_seed_datums,
    build_tinker_opd_datums,
    recover_teacher_prompt_ids,
)


class _CharEnc:
    family = "qwen3"

    def encode(self, text: str) -> list[int]:
        return [ord(c) % 200 + 1 for c in str(text)]


def _step(**kwargs) -> ProjectedTrainingStep:
    meta = dict(kwargs.pop("metadata", None) or {})
    meta.setdefault("student_prompt_token_ids", [1, 2, 3])
    return ProjectedTrainingStep(
        prompt_reduced=kwargs.get("prompt_reduced", "PREFIX"),
        target_text=kwargs.get("target_text", "aa"),
        target_action=kwargs.get("target_action") or {"name": "end_search", "arguments": {}},
        token_mask=kwargs.get("token_mask"),
        weight=kwargs.get("weight", 1.0),
        projection_kind="direct",
        projection_confidence=kwargs.get("confidence", 1.0),
        metadata=meta,
        student_snapshot=kwargs.get("student_snapshot"),
    )


def test_gap_needs_teacher_context_ce_does_not():
    ce = collection_needs(collection_mode="rl+opd", opd_loss="sr_opd_ce")
    gap = collection_needs(collection_mode="rl+opd", opd_loss="sr_opd_projected_gap")
    sampled = collection_needs(collection_mode="rl+opd", opd_loss="sr_opd_sampled_gap")
    assert ce.need_student_snapshot is True
    assert ce.need_teacher_context is False
    assert gap.need_teacher_context is True
    assert sampled.need_teacher_context is True
    assert gap.need_debug_view is False


def test_gap_missing_teacher_ids_are_dropped_not_rebuilt():
    step = _step(target_text="aa", metadata={"student_prompt_token_ids": [9, 8]})
    datums, stats = build_projected_seed_datums(
        [step], lambda_opd=0.01, encode_fn=lambda t: [1] * len(t), policy_version="v1"
    )
    assert datums == []
    assert stats["n_skip_missing_teacher"] == 1


def test_gap_does_not_copy_student_snapshot_as_teacher():
    from trim.adapters.components import minus_mask
    from trim.state.snapshot import capture_snapshot

    snap = capture_snapshot(
        query_id="q0",
        step=7,
        harness_mask=minus_mask("auto_populate_first_search"),
        working_memory={"query": "who founded acme", "accessible_doc_ids": ["d1"]},
        metadata={"component_id": "auto_populate_first_search"},
    )
    ids = recover_teacher_prompt_ids(
        teacher_ids=None,
        metadata={},
        snapshot=snap,
        encode=lambda t: [1] * len(t),
        model_enc=_CharEnc(),
    )
    assert ids == []


def test_gap_recovers_teacher_from_explicit_prompt_full():
    ids = recover_teacher_prompt_ids(
        teacher_ids=None,
        metadata={"prompt_full": "FULL"},
        snapshot=None,
        encode=lambda t: [7] * len(t),
        model_enc=None,
    )
    assert ids == []
    ids = recover_teacher_prompt_ids(
        teacher_ids=None,
        metadata={"prompt_full": "FULL"},
        snapshot=None,
        encode=lambda t: [7] * len(t),
        model_enc=None,
        allow_offline=True,
    )
    assert ids == [7, 7, 7, 7]


def test_prompt_context_refs_are_excluded_from_snapshot_hash():
    from trim.adapters.components import minus_mask
    from trim.state.snapshot import capture_snapshot

    snap = capture_snapshot(
        query_id="q0",
        step=7,
        harness_mask=minus_mask("auto_populate_first_search"),
        working_memory={"query": "who founded acme", "accessible_doc_ids": ["d1"]},
    )
    before = snap.content_hash()
    attach_prompt_context(snap, acts=[], wm_text="student wm", teacher_wm_text="teacher wm extra")
    assert snap.content_hash() == before
    assert snap.metadata["_teacher_wm_text"] == "teacher wm extra"
    assert snap.metadata["_rollout_wm_text"] == "student wm"


def test_empty_visible_set_rejects_doc_action():
    step = _step(
        target_action={"name": "read_document", "arguments": {"doc_id": "d1"}},
        metadata={"student_prompt_token_ids": [1, 2], "visible_doc_ids": []},
    )
    with pytest.raises(ValueError, match="not visible"):
        build_tinker_opd_datums([step], lambda_opd=0.1, policy_version="v1")


def test_d1_is_not_visible_because_d10_is_in_prompt():
    enc = _CharEnc()
    prompt = "working memory\n  - d10: long document about d10\n"
    prompt_ids = enc.encode(prompt)
    visible = prompt_visible_doc_ids_from_prompt(
        prompt_ids=prompt_ids,
        accessible_ids=["d1", "d10"],
        enc=enc,
    )
    assert visible == ["d10"]
    assert "d1" not in visible


def test_search_allowed_when_prompt_has_no_docs():
    visible = visible_doc_ids_for_decision(
        metadata={"prompt_visible_doc_ids": []},
        prompt_ids=[1, 2],
        enc=_CharEnc(),
    )
    assert visible == []
    from trim.training.action_encoding import assert_action_visible

    assert_action_visible({"name": "search_corpus", "arguments": {"query": "x"}}, visible)


def test_projected_gap_uses_effective_weight_and_skips_zero():
    from trim.training.tinker_opd_datum import build_projected_seed_datums, supervised_weight_sum

    steps = [
        _step(target_text="aa", weight=1.0, metadata={"student_prompt_token_ids": [1], "teacher_prompt_token_ids": [3, 4]}),
        _step(target_text="aa", weight=0.01, metadata={"student_prompt_token_ids": [1], "teacher_prompt_token_ids": [3, 4]}),
        _step(target_text="aa", weight=0.0, metadata={"student_prompt_token_ids": [1], "teacher_prompt_token_ids": [3, 4]}),
    ]
    datums, stats = build_projected_seed_datums(
        steps, lambda_opd=0.01, encode_fn=lambda t: [1] * len(t), policy_version="v1"
    )
    assert stats["n_skip_zero_mask"] == 1
    assert len(datums) == 2
    w0 = sum(datums[0].weights)
    w1 = sum(datums[1].weights)
    assert w0 == pytest.approx(2.0)
    assert w1 == pytest.approx(0.02)
    assert w0 / w1 == pytest.approx(100.0)
    assert supervised_weight_sum(datums) == pytest.approx(2.02)


def test_zero_mask_ce_row_does_not_enter_denominator():
    steps = [
        _step(target_text="aa", token_mask=[False, False], weight=1.0),
        _step(target_text="aa", token_mask=[True, True], weight=1.0),
    ]
    datums = build_tinker_opd_datums(steps, lambda_opd=0.1, encode_fn=lambda t: [1, 1], policy_version="v1")
    assert len(datums) == 1
    assert sum(datums[0].weights) == pytest.approx(0.1)


def test_ce_canonical_action_comes_from_target_action_not_xml_parse():
    from trim.eval.model_tokenizer import parse_qwen_tool_call

    class _FakeQwenEnc:
        family = "qwen3"
        stop_token_ids = [151645]

        def encode(self, text: str) -> list[int]:
            return [ord(c) % 200 + 1 for c in str(text)] + [151645]

        def decode_tokens(self, ids: list[int]) -> str:
            return "".join(chr((int(tid) - 1) % 200) for tid in ids if int(tid) != 151645)

        def parse_tool_call(self, text: str, completion_ids=None):
            return parse_qwen_tool_call(text, completion_ids=completion_ids)

    action = {"name": "curate", "arguments": {"add_ids": ["d2"]}}
    enc = _FakeQwenEnc()
    encoded = encode_supervised_action(
        target_action=action,
        target_text="<tool_call>not-a-name</tool_call>",
        encode=enc.encode,
        model_enc=enc,
    )
    assert encoded.canonical_action["name"] == "curate"
    assert encoded.termination_kind == "eos"
    step = _step(target_action=action, metadata={"student_prompt_token_ids": [1, 2]})
    datums = build_tinker_opd_datums(
        [step], lambda_opd=0.1, encode_fn=enc.encode, policy_version="v1", model_enc=enc
    )
    assert datums[0].target_action["name"] == "curate"


def test_row_input_length_reads_opd_dataclass_and_dict():
    from trim.integrations.verl.batch_adapter import row_input_length

    datum = TinkerOPDDatum(
        model_input="p",
        prompt_token_ids=[1] * 100,
        target_tokens=[0] * 100 + [2] * 6000,
        weights=[0.0] * 100 + [1.0] * 6000,
        policy_version="v1",
        n_supervised_tokens=6000,
        teacher_prompt_token_ids=[3] * 50,
    )
    assert row_input_length(datum) == 6100
    as_dict = {
        "prompt_ids": [1] * 100,
        "target_ids": [2] * 6000,
        "teacher_prompt_token_ids": [3] * 50,
    }
    assert row_input_length(as_dict) == 6100


def test_dummy_opd_zero_loss_keeps_forward_graph():
    p1 = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    p2 = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    logp = (p1 * 0.5 + p2 * 0.25).sum().reshape(1)
    loss = logp.float().sum() * 0.0
    loss.backward()
    assert p1.grad is not None and p2.grad is not None
    assert torch.count_nonzero(p1.grad) + torch.count_nonzero(p2.grad) >= 0


def test_harmony_renderer_is_accepted_by_strict_parser():
    from trim.eval.harmony_runtime import parse_harmony_tool_call

    class _Harmony:
        family = "gpt-oss"

    text = render_rollout_action_text(
        _Harmony(), {"name": "curate", "arguments": {"add_ids": ["d2"], "remove_ids": []}}
    )
    parsed = parse_harmony_tool_call(text)
    assert parsed.parsed is True
    assert parsed.tool_name == "curate"
    assert parsed.error is None


def test_verify_and_budget_teachers_skip_synthetic_defaults():
    from trim.training.token_budget_marker_teacher import teacher_events_from_wm as budget_events
    from trim.training.verify_tool_teacher import teacher_events_from_wm as verify_events

    verify = verify_events({"query": "q", "documents": [{"id": "d1", "text": "abc"}]})
    assert verify[0].metadata["skip_reason"] == "no_real_verifier"
    assert "verified" not in (verify[0].observation or {})
    empty_budget = budget_events({"query": "q", "documents": [{"id": "d1"}]})
    assert empty_budget[0].metadata["skip_reason"] == "empty_token_budget_marker"
    marked = budget_events({"query": "q", "token_budget_marker": "budget=12", "documents": [{"id": "d1"}]})
    assert marked[0].metadata["skip_reason"] == "no_capability_derived_action"


def test_projection_audit_splits_attempt_and_unregistered():
    from trim.training.four_cell_runtime import teacher_for
    from trim.training.opd_dataset import ProjectionAudit, project_and_materialize
    from trim.training.rl_opd_types import StudentDecisionPoint
    from trim.adapters.components import minus_mask
    from trim.state.snapshot import capture_snapshot

    snap = capture_snapshot(
        query_id="q",
        query_text="qtext",
        step=0,
        harness_mask=minus_mask("content_dedup"),
        working_memory={"query": "qtext", "documents": [{"id": "d1", "text": "abc"}]},
    )
    point = StudentDecisionPoint(
        episode_id="e",
        query_id="q",
        rollout_idx=0,
        turn_id=0,
        policy_version="v0",
        pre_action_snapshot=snap,
        pre_action_snapshot_hash=snap.content_hash(),
        student_model_input="",
        student_action_tokens=[],
        student_action_text="",
        action_tool_names=[],
        post_action_snapshot=snap,
    )
    events = teacher_for("content_dedup", teacher_kind="upstream")(point)
    audit = ProjectionAudit()
    project_and_materialize(
        student_snapshot=snap,
        teacher_events=events,
        student_mask=snap.harness_mask,
        component_id="content_dedup",
        audit=audit,
    )
    stats = audit.component_stats["content_dedup"]
    assert stats["attempt_count"] == 1
    assert stats["trigger_count"] == 0
    assert stats["skip_unregistered"] == 1
    assert audit.n_unregistered == 1
    assert stats["supervised_whitespace_tokens"] == stats["supervised_tokens"]


def test_train_contract_accepts_trim_gap_on_verl_and_rejects_trim_ce():
    from trim.training.opd_train_contract import (
        assert_train_contract,
        normalize_opd_loss,
        weights_look_like_unnormalized_gap_mask,
    )

    assert normalize_opd_loss(None, method="trim") == "sr_opd_projected_gap"
    assert normalize_opd_loss("", method="trim") == "sr_opd_projected_gap"
    payload = assert_train_contract("trim", "sr_opd_projected_gap", "verl")
    assert payload["actor_objective"] == "gap"
    assert payload["actor_wrap"] == "fsdp2"
    payload_ddp = assert_train_contract("trim", None, "torch_ddp_lora")
    assert payload_ddp["opd_loss"] == "sr_opd_projected_gap"
    assert payload_ddp["actor_wrap"] == "ddp"
    assert_train_contract("rl", "sr_opd_ce", "verl")
    assert_train_contract("rl+opd", "sr_opd_ce", "fsdp2")
    with pytest.raises(SystemExit):
        assert_train_contract("trim", "sr_opd_ce", "verl")
    with pytest.raises(SystemExit):
        assert_train_contract("trim", "sr_opd_sampled_gap", "hf_debug")
    with pytest.raises(SystemExit):
        assert_train_contract("rl+opd", "sr_opd_projected_gap", "verl")
    assert weights_look_like_unnormalized_gap_mask([1.0, 1.0, 0.0]) is True
    assert weights_look_like_unnormalized_gap_mask([0.0]) is False
    assert weights_look_like_unnormalized_gap_mask([0.001, 0.002]) is False


def test_padding_stats_exclude_dummy_and_count_real_tokens():
    from trim.integrations.verl.batch_adapter import chunk_padding_stats, dummy_opd_row, plan_joint_sync_batches

    rows = [
        {"prompt_ids": [1] * 4, "target_ids": [2] * 6, "weights": [0.1] * 6},
        {"prompt_ids": [1] * 10, "target_ids": [2] * 10, "weights": [0.1] * 10},
    ]
    plan = plan_joint_sync_batches([], rows, rank=0, world_size=1, micro_batch_size=2)
    assert plan.opd_real_tokens > 0
    assert plan.opd_padded_tokens >= plan.opd_real_tokens
    dummy_stats = chunk_padding_stats([[dummy_opd_row()]])
    assert dummy_stats["n_rows"] == 0.0
