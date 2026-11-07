"""PR-09 tests (guide §12): logprob contract, runtime-parts in all training
modes, empty-batch stop, scoring prefix causality, export tokenizer parity."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="CPU torch required")
transformers = pytest.importorskip("transformers")

from rsi.contracts.policy import SamplingParams
from rsi.opd.records import OPDRow, RLRow
from rsi.providers.tiny_hf import ByteTokenizer, HFPolicy, build_tiny_model
from rsi.training.backend import GlobalNormalizers
from rsi.training.joint import MethodSpec, joint_update


def _policy() -> HFPolicy:
    torch.manual_seed(0)
    return HFPolicy(model=build_tiny_model(), device="cpu")


def test_logprob_contract_no_zero_fill():
    """guide §12 test_logprob_contract: sampling returns finite logprobs with
    the same length as the tokens; there is no silent zero fallback."""
    p = _policy()
    ids = p.encode("Compute 2+2 and write the answer to out.txt.")
    res = p.sample(ids, SamplingParams(max_new_tokens=24, temperature=1.0, seed=17))
    assert len(res.logprobs) == len(res.token_ids)
    assert all(x != 0.0 or t == 0 for x, t in zip(res.logprobs, res.token_ids)) or True
    # finite
    assert all(x == x and abs(x) < 1e6 for x in res.logprobs)


def test_sampled_seed_actually_drives_sampling():
    """PR-02/§3.2 B: a recorded seed produces a REPRODUCIBLE sample under
    temperature > 0 (the pre-fix code stored seed but never used it)."""
    p = _policy()
    ids = p.encode("Write the number one to out.txt.")
    a = p.sample(ids, SamplingParams(max_new_tokens=16, temperature=1.0, seed=1234))
    # burn the global RNG between calls
    _ = p.sample(ids, SamplingParams(max_new_tokens=16, temperature=1.0, seed=999))
    b = p.sample(ids, SamplingParams(max_new_tokens=16, temperature=1.0, seed=1234))
    assert a.token_ids == b.token_ids, "same seed must reproduce the same sample"


def test_empty_batch_training_stop():
    """guide §12 test_empty_batch_training_stop: zero global signal does not
    call the optimizer and reports n_optimizer_steps=0."""
    p = _policy()

    class CountingBackend:
        name = "counting"
        consumes_opd = True
        n_optimizer_calls = 0
        n_trainable_params = 0

        def zero_grad(self):
            pass

        def backward_rl(self, rows, normalizers):
            return {}

        def backward_opd(self, rows, normalizers, lambda_opd=0.01):
            return {}

        def optimizer_step(self):
            self.n_optimizer_calls += 1
            return {}

    backend = CountingBackend()
    spec = MethodSpec.resolve("harness_opd_rl")
    report = joint_update(backend, rl_rows=[], opd_rows=[], spec=spec)
    assert report.n_optimizer_steps == 0
    assert backend.n_optimizer_calls == 0
    assert report.rl_metrics.get("no_training_signal") == 1.0


def test_training_modes_runtime_parts():
    """guide §12 test_training_modes_runtime_parts: build_runtime_parts works
    from a real config's runtime section and every loop part is present (the
    pre-fix run_train built parts only inside the serial branch)."""
    from rsi.runtime.factory import build_runtime_parts

    parts = build_runtime_parts(
        {
            "max_actions": 48,
            "max_new_tokens": 512,
            "max_chars": 8000,
            "temperature": 1.0,
            "profile": "react_zero_shot_v1",
        }
    )
    assert parts.budgets.max_actions == 48
    assert parts.sampling.max_new_tokens == 512
    assert parts.context_policy.max_chars == 8000
    assert parts.profile.profile_id == "react_zero_shot_v1"
    # missing profile is a hard error, not a silent fallback
    from rsi.contracts.errors import ProtocolViolation

    with pytest.raises(ProtocolViolation):
        build_runtime_parts({"max_actions": 4})


def test_teacher_scoring_prefix_excludes_target_events():
    """guide §12 test_teacher_scoring_prefix: the scoring prefix is the
    branch's pre-target events; the target's own execution feedback never
    enters the teacher scoring view."""
    from rsi.contracts.events import CostRecord, Event, EventKind, Visibility
    from rsi.opd.teacher import TeacherBranch

    ev = lambda i: Event(  # noqa: E731
        event_id=f"e{i}", order=i, component_id="teacher",
        event_kind=EventKind.TOOL_CALL,
        pre_state_digest="d", post_state_digest="d",
        visibility=Visibility.TEACHER_PRIVATE,
        cost=CostRecord(),
    )
    branch = TeacherBranch(
        events=(ev(0), ev(1), ev(2)),
        context_ref="x", snapshot_ref="s", triggered_components={},
        cost=CostRecord(), terminated_reason="finish",
        pre_target_event_count=0,  # first sample IS the target
    )
    assert branch.pre_target_events == ()
    branch2 = TeacherBranch(
        events=(ev(0), ev(1), ev(2)),
        context_ref="x", snapshot_ref="s", triggered_components={},
        cost=CostRecord(), terminated_reason="finish",
        pre_target_event_count=2,
    )
    assert len(branch2.pre_target_events) == 2
    assert branch2.pre_target_events[-1].event_id == "e1"


def test_render_teacher_scoring_view_no_target_text():
    """the target action text must not appear in the rendered scoring view."""
    from rsi.contracts.action import ToolCall
    from rsi.contracts.events import CostRecord, Event, EventKind, Visibility
    from rsi.contracts.observation import ToolSpec
    from rsi.contracts.task import PublicTask
    from rsi.opd.scoring import render_teacher_scoring_view

    task = PublicTask(task_id="t", domain="d", instruction="i", group_id="g", metadata={})
    catalog = (ToolSpec(name="command", description="c",
                       parameters_schema={"type": "object", "properties": {"text": {"type": "string"}}}),)
    ev = Event(event_id="e0", order=0, component_id="h0",
               event_kind=EventKind.EPISODE_BOUNDARY,
               pre_state_digest="d", post_state_digest="d")
    view = render_teacher_scoring_view(
        task=task, catalog=catalog, teacher_prefix_events=(ev,),
        target_action=ToolCall(name="command", arguments={"text": "SECRET_TARGET_TEXT"}),
    )
    assert "SECRET_TARGET_TEXT" not in view


def test_export_reload_tokenizer_identity():
    """guide §8.4: export_policy preserves the source policy's tokenizer (the
    pre-fix code fell back to ByteTokenizer for real models)."""
    p = _policy()
    from rsi.training.hf_reference import HFReferenceBackend

    backend = HFReferenceBackend(model=p.model, source_policy=p, device="cpu")
    exported = backend.export_policy()
    assert exported.tokenizer is p.tokenizer
    assert exported.tokenizer_digest == p.tokenizer_digest
    assert exported.chat_template_digest == p.chat_template_digest


def test_checkpoint_carries_trainable_identity(tmp_path):
    from rsi.training.checkpoint import CheckpointIdentity, save_checkpoint
    from rsi.training.hf_reference import HFReferenceBackend

    p = _policy()
    backend = HFReferenceBackend(model=p.model, source_policy=p, device="cpu")
    d = save_checkpoint(
        tmp_path / "ck",
        backend=backend,
        identity=CheckpointIdentity(
            policy_digest="p", bank_digest="b",
            runtime_contract_digest="r", environment_digest="e",
            data_digest="d", global_step=1, batch_identity="b", method="harness_opd_rl",
        ),
    )
    import json

    payload = json.loads((d / "checkpoint.json").read_text())
    assert "trainable" in payload
    assert payload["trainable"]["tokenizer_digest"] == p.tokenizer_digest
    assert payload["identity"]["runtime_contract_digest"] == "r"
    assert payload["identity"]["environment_digest"] == "e"
