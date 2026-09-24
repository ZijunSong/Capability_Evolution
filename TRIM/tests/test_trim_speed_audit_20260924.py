"""Regressions for the 2026-09-24 TRIM speed/correctness audit."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from trim.training.hf_rl_batch import gather_response_logprobs
from trim.training.opd_batch_health import assess_opd_health
from trim.training.opd_prompt_encoding import attach_prompt_context
from trim.training.rl_opd_types import OPD_LOSS_PROJECTED_GAP
from trim.training.runtime_manifest import finalize_step_timing
from trim.training.tinker_opd_datum import (
    build_projected_seed_datums,
    explain_teacher_recovery_failure,
    recover_teacher_prompt_ids,
)
from trim.training.tinker_rl_opd_trainer import materialize_teacher_prompt_ids


class _SnapEnc:
    family = "qwen3"
    calls: list[tuple] = []

    def encode(self, text: str) -> list[int]:
        return [ord(ch) % 200 + 1 for ch in str(text)]

    def build_first_turn_prompt_ids(self, query: str) -> list[int]:
        self.calls.append(("first", query))
        return [1]

    def build_continuation_prompt_ids(self, query: str, actions_obs=None, wm_text: str = "") -> list[int]:
        self.calls.append(("cont", wm_text))
        return [91, 92, 93]


def _snapshot(*, teacher_wm: str = ""):
    from trim.adapters.components import minus_mask
    from trim.state.snapshot import capture_snapshot

    snap = capture_snapshot(
        query_id="q0",
        step=3,
        harness_mask=minus_mask("auto_populate_first_search"),
        working_memory={"query": "who founded acme", "accessible_doc_ids": ["d1"]},
    )
    if teacher_wm:
        attach_prompt_context(snap, acts=[], wm_text="student wm", teacher_wm_text=teacher_wm)
    return snap


def test_online_prompt_full_does_not_block_snapshot_recovery():
    enc = _SnapEnc()
    enc.calls = []
    seen: list[str] = []

    def encode(text: str) -> list[int]:
        seen.append(text)
        return [7] * len(text)

    ids = recover_teacher_prompt_ids(
        teacher_ids=None,
        metadata={"prompt_full": "DEBUG TEXT"},
        snapshot=_snapshot(teacher_wm="teacher secret"),
        encode=encode,
        model_enc=enc,
        allow_offline=False,
    )
    assert ids == [91, 92, 93]
    assert seen == []
    assert enc.calls and enc.calls[-1][0] == "cont"
    assert "teacher secret" in enc.calls[-1][1]


def test_online_prompt_full_without_snapshot_still_fails():
    seen: list[str] = []

    def encode(text: str) -> list[int]:
        seen.append(text)
        return [7]

    ids = recover_teacher_prompt_ids(
        teacher_ids=None,
        metadata={"prompt_full": "DEBUG TEXT"},
        snapshot=None,
        encode=encode,
        model_enc=_SnapEnc(),
        allow_offline=False,
    )
    assert ids == []
    assert seen == []
    assert (
        explain_teacher_recovery_failure(
            metadata={"prompt_full": "DEBUG TEXT"},
            snapshot=None,
            model_enc=_SnapEnc(),
            allow_offline=False,
        )
        == "online_debug_prompt_full_without_snapshot"
    )


def test_explicit_teacher_ids_win_over_prompt_full_and_snapshot():
    ids = recover_teacher_prompt_ids(
        teacher_ids=[4, 5],
        metadata={"prompt_full": "DEBUG", "teacher_prompt_token_ids": [8]},
        snapshot=_snapshot(teacher_wm="teacher secret"),
        encode=lambda text: [7],
        model_enc=_SnapEnc(),
        allow_offline=False,
    )
    assert ids == [4, 5]
    ids = recover_teacher_prompt_ids(
        teacher_ids=None,
        metadata={"prompt_full": "DEBUG", "teacher_prompt_token_ids": [8, 9]},
        snapshot=_snapshot(teacher_wm="teacher secret"),
        encode=lambda text: [7],
        model_enc=_SnapEnc(),
        allow_offline=False,
    )
    assert ids == [8, 9]


class _TeacherOnlyEnc:
    """Teacher prompt encoder. Action text falls through to UTF-8 so the action roundtrip stays exact."""

    family = "plain"
    calls: list[tuple] = []

    def build_continuation_prompt_ids(self, query: str, actions_obs=None, wm_text: str = "") -> list[int]:
        self.calls.append(("cont", wm_text))
        return [91, 92, 93]

    def build_first_turn_prompt_ids(self, query: str) -> list[int]:
        self.calls.append(("first", query))
        return [1]


def test_projected_seed_keeps_datum_when_debug_prompt_full_and_snapshot_exist():
    enc = _TeacherOnlyEnc()
    enc.calls = []
    snap = _snapshot(teacher_wm="teacher secret")
    from trim.training.opd_dataset import ProjectedTrainingStep

    step = ProjectedTrainingStep(
        prompt_reduced="PREFIX",
        target_text="aa",
        target_action={"name": "end_search", "arguments": {}},
        token_mask=None,
        weight=1.0,
        projection_kind="direct",
        projection_confidence=1.0,
        metadata={
            "student_prompt_token_ids": [1, 2, 3],
            "prompt_full": "DEBUG TEXT THAT MUST NOT BE THE TEACHER PREFIX",
        },
        student_snapshot=snap.to_dict(),
    )
    datums, stats = build_projected_seed_datums(
        [step],
        lambda_opd=0.01,
        encode_fn=lambda text: (_ for _ in ()).throw(AssertionError(text)),
        policy_version="v1",
        opd_loss=OPD_LOSS_PROJECTED_GAP,
        model_enc=enc,
        allow_offline=False,
    )
    assert stats["n_skip_missing_teacher"] == 0
    assert stats["n_kept"] == 1
    assert len(datums) == 1
    assert datums[0].teacher_prompt_token_ids == [91, 92, 93]


def test_materialize_records_recovery_failure_reason():
    from trim.training.rl_opd_types import StudentDecisionPoint

    snap = _snapshot()
    point = StudentDecisionPoint(
        episode_id="e0",
        query_id="q0",
        rollout_idx=0,
        turn_id=2,
        policy_version="v1",
        pre_action_snapshot=snap,
        pre_action_snapshot_hash=snap.content_hash(),
        student_model_input="",
        student_action_tokens=[1],
        student_action_text="",
        action_tool_names=["search_corpus"],
    )
    stats: dict = {}
    materialize_teacher_prompt_ids(
        [point],
        encode_fn=lambda text: [1],
        model_enc=_SnapEnc(),
        recovery_stats=stats,
    )
    assert point.teacher_prompt_token_ids == []
    assert stats["n_failed"] == 1
    assert stats["reasons"]["snapshot_missing_teacher_context"] == 1


def test_missing_teacher_after_projection_is_fatal_immediately():
    health = assess_opd_health(
        lambda_opd=0.01,
        opd_loss=OPD_LOSS_PROJECTED_GAP,
        n_rl=10,
        n_opd=0,
        projection_stats={
            "n_projected_training_steps": 4,
            "n_skip_missing_teacher": 4,
            "n_kept": 0,
            "missing_teacher_reasons": {"online_debug_prompt_full_snapshot_has_no_teacher_context": 4},
        },
        opd_empty_streak=0,
        max_empty_opd=3,
    )
    assert health["fatal"] is True
    assert "teacher context" in health["message"]
    assert health["effective_update_type"] == "rl_only"


def test_untriggered_teacher_allows_two_empty_opd_batches_then_stops():
    stats = {"n_untriggered": 3, "n_projected_training_steps": 0, "n_kept": 0}
    first = assess_opd_health(
        lambda_opd=0.01,
        opd_loss=OPD_LOSS_PROJECTED_GAP,
        n_rl=4,
        n_opd=0,
        projection_stats=stats,
        opd_empty_streak=0,
        max_empty_opd=3,
    )
    assert first["fatal"] is False
    assert first["opd_empty_streak"] == 1
    second = assess_opd_health(
        lambda_opd=0.01,
        opd_loss=OPD_LOSS_PROJECTED_GAP,
        n_rl=4,
        n_opd=0,
        projection_stats=stats,
        opd_empty_streak=first["opd_empty_streak"],
        max_empty_opd=3,
    )
    assert second["fatal"] is False
    third = assess_opd_health(
        lambda_opd=0.01,
        opd_loss=OPD_LOSS_PROJECTED_GAP,
        n_rl=4,
        n_opd=0,
        projection_stats=stats,
        opd_empty_streak=second["opd_empty_streak"],
        max_empty_opd=3,
    )
    assert third["fatal"] is True
    assert third["opd_empty_streak"] == 3
    recovered = assess_opd_health(
        lambda_opd=0.01,
        opd_loss=OPD_LOSS_PROJECTED_GAP,
        n_rl=4,
        n_opd=2,
        projection_stats={"n_projected_training_steps": 2, "n_kept": 2},
        opd_empty_streak=2,
        max_empty_opd=3,
    )
    assert recovered["fatal"] is False
    assert recovered["opd_empty_streak"] == 0
    assert recovered["effective_update_type"] == "rl_opd_joint"


def test_rl_only_run_does_not_use_opd_empty_streak():
    health = assess_opd_health(
        lambda_opd=0.0,
        opd_loss="sr_opd_ce",
        n_rl=3,
        n_opd=0,
        projection_stats={},
        opd_empty_streak=2,
        max_empty_opd=3,
    )
    assert health["fatal"] is False
    assert health["opd_empty_streak"] == 2
    assert health["effective_update_type"] == "rl_only"


def test_selective_logprob_matches_log_softmax_and_grad(monkeypatch):
    monkeypatch.setattr("trim.training.hf_rl_batch._LOGPROB_VOCAB_CHUNK", 3)
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 11, requires_grad=True)
    ref = logits.detach().clone().requires_grad_(True)
    responses = [[2, 3], []]
    got = gather_response_logprobs(logits, responses, max_resp=2)
    window = ref[:, :-1, :]
    full = torch.log_softmax(window.float(), dim=-1)
    ids = torch.tensor([2, 3])
    expected = full[0, 0:2].gather(1, ids.view(-1, 1)).squeeze(1)
    assert got[0].dtype == torch.float32
    assert torch.allclose(got[0], expected, atol=1e-5)
    assert got[1].numel() == 0
    got[0].sum().backward()
    expected.sum().backward()
    assert logits.grad is not None and ref.grad is not None
    assert torch.allclose(logits.grad, ref.grad, atol=1e-5)


def test_bf16_logprob_stays_fp32():
    logits = torch.zeros(1, 3, 5, dtype=torch.bfloat16)
    logits[0, 0, 2] = 8.0
    logits[0, 1, 3] = 8.0
    logps = gather_response_logprobs(logits, [[2, 3]], max_resp=2)
    assert logps[0].dtype == torch.float32
    assert torch.isfinite(logps[0]).all()


def test_step_timing_residual_ignores_overlapping_components():
    out = finalize_step_timing(
        {
            "_wall": 10.0,
            "rollout_generate_s": 4.0,
            "gather_s": 1.0,
            "rollout_component_model_s": 4.0,
        }
    )
    assert out["step_wall_s"] == 10.0
    assert out["unattributed_s"] == 5.0
    assert out["rollout_component_model_s"] == 4.0


def test_policy_pickle_rejects_mismatched_version(tmp_path: Path):
    from trim.training.dist_runtime import read_policy_pickle, write_policy_pickle

    path = tmp_path / "shard.pkl"
    write_policy_pickle(path, [{"row": 1}], policy_version="v1")
    assert read_policy_pickle(path, policy_version="v1") == [{"row": 1}]
    with pytest.raises(RuntimeError, match="policy_version"):
        read_policy_pickle(path, policy_version="v2")


def test_rollout_shard_dir_is_scoped_by_run_id(tmp_path: Path, monkeypatch):
    from trim.training.dist_runtime import resolve_rollout_shard_dir

    monkeypatch.setenv("TRIM_ROLLOUT_SHARD_DIR", str(tmp_path))
    path = resolve_rollout_shard_dir(out=tmp_path / "out", run_id="job 1", single_node=True)
    assert path == tmp_path / "job_1" / "rollout_shards"
    assert path.is_dir()


def test_ddp_save_uses_adapter_pretrained_not_full_state(tmp_path: Path):
    from trim.integrations.verl.fsdp2_actor import FSDP2CispoActor

    seen: dict = {}

    class Core:
        peft_config = {"default": object()}

        def save_pretrained(self, path, **kwargs):
            seen["kwargs"] = kwargs
            Path(path).mkdir(parents=True, exist_ok=True)

    class Wrap:
        def __init__(self):
            self.module = Core()

    class Tok:
        def save_pretrained(self, path):
            seen["tokenizer"] = str(path)

    actor = FSDP2CispoActor.__new__(FSDP2CispoActor)
    actor.rank = 0
    actor.world_size = 1
    actor.wrap = "ddp"
    actor.model = Wrap()
    actor.tokenizer = Tok()
    actor.save_adapter(tmp_path / "adapter")
    assert seen["kwargs"]["safe_serialization"] is True
    assert seen["kwargs"]["save_embedding_layers"] is False
    assert "state_dict" not in seen["kwargs"]
    assert seen["tokenizer"]


def test_turn_audit_keeps_one_sample_per_kind():
    from trim.training.batched_env_rollout import turn_audit_from_groups

    group = SimpleNamespace(
        trajectory_group={
            "turn_diags": [
                {
                    "episode_id": "e0",
                    "turn_id": 0,
                    "tool_name": "search_corpus",
                    "structurally_valid": True,
                    "executed_ok": True,
                    "generation_finish_reason": "stop",
                    "episode_end_reason": "",
                    "parse_method": "tool_call",
                },
                {
                    "episode_id": "e0",
                    "turn_id": 1,
                    "tool_name": "end_search",
                    "structurally_valid": True,
                    "executed_ok": True,
                    "generation_finish_reason": "stop",
                    "episode_end_reason": "done",
                    "parse_method": "tool_call",
                },
            ],
            "rl_rows": [
                {"episode_id": "e0", "turn_id": 0, "prompt_ids": [1, 2], "action_ids": [9]},
                {"episode_id": "e0", "turn_id": 1, "prompt_ids": [3], "action_ids": [8]},
            ],
        }
    )
    audit = turn_audit_from_groups([group])
    kinds = {sample["kind"] for sample in audit["samples"]}
    assert "tool_call" in kinds
    assert "end_search" in kinds
    assert audit["counts"]["n_tool_call"] == 1
    assert audit["counts"]["n_end_search"] == 1
    end = next(sample for sample in audit["samples"] if sample["kind"] == "end_search")
    assert end["action_token_ids"] == [8]
