from __future__ import annotations

from types import SimpleNamespace

from trim.training.sft_ablation.teacher_archive import ARCHIVE_SCHEMA, FrozenTeacherSession
from trim.training.teacher_sft.dataset import build_teacher_sft_rows


class Enc:
    family = "plain"

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode_tokens(self, ids: list[int]) -> str:
        return bytes(ids).decode("utf-8")


def test_archive_generation_calls_the_model_and_does_not_step_an_optimizer():
    seen = {}

    def generate(requests):
        seen["called"] = True
        return requests

    def rollout(fn, rows, **kwargs):
        assert fn is generate
        fn([])
        assert kwargs["teacher_mode"] is False
        assert kwargs["enable_rl"] is False
        assert kwargs["harness_mask"]["verify_tool"] is True
        seen["rows"] = rows
        return []

    session = FrozenTeacherSession(generate, policy_version="teacher-init")
    archive = session.collect(
        [{"query_id": "q"}],
        rollout,
        group_size=1,
        max_turns=2,
        max_new_tokens=8,
        seed=1,
        enc=Enc(),
        train_env="upstream",
    )
    assert seen["called"] is True
    assert session.optimizer_steps == 0
    assert archive["teacher_mode"] is False
    assert archive["schema_version"] == ARCHIVE_SCHEMA


def test_teacher_rows_keep_verify_and_importance_and_pre_action_prompt():
    archive = {
        "schema_version": ARCHIVE_SCHEMA,
        "policy_version": "teacher-init",
        "mask": "all",
        "teacher_mode": False,
        "trajectories": [
            {
                "trajectory_id": "q:r0",
                "query_id": "q",
                "policy_version": "teacher-init",
                "decisions": [
                    {
                        "decision_id": "d-verify",
                        "prompt_ids": [11, 12, 13],
                        "action": {"name": "verify", "arguments": {"claim": "c", "doc_id": "d1"}},
                        "executed_ok": True,
                        "structurally_valid": True,
                        "pre_state_hash": "pre",
                    },
                    {
                        "decision_id": "d-curate",
                        "prompt_ids": [11, 12, 13],
                        "action": {
                            "name": "curate",
                            "arguments": {"add_ids": ["d1"], "remove_ids": [], "importance": {"d1": 2}},
                        },
                        "executed_ok": True,
                        "structurally_valid": True,
                        "pre_state_hash": "pre",
                    },
                ],
            }
        ],
    }
    rows, rejected = build_teacher_sft_rows(
        archive,
        Enc(),
        {"max_model_len": 8192, "max_new_tokens": 2048},
    )
    assert rejected == []
    by_id = {row.decision_id: row for row in rows}
    assert by_id["d-verify"].metadata["native_action"]["name"] == "verify"
    assert by_id["d-curate"].metadata["native_action"]["arguments"]["importance"] == {"d1": 2}
    assert by_id["d-verify"].prompt_ids == [11, 12, 13]
    assert b"OBS_SECRET" not in bytes(by_id["d-verify"].prompt_ids)
    assert all(row.weight == 1.0 for row in rows)
    assert all(row.prompt_owner == "teacher_full" for row in rows)
