"""CPU contracts for the TRIM FSDP2 / verl adapter (T3)."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from trim.cli.launch import parse_train_args
from trim.integrations.verl.batch_adapter import (
    drop_constant_reward_groups,
    expand_episode_rows,
    plan_joint_sync_batches,
    shard_training_rows,
    training_rows_from_groups,
)
from trim.integrations.verl.joint_objective import (
    assert_one_optimizer_step,
    cispo_clip_bounds,
    cispo_token_loss,
    resolved_cispo_config,
    verl_cispo_clip_config,
)
from trim.training.dist_runtime import training_backend_from_argv, visible_cuda_count


def test_cispo_maps_to_zero_five():
    clip = verl_cispo_clip_config()
    low, high = cispo_clip_bounds(**clip)
    assert clip == {"clip_ratio_low": 1.0, "clip_ratio_high": 4.0}
    assert (low, high) == (0.0, 5.0)
    cfg = resolved_cispo_config()
    assert cfg["loss_mode"] == "cispo"
    assert cfg["ppo_epochs"] == 1
    assert cfg["clip_low_threshold"] == 0.0
    assert cfg["clip_high_threshold"] == 5.0


def test_cispo_token_loss_clips_and_one_step():
    new = torch.tensor([0.0, -1.0], dtype=torch.float32, requires_grad=True)
    old = torch.tensor([-4.0, -1.0], dtype=torch.float32)
    loss = cispo_token_loss(new, old, advantage=1.0, clip_low=0.0, clip_high=5.0)
    loss.backward()
    assert new.grad is not None
    assert_one_optimizer_step(1)


def test_expand_episode_rows_broadcasts_existing_and_missing_adv():
    rows = [
        {"query_id": "q", "episode_id": "q_r0", "reward": 1.0, "advantage": 0.5, "action_ids": [1]},
        {"query_id": "q", "episode_id": "q_r0", "reward": 1.0, "advantage": 0.5, "action_ids": [2]},
    ]
    out = expand_episode_rows(rows)
    assert [r["advantage"] for r in out] == [0.5, 0.5]
    raw = [
        {"query_id": "q", "episode_id": "q_r0", "reward": 2.0, "action_ids": [1]},
        {"query_id": "q", "episode_id": "q_r1", "reward": 0.0, "action_ids": [2]},
        {"query_id": "q", "episode_id": "q_r1", "reward": 0.0, "action_ids": [3]},
    ]
    filled = expand_episode_rows(raw)
    assert filled[0]["advantage"] == filled[0]["advantage"]
    assert filled[1]["advantage"] == filled[2]["advantage"]
    assert filled[0]["advantage"] != filled[1]["advantage"]


def test_drop_constant_reward_groups_and_shard():
    keep = SimpleNamespace(
        query_id="q0",
        terminal_rewards=[0.1, 0.9],
        trajectory_group={
            "rl_rows": [
                {
                    "query_id": "q0",
                    "episode_id": "q0_r0",
                    "reward": 0.1,
                    "action_ids": [1, 2],
                    "token_logprobs": [-0.1, -0.2],
                    "action_mask": [1, 1],
                    "effective_prompt_ids": [9],
                }
            ]
        },
    )
    drop = SimpleNamespace(query_id="q1", terminal_rewards=[0.4, 0.4], trajectory_group={"rl_rows": []})
    kept, n_drop = drop_constant_reward_groups([keep, drop])
    assert n_drop == 1
    assert kept == [keep]
    rows = training_rows_from_groups(kept)
    assert len(rows) == 1
    shards = [shard_training_rows(rows, rank=r, world_size=8) for r in range(8)]
    assert sum(len(s) for s in shards) == 1
    assert shards[0][0]["action_ids"] == [1, 2]


def test_parse_train_backend_torch_ddp_lora(tmp_path):
    args, _spec = parse_train_args(
        [
            "--train_method",
            "rl",
            "--component",
            "all",
            "--training-backend",
            "torch_ddp_lora",
            "--expected-world-size",
            "8",
            "--train-steps",
            "100",
            "--max-turns",
            "40",
            "--out",
            str(tmp_path / "out"),
            "--validate-only",
        ]
    )
    assert args.training_backend == "torch_ddp_lora"
    assert args.expected_world_size == 8


def test_parse_train_backend_verl(tmp_path):
    args, _spec = parse_train_args(
        [
            "--train_method",
            "rl",
            "--component",
            "all",
            "--training-backend",
            "verl",
            "--train-steps",
            "100",
            "--max-turns",
            "40",
            "--out",
            str(tmp_path / "out"),
            "--validate-only",
        ]
    )
    assert args.training_backend == "verl"
    assert args.train_steps == 100
    assert args.max_turns == 40
    assert args.enforce_eager is False


def test_training_backend_from_argv_and_visible_count(monkeypatch):
    assert training_backend_from_argv(["--train_method", "rl"]) == "hf_debug"
    assert training_backend_from_argv(["--training-backend", "verl", "--train-steps", "100"]) == "verl"
    assert training_backend_from_argv(["--training-backend=fsdp2"]) == "fsdp2"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    assert visible_cuda_count() == 8
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7,8")
    assert visible_cuda_count() == 9


def test_global_batch_plan_equal_rounds_and_dummy():
    rows = []
    for i in range(17):
        n = 64 * ((i % 5) + 1)
        rows.append(
            {
                "row_id": i,
                "prompt_ids": [1] * n,
                "action_ids": [2, 3],
                "action_mask": [1, 1],
                "token_logprobs": [-0.1, -0.2],
                "advantage": 0.5,
            }
        )
    plans = [
        plan_joint_sync_batches(rows, [], rank=r, world_size=8, micro_batch_size=4)
        for r in range(8)
    ]
    rounds = {p.n_rl_rounds for p in plans}
    assert len(rounds) == 1
    assert plans[0].n_sync_rounds == plans[0].n_rl_rounds
    assert sum(1 for p in plans for chunk in p.rl_chunks) == plans[0].n_rl_rounds * 8
    assert plans[0].n_dummy_rl == (8 - (plans[0].n_global_rl_microbatches % 8)) % 8
    real = [row for p in plans for chunk in p.rl_chunks for row in chunk if not row.get("is_dummy")]
    assert len(real) == 17


def test_resolve_resume_optimizer_ddp_and_fsdp2(tmp_path):
    from trim.integrations.verl.trainer_adapter import resolve_resume_optimizer_path

    ckpt = tmp_path / "step_000001"
    ckpt.mkdir()
    (ckpt / "optimizer.rank0000.pt").write_bytes(b"opt")
    assert resolve_resume_optimizer_path(ckpt, rank=3, world_size=8, wrap="ddp").endswith("optimizer.rank0000.pt")
    try:
        resolve_resume_optimizer_path(ckpt, rank=3, world_size=8, wrap="fsdp2")
        raised = False
    except SystemExit:
        raised = True
    assert raised is True


def test_unsupported_verl_method_scape():
    from trim.integrations.verl.trainer_adapter import _unsupported_method

    try:
        _unsupported_method("trim")
        raised = False
    except SystemExit:
        raised = True
    assert raised is True
    _unsupported_method("rl")
    _unsupported_method("rl+opd")
    try:
        _unsupported_method("rl+opd", "sr_opd_projected_gap")
        raised_gap = False
    except SystemExit:
        raised_gap = True
    assert raised_gap is True
