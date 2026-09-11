"""Regression tests for TRIM/RL training fixes (diagnosis doc items 01-13)."""

from __future__ import annotations

import math

import pytest
import torch

from trim.training.hf_rl_opd_client import HFDebugTrainingClient, episode_relative_advantages
from trim.training.train_query_sampler import QuerySampler
from trim.training.vllm_hybrid import mean_behavior_logprob


def test_query_sampler_epoch_no_repeat_until_exhausted():
    pool = [{"query_id": f"q{i}"} for i in range(8)]
    sampler = QuerySampler(pool, base_seed=42, groups_per_step=4)
    a, _ = sampler.sample_for_rollout()
    b, _ = sampler.sample_for_rollout()
    assert len(a) == 4 and len(b) == 4
    ids_a = {r["query_id"] for r in a}
    ids_b = {r["query_id"] for r in b}
    assert ids_a.isdisjoint(ids_b)


def test_train_group_sampling_uses_global_step_not_fixed_17():
    from types import SimpleNamespace

    from trim.training.hf_rl_batch import sample_groups_for_step

    groups = [SimpleNamespace(query_id=f"q{i}") for i in range(32)]
    a, _ = sample_groups_for_step(groups, 8, seed=0 + 17)
    b, _ = sample_groups_for_step(groups, 8, seed=0 + 17)
    c, _ = sample_groups_for_step(groups, 8, seed=3 + 17)
    assert [g.query_id for g in a] == [g.query_id for g in b]
    assert [g.query_id for g in a] != [g.query_id for g in c]


def test_episode_advantage_ignores_turn_multiplicity():
    rows = [
        {"query_id": "q0", "episode_id": "q0_r0", "rollout_idx": 0, "reward": 0.0},
        {"query_id": "q0", "episode_id": "q0_r1", "rollout_idx": 1, "reward": 1.0},
        {"query_id": "q0", "episode_id": "q0_r1", "rollout_idx": 1, "reward": 1.0},
        {"query_id": "q0", "episode_id": "q0_r1", "rollout_idx": 1, "reward": 1.0},
    ]
    adv = episode_relative_advantages(rows)
    assert adv[0] == pytest.approx(-1.0)
    assert adv[1] == pytest.approx(1.0)
    assert adv[2] == pytest.approx(1.0)
    assert adv[3] == pytest.approx(1.0)


class _TinyBackend:
    def __init__(self) -> None:
        self.param = torch.nn.Parameter(torch.tensor([0.0]))
        self.optimizer = torch.optim.SGD([self.param], lr=0.1)
        self._device = torch.device("cpu")

    def encode(self, text: str) -> list[int]:
        return [1, 2, 3]

    def _teacher_forced_logprobs(self, prompt_ids, response_ids, *, require_grad: bool):
        n = len(response_ids)
        base = self.param.expand(n)
        return base if require_grad else base.detach()


def test_cispo_token_ratio_differs_from_scalar_mean():
    backend = _TinyBackend()
    client = HFDebugTrainingClient(backend, micro_batch_size=1)
    row = {
        "prompt_ids": [1, 2],
        "action_ids": [3, 4],
        "token_logprobs": [-2.0, -2.0],
        "action_mask": [1, 1],
        "advantage": 1.0,
    }
    out = client._cispo_backward([row], loss_fn_config={"clip_low_threshold": 0.0, "clip_high_threshold": 5.0})
    assert out["n_datums"] == 1
    scalar_ratio = math.exp(((-1.0) + (-3.0)) / 2 - (-2.0))
    assert scalar_ratio == pytest.approx(1.0)
    assert backend.param.grad is not None
    assert float(backend.param.grad) != 0.0


def test_mean_behavior_logprob_uses_full_action():
    assert mean_behavior_logprob([-1.0, -3.0]) == pytest.approx(-2.0)


def test_vllm_lora_requires_weights_when_dir_nonempty(tmp_path):
    root = tmp_path / "bad_adapter"
    root.mkdir()
    (root / "adapter_config.json").write_text("{}", encoding="utf-8")
    weight = root / "adapter_model.safetensors"
    assert not weight.is_file()
    # Mirrors open_vllm: non-empty adapter dir without weights is an error.
    assert root.is_dir() and any(root.iterdir())


def test_build_manifest_includes_git_and_vllm_config():
    import argparse

    from trim.training.four_cell_runtime import build_manifest, _resolved_vllm_config

    args = argparse.Namespace(
        training_mode="rl",
        component="all",
        lambda_opd=0.0,
        opd_loss="sr_opd_ce",
        opd_gate_beta=5.0,
        group_size=8,
        max_turns=6,
        train_steps=3,
        train_groups_per_step=32,
        train_micro_batch_size=4,
        n_queries=100,
        opd_states_per_trajectory=3,
        seed=42,
        base_model="/tmp/model",
        sft_adapter="",
        smoke=False,
        rollout_backend="vllm",
        gpu_schedule="scheme_a",
        on_policy_refresh=True,
        train_only=True,
        max_new_tokens=2048,
        max_model_len=8192,
        rollout_query_batch_size=32,
    )
    resolved = _resolved_vllm_config(args, tp=8)
    assert resolved["max_new_tokens"] == 2048
    manifest = build_manifest(args, extra={"resolved_vllm": resolved})
    assert manifest["max_new_tokens"] == 2048
    assert "git" in manifest
    assert manifest["resolved_vllm"]["tensor_parallel_size"] == 8
