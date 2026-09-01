from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import torch

from scape.training.rl_opd_policy_version import PolicyVersionMismatch
from scape.training.rl_opd_types import UPDATE_OPD_ONLY_ZERO_RL, UPDATE_RL_ONLY, UPDATE_RL_OPD_JOINT
from scape.training.tinker_opd_datum import TinkerOPDDatum
from scape.training.tinker_rl_opd_trainer import hybrid_train_substep


@dataclass
class FakeTrainingClient:
    dim: int = 4
    lr: float = 1.0
    W: torch.Tensor = field(init=False)
    grad: torch.Tensor = field(init=False)
    calls: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self.W = torch.zeros(self.dim)
        self.grad = torch.zeros(self.dim)

    async def forward_backward_async(self, data, loss_fn, loss_fn_config=None):
        self.calls.append(("fb", loss_fn, len(list(data))))
        if loss_fn == "cispo":
            g = torch.tensor([1.0, 0.0, 0.0, 0.0])
            loss = 1.0
        else:
            scale = 0.0
            for row in data:
                weights = getattr(row, "weights", None)
                if weights is None and isinstance(row, dict):
                    weights = row.get("weights") or []
                scale += float(sum(weights or [1.0]))
            g = torch.tensor([0.0, 1.0, 0.0, 0.0]) * (scale or 1.0)
            loss = float(scale or 1.0)
        self.grad = self.grad + g
        return {"loss": loss}

    async def optim_step_async(self, adam_params):
        del adam_params
        self.calls.append(("opt",))
        self.W = self.W - self.lr * self.grad
        self.grad = torch.zeros(self.dim)


def _opd(n_tok: int, weight: float) -> TinkerOPDDatum:
    return TinkerOPDDatum(
        model_input="prefix",
        prompt_token_ids=[0],
        target_tokens=[0] + [1] * n_tok,
        weights=[0.0] + [weight / n_tok] * n_tok,
        policy_version="v1",
        n_supervised_tokens=n_tok,
    )


def _run(**kwargs):
    return asyncio.run(hybrid_train_substep(**kwargs))


def test_hybrid_substep_one_optimizer():
    client = FakeTrainingClient()
    metrics = _run(
        training_client=client,
        rl_datums=[{"n_tokens": 4}],
        opd_datums=[_opd(2, 0.1)],
        rl_loss_fn="cispo",
        rl_loss_fn_config={"clip_high_threshold": 5},
        lambda_opd=0.1,
        adam_params={},
        policy_version="v17",
    )
    assert client.calls == [
        ("fb", "cispo", 1),
        ("fb", "cross_entropy", 1),
        ("opt",),
    ]
    assert metrics.n_rl_forward_backward == 1
    assert metrics.n_opd_forward_backward == 1
    assert metrics.n_optimizer_steps == 1
    assert metrics.update_type == UPDATE_RL_OPD_JOINT


def test_scape_rl_substep_uses_reverse_kl_fb():
    client = FakeTrainingClient()
    metrics = _run(
        training_client=client,
        rl_datums=[{"n_tokens": 4}],
        opd_datums=[_opd(2, 0.1)],
        rl_loss_fn="cispo",
        rl_loss_fn_config={"clip_high_threshold": 5},
        lambda_opd=0.1,
        adam_params={},
        policy_version="v17",
        opd_loss="sr_opd_reverse_kl",
    )
    assert client.calls == [
        ("fb", "cispo", 1),
        ("fb", "reverse_kl", 1),
        ("opt",),
    ]
    assert metrics.n_opd_forward_backward == 1


def test_lambda_zero_skips_opd_fb_when_no_opd_datums():
    client = FakeTrainingClient()
    metrics = _run(
        training_client=client,
        rl_datums=[{"n_tokens": 3}],
        opd_datums=[],
        rl_loss_fn="cispo",
        rl_loss_fn_config={},
        lambda_opd=0.0,
        adam_params={},
        policy_version="v1",
    )
    assert client.calls == [("fb", "cispo", 1), ("opt",)]
    assert metrics.update_type == UPDATE_RL_ONLY
    assert metrics.n_opd_forward_backward == 0


def test_reject_all_still_runs_rl():
    client = FakeTrainingClient()
    metrics = _run(
        training_client=client,
        rl_datums=[{"n_tokens": 2}, {"n_tokens": 2}],
        opd_datums=[],
        rl_loss_fn="cispo",
        rl_loss_fn_config={},
        lambda_opd=0.1,
        adam_params={},
        policy_version="v1",
    )
    assert client.calls == [("fb", "cispo", 2), ("opt",)]
    assert metrics.update_type == UPDATE_RL_ONLY


def test_constant_reward_opd_only():
    client = FakeTrainingClient()
    metrics = _run(
        training_client=client,
        rl_datums=[],
        opd_datums=[_opd(3, 0.2)],
        rl_loss_fn="cispo",
        rl_loss_fn_config={},
        lambda_opd=0.2,
        adam_params={},
        policy_version="v1",
    )
    assert client.calls == [("fb", "cross_entropy", 1), ("opt",)]
    assert metrics.update_type == UPDATE_OPD_ONLY_ZERO_RL
    assert metrics.n_rl_tokens == 0
    assert metrics.n_opd_tokens > 0


def test_joint_update_differs_from_single_branch():
    rl = [{"n_tokens": 4}]
    opd = [_opd(2, 0.5)]

    a = FakeTrainingClient()
    _run(
        training_client=a,
        rl_datums=rl,
        opd_datums=[],
        rl_loss_fn="cispo",
        rl_loss_fn_config={},
        lambda_opd=0.5,
        adam_params={},
        policy_version="v1",
    )
    b = FakeTrainingClient()
    _run(
        training_client=b,
        rl_datums=[],
        opd_datums=opd,
        rl_loss_fn="cispo",
        rl_loss_fn_config={},
        lambda_opd=0.5,
        adam_params={},
        policy_version="v1",
    )
    ab = FakeTrainingClient()
    _run(
        training_client=ab,
        rl_datums=rl,
        opd_datums=opd,
        rl_loss_fn="cispo",
        rl_loss_fn_config={},
        lambda_opd=0.5,
        adam_params={},
        policy_version="v1",
    )
    assert not torch.allclose(ab.W, a.W)
    assert not torch.allclose(ab.W, b.W)

    small = FakeTrainingClient()
    _run(
        training_client=small,
        rl_datums=rl,
        opd_datums=[_opd(2, 0.1)],
        rl_loss_fn="cispo",
        rl_loss_fn_config={},
        lambda_opd=0.1,
        adam_params={},
        policy_version="v1",
    )
    large = FakeTrainingClient()
    _run(
        training_client=large,
        rl_datums=rl,
        opd_datums=[_opd(2, 1.0)],
        rl_loss_fn="cispo",
        rl_loss_fn_config={},
        lambda_opd=1.0,
        adam_params={},
        policy_version="v1",
    )
    assert abs(float(large.W[1])) > abs(float(small.W[1]))


def test_version_mismatch_before_any_fb():
    client = FakeTrainingClient()
    try:
        _run(
            training_client=client,
            rl_datums=[{"n_tokens": 1}],
            opd_datums=[],
            rl_loss_fn="cispo",
            rl_loss_fn_config={},
            lambda_opd=0.0,
            adam_params={},
            policy_version="v10",
            rollout_policy="v10",
            train_policy="v10",
            harness_teacher_policy="v11",
        )
        raise AssertionError("expected PolicyVersionMismatch")
    except PolicyVersionMismatch:
        pass
    assert client.calls == []
