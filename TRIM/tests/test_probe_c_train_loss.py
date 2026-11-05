"""Projected-gap loss numerics and reset/restore (§16.4, §16.7)."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from trim.probes.probe_c_train import (  # noqa: E402
    CandidateResetManager,
    MicroUpdateTrainer,
    TrainContractError,
    checksum_payload,
    effective_weight_sum,
    frozen_projected_gap_numerator,
    plan_micro_batches,
    scale_c1_loss,
)
from trim.training.sr_opd_loss import (  # noqa: E402
    gated_action_gap_per_token,
    gated_action_gap_weighted_sum,
    scale_projected_gap_loss,
)
from trim.probes.probe_c_cache import CachedSupervisionRow, TeacherTarget  # noqa: E402
from trim.probes.probe_c_data import validate_state_row  # noqa: E402
from trim.probes.probe_c_contract import make_add_pair  # noqa: E402
from tests.probe_c_fixtures import make_state_row  # noqa: E402


def test_loss_matches_reference_projected_gap():
    s = torch.tensor([-1.0, -2.0, -0.5], requires_grad=True)
    t = torch.tensor([-0.5, -2.5, -0.5])
    w = torch.tensor([1.0, 0.0, 2.0])
    beta = 5.0
    mine = frozen_projected_gap_numerator(s, t, w, gate_beta=beta)
    ref = gated_action_gap_weighted_sum(s, t, w, gate_beta=beta)
    assert torch.allclose(mine, ref)
    # Manual per-token expansion: g_j (ℓT_j − ℓS_j) with detached gate.
    expected = 0.0
    for j in range(3):
        delta = (t[j] - s[j]).detach()
        g = torch.sigmoid(delta * beta)
        expected = expected + w[j] * g * (t[j] - s[j])
    assert torch.allclose(mine, expected.reshape(()))


def test_teacher_detached_student_has_grad():
    s = torch.tensor([-1.0, -2.0], requires_grad=True)
    t = torch.tensor([-0.3, -0.4], requires_grad=True)  # even if attached
    w = torch.tensor([1.0, 1.0])
    numer = frozen_projected_gap_numerator(s, t, w)
    loss = scale_c1_loss(numer, z_gap=2.0, loss_scale=1.0, world_size=1)
    loss.backward()
    assert s.grad is not None and float(s.grad.abs().sum()) > 0
    assert t.grad is None or float(t.grad.abs().sum()) == 0


def test_scale_matches_formal_contract():
    numer = torch.tensor(2.0)
    got = scale_c1_loss(numer, z_gap=4.0, loss_scale=1.0, world_size=2)
    ref = scale_projected_gap_loss(numer, lambda_opd=1.0, z_gap=4.0, world_size=2)
    assert torch.allclose(got, ref)
    assert math.isclose(float(got), 1.0)  # 2 * 1 * 2 / 4


def test_z_zero_skips_without_clamp():
    numer = torch.tensor(2.0, requires_grad=True)
    loss = scale_c1_loss(numer, z_gap=0.0, loss_scale=1.0, world_size=1)
    assert float(loss) == 0.0
    # zero gradient — no clamp_min(1) resurrection
    loss.backward()
    assert float(numer.grad) == 0.0


def test_effective_weight_sum_excludes_zero_weights():
    assert effective_weight_sum([1.0, 0.0, 2.0]) == 3.0
    assert effective_weight_sum([1.0, 0.0], [True, True]) == 1.0


def _rows(n: int = 3) -> list[CachedSupervisionRow]:
    pair = make_add_pair(["sentence_compress"], git_sha="g1")
    out = []
    for i in range(n):
        state = validate_state_row(
            make_state_row(
                query_id=f"q{i}",
                student_mask=pair.student_mask,
                teacher_mask=pair.teacher_mask,
            ),
            pair=pair,
        )
        target = TeacherTarget(
            cache_key=f"k{i}",
            state_id=state.state_id,
            query_id=state.query_id,
            component_id="sentence_compress",
            target_action={"name": "curate", "arguments": {"add_ids": ["d1"]}},
            target_action_ids=[10, 11],
            teacher_token_logprobs=[-0.1, -0.2],
            weights=[1.0, 1.0],
            student_prompt_ids_hash="ph",
        )
        out.append(CachedSupervisionRow(state=state, target=target))
    return out


def test_plan_micro_batches_token_budget():
    rows = _rows(4)  # 8 effective tokens total (2 each)
    plan = plan_micro_batches(rows, action_tokens_per_optimizer_step=4)
    assert len(plan.batches) == 2
    assert plan.tokens_per_batch == [4, 4]
    plan2 = plan_micro_batches(rows, action_tokens_per_optimizer_step=1024)
    assert plan2.leftover_tokens == 8
    assert plan2.n_unique_tokens == 8
    assert plan2.n_reused_rows == 0
    # Repeated sampling is flagged as reuse, not new data.
    plan3 = plan_micro_batches(rows, action_tokens_per_optimizer_step=1024, repeat=3)
    assert plan3.n_reused_rows == 8


def test_microbatch_accumulation_equals_single_shot():
    """Grad-accum with one global Z == single batch scaled loss (§7.3/7.4)."""

    def make_model():
        torch.manual_seed(0)
        return torch.nn.Linear(4, 3)

    s_tokens = [torch.randn(3, 4) for _ in range(2)]
    targets = [torch.tensor([0, 1, 2]), torch.tensor([2, 1, 0])]
    t_logp = [torch.tensor([-0.1, -0.2, -0.3]), torch.tensor([-0.3, -0.2, -0.1])]
    weights = [torch.tensor([1.0, 1.0, 1.0]), torch.tensor([1.0, 0.0, 1.0])]
    z = float(sum(float(w.sum()) for w in weights))

    # Single shot: concatenate along the token dimension.
    m1 = make_model()
    s_cat = torch.cat(s_tokens, dim=0)
    tgt_cat = torch.cat(targets)
    t_cat = torch.cat(t_logp)
    w_cat = torch.cat(weights)
    logits = m1(s_cat)
    student = logits.gather(-1, tgt_cat.unsqueeze(-1)).squeeze(-1)
    loss1 = scale_c1_loss(
        frozen_projected_gap_numerator(student, t_cat, w_cat),
        z_gap=z,
        loss_scale=1.0,
        world_size=1,
    )
    loss1.backward()
    g1 = m1.weight.grad.clone()

    # Accumulated: same global Z per microbatch.
    m2 = make_model()
    for s_tok, tgt, t_lp, w in zip(s_tokens, targets, t_logp, weights):
        logits = m2(s_tok)
        student = logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        loss_i = scale_c1_loss(
            frozen_projected_gap_numerator(student, t_lp, w),
            z_gap=z,
            loss_scale=1.0,
            world_size=1,
        )
        loss_i.backward()
    g2 = m2.weight.grad.clone()
    assert torch.allclose(g1, g2, atol=1e-6)


def test_z_zero_batch_reports_no_support():
    class _Backend:
        def __init__(self):
            self.calls = 0

        def token_logprobs(self, p, t):
            return [-0.1] * len(t)

        def train_microbatch(self, **kw):
            self.calls += 1
            return {"numerator": 0.0}

        def optimizer_step(self):
            return {"grad_norm": 1.0}

        def trainable_checksum(self):
            return "cs"

        def save_trainable(self):
            return {"w": [1.0]}

        def load_trainable(self, payload):
            return None

    rows = _rows(1)
    rows[0].target.weights = [0.0, 0.0]
    backend = _Backend()
    trainer = MicroUpdateTrainer(backend, action_tokens_per_optimizer_step=1)
    run = trainer.train_to_budget(rows, target_steps=1, component_id="m")
    assert backend.calls == 0
    assert run.steps[0].skipped is True
    assert run.steps[0].skip_reason == "NO_SUPPORT"


def test_candidate_reset_restores_theta0():
    class _Backend:
        def __init__(self):
            self.state = {"w": [1.0, 2.0]}
            self.opt = {"mom": 0.0}

        def token_logprobs(self, p, t):
            return [-0.1] * len(t)

        def train_microbatch(self, **kw):
            self.state = {"w": [9.0, 9.0]}
            self.opt = {"mom": 9.0}
            return {"numerator": 1.0}

        def optimizer_step(self):
            return {"grad_norm": 1.0}

        def trainable_checksum(self):
            return checksum_payload(self.save_trainable())

        def save_trainable(self):
            return {"model": dict(self.state), "optimizer": dict(self.opt)}

        def load_trainable(self, payload):
            self.state = dict(payload["model"])
            self.opt = dict(payload["optimizer"])

    backend = _Backend()
    reset = CandidateResetManager(backend)
    theta0 = reset.capture_theta0()
    reset.reset_exact(seed=1)
    reset.verify_theta0()
    backend.train_microbatch()  # candidate A "training"
    with pytest.raises(TrainContractError, match="theta0 checksum drift"):
        reset.verify_theta0()
    reset.reset_exact(seed=2)  # candidate B restarts from the same θ0
    reset.verify_theta0()
    assert reset.theta0_checksum == theta0
    assert reset.n_resets == 2


def test_verify_updated_detects_no_movement():
    class _Backend:
        def token_logprobs(self, p, t):
            return [-0.1] * len(t)

        def train_microbatch(self, **kw):
            return {"numerator": 0.0}

        def optimizer_step(self):
            return {}

        def trainable_checksum(self):
            return checksum_payload(self.save_trainable())

        def save_trainable(self):
            return {"model": {"w": 1.0}}

        def load_trainable(self, payload):
            return None

    backend = _Backend()
    reset = CandidateResetManager(backend)
    reset.capture_theta0()
    with pytest.raises(TrainContractError, match="unchanged"):
        reset.verify_updated()


def test_gap_length_mismatch_is_error():
    from trim.training.sr_opd_loss import gated_action_gap_per_token

    with pytest.raises(ValueError, match="length mismatch"):
        gated_action_gap_per_token(
            torch.tensor([-1.0, -1.0]), torch.tensor([-1.0])
        )
