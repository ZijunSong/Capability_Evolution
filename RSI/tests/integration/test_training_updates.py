"""Training acceptance: test_joint_step_count / test_opd_affects_update /
test_export_reload_equivalence / test_distributed_normalization /
test_exact_prompt_ids.  Uses a real tiny HF model (marked integration: real
gradients, CPU-sized)."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsi.opd.losses import gated_projected_gap_loss
from rsi.opd.records import OPDRow, RLRow, build_opd_row, build_rl_row
from rsi.providers.tiny_hf import HFPolicy, build_tiny_model
from rsi.training.backend import GlobalNormalizers
from rsi.training.export import export_adapter, verify_export_reload_equivalence
from rsi.training.hf_reference import HFReferenceBackend
from rsi.training.joint import MethodSpec, joint_update

pytestmark = pytest.mark.integration


def _mk_rl_row(i: int) -> RLRow:
    return RLRow(
        row_id=f"rl{i}",
        episode_id="ep",
        decision_index=i,
        prompt_token_ids=(256, 65, 66, 67),
        action_token_ids=(68, 69),
        old_logprobs=(-1.0, -1.2),
        action_mask=(1, 1),
        advantage=0.5,
        reward_ref="t",
        behavior_policy_version="p0",
    )


def _mk_opd_row(i: int, teacher_lp=(-0.3, -0.4)) -> OPDRow:
    return OPDRow(
        row_id=f"opd{i}",
        episode_id="ep",
        decision_index=i,
        student_prompt_token_ids=(256, 65, 66, 67),
        teacher_prompt_token_ids=(256, 67, 68, 69),
        target_token_ids=(68, 69),
        target_mask=(1, 1),
        target_action=None,
        target_kind="native",
        alignment_route="native_anchor",
        source_event_ids=("e0",),
        realizability_json={},
        provenance_json={},
        token_weight=(1.0, 1.0),
        student_logprobs=(-1.0, -1.1),
        teacher_logprobs=teacher_lp,
    )


def test_joint_step_count():
    torch.manual_seed(0)
    model = build_tiny_model()
    backend = HFReferenceBackend(model=model, lr=1e-2)
    policy_snapshot = copy.deepcopy(model.state_dict())
    rl_rows = [_mk_rl_row(0), _mk_rl_row(1)]
    opd_rows = [_mk_opd_row(0)]
    spec = MethodSpec.resolve("harness_opd_rl")
    before_steps = backend.step_count
    report = joint_update(backend, rl_rows=rl_rows, opd_rows=opd_rows, spec=spec, lambda_opd=0.1)
    assert report.n_optimizer_steps == 1
    assert backend.step_count == before_steps + 1  # one step, not two
    assert report.n_rl_rows == 2 and report.n_opd_rows == 1
    # weights changed
    changed = any(
        not torch.equal(policy_snapshot[k], v) for k, v in model.state_dict().items()
    )
    assert changed


def test_opd_affects_update():
    torch.manual_seed(0)
    model_a = build_tiny_model()
    model_b = copy.deepcopy(model_a)
    rows_rl = [_mk_rl_row(0)]
    rows_opd = [_mk_opd_row(0, teacher_lp=(-0.05, -0.05))]  # teacher much better

    def run(model, lambda_opd):
        backend = HFReferenceBackend(model=model, lr=1e-2)
        spec = MethodSpec.resolve("harness_opd_rl")
        joint_update(backend, rl_rows=rows_rl, opd_rows=rows_opd, spec=spec, lambda_opd=lambda_opd)
        return copy.deepcopy(model.state_dict())

    state0 = run(model_a, 0.0)
    state1 = run(model_b, 1.0)
    diffs = [
        float((state0[k] - state1[k]).abs().sum())
        for k in state0
        if state0[k].dtype.is_floating_point
    ]
    assert max(diffs) > 0.0  # non-zero OPD supervision changed the update


def test_export_reload_equivalence(tmp_path):
    torch.manual_seed(0)
    model = build_tiny_model()
    backend = HFReferenceBackend(model=model, lr=1e-2)
    joint_update(backend, rl_rows=[_mk_rl_row(0)], opd_rows=[_mk_opd_row(0)], spec=MethodSpec.resolve("harness_opd_rl"), lambda_opd=0.1)
    payload = export_adapter(backend, tmp_path / "export")
    assert (tmp_path / "export" / "policy_artifact.json").is_file()

    reloaded = HFPolicy(build_tiny_model())  # fresh random init
    state = torch.load(tmp_path / "export" / "model.pt", map_location="cpu", weights_only=True)
    reloaded.model.load_state_dict(state)
    result = verify_export_reload_equivalence(
        HFPolicy(model=backend.model), reloaded, probe_prompt="probe prompt", probe_target="target"
    )
    assert result["ok"]


def test_distributed_normalization():
    """Single-process vs two-rank weighted-sum form: no double global denom."""
    rows = [_mk_opd_row(0), _mk_opd_row(1)]
    norm1 = GlobalNormalizers.from_rows([_mk_rl_row(0)], rows, world_size=1)
    norm2 = GlobalNormalizers.from_rows([_mk_rl_row(0)], rows, world_size=2)
    assert norm1.global_opd_weight == norm2.global_opd_weight == 4.0
    assert norm1.global_action_tokens == 2

    numer = torch.tensor(1.0)
    # world-size mean-reduction undo: gradient-equivalent totals across ranks
    total_rank_sum = sum(
        float(
            gated_projected_gap_loss(
                torch.tensor([0.0], requires_grad=True),
                torch.tensor([1.0]),
                torch.ones(1),
                torch.ones(1),
            ).detach()
        )
        for _ in range(2)
    )
    from rsi.opd.losses import scale_projected_gap_loss

    s2 = scale_projected_gap_loss(numer, lambda_opd=1.0, z_gap=norm2.global_opd_weight, world_size=2)
    s1 = scale_projected_gap_loss(numer, lambda_opd=1.0, z_gap=norm1.global_opd_weight, world_size=1)
    # 2-rank form scales by world_size to undo mean-reduction
    assert abs(float(s2.detach()) - 2 * float(s1.detach())) < 1e-9
    assert total_rank_sum > 0  # sanity: the gap term is non-zero


def test_exact_prompt_ids(tmp_path):
    """Training must consume the recorded prefixes; a wrong re-render is
    rejected by construction (rows only carry recorded ids)."""
    from rsi.contracts.trace import DecisionRecord

    d = DecisionRecord(
        campaign_id="c",
        round_index=0,
        batch_index=0,
        episode_id="e",
        decision_index=0,
        environment_snapshot_ref="s",
        public_view_digest="v",
        runtime_contract_digest="rt",
        policy_digest="p",
        tokenizer_digest="tb",
        chat_template_digest="cb",
        bank_digest="b",
        student_prompt_token_ids=(256, 104, 105),
        sampled_action_token_ids=(123, 124),
        old_logprobs=(-1.0, -1.0),
        action_mask=(1, 1),
    )
    row = build_rl_row(d, advantage=0.1, reward_ref="t")
    assert row.prompt_token_ids == (256, 104, 105)  # exact recorded ids, not re-rendered
    assert row.action_token_ids == (123, 124)
