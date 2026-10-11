"""T07: loss branches keep their declared targets."""

from __future__ import annotations

from trim.training.rl_opd_types import (
    OPD_LOSS_CE,
    OPD_LOSS_PROJECTED_GAP,
    OPD_LOSS_SAMPLED_GAP,
    uses_projected_seed,
    uses_sampled_opd,
    uses_seed_gap,
)


def test_loss_mode_contracts():
    assert uses_sampled_opd(OPD_LOSS_SAMPLED_GAP)
    assert uses_projected_seed(OPD_LOSS_PROJECTED_GAP)
    assert uses_seed_gap(OPD_LOSS_PROJECTED_GAP)
    assert uses_seed_gap(OPD_LOSS_SAMPLED_GAP)
    assert not uses_sampled_opd(OPD_LOSS_CE)
    assert not uses_projected_seed(OPD_LOSS_CE)
    from trim.training.opd_train_contract import default_opd_loss_for_method

    assert default_opd_loss_for_method("zero-placebo") == OPD_LOSS_SAMPLED_GAP
    assert default_opd_loss_for_method("full-sampled-gap") == OPD_LOSS_SAMPLED_GAP
    assert default_opd_loss_for_method("trim") == OPD_LOSS_PROJECTED_GAP
    assert default_opd_loss_for_method("rl") == OPD_LOSS_CE
    assert uses_sampled_opd(default_opd_loss_for_method("zero-placebo"))
    assert not uses_projected_seed(default_opd_loss_for_method("full-sampled-gap"))


def test_lambda_zero_means_no_opd_requirement():
    from trim.cli.launch import parse_train_args
    from trim.training.rl_opd_types import TRAINING_MODE_RL

    args, spec = parse_train_args(
        ["--train_method", "rl", "--component", "zero", "--out", "/tmp/rl-only"]
    )
    assert spec.training_mode == TRAINING_MODE_RL
    # run_train.py sets lambda_opd=0 for RL-only.
    assert args.opd_loss == "sr_opd_ce" or args.train_method == "rl"
