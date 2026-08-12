"""Config diff / single-factor drift tests."""

from __future__ import annotations

import pytest

from experiments.common.config_diff import assert_single_factor_diff, config_diff
from experiments.common.spec import ExperimentSpec


def _spec(**kw):
    base = dict(
        experiment_id="a",
        group="g",
        method="m",
        capability="duplicate_evidence",
        variant="a",
        changed_factor="none",
        base_model="Qwen2.5-7B-Instruct",
        dataset="browsecomp_plus",
        runtime_config="harness/configs/modules_minimal_v2.yaml",
        seed=42,
        output_dir="outputs/iclr_ablations/g/a/seed_42",
        objective="discriminative_ce",
        lora_rank=64,
    )
    base.update(kw)
    return ExperimentSpec(**base)


def test_detects_undeclared_drift():
    a = _spec()
    b = _spec(
        experiment_id="b",
        variant="b",
        changed_factor="objective",
        objective="operation_ce",
        lora_rank=16,  # undeclared
        output_dir="outputs/iclr_ablations/g/b/seed_42",
    )
    with pytest.raises(ValueError):
        assert_single_factor_diff(a, b)


def test_allows_declared_objective_change():
    a = _spec()
    b = _spec(
        experiment_id="b",
        variant="b",
        changed_factor="objective",
        objective="operation_ce",
        output_dir="outputs/iclr_ablations/g/b/seed_42",
    )
    d = assert_single_factor_diff(a, b)
    assert d["ok"]
    assert config_diff(a, b)["n_changed"] >= 1
