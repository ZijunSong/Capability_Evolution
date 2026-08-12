"""ExperimentSpec tests."""

from __future__ import annotations

import pytest

from experiments.common.spec import ExperimentSpec


def _minimal(**kw):
    base = dict(
        experiment_id="t1",
        group="g",
        method="m",
        capability="duplicate_evidence",
        variant="v",
        changed_factor="none",
        base_model="Qwen2.5-7B-Instruct",
        dataset="browsecomp_plus",
        runtime_config="harness/configs/modules_minimal_v2.yaml",
        seed=42,
        output_dir="outputs/iclr_ablations/g/v/seed_42",
    )
    base.update(kw)
    return ExperimentSpec(**base)


def test_required_fields():
    with pytest.raises(ValueError):
        ExperimentSpec(
            experiment_id="x",
            group="g",
            method="m",
            capability="c",
            variant="v",
            changed_factor="none",
            base_model="m",
            dataset="d",
            runtime_config="r",
            seed=1,
            output_dir="",
        )


def test_roundtrip_yaml(tmp_path):
    s = _minimal()
    path = tmp_path / "s.yaml"
    s.to_yaml(path)
    s2 = ExperimentSpec.from_yaml(path)
    assert s2.experiment_id == s.experiment_id
    assert s2.lora_rank == 64
    assert s2.rollout_seed == 42


def test_extras_unknown_keys():
    s = ExperimentSpec.from_dict(
        {
            **_minimal().to_dict(),
            "custom_knob": 123,
        }
    )
    assert s.extras["custom_knob"] == 123
