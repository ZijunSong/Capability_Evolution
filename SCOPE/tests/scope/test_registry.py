"""Registry tests."""

from __future__ import annotations

from experiments.common.registry import ExperimentRegistry


def test_registry_loads_and_resolves():
    reg = ExperimentRegistry()
    assert len(reg.ids()) >= 50
    spec = reg.resolve("a1_same_state_on_policy")
    assert spec.variant == "a1_same_state_on_policy"
    assert spec.output_dir
    assert "iclr_ablations" in spec.output_dir


def test_registry_validate():
    reg = ExperimentRegistry()
    errs = reg.validate()
    assert errs == []


def test_by_group():
    reg = ExperimentRegistry()
    ids = reg.by_group("a1_supervision_source")
    assert "a1_same_state_on_policy" in ids
    assert len(ids) == 4
