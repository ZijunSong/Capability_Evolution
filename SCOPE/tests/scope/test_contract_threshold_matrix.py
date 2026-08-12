"""A8 contract × threshold matrix tests."""

from __future__ import annotations

from experiments.common.registry import ExperimentRegistry


def test_a8_four_variants_present():
    reg = ExperimentRegistry()
    ids = reg.by_group("a8_contract_threshold")
    assert set(ids) == {
        "a8_old_contract_tau0",
        "a8_old_contract_calibrated_tau",
        "a8_fixed_contract_tau0",
        "a8_fixed_contract_calibrated_tau",
    }
    for eid in ids:
        spec = reg.resolve(eid)
        assert spec.changed_factor == "contract_threshold"
        assert "contract" in spec.extras
