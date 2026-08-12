"""Baseline adapter dry-run tests."""

from __future__ import annotations

from experiments.baselines.adapters.seed_adapter import SeedAdapter
from experiments.baselines.adapters.scope_env_adapter import ScopeEnvAdapter
from experiments.common.registry import ExperimentRegistry


def test_scope_env_dry_run():
    reg = ExperimentRegistry()
    spec = reg.resolve("b0_base_hmin", dry_run=True, smoke_query_limit=2)
    dry = ScopeEnvAdapter().dry_run(spec)
    assert dry["dry_run"] is True
    assert dry["command"]


def test_seed_adapter_builds_command(tmp_path):
    reg = ExperimentRegistry()
    spec = reg.resolve(
        "b_seed_dryrun",
        dry_run=True,
        smoke_query_limit=2,
        output_dir=str(tmp_path / "seed"),
    )
    adapter = SeedAdapter()
    dry = adapter.dry_run(spec)
    assert "SEED" in " ".join(dry["command"]) or dry["command"]
    assert dry["data"]["n_queries"] == 2
