from experiments.baselines.adapters.base import BaselineAdapter
from experiments.baselines.adapters.opid_adapter import OpidAdapter
from experiments.baselines.adapters.scope_env_adapter import ScopeEnvAdapter
from experiments.baselines.adapters.sdar_adapter import SdarAdapter
from experiments.baselines.adapters.seed_adapter import SeedAdapter

ADAPTERS = {
    "SEED": SeedAdapter,
    "OPID": OpidAdapter,
    "SDAR": SdarAdapter,
    "scope_env": ScopeEnvAdapter,
}

__all__ = ["BaselineAdapter", "ADAPTERS", "SeedAdapter", "OpidAdapter", "SdarAdapter", "ScopeEnvAdapter"]
