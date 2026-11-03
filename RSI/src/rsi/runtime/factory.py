"""Single runtime factory: every entrypoint builds the same loop parts.

Eval (serial), parallel panel workers, evolve panel_runner, the collector and
the OPD teacher previously each parsed two or three runtime keys and silently
fell back to defaults for the rest — e.g. ``max_chars`` from the config never
reached the ContextPolicy, so a config asking for a 24000-char window kept
rendering 8000.  Build everything here once; callers pass the parsed sections
through.

PR-1 of the 2026-11-03 audit (docs/audit/20261103_measurement_review.md §4.4).
"""

from __future__ import annotations

from dataclasses import dataclass

from rsi.contracts.policy import SamplingParams
from rsi.runtime.budgets import BudgetSpec
from rsi.runtime.context import ContextPolicy
from rsi.runtime.profiles import H0Profile, profile_from_runtime_config


@dataclass(frozen=True)
class RuntimeParts:
    budgets: BudgetSpec
    sampling: SamplingParams
    context_policy: ContextPolicy
    profile: H0Profile


def build_runtime_parts(
    runtime_cfg: dict,
    *,
    max_chars_default: int = 8000,
) -> RuntimeParts:
    """Parse the ``runtime`` config section into the three loop parts.

    Every key the runtime contract defines is read here; nothing falls back to
    a locally-copied default that can drift from the config.
    """
    budgets = BudgetSpec(
        max_actions=int(runtime_cfg.get("max_actions", 20)),
        max_extra_llm_calls=int(runtime_cfg.get("max_extra_llm_calls", 2)),
        max_actor_output_tokens=int(runtime_cfg.get("max_actor_output_tokens", 4096)),
        max_tool_calls=int(runtime_cfg.get("max_tool_calls", 40)),
        max_wall_ms=float(runtime_cfg.get("max_wall_ms", 600_000.0)),
        max_network_retries=int(runtime_cfg.get("max_network_retries", 2)),
    )
    sampling = SamplingParams(
        max_new_tokens=int(runtime_cfg.get("max_new_tokens", 256)),
        temperature=float(runtime_cfg.get("temperature", 1.0)),
    )
    # Guide §5.2: runtime.profile is REQUIRED — no silent legacy fallback.
    profile: H0Profile = profile_from_runtime_config(runtime_cfg)
    context_policy = ContextPolicy(
        max_chars=int(runtime_cfg.get("max_chars", max_chars_default)),
        profile=profile,
    )
    return RuntimeParts(budgets=budgets, sampling=sampling, context_policy=context_policy, profile=profile)
