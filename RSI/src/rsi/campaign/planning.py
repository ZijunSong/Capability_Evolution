"""Campaign plan artifacts (``campaign plan``).

Plan artifacts always carry ``executed=false``.  ``campaign run`` never
substitutes a plan for an execution; missing backend capability is a non-zero
exit, not a successful plan report.
"""

from __future__ import annotations

from typing import Any

from rsi.config import CampaignConfig
from rsi.campaign.state_machine import MAIN_STATES


def build_plan(cfg: CampaignConfig) -> dict[str, Any]:
    suite = cfg.section("suite")
    evo = cfg.section("evolution")
    training = cfg.section("training")
    return {
        "schema_version": "rsi_campaign_plan_v1",
        "campaign_id": cfg.campaign_id,
        "executed": False,
        "config_digest": cfg.digest(),
        "max_rounds": cfg.max_rounds,
        "planned_states": list(MAIN_STATES),
        "domains": list(suite.get("domains") or []),
        "shared_model": suite.get("shared_model", True),
        "shared_bank": suite.get("shared_bank", True),
        "evolution": {
            "max_generations": evo.get("max_generations"),
            "candidates_per_generation": evo.get("candidates_per_generation"),
            "max_candidate_evaluations": evo.get("max_candidate_evaluations"),
            "seed_mode": evo.get("seed_mode"),
            "proposer_mode": evo.get("proposer_mode"),
        },
        "training": {
            "method": training.get("method"),
            "backend": training.get("backend"),
            "max_optimizer_steps_per_round": training.get("max_optimizer_steps_per_round"),
        },
        "notes": "Plan only. Numbers here are planned budgets, not results.",
    }
