"""CLI service for harness evolution (`rsi.cli evolve`)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rsi.config import CampaignConfig
from rsi.contracts.errors import ProtocolViolation
from rsi.evolution.archive import Archive
from rsi.evolution.packet import build_packet
from rsi.evolution.proposer import APIProposer, SelfProposer
from rsi.evolution.search import EvolutionBudgets, EvolutionSearch
from rsi.evaluation.service import run_panel
from rsi.harness.loader import load_bank_ref, load_empty_bank
from rsi.providers.factory import build_policy


def _build_proposer(cfg: CampaignConfig, policy: Any):
    evo = cfg.section("evolution")
    mode = str(evo.get("proposer_mode", "self"))
    if mode == "self":
        return SelfProposer(
            policy=policy,
            max_new_tokens=int(evo.get("proposer_max_new_tokens", 1024)),
            max_retries=int(evo.get("proposer_max_retries", 2)),
        )
    if mode == "assisted":
        api_cfg = cfg.section("providers") or cfg.raw.get("proposer_api") or {}
        if not api_cfg:
            raise ProtocolViolation("proposer_mode=assisted requires a providers/proposer_api section")
        from rsi.providers.openai_compat import OpenAICompatClient

        client = OpenAICompatClient(
            base_url=str(api_cfg["base_url"]),
            model=str(api_cfg["model"]),
            api_key_env=str(api_cfg.get("api_key_env", "RSI_API_KEY")),
        )
        return APIProposer(client=client)
    raise ProtocolViolation(f"unknown proposer_mode {mode!r} (fixture mode is test-only)")


def run_evolve(cfg: CampaignConfig, *, initial_bank_ref: str | None = None) -> dict[str, Any]:
    evo = cfg.section("evolution")
    runs_root = cfg.resolve_path(str(cfg.section("execution").get("runs_root", "runs")))
    assert runs_root is not None
    run_dir = runs_root / cfg.campaign_id
    run_dir.mkdir(parents=True, exist_ok=True)
    archive = Archive(run_dir / "evolve" / "archive")

    policy = build_policy(cfg)
    max_per_domain = evo.get("dev_panel_per_domain")

    def panel_runner(bank):
        result = run_panel(
            cfg,
            policy=policy,
            bank=bank,
            role=str(cfg.section("suite").get("selection_role", "harness_select")),
            seed=cfg.seed,
            max_per_domain=int(max_per_domain) if max_per_domain else None,
        )
        scores: dict[str, dict[str, float]] = {}
        for rec in result.records:
            scores.setdefault(rec.domain, {})[rec.task_id] = rec.value
        per_task = {
            tid: {
                "value": meta["value"],
                "terminated_reason": meta["terminated_reason"],
                "trigger_stats": {},
            }
            for tid, meta in result.per_task.items()
        }
        return {
            "scores": scores,
            "per_task": per_task,
            "cost": result.costs.to_json(),
            "protocol_ok": result.costs.infra_failures == 0,
            "budget_ok": True,
            "policy_digest": policy.policy_artifact.policy_id,
        }

    # Collect known task ids for the anti-task-id-constant audit.
    known_ids: list[str] = []
    from rsi.environments.registry import global_registry

    registry = global_registry()
    for domain in cfg.section("suite").get("domains") or []:
        provider = registry.get(domain).task_provider_factory()
        known_ids.extend(t.task_id for t in provider.public_tasks("harness_select"))
        known_ids.extend(t.task_id for t in provider.public_tasks("work_train"))

    initial_bank = load_bank_ref(initial_bank_ref) if initial_bank_ref else load_empty_bank()
    search = EvolutionSearch(
        proposer=_build_proposer(cfg, policy),
        archive=archive,
        panel_runner=panel_runner,
        packet_builder=build_packet,
        budgets=EvolutionBudgets(
            max_generations=int(evo.get("max_generations", 8)),
            candidates_per_generation=int(evo.get("candidates_per_generation", 2)),
            max_candidate_evaluations=int(evo.get("max_candidate_evaluations", 16)),
            plateau_patience=int(evo.get("plateau_patience", 3)),
        ),
        parent_bank_root=initial_bank.root if initial_bank_ref else None,
        candidates_root=run_dir / "candidates",
        known_task_ids=known_ids,
        seed_mode=str(evo.get("seed_mode", "empty_each_round")),
    )
    result = search.run()
    payload = result.to_json()
    payload["command"] = "evolve"
    payload["config_digest"] = cfg.digest()
    out = run_dir / "evolve" / "evolve_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), "utf-8")
    if result.champion is not None:
        champion_ref = {
            "bank_dir": str(result.champion.bank_dir),
            "bank_digest": result.champion.bank_digest,
            "candidate_id": result.champion.candidate_id,
            "hypothesis": result.champion.hypothesis,
        }
        champ_path = run_dir / "evolve" / "champion.json"
        champ_path.write_text(json.dumps(champion_ref, indent=2, sort_keys=True), "utf-8")
        payload["champion_ref"] = str(champ_path)
    payload["result_path"] = str(out)
    return payload
