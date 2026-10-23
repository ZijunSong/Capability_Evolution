"""Evaluation service: run a policy under minimal/bank harness on a split panel.

Real episodes through the H0 loop, official per-domain scorers, per-task raw
records, cost accounting, equal-domain macro aggregation.  Toy/synthetic runs
are flagged; report merge refuses to fold them into formal results.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rsi.config import CampaignConfig
from rsi.contracts.errors import ProtocolViolation
from rsi.contracts.events import CostRecord, total_cost
from rsi.contracts.observation import ScoreRecord
from rsi.evaluation.aggregation import SuiteResult, aggregate_scores
from rsi.evaluation.costs import CostLedger
from rsi.environments.registry import global_registry
from rsi.harness.loader import LoadedBank, load_bank_ref, load_empty_bank
from rsi.providers.factory import build_policy
from rsi.runtime.budgets import BudgetSpec
from rsi.runtime.loop import MinimalLoop


@dataclass
class EvalRunResult:
    suite: SuiteResult
    records: tuple[ScoreRecord, ...]
    costs: CostLedger
    per_task: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "suite": self.suite.to_json(),
            "records": [
                {
                    "task_id": r.task_id,
                    "domain": r.domain,
                    "value": r.value,
                    "success": r.success,
                    "metric_name": r.metric_name,
                    "synthetic": r.synthetic,
                }
                for r in self.records
            ],
            "costs": self.costs.to_json(),
            "per_task": self.per_task,
        }


def resolve_bank(harness: str, bank_ref: str | None) -> LoadedBank:
    if harness == "minimal":
        return load_empty_bank()
    if harness == "bank":
        if not bank_ref:
            raise ProtocolViolation("--harness bank requires --bank <manifest/ref>")
        return load_bank_ref(bank_ref)
    raise ProtocolViolation(f"unknown harness mode {harness!r}")


def run_panel(
    cfg: CampaignConfig,
    *,
    policy: Any,
    bank: LoadedBank,
    role: str,
    seed: int,
    round_index: int | None = None,
    max_per_domain: int | None = None,
    campaign_id: str | None = None,
) -> EvalRunResult:
    suite_cfg = cfg.section("suite")
    runtime_cfg = cfg.section("runtime")
    domains = list(suite_cfg.get("domains") or [])
    budgets = BudgetSpec(
        max_actions=int(runtime_cfg.get("max_actions", 20)),
        max_extra_llm_calls=int(runtime_cfg.get("max_extra_llm_calls", 2)),
    )
    from rsi.contracts.policy import SamplingParams

    sampling = SamplingParams(
        max_new_tokens=int(runtime_cfg.get("max_new_tokens", 256)),
        temperature=float(runtime_cfg.get("temperature", 1.0)),
    )
    registry = global_registry()
    records: list[ScoreRecord] = []
    costs = CostLedger()
    per_task: dict[str, dict[str, Any]] = {}
    campaign = campaign_id or cfg.campaign_id

    for domain in domains:
        bundle = registry.get(domain)
        provider = bundle.task_provider_factory()
        scorer = bundle.scorer_factory()
        tasks = provider.public_tasks(role, round_index=round_index)
        if max_per_domain is not None:
            tasks = tasks[:max_per_domain]
        for i, task in enumerate(tasks):
            env = bundle.adapter_factory()
            loop = MinimalLoop(
                env=env,
                policy=policy,
                bank=bank,
                budgets=budgets,
                sampling=sampling,
                campaign_id=campaign,
            )
            t0 = time.perf_counter()
            result = loop.run_episode(task, seed=seed + i)
            wall_ms = (time.perf_counter() - t0) * 1000.0
            artifact = result.final_artifact
            if artifact is None:
                from rsi.contracts.observation import ArtifactRef

                artifact = ArtifactRef(ref_id=task.task_id, kind="answer")
            record = scorer.evaluate(artifact, provider.private_ref(task.task_id))
            records.append(record)
            ev_cost = total_cost(result.trace.events)
            costs.add_episode(
                success=record.success,
                protocol_valid=result.trace.terminated_reason not in ("max_actions",),
                completed=result.trace.terminated_reason == "finish",
                cost=ev_cost,
                wall_ms=wall_ms,
            )
            per_task[task.task_id] = {
                "domain": domain,
                "value": record.value,
                "terminated_reason": result.trace.terminated_reason,
                "n_decisions": len(result.trace.decisions),
                "wall_ms": wall_ms,
                "bank_digest": bank.bank_digest,
                "policy_digest": policy.policy_artifact.policy_id,
                "synthetic": record.synthetic,
            }
            env.close()

    suite = aggregate_scores(records, aggregate=str(suite_cfg.get("aggregate", "equal_domain_macro")))
    return EvalRunResult(suite=suite, records=tuple(records), costs=costs, per_task=per_task)


def run_eval(
    cfg: CampaignConfig,
    *,
    harness: str = "minimal",
    bank_ref: str | None = None,
    split: str = "harness_select",
    seed: int | None = None,
    policy: Any | None = None,
    max_per_domain: int | None = None,
    write: bool = True,
) -> dict[str, Any]:
    bank = resolve_bank(harness, bank_ref)
    pol = policy if policy is not None else build_policy(cfg)
    run_seed = cfg.seed if seed is None else seed
    result = run_panel(
        cfg,
        policy=pol,
        bank=bank,
        role=split,
        seed=run_seed,
        max_per_domain=max_per_domain,
    )
    payload: dict[str, Any] = {
        "ok": True,
        "command": "eval",
        "harness": harness,
        "bank_ref": bank_ref or "minimal",
        "bank_digest": bank.bank_digest,
        "policy_digest": pol.policy_artifact.policy_id,
        "split": split,
        "seed": run_seed,
        "config_digest": cfg.digest(),
        "result": result.to_json(),
    }
    if write:
        runs_root = cfg.resolve_path(str(cfg.section("execution").get("runs_root", "runs")))
        assert runs_root is not None
        out_dir = runs_root / cfg.campaign_id / "eval"
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = f"{split}_{harness}_{bank.bank_digest[:8]}_{run_seed}"
        out_path = out_dir / f"eval_{tag}.json"
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), "utf-8")
        payload["result_path"] = str(out_path)
    return payload
