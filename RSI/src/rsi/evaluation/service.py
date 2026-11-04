"""Evaluation service: run a policy under minimal/bank harness on a split panel.

Real episodes through the H0 loop, official per-domain scorers, per-task raw
records, cost accounting, equal-domain macro aggregation.  Toy/synthetic runs
are flagged; report merge refuses to fold them into formal results.
"""

from __future__ import annotations

import json
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
from rsi.runtime.clock import monotonic_ms
from rsi.evaluation.task_resolver import TaskResolver, install_materialize_task_id
from rsi.runtime.budgets import BudgetSpec
from rsi.runtime.loop import MinimalLoop


@dataclass
class EvalRunResult:
    suite: SuiteResult
    records: tuple[ScoreRecord, ...]
    costs: CostLedger
    per_task: dict[str, dict[str, Any]] = field(default_factory=dict)
    skipped_domains: dict[str, str] = field(default_factory=dict)

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
            "skipped_domains": self.skipped_domains,
        }


def _bounded_steps(trace, *, max_steps: int = 30, max_chars: int = 400) -> list[dict]:
    """Bounded per-step evidence from the episode trace: action text +
    observation head/tail.  Caps keep one panel's json manageable (30 steps x
    400 chars ≈ 12KB/task worst case)."""
    steps: list[dict] = []
    for ev in list(trace.events)[:max_steps]:
        entry: dict = {"order": ev.order, "kind": ev.event_kind.value}
        if ev.action is not None:
            from rsi.contracts.action import encode_action

            try:
                entry["action"] = encode_action(ev.action)[:max_chars]
            except Exception:  # noqa: BLE001 - encode is best-effort
                entry["action"] = None
        if ev.observation is not None:
            content = str(ev.observation.content or "")
            entry["obs_head"] = content[: max_chars // 2]
            if len(content) > max_chars // 2:
                entry["obs_tail"] = content[-(max_chars // 4):]
            entry["obs_len"] = len(content)
            if getattr(ev.observation, "ok", True) is False:
                entry["obs_ok"] = False
        steps.append(entry)
    return steps


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
    resolver = TaskResolver(cfg)
    from rsi.runtime.factory import build_runtime_parts

    parts = build_runtime_parts(runtime_cfg)
    budgets, sampling, context_policy = parts.budgets, parts.sampling, parts.context_policy
    registry = global_registry()
    records: list[ScoreRecord] = []
    costs = CostLedger()
    per_task: dict[str, dict[str, Any]] = {}
    skipped_domains: dict[str, str] = {}
    campaign = campaign_id or cfg.campaign_id

    for domain in domains:
        executable, why = registry.probe_executable(domain)
        if not executable:
            # Loud skip — identical for baseline and every candidate, so
            # comparisons stay valid; recorded for the report.
            skipped_domains[domain] = why
            continue
        bundle = registry.get(domain)
        provider = bundle.task_provider_factory()
        scorer = bundle.scorer_factory()
        # PR-3 (audit §10.5): when the config declares a frozen split manifest,
        # it — not a fresh directory enumeration — decides the task set.
        if resolver.authoritative:
            tasks = resolver.resolve(
                install_materialize_task_id(provider),
                domain=domain,
                role=role,
                round_index=round_index,
            )
        else:
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
                context_policy=context_policy,
                campaign_id=campaign,
            )
            t0 = monotonic_ms()
            result = loop.run_episode(task, seed=seed + i)
            wall_ms = monotonic_ms() - t0
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
                "metric_name": record.metric_name,
                "success": record.success,
                "details": dict(record.details) if isinstance(record.details, dict) else {},
                "terminated_reason": result.trace.terminated_reason,
                "n_decisions": len(result.trace.decisions),
                "wall_ms": wall_ms,
                "bank_digest": bank.bank_digest,
                "policy_digest": policy.policy_artifact.policy_id,
                "synthetic": record.synthetic,
                "component_errors": dict(result.component_errors),
                # E1-pre (guide §9): bounded real-step evidence — the actual
                # actions and observations (truncated) so the evolver can see
                # WHAT failed, not just that it failed.  Persisted per task in
                # the eval json; EvidenceStore reads these directly.
                "steps": _bounded_steps(result.trace),
            }
            env.close()

    suite = aggregate_scores(records, aggregate=str(suite_cfg.get("aggregate", "equal_domain_macro")))
    return EvalRunResult(
        suite=suite,
        records=tuple(records),
        costs=costs,
        per_task=per_task,
        skipped_domains=skipped_domains,
    )


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
    workers: int | None = None,
    provider_override: str | None = None,
) -> dict[str, Any]:
    bank = resolve_bank(harness, bank_ref)
    run_seed = cfg.seed if seed is None else seed
    if workers is None:
        import os

        workers = int(os.environ.get("RSI_PANEL_WORKERS", 1) or 1)
    if workers >= 2 and policy is None:
        # Multi-GPU panel workers rebuild the policy per process; only valid
        # when the policy comes from the config (no injected instance).
        # Deliberately NOT building a parent-side policy here: a second 54GB
        # materialization in the parent was a co-cause of the kygt9 OOM chain.
        from rsi.evaluation.parallel import run_panel_mp
        from rsi.providers.factory import policy_identity_digest

        bank_ref_for_worker = bank_ref if harness == "bank" else None
        if provider_override:
            raise ProtocolViolation(
                "--provider override requires the serial path (workers=1); "
                "parallel workers rebuild policies from config only"
            )
        result = run_panel_mp(
            cfg,
            bank_ref=bank_ref_for_worker,
            role=split,
            seed=run_seed,
            max_per_domain=max_per_domain,
            workers=workers,
            require_gpu=True,
        )
        pol_policy_digest = policy_identity_digest(cfg)
    else:
        pol = policy if policy is not None else build_policy(cfg, provider_override=provider_override)
        pol_policy_digest = pol.policy_artifact.policy_id
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
        "policy_digest": pol_policy_digest,
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
