"""CLI service for the real distillability probe (`rsi.cli probe-opd`)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rsi.config import CampaignConfig
from rsi.evaluation.service import resolve_bank
from rsi.harness.loader import load_bank_ref
from rsi.opd.probe import run_probe_on_traces
from rsi.providers.factory import build_policy
from rsi.runtime.budgets import BudgetSpec
from rsi.runtime.loop import MinimalLoop


def run_probe_opd(cfg: CampaignConfig, *, bank_ref: str, policy: Any | None = None) -> dict[str, Any]:
    bank = load_bank_ref(bank_ref)
    policy = policy if policy is not None else build_policy(cfg)
    opd_cfg = cfg.section("opd")
    runtime_cfg = cfg.section("runtime")
    budgets = BudgetSpec(
        max_actions=int(runtime_cfg.get("max_actions", 20)),
        max_extra_llm_calls=int(runtime_cfg.get("max_extra_llm_calls", 2)),
    )
    from rsi.contracts.policy import SamplingParams

    sampling = SamplingParams(max_new_tokens=int(runtime_cfg.get("max_new_tokens", 256)))

    def trace_runner(env, task):
        loop = MinimalLoop(
            env=env,
            policy=policy,
            bank=bank,  # Student side runs H0 only... probe rollouts are H0;
            budgets=budgets,
            sampling=sampling,
            campaign_id=cfg.campaign_id,
        )
        # Student rollouts must use H0 alone: pass the empty bank for the loop
        # while the Teacher fork uses the champion bank.
        from rsi.harness.loader import load_empty_bank

        loop.bank = load_empty_bank()
        result = loop.run_episode(task, seed=cfg.seed)
        return result.trace

    from rsi.environments.registry import global_registry

    registry = global_registry()
    tasks = []
    for domain in cfg.section("suite").get("domains") or []:
        provider = registry.get(domain).task_provider_factory()
        env_factory = registry.get(domain).adapter_factory
        tasks.append((domain, provider, env_factory))

    # probe panel: work_train decisions (allowed for evolution diagnosis + OPD)
    all_reports = []
    for domain, provider, env_factory in tasks:
        panel = provider.public_tasks("work_train")
        max_n = int(cfg.section("evolution").get("dev_panel_per_domain", 8))
        report = run_probe_on_traces(
            policy=policy,
            teacher_policy=policy,  # batch-snapshot self-distillation: same weights
            env_factory=env_factory,
            bank=bank,
            tasks=panel[:max_n],
            trace_runner=trace_runner,
            gate_beta=float(opd_cfg.get("gate_beta", 5.0)),
            uncertain_realizability=str(opd_cfg.get("uncertain_realizability", "skip")),
            max_targets_per_decision=int(opd_cfg.get("max_targets_per_decision", 1)),
            max_shadow_events=int(opd_cfg.get("max_shadow_events", 8)),
            sampling=sampling,
            budgets=budgets,
            synthetic=bool(registry.get(domain).synthetic),
        )
        all_reports.append((domain, report))

    ok = all(r.admitted for _, r in all_reports) and bool(all_reports)
    payload: dict[str, Any] = {
        "ok": ok,
        "command": "probe-opd",
        "bank_ref": bank_ref,
        "bank_digest": bank.bank_digest,
        "policy_digest": policy.policy_artifact.policy_id,
        "domains": {d: r.to_json() for d, r in all_reports},
        "stopped_reason": None if ok else "INSUFFICIENT_OPD_SUPPORT",
    }
    runs_root = cfg.resolve_path(str(cfg.section("execution").get("runs_root", "runs")))
    assert runs_root is not None
    out_dir = runs_root / cfg.campaign_id / "opd"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "probe_report.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), "utf-8")
    payload["result_path"] = str(out_path)
    return payload
