"""CONFIRM_HARNESS: paired ΔH = B − A measurement (task book §10.2).

Teacher arm  B = (θ, H0 + champion bank)   vs   Student arm  A = (θ, H0)

Both arms run the SAME fresh ``confirm_h`` panel with the same per-task seeds
(one-shot: the confirm ledger records the consumption).  Output: per-domain
and macro ΔH with paired wins/losses and per-task table.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rsi.config import CampaignConfig
from rsi.evaluation.paired import paired_deltas
from rsi.evaluation.parallel import run_panel_mp
from rsi.evaluation.service import run_panel
from rsi.campaign.splits import ConsumptionLedger
from rsi.harness.loader import load_bank_ref, load_empty_bank


def _panel(cfg: CampaignConfig, bank_ref: str | None, *, role: str, seed: int, max_per_domain: int, workers: int):
    if workers >= 2:
        return run_panel_mp(
            cfg,
            bank_ref=bank_ref,
            role=role,
            seed=seed,
            max_per_domain=max_per_domain,
            workers=workers,
            require_gpu=workers >= 2,
        )
    from rsi.providers.factory import build_policy

    bank = load_bank_ref(bank_ref) if bank_ref else load_empty_bank()
    policy = build_policy(cfg)
    return run_panel(
        cfg,
        policy=policy,
        bank=bank,
        role=role,
        seed=seed,
        max_per_domain=max_per_domain,
    )


def run_confirm_harness(
    cfg: CampaignConfig,
    *,
    bank_ref: str,
    panel_per_domain: int = 16,
    round_index: int = 0,
    workers: int | None = None,
) -> dict[str, Any]:
    import os

    if workers is None:
        workers = int(os.environ.get("RSI_PANEL_WORKERS") or cfg.section("execution").get("panel_workers", 1) or 1)
    runs_root = cfg.resolve_path(str(cfg.section("execution").get("runs_root", "runs")))
    assert runs_root is not None
    run_dir = runs_root / cfg.campaign_id
    ledger = ConsumptionLedger(run_dir / "confirm_consumption.json")
    ledger.consume(f"confirm_h[{round_index}]", context="CONFIRM_HARNESS")

    out_dir = run_dir / "confirm_h"
    out_dir.mkdir(parents=True, exist_ok=True)
    a = _panel(cfg, None, role="confirm_h", seed=cfg.seed, max_per_domain=panel_per_domain, workers=workers)
    (out_dir / f"armA_student_r{round_index}.json").write_text(json.dumps(a.to_json(), indent=2, sort_keys=True))
    b = _panel(cfg, bank_ref, role="confirm_h", seed=cfg.seed, max_per_domain=panel_per_domain, workers=workers)
    (out_dir / f"armB_teacher_r{round_index}.json").write_text(json.dumps(b.to_json(), indent=2, sort_keys=True))

    domains: dict[str, Any] = {}
    all_wins = all_losses = all_ties = 0
    for dom in sorted(a.suite.domains, key=lambda d: d.domain):
        da = dom.per_task
        db = next((d.per_task for d in b.suite.domains if d.domain == dom.domain), {})
        common = sorted(set(da) & set(db))
        wins = sum(1 for t in common if db[t] > da[t])
        losses = sum(1 for t in common if db[t] < da[t])
        ties = len(common) - wins - losses
        all_wins += wins
        all_losses += losses
        all_ties += ties
        mean_a = sum(da[t] for t in common) / len(common) if common else 0.0
        mean_b = sum(db[t] for t in common) / len(common) if common else 0.0
        domains[dom.domain] = {
            "n_paired": len(common),
            "teacher_mean": mean_b,
            "student_mean": mean_a,
            "delta_h": mean_b - mean_a,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "per_task": {t: {"A_student": da[t], "B_teacher": db[t]} for t in common},
        }

    macro_delta = sum(d["delta_h"] for d in domains.values()) / len(domains) if domains else 0.0
    report = {
        "stage": "CONFIRM_HARNESS",
        "consumed": f"confirm_h[{round_index}]",
        "bank_ref": bank_ref,
        "bank_digest": load_bank_ref(bank_ref).bank_digest,
        "seed": cfg.seed,
        "panel_per_domain": panel_per_domain,
        "teacher_arm": "H0+champion_bank",
        "student_arm": "H0_only",
        "delta_h_macro": macro_delta,
        "wins_losses_ties": {"wins": all_wins, "losses": all_losses, "ties": all_ties},
        "domains": domains,
        "synthetic": all(dom.synthetic for dom in a.suite.domains),
        "interpretation": (
            "ΔH = B − A on the SAME fresh confirm_h panel with identical seeds. "
            "Positive ΔH = the champion harness improves the model; ties indicate "
            "the base model's action quality saturates panel separation."
        ),
    }
    out_path = out_dir / f"confirm_h_r{round_index}.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False), "utf-8")
    report["result_path"] = str(out_path)
    return report
