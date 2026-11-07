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

    # PR-05 (guide §5.2): a confirm run consumes the fresh confirm_h split —
    # when the manifest is authoritative it must actually CONTAIN that split.
    # The pre-fix behavior let the provider fall back to another unknown task
    # set (e.g. work_train), silently burning the "fresh" property: the
    # confirm then re-measured tasks already used for selection/tuning.
    suite_cfg_confirm = cfg.section("suite")
    if bool(suite_cfg_confirm.get("manifest_is_authoritative", False)):
        from rsi.evaluation.data_prep import load_split_manifests

        manifests = load_split_manifests(cfg)
        missing = []
        for dom in suite_cfg_confirm.get("domains") or []:
            manifest = manifests.get(dom)
            roles = {a.role for a in manifest.assignments} if manifest else set()
            if "confirm_h" not in roles:
                missing.append(dom)
        if missing:
            raise ProtocolViolation(
                f"confirm_h split MISSING from the authoritative manifest for "
                f"domains {missing} — refusing to run a fallback-panel confirm "
                "(guide §5.2: a confirm without a real fresh split is not a "
                "confirmation; build a v2 manifest with a confirm_h role first)"
            )

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
    all_strict_wins = all_strict_losses = all_strict_ties = 0
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
        # PR-03 (guide §4.2): the paired delta is computed on BOTH metrics.
        # The value metric (AppWorld: partial rate; WebShop: reward) is the
        # dense signal; the STRICT metric (success labels) is the formal
        # teacher-admission gate.  For domains where value IS strict (metric
        # == strict success), the two sets coincide.
        strict_a = dict(zip(sorted(da), (1.0 if s else 0.0 for s in dom.successes), strict=False))
        strict_b_dom = next((d for d in b.suite.domains if d.domain == dom.domain), None)
        strict_b = {}
        if strict_b_dom is not None:
            strict_b = dict(
                zip(sorted(strict_b_dom.per_task),
                    (1.0 if s else 0.0 for s in strict_b_dom.successes),
                    strict=False)
            )
        swins = sum(1 for t in common if strict_b.get(t, 0.0) > strict_a.get(t, 0.0))
        slosses = sum(1 for t in common if strict_b.get(t, 0.0) < strict_a.get(t, 0.0))
        sties = len(common) - swins - slosses
        all_strict_wins += swins
        all_strict_losses += slosses
        all_strict_ties += sties
        strict_mean_a = sum(strict_a.get(t, 0.0) for t in common) / len(common) if common else 0.0
        strict_mean_b = sum(strict_b.get(t, 0.0) for t in common) / len(common) if common else 0.0
        domains[dom.domain] = {
            "n_paired": len(common),
            "teacher_mean": mean_b,
            "student_mean": mean_a,
            "delta_h": mean_b - mean_a,
            # strict-metric paired delta (formal admission gate)
            "strict_teacher_mean": strict_mean_b,
            "strict_student_mean": strict_mean_a,
            "delta_h_strict": strict_mean_b - strict_mean_a,
            "strict_wins": swins,
            "strict_losses": slosses,
            "strict_ties": sties,
            "metric_name": dom.metric_name,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "per_task": {t: {"A_student": da[t], "B_teacher": db[t]} for t in common},
        }

    macro_delta = sum(d["delta_h"] for d in domains.values()) / len(domains) if domains else 0.0
    macro_delta_strict = (
        sum(d["delta_h_strict"] for d in domains.values()) / len(domains) if domains else 0.0
    )
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
        "delta_h_macro_strict": macro_delta_strict,
        "wins_losses_ties": {"wins": all_wins, "losses": all_losses, "ties": all_ties},
        "strict_wins_losses_ties": {
            "wins": all_strict_wins,
            "losses": all_strict_losses,
            "ties": all_strict_ties,
        },
        "domains": domains,
        "synthetic": all(dom.synthetic for dom in a.suite.domains),
        "interpretation": (
            "ΔH = B − A on the SAME fresh confirm_h panel with identical seeds. "
            "Positive ΔH = the champion harness improves the model; ties indicate "
            "the base model's action quality saturates panel separation. "
            "delta_h_strict is the formal teacher-admission metric; delta_h on "
            "value (AppWorld partial / WebShop reward) supports PARTIAL_SIGNAL_ONLY at most."
        ),
    }
    out_path = out_dir / f"confirm_h_r{round_index}.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False), "utf-8")
    report["result_path"] = str(out_path)
    return report
