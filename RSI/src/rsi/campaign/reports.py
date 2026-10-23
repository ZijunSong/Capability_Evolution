"""Machine-readable campaign reports (task book §17).

The final report distinguishes ``implementation_smoke_passed``,
``real_training_executed``, ``n_successful_promotions``,
``shared_multidomain_evaluated``, ``final_test_unsealed`` and
``stopped_reason`` — one vague ``success`` field may never paper over levels.
Synthetic (toy/mock) artifacts are refused from formal result merges.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    if path.is_file():
        return json.loads(path.read_text("utf-8"))
    return None


def _iter_round_artifacts(run_dir: Path) -> list[dict[str, Any]]:
    rounds_dir = run_dir / "rounds"
    out: list[dict[str, Any]] = []
    if not rounds_dir.is_dir():
        return out
    for rdir in sorted(rounds_dir.iterdir()):
        if not rdir.is_dir():
            continue
        entry: dict[str, Any] = {"round": rdir.name}
        for name in (
            "baseline",
            "evolution",
            "confirm_harness",
            "probe",
            "training",
            "select_checkpoint",
            "four_cells",
            "promotion_decision",
        ):
            payload = _read_json(rdir / f"{name}.json")
            if payload is not None:
                entry[name] = payload
        out.append(entry)
    return out


def build_report(run_dir: Path | str) -> dict[str, Any]:
    run_dir = Path(run_dir)
    state = _read_json(run_dir / "state.json") or {}
    campaign = _read_json(run_dir / "campaign.json") or {}
    rounds = _iter_round_artifacts(run_dir)

    n_promotions = 0
    real_training = False
    for r in rounds:
        dec = r.get("promotion_decision") or {}
        if dec.get("accepted"):
            n_promotions += 1
        train = r.get("training") or {}
        if train.get("global_steps", 0) and train.get("step_reports"):
            real_training = True

    final_frozen = (run_dir / "final" / "FROZEN").is_file()
    final_test = _read_json(run_dir / "final" / "final_test.json")
    domains = set()
    for r in rounds:
        for key in ("baseline", "four_cells"):
            payload = r.get(key) or {}
            suite = payload.get("suite") or payload
            for dom in (suite.get("domains") or []) if isinstance(suite, dict) else []:
                if isinstance(dom, dict) and "domain" in dom:
                    domains.add(dom["domain"])

    synthetic = False
    for r in rounds:
        baseline = r.get("baseline") or {}
        if baseline.get("suite", {}).get("synthetic"):
            synthetic = True

    report: dict[str, Any] = {
        "campaign_id": campaign.get("campaign_id"),
        "run_dir": str(run_dir),
        "state": state.get("state"),
        "stopped_reason": state.get("state") if state.get("state") not in ("DONE",) else None,
        "implementation_smoke_passed": state.get("state") not in (None, "PREFLIGHT_FAILED", "ERROR"),
        "real_training_executed": real_training,
        "n_successful_promotions": n_promotions,
        "shared_multidomain_evaluated": len(domains) >= 2,
        "domains_evaluated": sorted(domains),
        "final_test_unsealed": final_test is not None,
        "final_frozen": final_frozen,
        "synthetic_results": synthetic,
        "rounds": [
            {
                "round": r.get("round"),
                "promotion": (r.get("promotion_decision") or {}).get("accepted"),
                "four_cells": (r.get("four_cells") or {}),
                "confirm_harness": (r.get("confirm_harness") or {}),
            }
            for r in rounds
        ],
    }
    if synthetic:
        report["formal_eligible"] = False
        report["formal_block_reason"] = "synthetic (toy/mock) artifacts present; merge refused"
    else:
        report["formal_eligible"] = n_promotions > 0 and real_training
    return report
