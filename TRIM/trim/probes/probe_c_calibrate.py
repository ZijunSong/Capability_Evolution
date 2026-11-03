"""Probe C3 calibration: does the cheap probe predict 50-step outcomes?

CLI:

    python -m trim.probes.probe_c_calibrate --config configs/probes/probe_c_qwen35_9b.json \
        --c1 METRICS_OR_SCREEN.json --c3 C3_50STEP_RESULTS.jsonl --out DIR

Rules (§11):
- Legacy 50-step runs without the required manifest fields are labelled
  ``legacy/unverified``; they are never re-labelled as add experiments.
- Calibration must include at least one candidate that the cheap probe
  *filtered out*, otherwise only "kept successes" are visible and the
  false-negative rate cannot be estimated.
- Candidates without a finished C3 run are missing data — never filled as
  negative samples.
- Thresholds are chosen on C_select; the prediction relation is evaluated on
  C_audit. Rules developed on 9B are frozen before any 27B check; 27B must
  collect its own states and scores.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from trim.eval.probe_c_metrics import MISSING
from trim.probes.probe_c_contract import ContractError, load_config

C3_REQUIRED_MANIFEST_FIELDS = (
    "model_id",
    "model_revision",
    "harness_pair",  # actual H0/H_m or Full-minus-m masks
    "data_splits",
    "train_seeds",
    "eval_seeds",
    "objective",
    "rollout_refresh",
    "n_supervised_tokens",
    "reward_definition",
    "checkpoints",
    "cost",
)


class CalibrateError(ContractError):
    pass


# ---------------------------------------------------------------------------
# ranking correlation
# ---------------------------------------------------------------------------


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | str:
    """Spearman rank correlation with average ranks for ties."""
    if len(xs) != len(ys):
        raise CalibrateError("spearman: length mismatch")
    n = len(xs)
    if n < 2:
        return MISSING
    rx, ry = _ranks(list(xs)), _ranks(list(ys))
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    if dx == 0 or dy == 0:
        return MISSING
    return float(num / (dx * dy))


# ---------------------------------------------------------------------------
# data rows
# ---------------------------------------------------------------------------


@dataclass
class C1Row:
    component_id: str
    train_seed: int
    fixed_budget: int
    r_d: float | str = MISSING  # D_K improvement (absolute drop)
    r_n: float | str = MISSING  # NLL improvement (absolute drop)
    status: str = "NOT_RUN"
    n_supervised_tokens: int = 0
    covered_queries: int = 0
    split: str = "C_select"


@dataclass
class C3Row:
    component_id: str
    g_50: float | str = MISSING  # J(theta_m^50, H0) - J(theta0, H0)
    e_50: float | str = MISSING  # HarnessOPD - RL-only
    verified: bool = False
    manifest_missing: list[str] = field(default_factory=list)
    beneficial: bool | None = None  # None = C3 not finished (missing data)

    @property
    def label(self) -> str:
        if not self.verified:
            return "legacy/unverified"
        return "verified"


def load_c1_rows(path: str | Path) -> list[C1Row]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_rows = payload.get("rows") if isinstance(payload, dict) else payload
    out: list[C1Row] = []
    for r in raw_rows or []:
        out.append(
            C1Row(
                component_id=str(r.get("component_id") or ""),
                train_seed=int(r.get("train_seed") or 0),
                fixed_budget=int(r.get("fixed_budget") or 0),
                r_d=r.get("r_d", r.get("d_k_improvement", MISSING)),
                r_n=r.get("r_n", r.get("nll_improvement", MISSING)),
                status=str(r.get("status") or "NOT_RUN"),
                n_supervised_tokens=int(r.get("supervised_tokens") or 0),
                covered_queries=int(r.get("covered_queries") or 0),
                split=str(r.get("split") or "C_select"),
            )
        )
    return out


def load_c3_rows(path: str | Path) -> list[C3Row]:
    p = Path(path)
    lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    out: list[C3Row] = []
    for ln in lines:
        r = json.loads(ln)
        manifest = dict(r.get("manifest") or {})
        missing = [k for k in C3_REQUIRED_MANIFEST_FIELDS if k not in manifest]
        out.append(
            C3Row(
                component_id=str(r.get("component_id") or ""),
                g_50=r.get("g_50", MISSING),
                e_50=r.get("e_50", MISSING),
                verified=not missing,
                manifest_missing=missing,
                beneficial=r.get("beneficial"),
            )
        )
    return out


# ---------------------------------------------------------------------------
# calibration questions
# ---------------------------------------------------------------------------


def prediction_scatter(
    c1_rows: Sequence[C1Row], c3_rows: Sequence[C3Row]
) -> dict[str, Any]:
    """Q1: is fixed-budget R_D/R_N directionally related to E^50/G^50?"""
    c3_by_id = {r.component_id: r for r in c3_rows}
    points: list[dict[str, Any]] = []
    xs_rn_e: list[float] = []
    ys_n_e: list[float] = []
    xs_rd_e: list[float] = []
    ys_d_e: list[float] = []
    xs_rn_g: list[float] = []
    ys_n_g: list[float] = []
    xs_rd_g: list[float] = []
    ys_d_g: list[float] = []
    for row in c1_rows:
        c3 = c3_by_id.get(row.component_id)
        if c3 is None:
            continue
        point = {
            "component_id": row.component_id,
            "r_n": row.r_n,
            "r_d": row.r_d,
            "e_50": c3.e_50,
            "g_50": c3.g_50,
            "c3_label": c3.label,
            "n_supervised_tokens": row.n_supervised_tokens,
            "covered_queries": row.covered_queries,
            "confounds": {
                "initial_gap": MISSING,
                "token_count": row.n_supervised_tokens,
                "coverage": row.covered_queries,
            },
        }
        points.append(point)
        if c3.e_50 != MISSING:
            if row.r_n != MISSING:
                xs_rn_e.append(float(row.r_n))
                ys_n_e.append(float(c3.e_50))
            if row.r_d != MISSING:
                xs_rd_e.append(float(row.r_d))
                ys_d_e.append(float(c3.e_50))
        if c3.g_50 != MISSING:
            if row.r_n != MISSING:
                xs_rn_g.append(float(row.r_n))
                ys_n_g.append(float(c3.g_50))
            if row.r_d != MISSING:
                xs_rd_g.append(float(row.r_d))
                ys_d_g.append(float(c3.g_50))
    return {
        "points": points,
        "spearman": {
            "r_n_vs_e_50": spearman(xs_rn_e, ys_n_e),
            "r_d_vs_e_50": spearman(xs_rd_e, ys_d_e),
            "r_n_vs_g_50": spearman(xs_rn_g, ys_n_g),
            "r_d_vs_g_50": spearman(xs_rd_g, ys_d_g),
        },
        "n_paired": len(points),
        "note": "component-level uncertainty needs multiple train seeds; "
        "C1 selection noise is not a final statistical claim",
    }


def selection_confusion(
    c1_rows: Sequence[C1Row], c3_rows: Sequence[C3Row]
) -> dict[str, Any]:
    """Q3: PROMOTE hit rate / DEPRIORITIZE false negatives. Missing stays missing."""
    c3_by_id = {r.component_id: r for r in c3_rows}
    by_status: dict[str, list[str]] = {}
    for row in c1_rows:
        by_status.setdefault(row.status, []).append(row.component_id)
    out: dict[str, Any] = {}
    for status, ids in by_status.items():
        uniq = sorted(set(ids))
        finished = []
        missing = []
        for cid in uniq:
            c3 = c3_by_id.get(cid)
            if c3 is None or c3.beneficial is None:
                missing.append(cid)
            else:
                finished.append((cid, bool(c3.beneficial)))
        n_beneficial = sum(1 for _, b in finished if b)
        out[status] = {
            "components": uniq,
            "c3_finished": [cid for cid, _ in finished],
            "c3_missing": missing,  # NOT counted as negative samples
            "n_beneficial_of_finished": n_beneficial,
            "beneficial_rate_of_finished": (
                n_beneficial / len(finished) if finished else MISSING
            ),
        }
    return {
        "by_c1_status": out,
        "note": "at least one filtered-out (DEPRIORITIZE) candidate must enter "
        "C3; incomplete C3 rows are never filled as negatives",
    }


def strategy_comparison(
    strategies: Mapping[str, Sequence[str]],
    *,
    c3_rows: Sequence[C3Row],
    total_budget_gpu_hours: float,
    costs: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Q2: A-only vs A+B vs A+B+C top-k under the *same total budget*.

    Probe overhead counts against the budget. Components not selected still
    incur their accounting entry when their probe cost was paid.
    """
    c3_by_id = {r.component_id: r for r in c3_rows}
    costs = dict(costs or {})
    out: dict[str, Any] = {}
    all_selected: set[str] = set()
    for name, picks in strategies.items():
        all_selected.update(picks)
        quality = []
        spent = 0.0
        for cid in picks:
            spent += float(costs.get(cid, 0.0))
            c3 = c3_by_id.get(cid)
            quality.append(None if c3 is None or c3.g_50 == MISSING else float(c3.g_50))
        measured = [q for q in quality if q is not None]
        out[name] = {
            "selected": list(picks),
            "quality_g_50": quality,
            "mean_quality_of_measured": (
                sum(measured) / len(measured) if measured else MISSING
            ),
            "spent_gpu_hours": spent,
            "within_total_budget": spent <= float(total_budget_gpu_hours),
        }
    best = max(
        (
            (name, float(v["mean_quality_of_measured"]))
            for name, v in out.items()
            if v["mean_quality_of_measured"] != MISSING
        ),
        key=lambda kv: kv[1],
        default=None,
    )
    for name, v in out.items():
        if best is not None and v["mean_quality_of_measured"] != MISSING:
            v["regret_vs_best"] = best[1] - float(v["mean_quality_of_measured"])
        else:
            v["regret_vs_best"] = MISSING
    return {
        "strategies": out,
        "best_strategy": None if best is None else best[0],
        "unselected_but_accounted": sorted(
            set(c3_by_id) - all_selected
        ),
        "note": "budgets include probe overhead; bare A×B×C products are not "
        "multiplied",
    }


def cost_accounting(ledger_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Q4 / §17: C_new vs all-50-step totals, one-shot vs amortized."""
    def _sum(stage: str, key: str = "gpu_hours") -> float:
        return sum(float(r.get(key) or 0.0) for r in ledger_rows if r.get("stage") == stage)

    c_states = _sum("collect")
    c_shared = _sum("shared_encoding") + _sum("cache_shared")
    c_t = {}
    c_u = {}
    c_m = {}
    c2 = {}
    c3 = {}
    for r in ledger_rows:
        cid = str(r.get("component_id") or "unknown")
        stage = str(r.get("stage") or "")
        hours = float(r.get("gpu_hours") or 0.0)
        if stage == "cache":
            c_t[cid] = c_t.get(cid, 0.0) + hours
        elif stage == "screen":
            c_u[cid] = c_u.get(cid, 0.0) + hours
        elif stage == "screen_eval":
            c_m[cid] = c_m.get(cid, 0.0) + hours
        elif stage == "confirm":
            c2[cid] = c2.get(cid, 0.0) + hours
        elif stage == "calibrate_c3":
            c3[cid] = c3.get(cid, 0.0) + hours

    def _total(d: Mapping[str, float]) -> float:
        return sum(d.values())

    c_new = (
        c_states
        + c_shared
        + _total(c_t)
        + _total(c_u)
        + _total(c_m)
        + _total(c2)
        + _total(c3)
    )
    return {
        "C_states": c_states,
        "C_shared": c_shared,
        "C_T_by_component": c_t,
        "C_U_by_component": c_u,
        "C_M_by_component": c_m,
        "C2_by_component": c2,
        "C3_by_component": c3,
        "C_new_total_gpu_hours": c_new,
        "task_calls": sum(int(r.get("n_task_calls") or 0) for r in ledger_rows),
        "teacher_calls": sum(int(r.get("n_teacher_calls") or 0) for r in ledger_rows),
        "prefill_tokens": sum(int(r.get("n_prefill_tokens") or 0) for r in ledger_rows),
        "processed_tokens": sum(int(r.get("n_processed_tokens") or 0) for r in ledger_rows),
        "peak_mem_gb_max": max(
            (float(r.get("peak_mem_gb") or 0.0) for r in ledger_rows), default=0.0
        ),
        "note": "one-shot calibration cost and amortized cost for later rounds "
        "are reported separately; the first calibration is never hidden",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def calibrate(
    *,
    c1_rows: Sequence[C1Row],
    c3_rows: Sequence[C3Row],
    ledger_rows: Sequence[Mapping[str, Any]] = (),
    strategies: Mapping[str, Sequence[str]] | None = None,
    total_budget_gpu_hours: float = 0.0,
    require_filtered_candidate: bool = True,
) -> dict[str, Any]:
    statuses = {r.status for r in c1_rows}
    has_filtered = "DEPRIORITIZE" in statuses and any(
        r.beneficial is not None for r in c3_rows if r.component_id in {c.component_id for c in c1_rows if c.status == "DEPRIORITIZE"}
    )
    warnings: list[str] = []
    if require_filtered_candidate and not has_filtered:
        warnings.append(
            "no filtered-out (DEPRIORITIZE) candidate has a finished C3 run; "
            "false-negative rate cannot be estimated from kept successes alone"
        )
    unverified = [r.component_id for r in c3_rows if not r.verified]
    if unverified:
        warnings.append(
            f"C3 rows without required manifest fields (labelled legacy/unverified): "
            f"{unverified}"
        )
    result: dict[str, Any] = {
        "prediction": prediction_scatter(c1_rows, c3_rows),
        "selection": selection_confusion(c1_rows, c3_rows),
        "costs": cost_accounting(ledger_rows),
        "warnings": warnings,
    }
    if strategies:
        result["strategies"] = strategy_comparison(
            strategies,
            c3_rows=c3_rows,
            total_budget_gpu_hours=total_budget_gpu_hours,
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="probe_c_calibrate")
    parser.add_argument("--config", required=True)
    parser.add_argument("--c1", required=True, help="SCREEN.json / C1 metrics json")
    parser.add_argument("--c3", required=True, help="C3 50-step results jsonl")
    parser.add_argument("--ledger", default="", help="COST_LEDGER.jsonl")
    parser.add_argument("--out", default="", help="output directory")
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config, stage="calibrate")
    except ContractError as exc:
        print(f"[probe_c_calibrate] REFUSED: {exc}")
        return 2
    c1_rows = load_c1_rows(args.c1)
    c3_rows = load_c3_rows(args.c3)
    ledger_rows: list[Mapping[str, Any]] = []
    if args.ledger:
        ledger_rows = [
            json.loads(ln)
            for ln in Path(args.ledger).read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
    result = calibrate(c1_rows=c1_rows, c3_rows=c3_rows, ledger_rows=ledger_rows)
    out_dir = Path(args.out or cfg.output_root or "outputs/probe_c") / "calibrate"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "PROBE_C_CALIBRATION.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True)[:4000])
    print(f"[probe_c_calibrate] wrote {out_dir / 'PROBE_C_CALIBRATION.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
