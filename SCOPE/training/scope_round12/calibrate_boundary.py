#!/usr/bin/env python3
"""Phase B — cross-view aggregation + scalar / dual-view boundary calibration.

Only offline_valid labels may be used for selecting tau / lambda.
base_live is evaluated once after freeze.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.scope.canonical_rollback_scorer import decide_from_saved_logits

OUT = _REPO / "outputs" / "scope_round12" / "phase_b_operation_boundary"
R11 = _REPO / "outputs" / "scope_round11" / "phase_b"
CROSS = OUT / "cross_view_replays"

MODELS = {
    "M0": "factorized_full_stage1_seed42",
    "M1": "factorized_main_seed42",
    "M2": "r10_main_noweight_seed42",
}
# M2 lives under round10 followup; path override for legacy fallback
M2_LEGACY = (
    _REPO
    / "outputs"
    / "scope_round10_followup"
    / "phase_b"
    / "r10_main_noweight_seed42"
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def margin_of(row: dict) -> float:
    s = row.get("vllm_logits") or row.get("canonical_logits") or {}
    return float(s.get("ROLLBACK_TO", -1e9)) - float(s.get("CONTINUE", -1e9))


def op_metrics(rows: list[dict], *, threshold: float = 0.0, margins: list[float] | None = None) -> dict:
    tp = fp = tn = fn = 0
    margin_vals = []
    prior = {"CONTINUE": 0, "ROLLBACK_TO": 0, "REPLAN": 0}
    for i, r in enumerate(rows):
        g = r.get("gold_operation")
        if g not in ("CONTINUE", "ROLLBACK_TO"):
            continue
        if margins is not None:
            # decide from fused/custom margin vs threshold using same contract:
            # ROLLBACK if margin >= threshold (approx: score_rb - score_c >= threshold when rb wins)
            m = margins[i]
            pred = "ROLLBACK_TO" if m >= threshold else "CONTINUE"
        else:
            pred = decide_from_saved_logits(r, threshold=threshold, disable_replan=True).pred_operation
            m = margin_of(r)
        margin_vals.append(m)
        prior[pred] = prior.get(pred, 0) + 1
        if g == "CONTINUE":
            if pred == "CONTINUE":
                tn += 1
            else:
                fp += 1
        else:
            if pred == "ROLLBACK_TO":
                tp += 1
            else:
                fn += 1
    n = max(tn + fp + tp + fn, 1)
    cr = tn / max(tn + fp, 1)
    rr = tp / max(tp + fn, 1)
    bal = 0.5 * (cr + rr)
    mv = sorted(margin_vals)

    def q(p: float) -> float | None:
        if not mv:
            return None
        return mv[min(len(mv) - 1, max(0, int(p * (len(mv) - 1))))]

    mean = sum(margin_vals) / max(len(margin_vals), 1)
    var = sum((x - mean) ** 2 for x in margin_vals) / max(len(margin_vals), 1)
    return {
        "n": n,
        "balanced_accuracy": bal,
        "ContinueRecall": cr,
        "RollbackRecall": rr,
        "prediction_prior": {k: v / n for k, v in prior.items()},
        "margin_mean": mean,
        "margin_std": math.sqrt(var),
        "margin_q01": q(0.01),
        "margin_q05": q(0.05),
        "margin_q25": q(0.25),
        "margin_q50": q(0.50),
        "margin_q75": q(0.75),
        "margin_q95": q(0.95),
        "margin_q99": q(0.99),
    }


def resolve_replay(model: str, view: str, split: str) -> Path | None:
    """Prefer Round12 cross-view outputs; fall back to Round11 native-view replays."""
    tag = "offline_valid" if split == "offline_valid" else "holdout"
    # Job dirs use V0/V1; views are A0/A1
    view_to_job = {"A0": "V0", "A1": "V1"}
    job_suffix = view_to_job.get(view, view)
    candidates = [
        CROSS / f"{model}_{job_suffix}" / f"eval_{tag}" / "canonical_vllm_replay.jsonl",
        CROSS / f"{model}_{view}" / f"eval_{tag}" / "canonical_vllm_replay.jsonl",
    ]
    for r12 in candidates:
        if r12.exists() and r12.stat().st_size > 0:
            return r12
    # Fallbacks for native training views already scored in R11/R10
    if model == "M0" and view == "A0":
        p = R11 / "factorized_full_stage1_seed42" / f"eval_{tag}" / "canonical_vllm_replay.jsonl"
        return p if p.exists() else None
    if model == "M1" and view == "A1":
        p = R11 / "factorized_main_seed42" / f"eval_{tag}" / "canonical_vllm_replay.jsonl"
        return p if p.exists() else None
    if model == "M2" and view == "A0":
        # R10 effective input ≈ A0 full
        p = M2_LEGACY / f"eval_{tag}" / "canonical_vllm_replay.jsonl"
        if not p.exists():
            # alternate naming
            alt = M2_LEGACY / "eval_holdout" / "canonical_vllm_replay.jsonl" if tag == "holdout" else None
            return alt if alt and alt.exists() else (p if p.exists() else None)
        return p
    return None


def build_cross_view_matrix() -> dict:
    matrix = {}
    for model in ("M0", "M1", "M2"):
        for view in ("A0", "A1"):
            cell = {"model": model, "view": view, "splits": {}}
            for split in ("offline_valid", "base_live"):
                path = resolve_replay(model, view, split)
                if path is None:
                    cell["splits"][split] = {"available": False}
                    continue
                rows = load_jsonl(path)
                cell["splits"][split] = {"available": True, "path": str(path), **op_metrics(rows, threshold=0.0)}
            matrix[f"{model}_{view}"] = cell
    return matrix


def select_scalar_tau(offline_rows: list[dict]) -> tuple[float, dict]:
    best = None
    best_tau = 0.0
    best_m = None
    for i in range(0, 1001):
        tau = i / 100.0
        m = op_metrics(offline_rows, threshold=tau)
        key = (m["ContinueRecall"] if False else min(m["ContinueRecall"], m["RollbackRecall"]), m["balanced_accuracy"], -abs(tau))
        key = (min(m["ContinueRecall"], m["RollbackRecall"]), m["balanced_accuracy"], -abs(tau))
        if best is None or key > best:
            best = key
            best_tau = tau
            best_m = m
    assert best_m is not None
    return best_tau, best_m


def zscore(vals: list[float], mean: float, std: float) -> list[float]:
    s = std if std > 1e-8 else 1.0
    return [(v - mean) / s for v in vals]


def select_dual_view(full_off: list[dict], state_off: list[dict]) -> dict:
    assert len(full_off) == len(state_off)
    m_full = [margin_of(r) for r in full_off]
    m_state = [margin_of(r) for r in state_off]
    mean_f, mean_s = sum(m_full) / len(m_full), sum(m_state) / len(m_state)
    std_f = math.sqrt(sum((x - mean_f) ** 2 for x in m_full) / len(m_full))
    std_s = math.sqrt(sum((x - mean_s) ** 2 for x in m_state) / len(m_state))
    zf = zscore(m_full, mean_f, std_f)
    zs = zscore(m_state, mean_s, std_s)

    best = None
    best_cfg = None
    for lam in (0.25, 0.50, 0.75, 1.00):
        fused = [a + lam * b for a, b in zip(zf, zs)]
        for i in range(-500, 501):
            tau = i / 100.0
            m = op_metrics(full_off, threshold=tau, margins=fused)
            key = (min(m["ContinueRecall"], m["RollbackRecall"]), m["balanced_accuracy"], -lam, -abs(tau))
            if best is None or key > best:
                best = key
                best_cfg = {
                    "lambda": lam,
                    "tau": tau,
                    "offline": m,
                    "norm": {"mean_full": mean_f, "std_full": std_f, "mean_state": mean_s, "std_state": std_s},
                }
    assert best_cfg is not None
    return best_cfg


def gate_pass(m: dict) -> bool:
    return (
        m["ContinueRecall"] >= 0.70
        and m["RollbackRecall"] >= 0.70
        and m["balanced_accuracy"] >= 0.70
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    matrix = build_cross_view_matrix()
    (OUT / "CROSS_VIEW_MATRIX.json").write_text(json.dumps(matrix, indent=2) + "\n", encoding="utf-8")

    md = ["# CROSS_VIEW_MATRIX\n\n", "| model×view | split | bal | CR | RR | margin_mean |\n", "|---|---|---:|---:|---:|---:|\n"]
    for k, cell in matrix.items():
        for split, m in cell["splits"].items():
            if not m.get("available"):
                md.append(f"| {k} | {split} | — | — | — | — |\n")
            else:
                md.append(
                    f"| {k} | {split} | {m['balanced_accuracy']:.3f} | {m['ContinueRecall']:.3f} | "
                    f"{m['RollbackRecall']:.3f} | {m['margin_mean']:.3f} |\n"
                )
    (OUT / "CROSS_VIEW_MATRIX.md").write_text("".join(md), encoding="utf-8")

    # B2 scalar on M0+V0
    off_path = resolve_replay("M0", "A0", "offline_valid")
    live_path = resolve_replay("M0", "A0", "base_live")
    if off_path is None or live_path is None:
        raise SystemExit("M0×A0 replays missing — cannot calibrate scalar boundary")
    off = load_jsonl(off_path)
    live = load_jsonl(live_path)
    tau, off_m = select_scalar_tau(off)
    live_m = op_metrics(live, threshold=tau)
    scalar = {
        "model": "M0",
        "view": "A0",
        "tau_scalar": tau,
        "offline_valid": off_m,
        "base_live": live_m,
        "SCALAR_BOUNDARY_REPAIR_PASS": gate_pass(live_m),
        "selection_objective": "maximize min(CR,RR) then bal then min|tau|; labels=offline_valid only",
    }
    (OUT / "SCALAR_CALIBRATION.json").write_text(json.dumps(scalar, indent=2) + "\n", encoding="utf-8")

    dual = {"skipped": True, "reason": "scalar passed"}
    solution = "scalar"
    stop = False
    if not scalar["SCALAR_BOUNDARY_REPAIR_PASS"]:
        off_s_path = resolve_replay("M0", "A1", "offline_valid")
        live_s_path = resolve_replay("M0", "A1", "base_live")
        if off_s_path is None or live_s_path is None:
            dual = {
                "skipped": False,
                "available": False,
                "reason": "M0×A1 cross-view replay not ready",
                "DUAL_VIEW_BOUNDARY_REPAIR_PASS": False,
            }
            solution = "no-pass"
            stop = True
        else:
            off_s = load_jsonl(off_s_path)
            live_s = load_jsonl(live_s_path)
            # align by event_id/query:turn
            def key(r):
                return r.get("event_id") or f"{r.get('query_id')}:{r.get('turn')}"

            idx_s_off = {key(r): r for r in off_s}
            idx_s_live = {key(r): r for r in live_s}
            off_full = [r for r in off if key(r) in idx_s_off]
            off_state = [idx_s_off[key(r)] for r in off_full]
            live_full = [r for r in live if key(r) in idx_s_live]
            live_state = [idx_s_live[key(r)] for r in live_full]
            cfg = select_dual_view(off_full, off_state)
            # apply frozen norm+lambda+tau on live
            norm = cfg["norm"]
            zf = zscore([margin_of(r) for r in live_full], norm["mean_full"], norm["std_full"])
            zs = zscore([margin_of(r) for r in live_state], norm["mean_state"], norm["std_state"])
            fused_live = [a + cfg["lambda"] * b for a, b in zip(zf, zs)]
            live_dual = op_metrics(live_full, threshold=cfg["tau"], margins=fused_live)
            dual = {
                "skipped": False,
                "available": True,
                "lambda": cfg["lambda"],
                "tau": cfg["tau"],
                "norm": norm,
                "offline_valid": cfg["offline"],
                "base_live": live_dual,
                "DUAL_VIEW_BOUNDARY_REPAIR_PASS": gate_pass(live_dual),
                "note": "diagnostic / teacher boundary only; not hard capability internalization",
            }
            if dual["DUAL_VIEW_BOUNDARY_REPAIR_PASS"]:
                solution = "dual_view_teacher"
                stop = False
            else:
                solution = "no-pass"
                stop = True
    (OUT / "DUAL_VIEW_FUSION.json").write_text(json.dumps(dual, indent=2) + "\n", encoding="utf-8")

    decision = {
        "OPERATION_BOUNDARY_SOLUTION": solution,
        "SCALAR_BOUNDARY_REPAIR_PASS": scalar["SCALAR_BOUNDARY_REPAIR_PASS"],
        "DUAL_VIEW_BOUNDARY_REPAIR_PASS": bool(dual.get("DUAL_VIEW_BOUNDARY_REPAIR_PASS")),
        "STOP_AFTER_OPERATION_BOUNDARY": stop,
        "allow_phase_c_mainline": solution == "scalar",
        "note": (
            "Phase C mainline (3-seed full_stage1) only when SCALAR_BOUNDARY_REPAIR_PASS; "
            "dual_view_teacher is diagnostic and does not unlock closed-loop by itself."
        ),
    }
    (OUT / "BARRIER_B_DECISION.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
