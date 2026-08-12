#!/usr/bin/env python3
"""Phase A1: fit HF↔vLLM operation agreement calibration on offline_valid only.

Forbidden: using base_live to tune threshold / affine params.
Allowed: scalar threshold or affine m' = a*m + b, maximize top-1 agreement,
freeze, evaluate once on base_live.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.scope.decide_rollback_operation import decide_rollback_operation

R10 = _REPO / "outputs/scope_round10"
FOLLOWUP = _REPO / "outputs/scope_round10_followup"
SEEDS = (42, 43, 44)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def margin_from_logits(logits: dict) -> float:
    """Signed margin: CONTINUE - ROLLBACK_TO (positive → prefer CONTINUE)."""
    return float(logits.get("CONTINUE", -1e9)) - float(logits.get("ROLLBACK_TO", -1e9))


def decide_binary(margin: float, threshold: float = 0.0) -> str:
    # Prefer CONTINUE when margin > threshold (CONTINUE score higher than ROLLBACK by > thr)
    # Align with decide_rollback_operation: ROLLBACK wins if score_roll > score_cont and margin>=thr
    # Here margin = C - R; ROLLBACK when -margin >= threshold i.e. margin <= -threshold
    return "CONTINUE" if margin > -threshold else "ROLLBACK_TO"


def paired_rows(seed: int, split: str) -> list[dict]:
    base = R10 / "phase_a" / f"seed{seed}" / split
    hf_path = base / "hf_float32_replay.jsonl"
    vl_path = base / "vllm_fixed_replay.jsonl"
    if not hf_path.exists() or not vl_path.exists():
        raise FileNotFoundError(f"missing replay for seed{seed}/{split}")
    hf = {r.get("event_id"): r for r in load_jsonl(hf_path)}
    vl = {r.get("event_id"): r for r in load_jsonl(vl_path)}
    keys = sorted(set(hf) & set(vl))
    out = []
    for k in keys:
        h, v = hf[k], vl[k]
        out.append(
            {
                "event_id": k,
                "hf_logits": h.get("hf_logits") or {},
                "vllm_logits": v.get("vllm_logits") or {},
                "hf_pred": h.get("pred_operation"),
                "vllm_pred": v.get("pred_operation"),
                "prompt_sha256": h.get("prompt_sha256"),
                "vllm_prompt_sha256": v.get("prompt_sha256"),
                "candidate_list_sha256": h.get("candidate_list_sha256"),
                "vllm_candidate_list_sha256": v.get("candidate_list_sha256"),
            }
        )
    return out


def raw_decide(logits: dict, threshold: float = 0.0) -> str:
    d = decide_rollback_operation(
        score_continue=float(logits.get("CONTINUE", -1e9)),
        score_replan=float(logits.get("REPLAN", -1e9)),
        score_rollback=float(logits.get("ROLLBACK_TO", -1e9)),
        threshold=threshold,
        disable_replan=True,
    )
    return d.predicted_operation.value


def agreement_stats(rows: list[dict], pred_a: list[str], pred_b: list[str]) -> dict:
    n = len(rows)
    agree = sum(1 for a, b in zip(pred_a, pred_b) if a == b)
    c2r = sum(1 for a, b in zip(pred_a, pred_b) if a == "CONTINUE" and b == "ROLLBACK_TO")
    r2c = sum(1 for a, b in zip(pred_a, pred_b) if a == "ROLLBACK_TO" and b == "CONTINUE")
    margins_hf = [margin_from_logits(r["hf_logits"]) for r in rows]
    margins_vl = [margin_from_logits(r["vllm_logits"]) for r in rows]
    near = sum(
        1
        for a, b, mh, mv in zip(pred_a, pred_b, margins_hf, margins_vl)
        if a != b and min(abs(mh), abs(mv)) < 0.05
    )
    qs = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    return {
        "n": n,
        "agreement": agree / max(n, 1),
        "mismatch": n - agree,
        "C_to_R_flips": c2r,
        "R_to_C_flips": r2c,
        "near_boundary_mismatch_count": near,
        "hf_margin_quantiles": {
            str(q): float(np.quantile(margins_hf, q)) for q in qs
        },
        "vllm_margin_quantiles": {
            str(q): float(np.quantile(margins_vl, q)) for q in qs
        },
        "margin_diff_quantiles": {
            str(q): float(np.quantile(np.array(margins_vl) - np.array(margins_hf), q))
            for q in qs
        },
    }


def fit_threshold(rows: list[dict]) -> dict:
    """Map vLLM margin via threshold so vLLM decisions match HF decisions."""
    hf_preds = [raw_decide(r["hf_logits"]) for r in rows]
    vl_margins = np.array([margin_from_logits(r["vllm_logits"]) for r in rows])
    # Search tau such that decide_binary(vl_margin, tau) ~= hf_pred
    # CONTINUE if margin > -tau
    candidates = sorted(set(np.round(np.linspace(-2.0, 2.0, 801), 4).tolist()))
    best = None
    for tau in candidates:
        preds = [decide_binary(float(m), tau) for m in vl_margins]
        agr = sum(1 for a, b in zip(hf_preds, preds) if a == b) / max(len(rows), 1)
        if best is None or agr > best["agreement"]:
            best = {"method": "threshold", "threshold": float(tau), "agreement": agr, "a": 1.0, "b": 0.0}
    return best or {"method": "threshold", "threshold": 0.0, "agreement": 0.0, "a": 1.0, "b": 0.0}


def fit_affine(rows: list[dict]) -> dict:
    """Fit m' = a*m + b on vLLM margin to match HF margin sign / decisions."""
    hf_margins = np.array([margin_from_logits(r["hf_logits"]) for r in rows], dtype=np.float64)
    vl_margins = np.array([margin_from_logits(r["vllm_logits"]) for r in rows], dtype=np.float64)
    hf_preds = [raw_decide(r["hf_logits"]) for r in rows]
    # Least squares: hf ≈ a*vl + b
    A = np.vstack([vl_margins, np.ones(len(vl_margins))]).T
    try:
        coef, _, _, _ = np.linalg.lstsq(A, hf_margins, rcond=None)
        a0, b0 = float(coef[0]), float(coef[1])
    except Exception:
        a0, b0 = 1.0, 0.0
    best = None
    for da in np.linspace(-0.5, 0.5, 21):
        for db in np.linspace(-0.5, 0.5, 41):
            a, b = a0 + da, b0 + db
            mapped = a * vl_margins + b
            preds = ["CONTINUE" if m > 0 else "ROLLBACK_TO" for m in mapped]
            agr = sum(1 for x, y in zip(hf_preds, preds) if x == y) / max(len(rows), 1)
            if best is None or agr > best["agreement"]:
                best = {"method": "affine", "a": float(a), "b": float(b), "threshold": 0.0, "agreement": agr}
    return best or {"method": "affine", "a": 1.0, "b": 0.0, "threshold": 0.0, "agreement": 0.0}


def apply_calib(rows: list[dict], calib: dict) -> list[str]:
    preds = []
    for r in rows:
        m = margin_from_logits(r["vllm_logits"])
        if calib["method"] == "affine":
            m = calib["a"] * m + calib["b"]
            preds.append("CONTINUE" if m > 0 else "ROLLBACK_TO")
        else:
            preds.append(decide_binary(m, calib["threshold"]))
    return preds


def evaluate_seed(seed: int) -> dict:
    off = paired_rows(seed, "offline_valid")
    live = paired_rows(seed, "base_live")
    hf_off = [raw_decide(r["hf_logits"]) for r in off]
    vl_off = [raw_decide(r["vllm_logits"]) for r in off]
    hf_live = [raw_decide(r["hf_logits"]) for r in live]
    vl_live = [raw_decide(r["vllm_logits"]) for r in live]
    raw_off = agreement_stats(off, hf_off, vl_off)
    raw_live = agreement_stats(live, hf_live, vl_live)

    thr = fit_threshold(off)
    aff = fit_affine(off)
    best = thr if thr["agreement"] >= aff["agreement"] else aff

    cal_off_preds = apply_calib(off, best)
    cal_live_preds = apply_calib(live, best)
    cal_off = agreement_stats(off, hf_off, cal_off_preds)
    cal_live = agreement_stats(live, hf_live, cal_live_preds)

    seed_dir = FOLLOWUP / "phase_a" / "calibration" / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "calibration_fit_split": "offline_valid",
        "forbidden_fit_on_base_live": True,
        "chosen_calibration": best,
        "threshold_candidate": thr,
        "affine_candidate": aff,
        "offline_raw": raw_off,
        "offline_calibrated": cal_off,
        "base_live_raw": raw_live,
        "base_live_calibrated": cal_live,
        "prompt_hash_mismatch_offline": sum(
            1 for r in off if r["prompt_sha256"] != r["vllm_prompt_sha256"]
        ),
        "candidate_hash_mismatch_offline": sum(
            1 for r in off if r["candidate_list_sha256"] != r["vllm_candidate_list_sha256"]
        ),
        "prompt_hash_mismatch_base_live": sum(
            1 for r in live if r["prompt_sha256"] != r["vllm_prompt_sha256"]
        ),
        "candidate_hash_mismatch_base_live": sum(
            1 for r in live if r["candidate_list_sha256"] != r["vllm_candidate_list_sha256"]
        ),
        "gate_a1_pass_seed": (
            abs(cal_off["agreement"] - 1.0) < 1e-12
            and abs(cal_live["agreement"] - 1.0) < 1e-12
        ),
    }
    (seed_dir / "SEED_CALIBRATION.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def write_report(per_seed: list[dict]) -> dict:
    lines = [
        "# CALIBRATION_REPORT (Phase A1)",
        "",
        "Fit restricted to `offline_valid` only. `base_live` is a one-shot freeze evaluation.",
        "",
    ]
    all_pass = True
    for p in per_seed:
        seed = p["seed"]
        ok = p["gate_a1_pass_seed"]
        all_pass = all_pass and ok
        c = p["chosen_calibration"]
        lines += [
            f"## seed{seed}",
            "",
            f"- method: `{c['method']}` params={{{json.dumps({k: c[k] for k in c if k != 'agreement'})}}}",
            f"- offline raw agreement: **{p['offline_raw']['agreement']:.6f}** (mismatch={p['offline_raw']['mismatch']})",
            f"- offline calibrated agreement: **{p['offline_calibrated']['agreement']:.6f}** (mismatch={p['offline_calibrated']['mismatch']})",
            f"- base_live raw agreement: **{p['base_live_raw']['agreement']:.6f}** (mismatch={p['base_live_raw']['mismatch']})",
            f"- base_live calibrated agreement: **{p['base_live_calibrated']['agreement']:.6f}** (mismatch={p['base_live_calibrated']['mismatch']})",
            f"- C→R flips (cal live): {p['base_live_calibrated']['C_to_R_flips']}",
            f"- R→C flips (cal live): {p['base_live_calibrated']['R_to_C_flips']}",
            f"- near-boundary mismatch (cal live): {p['base_live_calibrated']['near_boundary_mismatch_count']}",
            f"- margin quantiles (vLLM offline): `{json.dumps(p['offline_raw']['vllm_margin_quantiles'], indent=None)}`",
            f"- A1 seed pass: `{ok}`",
            "",
        ]
    lines += [
        "## Gate A1",
        "",
        "Requires operation agreement == 1.000000 on 3 seeds × {offline_valid, base_live}.",
        f"",
        f"**GATE_A1_PASS = {all_pass}**",
        "",
        "If false: do not tune base_live further; proceed to A2 Canonical single-backend contract.",
        "",
    ]
    report_path = FOLLOWUP / "phase_a" / "calibration" / "CALIBRATION_REPORT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "GATE_A1_PASS": all_pass,
        "per_seed": per_seed,
        "next": "declare Gate A fixed" if all_pass else "enter A2 CanonicalRollbackOperationScorer",
    }
    (FOLLOWUP / "phase_a" / "calibration" / "CALIBRATION_SUMMARY.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, choices=SEEDS, default=None)
    args = p.parse_args()
    seeds = [args.seed] if args.seed is not None else list(SEEDS)
    per_seed = [evaluate_seed(s) for s in seeds]
    if args.seed is None:
        summary = write_report(per_seed)
        print(json.dumps({"GATE_A1_PASS": summary["GATE_A1_PASS"], "next": summary["next"]}, indent=2))
    else:
        print(json.dumps(per_seed[0], indent=2))


if __name__ == "__main__":
    main()
