"""Probe C offline diagnosis + re-judge (guide §17.2, R0 pure-CPU path).

- ``probe_c_diagnose``: read an OLD run's TEACHER_FUNNEL.jsonl / CACHE.json and
  re-bucket rejections by structured reason (v4 attempted/passed semantics +
  materialize reason codes). Never mutates the old run.
- ``probe_c_rejudge``: re-score old SCREEN rows under GATE_V4 from their
  recorded pre/post values. Fields the old run lacks stay MISSING — never 0.

Both are offline: no GPU, no state collection, no silent fallbacks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from trim.eval.probe_c_metrics import (
    GATE_POLICY_V4,
    MISSING,
    RegenStats,
    judge_c1,
)
from trim.probes.probe_c_cache import (
    FunnelCounts,
    FunnelRow,
    STATUS_READY,
)

DIAG_VERSION = "probe_c_diagnose_v4"
REJUDGE_VERSION = "probe_c_rejudge_v4"


# ---------------------------------------------------------------------------
# diagnose
# ---------------------------------------------------------------------------


def _load_funnel_rows(out_dir: Path) -> dict[str, list[dict[str, Any]]]:
    by_component: dict[str, list[dict[str, Any]]] = {}
    roots = [out_dir, out_dir / "runner" / "*" if out_dir.name != "*" else out_dir]
    # Old runs nest cache under runner/<model_id>/; accept both layouts.
    for pattern in (
        "cache/*/TEACHER_FUNNEL.jsonl",
        "*/*/cache/*/TEACHER_FUNNEL.jsonl",
    ):
        for path in sorted(out_dir.glob(pattern)):
            cid = path.parent.name
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            by_component[cid] = rows
        if by_component:
            break
    if not by_component:
        raise FileNotFoundError(
            f"no TEACHER_FUNNEL.jsonl under {out_dir}/cache/*/ or "
            f"{out_dir}/*/cache/*/; point --input at the OLD run root (the dir "
            "containing cache/<component>/ or runner/<model>/cache/<component>/)"
        )
    return by_component


def diagnose(out_dir: Path, *, out: Path) -> dict[str, Any]:
    """Re-bucket old funnel rows under v4 attempted/passed semantics."""
    by_component = _load_funnel_rows(out_dir)
    report: dict[str, Any] = {
        "diagnosis_version": DIAG_VERSION,
        "input": str(out_dir),
        "components": {},
        "note": "legacy stage_reached counts rows that ENTERED a stage; v4 "
        "passed counts are derived from status==READY rows only (legacy rows "
        "cannot recover finer pass semantics — per-stage reason buckets below "
        "are the authoritative rejection split)",
    }
    for cid, raw_rows in by_component.items():
        rows = [
            FunnelRow(**{k: v for k, v in r.items() if k in FunnelRow.__dataclass_fields__})
            for r in raw_rows
        ]
        funnel = FunnelCounts(component_id=cid, rows=rows)
        # Rejection buckets by (stage_reached, reject_reason) — the actionable
        # split for ccso's 95-direct question (guide §4.4).
        buckets: dict[str, int] = {}
        for r in rows:
            reason = (r.reject_reason or "").split(":")[0] or "none"
            key = f"{r.stage_reached}|{reason}"
            buckets[key] = buckets.get(key, 0) + 1
        # Specific counts the guide asks for first.
        n_reached_direct = sum(
            1 for r in rows if r.stage_reached == "projection_direct"
        )
        n_ready = sum(1 for r in rows if r.status == STATUS_READY)
        capacity_like = sum(
            1
            for r in rows
            if "CAPACITY" in (r.reject_reason or "").upper()
        )
        report["components"][cid] = {
            "n_rows": len(rows),
            "n_status_ready": n_ready,
            "n_reached_direct_stage": n_reached_direct,
            "n_ready_at_direct": n_ready,
            "n_capacity_like_rejects": capacity_like,
            "rejection_buckets": dict(sorted(buckets.items(), key=lambda kv: -kv[1])),
            "v4_summary": funnel.to_dict_v4(),
        }
    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "DIAGNOSIS.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return report


# ---------------------------------------------------------------------------
# rejudge
# ---------------------------------------------------------------------------


def _r_from_pre_post(row: Mapping[str, Any], pre_key: str, post_key: str) -> Any:
    pre, post = row.get(pre_key, MISSING), row.get(post_key, MISSING)
    if pre in (MISSING, None) or post in (MISSING, None):
        return MISSING
    try:
        return float(pre) - float(post)
    except (TypeError, ValueError):
        return MISSING


def rejudge(screen_path: Path, *, out: Path, policy: str = GATE_POLICY_V4) -> dict[str, Any]:
    """Re-score old SCREEN rows under GATE_V4 from recorded pre/post values.

    Old rows carry d_k_pre/post + nll_pre/post + generated_invalid_*/pre —
    enough for point estimates. Old CI / guard wiring did not exist: guard is
    MISSING and can never produce a new PROMOTE (guide §9.2/§8.4).
    """
    screen = json.loads(Path(screen_path).read_text(encoding="utf-8"))
    rows = screen.get("rows") or []
    rejudged: list[dict[str, Any]] = []
    for r in rows:
        r_n = _r_from_pre_post(r, "nll_pre", "nll_post")
        r_d = _r_from_pre_post(r, "d_k_pre", "d_k_post")
        pre_regen = post_regen = None
        inv_pre, inv_post = r.get("generated_invalid_pre"), r.get("generated_invalid_post")
        if inv_pre not in (MISSING, None) and inv_post not in (MISSING, None):
            try:
                # Reconstruct minimal RegenStats from the recorded rates.
                pre_regen = _stats_from_rate(float(inv_pre))
                post_regen = _stats_from_rate(float(inv_post))
            except (TypeError, ValueError):
                pre_regen = post_regen = None
        verdict = judge_c1(
            pre_js=r.get("d_k_pre", MISSING),
            post_js=r.get("d_k_post", MISSING),
            pre_nll=r.get("nll_pre", MISSING),
            post_nll=r.get("nll_post", MISSING),
            pre_name_js=MISSING,
            post_name_js=MISSING,
            pre_regen=pre_regen,
            post_regen=post_regen,
            support_status="READY",
            ci_nll=None,
            ci_dk=None,
            require_generation_guard=False,  # legacy rows: guard never wired
            absolute_quality_status="QUALITY_UNRESOLVED",
            policy_version=policy,
        )
        rejudged.append(
            {
                "component_id": r.get("component_id"),
                "train_seed": r.get("train_seed"),
                "fixed_budget": r.get("fixed_budget"),
                "legacy_status": r.get("status"),
                "rejudged_status": verdict.status,
                "rejudged_reason": verdict.reason,
                "R_N": r_n,
                "R_D": r_d,
                "detail": verdict.to_dict(),
            }
        )
    report = {
        "rejudge_version": REJUDGE_VERSION,
        "policy_version": policy,
        "input": str(screen_path),
        "n_rows": len(rows),
        "rows": rejudged,
        "note": "legacy rows have no CI and no wired generation guard: no row "
        "can re-judge to PROMOTE under v4; PROMOTE candidates must re-run "
        "screen with repaired measurement (R2)",
    }
    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "REJUDGED.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return report


def _stats_from_rate(rate: float, *, n: int = 100) -> RegenStats:
    """Minimal RegenStats reconstructing a recorded invalid rate.

    ``n`` is synthetic (old rows kept only the rate); the guard consumes the
    rate, so the reconstruction preserves the decision, not the denominator.
    """
    n_bad = int(round(float(rate) * n))
    stats = RegenStats(n_attempted=n, n_ok=n - n_bad)
    return stats


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="probe_c_diagnose")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_diag = sub.add_parser("diagnose", help="re-bucket old funnel rejections (CPU only)")
    p_diag.add_argument("--input", required=True, help="OLD run root (contains cache/<cid>/)")
    p_diag.add_argument("--out", required=True, help="output dir for DIAGNOSIS.json")

    p_rej = sub.add_parser("rejudge", help="re-score old SCREEN.json under GATE_V4")
    p_rej.add_argument("--input", required=True, help="old SCREEN.json path")
    p_rej.add_argument("--out", required=True, help="output dir for REJUDGED.json")
    p_rej.add_argument("--policy", default=GATE_POLICY_V4)

    args = parser.parse_args(argv)
    try:
        if args.cmd == "diagnose":
            report = diagnose(Path(args.input), out=Path(args.out))
            summary = {
                cid: {
                    "n_rows": c["n_rows"],
                    "n_status_ready": c["n_status_ready"],
                    "top_buckets": dict(
                        list(c["rejection_buckets"].items())[:5]
                    ),
                }
                for cid, c in report["components"].items()
            }
            print(json.dumps(summary, indent=2, ensure_ascii=False)[:3000])
        else:
            report = rejudge(Path(args.input), out=Path(args.out), policy=args.policy)
            by_status: dict[str, int] = {}
            for r in report["rows"]:
                by_status[r["rejudged_status"]] = by_status.get(r["rejudged_status"], 0) + 1
            print(json.dumps({"n_rows": report["n_rows"], "by_status": by_status}, indent=2))
        return 0
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"[probe_c_diagnose] REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["diagnose", "rejudge"]
