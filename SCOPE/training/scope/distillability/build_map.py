"""Build distillability map and human-readable E0 report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from harness.capability.capability_id import CapabilityId, E0_PROBE_CAPABILITIES
from training.scope.distillability.metrics import (
    GLOBAL_METRICS,
    capability_specific_metrics,
    episodes_by_query,
)
from training.scope.distillability.modes import DistillabilityMode
from training.scope.distillability.registry import get_probe_spec
from training.scope.distillability.schema import CapabilityDistillabilityResult, ProcAuditStats
from training.scope.distillability.statistics import compute_distillability


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build E0 distillability map")
    p.add_argument("--root", default="outputs/scope_e0_distillability")
    p.add_argument(
        "--output-map",
        default="artifacts/capability/distillability_map.json",
    )
    p.add_argument(
        "--output-report",
        default="outputs/scope_e0_distillability/E0_REPORT.md",
    )
    p.add_argument(
        "--min-effect-sizes",
        default="0.005,0.01,0.02",
        help="Comma-separated thresholds for sensitivity analysis",
    )
    p.add_argument("--primary-metric", default="recall")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _load_episodes(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _decision_for(
    result: CapabilityDistillabilityResult,
    primary,
    min_effect: float,
) -> tuple[str, str]:
    cap = CapabilityId(result.capability_id)
    if not result.probe_supported:
        return "INCONCLUSIVE", "probe not supported"
    if primary is None or not primary.probe_valid:
        if abs(primary.delta_full if primary else 0.0) < min_effect:
            return "INCONCLUSIVE", "delta_full below min_effect_size"
        return "INCONCLUSIVE", primary.invalid_reason if primary else "missing primary metric"

    P = primary.P_raw
    d_full = primary.delta_full
    notes: list[str] = []

    if cap == CapabilityId.DETERMINISTIC_TRUNCATION:
        return "RUNTIME", "deterministic truncation is runtime-only"

    if cap == CapabilityId.EXTERNAL_VERIFICATION:
        if P is not None and P < 0.5 and d_full > min_effect:
            return "RUNTIME", "benefit depends on external verifier information"
        if P is not None and P >= 0.7:
            notes.append("unexpected high P — check external_call_count")
            return "INCONCLUSIVE", "; ".join(notes)

    if P is not None and P >= 0.7 and d_full > min_effect:
        return "DISTILL", f"P={P:.2f}, delta_full={d_full:.4f}"
    if P is not None and 0.3 <= P < 0.7:
        return "HYBRID", f"partial procedural recovery P={P:.2f}"
    if P is not None and P < 0.3 and d_full > min_effect:
        return "RUNTIME", f"low procedural share P={P:.2f}"

    return "INCONCLUSIVE", "effect or CI inconclusive"


def build_capability_result(
    capability: CapabilityId,
    root: Path,
    *,
    min_effect: float,
    primary_metric: str,
    bootstrap: int,
    seed: int,
) -> CapabilityDistillabilityResult:
    spec = get_probe_spec(capability)
    off_eps = _load_episodes(root / capability.value / "off" / "episodes.jsonl")
    proc_eps = _load_episodes(root / capability.value / "proc" / "episodes.jsonl")
    full_eps = _load_episodes(root / capability.value / "full" / "episodes.jsonl")

    off_by_q = {q: float(ep.get(primary_metric, 0.0)) for q, ep in episodes_by_query(off_eps).items()}
    proc_by_q = {q: float(ep.get(primary_metric, 0.0)) for q, ep in episodes_by_query(proc_eps).items()}
    full_by_q = {q: float(ep.get(primary_metric, 0.0)) for q, ep in episodes_by_query(full_eps).items()}

    metrics: dict[str, Any] = {}
    for metric in GLOBAL_METRICS:
        metrics[metric] = compute_distillability(
            metric=metric,
            off_by_q={q: float(episodes_by_query(off_eps).get(q, {}).get(metric, 0.0)) for q in off_by_q},
            proc_by_q={q: float(episodes_by_query(proc_eps).get(q, {}).get(metric, 0.0)) for q in proc_by_q},
            full_by_q={q: float(episodes_by_query(full_eps).get(q, {}).get(metric, 0.0)) for q in full_by_q},
            min_effect_size=min_effect,
            n_resamples=bootstrap,
            seed=seed,
        )

    proc_audit = ProcAuditStats()
    proc_summary_path = root / capability.value / "proc" / "summary.json"
    if proc_summary_path.exists():
        proc_summary = json.loads(proc_summary_path.read_text(encoding="utf-8"))
        audit_raw = proc_summary.get("proc_audit") or {}
        proc_audit = ProcAuditStats(
            visibility_violation_rate=float(audit_raw.get("visibility_violation_rate", 0.0)),
            new_observation_from_proc=int(audit_raw.get("new_observation_from_proc", 0)),
            external_call_from_proc=int(audit_raw.get("external_call_from_proc", 0)),
            hidden_field_access=int(audit_raw.get("hidden_field_access", 0)),
            state_mutation_rate=float(audit_raw.get("state_mutation_rate", 0.0)),
            n_proc_interventions=int(audit_raw.get("n_proc_interventions", 0)),
            n_shadow_calls=int(audit_raw.get("n_shadow_calls", 0)),
        )

    full_manifest = root / capability.value / "full" / "manifest.json"
    full_reused = False
    if full_manifest.exists():
        fm = json.loads(full_manifest.read_text(encoding="utf-8"))
        full_reused = bool(fm.get("reused_from"))

    result = CapabilityDistillabilityResult(
        capability_id=capability.value,
        probe_supported=spec.probe_supported,
        metrics=metrics,
        capability_metrics={
            "off": capability_specific_metrics(capability.value, off_eps),
            "proc": capability_specific_metrics(capability.value, proc_eps),
            "full": capability_specific_metrics(capability.value, full_eps),
        },
        proc_audit=proc_audit,
        full_reused=full_reused,
    )
    primary = metrics.get(primary_metric)
    decision, notes = _decision_for(result, primary, min_effect)
    if not proc_audit.information_safe and spec.proc_supported:
        decision = "INCONCLUSIVE"
        notes = f"PROC information-safe audit failed: {notes}"
    result.decision = decision
    result.decision_notes = notes
    return result


def render_report(
    results: dict[str, CapabilityDistillabilityResult],
    *,
    min_effect: float,
    sensitivity: list[float],
) -> str:
    lines = [
        "# SCOPE E0 Capability Distillability Report",
        "",
        f"Primary metric: recall | min_effect_size={min_effect}",
        "",
        "## Summary",
        "",
        "| Capability | OFF | PROC | FULL | Δproc | Δfull | P | CI(P) | Probe valid | Decision |",
        "| ---------- | --: | ---: | ---: | ----: | ----: | -: | -- | ----------- | -------- |",
    ]
    for cap, res in results.items():
        primary = res.metrics.get("recall")
        if primary is None:
            continue
        ci = primary.ci95.get("P", [0, 0])
        ci_s = f"[{ci[0]:.2f},{ci[1]:.2f}]" if primary.probe_valid else "n/a"
        p_s = f"{primary.P_raw:.2f}" if primary.P_raw is not None else "n/a"
        lines.append(
            f"| {cap} | {primary.R_off:.4f} | {primary.R_proc:.4f} | {primary.R_full:.4f} "
            f"| {primary.delta_proc:+.4f} | {primary.delta_full:+.4f} | {p_s} | {ci_s} "
            f"| {primary.probe_valid} | {res.decision} |"
        )

    lines.extend(["", "## Per-capability detail", ""])
    for cap, res in results.items():
        lines.append(f"### {cap}")
        lines.append(f"- Decision: **{res.decision}** — {res.decision_notes}")
        lines.append(f"- FULL reused: {res.full_reused}")
        if res.proc_audit.n_shadow_calls:
            lines.append(f"- PROC audit: {json.dumps(res.proc_audit.to_dict())}")
        primary = res.metrics.get("recall")
        if primary:
            lines.append(
                f"- Paired PROC vs OFF: wins={primary.paired_wins} "
                f"losses={primary.paired_losses} ties={primary.paired_ties}"
            )
        cap_m = res.capability_metrics
        if cap_m:
            lines.append(f"- Capability metrics: `{json.dumps(cap_m)}`")
        lines.append("")

    lines.extend(["## Threshold sensitivity", ""])
    for thr in sensitivity:
        lines.append(f"- min_effect_size={thr}")
    lines.append("")
    lines.append("## Next steps (830q)")
    lines.append("")
    for cap, res in results.items():
        primary = res.metrics.get("recall")
        if primary and primary.probe_valid and abs(primary.delta_full) >= min_effect:
            lines.append(f"- **{cap}**: candidate for 830q (delta_full={primary.delta_full:+.4f})")
        else:
            lines.append(f"- {cap}: skip 830q — INCONCLUSIVE / LOW_EFFECT")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    min_effect = float(args.min_effect_sizes.split(",")[0])
    sensitivity = [float(x) for x in args.min_effect_sizes.split(",") if x.strip()]

    results: dict[str, CapabilityDistillabilityResult] = {}
    for cap in E0_PROBE_CAPABILITIES:
        results[cap.value] = build_capability_result(
            cap,
            root,
            min_effect=min_effect,
            primary_metric=args.primary_metric,
            bootstrap=args.bootstrap,
            seed=args.seed,
        )

    out_map = Path(args.output_map)
    out_map.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v.to_dict() for k, v in results.items()}
    payload["_meta"] = {
        "min_effect_size": min_effect,
        "sensitivity_thresholds": sensitivity,
        "primary_metric": args.primary_metric,
    }
    out_map.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    report = render_report(results, min_effect=min_effect, sensitivity=sensitivity)
    out_report = Path(args.output_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(report, encoding="utf-8")
    print(f"Wrote {out_map} and {out_report}", flush=True)


if __name__ == "__main__":
    main()
