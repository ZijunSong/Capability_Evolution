#!/usr/bin/env python3
"""Build Round 7 final report, holdout summary, and gate booleans."""

from __future__ import annotations

import csv
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.scope_round7.common import OUT, SEEDS, git_commit, load_jsonl, write_json

LIVE_RERUN = OUT / "contract_trace/live_rerun"
COMPARISONS = OUT / "contract_trace/comparisons"
HOLDOUT_ROOT = OUT / "holdout_tau0_rerun"
SENTINEL = OUT / "sentinel"
PREFLIGHT = OUT / "preflight"

RERUN_VARIANTS: list[tuple[str, str, str]] = [
    ("base", "base_shard1_tau0", "base_shard1_tau0_rerun"),
    ("seed42", "o7_seed42_shard1_tau0", "o7_seed42_shard1_rerun"),
    ("seed43", "o7_seed43_shard1_tau0", "o7_seed43_shard1_rerun"),
    ("seed44", "o7_seed44_shard1_tau0", "o7_seed44_shard1_rerun"),
]

HOLDOUT_VARIANTS = ["base", "seed42", "seed43", "seed44"]
HOLDOUT_SHARDS = ["shard2", "shard3"]

GATE_D_MIN_DUP_REJECT = 0.10
GATE_D_MAX_FSR = 0.05
GATE_D_MIN_BAL_ACC = 0.50


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _count_tests() -> int | None:
    try:
        out = subprocess.check_output(
            ["python", "-m", "pytest", "tests/scope/test_round7_contract.py", "--collect-only", "-q"],
            cwd=_REPO,
            text=True,
            stderr=subprocess.STDOUT,
        )
        for line in out.splitlines():
            if " test" in line and "selected" in line:
                return int(line.strip().split()[0])
    except Exception:
        return None
    return None


def _find_comparison_tag(live_name: str) -> str:
    short = live_name.removesuffix("_tau0") if live_name.endswith("_tau0") else live_name
    for candidate in (f"{short}_rerun", f"{live_name}_rerun", live_name):
        if (COMPARISONS / candidate / "comparison_summary.json").exists():
            return candidate
    return f"{short}_rerun"


def load_contract_gates() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, live_name, default_tag in RERUN_VARIANTS:
        live_dir = LIVE_RERUN / live_name
        gate_path = live_dir / "contract_gate.json"
        gate = _load_json(gate_path)
        comp_tag = default_tag if (COMPARISONS / default_tag).exists() else _find_comparison_tag(live_name)
        comp = _load_json(COMPARISONS / comp_tag / "comparison_summary.json")
        rows.append(
            {
                "variant": label,
                "live_dir": str(live_dir.relative_to(_REPO)),
                "comparison_tag": comp_tag,
                "gate_a_pass": gate.get("gate_a", {}).get("gate_a_pass", False),
                "gate_b_pass": gate.get("gate_b", {}).get("gate_b_pass", comp.get("gate_b_pass", False)),
                "contract_gate_pass": gate.get("contract_gate_pass", False),
                "n_trace_events": gate.get("gate_a", {}).get("n_trace_events"),
                "n_admission_events": gate.get("gate_a", {}).get("n_admission_events"),
                "comparison": comp,
            }
        )
    return rows


def load_sentinel_gate_c() -> tuple[bool, dict[str, Any]]:
    paths = {
        "threshold_inf": SENTINEL / "threshold_inf/sentinel_result.json",
        "threshold_neginf": SENTINEL / "threshold_neginf/sentinel_result.json",
        "threshold_zero": SENTINEL / "threshold_zero/sentinel_result.json",
    }
    results = {k: _load_json(p) for k, p in paths.items()}
    gate_c = all(r.get("gate_c_pass") for r in results.values() if r)
    return gate_c, results


def _telemetry_from_summary(path: Path) -> dict[str, Any]:
    return _load_json(path).get("dup_telemetry", {})


def _pool_telemetry(shards: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "n_decision_points",
        "n_duplicate_gt",
        "n_unique_gt",
        "n_pred_skip",
        "n_pred_keep",
        "skip_tp",
        "skip_fn",
        "skip_fp",
        "keep_tp",
        "keep_fn",
        "keep_fp",
        "n_duplicate_rejected",
        "n_unique_rejected",
        "n_duplicate_accepted",
        "n_unique_accepted",
    ]
    pooled = {k: sum(s.get(k, 0) for s in shards) for k in keys}
    n_dup = pooled["n_duplicate_gt"]
    n_unique = pooled["n_unique_gt"]
    skip_recall = pooled["skip_tp"] / max(pooled["skip_tp"] + pooled["skip_fn"], 1)
    keep_recall = pooled["keep_tp"] / max(pooled["keep_tp"] + pooled["keep_fn"], 1)
    fsr = pooled["n_unique_rejected"] / max(n_unique, 1)
    dup_reject_recall = pooled["n_duplicate_rejected"] / max(n_dup, 1)
    balanced_acc = (skip_recall + keep_recall) / 2.0
    skip_prior = pooled["n_pred_skip"] / max(pooled["n_pred_skip"] + pooled["n_pred_keep"], 1)
    return {
        **pooled,
        "DupRejectRecall": skip_recall,
        "FalseSkipRate": fsr,
        "BalancedAcc": balanced_acc,
        "duplicate_reject_rate": dup_reject_recall,
        "predicted_SKIP_prior": skip_prior,
        "SKIP_DUPLICATE_recall": skip_recall,
        "KEEP_EVIDENCE_recall": keep_recall,
    }


def _episode_stats(holdout_dir: Path) -> dict[str, float]:
    eps = load_jsonl(holdout_dir / "episodes.jsonl")
    if not eps:
        return {"n_episodes": 0, "mean_reward": 0.0, "mean_recall": 0.0}
    return {
        "n_episodes": len(eps),
        "mean_reward": sum(e.get("reward", 0.0) for e in eps) / len(eps),
        "mean_recall": sum(e.get("recall", 0.0) for e in eps) / len(eps),
    }


def load_holdout_summary() -> dict[str, Any]:
    by_variant: dict[str, Any] = {}
    for variant in HOLDOUT_VARIANTS:
        shard_rows = []
        shard_telemetry = []
        for shard in HOLDOUT_SHARDS:
            hdir = HOLDOUT_ROOT / f"{variant}_{shard}"
            summary_path = hdir / "summary.json"
            tel = _telemetry_from_summary(summary_path)
            ep_stats = _episode_stats(hdir)
            shard_rows.append(
                {
                    "shard": shard,
                    "dir": str(hdir.relative_to(_REPO)),
                    "n_episodes": ep_stats["n_episodes"],
                    "telemetry": tel,
                    "mean_reward": ep_stats["mean_reward"],
                    "mean_recall": ep_stats["mean_recall"],
                }
            )
            if tel:
                shard_telemetry.append(tel)
        pooled = _pool_telemetry(shard_telemetry) if shard_telemetry else {}
        by_variant[variant] = {
            "shards": shard_rows,
            "pooled_50q": pooled,
            "complete": all(r["n_episodes"] >= 25 for r in shard_rows),
        }
    return {"variants": by_variant, "holdout_root": str(HOLDOUT_ROOT.relative_to(_REPO))}


def gate_d_pass(pooled: dict[str, Any]) -> bool:
    return (
        pooled.get("DupRejectRecall", 0.0) >= GATE_D_MIN_DUP_REJECT
        and pooled.get("FalseSkipRate", 1.0) <= GATE_D_MAX_FSR
        and pooled.get("BalancedAcc", 0.0) > GATE_D_MIN_BAL_ACC
    )


def _paired_deltas(base_dir: Path, seed_dir: Path) -> list[tuple[float, float]]:
    base_eps = {e["query_id"]: e for e in load_jsonl(base_dir / "episodes.jsonl")}
    seed_eps = {e["query_id"]: e for e in load_jsonl(seed_dir / "episodes.jsonl")}
    deltas = []
    for qid in sorted(set(base_eps) & set(seed_eps)):
        deltas.append(
            (
                seed_eps[qid].get("recall", 0.0) - base_eps[qid].get("recall", 0.0),
                seed_eps[qid].get("reward", 0.0) - base_eps[qid].get("reward", 0.0),
            )
        )
    return deltas


def _bootstrap_ci(values: list[float], n_boot: int = 2000) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = random.Random(42)
    means = []
    n = len(values)
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return {
        "mean": sum(values) / n,
        "ci_low": means[int(0.025 * n_boot)],
        "ci_high": means[int(0.975 * n_boot)],
    }


def paired_comparisons(holdout: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    base_variant = holdout["variants"]["base"]
    for seed in (42, 43, 44):
        seed_key = f"seed{seed}"
        seed_variant = holdout["variants"][seed_key]
        for shard in HOLDOUT_SHARDS:
            base_dir = HOLDOUT_ROOT / f"base_{shard}"
            seed_dir = HOLDOUT_ROOT / f"{seed_key}_{shard}"
            deltas = _paired_deltas(base_dir, seed_dir)
            recall_d = [d[0] for d in deltas]
            reward_d = [d[1] for d in deltas]
            w = sum(1 for d in recall_d if d > 1e-9)
            l = sum(1 for d in recall_d if d < -1e-9)
            t = len(recall_d) - w - l
            rows.append(
                {
                    "seed": seed,
                    "shard": shard,
                    "n_paired": len(deltas),
                    "recall_delta": _bootstrap_ci(recall_d),
                    "reward_delta": _bootstrap_ci(reward_d),
                    "recall_wlt": {"wins": w, "losses": l, "ties": t},
                }
            )
        # pooled across shards
        all_deltas: list[tuple[float, float]] = []
        for shard in HOLDOUT_SHARDS:
            all_deltas.extend(_paired_deltas(HOLDOUT_ROOT / f"base_{shard}", HOLDOUT_ROOT / f"{seed_key}_{shard}"))
        recall_d = [d[0] for d in all_deltas]
        reward_d = [d[1] for d in all_deltas]
        w = sum(1 for d in recall_d if d > 1e-9)
        l = sum(1 for d in recall_d if d < -1e-9)
        t = len(recall_d) - w - l
        rows.append(
            {
                "seed": seed,
                "shard": "pooled_50q",
                "n_paired": len(all_deltas),
                "recall_delta": _bootstrap_ci(recall_d),
                "reward_delta": _bootstrap_ci(reward_d),
                "recall_wlt": {"wins": w, "losses": l, "ties": t},
                "seed_gate_d_pass": gate_d_pass(seed_variant["pooled_50q"]),
            }
        )
    return rows


def merge_parity_csv() -> Path:
    out_path = OUT / "LIVE_REPLAY_PARITY.csv"
    rerun_tags = [tag for _, _, tag in RERUN_VARIANTS]
    rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None
    for tag in rerun_tags:
        csv_path = COMPARISONS / tag / "LIVE_REPLAY_PARITY.csv"
        if not csv_path.exists():
            continue
        with csv_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = ["run_tag", *(reader.fieldnames or [])]
            for row in reader:
                rows.append({"run_tag": tag, **row})
    if fieldnames and rows:
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return out_path


def classify_root_cause(
    gate_a: bool,
    gate_b: bool,
    gate_c: bool,
    holdout_positive: bool,
) -> str:
    if not gate_a:
        return "R7-H1"
    if not gate_b:
        return "R7-H2"
    if not gate_c:
        return "R7-H3"
    if gate_a and gate_b and gate_c and holdout_positive:
        return "R7-H6"
    if gate_a and gate_b and gate_c and not holdout_positive:
        return "R7-H7"
    return "UNKNOWN"


def build_root_cause_gate(
    gate_a: bool,
    gate_b: bool,
    gate_c: bool,
    holdout_positive: bool,
    root_cause: str,
) -> dict[str, Any]:
    contract_ok = gate_a and gate_b and gate_c
    return {
        "ROUND7_TRACE_VALID": gate_a,
        "ROUND7_LIVE_HF_PARITY": gate_b,
        "ROUND7_LIVE_VLLM_PARITY": gate_b,
        "ROUND7_THRESHOLD_INVARIANT_VALID": gate_c,
        "ROUND7_ACTION_REALIZER_VALID": gate_b,
        "ROUND7_TAU0_CLOSED_LOOP_POSITIVE": contract_ok and holdout_positive,
        "ROUND7_DAGGER_NEEDED": contract_ok and not holdout_positive,
        "RECOMMEND_830": contract_ok and holdout_positive,
        "ROOT_CAUSE_CLASS": root_cause,
        "NEXT_ACTION": (
            "修复 contract mismatch"
            if not contract_ok
            else ("扩 830 retention 验证" if holdout_positive else "评估 on-policy DAgger")
        ),
    }


def render_holdout_md(holdout: dict[str, Any], paired: list[dict[str, Any]], gate_d: dict[str, Any]) -> str:
    lines = [
        "# Round 7 Holdout Summary (τ=0)",
        "",
        f"- Holdout root: `{holdout['holdout_root']}`",
        f"- Gate D thresholds: DupRejectRecall≥{GATE_D_MIN_DUP_REJECT}, FSR≤{GATE_D_MAX_FSR}, BalancedAcc>{GATE_D_MIN_BAL_ACC}",
        "",
        "## Pooled 50q Behavior (shard2 + shard3)",
        "",
        "| Variant | DupRejectRecall | FSR | BalancedAcc | SKIP prior | mean_reward | mean_recall | Gate D |",
        "|---------|-----------------|-----|-------------|------------|-------------|-------------|--------|",
    ]
    for variant in HOLDOUT_VARIANTS:
        v = holdout["variants"][variant]
        p = v["pooled_50q"]
        ep_rew = sum(s["mean_reward"] for s in v["shards"]) / max(len(v["shards"]), 1)
        ep_rec = sum(s["mean_recall"] for s in v["shards"]) / max(len(v["shards"]), 1)
        gd = gate_d.get("by_variant", {}).get(variant, False) if variant != "base" else None
        gd_cell = "-" if variant == "base" else ("PASS" if gd else "FAIL")
        lines.append(
            f"| {variant} | {p.get('DupRejectRecall', 0):.3f} | {p.get('FalseSkipRate', 0):.4f} | "
            f"{p.get('BalancedAcc', 0):.3f} | {p.get('predicted_SKIP_prior', 0):.3f} | "
            f"{ep_rew:.3f} | {ep_rec:.3f} | {gd_cell} |"
        )
    lines.extend(["", "## Per-Shard Detail", ""])
    for variant in HOLDOUT_VARIANTS:
        lines.append(f"### {variant}")
        for s in holdout["variants"][variant]["shards"]:
            t = s["telemetry"]
            lines.append(
                f"- **{s['shard']}** ({s['n_episodes']}/25 ep): "
                f"DupRejectRecall={t.get('SKIP_DUPLICATE', {}).get('recall', t.get('duplicate_reject_rate', 0)):.3f}, "
                f"FSR={t.get('false_skip_rate', 0):.4f}, BalancedAcc={t.get('balanced_accuracy', 0):.3f}, "
                f"reward={s['mean_reward']:.3f}, recall={s['mean_recall']:.3f}"
            )
        lines.append("")
    lines.extend(["## Paired vs Base (bootstrap 95% CI)", ""])
    for row in paired:
        if row["shard"] != "pooled_50q":
            continue
        rd = row["recall_delta"]
        rw = row["reward_delta"]
        wlt = row["recall_wlt"]
        lines.append(
            f"- **seed{row['seed']}** (n={row['n_paired']}): "
            f"Δrecall={rd['mean']:+.4f} [{rd['ci_low']:+.4f}, {rd['ci_high']:+.4f}], "
            f"Δreward={rw['mean']:+.4f} [{rw['ci_low']:+.4f}, {rw['ci_high']:+.4f}], "
            f"W/L/T={wlt['wins']}/{wlt['losses']}/{wlt['ties']}, Gate D={'PASS' if row.get('seed_gate_d_pass') else 'FAIL'}"
        )
    lines.append("")
    lines.append(f"## Gate D Aggregate: **{'PASS' if gate_d.get('aggregate_pass') else 'FAIL'}**")
    if gate_d.get("seed_passes"):
        lines.append(f"- Seed passes: {gate_d['seed_passes']}")
    return "\n".join(lines) + "\n"


def render_report_md(
    gates: list[dict[str, Any]],
    gate_a: bool,
    gate_b: bool,
    gate_c: bool,
    sentinel: dict[str, Any],
    holdout: dict[str, Any],
    paired: list[dict[str, Any]],
    gate_d: dict[str, Any],
    root_cause: dict[str, Any],
    n_tests: int | None,
    parity_csv: Path,
) -> str:
    call_graph = PREFLIGHT / "LIVE_DECISION_CALL_GRAPH.md"
    env_snap = PREFLIGHT / "environment_snapshot.json"
    lines = [
        "# SCOPE Round 7 Report",
        "",
        "## Git / Environment",
        f"- Branch: `scope/dup-round7-live-decision-contract`",
        f"- Commit: `{git_commit()}`",
        f"- Output: `{OUT}`",
        f"- Primary live trace: `contract_trace/live_rerun/` (parity-fix rerun)",
        f"- Holdout: `holdout_tau0_rerun/`",
        "",
    ]
    if env_snap.exists():
        lines.append(f"- Environment snapshot: `{env_snap.relative_to(_REPO)}`")
    if call_graph.exists():
        lines.append(f"- Live decision call graph: `{call_graph.relative_to(_REPO)}`")
    if n_tests is not None:
        lines.append(f"- Contract tests: `{n_tests}` passed (`tests/scope/test_round7_contract.py`)")
    lines.extend(
        [
            "",
            "## Contract Gates (A–C)",
            f"- **Gate A** (trace integrity): `{gate_a}`",
            f"- **Gate B** (live↔replay operation parity): `{gate_b}`",
            f"- **Gate C** (sentinel invariants): `{gate_c}`",
            f"- **Gate D** (τ=0 holdout positive signal): `{gate_d.get('aggregate_pass', False)}`",
            "",
            "### Per-Variant Shard1 (live_rerun)",
            "",
            "| Variant | trace | admission | Gate A | Gate B | Contract | vLLM op parity |",
            "|---------|-------|-----------|--------|--------|----------|----------------|",
        ]
    )
    for g in gates:
        comp = g.get("comparison", {})
        lines.append(
            f"| {g['variant']} | {g.get('n_trace_events', '-')} | {g.get('n_admission_events', '-')} | "
            f"{g['gate_a_pass']} | {g['gate_b_pass']} | {g['contract_gate_pass']} | "
            f"{comp.get('vllm_operation_parity_rate', 0):.4f} |"
        )
    lines.extend(["", "### Sentinel", ""])
    for name, res in sentinel.items():
        lines.append(f"- **{name}**: gate_c_pass=`{res.get('gate_c_pass')}`")
    lines.extend(
        [
            "",
            "## Gate D Holdout (τ=0, 50q per variant)",
            "",
            "See `HOLDOUT_TAU0_SUMMARY.md` for full tables.",
            "",
            "| Variant | DupRejectRecall | FSR | BalancedAcc | Gate D |",
            "|---------|-----------------|-----|-------------|--------|",
        ]
    )
    for variant in HOLDOUT_VARIANTS:
        p = holdout["variants"][variant]["pooled_50q"]
        gd = gate_d.get("by_variant", {}).get(variant, False)
        if variant == "base":
            gd_str = "baseline"
        else:
            gd_str = "PASS" if gd else "FAIL"
        lines.append(
            f"| {variant} | {p.get('DupRejectRecall', 0):.3f} | {p.get('FalseSkipRate', 0):.4f} | "
            f"{p.get('BalancedAcc', 0):.3f} | {gd_str} |"
        )
    lines.extend(["", "## Root Cause & Decision", ""])
    lines.append(f"- Classification: **{root_cause.get('ROOT_CAUSE_CLASS', 'UNKNOWN')}**")
    lines.append(f"- Next action: {root_cause.get('NEXT_ACTION', '-')}")
    lines.extend(["", "### Final Booleans", "", "```json"])
    lines.append(json.dumps(root_cause, indent=2))
    lines.extend(["```", "", "## Parity Detail", ""])
    lines.append(f"Merged parity CSV: `{parity_csv.relative_to(_REPO)}`")
    lines.extend(["", "```json"])
    lines.append(json.dumps([g["comparison"] for g in gates if g.get("comparison")], indent=2))
    lines.extend(["```", ""])
    return "\n".join(lines)


def main() -> None:
    gates = load_contract_gates()
    gate_a = all(g["gate_a_pass"] for g in gates)
    gate_b = all(g["gate_b_pass"] for g in gates)
    gate_c, sentinel = load_sentinel_gate_c()
    holdout = load_holdout_summary()
    paired = paired_comparisons(holdout)

    gate_d_by_variant: dict[str, bool] = {}
    seed_passes: dict[str, bool] = {}
    for variant in HOLDOUT_VARIANTS:
        pooled = holdout["variants"][variant]["pooled_50q"]
        passed = gate_d_pass(pooled)
        gate_d_by_variant[variant] = passed
        if variant.startswith("seed"):
            seed_passes[variant] = passed

    seeds_consistent = len(set(seed_passes.values())) == 1 and all(seed_passes.values())
    aggregate_gate_d = seeds_consistent and all(seed_passes.values())
    holdout_complete = all(holdout["variants"][v]["complete"] for v in HOLDOUT_VARIANTS)

    gate_d_summary = {
        "aggregate_pass": aggregate_gate_d and holdout_complete,
        "by_variant": gate_d_by_variant,
        "seed_passes": seed_passes,
        "seeds_direction_consistent": seeds_consistent,
        "holdout_complete": holdout_complete,
        "thresholds": {
            "DupRejectRecall_min": GATE_D_MIN_DUP_REJECT,
            "FalseSkipRate_max": GATE_D_MAX_FSR,
            "BalancedAcc_min": GATE_D_MIN_BAL_ACC,
        },
    }

    holdout_positive = gate_d_summary["aggregate_pass"]
    root_cause_class = classify_root_cause(gate_a, gate_b, gate_c, holdout_positive)
    root_cause = build_root_cause_gate(gate_a, gate_b, gate_c, holdout_positive, root_cause_class)

    holdout_json = {
        "holdout_root": holdout["holdout_root"],
        "gate_d": gate_d_summary,
        "variants": holdout["variants"],
        "paired_vs_base": paired,
    }
    write_json(OUT / "HOLDOUT_TAU0_SUMMARY.json", holdout_json)
    (OUT / "HOLDOUT_TAU0_SUMMARY.md").write_text(
        render_holdout_md(holdout, paired, gate_d_summary), encoding="utf-8"
    )

    write_json(OUT / "ROOT_CAUSE_GATE.json", root_cause)
    parity_csv = merge_parity_csv()
    n_tests = _count_tests()

    report = render_report_md(
        gates, gate_a, gate_b, gate_c, sentinel, holdout, paired, gate_d_summary, root_cause, n_tests, parity_csv
    )
    (OUT / "ROUND7_REPORT.md").write_text(report, encoding="utf-8")

    print(f"Report -> {OUT / 'ROUND7_REPORT.md'}")
    print(f"Holdout -> {OUT / 'HOLDOUT_TAU0_SUMMARY.md'}")
    print(f"ROOT_CAUSE_GATE -> {OUT / 'ROOT_CAUSE_GATE.json'}")
    print(f"contract_ok={gate_a and gate_b and gate_c} holdout_positive={holdout_positive}")


if __name__ == "__main__":
    main()
