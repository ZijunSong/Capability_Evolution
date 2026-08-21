#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
STREAM_SRC = REPO / "outputs" / "0816_2_importance_proper_fork_formal_stream"
DEFAULT_K8_8424 = REPO / "outputs" / "0817_importance_k8_seed8424_full_rerun" / "shards" / "importance_tagging_K8.jsonl"
DEFAULT_OUT = REPO / "outputs" / "0816_2_importance_proper_fork_formal_final"


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def summarize(k: int, seed: int, path: Path, rows: list[dict]) -> dict:
    vals = [float(r.get("branch_T_minus_S", 0.0)) for r in rows]
    qids = {str(r.get("query_id")) for r in rows}
    states = {str(r.get("snapshot_hash")) for r in rows}
    takeover = sum(1 for r in rows if r.get("full_harness_takeover"))
    return {
        "K": k,
        "seed": seed,
        "file": str(path),
        "n_states": len(rows),
        "unique_qids": len(qids),
        "unique_snapshots": len(states),
        "mean_branch_T_minus_S": statistics.mean(vals) if vals else 0.0,
        "median_branch_T_minus_S": statistics.median(vals) if vals else 0.0,
        "positive_count": sum(v > 0 for v in vals),
        "negative_count": sum(v < 0 for v in vals),
        "zero_count": sum(v == 0 for v in vals),
        "full_harness_takeover_count": takeover,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--stream-src", type=Path, default=STREAM_SRC)
    ap.add_argument("--k8-seed8424", type=Path, default=DEFAULT_K8_8424)
    ap.add_argument("--allow-incomplete", action="store_true", help="Write diagnostics even if a shard has fewer than 512 rows.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    shards = [
        (4, 8423, args.stream_src / "K4_seed8423" / "shards" / "importance_tagging_K4.jsonl"),
        (4, 8424, args.stream_src / "K4_seed8424" / "shards" / "importance_tagging_K4.jsonl"),
        (8, 8423, args.stream_src / "K8_seed8423" / "shards" / "importance_tagging_K8.jsonl"),
        (8, 8424, args.k8_seed8424),
    ]

    all_rows = []
    summaries = []
    for k, seed, path in shards:
        rows = load_rows(path)
        if len(rows) < 512 and not args.allow_incomplete:
            raise SystemExit(f"Shard incomplete: K{k} seed{seed} has {len(rows)} rows at {path}")
        for i, row in enumerate(rows):
            row = dict(row)
            row["formal_K"] = k
            row["formal_seed"] = seed
            row["formal_row_id"] = f"importance_K{k}_seed{seed}_{i:04d}"
            row["source_shard"] = str(path)
            all_rows.append(row)
        summaries.append(summarize(k, seed, path, rows))

    with (out / "IMPORTANCE_PROPER_SUMMARY.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    write_jsonl(out / "IMPORTANCE_PROPER_VALUE_PER_STATE.jsonl", all_rows)

    k4_ok = all(s["K"] != 4 or s["mean_branch_T_minus_S"] > 0 for s in summaries)
    k8_ok = all(s["K"] != 8 or s["mean_branch_T_minus_S"] > 0 for s in summaries)
    n_ok = all(s["n_states"] >= 512 for s in summaries)
    takeover_ok = all(s["full_harness_takeover_count"] == 0 for s in summaries)
    gate_passed = bool(k4_ok and k8_ok and n_ok and takeover_ok)
    gate = {
        "status": "proper_fork_formal_gate_failed" if not gate_passed else "proper_fork_formal_gate_passed",
        "component": "importance_tagging",
        "contract": "same xi_t; full branch importance_tagging ON first action; reduced branch importance_tagging OFF first action; continuation reduced policy; no full-harness takeover",
        "stream_source_dir": str(args.stream_src),
        "k8_seed8424_source": str(args.k8_seed8424),
        "output_dir": str(out),
        "n_states_per_shard_required": 512,
        "seeds": [8423, 8424],
        "K": [4, 8],
        "rows": summaries,
        "n_ok": n_ok,
        "no_full_harness_takeover": takeover_ok,
        "proper_K4_positive": k4_ok,
        "K8_direction_consistent_positive": k8_ok,
        "gate_passed": gate_passed,
        "decision": "discard_importance_tagging_as_positive_component; do_not_start_importance_lora_opd" if not gate_passed else "eligible_for_importance_lora_opd",
    }
    (out / "IMPORTANCE_K4_K8_GATE.json").write_text(json.dumps(gate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# IMPORTANCE_MECHANISM_ANALYSIS",
        "",
        f"- status: `{gate['status']}`",
        "- action: do not start importance LoRA OPD from this formal gate result" if not gate_passed else "- action: importance LoRA OPD allowed by formal gate",
        "- contract: same xi_t, importance ON vs OFF first fork action, reduced continuation, no full-harness takeover",
        "",
        "## Formal Proper Fork Summary",
        "",
        "| seed | K | n | mean T-S | median T-S | pos | neg | zero | takeover |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['seed']} | {s['K']} | {s['n_states']} | {s['mean_branch_T_minus_S']:.9f} | {s['median_branch_T_minus_S']:.9f} | {s['positive_count']} | {s['negative_count']} | {s['zero_count']} | {s['full_harness_takeover_count']} |"
        )
    if gate_passed:
        interpretation = "The formal K4/K8 gate is positive under the configured decision rule. Actual-LoRA importance OPD may proceed, subject to mechanism/target audit."
    else:
        interpretation = "The formal K4/K8 gate is not positive under the configured decision rule. Under 0816-2, importance_tagging actual-LoRA OPD is blocked and should not be launched from this component."
    lines += [
        "",
        "## Interpretation",
        "",
        interpretation,
        "",
        "This formal gate supersedes the earlier approximate REAL_INFLUENCE_POSITIVE and the 64-state smoke gate for launch/no-launch decisions.",
    ]
    (out / "IMPORTANCE_MECHANISM_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    audit = [
        "# IMPORTANCE_DATA_AUDIT",
        "",
        f"- stream_source_dir: `{args.stream_src}`",
        f"- K8 seed8424 source: `{args.k8_seed8424}`",
        "- formal shards: K4/K8 x seeds 8423/8424",
        "- required rows per shard: 512",
        f"- rows per shard observed: `{[(s['K'], s['seed'], s['n_states']) for s in summaries]}`",
        f"- no full-harness takeover: `{takeover_ok}`",
        f"- result: `{gate['status']}`",
    ]
    (out / "IMPORTANCE_DATA_AUDIT.md").write_text("\n".join(audit) + "\n", encoding="utf-8")

    handoff = {
        "component": "importance_tagging",
        "status": gate["status"],
        "proper_fork_formal_complete": n_ok,
        "actual_lora_started": False,
        "actual_lora_blocked_reason": None if gate_passed else "proper K4/K8 formal fork is not positive",
        "gate": gate,
        "recommended_next_action": "mechanism_target_audit_then_actual_lora" if gate_passed else "switch_to_next_high_level_structured_control_candidate_or_run_substantive_contract_audit; do_not_tune_loss_to_rescue_importance",
    }
    (out / "H1002_IMPORTANCE_HANDOFF.json").write_text(json.dumps(handoff, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    manifest = {
        "stage": "0816_2_importance_proper_fork_formal_final",
        "status": gate["status"],
        "stream_source_dir": str(args.stream_src),
        "k8_seed8424_source": str(args.k8_seed8424),
        "generated_files": [
            "IMPORTANCE_PROPER_VALUE_PER_STATE.jsonl",
            "IMPORTANCE_PROPER_SUMMARY.csv",
            "IMPORTANCE_K4_K8_GATE.json",
            "IMPORTANCE_DATA_AUDIT.md",
            "IMPORTANCE_MECHANISM_ANALYSIS.md",
            "H1002_IMPORTANCE_HANDOFF.json",
            "SHA256SUMS",
        ],
    }
    (out / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    subprocess.run("find . -type f -not -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS", cwd=out, shell=True, check=True)
    print(json.dumps({"status": gate["status"], "output_dir": str(out), "gate_passed": gate_passed}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
