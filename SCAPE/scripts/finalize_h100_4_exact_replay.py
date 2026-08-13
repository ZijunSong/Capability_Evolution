#!/usr/bin/env python3
"""Finalize H100-4 exact replay cross-machine reproducibility artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scape.common.manifest import build_run_manifest, finalize_run_manifest, write_run_manifest
from scape.common.sha256sums import write_sha256sums
from scape.common.status import write_status_live

CELLS = [
    ("subtractive_curation", 2),
    ("subtractive_curation", 4),
    ("importance_tagging", 2),
    ("importance_tagging", 4),
    ("verify_tool", 2),
    ("verify_tool", 4),
]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({k for r in rows for k in r})
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mx = sum(xs) / len(xs); my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs); vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def _ranks(vals: list[float]) -> list[float]:
    order = sorted(enumerate(vals), key=lambda kv: kv[1])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and order[j][1] == order[i][1]:
            j += 1
        avg = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            ranks[order[k][0]] = avg
        i = j
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    return _pearson(_ranks(xs), _ranks(ys)) if len(xs) >= 2 else None


def _sign(x: float) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


def _topk_agreement(rows: list[dict[str, Any]], k: int = 16) -> float | None:
    if not rows:
        return None
    kk = min(k, len(rows))
    h2 = {r["state_id"] for r in sorted(rows, key=lambda r: float(r["utility_h1002"]), reverse=True)[:kk]}
    h4 = {r["state_id"] for r in sorted(rows, key=lambda r: float(r["utility_h1004"]), reverse=True)[:kk]}
    return len(h2 & h4) / max(1, kk)


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    xs = [float(r["utility_h1002"]) for r in rows]
    ys = [float(r["utility_h1004"]) for r in rows]
    diffs = [abs(y - x) for x, y in zip(xs, ys)]
    return {
        "n_states": len(rows),
        "pearson": _pearson(xs, ys),
        "spearman": _spearman(xs, ys),
        "mean_absolute_difference": sum(diffs) / len(diffs) if diffs else None,
        "sign_agreement": sum(1 for x, y in zip(xs, ys) if _sign(x) == _sign(y)) / len(xs) if xs else None,
        "top_k_agreement_k16": _topk_agreement(rows, 16),
    }


def _component_ranking(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    out = []
    for comp in sorted({r["component"] for r in rows}):
        vals = [float(r[key]) for r in rows if r["component"] == comp]
        out.append({"component": comp, f"mean_{key}": sum(vals) / len(vals), "n_states": len(vals)})
    return sorted(out, key=lambda r: r[f"mean_{key}"], reverse=True)


def _ranking_agreement(rows: list[dict[str, Any]]) -> dict[str, Any]:
    r2 = _component_ranking(rows, "utility_h1002")
    r4 = _component_ranking(rows, "utility_h1004")
    names2 = [r["component"] for r in r2]
    names4 = [r["component"] for r in r4]
    return {
        "h1002_ranking": r2,
        "h1004_ranking": r4,
        "exact_order_match": names2 == names4,
        "top_component_match": bool(names2 and names4 and names2[0] == names4[0]),
        "spearman_over_component_means": _spearman(
            [next(r[f"mean_utility_h1002"] for r in r2 if r["component"] == c) for c in sorted(set(names2) & set(names4))],
            [next(r[f"mean_utility_h1004"] for r in r4 if r["component"] == c) for c in sorted(set(names2) & set(names4))],
        ) if len(set(names2) & set(names4)) >= 2 else None,
    }


def _decision(agreement: dict[str, Any]) -> str:
    pear = agreement["overall"].get("pearson")
    spear = agreement["overall"].get("spearman")
    sign = agreement["overall"].get("sign_agreement")
    rank = agreement["component_ranking_agreement"]
    if pear is not None and spear is not None and sign is not None and pear >= 0.85 and spear >= 0.85 and sign >= 0.75 and rank.get("top_component_match"):
        return "MACHINE_REPRODUCIBLE"
    return "MACHINE_SENSITIVE"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=REPO / "outputs" / "h100_4_utility_exact_replay")
    args = ap.parse_args()
    out = args.out_dir
    rows: list[dict[str, Any]] = []
    for comp, k in CELLS:
        rows.extend(_load_jsonl(out / "shards" / f"{comp}_K{k}.jsonl"))
    if not rows:
        raise RuntimeError("no exact replay shard rows found")
    per_state = [{
        "state_id": r["state_id"],
        "query_id": r["query_id"],
        "snapshot_hash": r["snapshot_hash"],
        "component": r["component"],
        "K": r["K"],
        "seed": r.get("seed"),
        "utility_h1002": r["utility_h1002"],
        "utility_h1004": r["utility_h1004"],
        "difference": r["difference"],
    } for r in rows]
    _write_csv(out / "EXACT_REPLAY_PER_STATE.csv", per_state)
    noise_rows = []
    for r in _load_jsonl(out / "shards" / "replay_noise.jsonl"):
        noise_rows.append({
            "state_id": r["state_id"],
            "query_id": r["query_id"],
            "snapshot_hash": r["snapshot_hash"],
            "component": r["component"],
            "K": r["K"],
            "seed": r.get("seed"),
            "replay_noise": r["replay_noise"],
            "runner": r.get("runner"),
        })
    if noise_rows:
        _write_csv(out / "REPLAY_NOISE.csv", noise_rows)
    by_cell = {}
    for comp, k in sorted({(r["component"], int(r["K"])) for r in per_state}):
        by_cell[f"{comp}_K{k}"] = _summary([r for r in per_state if r["component"] == comp and int(r["K"]) == k])
    agreement = {
        "overall": _summary(per_state),
        "by_component_K": by_cell,
        "component_ranking_agreement": _ranking_agreement(per_state),
    }
    decision = _decision(agreement)
    agreement["decision"] = decision
    (out / "CROSS_MACHINE_AGREEMENT.json").write_text(json.dumps(agreement, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = [
        "# CROSS_MACHINE_AGREEMENT",
        "",
        f"- decision: `{decision}`",
        f"- n_states: {agreement['overall']['n_states']}",
        f"- Pearson: {agreement['overall']['pearson']}",
        f"- Spearman: {agreement['overall']['spearman']}",
        f"- mean absolute difference: {agreement['overall']['mean_absolute_difference']}",
        f"- sign agreement: {agreement['overall']['sign_agreement']}",
        f"- top-k agreement (k=16): {agreement['overall']['top_k_agreement_k16']}",
        "",
        "## Component ranking agreement",
        "",
        f"- exact order match: {agreement['component_ranking_agreement']['exact_order_match']}",
        f"- top component match: {agreement['component_ranking_agreement']['top_component_match']}",
        f"- component-mean Spearman: {agreement['component_ranking_agreement']['spearman_over_component_means']}",
        "",
        "| H100-2 rank | H100-2 mean | H100-4 rank | H100-4 mean |",
        "|---|---:|---|---:|",
    ]
    r2 = agreement["component_ranking_agreement"]["h1002_ranking"]
    r4 = agreement["component_ranking_agreement"]["h1004_ranking"]
    for i in range(max(len(r2), len(r4))):
        a = r2[i] if i < len(r2) else {"component": "", "mean_utility_h1002": ""}
        b = r4[i] if i < len(r4) else {"component": "", "mean_utility_h1004": ""}
        md.append(f"| `{a['component']}` | {a['mean_utility_h1002']} | `{b['component']}` | {b['mean_utility_h1004']} |")
    (out / "CROSS_MACHINE_AGREEMENT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    decision_payload = {"decision": decision, "agreement": agreement, "utility_probe_selector_status": "allowed" if decision == "MACHINE_REPRODUCIBLE" else "pause_and_investigate"}
    (out / "UTILITY_REPRO_DECISION.json").write_text(json.dumps(decision_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = build_run_manifest(run_id="h1004_utility_exact_replay_20260813", stage="h100_4_utility_exact_replay", command=sys.argv, repo_root=REPO, output_dir=out, input_paths={"per_state": out / "EXACT_REPLAY_PER_STATE.csv"}, extra={"python": "/opt/scape-hf-scorer/bin/python", "training": False, "four_h100_only": True, "decision": decision})
    write_run_manifest(out / "RUN_MANIFEST.json", finalize_run_manifest(manifest, exit_code=0, completed_shards=[f"{c}_K{k}" for c, k in CELLS if (out / "shards" / f"{c}_K{k}.jsonl").exists()] + (["replay_noise"] if noise_rows else []) + ["aggregation"]))
    files = [p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS"]
    write_sha256sums(out, files)
    write_status_live(out / "STATUS_LIVE.md", stage="h100_4_utility_exact_replay", run_id="h1004_utility_exact_replay_20260813", n_expected=8, n_finished=8, errors=[], extra={"decision": decision})
    print(json.dumps({"out_dir": str(out), "decision": decision, "rows": len(per_state)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
