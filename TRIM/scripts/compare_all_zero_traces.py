"""Compare paired bcplus_test_50 all vs zero eval traces."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def summary_row(payload: dict) -> dict:
    settings = payload.get("settings") or []
    return settings[0] if settings else payload


def mean(xs: list[float]) -> float:
    return sum(xs) / max(1, len(xs))


def main(all_dir: str, zero_dir: str) -> int:
    all_root = Path(all_dir)
    zero_root = Path(zero_dir)
    all_sum = load_json(all_root / "FOUR_CELL_OFFICIAL_SUMMARY.json")
    zero_sum = load_json(zero_root / "FOUR_CELL_OFFICIAL_SUMMARY.json")
    all_tr = load_jsonl(all_root / "harness" / "PER_QUERY.jsonl")
    zero_tr = load_jsonl(zero_root / "harness" / "PER_QUERY.jsonl")
    if len(all_tr) != len(zero_tr):
        print(f"FAIL: trace count all={len(all_tr)} zero={len(zero_tr)}")
        return 1
    by_z = {str(t["query_id"]): t for t in zero_tr}
    n_auto = 0
    n_zero_auto = 0
    n_same_tools = 0
    n_same_curated = 0
    diffs = []
    for a in all_tr:
        z = by_z[str(a["query_id"])]
        a_eff = a.get("runtime_effects") or {}
        z_eff = z.get("runtime_effects") or {}
        if int(a_eff.get("auto_populate_first_search") or 0) > 0 or a.get("auto_seed"):
            n_auto += 1
        if int(z_eff.get("auto_populate_first_search") or 0) > 0 or z.get("auto_seed"):
            n_zero_auto += 1
        if list(a.get("tool_names") or []) == list(z.get("tool_names") or []):
            n_same_tools += 1
        if float(a.get("n_curated") or 0) == float(z.get("n_curated") or 0):
            n_same_curated += 1
        diffs.append(
            {
                "query_id": a["query_id"],
                "all_n_curated": a.get("n_curated"),
                "zero_n_curated": z.get("n_curated"),
                "all_auto": a.get("auto_seed") or a_eff.get("auto_populate_first_search"),
                "zero_auto": z.get("auto_seed") or z_eff.get("auto_populate_first_search"),
                "all_tools": a.get("tool_names"),
                "zero_tools": z.get("tool_names"),
            }
        )
    a_row = summary_row(all_sum)
    z_row = summary_row(zero_sum)
    report = {
        "n": len(all_tr),
        "all_n_curated_mean": mean([float(t.get("n_curated") or 0) for t in all_tr]),
        "zero_n_curated_mean": mean([float(t.get("n_curated") or 0) for t in zero_tr]),
        "all_auto_populate_queries": n_auto,
        "zero_auto_populate_queries": n_zero_auto,
        "identical_tool_seq": n_same_tools,
        "identical_n_curated": n_same_curated,
        "all_recall": a_row.get("recall"),
        "zero_recall": z_row.get("recall"),
        "all_legal": a_row.get("legal_action_rate"),
        "zero_legal": a_row.get("legal_action_rate") and z_row.get("legal_action_rate"),
        "zero_legal_action_rate": z_row.get("legal_action_rate"),
        "all_claim": all_sum.get("claim_status"),
        "zero_claim": zero_sum.get("claim_status"),
    }
    failures = []
    if n_auto < max(1, int(0.5 * len(all_tr))):
        failures.append(f"all auto_populate too rare: {n_auto}/{len(all_tr)}")
    if n_zero_auto != 0:
        failures.append(f"zero auto_populate leaked: {n_zero_auto}")
    if n_same_tools == len(all_tr):
        failures.append("all tool sequences identical to zero")
    if abs(report["all_n_curated_mean"] - report["zero_n_curated_mean"]) < 1e-6 and n_same_curated == len(all_tr):
        failures.append("n_curated identical — mask likely still inert")
    report["pass"] = not failures
    report["failures"] = failures
    report["examples"] = diffs[:5]
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
