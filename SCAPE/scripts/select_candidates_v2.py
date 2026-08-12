#!/usr/bin/env python3
"""Build CANDIDATE_SELECTION_V2 from H100 import JSONs (Gate C/R/I/P).

Does not hand-fix from old provisional A/B.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


RUNTIMEISH = {
    "chunk_neighbors",
    "content_dedup",
    "token_budget_marker",
    "retrieval_executor",
    "exact_accounting",
    "persistent_store",
    "hard_budget_enforcement",
    "cheap_deterministic_runtime_checks",
}


def _load(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _components_from_contribution(obj: Any) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if obj is None:
        return out
    rows = obj.get("components") if isinstance(obj, dict) else obj
    if isinstance(obj, dict) and "rows" in obj and rows is None:
        rows = obj["rows"]
    if not isinstance(rows, list):
        # try dict mapping
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, dict) and ("delta" in v or "full" in v or "contribution" in v):
                    out[str(k)] = v
        return out
    for r in rows:
        if not isinstance(r, dict):
            continue
        cid = r.get("component_id") or r.get("component") or r.get("name")
        if cid:
            out[str(cid)] = r
    return out


def gate_c(row: dict[str, Any]) -> bool:
    # full > minus_m on a core retrieval metric
    for key in ("delta_full_minus", "delta", "delta_ndcg", "delta_recall", "contribution"):
        if key in row:
            try:
                return float(row[key]) > 0
            except (TypeError, ValueError):
                pass
    full = row.get("full")
    minus = row.get("minus") or row.get("minus_m")
    if isinstance(full, (int, float)) and isinstance(minus, (int, float)):
        return float(full) > float(minus)
    return bool(row.get("gate_c") or row.get("pass_contribution"))


def gate_i(row: dict[str, Any]) -> bool:
    for key in ("I_name", "i_name", "influence_name", "I"):
        if key in row:
            try:
                return float(row[key]) > float(row.get("null_control", 0.0) or 0.0)
            except (TypeError, ValueError):
                pass
    return bool(row.get("gate_i") or row.get("pass_influence"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--imports-root", type=Path, default=Path("imports"))
    ap.add_argument("--out", type=Path, default=Path("outputs/scape_prestage_v2"))
    args = ap.parse_args()

    paths = {
        "contribution": args.imports_root / "h100_1" / "CONTRIBUTION_CONFIRM.json",
        "replication": args.imports_root / "h100_2" / "LOO_REPLICATION_V2.json",
        "influence": args.imports_root / "h100_3" / "REAL_INFLUENCE_BY_COMPONENT.json",
        "confirm": args.imports_root / "h100_4" / "CANDIDATE_RECOMMENDATION_FOR_H20.json",
    }
    missing = [k for k, p in paths.items() if not p.is_file()]
    args.out.mkdir(parents=True, exist_ok=True)
    if missing:
        wait = {
            "status": "WAITING_FOR_H100",
            "missing": missing,
            "paths": {k: str(v) for k, v in paths.items()},
            "legacy_scope_path_used": False,
            "note": "Do not hand-fix Candidate A/B from old provisional map.",
        }
        (args.out / "WAITING_FOR_H100.json").write_text(
            json.dumps(wait, indent=2) + "\n", encoding="utf-8"
        )
        (args.out / "WAITING_FOR_H100.md").write_text(
            "# Waiting for H100 V2 imports\n\nMissing: "
            + ", ".join(missing)
            + "\n\nDrop JSON files into `imports/h100_{1..4}/` then re-run.\n",
            encoding="utf-8",
        )
        print(json.dumps(wait, indent=2))
        return 2

    contrib = _components_from_contribution(_load(paths["contribution"]))
    repl = _components_from_contribution(_load(paths["replication"]))
    infl = _components_from_contribution(_load(paths["influence"]))
    rec = _load(paths["confirm"]) or {}

    # Copy raw
    for name, p in paths.items():
        data = _load(p)
        (args.out / p.name).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    scores: list[dict[str, Any]] = []
    all_ids = sorted(set(contrib) | set(repl) | set(infl))
    for cid in all_ids:
        c_row = contrib.get(cid, {})
        r_row = repl.get(cid, {})
        i_row = infl.get(cid, {})
        gc = gate_c(c_row) or gate_c(r_row)
        # Gate R: agree direction if both present, else one-sided ok
        gr = True
        if c_row and r_row:
            try:
                gr = (float(c_row.get("delta", 0)) >= 0 and float(r_row.get("delta", 0)) >= 0) or (
                    gate_c(c_row) and not (float(r_row.get("delta", 0)) < 0)
                )
            except (TypeError, ValueError):
                gr = True
        gi = gate_i(i_row)
        gp = cid not in RUNTIMEISH and not bool(c_row.get("runtime_anchor"))
        ok = gc and gr and gi and gp
        score = 0.0
        for row in (c_row, r_row, i_row):
            for key in ("delta", "I_name", "I", "influence"):
                if key in row:
                    try:
                        score += abs(float(row[key]))
                    except (TypeError, ValueError):
                        pass
        scores.append(
            {
                "component_id": cid,
                "gate_c": gc,
                "gate_r": gr,
                "gate_i": gi,
                "gate_p": gp,
                "pass": ok,
                "score": score,
            }
        )

    passed = [s for s in scores if s["pass"]]
    passed.sort(key=lambda x: -x["score"])
    # Prefer H100-4 recommendation order if present
    rec_list = []
    if isinstance(rec, dict):
        rec_list = rec.get("top") or rec.get("candidates") or rec.get("recommended") or []
    if isinstance(rec_list, list) and rec_list:
        order = []
        for item in rec_list:
            if isinstance(item, str):
                order.append(item)
            elif isinstance(item, dict):
                order.append(str(item.get("component_id") or item.get("name")))
        passed.sort(
            key=lambda x: (order.index(x["component_id"]) if x["component_id"] in order else 999, -x["score"])
        )

    top2 = passed[:2]
    selection = {
        "schema_version": "candidate_selection_v2",
        "legacy_scope_path_used": False,
        "n_evaluated": len(scores),
        "n_passed": len(passed),
        "Candidate_A": top2[0]["component_id"] if len(top2) > 0 else None,
        "Candidate_B": top2[1]["component_id"] if len(top2) > 1 else None,
        "ranked_pass": passed,
        "all": scores,
        "note": "LOCAL_COMPAT_ONLY if retrieval backend is local BM25",
    }
    (args.out / "CANDIDATE_SELECTION_V2.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )
    md = [
        "# CANDIDATE_SELECTION_V2",
        "",
        f"- Candidate A: `{selection['Candidate_A']}`",
        f"- Candidate B: `{selection['Candidate_B']}`",
        f"- passed: {selection['n_passed']} / {selection['n_evaluated']}",
        "- legacy_scope_path_used: false",
        "",
        "## Ranked",
        "",
    ]
    for s in passed:
        md.append(
            f"- `{s['component_id']}` score={s['score']:.4f} "
            f"C/R/I/P={s['gate_c']}/{s['gate_r']}/{s['gate_i']}/{s['gate_p']}"
        )
    (args.out / "CANDIDATE_SELECTION_V2.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(selection, indent=2))
    return 0 if selection["Candidate_A"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
