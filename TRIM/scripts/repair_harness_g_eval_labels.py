#!/usr/bin/env python3
"""Attach frozen gold/evidence labels to an existing Harness-G result dir.

This repairs the export envelope only. It does not rewrite trajectories or
promote a run with execution exceptions into a healthy official eval.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

_TRIM = Path(__file__).resolve().parents[1]
if str(_TRIM) not in sys.path:
    sys.path.insert(0, str(_TRIM))

from trim.eval.harness_g_contract import official_metric_counts


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _labels_from_queries(run: Path) -> dict[str, dict]:
    labels: dict[str, dict] = {}
    for path in sorted(run.glob("shards/*/queries.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload:
            qid = str(row["query_id"])
            if qid in labels:
                raise SystemExit(f"duplicate planned query {qid} in {path}")
            labels[qid] = row
    return labels


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Repair missing gold labels on Harness-G results")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    src = args.run_dir
    dest = args.out_dir
    if dest.exists():
        raise SystemExit(f"refusing to overwrite {dest}")
    shutil.copytree(src, dest, dirs_exist_ok=False)
    labels = _labels_from_queries(src)
    if not labels:
        raise SystemExit(f"no shards/*/queries.json under {src}")
    pq = dest / "harness" / "PER_QUERY.jsonl"
    if not pq.is_file():
        pq = dest / "PER_QUERY.jsonl"
    if not pq.is_file():
        raise SystemExit(f"PER_QUERY.jsonl missing in {dest}")
    rows = _load_jsonl(pq)
    repaired: list[dict] = []
    for row in rows:
        qid = str(row["query_id"])
        if qid not in labels:
            raise SystemExit(f"query_id {qid} missing from frozen queries.json")
        lab = labels[qid]
        gold = list(lab.get("gold_docids") or [])
        evidence = list(lab.get("evidence_docids") or [])
        if not gold:
            raise SystemExit(f"empty gold_docids for {qid}")
        row = dict(row)
        row["gold_docids"] = gold
        row["evidence_docids"] = evidence
        row["official_split"] = lab.get("official_split") or row.get("official_split")
        row["label_repair_source"] = "shards/*/queries.json"
        expected = official_metric_counts(
            selected=row.get("selected_docids") or [],
            observed=row.get("observed_docids") or [],
            gold=gold,
            evidence=evidence,
        )
        row["recall"] = expected["recall"]
        row["trajectory_recall"] = expected["trajectory_recall"]
        repaired.append(row)
    pq.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in repaired), encoding="utf-8")
    note = dest / "LABEL_REPAIR.json"
    note.write_text(
        json.dumps(
            {
                "source_run": str(src),
                "n_rows": len(repaired),
                "label_source": "shards/*/queries.json",
                "note": "Export/report repair only. Original trajectories unchanged; not a healthy eval upgrade.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"ok": True, "out": str(dest), "n_rows": len(repaired)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
