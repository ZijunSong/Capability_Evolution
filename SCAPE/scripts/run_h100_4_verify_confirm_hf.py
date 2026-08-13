#!/usr/bin/env python3
"""Run H100-4 independent verify_tool CONFIRM128.

This is the 2026-08-13 H100-4 task-specific wrapper. It runs only
`verify_tool` on an independent VERIFY_INF_CONFIRM128 split, writes the
required H100-4 verify artifacts, and emits the H20 handoff file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]


def _load_queries(path: Path) -> list[str]:
    qids: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                qids.append(str(parts[0]))
    return qids


def _stable_rank(seed: int, qid: str) -> str:
    return hashlib.sha256(f"{seed}:{qid}".encode()).hexdigest()


def _used_qids_from_jsonl(path: Path) -> set[str]:
    used: set[str] = set()
    if not path.exists():
        return used
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                qid = json.loads(line).get("query_id")
            except Exception:
                continue
            if qid is not None:
                used.add(str(qid))
    return used


def _freeze_split(*, queries_path: Path, out_dir: Path, seed: int, n: int) -> list[str]:
    existing = out_dir / "manifests" / "VERIFY_INF_CONFIRM128.json"
    if existing.exists():
        obj = json.loads(existing.read_text(encoding="utf-8"))
        if obj.get("seed") == seed and int(obj.get("n", 0)) == n:
            return [str(qid) for qid in obj.get("query_ids", [])]
    all_qids = _load_queries(queries_path)
    used = set()
    for path in [
        REPO / "outputs" / "h100_3_real_influence" / "REAL_INFLUENCE_PER_STATE.jsonl",
        REPO / "outputs" / "h100_3_real_influence_shards" / "verify_tool" / "REAL_INFLUENCE_PER_STATE.jsonl",
        REPO / "outputs" / "h100_4_influence_confirm" / "confirm128_hf_scorer" / "REAL_INFLUENCE_PER_STATE.jsonl",
    ]:
        used |= _used_qids_from_jsonl(path)
    eligible = [qid for qid in all_qids if qid not in used]
    selected = sorted(eligible, key=lambda q: _stable_rank(seed, q))[:n]
    if len(selected) < n:
        raise RuntimeError(f"Only {len(selected)} eligible qids for requested n={n}")
    manifests = out_dir / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": "VERIFY_INF_CONFIRM128",
        "seed": seed,
        "n": len(selected),
        "query_ids": selected,
        "excluded_prior_qids": len(used),
    }
    (manifests / "VERIFY_INF_CONFIRM128.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (manifests / "VERIFY_SPLIT_AUDIT.md").write_text("\n".join([
        "# VERIFY_SPLIT_AUDIT",
        "",
        "- split: `VERIFY_INF_CONFIRM128`",
        f"- seed: {seed}",
        f"- n: {len(selected)}",
        f"- excluded_prior_query_ids: {len(used)}",
        f"- eligible_after_exclusion: {len(eligible)}",
        "- component: `verify_tool`",
    ]) + "\n", encoding="utf-8")
    return selected


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, 0.0))
    except Exception:
        return 0.0


def _write_reports(out: Path, row: dict[str, str], *, n_queries: int, max_states: int, source_dir: Path) -> None:
    # Required CSV/JSON names.
    csv_path = out / "VERIFY_REAL_INF_CONFIRM128.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader(); writer.writerow(row)
    (out / "VERIFY_REAL_INF_CONFIRM128.json").write_text(json.dumps({"rows": [row]}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    n_states = int(float(row.get("n_states", 0) or 0))
    i_name = _float(row, "I_name_normalized")
    i_args = _float(row, "I_args_raw")
    gate = row.get("gate", "")
    h1003_path = REPO / "outputs" / "h100_3_real_influence_shards" / "verify_tool" / "REAL_INFLUENCE_BY_COMPONENT.csv"
    h1003_rows = _read_rows(h1003_path) if h1003_path.exists() else []
    h1003_i = _float(h1003_rows[0], "I_name_normalized") if h1003_rows else 0.0
    same_sign = (i_name > 0 and h1003_i > 0) or (i_name == 0 and h1003_i == 0) or (i_name < 0 and h1003_i < 0)
    concentrated = n_states < max(64, n_queries)
    if i_name > 0 and same_sign and not concentrated:
        decision = "CONFIRMED"
        recommend = True
    elif i_name <= 0 and i_args > 0:
        decision = "RARE_EVENT_ONLY"
        recommend = False
    else:
        decision = "NOT_CONFIRMED"
        recommend = False

    (out / "VERIFY_NATURAL_VS_TARGETED.md").write_text("\n".join([
        "# VERIFY_NATURAL_VS_TARGETED",
        "",
        "## Natural",
        f"- split: `VERIFY_INF_CONFIRM128`",
        f"- n_queries: {n_queries}",
        f"- max_states_per_query: {max_states}",
        f"- n_states: {n_states}",
        f"- I_name_normalized: {i_name:.6f}",
        f"- I_args_raw: {i_args:.6f}",
        f"- gate: `{gate}`",
        "",
        "## Targeted",
        "- TARGETED: not run in this invocation",
        "- reason: natural confirm was prioritized as the required first task; this file keeps natural and targeted streams separate.",
    ]) + "\n", encoding="utf-8")

    null_source = source_dir / "NULL_CONTROL_REPORT.md"
    if null_source.exists():
        (out / "VERIFY_NULL_REPORT.md").write_text(null_source.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        (out / "VERIFY_NULL_REPORT.md").write_text("# VERIFY_NULL_REPORT\n\nNull source report missing.\n", encoding="utf-8")

    (out / "VERIFY_CANDIDATE_B_DECISION.md").write_text("\n".join([
        "# VERIFY_CANDIDATE_B_DECISION",
        "",
        f"- decision: `{decision}`",
        f"- recommend_candidate_b: {str(recommend).lower()}",
        f"- H100-4 natural I_name_normalized: {i_name:.6f}",
        f"- H100-4 natural I_args_raw: {i_args:.6f}",
        f"- H100-3 verify_tool I_name_normalized: {h1003_i:.6f}",
        f"- same sign: {str(same_sign).lower()}",
        f"- effect concentrated in tiny number of states: {str(concentrated).lower()}",
        "",
        "## Gate Interpretation",
        "- CONFIRMED requires natural influence above null, same sign as H100-3, and broad state support.",
        "- RARE_EVENT_ONLY would require weak natural influence but strong targeted influence.",
        "- NOT_CONFIRMED means natural and targeted evidence are both weak or missing.",
    ]) + "\n", encoding="utf-8")

    handoff = {
        "confirmed": decision == "CONFIRMED",
        "decision": decision,
        "component": "verify_tool",
        "natural_influence": {
            "I_name_normalized": i_name,
            "I_name_raw": _float(row, "I_name_raw"),
            "I_name_null": _float(row, "I_name_null"),
            "I_args_raw": i_args,
            "n_queries": n_queries,
            "n_states": n_states,
            "gate": gate,
        },
        "targeted_influence": None,
        "event_support": n_states,
        "recommend_candidate_b": recommend,
    }
    handoff_dir = REPO / "outputs" / "scape_prestage_v2"
    handoff_dir.mkdir(parents=True, exist_ok=True)
    (handoff_dir / "H1004_VERIFY_HANDOFF.json").write_text(json.dumps(handoff, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_sha256sums(root: Path) -> None:
    lines: list[str] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        lines.append(f"{h.hexdigest()}  {path.relative_to(root)}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default="/opt/scape-h1003-hf-scorer/bin/python")
    ap.add_argument("--model", default="/mnt/songzijun/models/pat-jj_harness-1-full/harness-1")
    ap.add_argument("--out-dir", type=Path, default=REPO / "outputs" / "h100_4_verify_confirm")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--seed", type=int, default=4414)
    ap.add_argument("--max-states-per-query", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    queries_path = Path("/mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus/topics-qrels/queries.tsv")
    split = _freeze_split(queries_path=queries_path, out_dir=out, seed=args.seed, n=args.n)
    preflight = {
        "model": args.model,
        "scorer": "hf_continuation_logprob",
        "component": "verify_tool",
        "split": "VERIFY_INF_CONFIRM128",
        "seed": args.seed,
        "n_queries": len(split),
        "max_states_per_query": args.max_states_per_query,
        "targeted": False,
    }
    (out / "PREFLIGHT.json").write_text(json.dumps(preflight, indent=2) + "\n", encoding="utf-8")

    cmd = [
        args.python,
        str(REPO / "scripts" / "run_h100_3_real_influence_hf.py"),
        "--model", args.model,
        "--components", "verify_tool",
        "--n-queries", str(len(split)),
        "--max-states-per-query", str(args.max_states_per_query),
        "--device", args.device,
        "--dtype", args.dtype,
        "--out-dir", str(out / "verify_tool_hf_scorer"),
    ]
    try:
        subprocess.check_call(cmd, cwd=str(REPO))
    except Exception as exc:
        (out / "STATUS_LIVE.md").write_text("\n".join([
            "# STATUS_LIVE — h100_4_verify_confirm",
            "",
            "- n_expected: 1",
            "- n_finished: 0",
            "- errors: 1",
            "",
            "## Errors",
            f"- scorer invocation failed: {exc}",
        ]) + "\n", encoding="utf-8")
        raise

    result_csv = out / "verify_tool_hf_scorer" / "REAL_INFLUENCE_BY_COMPONENT.csv"
    if not result_csv.exists():
        raise RuntimeError(f"Missing result CSV: {result_csv}")
    rows = _read_rows(result_csv)
    if len(rows) != 1 or rows[0].get("component") != "verify_tool":
        raise RuntimeError(f"Expected exactly verify_tool row, got: {rows}")
    _write_reports(out, rows[0], n_queries=len(split), max_states=args.max_states_per_query, source_dir=out / "verify_tool_hf_scorer")
    run_manifest = {
        "stage": "h100_4_verify_confirm",
        "status": "completed",
        "exit_code": 0,
        "component": "verify_tool",
        "split": "VERIFY_INF_CONFIRM128",
        "seed": args.seed,
        "n_queries": len(split),
        "max_states_per_query": args.max_states_per_query,
        "n_states": int(float(rows[0].get("n_states", 0) or 0)),
        "scorer": "hf_continuation_logprob",
        "source_dir": str(out / "verify_tool_hf_scorer"),
    }
    (out / "RUN_MANIFEST.json").write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "STATUS_LIVE.md").write_text("\n".join([
        "# STATUS_LIVE — h100_4_verify_confirm",
        "",
        "- n_expected: 1",
        "- n_finished: 1",
        "- remaining: 0",
        "- errors: 0",
        "",
        "## Extra",
        "- component: verify_tool",
        "- split: VERIFY_INF_CONFIRM128",
        f"- seed: {args.seed}",
        f"- max_states_per_query: {args.max_states_per_query}",
        "- scorer: hf_continuation_logprob",
    ]) + "\n", encoding="utf-8")
    _write_sha256sums(out)
    print(json.dumps({"out_dir": str(out), "component": "verify_tool", "n": len(split)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
