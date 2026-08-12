#!/usr/bin/env python3
"""Run H100-4 CONFIRM128 real-model influence confirmation.

This wrapper uses the canonical HF continuation scorer implemented by
`scripts/run_h100_3_real_influence_hf.py`, but freezes an independent
REAL_INF_CONFIRM128 split and writes H100-4 handoff artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load_h1003_top(path: Path) -> list[str]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        rows = obj.get("top_candidates") or obj.get("rows") or obj
        if isinstance(rows, list):
            out = []
            for r in rows:
                out.append(str(r.get("component") or r.get("component_id")))
            return [x for x in out if x and x != "None"][:3]
    if isinstance(obj, list):
        out = []
        for r in obj:
            if isinstance(r, dict):
                out.append(str(r.get("component") or r.get("component_id")))
            else:
                out.append(str(r))
        return [x for x in out if x and x != "None"][:3]
    raise ValueError(f"Unsupported top-candidate shape: {path}")


def _load_queries(path: Path) -> list[str]:
    qids = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                qids.append(str(parts[0]))
    return qids


def _stable_rank(seed: int, qid: str) -> str:
    return hashlib.sha256(f"{seed}:{qid}".encode()).hexdigest()


def _freeze_split(*, queries_path: Path, h1003_state_path: Path, out_dir: Path, seed: int, n: int) -> list[str]:
    all_qids = _load_queries(queries_path)
    used = set()
    if h1003_state_path.exists():
        with h1003_state_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    used.add(str(json.loads(line).get("query_id")))
                except Exception:
                    pass
    eligible = [qid for qid in all_qids if qid not in used]
    selected = sorted(eligible, key=lambda q: _stable_rank(seed, q))[:n]
    manifests = out_dir / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / "REAL_INF_CONFIRM128.json").write_text(json.dumps({"name": "REAL_INF_CONFIRM128", "seed": seed, "n": len(selected), "query_ids": selected, "excluded_h1003_qids": len(used)}, indent=2) + "\n", encoding="utf-8")
    (manifests / "CONFIRM_SPLIT_AUDIT.md").write_text("\n".join([
        "# CONFIRM_SPLIT_AUDIT",
        "",
        "- split: `REAL_INF_CONFIRM128`",
        f"- seed: {seed}",
        f"- n: {len(selected)}",
        f"- excluded_h1003_query_ids: {len(used)}",
        f"- eligible_after_exclusion: {len(eligible)}",
    ]) + "\n", encoding="utf-8")
    return selected


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_recommendation(out_dir: Path, h1003_csv: Path, h1004_csv: Path) -> None:
    h3 = {r["component"]: r for r in _read_rows(h1003_csv)} if h1003_csv.exists() else {}
    h4 = {r["component"]: r for r in _read_rows(h1004_csv)}
    components = sorted(h4, key=lambda c: float(h4[c].get("I_name_normalized", 0.0)) + float(h4[c].get("I_args_raw", 0.0)), reverse=True)
    semantic = [c for c in components if c not in {"chunk_neighbors", "content_dedup", "token_budget_marker"}]
    recs = semantic[:2]
    runtime = [c for c in components if c in {"chunk_neighbors", "content_dedup", "token_budget_marker"}][:2]
    payload = {
        "candidates": [{"label": chr(ord("A") + i), "component": c, "h1003": h3.get(c, {}), "h1004": h4.get(c, {})} for i, c in enumerate(recs)],
        "runtime_controls": runtime,
        "note": "Generated from HF continuation scorer CONFIRM output; official Chroma remains separate.",
    }
    (out_dir / "CANDIDATE_RECOMMENDATION_FOR_H20.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = ["# CANDIDATE_RECOMMENDATION_FOR_H20", ""]
    for item in payload["candidates"]:
        md.append(f"- Candidate {item['label']}: `{item['component']}`")
    md.append("")
    md.append("## Runtime Controls")
    for c in runtime:
        md.append(f"- `{c}`")
    (out_dir / "CANDIDATE_RECOMMENDATION_FOR_H20.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def _write_preflight(out_dir: Path, *, model: str, components: list[str], n: int, max_states: int) -> None:
    (out_dir / "PREFLIGHT.md").write_text("\n".join([
        "# PREFLIGHT",
        "",
        f"- model: `{model}`",
        "- scorer: HF continuation logprob scorer from `run_h100_3_real_influence_hf.py`",
        f"- components: {', '.join(components)}",
        f"- split: REAL_INF_CONFIRM128 n={n}",
        f"- max_states_per_query: {max_states}",
        "- Chroma/API credentials: not required for HF scorer confirmation",
    ]) + "\n", encoding="utf-8")


def _sha256sums(root: Path) -> None:
    lines = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        lines.append(f"{h.hexdigest()}  {path.relative_to(root)}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default="/opt/scape-h1003-hf-scorer/bin/python")
    ap.add_argument("--model", default="/mnt/songzijun/models/pat-jj_harness-1-full/harness-1")
    ap.add_argument("--top-candidates", type=Path, default=REPO / "outputs" / "h100_3_influence_qrel" / "TOP_CANDIDATES_FOR_CONFIRM.json")
    ap.add_argument("--h1003-real-csv", type=Path, default=REPO / "outputs" / "h100_3_influence_qrel" / "INFLUENCE_BY_COMPONENT.csv")
    ap.add_argument("--out-dir", type=Path, default=REPO / "outputs" / "h100_4_influence_confirm")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--seed", type=int, default=4404)
    ap.add_argument("--max-states-per-query", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    components = _load_h1003_top(args.top_candidates)
    if not components:
        raise RuntimeError("No H100-3 top candidates found")
    queries_path = Path("/mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus/topics-qrels/queries.tsv")
    split = _freeze_split(queries_path=queries_path, h1003_state_path=REPO / "outputs" / "h100_3_influence_qrel" / "INFLUENCE_PER_STATE.jsonl", out_dir=out, seed=args.seed, n=args.n)
    _write_preflight(out, model=args.model, components=components, n=len(split), max_states=args.max_states_per_query)

    cmd = [
        args.python,
        str(REPO / "scripts" / "run_h100_3_real_influence_hf.py"),
        "--model", args.model,
        "--components", *components,
        "--n-queries", str(len(split)),
        "--max-states-per-query", str(args.max_states_per_query),
        "--device", args.device,
        "--dtype", args.dtype,
        "--out-dir", str(out / "confirm128_hf_scorer"),
    ]
    subprocess.check_call(cmd, cwd=str(REPO))
    result_csv = out / "confirm128_hf_scorer" / "REAL_INFLUENCE_BY_COMPONENT.csv"
    if not result_csv.exists():
        raise RuntimeError(f"Missing result CSV: {result_csv}")

    rows = _read_rows(result_csv)
    out_csv = out / "REAL_INFLUENCE_CONFIRM_BY_COMPONENT.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    (out / "REAL_INFLUENCE_CONFIRM_BY_COMPONENT.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = ["# REAL_INFLUENCE_CONFIRM", "", "| component | n_states | I_name_normalized | I_args_raw | gate |", "|---|---:|---:|---:|---|"]
    for r in rows:
        md.append(f"| {r['component']} | {r['n_states']} | {float(r['I_name_normalized']):.6f} | {float(r['I_args_raw']):.6f} | {r['gate']} |")
    (out / "REAL_INFLUENCE_CONFIRM.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    _write_recommendation(out, args.h1003_real_csv, out_csv)
    _sha256sums(out)
    print(json.dumps({"out_dir": str(out), "components": components, "n": len(split), "max_states_per_query": args.max_states_per_query}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
