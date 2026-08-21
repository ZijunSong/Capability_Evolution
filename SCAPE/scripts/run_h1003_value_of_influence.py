#!/usr/bin/env python3
"""H100-3 advantage-conditioned influence breadth probe.

Runs a seven-component K=4 one-step corrective fork map.  This script reuses the
H100-2 live fork/replay contract: same xi_t, S branch executes reduced action,
T branch executes full-view teacher action, both continue with the same reduced
continuation policy.  It is breadth-oriented and does not train weights.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

H1002_SCRIPTS = Path("/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-2/SCAPE/scripts")
H1003_SCRIPTS = REPO / "scripts"
for p in (H1002_SCRIPTS, H1003_SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_h100_2_live_fork_replay as live
from scape.common.hashing import stable_split
from scape.common.manifest import build_run_manifest, finalize_run_manifest, write_run_manifest
from scape.common.sha256sums import write_sha256sums
from scape.common.status import write_status_live

COMPONENTS = (
    "evidence_graph",
    "verify_tool",
    "importance_tagging",
    "subtractive_curation",
    "content_dedup",
    "chunk_neighbors",
    "auto_populate_first_search",
)
RUNTIME_CONTROLS = {"content_dedup", "chunk_neighbors"}
DEFAULT_OUT = REPO / "outputs" / "h100_3_advantage_conditioned_influence"
PRESTAGE = REPO / "outputs" / "scape_prestage_v4" / "H1003_VALUE_OF_INFLUENCE_HANDOFF.json"


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _median(vals: list[float]) -> float:
    return float(statistics.median(vals)) if vals else 0.0


def _stable_u(seed: int, label: str) -> float:
    h = hashlib.sha256(f"{seed}:{label}".encode()).hexdigest()[:16]
    return int(h, 16) / float(16**16 - 1)


def _bootstrap_ci(vals: list[float], *, seed: int, n_boot: int = 500) -> list[float]:
    if not vals:
        return [0.0, 0.0]
    n = len(vals)
    means = []
    for b in range(n_boot):
        sample = [vals[int(_stable_u(seed, f"{b}:{i}") * n) % n] for i in range(n)]
        means.append(_mean(sample))
    means.sort()
    return [means[int(0.025 * (n_boot - 1))], means[int(0.975 * (n_boot - 1))]]


def _spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    def ranks(vals: list[float]) -> list[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        out = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            r = (i + j) / 2.0
            for k in range(i, j + 1):
                out[order[k]] = r
            i = j + 1
        return out
    rx, ry = ranks(xs), ranks(ys)
    mx, my = _mean(rx), _mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    deny = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / max(denx * deny, 1e-12)


def _entropy(probs: dict[str, float]) -> float:
    vals = [float(v) for v in probs.values() if float(v) > 0]
    total = sum(vals)
    if total <= 0:
        return 0.0
    return -sum((v / total) * math.log(max(v / total, 1e-12)) for v in vals)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({k for r in rows for k in r})
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _load_queries(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                out[str(parts[0])] = parts[1]
    return out


def _load_qrels(path: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                out.setdefault(str(parts[0]), set()).add(str(parts[2]))
    return out


def freeze_qids(args: argparse.Namespace, queries: dict[str, str], qrels: dict[str, set[str]], out: Path) -> list[str]:
    eligible = sorted(set(queries) & set(qrels))
    selected, _ = stable_split(eligible, seed=args.seed, n_take=args.n_queries_pool)
    man = {"name": "VAI_K4_7COMP128", "seed": args.seed, "n_query_pool": len(selected), "query_ids": selected, "components": list(COMPONENTS)}
    (out / "manifests").mkdir(parents=True, exist_ok=True)
    (out / "manifests" / "VAI_K4_7COMP128.json").write_text(json.dumps(man, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return selected


def _collect_zero_signal_states(component: str, qids: list[str], queries: dict[str, str], qrels: dict[str, set[str]], searcher: Any, scorer: Any, renderer: Any, n_states: int) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    for qid in qids:
        base = live.LiveState(qid=qid, query=queries[qid], gold=qrels.get(qid, set()), searcher=searcher, component=component, branch_seed=f"zero:{component}:{qid}")
        for _ in range(8):
            a_s, d_s, _ = live.policy_action(base, scorer, renderer, component=component, full=False)
            a_t, d_t, _ = live.policy_action(base, scorer, renderer, component=component, full=True)
            div = live.action_distance(a_s, a_t)
            states.append({
                "component": component,
                "query_id": qid,
                "turn_id": base.step,
                "snapshot": base.snapshot().to_dict(),
                "snapshot_hash": base.snapshot().content_hash(),
                "a_S": a_s,
                "a_T": a_t,
                "P_tool_reduced": d_s["tool_name_probs"],
                "P_tool_full": d_t["tool_name_probs"],
                "divergence": div,
                "divergence_type": "tool-name" if a_s.get("name") != a_t.get("name") else ("args-only" if div >= live.ARG_THRESHOLD else "zero-signal"),
            })
            if len(states) >= n_states:
                return states
            base.execute(a_s)
    return states


def run_component(args: argparse.Namespace) -> int:
    out = args.out_dir
    cdir = out / args.component
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "shards").mkdir(exist_ok=True)
    queries = _load_queries(args.browsecomp_root / "topics-qrels" / "queries.tsv")
    qrels = _load_qrels(args.browsecomp_root / "topics-qrels" / "qrel_evidence.txt")
    qids = freeze_qids(args, queries, qrels, out)
    searcher = live.LuceneSearcher(str(args.index_path))
    scorer = live.HFContinuationScorer(args.model, device=args.device, dtype=args.dtype, max_prompt_tokens=args.max_prompt_tokens)
    renderer = live.DualViewRenderer()
    manifest = build_run_manifest(
        run_id=f"h1003_vai_{args.component}_{args.seed}",
        stage="h100_3_advantage_conditioned_influence_component",
        command=[sys.executable, "scripts/run_h1003_value_of_influence.py", "--component", args.component],
        repo_root=REPO,
        output_dir=cdir,
        input_paths={"queries": args.browsecomp_root / "topics-qrels" / "queries.tsv", "qrels": args.browsecomp_root / "topics-qrels" / "qrel_evidence.txt"},
        extra={"component": args.component, "seed": args.seed, "n_states": args.n_states, "K": 4, "training": False, "LOCAL_COMPAT_ONLY": True},
    )
    write_run_manifest(cdir / "RUN_MANIFEST.json", manifest)
    write_status_live(cdir / "STATUS_LIVE.md", stage="h100_3_advantage_conditioned_influence", run_id=manifest["run_id"], n_expected=args.n_states, n_finished=0, errors=[], extra={"component": args.component})
    states = live.collect_candidate_states(args.component, qids, queries, qrels, searcher, scorer, renderer, args.n_states)
    if len(states) < args.n_states:
        zero_states = _collect_zero_signal_states(args.component, qids, queries, qrels, searcher, scorer, renderer, args.n_states)
        seen = {s["snapshot_hash"] for s in states}
        states.extend(s for s in zero_states if s["snapshot_hash"] not in seen)
        states = states[:args.n_states]
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(states):
        start = live.state_from_snapshot(item["snapshot"], queries[item["query_id"]], qrels[item["query_id"]], searcher, args.component)
        s_final, s_trace = live.run_branch(start, item["a_S"], k=4, scorer=scorer, renderer=renderer, component=args.component, label="S")
        t_final, t_trace = live.run_branch(start, item["a_T"], k=4, scorer=scorer, renderer=renderer, component=args.component, label="T")
        sm, tm = s_final.metrics(), t_final.metrics()
        i_name = float(item.get("divergence", 0.0))
        adv = tm["objective_utility"] - sm["objective_utility"]
        row = {
            "split": "VAI_K4_7COMP128",
            "seed": args.seed,
            "component": args.component,
            "state_id": f"{args.component}_{idx:03d}",
            "query_id": item["query_id"],
            "turn_id": item["turn_id"],
            "tool_type": item.get("divergence_type"),
            "argument_class": "tool_name" if item.get("divergence_type") == "tool-name" else ("zero_signal" if item.get("divergence_type") == "zero-signal" else "arguments"),
            "snapshot_hash": item["snapshot_hash"],
            "a_S": item["a_S"],
            "a_T": item["a_T"],
            "name_disagreement": item["a_S"].get("name") != item["a_T"].get("name"),
            "teacher_entropy": _entropy(item.get("P_tool_full") or {}),
            "I_name": i_name,
            "I_name_normalized": i_name,
            "advantage": adv,
            "A_m_state": adv,
            "VAI_state": i_name * (1 if adv > 0 else (-1 if adv < 0 else 0)),
            "delta_reward": adv,
            "delta_curated_recall": tm["curated_evidence_gain"] - sm["curated_evidence_gain"],
            "delta_trajectory_recall": tm["evidence_coverage"] - sm["evidence_coverage"],
            "delta_evidence_coverage": tm["evidence_coverage"] - sm["evidence_coverage"],
            "delta_state_potential": adv,
            "delta_calls": tm["tool_search_cost"] - sm["tool_search_cost"],
            "branch_S_metrics": sm,
            "branch_T_metrics": tm,
            "branch_S_trace": s_trace,
            "branch_T_trace": t_trace,
            "full_harness_takeover": False,
            "runner": "h1003_value_of_influence_k4",
        }
        rows.append(row)
        if (idx + 1) % 16 == 0:
            write_status_live(cdir / "STATUS_LIVE.md", stage="h100_3_advantage_conditioned_influence", run_id=manifest["run_id"], n_expected=args.n_states, n_finished=idx + 1, errors=[], extra={"component": args.component})
    (cdir / "VALUE_OF_INFLUENCE_PER_STATE.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    vals = [float(r["advantage"]) for r in rows]
    summary = {
        "component": args.component,
        "n_states": len(rows),
        "mean_advantage": _mean(vals),
        "median_advantage": _median(vals),
        "positive_fraction": sum(1 for v in vals if v > 0) / max(1, len(vals)),
        "spearman_I_advantage": _spearman([float(r["I_name_normalized"]) for r in rows], vals),
        "mean_I_name": _mean([float(r["I_name_normalized"]) for r in rows]),
        "mean_VAI": _mean([float(r["VAI_state"]) for r in rows]),
        "ci95": _bootstrap_ci(vals, seed=args.seed + len(args.component)),
        "classification": "RUNTIME_CONTROL" if args.component in RUNTIME_CONTROLS else "PENDING_AGGREGATION",
    }
    (cdir / "COMPONENT_SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_run_manifest(cdir / "RUN_MANIFEST.json", finalize_run_manifest(manifest, exit_code=0, completed_shards=[args.component]))
    write_status_live(cdir / "STATUS_LIVE.md", stage="h100_3_advantage_conditioned_influence", run_id=manifest["run_id"], n_expected=args.n_states, n_finished=len(rows), errors=[], extra={"component": args.component, "phase": "complete"})
    write_sha256sums(cdir, [p for p in cdir.rglob("*") if p.is_file() and p.name != "SHA256SUMS"])
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def finalize(args: argparse.Namespace) -> int:
    out = args.out_dir
    comp_dirs = [out / c for c in COMPONENTS if (out / c / "COMPONENT_SUMMARY.json").exists()]
    if len(comp_dirs) != len(COMPONENTS):
        raise RuntimeError(f"missing component summaries: {sorted(set(COMPONENTS) - {p.name for p in comp_dirs})}")
    per_state: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    quant_rows: list[dict[str, Any]] = []
    for cdir in comp_dirs:
        rows = _load_jsonl(cdir / "VALUE_OF_INFLUENCE_PER_STATE.jsonl")
        per_state.extend(rows)
        s = json.loads((cdir / "COMPONENT_SUMMARY.json").read_text(encoding="utf-8"))
        vals = [float(r["I_name_normalized"]) for r in rows]
        if vals:
            sorted_vals = sorted(vals)
            qs = [sorted_vals[len(vals)//4], sorted_vals[len(vals)//2], sorted_vals[(3*len(vals))//4]]
            buckets = [("Q1", float("-inf"), qs[0]), ("Q2", qs[0], qs[1]), ("Q3", qs[1], qs[2]), ("Q4", qs[2], float("inf"))]
            for label, lo, hi in buckets:
                br = [r for r in rows if (float(r["I_name_normalized"]) >= lo and float(r["I_name_normalized"]) < hi) or (label == "Q4" and float(r["I_name_normalized"]) >= lo)]
                quant_rows.append({"component": cdir.name, "quantile": label, "n": len(br), "mean_I_name": _mean([float(r["I_name_normalized"]) for r in br]), "mean_advantage": _mean([float(r["advantage"]) for r in br])})
        high_i = s["mean_I_name"] > 0.02
        high_a = s["mean_advantage"] > 0.0
        if cdir.name in RUNTIME_CONTROLS:
            cls = "RUNTIME_CONTROL"
        elif high_i and high_a:
            cls = "HIGH_I_HIGH_A"
        elif high_i and not high_a:
            cls = "HIGH_I_LOW_A"
        elif not high_i and high_a:
            cls = "LOW_I_HIGH_A"
        else:
            cls = "LOW_I_LOW_A"
        s["classification"] = cls
        summaries.append(s)
    (out / "VALUE_OF_INFLUENCE_PER_STATE.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in per_state) + "\n", encoding="utf-8")
    _write_csv(out / "VALUE_OF_INFLUENCE_BY_COMPONENT.csv", summaries)
    _write_csv(out / "VALUE_OF_INFLUENCE_BY_QUANTILE.csv", quant_rows)
    corr_lines = ["# INFLUENCE_ADVANTAGE_CORRELATION", "", "| component | n | mean I | mean advantage | Spearman(I, A) | class |", "|---|---:|---:|---:|---:|---|"]
    for s in summaries:
        corr_lines.append(f"| `{s['component']}` | {s['n_states']} | {s['mean_I_name']:.6f} | {s['mean_advantage']:.6f} | {s['spearman_I_advantage']:.6f} | `{s['classification']}` |")
    (out / "INFLUENCE_ADVANTAGE_CORRELATION.md").write_text("\n".join(corr_lines) + "\n", encoding="utf-8")
    fail_lines = ["# PRESTAGE_FAILURE_EXPLANATION", "", "Influence is treated as policy-effect evidence; downstream advantage determines supervision value. This run maps seven components with K=4 corrective fork/replay and no training.", ""]
    for s in summaries:
        fail_lines.append(f"- `{s['component']}`: `{s['classification']}`; mean_I={s['mean_I_name']:.6f}; mean_A={s['mean_advantage']:.6f}")
    (out / "PRESTAGE_FAILURE_EXPLANATION.md").write_text("\n".join(fail_lines) + "\n", encoding="utf-8")
    handoff = {"decision": "VALUE_OF_INFLUENCE_MAP_READY", "split": "VAI_K4_7COMP128", "seed": args.seed, "components": {s["component"]: s for s in summaries}, "official_chroma_parity": False, "LOCAL_COMPAT_ONLY": True, "runner": "h1003_value_of_influence_k4"}
    (out / "H1003_VALUE_OF_INFLUENCE_HANDOFF.json").write_text(json.dumps(handoff, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    PRESTAGE.parent.mkdir(parents=True, exist_ok=True)
    PRESTAGE.write_text(json.dumps(handoff, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = build_run_manifest(
        run_id="h1003_value_of_influence_20260814",
        stage="h100_3_advantage_conditioned_influence",
        command=[sys.executable, "scripts/run_h1003_value_of_influence.py", "--mode", "finalize"],
        repo_root=REPO,
        output_dir=out,
        input_paths={"component_root": out},
        extra={"components": list(COMPONENTS), "seed": args.seed, "training": False, "decision": handoff["decision"]},
    )
    write_run_manifest(out / "RUN_MANIFEST.json", finalize_run_manifest(manifest, exit_code=0, completed_shards=list(COMPONENTS)))
    write_status_live(out / "STATUS_LIVE.md", stage="h100_3_advantage_conditioned_influence", run_id=manifest["run_id"], n_expected=len(COMPONENTS), n_finished=len(COMPONENTS), errors=[], extra={"decision": handoff["decision"], "LOCAL_COMPAT_ONLY": True})
    write_sha256sums(out, [p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS"])
    print(json.dumps({"out_dir": str(out), "components": len(summaries), "decision": handoff["decision"]}, indent=2), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["component", "finalize"], default="component")
    ap.add_argument("--component", choices=list(COMPONENTS))
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--browsecomp-root", type=Path, default=Path("/mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus"))
    ap.add_argument("--index-path", type=Path, default=Path("/mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus/indexes/bm25"))
    ap.add_argument("--model", default=os.environ.get("HARNESS1_HF_MODEL", "/mnt/songzijun/models/harness-1"))
    ap.add_argument("--seed", type=int, default=3324)
    ap.add_argument("--n-states", type=int, default=128)
    ap.add_argument("--n-queries-pool", type=int, default=512)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    ap.add_argument("--max-prompt-tokens", type=int, default=3072)
    args = ap.parse_args()
    os.environ.setdefault("JAVA_HOME", "/usr/lib/jvm/java-21-openjdk-amd64")
    if args.mode == "finalize":
        return finalize(args)
    if not args.component:
        raise SystemExit("--component required")
    return run_component(args)


if __name__ == "__main__":
    raise SystemExit(main())
