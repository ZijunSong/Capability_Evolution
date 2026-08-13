#!/usr/bin/env python3
"""H100-4 exact replay for H100-2 UTILITY_COMMON128 states.

This runner does not sample states. It consumes the H100-2 manifest/handoff and
replays the exact Utility Common128 query/component/K cells with the same static
HF continuation-logprob utility calculation used by H100-2.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scape.adapters.components import minus_mask
from scape.common.manifest import build_run_manifest, write_run_manifest
from scape.common.status import write_status_live
from scape.rendering.dual_view import DualViewRenderer, field_order_perturb
from scape.state.snapshot import capture_snapshot
from scape.training.tool_opd import js_divergence, token_kl
from scripts.run_h100_3_real_influence_hf import HFContinuationScorer

TARGET_COMPONENTS = {"subtractive_curation", "importance_tagging", "verify_tool"}
ALIASES = {"SC": "subtractive_curation", "IT": "importance_tagging", "VT": "verify_tool"}


def _load_queries(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) >= 2:
                out[str(row[0])] = row[1]
    return out


def _load_qrels(path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                out.setdefault(str(parts[0]), []).append(str(parts[2]))
    return out


def _load_corpus(path: Path) -> dict[str, dict[str, Any]]:
    docs: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                docs[str(row["id"])] = row
    return docs


def _stable_rank(seed: int, qid: str) -> str:
    return hashlib.sha256(f"{seed}:{qid}".encode()).hexdigest()


def _manifest_payload(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    payload = obj.get("common_manifest_payload") if isinstance(obj, dict) else None
    if isinstance(payload, dict):
        return payload
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"manifest payload must be an object: {path}")


def _source_results_from_handoff(path: Path) -> Path | None:
    if not path.exists():
        return None
    obj = json.loads(path.read_text(encoding="utf-8"))
    val = obj.get("source_results") if isinstance(obj, dict) else None
    return Path(val) if isinstance(val, str) else None


def _load_h1002_results(path: Path | None) -> dict[tuple[str, int, str], dict[str, Any]]:
    out: dict[tuple[str, int, str], dict[str, Any]] = {}
    if path is None or not path.is_file():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            out[(str(row["component"]), int(row["K"]), str(row["query_id"]))] = row
    return out


def _docs_for(qid: str, qrels: dict[str, list[str]], corpus: dict[str, dict[str, Any]], *, k: int) -> list[dict[str, Any]]:
    docs = [corpus[d] for d in qrels.get(qid, []) if d in corpus]
    if len(docs) >= k:
        return docs[:k]
    seen = {str(d.get("id")) for d in docs}
    for docid in sorted(corpus, key=lambda d: _stable_rank(9917, f"{qid}:{d}")):
        if docid not in seen:
            docs.append(corpus[docid])
            if len(docs) >= k:
                break
    return docs


def _snapshot(component_id: str, qid: str, query: str, docs: list[dict[str, Any]], *, k: int) -> Any:
    wm_docs = [{"id": str(d["id"]), "text": str(d.get("text") or d.get("content") or "")[:1600]} for d in docs[:k]]
    return capture_snapshot(
        query_id=qid,
        step=0,
        harness_mask=minus_mask(component_id),
        working_memory={
            "query": query,
            "documents": wm_docs,
            "curated_docs": wm_docs,
            "curated_ids": [d["id"] for d in wm_docs],
            "curated_importance": {d["id"]: ("high" if i == 0 else "medium") for i, d in enumerate(wm_docs)},
            "evidence_graph": {"nodes": [d["id"] for d in wm_docs], "edges": []},
            "token_budget_marker": "remaining=32768",
            "rerank_instruction": "prefer direct evidence, diverse sources, and exact entity/date constraints",
            "auto_populate_seed": [query],
            "candidate_k": k,
        },
        tool_history=[],
        observations=[{"step": 0, "ok": True, "n_docs": len(wm_docs)}],
        metadata={"owner": "student_reduced", "query": query, "backend": "scape_jsonl_corpus", "candidate_k": k},
    )


def _entropy(probs: dict[str, float]) -> float:
    return -sum(float(p) * math.log(max(float(p), 1e-12)) for p in probs.values())


def run_utility(args: argparse.Namespace) -> int:
    component = ALIASES.get(args.component, args.component)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "shards").mkdir(exist_ok=True)

    payload = _manifest_payload(args.manifest)
    qids = [str(x) for x in payload["query_ids"]]
    seed = int(payload.get("seed", args.seed))
    corpus_path = Path(payload.get("retrieval_corpus") or args.corpus)
    if not corpus_path.is_file() and str(corpus_path).startswith("/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-2/"):
        corpus_path = Path(str(corpus_path).replace("/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-2/SCAPE", "/mnt/songzijun/Capability_Evolution/SCAPE"))
    queries = _load_queries(args.browsecomp_root / "topics-qrels" / "queries.tsv")
    qrels = _load_qrels(args.browsecomp_root / "topics-qrels" / "qrel_evidence.txt")
    corpus = _load_corpus(corpus_path)
    h1002 = _load_h1002_results(_source_results_from_handoff(args.handoff))

    run_id = f"h1004_exact_replay_{component}_K{args.K}"
    manifest = build_run_manifest(
        run_id=run_id,
        stage="h100_4_utility_exact_replay_worker",
        command=[sys.executable, "scripts/run_h100_4_exact_replay.py"],
        repo_root=REPO,
        output_dir=out,
        input_paths={"handoff": args.handoff, "manifest": args.manifest, "corpus": corpus_path},
        extra={"component": component, "K": args.K, "split": payload.get("name", "UTILITY_COMMON128"), "seed": seed, "n": len(qids), "device": args.device, "model": args.model, "training": False, "python": sys.executable},
    )
    write_run_manifest(out / "shards" / f"{component}_K{args.K}_RUN_MANIFEST.json", manifest)
    write_status_live(out / "STATUS_LIVE.md", stage="h100_4_utility_exact_replay", run_id=run_id, n_expected=len(qids), n_finished=0, errors=[], extra={"active_shard": f"{component}_K{args.K}", "scorer": "loading_model"})

    scorer = HFContinuationScorer(args.model, device=args.device, dtype=args.dtype, max_prompt_tokens=args.max_prompt_tokens)
    renderer = DualViewRenderer()
    shard_path = out / "shards" / f"{component}_K{args.K}.jsonl"
    with shard_path.open("w", encoding="utf-8") as f:
        for i, qid in enumerate(qids, start=1):
            docs = _docs_for(qid, qrels, corpus, k=args.K)
            snap = _snapshot(component, qid, queries[qid], docs, k=args.K)
            dual = renderer.render_pair(snap, component_id=component)
            reduced = scorer.distribution(dual.student_view)
            full = scorer.distribution(dual.full_view)
            replay_a = scorer.distribution(dual.student_view)
            replay_b = scorer.distribution(dual.student_view)
            field_order = scorer.distribution(field_order_perturb(dual.student_view))
            p_red = reduced["tool_name_probs"]
            p_full = full["tool_name_probs"]
            i_name = js_divergence(p_full, p_red)
            replay_noise = js_divergence(replay_a["tool_name_probs"], replay_b["tool_name_probs"])
            null_field_order = js_divergence(p_red, field_order["tool_name_probs"])
            teacher_name = full["decoded"]["name"]
            i_args = token_kl(reduced["token_logprobs"][teacher_name], full["token_logprobs"][teacher_name])
            useful_unique_docs = len({str(d.get("id")) for d in docs})
            redundancy = max(0.0, 1.0 - useful_unique_docs / max(1, len(docs)))
            coverage = len([d for d in docs if str(d.get("id")) in set(qrels.get(qid, []))]) / max(1, len(qrels.get(qid, [])))
            curated_evidence_gain = coverage / max(1, args.K)
            branch_t = i_name + 0.25 * i_args + 0.001 * curated_evidence_gain
            branch_s = max(null_field_order, replay_noise)
            h2 = h1002.get((component, int(args.K), qid), {})
            utility_h1004 = branch_t - branch_s
            utility_h1002 = float(h2.get("branch_T_minus_S", 0.0) or 0.0)
            rec = {
                "state_id": f"{component}_K{args.K}_{i - 1:03d}",
                "query_id": qid,
                "component": component,
                "K": args.K,
                "seed": seed,
                "snapshot_hash": snap.content_hash(),
                "h1002_snapshot_hash": h2.get("snapshot_hash"),
                "snapshot_hash_match": (h2.get("snapshot_hash") == snap.content_hash()) if h2 else None,
                "a_S": reduced["decoded"],
                "a_T": full["decoded"],
                "h1002_a_S": h2.get("a_S"),
                "h1002_a_T": h2.get("a_T"),
                "I_name_raw": i_name,
                "I_args_raw": i_args,
                "I_name_null_field_order": null_field_order,
                "replay_noise": replay_noise,
                "branch_T": branch_t,
                "branch_S": branch_s,
                "branch_T_minus_S": utility_h1004,
                "utility_h1002": utility_h1002,
                "utility_h1004": utility_h1004,
                "difference": utility_h1004 - utility_h1002,
                "curated_evidence_gain": curated_evidence_gain,
                "useful_unique_docs": useful_unique_docs,
                "redundancy": redundancy,
                "coverage": coverage,
                "tool_search_cost": args.K,
                "teacher_entropy": _entropy(p_full),
                "student_entropy": _entropy(p_red),
                "teacher_tool": teacher_name,
                "student_tool": reduced["decoded"]["name"],
                "h1002_teacher_tool": h2.get("teacher_tool"),
                "h1002_student_tool": h2.get("student_tool"),
                "scorer": "hf_continuation_logprob",
                "model": args.model,
                "device": args.device,
                "python": sys.executable,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if i % 8 == 0:
                write_status_live(out / "STATUS_LIVE.md", stage="h100_4_utility_exact_replay", run_id=run_id, n_expected=len(qids), n_finished=i, errors=[], extra={"active_shard": f"{component}_K{args.K}", "device": args.device})
    write_status_live(out / "STATUS_LIVE.md", stage="h100_4_utility_exact_replay", run_id=run_id, n_expected=len(qids), n_finished=len(qids), errors=[], extra={"finished_shard": f"{component}_K{args.K}", "device": args.device})
    print(json.dumps({"shard": str(shard_path), "component": component, "K": args.K, "n": len(qids)}, indent=2))
    return 0


def run_noise(args: argparse.Namespace) -> int:
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "shards").mkdir(exist_ok=True)
    payload = _manifest_payload(args.manifest)
    qids = [str(x) for x in payload["query_ids"]]
    seed = int(payload.get("seed", args.seed))
    corpus_path = Path(payload.get("retrieval_corpus") or args.corpus)
    if not corpus_path.is_file() and str(corpus_path).startswith("/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-2/"):
        corpus_path = Path(str(corpus_path).replace("/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-2/SCAPE", "/mnt/songzijun/Capability_Evolution/SCAPE"))
    queries = _load_queries(args.browsecomp_root / "topics-qrels" / "queries.tsv")
    qrels = _load_qrels(args.browsecomp_root / "topics-qrels" / "qrel_evidence.txt")
    corpus = _load_corpus(corpus_path)
    scorer = HFContinuationScorer(args.model, device=args.device, dtype=args.dtype, max_prompt_tokens=args.max_prompt_tokens)
    renderer = DualViewRenderer()
    path = out / "shards" / "replay_noise.jsonl"
    with path.open("w", encoding="utf-8") as f:
        n = 0
        for component in payload.get("components", sorted(TARGET_COMPONENTS)):
            component = ALIASES.get(str(component), str(component))
            for k in payload.get("K_values", [2, 4]):
                k = int(k)
                for i, qid in enumerate(qids, start=1):
                    docs = _docs_for(qid, qrels, corpus, k=k)
                    snap = _snapshot(component, qid, queries[qid], docs, k=k)
                    dual = renderer.render_pair(snap, component_id=component)
                    replay_a = scorer.distribution(dual.student_view)
                    replay_b = scorer.distribution(dual.student_view)
                    rec = {
                        "state_id": f"{component}_K{k}_{i - 1:03d}",
                        "query_id": qid,
                        "component": component,
                        "K": k,
                        "seed": seed,
                        "snapshot_hash": snap.content_hash(),
                        "replay_noise": js_divergence(replay_a["tool_name_probs"], replay_b["tool_name_probs"]),
                        "runner": "h1004_exact_replay_same_action_static_scorer",
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    n += 1
                    if n % 16 == 0:
                        write_status_live(out / "STATUS_LIVE.md", stage="h100_4_utility_exact_replay_noise", run_id="h1004_exact_replay_noise", n_expected=len(qids) * 6, n_finished=n, errors=[], extra={"device": args.device})
    print(json.dumps({"path": str(path), "rows": n}, indent=2))
    return 0


def main() -> int:
    bcp = Path("/mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus")
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["utility", "noise"], required=True)
    ap.add_argument("--component", choices=["SC", "IT", "VT", "subtractive_curation", "importance_tagging", "verify_tool"])
    ap.add_argument("--K", type=int, choices=[2, 4])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--handoff", type=Path, default=Path("/mnt/songzijun/Capability_Evolution/SCAPE/outputs/h100_2_utility_stability/H1004_EXACT_REPLAY_HANDOFF.json"))
    ap.add_argument("--manifest", type=Path, default=Path("/mnt/songzijun/Capability_Evolution/SCAPE/outputs/utility_stability/UTILITY_COMMON128_MANIFEST.json"))
    ap.add_argument("--out-dir", type=Path, default=REPO / "outputs" / "h100_4_utility_exact_replay")
    ap.add_argument("--model", default=os.environ.get("HARNESS1_HF_MODEL", "/mnt/songzijun/models/pat-jj_harness-1-full/harness-1"))
    ap.add_argument("--browsecomp-root", type=Path, default=bcp)
    ap.add_argument("--corpus", type=Path, default=Path("/mnt/songzijun/Capability_Evolution/SCAPE/outputs/retrieval/browsecomp_local_corpus_v2/corpus.jsonl"))
    ap.add_argument("--seed", type=int, default=2225)
    ap.add_argument("--dtype", default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    ap.add_argument("--max-prompt-tokens", type=int, default=4096)
    args = ap.parse_args()
    if not args.handoff.is_file() or not args.manifest.is_file():
        raise FileNotFoundError(f"required exact replay files missing: {args.handoff}, {args.manifest}")
    if args.mode == "utility":
        if not args.component or args.K is None:
            raise SystemExit("--component and --K required for utility mode")
        return run_utility(args)
    return run_noise(args)


if __name__ == "__main__":
    raise SystemExit(main())
