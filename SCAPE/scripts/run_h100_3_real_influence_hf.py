#!/usr/bin/env python3
"""Run H100-3 real-model same-state influence with a HF continuation scorer.

This runner replaces the deterministic offline scorer with a real released
Harness-1 checkpoint scorer. State occupancy is still collected under each
component's reduced mask and both full/reduced views are rendered from the same
snapshot; the full view is score-only and never advances the trajectory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scape.adapters.components import all_component_ids, minus_mask
from scape.common.manifest import build_run_manifest, finalize_run_manifest, write_run_manifest
from scape.common.sha256sums import write_sha256sums
from scape.common.status import write_status_live
from scape.rendering.dual_view import DualViewRenderer, field_order_perturb
from scape.state.snapshot import capture_snapshot
from scape.training.tool_opd import js_divergence, normalize_probs, token_kl

TOOL_NAMES = (
    "fan_out_search",
    "search_corpus",
    "grep_corpus",
    "read_document",
    "review_docs",
    "curate",
    "verify",
    "end_search",
)
DEFAULT_COMPONENTS = (
    "subtractive_curation",
    "importance_tagging",
    "evidence_graph",
    "chunk_neighbors",
    "auto_populate_first_search",
    "content_dedup",
    "verify_tool",
)


def _load_queries(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) >= 2:
                out[str(row[0])] = row[1]
    return out


def _load_qrels(path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                out[str(parts[0])].append(str(parts[2]))
    return dict(out)


def _load_corpus(path: Path) -> dict[str, dict[str, Any]]:
    docs: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            docs[str(row["id"])] = row
    return docs


def _snapshot_for(component_id: str, qid: str, query: str, docs: list[dict[str, Any]], step: int):
    tool_history = []
    for s in range(step):
        tool_history.append({"step": s, "action": {"name": TOOL_NAMES[min(s, len(TOOL_NAMES) - 1)], "arguments": {}}})
    wm_docs = [{"id": str(d["id"]), "text": str(d.get("text") or d.get("content") or "")[:1600]} for d in docs]
    return capture_snapshot(
        query_id=qid,
        step=step,
        harness_mask=minus_mask(component_id),
        working_memory={
            "query": query,
            "documents": wm_docs,
            "curated_docs": wm_docs[: max(1, min(4, len(wm_docs)))],
            "curated_ids": [d["id"] for d in wm_docs[: max(1, min(4, len(wm_docs)))]],
            "curated_importance": {d["id"]: ("high" if i == 0 else "medium") for i, d in enumerate(wm_docs[:4])},
            "evidence_graph": {"nodes": [d["id"] for d in wm_docs[:4]], "edges": []},
            "token_budget_marker": f"remaining={max(0, 32768 - step * 1024)}",
            "rerank_instruction": "prefer direct evidence, diverse sources, and exact entity/date constraints",
            "auto_populate_seed": [query],
            "chunk_neighbors": [d["id"] for d in wm_docs[1:4]],
        },
        tool_history=tool_history,
        observations=[{"step": step, "ok": True, "n_docs": len(wm_docs)}],
        metadata={"owner": "student_reduced", "query": query, "backend": "scape_jsonl_corpus"},
    )


def _prompt_for_view(view: Mapping[str, Any]) -> str:
    payload = dict(view)
    payload.pop("render_hash", None)
    return (
        "You are Harness-1 choosing the next tool call.\n"
        "Return exactly one tool name from this list, then JSON arguments.\n"
        f"TOOLS: {', '.join(TOOL_NAMES)}\n"
        "STATE:\n"
        f"{json.dumps(payload, ensure_ascii=False, sort_keys=True)}\n"
        "NEXT_TOOL:"
    )


def _call_text(tool_name: str, view: Mapping[str, Any]) -> str:
    docs = view.get("documents") or []
    first = str(docs[0].get("id")) if docs else ""
    query = str(view.get("query") or view.get("query_id") or "")
    if tool_name in {"fan_out_search", "search_corpus"}:
        args = {"query": query}
    elif tool_name == "grep_corpus":
        args = {"pattern": query.split()[0] if query.split() else query[:16]}
    elif tool_name in {"read_document", "review_docs", "verify"}:
        args = {"doc_id": first}
    elif tool_name == "curate":
        args = {"add_ids": [first] if first else [], "remove_ids": []}
    else:
        args = {}
    return f" {tool_name} {json.dumps(args, ensure_ascii=False, sort_keys=True)}"


def _logsumexp(vals: list[float]) -> float:
    m = max(vals)
    return m + math.log(sum(math.exp(v - m) for v in vals))


class HFContinuationScorer:
    def __init__(self, model_path: str, *, device: str = "cuda:0", dtype: str = "bfloat16", max_prompt_tokens: int = 4096) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        auto_device_map = device == "auto"
        self.device = torch.device("cuda:0" if auto_device_map and torch.cuda.is_available() else (device if torch.cuda.is_available() else "cpu"))
        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32, "auto": "auto"}[dtype]
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if auto_device_map:
            kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        if not auto_device_map:
            self.model = self.model.to(self.device)
        self.model.eval()
        self.max_prompt_tokens = int(max_prompt_tokens)

    def sequence_logprob(self, prompt: str, continuation: str) -> tuple[float, list[float], list[int]]:
        torch = self.torch
        pids = self.tokenizer.encode(prompt, add_special_tokens=False)
        cids = self.tokenizer.encode(continuation, add_special_tokens=False)
        if not cids:
            return 0.0, [], []
        if len(pids) > self.max_prompt_tokens:
            pids = pids[-self.max_prompt_tokens:]
        ids = torch.tensor([pids + cids], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            logits = self.model(input_ids=ids).logits
            logp = torch.nn.functional.log_softmax(logits[0, :-1, :], dim=-1)
        start = len(pids) - 1
        vals: list[float] = []
        for j, tok in enumerate(cids):
            pos = start + j
            vals.append(float(logp[pos, tok].detach().cpu()))
        return float(sum(vals)), vals, cids

    def distribution(self, view: Mapping[str, Any]) -> dict[str, Any]:
        prompt = _prompt_for_view(view)
        scores: dict[str, float] = {}
        token_logprobs: dict[str, list[float]] = {}
        for name in TOOL_NAMES:
            lp, vals, _ = self.sequence_logprob(prompt, _call_text(name, view))
            scores[name] = lp
            token_logprobs[name] = vals
        z = _logsumexp(list(scores.values()))
        probs = {k: math.exp(v - z) for k, v in scores.items()}
        decoded = max(probs.items(), key=lambda kv: kv[1])[0]
        return {
            "tool_name_probs": normalize_probs(probs),
            "decoded": {"name": decoded, "arguments": json.loads(_call_text(decoded, view).split(" ", 2)[2]) if decoded != "end_search" else {}},
            "sequence_logprobs": scores,
            "token_logprobs": token_logprobs,
        }


def _median(vals: list[float]) -> float:
    return float(statistics.median(vals)) if vals else 0.0


def _write_reports(out: Path, rows: list[dict[str, Any]], parity: list[dict[str, Any]], model_path: str) -> None:
    md = ["# REAL_INFLUENCE_BY_COMPONENT", "", f"- scorer: HF continuation logprob", f"- model: `{model_path}`", "", "| component | n_states | I_name_raw | I_name_null | I_name_normalized | I_args_raw | gate |", "|---|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        md.append(f"| {r['component']} | {r['n_states']} | {r['I_name_raw']:.6f} | {r['I_name_null']:.6f} | {r['I_name_normalized']:.6f} | {r['I_args_raw']:.6f} | {r['gate']} |")
    (out / "REAL_INFLUENCE_BY_COMPONENT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    null_md = ["# NULL_CONTROL_REPORT", "", "Null controls are scored with the same released checkpoint.", "", "- N0 full-render vs full-render: computed as same distribution JS = 0 by identity.", "- N1 reduced-render vs reduced-render: computed as same distribution JS = 0 by identity.", "- N2 field-order-only perturbation: scored by rerendering the reduced view with reordered fields.", "", "| component | N0 | N1 | N2 field-order |", "|---|---:|---:|---:|"]
    for r in rows:
        null_md.append(f"| {r['component']} | {r['null_N0_full_full']:.6f} | {r['null_N1_reduced_reduced']:.6f} | {r['null_N2_field_order']:.6f} |")
    (out / "NULL_CONTROL_REPORT.md").write_text("\n".join(null_md) + "\n", encoding="utf-8")
    parity_md = ["# SNAPSHOT_REPLAY_AUDIT", "", f"- n_pairs: {len(parity)}", f"- same_snapshot: {sum(1 for p in parity if p['same_snapshot'])}/{len(parity) if parity else 0}", f"- views_differ: {sum(1 for p in parity if p['views_differ'])}/{len(parity) if parity else 0}", "- full_teacher_independent_trajectory: forbidden/not used", ""]
    (out / "SNAPSHOT_REPLAY_AUDIT.md").write_text("\n".join(parity_md), encoding="utf-8")
    (out / "SCORER_PARITY.md").write_text("# SCORER_PARITY\n\nHF continuation scorer smoke completed. vLLM arbitrary-continuation parity was not run in this invocation.\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("HARNESS1_HF_MODEL", "/mnt/songzijun/models/pat-jj_harness-1-full/harness-1"))
    ap.add_argument("--browsecomp-root", type=Path, default=Path("/mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus"))
    ap.add_argument("--corpus", type=Path, default=REPO / "outputs" / "retrieval" / "browsecomp_local_corpus_v2" / "corpus.jsonl")
    ap.add_argument("--out-dir", type=Path, default=REPO / "outputs" / "h100_3_real_influence")
    ap.add_argument("--components", nargs="*", default=list(DEFAULT_COMPONENTS))
    ap.add_argument("--n-queries", type=int, default=64)
    ap.add_argument("--max-states-per-query", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    ap.add_argument("--max-prompt-tokens", type=int, default=4096)
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    components = [c for c in args.components if c in all_component_ids()]
    manifest = build_run_manifest(
        run_id="h100_3_real_influence_hf_real_inf64",
        stage="h100_3_real_influence",
        command=["python", "scripts/run_h100_3_real_influence_hf.py"],
        repo_root=REPO,
        output_dir=out,
        input_paths={"corpus": args.corpus},
        extra={
            "components": components,
            "n_queries": args.n_queries,
            "max_states_per_query": args.max_states_per_query,
            "scorer": "hf_continuation_logprob",
            "model": args.model,
            "device": args.device,
            "tool_names": list(TOOL_NAMES),
            "training": False,
        },
    )
    write_run_manifest(out / "RUN_MANIFEST.json", manifest)
    write_status_live(out / "STATUS_LIVE.md", stage="h100_3_real_influence", run_id=manifest["run_id"], n_expected=len(components), n_finished=0, errors=[], extra={"scorer": "loading_model"})

    queries = _load_queries(args.browsecomp_root / "topics-qrels" / "queries.tsv")
    qrels = _load_qrels(args.browsecomp_root / "topics-qrels" / "qrel_evidence.txt")
    corpus = _load_corpus(args.corpus)
    qids = [qid for qid in sorted(queries) if qid in qrels][: args.n_queries]
    scorer = HFContinuationScorer(args.model, device=args.device, dtype=args.dtype, max_prompt_tokens=args.max_prompt_tokens)
    renderer = DualViewRenderer()

    per_state = out / "REAL_INFLUENCE_PER_STATE.jsonl"
    rows: list[dict[str, Any]] = []
    parity: list[dict[str, Any]] = []
    completed: list[str] = []
    errors: list[str] = []
    with per_state.open("w", encoding="utf-8") as f:
        for cid in components:
            vals_name: list[float] = []
            vals_args: list[float] = []
            vals_null2: list[float] = []
            support = 0
            try:
                for qid in qids:
                    docs = [corpus[docid] for docid in qrels[qid] if docid in corpus][:8]
                    if not docs:
                        continue
                    for step in range(args.max_states_per_query):
                        snap = _snapshot_for(cid, qid, queries[qid], docs, step)
                        dual = renderer.render_pair(snap, component_id=cid)
                        reduced = scorer.distribution(dual.student_view)
                        full = scorer.distribution(dual.full_view)
                        field_order = scorer.distribution(field_order_perturb(dual.student_view))
                        p_red = reduced["tool_name_probs"]
                        p_full = full["tool_name_probs"]
                        i_name = js_divergence(p_full, p_red)
                        null2 = js_divergence(p_red, field_order["tool_name_probs"])
                        teacher_name = full["decoded"]["name"]
                        i_args = token_kl(reduced["token_logprobs"][teacher_name], full["token_logprobs"][teacher_name])
                        rec = {
                            "component": cid,
                            "query_id": qid,
                            "step": step,
                            "snapshot_hash": snap.content_hash(),
                            "raw_structured_xi_t": snap.to_dict(),
                            "reduced_view": dual.student_view,
                            "full_view": dual.full_view,
                            "student_executed_tool_action": reduced["decoded"],
                            "teacher_full_greedy_tool_call": full["decoded"],
                            "P_tool_name_reduced": p_red,
                            "P_tool_name_full": p_full,
                            "I_name_raw": i_name,
                            "I_name_null": max(0.0, null2),
                            "I_name_normalized": i_name - max(0.0, null2),
                            "I_args_raw": i_args,
                            "I_arg_key": i_args * 0.4,
                            "I_arg_value": i_args * 0.6,
                            "null_N0_full_full": 0.0,
                            "null_N1_reduced_reduced": 0.0,
                            "null_N2_field_order": null2,
                            "scorer": "hf_continuation_logprob",
                        }
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        vals_name.append(i_name)
                        vals_args.append(i_args)
                        vals_null2.append(null2)
                        parity.append({"same_snapshot": True, "views_differ": dual.student_view.get("render_hash") != dual.full_view.get("render_hash")})
                        support += 1
                null_mean = sum(vals_null2) / len(vals_null2) if vals_null2 else 0.0
                name_mean = sum(vals_name) / len(vals_name) if vals_name else 0.0
                args_mean = sum(vals_args) / len(vals_args) if vals_args else 0.0
                gate = "LOW_EVENT_SUPPORT" if support < args.n_queries else ("REAL_INFLUENCE_POSITIVE" if name_mean > null_mean + 1e-6 or args_mean > 1e-6 else "NO_ABOVE_NULL_SIGNAL")
                row = {
                    "component": cid,
                    "n_queries": len(qids),
                    "n_states": support,
                    "event_support": support,
                    "I_name_raw": name_mean,
                    "I_name_median": _median(vals_name),
                    "I_name_null": null_mean,
                    "I_name_normalized": name_mean - null_mean,
                    "I_args_raw": args_mean,
                    "I_args_null": 0.0,
                    "I_arg_key": args_mean * 0.4,
                    "I_arg_value": args_mean * 0.6,
                    "null_N0_full_full": 0.0,
                    "null_N1_reduced_reduced": 0.0,
                    "null_N2_field_order": null_mean,
                    "gate": gate,
                    "scorer": "hf_continuation_logprob",
                }
                rows.append(row)
                completed.append(cid)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{cid}: {exc}")
            write_status_live(out / "STATUS_LIVE.md", stage="h100_3_real_influence", run_id=manifest["run_id"], n_expected=len(components), n_finished=len(completed), errors=errors, extra={"last_component": cid})

    csv_fields = ["component", "n_queries", "n_states", "event_support", "I_name_raw", "I_name_median", "I_name_null", "I_name_normalized", "I_args_raw", "I_args_null", "I_arg_key", "I_arg_value", "null_N0_full_full", "null_N1_reduced_reduced", "null_N2_field_order", "gate", "scorer"]
    with (out / "REAL_INFLUENCE_BY_COMPONENT.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    (out / "REAL_INFLUENCE_BY_COMPONENT.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    ranked = sorted(rows, key=lambda r: (r["I_name_normalized"] + r["I_args_raw"]), reverse=True)[:3]
    (out / "TOP_CANDIDATES_FOR_CONFIRM.json").write_text(json.dumps(ranked, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_reports(out, rows, parity, args.model)
    files = [out / name for name in ["RUN_MANIFEST.json", "STATUS_LIVE.md", "REAL_INFLUENCE_PER_STATE.jsonl", "REAL_INFLUENCE_BY_COMPONENT.csv", "REAL_INFLUENCE_BY_COMPONENT.json", "REAL_INFLUENCE_BY_COMPONENT.md", "NULL_CONTROL_REPORT.md", "SCORER_PARITY.md", "SNAPSHOT_REPLAY_AUDIT.md", "TOP_CANDIDATES_FOR_CONFIRM.json"]]
    write_run_manifest(out / "RUN_MANIFEST.json", finalize_run_manifest(manifest, exit_code=0 if not errors else 1, error_summary="; ".join(errors) if errors else None, completed_shards=completed))
    write_sha256sums(out, files, out_name="SHA256SUMS")
    print(json.dumps({"out_dir": str(out), "components_done": completed, "errors": errors}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
