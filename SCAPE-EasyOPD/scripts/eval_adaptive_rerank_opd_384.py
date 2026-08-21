#!/usr/bin/env python3
"""Evaluate adaptive_rerank_instruction OPD conditions on the strict 384-query pool."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "/mnt/songzijun/models/Qwen3-30B-A3B-Instruct-2507"
BCP_ROOT = Path("/mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus")
TRAIN_POOL = Path("/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD/manifests/COMPONENT_SWEEP_TRAIN_POOL.json")
SPLIT_FILE = Path("/mnt/songzijun/Capability_Evolution/SCOPE/datagen/splits/browsecompplus_splits.json")
ADAPTER_ROOT = Path("/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD/outputs/component_sweep_0818/h100_2/formal_hf_adaptive_8gpu")
TOOLS = ("fan_out_search", "search_corpus", "grep_corpus", "read_document", "review_docs", "curate", "verify", "end_search")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_queries(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            qid, query = line.split("\t", 1)
            out[str(qid)] = query
    return out


def read_qrels(path: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) >= 4 and float(p[3]) > 0:
            out.setdefault(str(p[0]), set()).add(str(p[2]))
    return out


def training_ids(path: Path) -> set[str]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    rows = obj.get("queries", obj) if isinstance(obj, (dict, list)) else []
    return {str(r.get("query_id", r.get("id"))) for r in rows if isinstance(r, dict)}


def official_test_ids(path: Path) -> set[str]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    return {str(x) for x in obj.get("test_query_ids", [])}


def build_pool(out: Path) -> list[dict[str, Any]]:
    qpath = BCP_ROOT / "topics-qrels" / "queries.tsv"
    epath = BCP_ROOT / "topics-qrels" / "qrel_evidence.txt"
    gpath = BCP_ROOT / "topics-qrels" / "qrel_golds.txt"
    queries, evidence, golds = read_queries(qpath), read_qrels(epath), read_qrels(gpath)
    test_ids = official_test_ids(SPLIT_FILE)
    excluded = training_ids(TRAIN_POOL)
    eligible = set(queries) & set(evidence) & set(golds)
    ids = sorted(eligible - excluded, key=lambda x: int(x) if x.isdigit() else x)
    if len(ids) != 384 or set(ids) & excluded:
        raise RuntimeError(f"strict disjoint pool expected 384, got {len(ids)}")
    rows = []
    for qid in ids:
        rows.append({
            "query_id": qid,
            "query": queries[qid],
            "evidence_docids": sorted(evidence[qid]),
            "gold_docids": sorted(golds[qid]),
            "official_split": "test" if qid in test_ids else "train",
        })
    manifest = {
        "status": "FROZEN_VALID",
        "component": "adaptive_rerank_instruction",
        "pool_contract": "all official BrowseComp-Plus queries present in both qrels minus component training query IDs",
        "query_count": len(rows),
        "test_query_count": sum(r["official_split"] == "test" for r in rows),
        "training_overlap_query_ids": sorted(set(ids) & excluded),
        "queries": rows,
        "input_sha256": {str(p): sha256(p) for p in (qpath, epath, gpath, TRAIN_POOL, SPLIT_FILE)},
        "normalization": "split_at_first_underscore_v1",
        "test_split_source": str(SPLIT_FILE),
    }
    (out / "384_QUERY_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return rows


def remap_state(raw: dict[str, Any]) -> dict[str, Any]:
    return {k.replace(".lora_A.weight", ".lora_A.default.weight").replace(".lora_B.weight", ".lora_B.default.weight"): v for k, v in raw.items()}


def load_backend(adapter_dir: Path | None):
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, trust_remote_code=True, local_files_only=True, torch_dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
    reload_path = "base_no_adapter"
    if adapter_dir:
        cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
        lc = LoraConfig(task_type=cfg.get("task_type", "CAUSAL_LM"), r=int(cfg.get("r", 8)), lora_alpha=int(cfg.get("lora_alpha", 16)), lora_dropout=float(cfg.get("lora_dropout", 0.05)), target_modules=list(cfg.get("target_modules") or ["q_proj", "k_proj", "v_proj", "o_proj"]), bias=cfg.get("bias", "none"))
        try:
            model = PeftModel.from_pretrained(model, adapter_dir)
            reload_path = "peft_model_from_pretrained"
        except Exception:
            model = get_peft_model(model, lc)
            missing, unexpected = model.load_state_dict(remap_state(load_file(str(adapter_dir / "adapter_model.safetensors"))), strict=False)
            if [x for x in missing if "lora_" in x] or [x for x in unexpected if "lora_" in x]:
                raise RuntimeError("manual adapter reload mismatch")
            reload_path = "manual_safetensors_state_dict"
    model.eval()
    return tok, model, reload_path


def make_prompt(row: dict[str, Any], teacher: bool) -> str:
    state = {
        "component": "adaptive_rerank_instruction",
        "query_id": row["query_id"],
        "query": row["query"],
        "student_inference_privilege": False,
        "working_memory": {"curated_ids": [], "visible_doc_ids": [], "search_history": []},
        "adaptive_rerank_instruction": "enabled" if teacher else "disabled",
    }
    schemas = {
        "fan_out_search": {"required": ["queries"], "queries": "nonempty array of 2-3 strings; each query must be 5-12 words"},
        "search_corpus": {"required": ["query"], "query": "nonempty string"},
        "grep_corpus": {"required": ["pattern"], "pattern": "nonempty regex string"},
        "read_document": {"required": ["doc_id"], "doc_id": "visible document ID"},
        "review_docs": {"required": ["doc_ids"], "doc_ids": "nonempty array of visible document IDs"},
        "curate": {"required": ["add_ids", "remove_ids"], "add_ids": "array", "remove_ids": "array"},
        "end_search": {"required": ["reasoning"], "reasoning": "string"},
    }
    if teacher:
        state["adaptive_rerank_instruction_text"] = (
            "Focus on specific entities, dates, quantities, and direct multi-constraint evidence."
        )
    return json.dumps({
        "task": "Begin retrieval. Return exactly one legal Harness-1 tool call as a JSON object whose first key is tool. Output JSON immediately with no analysis or prose. Example: {\"tool\":\"fan_out_search\",\"queries\":[\"short query one\",\"short query two\"]}",
        "state": state,
        "tool_schemas": schemas,
    }, ensure_ascii=False, sort_keys=True)


def generate(tok, model, text: str) -> str:
    msgs = [{"role": "system", "content": "You are a SCAPE research agent. Return exactly one legal Harness-1 tool call."}, {"role": "user", "content": text}]
    rendered = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    if isinstance(rendered, dict) or hasattr(rendered, "input_ids"):
        ids = rendered["input_ids"] if isinstance(rendered, dict) else rendered.input_ids
    else:
        ids = rendered
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    ids = ids.to(model.device)
    with torch.inference_mode():
        out = model.generate(input_ids=ids, max_new_tokens=256, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[-1]:], skip_special_tokens=False)


def parse_action(text: str) -> dict[str, Any]:
    name = next((t for t in TOOLS if re.search(rf"\b{re.escape(t)}\b", text or "")), None)
    params: dict[str, Any] = {}
    matches = re.findall(r"\{.*?\}", text or "", flags=re.S)
    if matches:
        try:
            obj = json.loads(matches[-1])
            if isinstance(obj, dict):
                params = obj
        except json.JSONDecodeError:
            pass
    if isinstance(params.get("tool"), str):
        aliases = {"search": "search_corpus", "fanout_search": "fan_out_search", "finish": "end_search"}
        name = aliases.get(params["tool"], params["tool"])
        params.pop("tool", None)
    if name == "search_corpus" and "query" not in params and isinstance(params.get("q"), str):
        params["query"] = params["q"]
    executable = False
    if name == "search_corpus":
        executable = isinstance(params.get("query"), str) and bool(params["query"].strip())
    elif name == "fan_out_search":
        queries = params.get("queries")
        executable = isinstance(queries, list) and any(isinstance(q, str) and q.strip() for q in queries)
    elif name == "grep_corpus":
        executable = isinstance(params.get("pattern"), str) and bool(params["pattern"].strip())
    elif name in {"read_document", "review_docs", "curate"}:
        executable = bool(params.get("doc_id") or params.get("doc_ids") or params.get("add_ids") or params.get("remove_ids"))
    elif name in {"verify", "end_search"}:
        executable = True
    return {"tool_name": name, "params": params, "legal": name in TOOLS, "executable": executable}


def retrieval_query(action: dict[str, Any], row: dict[str, Any]) -> str | None:
    name, params = action["tool_name"], action["params"]
    if name == "search_corpus":
        return str(params.get("query"))
    if name == "fan_out_search":
        queries = params.get("queries")
        return str(queries[0]) if isinstance(queries, list) and queries else None
    if name == "grep_corpus":
        return str(params.get("pattern")) if isinstance(params.get("pattern"), str) else None
    return None


class _LocalHit:
    def __init__(self, docid: str, raw: str, score: float) -> None:
        self.docid, self.raw, self.score = docid, raw, score


class LocalCorpusSearcher:
    """Deterministic fallback used when the optional Java Lucene stack is unavailable."""
    def __init__(self, corpus_path: Path) -> None:
        self.docs = []
        with corpus_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    docid = str(row.get("id") or row.get("docid") or row.get("source"))
                    text = str(row.get("text") or row.get("contents") or row.get("content") or "")
                    if docid and text:
                        self.docs.append((docid, text))

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {x for x in text.lower().replace("_", " ").split() if len(x) > 2}

    def search(self, query: str, k: int = 100) -> list[_LocalHit]:
        q = self._tokens(query)
        scored = []
        for docid, text in self.docs:
            overlap = len(q & self._tokens(text))
            phrase = 1.0 if query.lower() in text.lower() else 0.0
            if overlap + phrase > 0:
                scored.append((overlap + phrase, docid, text))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [_LocalHit(d, json.dumps({"id": d, "contents": t}, ensure_ascii=False), s) for s, d, t in scored[:k]]


def build_searcher() -> tuple[Any, str]:
    index = BCP_ROOT / "indexes" / "bm25"
    corpus = Path("/mnt/songzijun/Capability_Evolution/SCAPE/outputs/retrieval/browsecomp_local_corpus_v2/corpus.jsonl")
    try:
        from pyserini.search.lucene import LuceneSearcher
        return LuceneSearcher(str(index)), "pyserini_lucene"
    except Exception:
        if not corpus.exists():
            raise RuntimeError(f"Lucene unavailable and fallback corpus missing: {corpus}")
        return LocalCorpusSearcher(corpus), "local_corpus_token_overlap"


def norm_doc(doc: str) -> str:
    return str(doc).split("_", 1)[0]


def evaluate(name: str, rows: list[dict[str, Any]], adapter: Path | None, teacher: bool, searcher: Any, out: Path) -> dict[str, Any]:
    tok, model, reload_path = load_backend(adapter)
    records = []
    for i, row in enumerate(rows, 1):
        generated = generate(tok, model, make_prompt(row, teacher))
        action = parse_action(generated)
        query = retrieval_query(action, row)
        retrieved = [str(h.docid) for h in searcher.search(query, 100)] if query and action["legal"] and action["executable"] else []
        activated = retrieved[:10]
        evidence = {norm_doc(x) for x in row["evidence_docids"]}
        gold = {norm_doc(x) for x in row["gold_docids"]}
        retrieved_norm = {norm_doc(x) for x in retrieved}
        activated_norm = {norm_doc(x) for x in activated}
        records.append({
            "query_id": row["query_id"], "official_split": row["official_split"], "setting": name,
            "generated_text": generated, **action, "retrieval_query": query,
            "retrieved_docids": retrieved, "top_retrieved_docids": activated,
            "evidence_recall_at_10": len(activated_norm & evidence) / max(1, len(evidence)),
            "evidence_recall_at_100": len(retrieved_norm & evidence) / max(1, len(evidence)),
            "gold_recall_at_10": len(activated_norm & gold) / max(1, len(gold)),
            "gold_recall_at_100": len(retrieved_norm & gold) / max(1, len(gold)),
        })
        if i % 32 == 0:
            print(json.dumps({"setting": name, "completed": i, "n": len(rows)}), flush=True)
    root = out / name
    root.mkdir(parents=True, exist_ok=True)
    with (root / "PER_QUERY.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def summary(subset: list[dict[str, Any]], split: str) -> dict[str, Any]:
        n = len(subset)
        return {"split": split, "n_queries": n, "legal_action_rate": sum(bool(r["legal"]) for r in subset) / max(1, n), "executable_action_rate": sum(bool(r["executable"]) for r in subset) / max(1, n), "evidence_recall_at_10": sum(r["evidence_recall_at_10"] for r in subset) / max(1, n), "evidence_recall_at_100": sum(r["evidence_recall_at_100"] for r in subset) / max(1, n), "gold_recall_at_10": sum(r["gold_recall_at_10"] for r in subset) / max(1, n), "gold_recall_at_100": sum(r["gold_recall_at_100"] for r in subset) / max(1, n)}

    result = {"setting": name, "adapter_reload_path": reload_path, "all_pool": summary(records, "all_pool"), "official_test": summary([r for r in records if r["official_split"] == "test"], "official_test")}
    (root / "SUMMARY.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--setting", choices=("TEACHER", "STUDENT_BEFORE_OPD", "STUDENT_AFTER_PURE_OPD", "STUDENT_AFTER_RL_PLUS_OPD"))
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_pool(args.output_dir)
    if args.limit:
        rows = rows[:args.limit]
    searcher, retrieval_backend = build_searcher()
    settings = [
        ("TEACHER", None, True),
        ("STUDENT_BEFORE_OPD", None, False),
        ("STUDENT_AFTER_PURE_OPD", ADAPTER_ROOT / "PURE_OPD_seed42" / "lora_checkpoint", False),
        ("STUDENT_AFTER_RL_PLUS_OPD", ADAPTER_ROOT / "RL_PLUS_OPD_seed42" / "lora_checkpoint", False),
    ]
    if args.setting:
        settings = [x for x in settings if x[0] == args.setting]
    summaries = [evaluate(name, rows, adapter, teacher, searcher, args.output_dir) for name, adapter, teacher in settings]
    payload = {"status": "ADAPTIVE_RERANK_INSTRUCTION_OPD_384_GENERATION_READY", "component": "adaptive_rerank_instruction", "query_count": len(rows), "test_query_count": sum(r["official_split"] == "test" for r in rows), "settings": summaries, "base_model": BASE_MODEL, "retrieval_backend": retrieval_backend, "retrieval": "generation-stage diagnostic only; formal scorer uses official Lucene Recall@5", "student_inference_privilege": False}
    (args.output_dir / "SUMMARY.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    files = [p for p in args.output_dir.rglob("*") if p.is_file() and p.name != "SHA256SUMS"]
    (args.output_dir / "SHA256SUMS").write_text("\n".join(f"{sha256(p)}  {p.relative_to(args.output_dir)}" for p in sorted(files)) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
