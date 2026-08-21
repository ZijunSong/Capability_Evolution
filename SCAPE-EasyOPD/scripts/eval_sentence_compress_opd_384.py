#!/usr/bin/env python3
"""Formal sentence_compress OPD evaluation on the strict 384-query pool."""
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
TOOLS = ("fan_out_search", "search_corpus", "grep_corpus", "read_document", "review_docs", "curate", "end_search")
ADAPTERS = {
    "PURE_OPD_seed42": "/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD/outputs/component_sweep_0818/h100_3_qwen3_faststart/global_step_1/actor/lora_adapter",
    "PURE_OPD_seed43": "/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD/outputs/component_sweep_0818/h100_3_qwen3_faststart/global_step_2/actor/lora_adapter",
    "RL_PLUS_OPD_seed42": "/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD/outputs/component_sweep_0818/h100_3_qwen3_faststart/global_step_3/actor/lora_adapter",
    "RL_PLUS_OPD_seed43": "/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD/outputs/component_sweep_0818/h100_3_qwen3_faststart/global_step_4/actor/lora_adapter",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_queries(path: Path) -> dict[str, str]:
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                qid, query = line.rstrip("\n").split("\t", 1)
                out[str(qid)] = query
    return out


def read_qrels(path: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 4 and float(p[3]) > 0:
                out.setdefault(str(p[0]), set()).add(str(p[2]))
    return out


def training_ids(path: Path) -> set[str]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    rows = obj.get("queries", obj) if isinstance(obj, (dict, list)) else []
    return {str(r.get("query_id", r.get("id"))) for r in rows if isinstance(r, dict)}


def build_pool(out: Path) -> list[dict[str, Any]]:
    qpath = BCP_ROOT / "topics-qrels" / "queries.tsv"
    epath = BCP_ROOT / "topics-qrels" / "qrel_evidence.txt"
    gpath = BCP_ROOT / "topics-qrels" / "qrel_golds.txt"
    queries, evidence, golds = read_queries(qpath), read_qrels(epath), read_qrels(gpath)
    eligible = set(queries) & set(evidence) & set(golds)
    excluded = training_ids(TRAIN_POOL)
    ids = sorted(eligible - excluded, key=lambda x: int(x) if x.isdigit() else x)
    if len(ids) != 384:
        raise RuntimeError(f"strict disjoint pool expected 384, got {len(ids)}")
    rows = [{"query_id": q, "query": queries[q], "evidence_docids": sorted(evidence[q]), "gold_docids": sorted(golds[q])} for q in ids]
    manifest = {
        "status": "FROZEN_VALID",
        "pool_contract": "all official BrowseComp-Plus queries present in both qrels minus component training query IDs",
        "query_count": len(rows),
        "excluded_training_query_ids": len(excluded),
        "training_overlap_query_ids": sorted(set(ids) & excluded),
        "queries": rows,
        "input_sha256": {str(p): sha256(p) for p in (qpath, epath, gpath, TRAIN_POOL)},
        "normalization": "split_at_first_underscore_v1",
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
        try:
            cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # Older verl export leaves target_parameters without a JSON value.
            cfg = {"task_type": "CAUSAL_LM", "r": 8, "lora_alpha": 16, "lora_dropout": 0.0, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"], "bias": "none"}
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


def prompt(row: dict[str, Any], teacher: bool) -> str:
    state = {"component": "sentence_compress", "query_id": row["query_id"], "query": row["query"], "student_inference_privilege": False, "visible_documents": [], "curated_ids": [], "search_history": []}
    if teacher:
        state["sentence_compress"] = {"compressed_observation": "Teacher privileged component context enabled."}
    return json.dumps({"task": "Choose exactly one legal Harness-1 tool call.", "state": state, "tools": list(TOOLS)}, ensure_ascii=False, sort_keys=True)


def generate_batch(tok, model, texts: list[str]) -> list[str]:
    messages = [[{"role": "system", "content": "You are a SCAPE research agent. Return exactly one legal Harness-1 tool call."}, {"role": "user", "content": text}] for text in texts]
    encoded = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", padding=True)
    ids = encoded["input_ids"] if hasattr(encoded, "__getitem__") else encoded
    mask = encoded["attention_mask"] if hasattr(encoded, "__getitem__") and "attention_mask" in encoded else None
    ids = ids.to(model.device)
    if mask is not None:
        mask = mask.to(model.device)
    with torch.inference_mode():
        out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=96, do_sample=False, pad_token_id=tok.eos_token_id)
    prompt_len = ids.shape[-1]
    return [tok.decode(row[prompt_len:], skip_special_tokens=False) for row in out]


def parse_action(text: str) -> dict[str, Any]:
    name = None
    params: dict[str, Any] = {}
    for m in re.finditer(r"\{.*?\}", text or "", flags=re.S):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        candidate = obj.get("tool") or obj.get("name") or obj.get("tool_name")
        if candidate in TOOLS:
            name = candidate
            params = obj.get("args") or obj.get("arguments") or obj.get("params") or obj.get("tool_input") or {}
            if not isinstance(params, dict):
                params = {}
            break
        for tool in TOOLS:
            if tool in obj:
                name, params = tool, obj.get(tool) or {}
                break
        if name:
            break
    if name is None:
        for tool in TOOLS:
            if re.search(rf"(?<![A-Za-z0-9_]){re.escape(tool)}(?![A-Za-z0-9_])", text or ""):
                name = tool
                break
    if name == "search_corpus" and "query" not in params and isinstance(params.get("q"), str):
        params["query"] = params["q"]
    legal = name is not None
    executable = False
    if name == "search_corpus":
        executable = isinstance(params.get("query"), str) and bool(params["query"].strip())
    elif name in {"fan_out_search", "grep_corpus"}:
        executable = True
    elif name in {"read_document", "review_docs"}:
        executable = bool(params.get("doc_id") or params.get("doc_ids"))
    elif name == "curate":
        executable = bool(params.get("add_ids") or params.get("remove_ids") or params.get("doc_ids"))
    elif name == "end_search":
        executable = True
    return {"tool_name": name, "params": params, "legal": legal, "executable": executable}


def search_query(action: dict[str, Any], row: dict[str, Any]) -> str | None:
    name, p = action.get("tool_name"), action.get("params") or {}
    if name in {"search_corpus", "fan_out_search"}:
        if isinstance(p.get("query"), str):
            return p["query"]
        if isinstance(p.get("queries"), list) and p["queries"]:
            return str(p["queries"][0])
        return row["query"]
    if name == "fan_out_search":
        qs = p.get("queries")
        return str(qs[0]) if isinstance(qs, list) and qs else row["query"]
    if name == "grep_corpus":
        return row["query"]
    return None


def norm_doc(doc: str) -> str:
    return str(doc).split("_", 1)[0]


def evaluate_setting(name: str, rows: list[dict[str, Any]], adapter: Path | None, teacher: bool, searcher: Any, out: Path) -> dict[str, Any]:
    tok, model, reload_path = load_backend(adapter)
    records = []
    for start in range(0, len(rows), 32):
        batch_rows = rows[start:start + 32]
        generated_batch = generate_batch(tok, model, [prompt(row, teacher) for row in batch_rows])
        for i, (row, generated) in enumerate(zip(batch_rows, generated_batch), start + 1):
            action = parse_action(generated)
            q = search_query(action, row)
            retrieved = []
            if q and action["legal"] and action["executable"]:
                retrieved = [str(h.docid) for h in searcher.search(q, 100)]
            gold = {norm_doc(x) for x in row["evidence_docids"]}
            hit = len({norm_doc(x) for x in retrieved} & gold)
            records.append({"query_id": row["query_id"], "setting": name, "generated_text": generated, **action, "retrieval_query": q, "retrieved_docids": retrieved, "evidence_recall": hit / max(1, len(gold)), "evidence_hits": hit, "evidence_qrel_count": len(gold)})
            if i % 32 == 0:
                print(json.dumps({"setting": name, "completed": i, "n": len(rows)}), flush=True)
    root = out / name
    root.mkdir(parents=True, exist_ok=True)
    with (root / "PER_QUERY.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {"setting": name, "n_queries": len(records), "legal_action_rate": sum(bool(r["legal"]) for r in records) / len(records), "executable_action_rate": sum(bool(r["executable"]) for r in records) / len(records), "test_evidence_recall": sum(r["evidence_recall"] for r in records) / len(records), "retrieval_nonempty_rate": sum(bool(r["retrieved_docids"]) for r in records) / len(records), "adapter_reload_path": reload_path}
    (root / "SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--only", default=None, help="comma-separated settings")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_pool(args.output_dir)
    if args.limit is not None:
        rows = rows[args.offset:args.offset + args.limit]
    elif args.offset:
        rows = rows[args.offset:]
    try:
        from pyserini.search.lucene import LuceneSearcher
        searcher = LuceneSearcher(str(BCP_ROOT / "indexes" / "bm25"))
        retrieval_backend = "pyserini_lucene"
    except Exception:
        class LocalHit:
            def __init__(self, docid, raw):
                self.docid, self.raw = docid, raw
        class LocalCorpusSearcher:
            def __init__(self, path):
                self.docs = []
                with path.open(encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            r = json.loads(line)
                            did = str(r.get("id") or r.get("docid") or r.get("source"))
                            text = str(r.get("text") or r.get("contents") or r.get("content") or "")
                            if did and text:
                                self.docs.append((did, text))
            def search(self, query, k=100):
                q = {x for x in query.lower().replace("_", " ").split() if len(x) > 2}
                scored = []
                for did, text in self.docs:
                    toks = set(text.lower().replace("_", " ").split())
                    overlap = len(q & toks)
                    if overlap:
                        scored.append((overlap, did, text))
                scored.sort(key=lambda x: (-x[0], x[1]))
                return [LocalHit(did, text) for _, did, text in scored[:k]]
        corpus = BCP_ROOT / "data" / "browsecomp_plus_decrypted.jsonl"
        searcher = LocalCorpusSearcher(corpus)
        retrieval_backend = "local_corpus_token_overlap_fallback"
    settings = [("TEACHER", None, True), ("STUDENT_BEFORE_OPD", None, False)]
    for label, adapter in ADAPTERS.items():
        settings.append(("STUDENT_AFTER_" + label, Path(adapter), False))
    if args.only:
        wanted = set(args.only.split(","))
        settings = [x for x in settings if x[0] in wanted]
    summaries = [evaluate_setting(n, rows, a, t, searcher, args.output_dir) for n, a, t in settings]
    payload = {"status": "SENTENCE_COMPRESS_OPD_384_READY", "component": "sentence_compress", "query_count": len(rows), "qrel": "qrel_evidence.txt", "normalization": "split_at_first_underscore_v1", "settings": summaries, "base_model": BASE_MODEL, "student_inference_privilege": False, "retrieval": "BrowseComp-Plus top-100 ordered docids", "retrieval_backend": retrieval_backend, "adapter_map": ADAPTERS}
    (args.output_dir / "SUMMARY.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    files = [p for p in args.output_dir.rglob("*") if p.is_file() and p.name != "SHA256SUMS"]
    (args.output_dir / "SHA256SUMS").write_text("\n".join(f"{sha256(p)}  {p.relative_to(args.output_dir)}" for p in sorted(files)) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
