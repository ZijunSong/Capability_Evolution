#!/usr/bin/env python3
"""Formal token_budget_marker OPD evaluation on the strict 384-query pool."""
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
    "PURE_OPD": "PURE_OPD_seed42",
    "RL_PLUS_OPD": "RL_PLUS_OPD_seed42",
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


def prompt(row: dict[str, Any], teacher: bool) -> str:
    state = {
        "component": "token_budget_marker",
        "query_id": row["query_id"],
        "query": row["query"],
        "student_inference_privilege": False,
        "visible_documents": [],
        "curated_ids": [],
        "search_history": [],
        "token_budget_marker": "remaining=8192" if teacher else None,
    }
    if not teacher:
        state.pop("token_budget_marker")
    return json.dumps({"task": "Choose exactly one legal Harness-1 tool call.", "state": state, "tools": list(TOOLS)}, ensure_ascii=False, sort_keys=True)


def generate(tok, model, text: str) -> str:
    msgs = [{"role": "system", "content": "You are a SCAPE research agent. Return exactly one legal Harness-1 tool call."}, {"role": "user", "content": text}]
    encoded = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    if isinstance(encoded, dict) or hasattr(encoded, "__getitem__"):
        ids = encoded["input_ids"]
    else:
        ids = encoded
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    ids = ids.to(model.device)
    with torch.inference_mode():
        out = model.generate(input_ids=ids, max_new_tokens=96, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[-1]:], skip_special_tokens=False)


def parse_action(text: str) -> dict[str, Any]:
    raw = text or ""
    name = next((t for t in TOOLS if re.search(rf"(?:to=|<tool_call>|\\b){re.escape(t)}\\b", raw)), None)
    params: dict[str, Any] = {}
    m = re.search(r"\{.*?\}(?:<\|im_end\|>)?", raw, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0).replace("<|im_end|>", ""))
            if isinstance(obj, dict):
                compact = obj.get("tool")
                if name is None and compact in TOOLS:
                    name = str(compact)
                nested = obj.get("args") or obj.get("arguments")
                params = dict(nested) if isinstance(nested, dict) else dict(obj)
                params.pop("tool", None)
                params.pop("args", None)
                params.pop("arguments", None)
        except json.JSONDecodeError:
            pass
    if name == "search_corpus" and "query" not in params and isinstance(params.get("q"), str):
        params["query"] = params["q"]
    legal = name is not None
    executable = False
    if name in {"search_corpus", "fan_out_search", "grep_corpus"}:
        executable = isinstance(params.get("query"), str) and bool(params["query"].strip()) if name == "search_corpus" else True
    elif name in {"read_document", "review_docs"}:
        executable = bool(params.get("doc_id") or params.get("doc_ids"))
    elif name == "curate":
        executable = bool(params.get("add_ids") or params.get("remove_ids") or params.get("doc_ids"))
    elif name == "end_search":
        executable = True
    return {"tool_name": name, "params": params, "legal": legal, "executable": executable}


def search_query(action: dict[str, Any], row: dict[str, Any]) -> str | None:
    name, p = action.get("tool_name"), action.get("params") or {}
    if name == "search_corpus":
        return str(p.get("query"))
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
    for i, row in enumerate(rows, 1):
        generated = generate(tok, model, prompt(row, teacher))
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
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_pool(args.output_dir)
    try:
        from pyserini.search.lucene import LuceneSearcher
    except Exception as exc:
        raise SystemExit(f"pyserini LuceneSearcher is required for official BM25 evaluation: {exc}")
    searcher = LuceneSearcher(str(BCP_ROOT / "indexes" / "bm25"))
    adapter_root = Path("/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD/outputs/component_sweep_0818/h100_4/token_budget_marker/formal_hf_token_budget_8gpu")
    settings = [("TEACHER", None, True), ("STUDENT_BEFORE_OPD", None, False)]
    for label, cell in ADAPTERS.items():
        summary = json.loads((adapter_root / cell / "summary.json").read_text(encoding="utf-8"))
        settings.append(("STUDENT_AFTER_" + label, Path(summary["adapter_path"]), False))
    summaries = [evaluate_setting(n, rows, a, t, searcher, args.output_dir) for n, a, t in settings]
    payload = {"status": "TOKEN_BUDGET_MARKER_OPD_384_READY", "component": "token_budget_marker", "query_count": len(rows), "qrel": "qrel_evidence.txt", "normalization": "split_at_first_underscore_v1", "settings": summaries, "base_model": BASE_MODEL, "student_inference_privilege": False, "retrieval": "BrowseComp-Plus official BM25 top-100 ordered docids"}
    (args.output_dir / "SUMMARY.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    files = [p for p in args.output_dir.rglob("*") if p.is_file() and p.name != "SHA256SUMS"]
    (args.output_dir / "SHA256SUMS").write_text("\n".join(f"{sha256(p)}  {p.relative_to(args.output_dir)}" for p in sorted(files)) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
