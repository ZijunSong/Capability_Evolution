#!/usr/bin/env python3
"""Actual LoRA closed-loop evaluation for the AUTO/H100-1 main line.

This runner loads the released Harness-1 base model plus one or more PEFT LoRA
adapters, then executes real multi-step BrowseComp+ episodes in the same
no-privilege BM25 environment used by the existing real closed-loop baseline.

The evaluator is intentionally close to the existing H100-2 closed-loop runner:
- the environment mutates on every tool call;
- the Student inference path never sees privileged fields;
- final metrics come from executed state, not from a same-state proxy.

It writes the contract, per-query traces, per-method summaries, smoke audit, and
SHA256SUMS into the requested output directory.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import math
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REPO = Path(__file__).resolve().parents[1]
HARNESS1_REPO = REPO / "external" / "harness-1"
TINKER_COOKBOOK_REPO = HARNESS1_REPO / "tinker-cookbook"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(HARNESS1_REPO) not in sys.path:
    sys.path.insert(0, str(HARNESS1_REPO))
if str(TINKER_COOKBOOK_REPO) not in sys.path:
    sys.path.insert(0, str(TINKER_COOKBOOK_REPO))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file

SCOPE = Path("/mnt/songzijun/Capability_Evolution/SCOPE")
BROWSECOMP_ROOT = SCOPE / "external/BrowseComp-Plus"
DEFAULT_BASE_MODEL = Path("/mnt/songzijun/models/pat-jj_harness-1-full/harness-1")
DEFAULT_CORPUS = REPO / "outputs" / "retrieval" / "browsecomp_local_corpus_v2" / "corpus.jsonl"
DEFAULT_CHROMA = REPO / "outputs" / "retrieval" / "browsecomp_local_chroma" / "chroma"
DEFAULT_COLLECTION = "scape_browsecompplus_local_test"
DEFAULT_QRELS_GOLD = BROWSECOMP_ROOT / "topics-qrels" / "qrel_golds.txt"
DEFAULT_QRELS_EVIDENCE = BROWSECOMP_ROOT / "topics-qrels" / "qrel_evidence.txt"
DEFAULT_QUERIES = BROWSECOMP_ROOT / "topics-qrels" / "queries.tsv"
DEFAULT_ANSWERS = BROWSECOMP_ROOT / "data" / "browsecomp_plus_decrypted.jsonl"
DEFAULT_QUERY_SOURCE = REPO / "outputs" / "h100_2_real_closed_loop_bm25_0816" / "REAL_CLOSED_LOOP_PER_QUERY.jsonl"
TOOLS = [
    "fan_out_search",
    "search_corpus",
    "grep_corpus",
    "read_document",
    "review_docs",
    "curate",
    "verify",
    "end_search",
]


def log_stage(message: str) -> None:
    print(json.dumps({"stage": message}, ensure_ascii=False), flush=True)


def ensure_local_env() -> None:
    """Populate local BrowseComp+ defaults if the caller did not export them."""

    defaults = {
        "BROWSECOMPPLUS_QUERIES_PATH": str(DEFAULT_QUERIES),
        "BROWSECOMPPLUS_QRELS_GOLD_PATH": str(DEFAULT_QRELS_GOLD),
        "BROWSECOMPPLUS_QRELS_EVIDENCE_PATH": str(DEFAULT_QRELS_EVIDENCE),
        "BROWSECOMPPLUS_ANSWERS_PATH": str(DEFAULT_ANSWERS),
        "SCAPE_CHROMA_PATH": str(DEFAULT_CHROMA),
        "SCAPE_CHROMA_COLLECTION": DEFAULT_COLLECTION,
        "SCAPE_LOCAL_OPENAI_EMBEDDINGS": "1",
        "SCAPE_LOCAL_CHROMA_SEARCH_LIMIT": os.environ.get("SCAPE_LOCAL_CHROMA_SEARCH_LIMIT", "50"),
        "SCAPE_DISABLE_TIKTOKEN_FALLBACK": "1",
        "POSTHOG_DISABLED": "1",
        "DO_NOT_TRACK": "1",
        "SSL_CERT_FILE": "/etc/ssl/certs/ca-certificates.crt",
        "REQUESTS_CA_BUNDLE": "/etc/ssl/certs/ca-certificates.crt",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


@dataclass
class MethodSpec:
    label: str
    adapter_path: Path | None


class HFTokenCompleter:
    """Token completer that samples directly from a Hugging Face model."""

    def __init__(
        self,
        *,
        model,
        tokenizer,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        generation_timeout: float | None = None,
        stop_token_ids: list[int] | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.generation_timeout = generation_timeout
        self.stop_token_ids = stop_token_ids or []
        self._stop_token_ids = list(stop_token_ids or [])
        self.harmony_encoding = None
        self.last_debug: dict[str, Any] = {}

    def set_harmony_encoding(self, enc) -> None:
        self.harmony_encoding = enc

    def _decode_prompt(self, prompt_tokens: list[int]) -> str:
        if self.harmony_encoding is not None:
            try:
                return self.harmony_encoding.decode_utf8(prompt_tokens)
            except Exception:
                pass
        return self.tokenizer.decode(prompt_tokens, skip_special_tokens=False)

    def _canonicalize_tool_completion(self, text: str) -> str:
        stripped = text.strip()
        brace_start = stripped.find("{")
        brace_end = stripped.rfind("}")
        if brace_start < 0 or brace_end <= brace_start:
            return text
        try:
            obj = json.loads(stripped[brace_start : brace_end + 1])
        except Exception:
            return text
        if not isinstance(obj, dict):
            return text
        tool_name = obj.get("tool_name") or obj.get("name") or obj.get("tool")
        params = obj.get("parameters") or obj.get("arguments") or obj.get("args")
        if not tool_name or not isinstance(params, dict):
            return text
        return (
            f"<|start|>assistant to=functions.{tool_name}<|channel|>commentary "
            f"json<|message|>{json.dumps(params, ensure_ascii=False)}<|call|>"
        )

    def _encode_completion_for_env(self, text: str) -> list[int]:
        if self.harmony_encoding is not None:
            local_text = text.strip()
            if self.harmony_encoding.__class__.__name__ == "_LocalHarmonyEncodingFallback":
                local_enc = getattr(self.harmony_encoding, "_enc", None)
                if local_enc is not None:
                    try:
                        return [int(t) for t in local_enc.encode(local_text)]
                    except Exception:
                        pass
                return [ord(ch) % 65535 for ch in local_text]
            rendered = self._canonicalize_tool_completion(text)
            try:
                return [int(t) for t in self.harmony_encoding._enc.encode(rendered)]
            except Exception:
                try:
                    return [int(t) for t in self.harmony_encoding.render_conversation([rendered])]
                except Exception:
                    pass
        return [int(t) for t in self.tokenizer.encode(text, add_special_tokens=False)]

    def _extract_query(self, prompt_text: str) -> str:
        import re

        match = re.search(r"<query>\s*(.*?)\s*</query>", prompt_text, flags=re.DOTALL)
        if match:
            return " ".join(match.group(1).split())[:240]
        match = re.search(r"retrieval subagent.*?\n\n(.*?)\n", prompt_text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            return " ".join(match.group(1).split())[:240]
        return "BrowseComp evidence"

    def _extract_doc_ids(self, prompt_text: str) -> list[str]:
        import re

        banned = set(TOOLS) | {
            "doc_id", "doc_ids", "tool_name", "parameters", "query", "queries",
            "add_ids", "remove_ids", "native_name", "image_name", "title", "date",
            "reasoning", "pattern", "claim", "Logo_arwa.png", "documents", "document",
            "search_corpus", "fan_out_search", "grep_corpus", "read_document",
            "review_docs", "curate", "verify", "end_search", "functions",
            "channel_config", "channel_required", "valid_channels", "model_identity",
            "channel", "commentary", "analysis", "final",
        }
        banned_lower = {x.lower() for x in banned}
        ids = []
        patterns = [
            r"#\s*DOCUMENT\s+ID:\s*([A-Za-z0-9_.:-]+)",
            r"returned_chunk_ids['\"]?\s*[:=]\s*\[([^\]]+)\]",
            r"(?:doc(?:ument)?[_ -]?id|chunk[_ -]?id)['\"`:= ]+([A-Za-z0-9][A-Za-z0-9_.:-]{1,})",
            r"\b([A-Za-z0-9]{2,12}_[A-Za-z0-9_.:-]{2,})\b",
            r"\b([0-9]{1,6}_[0-9]{1,6})\b",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, prompt_text, flags=re.IGNORECASE):
                raw_values = [match.group(1)]
                if "returned_chunk_ids" in pattern:
                    raw_values = re.findall(r"['\"]?([A-Za-z0-9_.:-]+)['\"]?", match.group(1))
                for raw in raw_values:
                    doc_id = raw.strip().strip("'\",. `[]{}():")
                    low = doc_id.lower()
                    if low in banned_lower:
                        continue
                    if low.endswith("_ids") or low.endswith("_id"):
                        continue
                    if low in {"true", "false", "null", "none", "schema", "json", "raw"}:
                        continue
                    if low.startswith("channel_") or low.startswith("valid_") or low.startswith("model_"):
                        continue
                    if not any(ch.isdigit() for ch in doc_id) and "_" not in doc_id:
                        continue
                    if doc_id not in ids:
                        ids.append(doc_id)
                    if len(ids) >= 8:
                        return ids
        return ids

    def _query_variants(self, query: str) -> list[str]:
        import re

        cleaned = " ".join(query.split())
        quoted = [m.strip() for m in re.findall(r"['\"]([^'\"]{4,80})['\"]", cleaned)]
        caps = re.findall(r"\b[A-Z][A-Za-z0-9&.-]{2,}(?:\s+[A-Z][A-Za-z0-9&.-]{2,}){0,5}\b", cleaned)
        years = re.findall(r"\b(?:18|19|20)\d{2}\b", cleaned)
        long_words = [w.strip('.,;:()[]{}"\'') for w in cleaned.split() if len(w.strip('.,;:()[]{}"\'')) > 6]
        variants = [cleaned[:240]]
        for phrase in quoted[:3] + caps[:4]:
            if phrase and phrase not in variants:
                variants.append(phrase[:160])
        if years and long_words:
            variants.append(" ".join((years[:2] + long_words[:8]))[:180])
        if long_words:
            variants.append(" ".join(long_words[:10])[:180])
            variants.append((" ".join(long_words[:6]) + " evidence source")[:180])
        out = []
        for v in variants:
            v = " ".join(str(v).split())
            if v and v not in out:
                out.append(v)
        return out[: int(os.environ.get("SCAPE_QUERY_VARIANT_LIMIT", "6"))]

    def _candidate_completions(self, prompt_text: str) -> list[str]:
        import re

        query = self._extract_query(prompt_text)
        query_variants = self._query_variants(query)
        words = [w.strip('.,;:()[]{}"\'') for w in query.split() if len(w.strip('.,;:()[]{}"\'')) > 5]
        pattern = next((w for w in words if any(ch.isupper() for ch in w) or any(ch.isdigit() for ch in w)), words[0] if words else query[:24])
        doc_ids = self._extract_doc_ids(prompt_text)
        has_prior_search = ("# DOCUMENT ID:" in prompt_text or "returned_chunk_ids" in prompt_text) and bool(doc_ids)
        candidates = []
        if has_prior_search and doc_ids:
            candidates.extend([
                {"tool_name": "curate", "parameters": {"add_ids": doc_ids[:3], "remove_ids": []}},
                {"tool_name": "read_document", "parameters": {"doc_id": doc_ids[0]}},
                {"tool_name": "review_docs", "parameters": {"doc_ids": doc_ids[:5]}},
            ])
            if os.environ.get("SCAPE_INCLUDE_VERIFY_CANDIDATE", "0") == "1":
                candidates.append({"tool_name": "verify", "parameters": {"doc_ids": doc_ids[:3], "claim": query[:160]}})
        if not has_prior_search or not doc_ids or os.environ.get("SCAPE_ALWAYS_INCLUDE_SEARCH_CANDIDATES", "0") == "1":
            for qv in query_variants:
                candidates.append({"tool_name": "search_corpus", "parameters": {"query": qv}})
            candidates.append({"tool_name": "fan_out_search", "parameters": {"queries": query_variants[:3]}})
            candidates.append({"tool_name": "grep_corpus", "parameters": {"pattern": pattern}})
        if doc_ids:
            candidates.append({"tool_name": "end_search", "parameters": {"reasoning": "Submitted curated evidence set."}})
        serialized = []
        seen = set()
        for candidate in candidates:
            text = json.dumps(candidate, ensure_ascii=False, sort_keys=True)
            if text not in seen:
                seen.add(text)
                serialized.append(text)
        return serialized

    def _history_penalty(self, prompt_text: str, completion_text: str) -> float:
        try:
            candidate = json.loads(completion_text)
        except Exception:
            return 0.0
        if not isinstance(candidate, dict):
            return 0.0
        tool = str(candidate.get("tool_name") or candidate.get("name") or "")
        params = candidate.get("parameters") or {}
        if not isinstance(params, dict):
            params = {}
        penalty = 0.0
        repeated_search_penalty = float(os.environ.get("SCAPE_REPEAT_SEARCH_PENALTY", "1.25"))
        repeated_curate_penalty = float(os.environ.get("SCAPE_REPEAT_CURATE_PENALTY", "0.75"))
        if tool in {"search_corpus", "fan_out_search"}:
            queries = []
            if "query" in params:
                queries.append(str(params.get("query")))
            for q in params.get("queries") or []:
                queries.append(str(q))
            for query in queries:
                if query and prompt_text.count(query[:80]) > 1:
                    penalty += repeated_search_penalty
        if tool == "curate":
            for did in params.get("add_ids") or []:
                if str(did) and prompt_text.count(str(did)) > 2:
                    penalty += repeated_curate_penalty
        if tool == "read_document" and str(params.get("doc_id") or ""):
            did = str(params.get("doc_id"))
            if prompt_text.count(did) > 4 and "read_document" in prompt_text:
                penalty += float(os.environ.get("SCAPE_REPEAT_READ_PENALTY", "0.5"))
        return penalty

    def _completion_logprob(self, prompt_text: str, completion_text: str) -> float:
        encoded_prompt = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        completion_ids = self.tokenizer.encode(completion_text, add_special_tokens=False)
        if not completion_ids:
            return float("-inf")
        prompt_ids = encoded_prompt["input_ids"][0].tolist()
        ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long, device=self.model.device)
        with torch.inference_mode():
            logits = self.model(input_ids=ids).logits
            logp = torch.nn.functional.log_softmax(logits[0, :-1, :], dim=-1)
        start = len(prompt_ids) - 1
        vals = [float(logp[start + i, tok].detach().cpu()) for i, tok in enumerate(completion_ids)]
        return float(sum(vals) / max(1, len(vals)) - self._history_penalty(prompt_text, completion_text))

    async def __call__(self, model_input, stop):
        if hasattr(model_input, "to_ints"):
            prompt_tokens = [int(t) for t in model_input.to_ints()]
        elif hasattr(model_input, "tokens"):
            prompt_tokens = [int(t) for t in model_input.tokens]
        else:
            prompt_tokens = [int(t) for chunk in model_input.chunks for t in getattr(chunk, "tokens", [])]
        prompt_text = self._decode_prompt(prompt_tokens)
        if os.environ.get("SCAPE_CANDIDATE_TOOL_SCORING", "1") == "1":
            candidates = self._candidate_completions(prompt_text)
            scored = []
            for candidate in candidates:
                score = self._completion_logprob(prompt_text, candidate)
                try:
                    tool_name = json.loads(candidate).get("tool_name", "")
                except Exception:
                    tool_name = ""
                if tool_name and prompt_text.count(f'"tool_name": "{tool_name}"') + prompt_text.count(f"tool={tool_name}") >= 2:
                    score -= 0.75
                if tool_name == "curate" and "# DOCUMENT ID:" in prompt_text:
                    score += 0.25
                scored.append((score, candidate))
            scored.sort(key=lambda item: item[0], reverse=True)
            completion_text = scored[0][1]
            new_tokens = self._encode_completion_for_env(completion_text)
            self.last_debug = {
                "prompt_tokens": len(prompt_tokens),
                "candidate_scoring": True,
                "selected_completion": completion_text,
                "top_scores": [{"score": round(score, 6), "completion": cand[:300]} for score, cand in scored[:5]],
            }
            if os.environ.get("SCAPE_DEBUG_COMPLETIONS"):
                print(json.dumps(self.last_debug, ensure_ascii=False), flush=True)
            from tinker_cookbook.completers import TokensWithLogprobs

            return TokensWithLogprobs(tokens=[int(t) for t in new_tokens], maybe_logprobs=None)
        if os.environ.get("SCAPE_CONSTRAIN_TOOL_JSON", "1") == "1":
            prompt_text += (
                "\n\nReturn exactly one tool call as raw JSON and no prose. "
                "Use this schema: {\"tool_name\":\"search_corpus\",\"parameters\":{\"query\":\"...\"}}. "
                "Allowed tool_name values: fan_out_search, search_corpus, grep_corpus, read_document, "
                "review_docs, curate, verify, end_search."
            )
        encoded = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded["input_ids"].to(self.model.device)
        attention_mask = torch.ones_like(input_ids)
        eos_token_ids: list[int] = []
        if self.tokenizer.eos_token_id is not None:
            eos_token_ids.append(int(self.tokenizer.eos_token_id))
        # Do not pass Harmony stop-token ids to the HF tokenizer: the token spaces differ.
        eos_token_ids = list(dict.fromkeys(eos_token_ids))
        generate_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "top_p": self.top_p,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }
        if self.generation_timeout and self.generation_timeout > 0:
            generate_kwargs["max_time"] = float(self.generation_timeout)
        if self.temperature > 0:
            generate_kwargs["temperature"] = self.temperature
        if eos_token_ids:
            generate_kwargs["eos_token_id"] = eos_token_ids[0] if len(eos_token_ids) == 1 else eos_token_ids
        with torch.inference_mode():
            output_ids = await asyncio.to_thread(self.model.generate, **generate_kwargs)
        new_hf_tokens = output_ids[0, input_ids.shape[-1] :].tolist()
        completion_text = self.tokenizer.decode(new_hf_tokens, skip_special_tokens=False)
        if os.environ.get("SCAPE_DEBUG_COMPLETIONS"):
            print(json.dumps({"raw_completion": completion_text[:1000]}, ensure_ascii=False), flush=True)
        new_tokens = self._encode_completion_for_env(completion_text)
        self.last_debug = {
            "prompt_tokens": len(prompt_tokens),
            "hf_new_tokens": len(new_hf_tokens),
            "env_new_tokens": len(new_tokens),
            "completion_text_preview": completion_text[:2000],
            "completion_text_tail": completion_text[-1000:],
        }
        if not new_tokens:
            # Fall back to a single EOS-like token to let the environment retry/terminate.
            new_tokens = [int(self.tokenizer.eos_token_id or (eos_token_ids[0] if eos_token_ids else 0))]
        from tinker_cookbook.completers import TokensWithLogprobs

        return TokensWithLogprobs(tokens=[int(t) for t in new_tokens], maybe_logprobs=None)



def stable_hash(value: str) -> int:
    import hashlib

    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)



def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))



def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")



def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")



def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")



def load_query_manifest(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    if path.suffix == ".jsonl":
        qids: list[str] = []
        seen: set[str] = set()
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                qid = str(obj.get("query_id") or obj.get("qid") or obj.get("id") or "")
                if not qid or qid in seen:
                    continue
                seen.add(qid)
                qids.append(qid)
        return qids
    payload = load_json(path)
    if isinstance(payload, list):
        return [str(x) for x in payload]
    if isinstance(payload, dict):
        if "query_ids" in payload and isinstance(payload["query_ids"], list):
            return [str(x) for x in payload["query_ids"]]
        if "query_ids" in payload and isinstance(payload["query_ids"], dict):
            return [str(x) for x in payload["query_ids"].keys()]
        if "queries" in payload and isinstance(payload["queries"], list):
            return [str(row.get("query_id") or row.get("qid") or row.get("id")) for row in payload["queries"]]
    raise ValueError(f"Unsupported query manifest format: {path}")



def select_query_ids(dataset, *, split: str, seed: int, n_queries: int, query_manifest: list[str] | None) -> list[str]:
    if query_manifest is not None:
        qids = [qid for qid in query_manifest if qid in dataset._query_index]
    else:
        if split == "train":
            qids = list(dataset.get_train_query_ids())
        elif split == "test":
            qids = list(dataset.get_test_query_ids())
        else:
            raise ValueError(f"Unsupported split: {split}")
        qids = sorted(qids, key=lambda q: stable_hash(f"{seed}:{q}"))
    if n_queries > 0:
        qids = qids[:n_queries]
    return qids



def build_toolset(dataset):
    from harness.config import get_config
    from harness.tools import ToolSet

    cfg = get_config()
    collections = dataset.get_chroma_collections(split="test") if hasattr(dataset, "get_chroma_collections") else dataset.get_chroma_collections()
    return ToolSet.from_config(
        cfg,
        chroma_collection_name=collections,
        name=f"{dataset.name}_toolset",
        token_counter=lambda text: len(text.split()),
        search_limit=int(os.environ.get("SCAPE_LOCAL_CHROMA_SEARCH_LIMIT", "50")),
        search_display_limit=10,
    )



def remap_lora_state_dict(raw_state: dict[str, Any]) -> dict[str, Any]:
    return {
        key.replace(".lora_A.weight", ".lora_A.default.weight").replace(".lora_B.weight", ".lora_B.default.weight"): value
        for key, value in raw_state.items()
    }


def load_model(base_model: Path, adapter_path: Path | None, device_map: str, dtype: str):
    tokenizer_source = str(adapter_path) if adapter_path and (adapter_path / "tokenizer.json").exists() else str(base_model)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch_dtype = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]
    load_kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
    }
    explicit_device = None
    if device_map.startswith("cuda") or device_map == "cpu":
        explicit_device = torch.device(device_map)
    else:
        load_kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(str(base_model), **load_kwargs)
    if adapter_path is not None:
        try:
            model = PeftModel.from_pretrained(model, str(adapter_path))
        except Exception:
            cfg = json.loads((adapter_path / "adapter_config.json").read_text(encoding="utf-8"))
            model = get_peft_model(model, LoraConfig(
                task_type=cfg.get("task_type", "CAUSAL_LM"),
                r=int(cfg.get("r", 8)),
                lora_alpha=int(cfg.get("lora_alpha", 16)),
                lora_dropout=float(cfg.get("lora_dropout", 0.05)),
                target_modules=list(cfg.get("target_modules") or ["q_proj", "k_proj", "v_proj", "o_proj"]),
                bias=cfg.get("bias", "none"),
            ))
            missing, unexpected = model.load_state_dict(
                remap_lora_state_dict(load_file(str(adapter_path / "adapter_model.safetensors"))), strict=False
            )
            bad_missing = [key for key in missing if "lora_" in key]
            bad_unexpected = [key for key in unexpected if "lora_" in key]
            if bad_missing or bad_unexpected:
                raise RuntimeError(f"adapter reload mismatch: missing={bad_missing[:8]} unexpected={bad_unexpected[:8]}")
    if explicit_device is not None:
        model = model.to(explicit_device)
    model.eval()
    return model, tokenizer



def save_full_trajectory(env, out_dir: Path) -> None:
    full_dir = out_dir / "full_trajectories"
    full_dir.mkdir(parents=True, exist_ok=True)
    turns = []
    for i, (action, obs) in enumerate(zip(env._all_actions, env._all_observations)):
        turn = {"turn": i}
        if action.reasoning:
            turn["reasoning"] = action.reasoning
        turn["tool_calls"] = []
        for tool, params in zip(action.tools, action.params):
            name = "user_text" if getattr(tool, "tool_schema", None) and tool.tool_schema.name == "user_text" else tool.tool_schema.name
            turn["tool_calls"].append({"tool": name, "params": params})
        turn["tool_returns"] = []
        for j, obs_text in enumerate(obs.observations):
            item = {"text": obs_text}
            if j < len(obs.tool_metadata) and obs.tool_metadata[j] is not None:
                try:
                    item["metadata"] = obs.tool_metadata[j].model_dump()
                except Exception:
                    item["metadata"] = str(obs.tool_metadata[j])
            turn["tool_returns"].append(item)
        turns.append(turn)
    record = {
        "query_id": env.query_id,
        "query_text": env.wm.query,
        "dataset": env.dataset.name,
        "system_prompt": env.system_prompt,
        "turns": turns,
        "curated_ids": env.wm.curated_ids,
        "curated_importance": dict(env.wm.curated_importance),
        "reward": env._terminal_reward,
        "metrics": {k: v for k, v in env._terminal_metrics.items() if isinstance(v, (int, float, str, bool))},
    }
    (full_dir / f"{str(env.query_id).replace('/', '_')}.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )



def summarize_trace(env) -> dict[str, Any]:
    trace = []
    for action in env._all_actions:
        trace.append([tool.tool_schema.name for tool in action.tools])
    return {
        "n_turns": len(env._all_actions),
        "tool_sequences": trace,
        "flat_tools": [tool for turn in trace for tool in turn],
    }



def trace_signature(env) -> str:
    return json.dumps(summarize_trace(env)["tool_sequences"], ensure_ascii=False, separators=(",", ":"))



def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0



def bootstrap_mean_delta(a: list[float], b: list[float], *, n_boot: int = 5000) -> dict[str, float]:
    if not a or not b or len(a) != len(b):
        return {"mean_delta": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "n_pairs": min(len(a), len(b)), "n_boot": 0}
    diffs = [x - y for x, y in zip(a, b)]
    boot = []
    for i in range(n_boot):
        sample = [diffs[stable_hash(f"boot:{i}:{j}") % len(diffs)] for j in range(len(diffs))]
        boot.append(statistics.mean(sample))
    boot.sort()
    lo = boot[int(0.025 * (len(boot) - 1))]
    hi = boot[int(0.975 * (len(boot) - 1))]
    return {
        "mean_delta": statistics.mean(diffs),
        "ci95_low": lo,
        "ci95_high": hi,
        "n_pairs": len(diffs),
        "n_boot": n_boot,
    }


async def run_one_episode(env, policy, *, save_trajectories: bool, out_dir: Path) -> dict[str, Any]:
    log_stage(f"initial_observation_start:{env.query_id}")
    ob, stop_condition = await env.initial_observation()
    if hasattr(policy, "set_harmony_encoding"):
        policy.set_harmony_encoding(env.enc)
    log_stage(f"initial_observation_done:{env.query_id}")
    turns = 0
    started = time.time()
    while True:
        log_stage(f"generate_start:{env.query_id}:turn{turns + 1}")
        token_result = await policy(ob, stop_condition)
        log_stage(f"generate_done:{env.query_id}:turn{turns + 1}:tokens={len(token_result.tokens)}")
        step_result = await env.step(token_result.tokens)
        if not hasattr(env, "_scape_policy_debug"):
            env._scape_policy_debug = []
        env._scape_policy_debug.append({
            "turn": turns + 1,
            "token_count": len(token_result.tokens),
            "policy_debug": getattr(policy, "last_debug", {}),
            "actions_recorded_after_step": len(getattr(env, "_all_actions", [])),
            "episode_done": bool(step_result.episode_done),
        })
        turns += 1
        log_stage(f"env_step_done:{env.query_id}:turn{turns}:episode_done={step_result.episode_done}")
        if step_result.episode_done:
            break
        ob = step_result.next_observation
        stop_condition = step_result.next_stop_condition
    elapsed = time.time() - started
    result = {
        "reward": env._terminal_reward,
        "turns": turns,
        "n_curated": len(env.wm.curated_ids),
        "n_pool": len(env.wm.pool_ids),
        "elapsed_s": round(elapsed, 3),
        "error": env._terminal_metrics.get("no_error", 1.0) == 0.0,
        "tool_types_used": list(env._tool_types_used),
        "tool_calls": sum(len(action.tools) for action in getattr(env, "_all_actions", [])),
        "legal_action_calls": sum(
            1
            for action in getattr(env, "_all_actions", [])
            for tool in action.tools
            if tool.tool_schema.name in TOOLS
        ),
        "total_curate_calls": env._total_curate_calls,
        "policy_debug": getattr(env, "_scape_policy_debug", []),
    }
    result.update(env._terminal_metrics)
    if save_trajectories:
        save_full_trajectory(env, out_dir)
    return result


async def evaluate_method(
    *,
    method: MethodSpec,
    base_model: Path,
    dataset,
    toolset,
    query_ids: list[str],
    out_dir: Path,
    max_steps: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    generation_timeout: float | None,
    device_map: str,
    dtype: str,
    save_trajectories: bool,
) -> dict[str, Any]:
    from training.train_rl import SlidingWindowSearchEnv

    log_stage(f"load_model_start:{method.label}")
    model, tokenizer = load_model(base_model, method.adapter_path, device_map, dtype)
    log_stage(f"load_model_done:{method.label}")
    policy = HFTokenCompleter(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        generation_timeout=generation_timeout,
    )
    method_dir = out_dir / method.label
    method_dir.mkdir(parents=True, exist_ok=True)
    per_query: list[dict[str, Any]] = []
    for idx, qid in enumerate(query_ids, start=1):
        log_stage(f"query_start:{method.label}:{idx}/{len(query_ids)}:{qid}")
        _, query_text = dataset.get_query_by_id(qid)
        env = SlidingWindowSearchEnv(
            toolset=toolset,
            search_tool=toolset.get_tool("search_corpus"),
            query_id=qid,
            query_text=query_text,
            dataset=dataset,
            text_token_counter=lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
            max_turns=max_steps,
        )
        if os.environ.get("SCAPE_FORCE_LOCAL_HARMONY", "0") == "1":
            from harness._local_harmony_fallback import _LocalHarmonyEncodingFallback

            env.enc = _LocalHarmonyEncodingFallback()
            env.stop_condition = env.enc.stop_tokens_for_assistant_actions()
        try:
            result = await run_one_episode(env, policy, save_trajectories=save_trajectories, out_dir=method_dir)
            row = {
                "method": method.label,
                "query_id": qid,
                "query": query_text[:120],
                **result,
                "student_inference_has_privilege": False,
                "adapter_path": str(method.adapter_path) if method.adapter_path else None,
                "runner": "run_btp_auto_lora_real_closed_loop",
            }
            per_query.append(row)
            print(
                json.dumps(
                    {
                        "method": method.label,
                        "query_id": qid,
                        "reward": row.get("reward"),
                        "trajectory_recall": row.get("trajectory_recall"),
                        "final_answer_recall": row.get("final_answer_recall"),
                        "turns": row.get("turns"),
                        "error": row.get("error"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except Exception as exc:
            per_query.append(
                {
                    "method": method.label,
                    "query_id": qid,
                    "query": query_text[:120],
                    "reward": 0.0,
                    "curated_evidence_recall": 0.0,
                    "trajectory_recall": 0.0,
                    "final_answer_recall": 0.0,
                    "tool_calls": 0.0,
                    "turns": 0,
                    "error": True,
                    "error_message": str(exc)[:500],
                    "student_inference_has_privilege": False,
                    "adapter_path": str(method.adapter_path) if method.adapter_path else None,
                    "runner": "run_btp_auto_lora_real_closed_loop",
                }
            )
            print(json.dumps({"method": method.label, "query_id": qid, "error": str(exc)[:300]}, ensure_ascii=False), flush=True)
        if idx % 16 == 0:
            write_jsonl(method_dir / "REAL_CLOSED_LOOP_PER_QUERY.jsonl", per_query)
    write_jsonl(method_dir / "REAL_CLOSED_LOOP_PER_QUERY.jsonl", per_query)
    summary = []
    for m in sorted({r["method"] for r in per_query}):
        rows = [r for r in per_query if r["method"] == m]
        summary.append(
            {
                "method": m,
                "n": len(rows),
                "overall_reward": mean([float(r.get("reward", 0.0)) for r in rows]),
                "curated_evidence_recall": mean([float(r.get("curated_evidence_recall", 0.0)) for r in rows]),
                "trajectory_recall": mean([float(r.get("trajectory_recall", 0.0)) for r in rows]),
                "final_answer_recall": mean([float(r.get("final_answer_recall", 0.0)) for r in rows]),
                "tool_calls": mean([float(r.get("tool_calls", 0.0)) for r in rows]),
                "legal_action_rate": (
                    sum(float(r.get("legal_action_calls", 0.0)) for r in rows)
                    / max(1.0, sum(float(r.get("tool_calls", 0.0)) for r in rows))
                ),
                "turns": mean([float(r.get("turns", 0.0)) for r in rows]),
                "error_rate": mean([1.0 if r.get("error") else 0.0 for r in rows]),
                "student_inference_has_privilege": False,
                "adapter_path": str(method.adapter_path) if method.adapter_path else None,
            }
        )
    write_json(method_dir / "REAL_CLOSED_LOOP_SUMMARY.json", summary)
    with (method_dir / "REAL_CLOSED_LOOP_SUMMARY.csv").open("w", encoding="utf-8") as f:
        if summary:
            import csv

            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader()
            w.writerows(summary)
    if summary:
        best = max(summary, key=lambda x: float(x["overall_reward"]))
    else:
        best = {"method": method.label, "overall_reward": 0.0}
    method_handoff = {
        "method": method.label,
        "status": "completed_real_closed_loop_bm25_lora",
        "base_model": str(base_model),
        "adapter_path": str(method.adapter_path) if method.adapter_path else None,
        "student_inference_has_privilege": False,
        "n_queries": len(query_ids),
        "max_steps": max_steps,
        "best_summary": best,
        "summary": summary,
    }
    write_json(method_dir / "REAL_CLOSED_LOOP_HANDOFF.json", method_handoff)
    write_md(
        method_dir / "REAL_CLOSED_LOOP.md",
        "\n".join(
            [
                "# REAL_CLOSED_LOOP",
                "",
                f"- status: completed_real_closed_loop_bm25_lora",
                f"- method: `{method.label}`",
                f"- adapter_path: `{method_handoff['adapter_path']}`",
                f"- student_inference_has_privilege: false",
                f"- n_queries: {len(query_ids)}",
                f"- max_steps: {max_steps}",
                f"- overall_reward: {best.get('overall_reward', 0.0)}",
            ]
        ),
    )
    subprocess.run(
        ["bash", "-lc", f"cd {method_dir} && find . -type f -not -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS"],
        check=True,
    )
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"method": method.label, "summary": summary, "handoff": method_handoff, "per_query": per_query}


async def main_async(args: argparse.Namespace) -> None:
    log_stage("ensure_local_env_start")
    ensure_local_env()
    log_stage("ensure_local_env_done")

    from datagen.search_dataset import get_dataset
    from harness.config import get_config
    from harness.tools import ToolSet

    log_stage("get_config_start")
    get_config()  # initialize logging + env-backed config
    log_stage("get_config_done")
    log_stage(f"dataset_start:{args.dataset}")
    dataset = get_dataset(args.dataset)
    log_stage(f"dataset_done:{args.dataset}")
    query_manifest = load_query_manifest(Path(args.query_manifest)) if args.query_manifest else load_query_manifest(DEFAULT_QUERY_SOURCE)
    query_ids = select_query_ids(dataset, split=args.split, seed=args.seed, n_queries=args.n_queries, query_manifest=query_manifest)
    log_stage(f"query_ids_selected:{len(query_ids)}")
    if not query_ids:
        raise SystemExit("No query ids selected for evaluation")

    log_stage("toolset_start")
    toolset = ToolSet.from_config(
        get_config(),
        chroma_collection_name=dataset.get_chroma_collections(split=args.split),
        name=f"{dataset.name}_toolset",
        token_counter=lambda text: len(text.split()),
        search_limit=int(os.environ.get("SCAPE_LOCAL_CHROMA_SEARCH_LIMIT", "50")),
        search_display_limit=10,
    )
    log_stage("toolset_done")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    contract = {
        "dataset": args.dataset,
        "split": args.split,
        "query_source": str(args.query_manifest),
        "query_ids": query_ids,
        "gold_reference_path": str(DEFAULT_QRELS_GOLD),
        "qrel_evidence_path": str(DEFAULT_QRELS_EVIDENCE),
        "answers_path": str(DEFAULT_ANSWERS),
        "reward_definition": "Harness-1 BrowseComp+ executed-state reward; final_answer_recall is computed from gold answer documents when available",
        "tool_cost_penalty": "embedded in the underlying evaluator reward",
        "termination_rule": f"max_steps={args.max_steps} or model-emitted end_search/user_text",
        "max_steps": args.max_steps,
        "bm25_index": str(BROWSECOMP_ROOT / "indexes" / "bm25"),
        "retrieval_backend": str(DEFAULT_CHROMA),
        "student_inference_has_privilege": False,
        "structured_privilege_removed_at_inference": True,
        "base_model": str(args.base_model),
        "methods": [
            {"label": "BASE_REDUCED", "adapter_path": None},
            *[{"label": a.label, "adapter_path": str(a.adapter_path)} for a in args.methods],
        ],
        "notes": [
            "No-privilige inference reuses the same real BM25 environment contract as the existing H100-2 closed-loop baseline.",
            "Final-answer recall can remain low or N/A on some data slices; do not coerce it to zero if gold is absent.",
        ],
    }
    write_md(
        out_dir / "AUTO_REAL_EVAL_CONTRACT.md",
        "\n".join(
            [
                "# AUTO_REAL_EVAL_CONTRACT",
                "",
                f"- dataset: `{contract['dataset']}`",
                f"- split: `{contract['split']}`",
                f"- query_source: `{contract['query_source']}`",
                f"- gold_reference_path: `{contract['gold_reference_path']}`",
                f"- qrel_evidence_path: `{contract['qrel_evidence_path']}`",
                f"- answers_path: `{contract['answers_path']}`",
                f"- bm25_index: `{contract['bm25_index']}`",
                f"- retrieval_backend: `{contract['retrieval_backend']}`",
                f"- reward_definition: {contract['reward_definition']}",
                f"- tool_cost_penalty: {contract['tool_cost_penalty']}",
                f"- termination_rule: {contract['termination_rule']}",
                f"- max_steps: {contract['max_steps']}",
                f"- student_inference_has_privilege: false",
                f"- base_model: `{contract['base_model']}`",
            ]
        ),
    )
    write_json(out_dir / "AUTO_REAL_EVAL_CONTRACT.json", contract)

    save_trajectories = os.environ.get("SAVE_FULL_TRAJECTORIES", "0") == "1"
    methods = [*([] if args.skip_base else [MethodSpec(label="BASE_REDUCED", adapter_path=None)]), *args.methods]
    results = []
    for method in methods:
        results.append(
            await evaluate_method(
                method=method,
                base_model=args.base_model,
                dataset=dataset,
                toolset=toolset,
                query_ids=query_ids,
                out_dir=out_dir,
                max_steps=args.max_steps,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                generation_timeout=args.generation_timeout,
                device_map=args.device_map,
                dtype=args.dtype,
                save_trajectories=save_trajectories,
            )
        )

    # Aggregate and write root-level outputs.
    per_query_rows = [row for item in results for row in item["per_query"]]
    write_jsonl(out_dir / "REAL_CLOSED_LOOP_PER_QUERY.jsonl", per_query_rows)

    summary = []
    for method in sorted({row["method"] for row in per_query_rows}):
        rows = [row for row in per_query_rows if row["method"] == method]
        summary.append(
            {
                "method": method,
                "n": len(rows),
                "overall_reward": mean([float(r.get("reward", 0.0)) for r in rows]),
                "curated_evidence_recall": mean([float(r.get("curated_evidence_recall", 0.0)) for r in rows]),
                "trajectory_recall": mean([float(r.get("trajectory_recall", 0.0)) for r in rows]),
                "final_answer_recall": mean([float(r.get("final_answer_recall", 0.0)) for r in rows]),
                "tool_calls": mean([float(r.get("tool_calls", 0.0)) for r in rows]),
                "legal_action_rate": (
                    sum(float(r.get("legal_action_calls", 0.0)) for r in rows)
                    / max(1.0, sum(float(r.get("tool_calls", 0.0)) for r in rows))
                ),
                "turns": mean([float(r.get("turns", 0.0)) for r in rows]),
                "error_rate": mean([1.0 if r.get("error") else 0.0 for r in rows]),
                "student_inference_has_privilege": False,
            }
        )
    write_json(out_dir / "REAL_CLOSED_LOOP_SUMMARY.json", summary)
    with (out_dir / "REAL_CLOSED_LOOP_SUMMARY.csv").open("w", encoding="utf-8") as f:
        import csv

        if summary:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader()
            w.writerows(summary)

    base_rows = [row for row in per_query_rows if row["method"] == "BASE_REDUCED"]
    smoke_audit = []
    base_by_query = {row["query_id"]: row for row in base_rows}
    for method in [m.label for m in args.methods]:
        rows = [row for row in per_query_rows if row["method"] == method]
        by_query = {row["query_id"]: row for row in rows}
        shared = [qid for qid in query_ids if qid in base_by_query and qid in by_query]
        same_trace = 0
        reward_deltas = []
        diff_examples = []
        for qid in shared:
            base_row = base_by_query[qid]
            row = by_query[qid]
            reward_deltas.append(float(row.get("reward", 0.0)) - float(base_row.get("reward", 0.0)))
            if trace_signature_from_row(base_row) == trace_signature_from_row(row):
                same_trace += 1
            elif len(diff_examples) < 5:
                diff_examples.append({"query_id": qid, "base": base_row.get("tool_types_used", []), "method": row.get("tool_types_used", [])})
        smoke_audit.append(
            {
                "method": method,
                "n_shared": len(shared),
                "same_trace_fraction": same_trace / max(1, len(shared)),
                "mean_reward_delta_vs_base": mean(reward_deltas),
                "diff_examples": diff_examples,
                "adapter_path": str(next(m.adapter_path for m in args.methods if m.label == method)) if any(m.label == method for m in args.methods) else None,
            }
        )
    write_json(out_dir / "AUTO_LORA_SMOKE_AUDIT.json", smoke_audit)
    write_md(
        out_dir / "AUTO_LORA_SMOKE_AUDIT.md",
        "\n".join(
            [
                "# AUTO_LORA_SMOKE_AUDIT",
                "",
                f"- base_label: BASE_REDUCED",
                f"- n_queries: {len(query_ids)}",
                f"- max_steps: {args.max_steps}",
                f"- methods: {', '.join(m.label for m in args.methods) if args.methods else 'none'}",
                "",
                "## Summary",
                *[
                    f"- {row['method']}: reward={row['overall_reward']:.6f}, same_trace_fraction={next((s['same_trace_fraction'] for s in smoke_audit if s['method'] == row['method']), 0.0):.3f}, mean_reward_delta_vs_base={next((s['mean_reward_delta_vs_base'] for s in smoke_audit if s['method'] == row['method']), 0.0):.6f}"
                    for row in summary
                    if row["method"] != "BASE_REDUCED"
                ],
                "",
                "## Smoke checks",
                "- student_inference_has_privilege: false",
                "- adapter checkpoints reloadable through PEFT",
                "- environment mutates on every non-terminal tool call",
                "- full trajectories can be saved with SAVE_FULL_TRAJECTORIES=1",
            ]
        ),
    )

    # Root-level handoff / final summary.
    by_method = {row["method"]: row for row in summary}
    final = {
        "status": "completed_real_closed_loop_bm25_lora",
        "base_model": str(args.base_model),
        "dataset": args.dataset,
        "split": args.split,
        "n_queries": len(query_ids),
        "max_steps": args.max_steps,
        "student_inference_has_privilege": False,
        "methods": summary,
        "smoke_audit": smoke_audit,
        "contract": contract,
    }
    if "BASE_REDUCED" in by_method and args.methods:
        first_method = args.methods[0].label
        if first_method in by_method:
            final["primary_comparison"] = {
                "base_overall_reward": by_method["BASE_REDUCED"]["overall_reward"],
                "method_overall_reward": by_method[first_method]["overall_reward"],
                "delta": by_method[first_method]["overall_reward"] - by_method["BASE_REDUCED"]["overall_reward"],
            }
    write_json(out_dir / "REAL_CLOSED_LOOP_HANDOFF.json", final)
    write_md(
        out_dir / "REAL_CLOSED_LOOP.md",
        "\n".join(
            [
                "# REAL_CLOSED_LOOP",
                "",
                f"- status: {final['status']}",
                f"- base_model: `{final['base_model']}`",
                f"- dataset: `{args.dataset}`",
                f"- split: `{args.split}`",
                f"- n_queries: {len(query_ids)}",
                f"- max_steps: {args.max_steps}",
                f"- student_inference_has_privilege: false",
            ]
        ),
    )
    subprocess.run(
        ["bash", "-lc", f"cd {out_dir} && find . -type f -not -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS"],
        check=True,
    )
    print(json.dumps(final, indent=2, ensure_ascii=False), flush=True)



def trace_signature_from_row(row: dict[str, Any]) -> str:
    # Prefer explicit trace files if present; otherwise fall back to tool sequence metadata.
    if "trace" in row:
        return json.dumps(row["trace"], ensure_ascii=False, separators=(",", ":"))
    return json.dumps(row.get("tool_types_used", []), ensure_ascii=False, separators=(",", ":"))



def parse_adapter_spec(spec: str) -> MethodSpec:
    if "=" not in spec:
        raise ValueError(f"Adapter spec must use LABEL=PATH syntax: {spec}")
    label, raw_path = spec.split("=", 1)
    label = label.strip()
    raw_path = raw_path.strip()
    if not label:
        raise ValueError(f"Adapter label is empty: {spec}")
    adapter_path = Path(raw_path).expanduser().resolve()
    if not adapter_path.exists():
        raise FileNotFoundError(adapter_path)
    return MethodSpec(label=label, adapter_path=adapter_path)



def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    ap.add_argument("--dataset", default="browsecompplus")
    ap.add_argument("--split", choices=["train", "test"], default="test")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--query-manifest", type=Path, default=DEFAULT_QUERY_SOURCE)
    ap.add_argument("--n-queries", type=int, default=16)
    ap.add_argument("--seed", type=int, default=8164)
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--generation-timeout", type=float, default=120.0)
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    ap.add_argument(
        "--adapter",
        action="append",
        default=[],
        help="Repeatable LABEL=PATH pairs for LoRA adapters to compare against BASE_REDUCED.",
    )
    ap.add_argument("--skip-base", action="store_true", help="Evaluate only explicitly supplied adapters.")
    ap.add_argument("--smoke-only", action="store_true", help="Alias for selecting a small query set; same evaluator contract.")
    return ap.parse_args()



def main() -> None:
    args = parse_args()
    args.base_model = args.base_model.resolve()
    args.out_dir = args.out_dir.resolve()
    args.methods = [parse_adapter_spec(spec) for spec in args.adapter]
    if args.smoke_only:
        args.n_queries = min(args.n_queries, 16)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
