"""Fingerprint the fixed H0 that every RSI round reuses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from trim.adapters.components import zero_mask
from trim.rsi.canonical import sha256_json, sha256_text
from trim.training.tool_mask import ACTION_ARG_SCHEMAS, legal_tool_names

CONTRACT_SCHEMA = "trim_base_contract_v1"
H0_TOOLS = (
    "search_corpus",
    "grep_corpus",
    "read_document",
    "curate",
    "end_search",
)


def build_base_contract(
    *,
    code_commit: str,
    harness: str = "Harness-1",
    protocol_version: str = CONTRACT_SCHEMA,
    tokenizer_sha256: str = "",
    chat_template_sha256: str = "",
    retrieval_backend: str = "local_bm25",
    corpus_version: str = "",
    index_sha256: str = "",
    top_k: int = 10,
    max_model_len: int = 8192,
    max_new_tokens: int = 2048,
    max_turns: int = 40,
    tool_timeout_s: float = 30.0,
    reward_protocol: str,
    metric_protocol: str,
    startup_prompt: str = "",
) -> dict[str, Any]:
    mask = zero_mask(harness)
    tools = [name for name in legal_tool_names(harness_mask=mask) if name in H0_TOOLS]
    schema = {name: sorted(ACTION_ARG_SCHEMAS.get(name, ())) for name in tools}
    if "verify" in tools:
        raise ValueError("H0 contract included verify; zero mask must not imply that tool")
    prompt_hash = sha256_text(startup_prompt)
    body = {
        "schema_version": CONTRACT_SCHEMA,
        "harness": harness,
        "protocol_version": protocol_version,
        "code_commit": code_commit,
        "tokenizer_sha256": tokenizer_sha256,
        "chat_template_sha256": chat_template_sha256,
        "action_parser": "trim.training.action_codec",
        "tool_names": tools,
        "tool_schema": schema,
        "visible_fields": ["query", "tool_history", "pool", "curated", "observations"],
        "serializable_state_fields": ["pool", "curated", "tool_history", "step", "ended"],
        "context_builder": "upstream_or_local_legacy_selected_at_launch",
        "retrieval_backend": retrieval_backend,
        "corpus_version": corpus_version,
        "index_sha256": index_sha256,
        "top_k": int(top_k),
        "max_model_len": int(max_model_len),
        "max_new_tokens": int(max_new_tokens),
        "max_turns": int(max_turns),
        "tool_timeout_s": float(tool_timeout_s),
        "zero_tool_schema_sha256": sha256_json(schema),
        "startup_prompt_sha256": prompt_hash,
        "reward_protocol": reward_protocol,
        "metric_protocol": metric_protocol,
        "advanced_components": "all_off",
    }
    body["contract_sha256"] = sha256_json({k: v for k, v in body.items() if k != "contract_sha256"})
    return body


def assert_base_contract(payload: Mapping[str, Any], *, expected_sha256: str | None = None) -> None:
    if payload.get("schema_version") != CONTRACT_SCHEMA:
        raise ValueError("base contract schema mismatch")
    if "verify" in set(payload.get("tool_names") or []):
        raise ValueError("base contract lists verify as an H0 tool")
    required = ("search_corpus", "read_document", "curate", "end_search")
    missing = [name for name in required if name not in set(payload.get("tool_names") or [])]
    if missing:
        raise ValueError(f"base contract is missing H0 tools: {missing}")
    if payload.get("reward_protocol") == payload.get("metric_protocol") and "reward" in str(payload.get("metric_protocol")):
        raise ValueError("reward protocol and metric protocol must be recorded separately")
    digest = payload.get("contract_sha256")
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("base contract hash does not match the campaign pin")


def write_contract(path: str | Path, payload: Mapping[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def smoke_record(*, init_ok: bool, search_ok: bool, read_ok: bool, curate_ok: bool, end_ok: bool, score_ok: bool) -> dict[str, Any]:
    mechanical = all([init_ok, search_ok, read_ok, curate_ok, end_ok, score_ok])
    return {
        "mechanical_ok": mechanical,
        "checks": {
            "init": init_ok,
            "search": search_ok,
            "read": read_ok,
            "curate": curate_ok,
            "end": end_ok,
            "score": score_ok,
        },
        "note": "A zero task score does not by itself fail the mechanical contract.",
    }
