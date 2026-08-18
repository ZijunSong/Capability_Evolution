#!/usr/bin/env python3
"""Harmony / tool-call runtime audit + parser contract tests (CPU)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scape.eval.harmony_runtime import (  # noqa: E402
    CANONICAL_TOOLS,
    build_first_turn_prompt_ids,
    canonical_examples,
    load_harmony_enc,
    parse_harmony_tool_call,
    run_parser_contract_tests,
    stop_ids_for_tool_actions,
)


def audit_chat_template(model_path: str) -> dict:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tmpl = getattr(tok, "chat_template", None) or ""
    special = list(getattr(tok, "additional_special_tokens", None) or [])
    enc = load_harmony_enc()
    probe = "Need a corpus search."
    hf_ids = tok.encode(probe, add_special_tokens=False)
    try:
        hy_ids = list(enc.encode(probe))
    except Exception as exc:  # noqa: BLE001
        hy_ids = []
        hy_err = str(exc)[:200]
    else:
        hy_err = None
    match = hf_ids == hy_ids and bool(hf_ids)
    stop = stop_ids_for_tool_actions(enc)
    return {
        "model_path": model_path,
        "tokenizer_class": type(tok).__name__,
        "has_chat_template": bool(tmpl),
        "chat_template_mentions_commentary": "commentary" in tmpl,
        "chat_template_mentions_functions": "functions" in tmpl or "to=" in tmpl,
        "n_special_tokens": len(special),
        "hf_harmony_plain_text_id_match": match,
        "hf_n": len(hf_ids),
        "harmony_n": len(hy_ids),
        "harmony_encode_error": hy_err,
        "stop_tokens_for_assistant_actions": stop,
        "stop_decoded": ["<|call|>", "<|return|>"],
        "must_not_stop_on_end": True,
        "pad_token_id": tok.pad_token_id,
        "eos_token_id": tok.eos_token_id,
    }


def audit_prompt_contract() -> dict:
    enc = load_harmony_enc()
    ids = build_first_turn_prompt_ids("What is the filing date of Apple's FY2023 10-K?", enc=enc)
    text = enc.decode_utf8(ids)
    checks = {
        "n_tokens": len(ids),
        "has_system": "<|start|>system" in text,
        "has_developer_tools": "namespace functions" in text or "type search_corpus" in text,
        "has_fan_out": "fan_out_search" in text,
        "has_end_search": "end_search" in text,
        "has_user_query": "Apple" in text and "10-K" in text,
        "ends_with_assistant_start": text.rstrip().endswith("<|start|>assistant"),
        "does_not_include_future_reward": "gold answer" not in text.lower(),
        "tail": text[-180:],
        "head": text[:400],
    }
    checks["ok"] = all(
        checks[k]
        for k in (
            "has_system",
            "has_developer_tools",
            "has_fan_out",
            "has_end_search",
            "has_user_query",
            "ends_with_assistant_start",
        )
    )
    return checks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model-path", default="/data/ppnm/models/gpt-oss-20b")
    args = ap.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    tests = run_parser_contract_tests()
    tmpl = audit_chat_template(args.model_path)
    prompt = audit_prompt_contract()
    examples = {k: parse_harmony_tool_call(v).to_dict() for k, v in canonical_examples().items()}
    payload = {
        "parser_contract": tests,
        "tokenizer_chat_template": tmpl,
        "first_turn_prompt_contract": prompt,
        "example_parses": examples,
        "canonical_tools": list(CANONICAL_TOOLS),
        "runtime_notes": {
            "sft_path": "openai_harmony.render_conversation + build_context",
            "eval_must_use_same_path": True,
            "previous_n4_eval_used_raw_english_prompt": True,
            "stop_on_call_or_return_not_end": True,
            "strict_parser_no_prose_fallback": True,
        },
    }
    (out / "HARMONY_RUNTIME_TESTS.json").write_text(json.dumps(payload, indent=2) + "\n")
    md = [
        "# HARMONY_RUNTIME_AUDIT",
        "",
        "Clean gpt-oss / Harness-1 tool-channel contract (2026-08-17).",
        "",
        "## Parser contract tests",
        "",
        f"- all_ok: **{tests['all_ok']}** ({tests['n_pass']}/{tests['n']})",
        "",
    ]
    for r in tests["rows"]:
        md.append(f"- `{r['id']}` expect={r['expect']} ok={r['ok']} parsed={r['parsed']} legal={r['legal']} name={r['tool_name']}")
    md += [
        "",
        "## Tokenizer / chat-template",
        "",
        f"- path: `{tmpl['model_path']}`",
        f"- has_chat_template: {tmpl['has_chat_template']}",
        f"- HF vs Harmony plain-text id match: {tmpl['hf_harmony_plain_text_id_match']}",
        f"- stop_tokens_for_assistant_actions: `{tmpl['stop_tokens_for_assistant_actions']}` → call/return",
        "",
        "## First-turn prompt (must match SFT)",
        "",
        f"- ok: **{prompt['ok']}**",
        f"- n_tokens: {prompt['n_tokens']}",
        f"- system: {prompt['has_system']}",
        f"- developer tools: {prompt['has_developer_tools']}",
        f"- ends with `<|start|>assistant`: {prompt['ends_with_assistant_start']}",
        "",
        "## Previous n=4 eval diagnosis",
        "",
        "The 0814 n=4 smoke (`run_clean_harness_eval.py`) fed **raw English instructions**",
        "through `tokenizer.encode`, without Harmony system/developer/tool schema, and",
        "used a greedy prose fallback parser. That is a **contract bug**, not evidence",
        "that FULL SFT never learned Harmony syntax. FULL s42 already emitted",
        "`assistant to=functions.search_corpus` even under the broken prompt.",
        "",
        "This round re-evaluates with `build_context` + `render_conversation_for_completion`",
        "and stops on `<|call|>` / `<|return|>` only.",
        "",
    ]
    (out / "HARMONY_RUNTIME_AUDIT.md").write_text("\n".join(md) + "\n")
    print(json.dumps({"all_ok": tests["all_ok"] and prompt["ok"], "tests": tests["n_pass"], "prompt_ok": prompt["ok"]}))
    return 0 if tests["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
