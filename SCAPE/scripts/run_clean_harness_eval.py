#!/usr/bin/env python3
"""Full-Harness smoke/eval for raw or Clean-SFT gpt-oss-20b (LOCAL_COMPAT_ONLY)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scape.training.clean_sft import CANONICAL_TOOLS, parse_tool_name

SMOKE_PROMPTS = [
    (
        "search",
        "You are a retrieval subagent. Emit a Harness-1 tool call only.\n"
        "User query: What is the filing date of Apple's FY2023 10-K?\n"
        "Call fan_out_search or search_corpus now.\n",
    ),
    (
        "read",
        "You already found doc_id=d12. Emit a read_document tool call for d12.\n",
    ),
    (
        "curate",
        "You have documents d12 and d18 in the pool. Emit a curate tool call "
        "that adds d12 with high importance.\n",
    ),
    (
        "end_search",
        "You have enough evidence. Emit an end_search tool call with a short reason.\n",
    ),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--base-model", default="/data/ppnm/models/gpt-oss-20b")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--tag", default="eval")
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    from scape.training.clean_sft import load_causal_lm
    import torch

    tok, model = load_causal_lm(
        args.model_path,
        device_map=f"cuda:{args.gpu}",
        base_model=args.base_model,
    )
    device = next(model.parameters()).device

    generations = []
    n_parse = 0
    n_legal = 0
    smoke_ok = {}
    for key, prompt in SMOKE_PROMPTS:
        ids = tok.encode(prompt, add_special_tokens=False)
        inp = torch.tensor([ids], device=device)
        with torch.no_grad():
            gen = model.generate(
                inp,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        text = tok.decode(gen[0, len(ids) :].tolist(), skip_special_tokens=False)
        name = parse_tool_name(text)
        parsed = name is not None
        legal = name in CANONICAL_TOOLS
        if parsed:
            n_parse += 1
        if legal:
            n_legal += 1
        want = {
            "search": {"fan_out_search", "search_corpus", "grep_corpus"},
            "read": {"read_document"},
            "curate": {"curate"},
            "end_search": {"end_search"},
        }[key]
        smoke_ok[key] = bool(name in want)
        generations.append(
            {
                "key": key,
                "tool_name": name,
                "parsed": parsed,
                "legal": legal,
                "text_head": text[:800],
            }
        )
    n = len(SMOKE_PROMPTS)
    summary = {
        "tag": args.tag,
        "model_path": args.model_path,
        "n": n,
        "tool_call_parse_rate": n_parse / n,
        "legal_tool_rate": n_legal / n,
        "invalid_tool_rate": 1.0 - (n_legal / n),
        "smoke_search_read_curate_end": smoke_ok,
        "smoke_all_ok": all(smoke_ok.values()),
        "LOCAL_COMPAT_ONLY": True,
        "legacy_scope_path_used": False,
        "generations": generations,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "DONE").write_text("ok\n")
    print(json.dumps({k: summary[k] for k in summary if k != "generations"}, indent=2))
    del model
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
