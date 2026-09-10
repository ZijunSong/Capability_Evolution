#!/usr/bin/env python3
"""Offline check for openai_harmony==0.0.8 HarmonyGptOss + tiktoken o200k_harmony."""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TIK = HERE / "share" / "tiktoken"
RS = HERE / "tiktoken_rs_cache"
PY = HERE / "tiktoken_cache"

os.environ.setdefault("TIKTOKEN_ENCODINGS_BASE", str(TIK))
os.environ.setdefault("TIKTOKEN_RS_CACHE_DIR", str(RS))
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(PY))

# Block accidental Azure hits during this check if the caller wants:
#   HTTP_PROXY=http://127.0.0.1:1 HTTPS_PROXY=http://127.0.0.1:1

def main() -> int:
    print("TIKTOKEN_ENCODINGS_BASE =", os.environ.get("TIKTOKEN_ENCODINGS_BASE"))
    print("TIKTOKEN_RS_CACHE_DIR   =", os.environ.get("TIKTOKEN_RS_CACHE_DIR"))
    print("TIKTOKEN_CACHE_DIR      =", os.environ.get("TIKTOKEN_CACHE_DIR"))
    for name in ("o200k_base.tiktoken", "o200k_harmony.tiktoken", "o200k_harmony.special_tokens.json"):
        p = TIK / name
        print(f"  {name}: exists={p.is_file()} bytes={p.stat().st_size if p.is_file() else 0}")

    import openai_harmony
    from openai_harmony import HarmonyEncodingName, load_harmony_encoding

    print("openai_harmony", getattr(openai_harmony, "__file__", "?"))
    enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    print("loaded HarmonyGptOss tokenizer_name=", getattr(enc, "name", None))

    def tok(s: str) -> int:
        ids = enc.encode(s, allowed_special="all") if hasattr(enc, "encode") else None
        if ids is None:
            # rust binding
            ids = list(enc.encode(s, allowed_special={s})) if False else None
        return ids

    # openai_harmony Python API: encode via render or encode
    checks = {
        "<|call|>": 200012,
        "<|return|>": 200002,
        "<|start|>": 200006,
        "<|message|>": 200008,
        "<|channel|>": 200005,
        "<|end|>": 200007,
        "<|constrain|>": 200003,
    }
    # Prefer encode if present
    encode = getattr(enc, "encode", None)
    ok = True
    if encode is not None:
        for s, expected in checks.items():
            try:
                ids = list(encode(s, allowed_special="all"))
            except TypeError:
                try:
                    ids = list(encode(s))
                except Exception as e:
                    print("FAIL encode", s, e)
                    ok = False
                    continue
            got = ids[0] if ids else None
            status = "OK" if got == expected else "FAIL"
            if got != expected:
                ok = False
            print(f"  {status} {s} -> {got} (expected {expected}) raw={ids}")
    else:
        print("WARN: HarmonyEncoding has no encode(); skip token-id probe")

    stops = []
    if hasattr(enc, "stop_tokens_for_assistant_actions"):
        stops = list(enc.stop_tokens_for_assistant_actions())
        print("stop_tokens_for_assistant_actions", stops)
        for needed in (200002, 200012):
            if needed not in [int(x) for x in stops]:
                print("FAIL missing stop token", needed)
                ok = False

    import tiktoken
    tenc = tiktoken.get_encoding("o200k_harmony")
    print("tiktoken o200k_harmony n_vocab", tenc.n_vocab)
    if tenc.n_vocab != 201088:
        print("FAIL n_vocab", tenc.n_vocab)
        ok = False
    for s, expected in checks.items():
        got = tenc.encode(s, allowed_special="all")
        if got != [expected]:
            print("FAIL tiktoken", s, got, "expected", [expected])
            ok = False
        else:
            print(f"  OK tiktoken {s} -> {got}")

    if ok:
        print("PASS: HarmonyGptOss + o200k_harmony special tokens loaded offline")
        return 0
    print("FAIL")
    return 1

if __name__ == "__main__":
    sys.exit(main())
