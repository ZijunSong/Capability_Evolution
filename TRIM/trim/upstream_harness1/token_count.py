"""Eval/train token counting for budget markers and local read truncation.

This is the counting convention used when no model tokenizer is loaded.
It is not Harmony and not a Qwen chat-template length.
"""

from __future__ import annotations

TOKEN_COUNT_MODE = "whitespace_words"


def whitespace_token_counter(text: str) -> int:
    return len(str(text).split())


def prefix_to_token_budget(text: str, budget: int, counter) -> str:
    """Keep a prefix that fits ``budget`` tokens. Empty only if budget <= 0."""
    raw = str(text or "")
    limit = int(budget)
    if limit <= 0 or not raw:
        return ""
    if counter(raw) <= limit:
        return raw
    lo, hi = 0, len(raw)
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        chunk = raw[:mid]
        if counter(chunk) <= limit:
            best = chunk
            lo = mid + 1
        else:
            hi = mid - 1
    return best
