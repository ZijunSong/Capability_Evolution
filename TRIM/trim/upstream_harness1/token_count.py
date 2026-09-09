"""Eval/train token counting for budget markers and local read truncation."""

from __future__ import annotations

from typing import Callable

TOKEN_COUNT_MODE = "whitespace_words"


def whitespace_token_counter(text: str) -> int:
    return len(str(text).split())


def counter_from_model_path(model_path: str | None) -> tuple[Callable[[str], int], str]:
    """Return a counter aligned with the served actor tokenizer."""
    from trim.eval.model_tokenizer import load_model_encoding

    enc = load_model_encoding(str(model_path or ""))

    def _count(text: str) -> int:
        return len(enc.encode(str(text or "")))

    return _count, str(enc.encoding_name)


def resolve_token_counter(model_path: str | None = None) -> tuple[Callable[[str], int], str]:
    """Prefer model tokenizer counting; fall back to whitespace only when loading fails."""
    path = str(model_path or "").strip()
    if path:
        try:
            return counter_from_model_path(path)
        except Exception:
            pass
    return whitespace_token_counter, TOKEN_COUNT_MODE


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
