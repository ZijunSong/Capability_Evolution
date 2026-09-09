"""Observation text matching the original Search/Grep formatter."""

from __future__ import annotations

DOC_TRUNCATION = 51200000
DOCUMENT_ID_PREFIX = "# DOCUMENT ID:"


def format_document_block(doc_id: str, text: str, tokens: int | None = None) -> str:
    suffix = f" ({tokens} tokens)" if tokens is not None else ""
    body = (text or "")[:DOC_TRUNCATION]
    return f"\n{DOCUMENT_ID_PREFIX} {doc_id}{suffix} \n{body}"


def format_search_observation(
    ids: list[str],
    documents: list[str],
    token_counts: list[int | None] | None = None,
    *,
    display_limit: int = 10,
) -> str:
    if not ids:
        return "No results found"
    counts = token_counts or [None] * len(ids)
    formatted = [
        format_document_block(doc_id, doc, tokens)
        for doc_id, doc, tokens in zip(ids, documents, counts)
    ][: int(display_limit)]
    return "\n".join(formatted)
