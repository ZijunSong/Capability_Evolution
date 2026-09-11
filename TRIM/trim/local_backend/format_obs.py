"""Observation text matching the original Search/Grep formatter."""

from __future__ import annotations

import os

# Full text cap when persisting to memory (not used for model-facing snippets).
DOC_TRUNCATION = int(os.environ.get("DOC_STORE_TRUNCATION", "51200000"))
DOCUMENT_ID_PREFIX = "# DOCUMENT ID:"

# Model-facing search/grep display budget (per doc + total).
DEFAULT_DISPLAY_SNIPPET_CHARS = int(os.environ.get("DISPLAY_SNIPPET_CHARS", "2048"))
MIN_DISPLAY_SNIPPET_CHARS = int(os.environ.get("MIN_DISPLAY_SNIPPET_CHARS", "256"))
DEFAULT_OBS_CHAR_BUDGET = int(os.environ.get("SEARCH_OBS_CHAR_BUDGET", "12000"))


def format_document_block(
    doc_id: str,
    text: str,
    tokens: int | None = None,
    *,
    snippet_chars: int = DEFAULT_DISPLAY_SNIPPET_CHARS,
) -> str:
    """Format one document for model-visible search/grep/read output."""
    body = text or ""
    total_len = len(body)
    if total_len > snippet_chars:
        body = (
            body[:snippet_chars]
            + f"\n... (document truncated, {total_len} chars total; "
            "use read_document/review_docs for full text)"
        )
    header = f"\n{DOCUMENT_ID_PREFIX} {doc_id}\n"
    if tokens is not None:
        header += f"({int(tokens)} tokens)\n"
    return header + body


def _minimal_document_header(doc_id: str) -> str:
    return (
        f"\n{DOCUMENT_ID_PREFIX} {doc_id}\n"
        "(content omitted — observation budget exceeded; use review_docs/read_document)"
    )


def format_search_observation(
    ids: list[str],
    documents: list[str],
    token_counts: list[int | None] | None = None,
    *,
    display_limit: int = 10,
    snippet_chars: int = DEFAULT_DISPLAY_SNIPPET_CHARS,
    total_char_budget: int = DEFAULT_OBS_CHAR_BUDGET,
) -> str:
    """Format ranked search/grep hits with per-document snippets and a total char budget."""
    if not ids:
        return "No results found"

    show_n = min(len(ids), int(display_limit))
    show_ids = ids[:show_n]
    show_docs = documents[:show_n]
    counts = (token_counts or [None] * len(ids))[:show_n]
    omitted_by_limit = max(0, len(ids) - show_n)

    per_doc_snippet = max(MIN_DISPLAY_SNIPPET_CHARS, int(snippet_chars))
    footer_reserve = 256
    budget = max(MIN_DISPLAY_SNIPPET_CHARS, int(total_char_budget) - footer_reserve)

    def build_blocks(snippet: int) -> tuple[list[str], int]:
        blocks = [
            format_document_block(doc_id, doc, tokens, snippet_chars=snippet)
            for doc_id, doc, tokens in zip(show_ids, show_docs, counts)
        ]
        total = sum(len(b) for b in blocks) + max(0, len(blocks) - 1)
        return blocks, total

    blocks, total = build_blocks(per_doc_snippet)
    while total > budget and per_doc_snippet > MIN_DISPLAY_SNIPPET_CHARS:
        per_doc_snippet = max(MIN_DISPLAY_SNIPPET_CHARS, per_doc_snippet // 2)
        blocks, total = build_blocks(per_doc_snippet)

    if total > budget:
        kept: list[str] = []
        running = 0
        dropped_with_content = 0
        for idx, block in enumerate(blocks):
            sep = 1 if kept else 0
            if running + sep + len(block) <= budget:
                kept.append(block)
                running += sep + len(block)
                continue
            minimal = _minimal_document_header(show_ids[idx])
            if running + sep + len(minimal) <= budget:
                kept.append(minimal)
                running += sep + len(minimal)
            dropped_with_content = len(blocks) - len(kept)
            break
        blocks = kept
        total = running
        budget_note = (
            f"[{dropped_with_content} result(s) omitted from display due to observation budget]"
            if dropped_with_content
            else None
        )
    else:
        budget_note = None

    footers: list[str] = []
    if omitted_by_limit:
        footers.append(
            f"[{omitted_by_limit} more result(s) not shown; display_limit={display_limit}]"
        )
    if budget_note:
        footers.append(budget_note)

    text = "\n".join(blocks)
    if footers:
        text = text + "\n" + "\n".join(footers)
    return text
