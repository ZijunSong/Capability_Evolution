"""Unit tests for weak-policy action repair (harness v2)."""

from harness.action_repair import (
    extract_document_ids_from_text,
    normalize_tool_params,
    should_block_early_end,
)


def test_block_early_end_too_soon():
    block, _ = should_block_early_end(
        turn=2, n_curated=0, n_pool=10, min_turns=8, min_curated=1
    )
    assert block is True


def test_allow_end_after_min_turns_with_curated():
    block, _ = should_block_early_end(
        turn=10, n_curated=3, n_pool=10, min_turns=8, min_curated=1
    )
    assert block is False


def test_normalize_search_alias():
    assert normalize_tool_params("search_corpus", {"q": "hello"})["query"] == "hello"


def test_normalize_fan_out_string_queries():
    out = normalize_tool_params("fan_out_search", {"queries": "single query"})
    assert out["queries"] == ["single query"]


def test_extract_document_xml():
    ids = extract_document_ids_from_text(
        "<Document id=abc123>\n<Justification>x</Justification>\n</Document>"
    )
    assert ids == ["abc123"]
