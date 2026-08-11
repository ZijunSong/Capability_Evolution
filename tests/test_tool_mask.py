from __future__ import annotations

from scape.training.tool_mask import (
    build_tool_token_mask,
    extract_argument_spans,
    extract_end_search_spans,
    extract_tool_name_spans,
)


SAMPLE = """\
to=search
{"query": "who invented x", "top_k": 5}
to=curate
{"add_ids": ["d1"], "remove_ids": []}
end_search
"""


def test_tool_name_span():
    spans = extract_tool_name_spans(SAMPLE)
    names = [s.text for s in spans]
    assert "search" in names
    assert "curate" in names
    assert all(s.kind == "tool_name" for s in spans)


def test_argument_key_span():
    keys, _vals = extract_argument_spans(SAMPLE)
    key_texts = {s.text for s in keys}
    assert "query" in key_texts
    assert "add_ids" in key_texts
    assert all(s.kind == "argument_key" for s in keys)


def test_argument_value_span():
    _keys, vals = extract_argument_spans(SAMPLE)
    val_texts = " ".join(s.text for s in vals)
    assert "who invented x" in val_texts or "d1" in val_texts
    assert all(s.kind == "argument_value" for s in vals)


def test_end_search_span():
    spans = extract_end_search_spans(SAMPLE)
    assert any(s.text == "end_search" for s in spans)
    mask = build_tool_token_mask(SAMPLE)
    kinds = {s.kind for s in mask}
    assert {"tool_name", "argument_key", "argument_value", "end_search"} <= kinds
