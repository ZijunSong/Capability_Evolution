from __future__ import annotations

from trim.eval.harmony_runtime import _canonicalize_tool_name, parse_harmony_tool_call
from trim.training.four_cell_runtime import parse_generated_action


def test_canonicalize_channel_leak_into_curate():
    assert _canonicalize_tool_name("curate?commentary") == "curate"
    assert _canonicalize_tool_name("curate!commentary") == "curate"
    assert _canonicalize_tool_name("curate.commentary") == "curate"
    assert _canonicalize_tool_name("curatecommentary") == "curate"
    assert _canonicalize_tool_name("curate=commentary") == "curate"
    assert _canonicalize_tool_name("curate...commentary") == "curate"
    assert _canonicalize_tool_name("curate…commentary") == "curate"
    assert _canonicalize_tool_name("curate<|start|>commentary") == "curate"
    assert _canonicalize_tool_name("read_document…commentary") == "read_document"
    assert _canonicalize_tool_name("search_corpus") == "search_corpus"
    assert _canonicalize_tool_name("functions.curate") == "curate"


def test_canonicalize_recovers_one_edit_typos():
    assert _canonicalize_tool_name("curute") == "curate"
    assert _canonicalize_tool_name("curtate") == "curate"
    assert _canonicalize_tool_name("curite") == "curate"
    assert _canonicalize_tool_name("seach_corpus") == "search_corpus"


def test_canonicalize_keeps_true_unknown_names():
    assert _canonicalize_tool_name("not_a_real_tool") == "not_a_real_tool"
    assert _canonicalize_tool_name("open_document") == "open_document"
    assert _canonicalize_tool_name("curure") == "curure"
    assert _canonicalize_tool_name("attempt_curture") == "attempt_curture"


def test_parse_regex_recovers_leaked_channel_as_legal_curate():
    text = (
        "<|channel|>analysis<|message|>Need to keep docs.<|end|>"
        "<|start|>assistant to=functions.curate?commentary<|channel|>commentary "
        '<|constrain|>json<|message|>{"add_ids": ["12", "34"]}<|call|>'
    )
    parsed = parse_harmony_tool_call(text)
    assert parsed.parsed is True
    assert parsed.legal is True
    assert parsed.tool_name == "curate"
    assert parsed.arguments == {"add_ids": ["12", "34"]}
    action, ok = parse_generated_action(text, None, enc=None)
    assert ok is True
    assert action["name"] == "curate"
    assert action["arguments"]["add_ids"] == ["12", "34"]


def test_parse_glued_curatecommentary_is_legal():
    text = (
        "<|start|>assistant to=functions.curatecommentary<|channel|>commentary "
        '<|constrain|>json<|message|>{"add_ids": ["7"]}<|call|>'
    )
    parsed = parse_harmony_tool_call(text)
    assert parsed.tool_name == "curate"
    assert parsed.legal is True


def test_parse_unknown_tool_stays_illegal():
    text = (
        "<|start|>assistant to=functions.not_a_real_tool<|channel|>commentary "
        '<|constrain|>json<|message|>{"query": "x"}<|call|>'
    )
    parsed = parse_harmony_tool_call(text)
    assert parsed.parsed is True
    assert parsed.legal is False
    assert parsed.tool_name == "not_a_real_tool"
