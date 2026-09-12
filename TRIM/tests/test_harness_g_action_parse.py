"""Harness-G action parse + Qwen JSON tool blobs."""

from __future__ import annotations

from trim.eval.harness_g_runtime import parse_harness_g_action
from trim.eval.model_tokenizer import parse_qwen_tool_call
from trim.training.parse_rollout_action import parse_generated_action


def test_parse_json_tool_key_init():
    action, ok = parse_harness_g_action('{"tool": "init"}<|im_end|>')
    assert ok is True
    assert action == {"name": "init", "arguments": {}}


def test_parse_json_name_arguments_select():
    action, ok = parse_harness_g_action('{"name":"select","arguments":{"sid":"d1:s0"}}')
    assert ok is True
    assert action["name"] == "select"
    assert action["arguments"]["sid"] == "d1:s0"


def test_parse_qwen_tool_blob_is_legal():
    parsed = parse_qwen_tool_call('{"tool": "init"}')
    assert parsed.parsed is True
    assert parsed.legal is True
    assert parsed.tool_name == "init"


def test_parse_generated_action_qwen_json_init():
    mask = {
        "answer_with": False,
        "bridge_entities": False,
        "entity_synonyms": False,
        "sentence_neighbors": False,
        "hybrid_init_retrieve": False,
        "invalid_target_filter": False,
        "lookup_dedup": False,
        "snc_frontier": False,
    }
    action, ok = parse_generated_action('{"tool":"init"}', None, enc=None, harness_mask=mask)
    assert ok is True
    assert action["name"] == "init"


def test_parse_harmony_select_still_works():
    mask = {
        "answer_with": False,
        "bridge_entities": False,
        "entity_synonyms": False,
        "sentence_neighbors": False,
        "hybrid_init_retrieve": False,
        "invalid_target_filter": False,
        "lookup_dedup": False,
        "snc_frontier": False,
    }
    text = (
        "<|start|>assistant to=functions.select<|channel|>commentary "
        '<|constrain|>json<|message|>{"sid": "d1:s0"}<|call|>'
    )
    action, ok = parse_generated_action(text, None, enc=None, harness_mask=mask)
    assert ok is True
    assert action["name"] == "select"
