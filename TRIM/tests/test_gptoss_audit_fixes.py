"""Regression tests for GPT-OSS bcplus_test_50 audit fixes (2026-09-11)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
HARNESS_ROOT = ROOT / "external" / "harness-1"
if str(HARNESS_ROOT) not in sys.path:
    sys.path.insert(0, str(HARNESS_ROOT))

from harness.tools import SEARCH_CORPUS_SCHEMA  # noqa: E402
from trim.eval.harness1_api_eval import _env_turn_count  # noqa: E402
from trim.upstream_harness1.api_adapter import parse_chat_completion  # noqa: E402
from trim.upstream_harness1.env_bridge import _validate_tool_params  # noqa: E402


def test_parse_nested_search_corpus_in_content():
    body = json.dumps(
        {
            "search_corpus": {
                "query": "W L article updated 2023 quotes F H 2019 study",
            }
        }
    )
    parsed = parse_chat_completion(
        {"choices": [{"finish_reason": "stop", "message": {"content": body}}]}
    )
    assert parsed.ok is False
    assert parsed.protocol_error == "tool_call_in_content"


def test_parse_type_fan_out_search_in_content():
    body = json.dumps({"type": "fan_out_search", "queries": ["a", "b"]})
    parsed = parse_chat_completion(
        {"choices": [{"finish_reason": "stop", "message": {"content": body}}]}
    )
    assert parsed.ok is False
    assert parsed.protocol_error == "tool_call_in_content"


def test_parse_recipient_functions_search_in_content():
    parsed = parse_chat_completion(
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": '{"recipient":"functions.search_corpus","query":"foo"}'},
                }
            ]
        }
    )
    assert parsed.ok is False
    assert parsed.protocol_error == "tool_call_in_content"


def test_validate_tool_params_requires_add_ids():
    tool = MagicMock()
    tool.tool_schema = SEARCH_CORPUS_SCHEMA
    _validate_tool_params(tool, {"query": "foo"})

    curate = MagicMock()
    curate.tool_schema = MagicMock(
        name="curate",
        required=["add_ids"],
        parameters={"add_ids": {"type": "array"}, "remove_ids": {"type": "array"}},
    )
    try:
        _validate_tool_params(curate, {"remove_ids": []})
        raised = False
    except ValueError as exc:
        raised = True
        assert "missing required parameter: add_ids" in str(exc)
    assert raised


def test_env_turn_count_does_not_fallback_to_generation_attempts():
    assert _env_turn_count({"num_turns": 0, "n_turns": 6}) == 0.0
    assert _env_turn_count({"num_turns": 3}) == 3.0


def _assert_rejects_toolish_content(content: str) -> None:
    parsed = parse_chat_completion(
        {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}
    )
    assert parsed.ok is False
    assert parsed.protocol_error == "tool_call_in_content"


def test_parse_rejects_bare_curate_json():
    _assert_rejects_toolish_content(json.dumps({"add_ids": ["123_0"], "remove_ids": []}))


def test_parse_rejects_bare_queries_json():
    _assert_rejects_toolish_content(json.dumps({"queries": ["foo", "bar"]}))


def test_parse_rejects_search_corpus_colon_prefix():
    _assert_rejects_toolish_content('search_corpus: {"query": "example query"}')


def test_parse_rejects_curate_function_call_syntax():
    _assert_rejects_toolish_content("curate({add_ids: ['123_0', '456_0']})")


def test_parse_rejects_harmony_call_marker_only():
    _assert_rejects_toolish_content("<|call|>")


def test_parse_rejects_functions_pkg_search_corpus():
    _assert_rejects_toolish_content("pkgs=functions search_corpus query=foo<|call|>")


def test_parse_rejects_add_ids_only_codeblock():
    _assert_rejects_toolish_content("```json\n{\"add_ids\": [\"123_0\"]}\n```")
