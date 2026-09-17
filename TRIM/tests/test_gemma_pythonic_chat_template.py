"""Gemma-3 pythonic chat template must accept Harness-1 retry transcripts."""

from __future__ import annotations

import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from trim.eval.model_profiles import _GEMMA3_PYTHONIC_CHAT_TEMPLATE

_TEMPLATE_PATH = Path(_GEMMA3_PYTHONIC_CHAT_TEMPLATE)


def _render(messages: list[dict], *, tools: list[dict] | None = None) -> str:
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_PATH.parent)),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )
    env.filters["tojson"] = lambda value, indent=None: json.dumps(value, indent=indent)
    env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(ValueError(msg))
    tmpl = env.get_template(_TEMPLATE_PATH.name)
    return tmpl.render(
        bos_token="<bos>",
        messages=messages,
        tools=tools,
        add_generation_prompt=True,
    )


_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_corpus",
            "description": "Search corpus",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
]


def test_template_allows_consecutive_user_format_retry():
    text = _render(
        [
            {"role": "user", "content": "Find documents about smoke tests."},
            {
                "role": "user",
                "content": "Your previous response could not be parsed as a valid tool call.",
            },
        ],
        tools=_TOOLS,
    )
    assert "MUST respond with a python list" in text
    assert "Find documents about smoke tests." in text
    assert "could not be parsed" in text
    assert text.endswith("<start_of_turn>model\n")


def test_template_renders_assistant_tool_call_and_tool_result():
    text = _render(
        [
            {"role": "user", "content": "Search the corpus."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search_corpus",
                            "arguments": {"query": "smoke test"},
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "doc-1: hello"},
        ],
        tools=_TOOLS,
    )
    assert "search_corpus(query=" in text
    assert "<tool_response>" in text
    assert "doc-1: hello" in text
