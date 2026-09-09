"""V01: same normalized action → same env semantics; protocol boundaries."""

from __future__ import annotations

from trim.upstream_harness1.api_adapter import PROTOCOL_CHAT_COMPLETIONS_V1, parse_chat_completion
from trim.upstream_harness1.model_serve import KNOWN_BASE_MODELS, vllm_serve_hint
from trim.upstream_harness1.v8d_flags import EVALUATION_PATH_LEGACY_LOCAL, EVALUATION_PATH_UPSTREAM_API


def test_evaluation_paths_are_named_not_merged():
    assert EVALUATION_PATH_UPSTREAM_API == "upstream_api"
    assert EVALUATION_PATH_LEGACY_LOCAL == "legacy_local"
    assert EVALUATION_PATH_UPSTREAM_API != EVALUATION_PATH_LEGACY_LOCAL


def test_known_actor_models_share_chat_protocol():
    assert "openai/gpt-oss-20b" in KNOWN_BASE_MODELS
    assert "Qwen/Qwen3-4B-Instruct-2507" in KNOWN_BASE_MODELS
    assert "pat-jj/harness-1" in KNOWN_BASE_MODELS
    hint = vllm_serve_hint("openai/gpt-oss-20b")
    assert "--enable-auto-tool-choice" in hint
    assert PROTOCOL_CHAT_COMPLETIONS_V1 == "chat_completions_v1"


def test_parse_and_train_action_share_tool_name():
    from trim.training.parse_rollout_action import parse_generated_action
    from trim.upstream_harness1.api_adapter import parse_chat_completion

    payload = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "tool_calls": [
                        {
                            "id": "c1",
                            "function": {"name": "end_search", "arguments": "{}"},
                        }
                    ]
                },
            }
        ]
    }
    api = parse_chat_completion(payload)
    text = (
        "<|start|>assistant to=functions.end_search<|channel|>commentary "
        "<|constrain|>json<|message|>{}<|call|>"
    )
    train, ok = parse_generated_action(text, None, enc=None, teacher_mode=True)
    assert api.ok and ok
    assert api.tool_calls[0]["name"] == train["name"] == "end_search"


def test_end_search_and_format_error_are_distinct_parses():
    end = parse_chat_completion(
        {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "tool_calls": [
                            {
                                "id": "c1",
                                "function": {"name": "end_search", "arguments": "{}"},
                            }
                        ]
                    },
                }
            ]
        }
    )
    assert end.ok and end.tool_calls[0]["name"] == "end_search"
    bad = parse_chat_completion({"choices": []})
    assert bad.ok is False


def test_schema_presence_for_all_default_zero():
    from trim.upstream_harness1.v8d_flags import actual_eval_mask, v8d_env_from_mask

    all_env = v8d_env_from_mask(actual_eval_mask([], preset="all", harness="Harness-1"))
    zero_env = v8d_env_from_mask(actual_eval_mask([], preset="zero", harness="Harness-1"))
    default_env = v8d_env_from_mask(actual_eval_mask([], preset="default", harness="Harness-1"))
    assert all_env["V8D_VERIFY_TOOL"] == "1"
    assert zero_env["V8D_VERIFY_TOOL"] == "0"
    assert default_env["V8D_VERIFY_TOOL"] == "1"
    assert default_env["V8D_CHUNK_NEIGHBORS"] == "0"
    assert all_env["V8D_CHUNK_NEIGHBORS"] == "1"
    # Import-time schema depends on these env values in an isolated process.
    assert all_env["V8D_IMPORTANCE_TAGGING"] == "1"
    assert zero_env["V8D_IMPORTANCE_TAGGING"] == "0"
