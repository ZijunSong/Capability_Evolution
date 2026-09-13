"""E01 / E03 / E08: pin, actual eval masks, API adapter, no eval-mode invert."""

from __future__ import annotations

import json

from trim.cli.launch import eval_mask_for_ids, parse_eval_args, student_mask_for_ids, teacher_mask_for_ids
from trim.upstream_harness1.api_adapter import (
    ServerParseError,
    parse_chat_completion,
    request_fingerprint,
)
from trim.upstream_harness1.env_bridge import (
    GPT_OSS_FORMAT_RETRY_PROMPT,
    QWEN_FORMAT_RETRY_PROMPT,
    _allowed_tool_names,
    format_retry_prompt,
    is_harmony_chat_model,
)
from trim.upstream_harness1.model_serve import identity_for_actor_rank, parse_actor_base_urls
from trim.upstream_harness1.pin import PINNED_UPSTREAM_COMMIT, pin_manifest
from trim.upstream_harness1.v8d_flags import (
    ABLATE_FLAGS_CLEARED_FOR_BASELINE,
    actual_eval_mask,
    all_enabled_mask,
    subprocess_env_for_mask,
    v8d_env_from_mask,
)


def test_pin_records_upstream_commit_and_interface_patches():
    pin = pin_manifest()
    assert pin["pinned_commit"] == PINNED_UPSTREAM_COMMIT
    assert pin["vendored_exists"] is True
    assert pin["is_git_submodule"] is False
    kinds = {p["kind"] for p in pin["interface_patches"]}
    assert kinds >= {"io_boundary", "behavior"}


def test_eval_all_default_zero_are_actual_switches():
    all_m = actual_eval_mask([], harness="Harness-1", preset="all")
    default_m = actual_eval_mask([], harness="Harness-1", preset="default")
    zero_m = actual_eval_mask([], harness="Harness-1", preset="zero")
    assert sum(all_m.values()) == 10
    assert sum(default_m.values()) == 8
    assert sum(zero_m.values()) == 0
    assert default_m["chunk_neighbors"] is False
    assert default_m["adaptive_rerank_instruction"] is False
    assert all_m["chunk_neighbors"] is True
    assert zero_m["verify_tool"] is False


def test_eval_explicit_list_is_exact_not_complement():
    mask = eval_mask_for_ids(["verify_tool"], harness="Harness-1")
    assert mask["verify_tool"] is True
    assert mask["evidence_graph"] is False
    assert mask["sentence_compress"] is False
    student = student_mask_for_ids(["verify_tool"], harness="Harness-1")
    assert student["verify_tool"] is False


def test_eval_cli_accepts_local_bm25_flags():
    args, spec = parse_eval_args(
        [
            "--component",
            "zero",
            "--out",
            "/tmp/eval-local-bm25",
            "--api-base-url",
            "http://127.0.0.1:8000/v1",
            "--retrieval-backend",
            "local_bm25",
            "--index-path",
            "/data/idx",
            "--corpus-path",
            "/data/corpus.jsonl",
            "--reranker",
            "none",
            "--offline",
        ]
    )
    assert args.retrieval_backend == "local_bm25"
    assert args.reranker == "none"
    assert args.offline is True
    assert spec.component_preset == "zero"


def test_eval_cli_adapter_does_not_invert_component():
    args, spec = parse_eval_args(
        [
            "--component",
            "all",
            "--adapter",
            "/tmp/fake-adapter",
            "--out",
            "/tmp/eval-all-adapter",
            "--api-base-url",
            "http://127.0.0.1:8000/v1",
        ]
    )
    mask = eval_mask_for_ids(spec.components, harness=spec.harness, preset=spec.component_preset)
    assert spec.component_preset == "all"
    assert sum(mask.values()) == 10
    assert args.adapter is not None


def test_zero_does_not_set_ablate_review_docs():
    env = v8d_env_from_mask(actual_eval_mask([], preset="zero", harness="Harness-1"))
    assert env["V8D_VERIFY_TOOL"] == "0"
    assert env["ABLATE_REVIEW_DOCS_UNAVAILABLE"] == "0"
    assert "ABLATE_REVIEW_DOCS_UNAVAILABLE" in ABLATE_FLAGS_CLEARED_FOR_BASELINE
    sub = subprocess_env_for_mask(actual_eval_mask([], preset="all", harness="Harness-1"))
    assert sub["V8D_CHUNK_NEIGHBORS"] == "1"
    assert sub["V8D_ADAPTIVE_RERANK_INSTRUCTION"] == "1"


def test_api_adapter_preserves_verify_importance_and_multi_calls():
    payload = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "reasoning_content": "check claim",
                    "tool_calls": [
                        {
                            "id": "call_verify",
                            "function": {
                                "name": "verify",
                                "arguments": json.dumps({"claim": "X happened in 1999"}),
                            },
                        },
                        {
                            "id": "call_curate",
                            "function": {
                                "name": "curate",
                                "arguments": json.dumps(
                                    {"add_ids": ["d1"], "remove_ids": [], "importance": {"d1": "high"}}
                                ),
                            },
                        },
                    ],
                },
            }
        ]
    }
    parsed = parse_chat_completion(payload)
    assert parsed.ok
    assert parsed.reasoning == "check claim"
    assert [c["name"] for c in parsed.tool_calls] == ["verify", "curate"]
    assert parsed.tool_calls[1]["arguments"]["importance"]["d1"] == "high"
    assert parsed.tool_calls[0]["id"] == "call_verify"


def test_api_adapter_does_not_invent_action_on_format_error():
    parsed = parse_chat_completion({"choices": [{"finish_reason": "stop", "message": {"content": ""}}]})
    assert parsed.ok is False
    assert parsed.tool_calls == []
    assert "Reasoning-only" in (parsed.parse_error or "")


def test_parse_actor_base_urls_comma_separated():
    urls = parse_actor_base_urls("http://127.0.0.1:8000/v1,http://127.0.0.1:8002/v1")
    assert urls == ["http://127.0.0.1:8000/v1", "http://127.0.0.1:8002/v1"]


def test_identity_for_actor_rank_round_robin():
    from trim.upstream_harness1.model_serve import ServedModelIdentity

    base = ServedModelIdentity(api_base_url="http://127.0.0.1:8000/v1", api_model="harness-1")
    urls = ["http://127.0.0.1:8000/v1", "http://127.0.0.1:8002/v1"]
    assert identity_for_actor_rank(base, urls, 0).api_base_url == urls[0]
    assert identity_for_actor_rank(base, urls, 1).api_base_url == urls[1]
    assert identity_for_actor_rank(base, urls, 2).api_base_url == urls[0]


def test_retry_fingerprint_is_stable():
    body = {"model": "x", "messages": [{"role": "user", "content": "q"}]}
    assert request_fingerprint(body) == request_fingerprint(dict(body))


def test_parse_rejects_corrupted_tool_name():
    parsed = parse_chat_completion(
        {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "tool_calls": [
                            {
                                "id": "c1",
                                "function": {
                                    "name": "search_corpus<|channel|>commentary",
                                    "arguments": '{"query": "test"}',
                                },
                            }
                        ]
                    },
                }
            ]
        }
    )
    assert parsed.ok is False
    assert parsed.protocol_error is not None
    assert "Corrupted tool name" in parsed.protocol_error


def test_parse_length_truncation_not_implicit_end():
    parsed = parse_chat_completion(
        {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": "partial answer text"},
                }
            ]
        }
    )
    assert parsed.ok is False
    assert parsed.protocol_error == "length_truncated"
    assert parsed.tool_calls == []


def test_parse_tool_call_in_content_without_structured_calls():
    parsed = parse_chat_completion(
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": '{"name": "search_corpus", "arguments": {"query": "foo"}}'
                    },
                }
            ]
        }
    )
    assert parsed.ok is False
    assert parsed.protocol_error == "tool_call_in_content"


def test_explicit_end_search_finish_reason():
    parsed = parse_chat_completion(
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
    assert parsed.ok
    assert parsed.episode_finish_reason == "explicit_end_search"


def test_implicit_user_text_finish_reason():
    parsed = parse_chat_completion(
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": "I am done searching."},
                }
            ]
        }
    )
    assert parsed.ok
    assert parsed.episode_finish_reason == "implicit_user_text"


def test_harmony_retry_prompt_differs_from_qwen():
    assert is_harmony_chat_model("gpt-oss-20b")
    assert is_harmony_chat_model("harness-1")
    assert not is_harmony_chat_model("Qwen3-4B-Instruct")
    assert GPT_OSS_FORMAT_RETRY_PROMPT != QWEN_FORMAT_RETRY_PROMPT
    assert "analysis channel" in format_retry_prompt(model="gpt-oss-20b")
    assert "analysis channel" not in format_retry_prompt(model="Qwen3-4B")


def test_classify_harmony_http_500_as_server_parse_error():
    err = ServerParseError(
        "HTTP 500",
        status=500,
        body='{"error": {"message": "HarmonyError: unexpected tokens remaining in message header"}}',
    )
    assert err.category == "model_output_parse_error"


def test_train_all_student_zero_teacher_ten():
    teacher = teacher_mask_for_ids(
        list(all_enabled_mask("Harness-1")), harness="Harness-1", preset="all"
    )
    student = student_mask_for_ids(list(all_enabled_mask("Harness-1")), harness="Harness-1")
    assert sum(teacher.values()) == 10
    assert sum(student.values()) == 0


def test_allowed_tool_names_reads_toolset_dict_keys():
    class _Schema:
        name = "search_corpus"

    class _Tool:
        tool_schema = _Schema()

    class _Toolset:
        tools = {
            "search_corpus": _Tool(),
            "grep_corpus": _Tool(),
            "fan_out_search": _Tool(),
        }

    class _Env:
        def _build_full_toolset(self):
            return _Toolset()

    assert _allowed_tool_names(_Env()) == {"search_corpus", "grep_corpus", "fan_out_search"}
