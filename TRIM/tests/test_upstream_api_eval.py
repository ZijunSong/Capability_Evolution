"""E01 / E03 / E08: pin, actual eval masks, API adapter, no eval-mode invert."""

from __future__ import annotations

import json

from trim.cli.launch import eval_mask_for_ids, parse_eval_args, student_mask_for_ids, teacher_mask_for_ids
from trim.upstream_harness1.api_adapter import parse_chat_completion, request_fingerprint
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
    assert kinds == {"io_boundary"}


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


def test_retry_fingerprint_is_stable():
    body = {"model": "x", "messages": [{"role": "user", "content": "q"}]}
    assert request_fingerprint(body) == request_fingerprint(dict(body))


def test_train_all_student_zero_teacher_ten():
    teacher = teacher_mask_for_ids(
        list(all_enabled_mask("Harness-1")), harness="Harness-1", preset="all"
    )
    student = student_mask_for_ids(list(all_enabled_mask("Harness-1")), harness="Harness-1")
    assert sum(teacher.values()) == 10
    assert sum(student.values()) == 0
