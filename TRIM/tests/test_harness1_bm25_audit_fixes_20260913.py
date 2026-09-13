"""Regression tests for Harness-1 + local BM25 audit fixes (2026-09-13)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
HARNESS_ROOT = ROOT / "external" / "harness-1"
for path in (HARNESS_ROOT, HARNESS_ROOT / "tinker-cookbook"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from harness.tools import UserTextTool  # noqa: E402
from harness.trajectory import Action, ActionBuilder  # noqa: E402
from harness.ultra_core import CURATE_SCHEMA, MAX_CURATED_DOCS, WorkingMemory  # noqa: E402
from training.train_rl import CurateTool, EndSearchTool  # noqa: E402
from trim.eval.harness1_api_eval import (  # noqa: E402
    _count_transport_retry_events,
    _query_sampling_seed,
    _transport_retry_events_for_turn,
    count_tool_calls_from_turns,
)
from trim.upstream_harness1.env_bridge import (  # noqa: E402
    SchemaValidationError,
    _validate_tool_params,
    action_from_parsed,
    format_retry_prompt,
)
from trim.upstream_harness1.fix_manifest import PUBLIC_FIX_VERSION


def test_curate_schema_allows_remove_only():
    assert "add_ids" not in CURATE_SCHEMA.required


def test_validate_remove_only_curate():
    tool = MagicMock()
    tool.tool_schema = CURATE_SCHEMA
    _validate_tool_params(tool, {"remove_ids": ["25036"]})


def test_validate_rejects_null_add_ids():
    tool = MagicMock()
    tool.tool_schema = CURATE_SCHEMA
    with pytest.raises(SchemaValidationError, match="must not be null"):
        _validate_tool_params(tool, {"add_ids": None})


def test_validate_rejects_bad_add_ids_items():
    tool = MagicMock()
    tool.tool_schema = CURATE_SCHEMA
    with pytest.raises(SchemaValidationError, match="expected string"):
        _validate_tool_params(tool, {"add_ids": [{}]})


def test_curate_retag_before_capacity_eviction():
    import harness.ultra_core as uc

    wm = WorkingMemory(query="cap test")
    wm.curated_ids = [f"d{i}" for i in range(MAX_CURATED_DOCS)]
    wm.curated_importance = {doc_id: "fair" for doc_id in wm.curated_ids}
    for doc_id in wm.curated_ids:
        wm.pool_id_set.add(doc_id)

    old_sub = uc.V8D_SUBTRACTIVE_CURATION
    old_tag = uc.V8D_IMPORTANCE_TAGGING
    uc.V8D_SUBTRACTIVE_CURATION = True
    uc.V8D_IMPORTANCE_TAGGING = True
    try:
        result = wm.curate(
            add_ids=["new"],
            remove_ids=[],
            importance={"d0": "low", "new": "fair"},
        )
    finally:
        uc.V8D_SUBTRACTIVE_CURATION = old_sub
        uc.V8D_IMPORTANCE_TAGGING = old_tag

    assert "new" in wm.curated_ids
    assert "d0" not in wm.curated_ids
    assert wm.curated_importance.get("d0") is None
    assert len(wm.curated_ids) == MAX_CURATED_DOCS
    assert "[ADDED]" in result


def test_retry_prompt_preserves_failed_request():
    prompt = format_retry_prompt(
        model="Qwen3-4B",
        error_hint="missing required parameter",
        failed_attempt={
            "tool_calls": [{"name": "curate", "arguments": {"remove_ids": ["25036"]}}],
            "error_kind": "schema_error",
            "error_detail": "missing required parameter: add_ids",
        },
    )
    assert "25036" in prompt
    assert "Failed request" in prompt
    assert "will not silently replay" in prompt


def test_transport_retry_events_are_not_double_counted():
    turn = {
        "transport_retry_events": [{"category": "server_error_unclassified"}],
        "api_response": {"_transport_retry_events": [{"category": "server_error_unclassified"}]},
    }
    assert len(_transport_retry_events_for_turn(turn)) == 1
    server, transport = _count_transport_retry_events(turn)
    assert server == 0
    assert transport == 1
    counts = count_tool_calls_from_turns([turn])
    assert counts["transport_error_count"] == 1.0


def test_schema_error_counts_zero_executed_tools():
    turn = {
        "env_exception": "missing required parameter",
        "error_kind": "schema_error",
        "api_response": {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"function": {"name": "curate", "arguments": "{}"}},
                        ]
                    }
                }
            ]
        },
        "n_executed_tool_calls": 0,
    }
    counts = count_tool_calls_from_turns([turn])
    assert counts["schema_error_count"] == 1.0
    assert counts["n_tool_calls"] == 0.0


def test_query_seed_is_deterministic_and_worker_independent():
    s1 = _query_sampling_seed(42, "661")
    s2 = _query_sampling_seed(42, "661")
    s3 = _query_sampling_seed(42, "662")
    assert s1 == s2
    assert s1 != s3


def _minimal_step_env() -> object:
    from training.train_rl import SlidingWindowSearchEnv

    env = object.__new__(SlidingWindowSearchEnv)
    env.wm = WorkingMemory(query="batch test")
    env.wm.pool_id_set.add("d1")
    env.query_id = "q1"
    env._current_turn = 0
    env._episode_ended = False
    env._all_actions = []
    env._all_observations = []
    env._wm_snapshots = [env.wm.snapshot()]
    env._result_summaries = []
    env._turns_since_curate = 0
    env._total_curate_calls = 0
    env._pool_size_at_last_curate = 0
    env._pending_verify_since_curate = False
    env._tool_types_used = set()
    env._format_retries = 0
    env.max_turns = 40
    env.stop_condition = None
    env.system_prompt = "test"
    env.enc = MagicMock()
    env._render_retry_context = lambda: [1, 2, 3]
    env._compute_terminal_reward = lambda: (1.0, {"recall": 1.0, "precision": 1.0, "num_turns": env._current_turn})
    env._save_trajectory = lambda: None
    return env


def test_step_action_executes_curate_before_end_search():
    from training.train_rl import SlidingWindowSearchEnv

    env = _minimal_step_env()
    action = Action(
        tools=[CurateTool(), EndSearchTool()],
        params=[{"add_ids": ["d1"], "remove_ids": []}, {"reasoning": "done"}],
        sources=["c1", "c2"],
    )
    result = asyncio.run(SlidingWindowSearchEnv.step_action(env, action))
    assert result.episode_done is True
    assert "d1" in env.wm.curated_ids


def test_step_action_rejects_end_search_before_curate():
    from training.train_rl import SlidingWindowSearchEnv

    env = _minimal_step_env()
    action = Action(
        tools=[EndSearchTool(), CurateTool()],
        params=[{"reasoning": "done"}, {"add_ids": ["d1"]}],
        sources=["c1", "c2"],
    )
    result = asyncio.run(SlidingWindowSearchEnv.step_action(env, action))
    assert result.episode_done is False
    assert result.metrics.get("format_retry") == 1.0
    assert "d1" not in env.wm.curated_ids


def test_action_from_parsed_accepts_remove_only_curate():
    parsed = SimpleNamespace(
        parse_error=None,
        protocol_error=None,
        reasoning=None,
        tool_calls=[{"name": "curate", "arguments": {"remove_ids": ["25036"]}, "id": "c1"}],
    )
    env = MagicMock()
    toolset = MagicMock()
    toolset.get_tool = lambda name: CurateTool()
    env._build_full_toolset.return_value = toolset
    mods = {
        "ActionBuilder": ActionBuilder,
        "UserTextTool": UserTextTool,
    }
    env._build_full_toolset.return_value.get_tool = lambda name: CurateTool()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("trim.upstream_harness1.env_bridge._allowed_tool_names", lambda _env: {"curate"})
        action = action_from_parsed(parsed, env, mods)
    assert action.tools[0].tool_schema.name == "curate"
    assert action.params[0]["remove_ids"] == ["25036"]


def test_public_fix_version_stamp():
    assert PUBLIC_FIX_VERSION == "harness1-bm25-fix-20260913"
