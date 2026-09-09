"""T02: training adapter uses original env handle, never silent BM25 fallback."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from trim.training.upstream_train_env import (
    TRAIN_ENV_LOCAL_LEGACY,
    TRAIN_ENV_UPSTREAM,
    apply_train_action,
    canonical_train_env,
    new_state_fn,
    state_from_upstream_env,
)
from trim.upstream_harness1.retrieval import RetrievalConfig


class _FakeTool:
    def __init__(self, name: str):
        self.tool_schema = SimpleNamespace(name=name)


class _FakeAction:
    def __init__(self, name: str, arguments: dict):
        self.tools = [_FakeTool(name)]
        self.arguments = arguments


class FakeUpstreamEnv:
    def __init__(self, query: str, store: dict):
        self.query_text = query
        self.query_id = "q1"
        self.wm = SimpleNamespace(
            query=query,
            curated_ids=[],
            pool_ids=list(store),
            doc_store={
                k: (dict(v) if isinstance(v, dict) else {"id": k, "text": str(v)})
                for k, v in store.items()
            },
            curated_importance={},
            auto_populated=False,
            evidence_graph=None,
            rerank_instruction=None,
            content_dedup=None,
            search_history=[],
        )
        self._episode_ended = False
        self._current_turn = 0
        self._all_actions = []
        self._all_observations = []
        self.applied = []

    def _handle_format_error(self, msg: str):
        self._episode_ended = True
        return SimpleNamespace(episode_done=True, reward=-0.2, metrics={"format_error": 1.0})

    async def step_parsed_dict(self, action: dict):
        self.applied.append(dict(action))
        name = str(action.get("name") or "")
        if name == "curate":
            add_ids = list((action.get("arguments") or {}).get("add_ids") or [])
            for did in add_ids:
                if did not in self.wm.curated_ids:
                    self.wm.curated_ids.append(did)
        if name == "end_search":
            self._episode_ended = True
        self._current_turn += 1
        self._all_observations.append(SimpleNamespace(observations=[f"ok:{name}"]))
        return SimpleNamespace(episode_done=self._episode_ended, reward=0.0, metrics={"no_error": 1.0})


def test_canonical_train_env_names():
    assert canonical_train_env("upstream") == TRAIN_ENV_UPSTREAM
    assert canonical_train_env("local_legacy") == TRAIN_ENV_LOCAL_LEGACY
    with pytest.raises(ValueError):
        canonical_train_env("silent_fallback")


def test_upstream_without_session_does_not_fall_back():
    with pytest.raises(RuntimeError, match="does not fall back"):
        new_state_fn(train_env="upstream", harness_mask=None, session=None)("q", {})


def test_same_action_same_transition_on_fake_upstream():
    store = {"d1": {"id": "d1", "text": "full " * 1000}}
    env = FakeUpstreamEnv("Who wrote Y?", store)
    st = state_from_upstream_env(env, query="Who wrote Y?", harness_mask={})
    assert len(st["documents"][0]["text"]) == len("full " * 1000)
    st, obs, ok = apply_train_action(st, {"name": "curate", "arguments": {"add_ids": ["d1"]}}, True)
    assert ok and "curate" in obs
    assert st["curated"]["d1"] is True
    env2 = FakeUpstreamEnv("Who wrote Y?", store)
    st2 = state_from_upstream_env(env2, query="Who wrote Y?", harness_mask={})
    apply_train_action(st2, {"name": "curate", "arguments": {"add_ids": ["d1"]}}, True)
    assert st["curated"] == st2["curated"]
    assert env.applied == env2.applied


def test_invalid_action_uses_original_format_error():
    env = FakeUpstreamEnv("q", {})
    st = state_from_upstream_env(env, query="q")
    st, obs, ok = apply_train_action(st, {"name": "verify", "arguments": {}}, False)
    assert ok is False
    assert st["ended"] is True
    assert "ERROR" in obs


def test_retrieval_from_mapping_accepts_list_notes():
    cfg = RetrievalConfig.from_mapping({"backend": "upstream", "notes": ["a", "b"]})
    assert cfg.notes == ("a", "b")
    cfg.assert_official_baseline()
    with pytest.raises(RuntimeError, match="local_bm25"):
        RetrievalConfig(backend="substitute_bm25").assert_official_baseline()
