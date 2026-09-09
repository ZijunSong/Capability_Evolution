"""Token-rollout adapter over original SlidingWindowSearchEnv (T02).

Official training ``--train-env upstream`` steps the pinned Harness-1 env.
``local_legacy`` is the named TRIM BM25 substitute and is never a silent fallback.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, Callable, Mapping

from trim.upstream_harness1.retrieval import RETRIEVAL_LOCAL_BM25, RETRIEVAL_UPSTREAM, RetrievalConfig

TRAIN_ENV_UPSTREAM = "upstream"
TRAIN_ENV_LOCAL_LEGACY = "local_legacy"

_UPSTREAM_HANDLE = "_upstream_env"
_TRAIN_ENV_KEY = "_train_env"


def canonical_train_env(value: str | None) -> str:
    key = str(value or TRAIN_ENV_UPSTREAM).strip().lower()
    if key in {TRAIN_ENV_UPSTREAM, "original", "harness1"}:
        return TRAIN_ENV_UPSTREAM
    if key in {TRAIN_ENV_LOCAL_LEGACY, "local", "legacy"}:
        return TRAIN_ENV_LOCAL_LEGACY
    raise ValueError(f"unknown train_env={value!r}; use {TRAIN_ENV_UPSTREAM} or {TRAIN_ENV_LOCAL_LEGACY}")


def is_upstream_state(st: Mapping[str, Any]) -> bool:
    return st.get(_TRAIN_ENV_KEY) == TRAIN_ENV_UPSTREAM and st.get(_UPSTREAM_HANDLE) is not None


def _run_coro(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError("upstream train step cannot nest inside a running event loop")


def wm_fields_from_env(env: Any) -> dict[str, Any]:
    wm = getattr(env, "wm", None)
    if wm is None:
        return {}
    store = dict(getattr(wm, "doc_store", None) or {})
    documents = []
    full_store: dict[str, Any] = {}
    for did, rec in store.items():
        if isinstance(rec, dict):
            text = str(rec.get("text") or rec.get("content") or "")
            item = {"id": str(did), "text": text, **{k: v for k, v in rec.items() if k not in {"text", "content"}}}
        else:
            text = str(rec)
            item = {"id": str(did), "text": text}
        documents.append({"id": str(did), "text": text})
        full_store[str(did)] = item
    curated = [str(x) for x in (getattr(wm, "curated_ids", None) or [])]
    pool = [str(x) for x in (getattr(wm, "pool_ids", None) or [])]
    query_text = str(getattr(wm, "query", None) or getattr(env, "query_text", "") or "")
    graph = getattr(wm, "evidence_graph", None)
    graph_payload = None
    if graph is not None and hasattr(graph, "__dict__"):
        graph_payload = {k: v for k, v in vars(graph).items() if not str(k).startswith("_")}
    elif graph is not None:
        graph_payload = graph
    return {
        "query": query_text,
        "query_text": query_text,
        "curated": {cid: True for cid in curated},
        "curated_ids": curated,
        "pool": {pid: True for pid in pool},
        "accessible_doc_ids": list(dict.fromkeys(pool + curated + list(full_store))),
        "documents": documents,
        "doc_store": full_store,
        "importance": dict(getattr(wm, "curated_importance", None) or {}),
        "curated_importance": dict(getattr(wm, "curated_importance", None) or {}),
        "auto_seed": bool(getattr(wm, "auto_populated", False)),
        "auto_populate_seed": bool(getattr(wm, "auto_populated", False)),
        "evidence_graph": graph_payload or {},
        "rerank_instruction": getattr(wm, "rerank_instruction", None),
        "content_dedup_state": getattr(wm, "content_dedup", None),
        "step": int(getattr(env, "_current_turn", 0) or 0),
        "ended": bool(getattr(env, "_episode_ended", False)),
        "n_tool_calls": int(getattr(env, "_current_turn", 0) or 0),
        "n_search_calls": len(getattr(wm, "search_history", None) or []),
        "tool_history": list(getattr(env, "_all_actions", None) or []),
    }


def state_from_upstream_env(
    env: Any,
    *,
    query: str,
    harness_mask: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    st = wm_fields_from_env(env)
    st["query"] = query or st.get("query") or ""
    st["query_text"] = st["query"]
    st[_TRAIN_ENV_KEY] = TRAIN_ENV_UPSTREAM
    st[_UPSTREAM_HANDLE] = env
    st["harness_mask"] = dict(harness_mask or {})
    st["ended"] = bool(getattr(env, "_episode_ended", False))
    return st


def sync_upstream_state(st: dict[str, Any]) -> dict[str, Any]:
    env = st.get(_UPSTREAM_HANDLE)
    if env is None:
        return st
    handle = env
    mask = st.get("harness_mask")
    st.update(wm_fields_from_env(env))
    st[_UPSTREAM_HANDLE] = handle
    st[_TRAIN_ENV_KEY] = TRAIN_ENV_UPSTREAM
    if mask is not None:
        st["harness_mask"] = mask
    return st


def _last_observation_text(env: Any) -> str:
    obs_list = getattr(env, "_all_observations", None) or []
    if not obs_list:
        return ""
    last = obs_list[-1]
    texts = getattr(last, "observations", None)
    if texts:
        return "\n".join(str(x) for x in texts)
    return str(last)


def apply_upstream_action(
    st: dict[str, Any],
    action: Mapping[str, Any],
    valid: bool,
    *,
    mods: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], str, bool]:
    env = st.get(_UPSTREAM_HANDLE)
    if env is None:
        raise RuntimeError("upstream train state is missing the original env handle")
    if not valid:
        result = env._handle_format_error("invalid parsed action")
        sync_upstream_state(st)
        st["ended"] = True
        st["invalid_tools"] = int(st.get("invalid_tools") or 0) + 1
        return st, f"ERROR: format ({getattr(result, 'metrics', {})})", False
    if hasattr(env, "step_parsed_dict"):
        result = _run_coro(env.step_parsed_dict(dict(action)))
    else:
        if mods is None:
            raise RuntimeError("upstream train step needs original modules to build Action")
        from trim.upstream_harness1.env_bridge import action_from_parsed

        parsed = SimpleNamespace(
            reasoning=None,
            parse_error=None,
            tool_calls=[
                {
                    "name": str(action.get("name") or ""),
                    "arguments": dict(action.get("arguments") or {}),
                    "id": "train",
                }
            ],
        )
        built = action_from_parsed(parsed, env, mods)
        result = _run_coro(env.step_action(built))
    sync_upstream_state(st)
    st["ended"] = bool(getattr(env, "_episode_ended", False) or getattr(result, "episode_done", False))
    obs = _last_observation_text(env)
    if getattr(result, "episode_done", False) and not obs:
        metrics = dict(getattr(result, "metrics", None) or {})
        obs = f"EPISODE_DONE {metrics}"
    return st, obs, True


def wm_text_for_train_state(st: Mapping[str, Any]) -> str:
    if is_upstream_state(st):
        env = st.get(_UPSTREAM_HANDLE)
        selected = getattr(env, "selected_context_window", None)
        if callable(selected):
            window = selected()
            return str((window or {}).get("wm_text") or "")
        wm = getattr(env, "wm", None)
        render = getattr(wm, "render", None) or getattr(wm, "text", None)
        if callable(render):
            return str(render())
        return str(getattr(wm, "text", "") or "")
    from trim.eval.local_search_env import wm_text

    return wm_text(st)


def apply_train_action(
    st: dict[str, Any],
    action: Mapping[str, Any],
    valid: bool,
    *,
    searcher: Any = None,
    search_k: int = 10,
    mods: Mapping[str, Any] | None = None,
    execute_local: Callable[..., Any] | None = None,
) -> tuple[dict[str, Any], str, bool]:
    if is_upstream_state(st):
        return apply_upstream_action(st, action, valid, mods=mods)
    if execute_local is None:
        from trim.eval.local_search_env import execute_tool as execute_local
    st, obs, ok = execute_local(
        st,
        action.get("name") if valid else None,
        action.get("arguments"),
        searcher=searcher,
        search_k=search_k,
    )
    return st, obs, bool(ok)


class UpstreamTrainSession:
    """One shared original toolset; one SlidingWindowSearchEnv per episode."""

    def __init__(self, mods: Mapping[str, Any], env_factory: Callable[..., Any]):
        self.mods = mods
        self.env_factory = env_factory

    def new_state(self, query: str, store: dict[str, Any], *, query_id: str = "", harness_mask=None) -> dict[str, Any]:
        env = self.env_factory(query=query, store=store, query_id=query_id or "train")
        init = getattr(env, "initial_observation", None)
        if callable(init):
            maybe = init()
            if asyncio.iscoroutine(maybe):
                _run_coro(maybe)
        st = state_from_upstream_env(env, query=query, harness_mask=harness_mask)
        st["_upstream_mods"] = self.mods
        return st


def open_upstream_train_session(
    *,
    retrieval: RetrievalConfig | None = None,
    harness_mask: Mapping[str, bool] | None = None,
    dataset: Any = None,
    max_turns: int = 40,
) -> UpstreamTrainSession:
    """Build original env + tools. Local BM25 replaces Chroma tools only; no local_search_env fallback."""
    import os

    retrieval = retrieval or RetrievalConfig()
    retrieval.assert_ready()
    if retrieval.backend == RETRIEVAL_LOCAL_BM25:
        os.environ["HARNESS1_FORBID_CHROMA"] = "1"
    from trim.upstream_harness1.env_bridge import build_eval_toolset, load_scoring_dataset, load_upstream_modules
    from trim.upstream_harness1.pin import ensure_harness1_on_path

    ensure_harness1_on_path()
    mods = load_upstream_modules()
    if dataset is None:
        dataset = load_scoring_dataset(retrieval.dataset)
    pack = build_eval_toolset(mods, retrieval, dataset=dataset, mask=harness_mask)
    Env = mods["SlidingWindowSearchEnv"]

    def factory(*, query: str, store: dict[str, Any], query_id: str) -> Any:
        env_kwargs: dict[str, Any] = {
            "toolset": pack.toolset,
            "search_tool": pack.search_tool,
            "query_id": query_id,
            "query_text": query,
            "dataset": pack.dataset,
            "max_turns": max_turns,
        }
        if pack.verifier_client is not None:
            env_kwargs["openai_client"] = pack.verifier_client
        env = Env(**env_kwargs)
        env._trim_seed_store = store
        env._trim_capability_log = pack.capability_log
        return env

    return UpstreamTrainSession(mods, factory)


def new_state_fn(
    *,
    train_env: str,
    harness_mask: Mapping[str, bool] | None,
    session: UpstreamTrainSession | None = None,
    is_harness_g: bool = False,
):
    train_env = canonical_train_env(train_env)
    if is_harness_g:
        from trim.eval.harness_g_env import new_state as g_new_state

        def new_state(query: str, store: dict[str, Any], query_id: str = "") -> dict[str, Any]:
            return g_new_state(query, store, harness_mask=harness_mask)

        return new_state
    if train_env == TRAIN_ENV_LOCAL_LEGACY:
        from trim.eval.local_search_env import new_state as h1_new_state

        def new_state(query: str, store: dict[str, Any], query_id: str = "") -> dict[str, Any]:
            st = h1_new_state(query, store, harness_mask=harness_mask)
            st[_TRAIN_ENV_KEY] = TRAIN_ENV_LOCAL_LEGACY
            st["query_id"] = query_id
            return st

        return new_state
    if session is None:
        raise RuntimeError(
            "train_env=upstream requires an original SlidingWindowSearchEnv session "
            f"(retrieval backend {RETRIEVAL_UPSTREAM} or {RETRIEVAL_LOCAL_BM25}). "
            "Official training does not fall back to TRIM local_search_env. "
            f"Pass --train-env {TRAIN_ENV_LOCAL_LEGACY} for the named substitute."
        )

    def new_state(query: str, store: dict[str, Any], query_id: str = "") -> dict[str, Any]:
        return session.new_state(query, store, query_id=query_id, harness_mask=harness_mask)

    return new_state
