"""Load original SlidingWindowSearchEnv after V8D flags are already set.

Importing this module does not import ultra_core. Call ``load_upstream_env``
only inside an isolated worker process.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from trim.upstream_harness1.api_adapter import chat_tools_from_upstream_schemas
from trim.upstream_harness1.pin import HARNESS1_ROOT, ensure_harness1_on_path
from trim.upstream_harness1.retrieval import (
    RETRIEVAL_LOCAL_BM25,
    RETRIEVAL_UPSTREAM,
    RetrievalConfig,
    canonical_dataset_name,
)


def load_upstream_modules() -> dict[str, Any]:
    """Import original env / tools. Caller must have set V8D_* first."""
    ensure_harness1_on_path()
    from harness.tools import (  # type: ignore[import-not-found]
        GREP_CORPUS_SCHEMA,
        READ_DOCUMENT_SCHEMA,
        SEARCH_CORPUS_SCHEMA,
        GrepCorpusTool,
        GrepCorpusToolCallMetadata,
        PruneChunksTool,
        ReadDocumentTool,
        SearchCorpusTool,
        SearchCorpusToolCallMetadata,
        Tool,
        ToolSet,
        UserTextTool,
    )
    from harness.trajectory import ActionBuilder, Observation, Trajectory  # type: ignore[import-not-found]
    from harness.utils import ProviderFormat  # type: ignore[import-not-found]
    from training.train_rl import SlidingWindowSearchEnv  # type: ignore[import-not-found]

    return {
        "SlidingWindowSearchEnv": SlidingWindowSearchEnv,
        "Tool": Tool,
        "ToolSet": ToolSet,
        "SearchCorpusTool": SearchCorpusTool,
        "GrepCorpusTool": GrepCorpusTool,
        "ReadDocumentTool": ReadDocumentTool,
        "PruneChunksTool": PruneChunksTool,
        "UserTextTool": UserTextTool,
        "SearchCorpusToolCallMetadata": SearchCorpusToolCallMetadata,
        "GrepCorpusToolCallMetadata": GrepCorpusToolCallMetadata,
        "SEARCH_CORPUS_SCHEMA": SEARCH_CORPUS_SCHEMA,
        "GREP_CORPUS_SCHEMA": GREP_CORPUS_SCHEMA,
        "READ_DOCUMENT_SCHEMA": READ_DOCUMENT_SCHEMA,
        "ActionBuilder": ActionBuilder,
        "Observation": Observation,
        "Trajectory": Trajectory,
        "ProviderFormat": ProviderFormat,
        "root": HARNESS1_ROOT,
    }


def ensure_browsecompplus_paths(*, bcp_root: Path | None = None) -> dict[str, str]:
    """Populate upstream harness Config paths for BrowseComp+ when unset."""
    if bcp_root is None:
        from trim.eval.official_query_pool import default_bcp_root

        bcp_root = default_bcp_root()
    if bcp_root is None:
        return {}
    candidates = {
        "BROWSECOMPPLUS_QRELS_GOLD_PATH": bcp_root / "topics-qrels" / "qrel_golds.txt",
        "BROWSECOMPPLUS_QRELS_EVIDENCE_PATH": bcp_root / "topics-qrels" / "qrel_evidence.txt",
        "BROWSECOMPPLUS_QUERIES_PATH": bcp_root / "topics-qrels" / "queries.tsv",
        "BROWSECOMPPLUS_ANSWERS_PATH": bcp_root / "data" / "browsecomp_plus_decrypted.jsonl",
    }
    applied: dict[str, str] = {}
    for env_key, path in candidates.items():
        if os.environ.get(env_key):
            continue
        if path.is_file():
            os.environ[env_key] = str(path)
            applied[env_key] = str(path)
    return applied


def load_scoring_dataset(name: str) -> Any:
    ensure_harness1_on_path()
    ensure_browsecompplus_paths()
    from datagen.search_dataset import get_dataset  # type: ignore[import-not-found]

    return get_dataset(canonical_dataset_name(name))


@dataclass
class EnvToolPack:
    toolset: Any
    search_tool: Any
    dataset: Any
    verifier_client: Any | None = None
    capability_log: dict[str, Any] = field(default_factory=dict)
    backend: str = RETRIEVAL_UPSTREAM
    token_counter: Any | None = None


def build_eval_toolset(
    mods: Mapping[str, Any],
    retrieval: RetrievalConfig,
    *,
    dataset: Any,
    mask: Mapping[str, bool] | None = None,
    token_counter: Any | None = None,
) -> EnvToolPack:
    from trim.upstream_harness1.token_count import TOKEN_COUNT_MODE, whitespace_token_counter

    counter = token_counter or whitespace_token_counter
    retrieval.assert_ready()
    if retrieval.backend == RETRIEVAL_LOCAL_BM25:
        from trim.local_backend.factory import build_local_toolset

        pack = build_local_toolset(mods, retrieval, dataset=dataset, token_counter=counter)
        if mask and mask.get("verify_tool") and pack.verifier_client is None:
            raise RuntimeError(
                "component includes verify_tool; local_bm25 requires --verify-base-url "
                "pointing at a local verifier. Do not stub verify with overlap or a fixed verdict."
            )
        return EnvToolPack(
            toolset=pack.toolset,
            search_tool=pack.search_tool,
            dataset=pack.dataset_wrapper if pack.dataset_wrapper is not None else dataset,
            verifier_client=pack.verifier_client,
            capability_log=pack.capability_log,
            backend=RETRIEVAL_LOCAL_BM25,
            token_counter=counter,
        )
    toolset, search_tool = build_upstream_toolset(
        mods, retrieval, dataset=dataset, token_counter=counter
    )
    return EnvToolPack(
        toolset=toolset,
        search_tool=search_tool,
        dataset=dataset,
        verifier_client=None,
        capability_log={
            "retrieval_backend": RETRIEVAL_UPSTREAM,
            "chroma_initialized": True,
            "token_count_mode": TOKEN_COUNT_MODE,
        },
        backend=RETRIEVAL_UPSTREAM,
        token_counter=counter,
    )


def build_upstream_toolset(
    mods: Mapping[str, Any],
    retrieval: RetrievalConfig,
    *,
    dataset: Any,
    token_counter: Any | None = None,
) -> tuple[Any, Any]:
    if retrieval.backend != RETRIEVAL_UPSTREAM:
        raise RuntimeError(
            f"build_upstream_toolset requires {RETRIEVAL_UPSTREAM}; got {retrieval.backend}"
        )
    from harness.config import get_config  # type: ignore[import-not-found]

    config = get_config()
    collection_names = dataset.get_chroma_collections(split=retrieval.collection_split)
    chroma_client = config.get_chroma_client()
    openai_client = config.get_openai_client()
    reranker = None
    if retrieval.reranker != "none":
        try:
            if retrieval.reranker == "vllm":
                from harness.rerank import VLLMQwen3Reranker  # type: ignore[import-not-found]

                reranker = VLLMQwen3Reranker(max_tokens=retrieval.read_max_tokens)
            else:
                from harness.rerank import BasetenReranker  # type: ignore[import-not-found]

                reranker = BasetenReranker(max_tokens=retrieval.read_max_tokens)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"upstream reranker {retrieval.reranker!r} is not available: {exc}. "
                "Do not silently disable it for the official baseline."
            ) from exc
    SearchCorpusTool = mods["SearchCorpusTool"]
    search_tool = SearchCorpusTool(
        chroma_client=chroma_client,
        openai_client=openai_client,
        chroma_collection_name=collection_names,
        reranker=reranker,
        snippet_max_chars=retrieval.snippet_max_chars,
        knn_limit=retrieval.search_knn_limit,
        search_limit=retrieval.search_limit,
        display_limit=retrieval.search_display_limit,
    )
    toolset = mods["ToolSet"](name=f"{retrieval.dataset}_toolset")
    toolset.add_tool(search_tool)
    toolset.add_tool(
        mods["GrepCorpusTool"](
            chroma_client=chroma_client,
            chroma_collection_name=collection_names,
        )
    )
    toolset.add_tool(
        mods["ReadDocumentTool"](
            chroma_client=chroma_client,
            chroma_collection_name=collection_names,
            reranker=reranker,
            token_counter=token_counter,
            max_tokens=retrieval.read_max_tokens,
        )
    )
    toolset.add_tool(mods["PruneChunksTool"]())
    return toolset, search_tool


def openai_tools_from_env(env: Any, mods: Mapping[str, Any]) -> list[dict[str, Any]]:
    ProviderFormat = mods["ProviderFormat"]
    full = env._build_full_toolset()
    # Chat Completions nested function schema (qwen/moonshot), not Responses flat form.
    raw = full.get_formats(ProviderFormat.QWEN_MOONSHOT)
    return chat_tools_from_upstream_schemas(raw)


# Harmony commentary/analysis channels are not valid chat.completions tool calls.
API_FORMAT_RETRY_PROMPT = (
    "Your previous response could not be parsed as a valid tool call. "
    "Please output a valid function call using the chat.completions tools interface "
    "(JSON arguments on a named tool). "
    "Do not emit Harmony channel markup such as commentary or analysis; "
    "put any reasoning in the assistant message, then call a tool."
)
DEFAULT_CURATE_NUDGE_PROMPT = (
    "IMPORTANT: You just searched without curating. Follow the search → curate rhythm: "
    "review the results from your last search and call curate NOW to add ALL plausibly "
    "relevant documents. Do not search again until you've curated."
)
DEFAULT_MAX_OBS_CHARS = 15000
DEFAULT_PROMPT_TOKEN_BUDGET = 30720
DEFAULT_CURATE_NUDGE_INTERVAL = 1


def clip_observation_text(text: str, max_chars: int = DEFAULT_MAX_OBS_CHARS) -> str:
    raw = str(text or "")
    limit = int(max_chars)
    if len(raw) <= limit:
        return raw
    return raw[:limit] + f"\n... (truncated, {len(raw)} chars total)"


def flatten_message_text(messages: Sequence[Mapping[str, Any]]) -> str:
    parts: list[str] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, Mapping) and item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
                else:
                    parts.append(str(item))
        elif content is not None:
            parts.append(str(content))
        for call in msg.get("tool_calls") or []:
            parts.append(json.dumps(call, ensure_ascii=False))
    return "\n".join(parts)


def _nudge_prompt_for_env(env: Any) -> str | None:
    interval = DEFAULT_CURATE_NUDGE_INTERVAL
    nudge_text = DEFAULT_CURATE_NUDGE_PROMPT
    try:
        from harness.ultra_core import CURATE_NUDGE_INTERVAL, CURATE_NUDGE_PROMPT  # type: ignore[import-not-found]

        interval = int(CURATE_NUDGE_INTERVAL)
        nudge_text = str(CURATE_NUDGE_PROMPT)
    except Exception:
        pass
    turns_since = int(getattr(env, "_turns_since_curate", 0) or 0)
    pool = 0
    wm = getattr(env, "wm", None)
    if wm is not None and hasattr(wm, "get_pool_size"):
        try:
            pool = int(wm.get_pool_size())
        except Exception:
            pool = 0
    if turns_since >= interval and pool > 0:
        return nudge_text
    return None


def _clip_observation(obs: Any, Observation: Any, max_chars: int) -> Any:
    texts = [clip_observation_text(t, max_chars) for t in list(getattr(obs, "observations", None) or [])]
    sources = list(getattr(obs, "sources", None) or [])
    meta = list(getattr(obs, "tool_metadata", None) or [None] * len(texts))
    try:
        return Observation(observations=texts, sources=sources, tool_metadata=meta)
    except Exception:
        return obs


def _truncate_wm_pool(wm_text: str | None, *, chars_to_cut: int) -> str | None:
    raw = str(wm_text or "")
    if not raw or chars_to_cut <= 0:
        return wm_text
    pool_start = raw.find("Document Pool:")
    hist_start = raw.find("Search History:")
    if pool_start <= 0 or hist_start <= pool_start:
        if len(raw) <= chars_to_cut:
            return raw[: max(0, len(raw) // 2)] + "\n...(WM truncated)\n"
        return raw[: max(0, len(raw) - chars_to_cut)] + "\n...(WM truncated)\n"
    pool_section = raw[pool_start:hist_start]
    cut = min(len(pool_section) - 100, chars_to_cut)
    if cut <= 0:
        return wm_text
    new_pool = pool_section[: len(pool_section) - cut] + "\n  ... (truncated for context)\n\n"
    return raw[:pool_start] + new_pool + raw[hist_start:]


def _message_token_count(messages: Sequence[Mapping[str, Any]], counter) -> int:
    return int(counter(flatten_message_text(messages)))


def _raise_over_budget(*, token_count: int, budget: int, stage: str) -> None:
    raise RuntimeError(
        f"API prompt still exceeds budget after {stage}: {token_count} > {budget}. "
        "Refuse to send a known over-budget request."
    )


def openai_messages_from_env(
    env: Any,
    mods: Mapping[str, Any],
    *,
    retry: bool = False,
    token_counter: Any | None = None,
    max_obs_chars: int | None = None,
    prompt_token_budget: int | None = None,
) -> list[dict[str, Any]]:
    """Messages from the original selected window, plus original nudge/retry/clip rules."""
    from trim.upstream_harness1.token_count import whitespace_token_counter

    window = env.selected_context_window()
    Trajectory = mods["Trajectory"]
    Observation = mods["Observation"]
    import uuid

    try:
        from harness.ultra_core import MAX_OBS_CHARS, PROMPT_TOKEN_BUDGET  # type: ignore[import-not-found]

        obs_limit = int(max_obs_chars if max_obs_chars is not None else MAX_OBS_CHARS)
        budget = int(prompt_token_budget if prompt_token_budget is not None else PROMPT_TOKEN_BUDGET)
    except Exception:
        obs_limit = int(max_obs_chars if max_obs_chars is not None else DEFAULT_MAX_OBS_CHARS)
        budget = int(prompt_token_budget if prompt_token_budget is not None else DEFAULT_PROMPT_TOKEN_BUDGET)

    counter = token_counter or getattr(env, "text_token_counter", None) or whitespace_token_counter
    actions = list(window.get("recent_actions") or [])
    observations = list(window.get("recent_observations") or [])
    summaries = list(window.get("result_summaries") or [])
    extra: list[str] = []
    if not retry:
        nudge = _nudge_prompt_for_env(env)
        if nudge:
            extra.append(nudge)
    else:
        extra.append(API_FORMAT_RETRY_PROMPT)

    wm_text = window.get("wm_text")
    obs_limit = int(obs_limit)
    tight = max(2000, obs_limit // 3)

    def _assemble_current(
        *,
        wm: str | None,
        act: list[Any],
        obs: list[Any],
        sums: list[Any],
        obs_chars: int,
    ) -> list[dict[str, Any]]:
        clipped_obs = [_clip_observation(o, Observation, obs_chars) for o in obs]
        entries: list[Any] = [
            Observation(observations=[env.system_prompt], sources=["user"], tool_metadata=[None])
        ]
        if wm:
            entries.append(Observation(observations=[wm], sources=["user"], tool_metadata=[None]))
        n = len(act)
        for i, (action, item) in enumerate(zip(act, clipped_obs)):
            entries.append(action)
            entries.append(item)
            if i < n - 1 and i < len(sums) and sums[i]:
                entries.append(
                    Observation(observations=[sums[i]], sources=["user"], tool_metadata=[None])
                )
        traj = Trajectory(actions_and_observations=entries, id=uuid.uuid4())
        messages = traj.to_openai_format()
        for text in extra:
            messages.append({"role": "user", "content": text})
        return messages

    messages = _assemble_current(wm=wm_text, act=actions, obs=observations, sums=summaries, obs_chars=obs_limit)

    # Pass 1: drop oldest recent turns.
    while _message_token_count(messages, counter) > budget and len(actions) > 1:
        actions = actions[1:]
        observations = observations[1:]
        if summaries:
            summaries = summaries[1:]
        messages = _assemble_current(wm=wm_text, act=actions, obs=observations, sums=summaries, obs_chars=obs_limit)

    # Pass 2: truncate WM pool section.
    if _message_token_count(messages, counter) > budget and wm_text:
        overshoot = _message_token_count(messages, counter) - budget
        wm_text = _truncate_wm_pool(wm_text, chars_to_cut=min(len(str(wm_text)), overshoot * 3))
        messages = _assemble_current(wm=wm_text, act=actions, obs=observations, sums=summaries, obs_chars=obs_limit)

    # Pass 3: aggressive WM cap + tighter observation clip.
    if _message_token_count(messages, counter) > budget:
        if wm_text and len(str(wm_text)) > 2000:
            wm_text = str(wm_text)[:2000] + "\n...(WM truncated)\n"
        tight = max(2000, obs_limit // 3)
        messages = _assemble_current(wm=wm_text, act=actions, obs=observations, sums=summaries, obs_chars=tight)

    # Pass 4: shrink remaining observations until within budget or floor reached.
    floor = 512
    step = max(256, obs_limit // 8)
    if _message_token_count(messages, counter) > budget:
        current_obs_limit = tight
        while _message_token_count(messages, counter) > budget and current_obs_limit > floor:
            current_obs_limit = max(floor, current_obs_limit - step)
            messages = _assemble_current(
                wm=wm_text, act=actions, obs=observations, sums=summaries, obs_chars=current_obs_limit
            )

    # Pass 5: minimal context (system prompt + tail only).
    if _message_token_count(messages, counter) > budget:
        minimal = [{"role": "user", "content": str(env.system_prompt or "")}]
        for text in extra:
            minimal.append({"role": "user", "content": text})
        messages = minimal

    if _message_token_count(messages, counter) > budget:
        _raise_over_budget(token_count=_message_token_count(messages, counter), budget=budget, stage="minimal_context")

    return messages


def action_from_parsed(parsed: Any, env: Any, mods: Mapping[str, Any]) -> Any:
    ActionBuilder = mods["ActionBuilder"]
    UserTextTool = mods["UserTextTool"]
    toolset = env._build_full_toolset()
    builder = ActionBuilder()
    if parsed.reasoning:
        builder.add_reasoning(parsed.reasoning)
    if parsed.parse_error:
        raise ValueError(parsed.parse_error)
    for call in parsed.tool_calls:
        name = str(call.get("name") or "")
        params = dict(call.get("arguments") or {})
        source = str(call.get("id") or "agent")
        if name in {"user_text", "UserTextTool"}:
            builder.add_tool_call(UserTextTool(), params if "text" in params else {"text": json.dumps(params)}, source)
            continue
        tool = toolset.get_tool(name)
        if tool is None:
            raise ValueError(f"Model requested unknown tool or tool not in toolset: {name}")
        builder.add_tool_call(tool, params, source)
    return builder.build()
