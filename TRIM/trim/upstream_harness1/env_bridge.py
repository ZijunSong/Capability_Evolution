"""Load original SlidingWindowSearchEnv after V8D flags are already set.

Importing this module does not import ultra_core. Call ``load_upstream_env``
only inside an isolated worker process.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
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


def load_scoring_dataset(name: str) -> Any:
    ensure_harness1_on_path()
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


def build_eval_toolset(
    mods: Mapping[str, Any],
    retrieval: RetrievalConfig,
    *,
    dataset: Any,
    mask: Mapping[str, bool] | None = None,
) -> EnvToolPack:
    retrieval.assert_ready()
    if retrieval.backend == RETRIEVAL_LOCAL_BM25:
        from trim.local_backend.factory import build_local_toolset

        pack = build_local_toolset(mods, retrieval, dataset=dataset)
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
        )
    toolset, search_tool = build_upstream_toolset(mods, retrieval, dataset=dataset)
    return EnvToolPack(
        toolset=toolset,
        search_tool=search_tool,
        dataset=dataset,
        verifier_client=None,
        capability_log={"retrieval_backend": RETRIEVAL_UPSTREAM, "chroma_initialized": True},
        backend=RETRIEVAL_UPSTREAM,
    )


def build_upstream_toolset(mods: Mapping[str, Any], retrieval: RetrievalConfig, *, dataset: Any) -> tuple[Any, Any]:
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


def openai_messages_from_env(env: Any, mods: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Use original Action/Observation objects via Trajectory.to_openai_format."""
    window = env.selected_context_window()
    Trajectory = mods["Trajectory"]
    Observation = mods["Observation"]
    import uuid

    entries: list[Any] = [
        Observation(observations=[env.system_prompt], sources=["user"], tool_metadata=[None])
    ]
    if window.get("wm_text"):
        entries.append(
            Observation(observations=[window["wm_text"]], sources=["user"], tool_metadata=[None])
        )
    actions = list(window.get("recent_actions") or [])
    observations = list(window.get("recent_observations") or [])
    summaries = list(window.get("result_summaries") or [])
    n = len(actions)
    for i, (action, obs) in enumerate(zip(actions, observations)):
        entries.append(action)
        entries.append(obs)
        if i < n - 1 and i < len(summaries) and summaries[i]:
            entries.append(
                Observation(observations=[summaries[i]], sources=["user"], tool_metadata=[None])
            )
    traj = Trajectory(actions_and_observations=entries, id=uuid.uuid4())
    return traj.to_openai_format()


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
