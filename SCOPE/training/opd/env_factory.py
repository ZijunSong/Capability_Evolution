"""Shared BrowseComp / Harness env construction for OPD rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import structlog

from datagen.search_dataset import SearchDataset, get_dataset
from harness.config import get_config
from harness.retrieval.bm25_backend import BrowseCompBm25Backend, resolve_bm25_index_path
from harness.retrieval.bm25_tools import (
    Bm25GrepCorpusTool,
    Bm25ReadDocumentTool,
    Bm25SearchCorpusTool,
)
from harness.retrieval.memory_backend import InMemoryBm25Backend
from harness.tools import (
    GrepCorpusTool,
    PruneChunksTool,
    ReadDocumentTool,
    SearchCorpusTool,
    Tool,
    ToolSet,
)
from training.train_rl import SEARCH_DISPLAY_LIMIT, SlidingWindowSearchEnv

logger = structlog.get_logger("training.opd.env_factory")


@dataclass
class RolloutRuntime:
    dataset: SearchDataset
    toolset: ToolSet
    search_tool: Tool
    text_token_counter: Callable[[str], int]
    retrieval_backend: str = "chroma"


def _default_token_counter() -> Callable[[str], int]:
    import tiktoken

    tiktoken_enc = tiktoken.get_encoding("o200k_harmony")
    return lambda text: len(tiktoken_enc.encode(text))


def _optional_reranker(
    reranker: str,
    text_token_counter: Callable[[str], int],
):
    if reranker == "none":
        return None
    try:
        if reranker == "vllm":
            from harness.rerank import VLLMQwen3Reranker

            return VLLMQwen3Reranker(
                token_counter=text_token_counter, max_tokens=4096
            )
        if reranker == "baseten":
            from harness.rerank import BasetenReranker

            return BasetenReranker(
                token_counter=text_token_counter, max_tokens=4096
            )
    except Exception as exc:
        logger.warning("reranker_unavailable", backend=reranker, error=str(exc)[:200])
    return None


def build_bm25_rollout_runtime(
    dataset_name: str = "browsecompplus",
    *,
    bm25_index_path: str | None = None,
    reranker: str = "none",
    backend: BrowseCompBm25Backend | InMemoryBm25Backend | None = None,
) -> RolloutRuntime:
    """Build Harness tools backed by BrowseComp+ BM25 (local index or in-memory smoke)."""
    dataset = get_dataset(dataset_name)
    text_token_counter = _default_token_counter()
    if backend is None:
        backend = BrowseCompBm25Backend(bm25_index_path or resolve_bm25_index_path())
    reranker_obj = _optional_reranker(reranker, text_token_counter)

    search_tool = Bm25SearchCorpusTool(
        backend,
        reranker=reranker_obj,
        token_counter=text_token_counter,
        snippet_max_chars=2048,
        display_limit=SEARCH_DISPLAY_LIMIT,
    )
    toolset = ToolSet(name=f"{dataset_name}_bm25_toolset")
    toolset.add_tool(search_tool)
    toolset.add_tool(
        Bm25GrepCorpusTool(backend, token_counter=text_token_counter)
    )
    toolset.add_tool(
        Bm25ReadDocumentTool(
            backend,
            reranker=reranker_obj,
            token_counter=text_token_counter,
            max_tokens=4096,
        )
    )
    toolset.add_tool(PruneChunksTool())
    return RolloutRuntime(
        dataset=dataset,
        toolset=toolset,
        search_tool=search_tool,
        text_token_counter=text_token_counter,
        retrieval_backend="bm25",
    )


def build_smoke_bm25_rollout_runtime(
    dataset_name: str = "browsecompplus",
) -> RolloutRuntime:
    """Offline smoke runtime: in-memory corpus, no API keys / Java / index."""
    return build_bm25_rollout_runtime(
        dataset_name,
        backend=InMemoryBm25Backend(),
        reranker="none",
    )


def build_rollout_runtime(
    dataset_name: str = "browsecompplus",
    *,
    collection_split: Literal["train", "test", "rl"] = "train",
    reranker: str = "baseten",
    retrieval: Literal["chroma", "bm25"] = "chroma",
    bm25_index_path: str | None = None,
) -> RolloutRuntime:
    """Build dataset + retrieval tools used by BrowseComp rollouts."""
    if retrieval == "bm25":
        return build_bm25_rollout_runtime(
            dataset_name,
            bm25_index_path=bm25_index_path,
            reranker=reranker,
        )

    config = get_config()
    dataset = get_dataset(dataset_name)
    collection_names = dataset.get_chroma_collections(split=collection_split)
    text_token_counter = _default_token_counter()

    chroma_client = config.get_chroma_client()
    openai_client = config.get_openai_client()
    reranker_obj = _optional_reranker(reranker, text_token_counter)

    search_tool = SearchCorpusTool(
        chroma_client=chroma_client,
        openai_client=openai_client,
        chroma_collection_name=collection_names,
        reranker=reranker_obj,
        snippet_max_chars=2048,
        display_limit=SEARCH_DISPLAY_LIMIT,
    )
    toolset = ToolSet(name=f"{dataset_name}_opd_toolset")
    toolset.add_tool(search_tool)
    toolset.add_tool(
        GrepCorpusTool(
            chroma_client=chroma_client,
            chroma_collection_name=collection_names,
            token_counter=text_token_counter,
        )
    )
    toolset.add_tool(
        ReadDocumentTool(
            chroma_client=chroma_client,
            chroma_collection_name=collection_names,
            reranker=reranker_obj,
            token_counter=text_token_counter,
            max_tokens=4096,
        )
    )
    toolset.add_tool(PruneChunksTool())
    return RolloutRuntime(
        dataset=dataset,
        toolset=toolset,
        search_tool=search_tool,
        text_token_counter=text_token_counter,
        retrieval_backend="chroma",
    )


def build_search_env(
    runtime: RolloutRuntime,
    *,
    query_id: str,
    query_text: str,
    max_turns: int,
) -> SlidingWindowSearchEnv:
    return SlidingWindowSearchEnv(
        toolset=runtime.toolset,
        search_tool=runtime.search_tool,  # type: ignore[arg-type]
        query_id=query_id,
        query_text=query_text,
        dataset=runtime.dataset,
        text_token_counter=runtime.text_token_counter,
        max_turns=max_turns,
    )
