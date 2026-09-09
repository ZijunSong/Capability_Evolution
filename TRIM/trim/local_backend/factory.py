"""Build original ToolSet with local search/grep/read backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from trim.local_backend.bm25 import LocalBm25Backend
from trim.local_backend.corpus_store import CorpusManifest, LocalCorpusStore
from trim.local_backend.corpus_validate import validate_corpus_against_index
from trim.local_backend.id_map import IdMap
from trim.local_backend.tools import LocalGrepCorpusTool, LocalReadDocumentTool, LocalSearchCorpusTool
from trim.upstream_harness1.retrieval import RetrievalConfig


EVAL_PROFILE_LOCAL_BM25 = "upstream_core_local_bm25"


@dataclass
class LocalToolPack:
    toolset: Any
    search_tool: Any
    store: LocalCorpusStore
    backend: LocalBm25Backend
    verifier_client: Any | None
    reranker: Any | None
    capability_log: dict[str, Any]
    dataset_wrapper: Any | None = None


class IdMappedDataset:
    def __init__(self, inner: Any, id_map: IdMap):
        self._inner = inner
        self._id_map = id_map

    def evaluate_results_recall(self, query_id: str, ids: list[str]) -> float:
        return self._inner.evaluate_results_recall(query_id, self._id_map.to_official_list(ids))

    def evaluate_results_precision(self, query_id: str, ids: list[str]) -> float:
        return self._inner.evaluate_results_precision(query_id, self._id_map.to_official_list(ids))

    def evaluate_results_final_answer_recall(self, query_id: str, ids: list[str]) -> float:
        return self._inner.evaluate_results_final_answer_recall(query_id, self._id_map.to_official_list(ids))

    def evaluate_results_f1_score(self, query_id: str, ids: list[str]) -> float:
        return self._inner.evaluate_results_f1_score(query_id, self._id_map.to_official_list(ids))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def load_local_store(retrieval: RetrievalConfig) -> LocalCorpusStore:
    id_map = IdMap.load(Path(retrieval.id_map_path) if retrieval.id_map_path else None)
    manifest = CorpusManifest(
        corpus_version=retrieval.corpus_version or "unknown",
        index_path=str(retrieval.index_path or ""),
        corpus_path=str(retrieval.corpus_path or ""),
        docstore_path=str(retrieval.docstore_path or ""),
        id_map_path=str(retrieval.id_map_path or ""),
    )
    if retrieval.docstore_path:
        store = LocalCorpusStore.from_sqlite(Path(retrieval.docstore_path), id_map=id_map, manifest=manifest)
    elif retrieval.corpus_path:
        store = LocalCorpusStore.from_jsonl(Path(retrieval.corpus_path), id_map=id_map, manifest=manifest)
    else:
        raise RuntimeError(
            "local_bm25 requires --corpus-path (jsonl) or --docstore-path (sqlite) "
            "with full document text. An index that only returns docids is not enough."
        )
    if retrieval.id_map_path:
        store.id_map.save(Path(retrieval.id_map_path))
    return store


def build_local_toolset(
    mods: Mapping[str, Any],
    retrieval: RetrievalConfig,
    *,
    dataset: Any = None,
    require_loopback: bool = True,
    token_counter: Any | None = None,
) -> LocalToolPack:
    retrieval.assert_local_bm25()
    from trim.upstream_harness1.token_count import whitespace_token_counter

    counter = token_counter or whitespace_token_counter
    token_count_mode = getattr(counter, "__name__", None) or "callable"
    store = load_local_store(retrieval)
    if not retrieval.index_path:
        raise RuntimeError("local_bm25 requires --index-path pointing at a Lucene/Pyserini index")
    backend = LocalBm25Backend(index_path=Path(retrieval.index_path), store=store)
    capability_log: dict[str, Any] = {
        "profile": EVAL_PROFILE_LOCAL_BM25,
        "retrieval_backend": "local_bm25",
        "reranker_requested": retrieval.reranker,
        "create_embeddings_called": False,
        "chroma_initialized": False,
        "adaptive_rerank_instruction_consumed_by_search": False,
        "chunk_neighbors_source": "parent_doc_ordinal",
        "chunk_strategy": store.manifest.chunk_strategy,
        "token_count_mode": token_count_mode,
        "corpus_n_documents": store.manifest.n_documents,
        "corpus_n_chunks": store.manifest.n_chunks,
        "read_unknown_id": 0,
        "read_success": 0,
        "grep_no_results": 0,
        "grep_success": 0,
        "grep_invalid_regex": 0,
        "grep_timeout": 0,
        "verify_requests": 0,
        "verify_valid_verdict": 0,
        "verify_empty_content": 0,
        "verify_parse_failures": 0,
        "verify_length_truncated": 0,
        "verify_calls": [],
    }
    corpus_validation = validate_corpus_against_index(
        store=store,
        index_num_docs=backend.num_docs(),
        index_path=Path(retrieval.index_path),
    )
    capability_log["corpus_validation"] = corpus_validation
    capability_log["index_num_docs"] = corpus_validation["index_num_docs"]
    reranker = None
    if retrieval.reranker not in {"none", "", None}:
        if retrieval.reranker in {"local", "local_model"}:
            if not retrieval.reranker_base_url:
                raise RuntimeError(
                    "local reranker requires --reranker-base-url. "
                    "Pass --reranker none if this experiment has no reranker."
                )
            from trim.local_backend.auxiliary import LocalHttpReranker

            reranker = LocalHttpReranker(
                base_url=retrieval.reranker_base_url,
                model=retrieval.reranker_model or "local-reranker",
                max_tokens=retrieval.read_max_tokens,
                token_counter=counter,
                require_loopback=require_loopback,
            )
            capability_log["reranker_identity"] = reranker.identity
        else:
            raise RuntimeError(
                f"local_bm25 does not construct cloud reranker {retrieval.reranker!r}. "
                "Use --reranker none or --reranker local with --reranker-base-url."
            )
    verifier = None
    if retrieval.verify_base_url:
        from trim.local_backend.auxiliary import LocalVerifierClient, OpenAIChatShim

        verifier = OpenAIChatShim(
            LocalVerifierClient(
                base_url=retrieval.verify_base_url,
                model=retrieval.verify_model,
                require_loopback=require_loopback,
                capability_log=capability_log,
            )
        )
        capability_log["verifier_base_url"] = retrieval.verify_base_url
        capability_log["verifier_model"] = retrieval.verify_model
    search = LocalSearchCorpusTool(
        backend=backend,
        store=store,
        metadata_cls=mods["SearchCorpusToolCallMetadata"] if "SearchCorpusToolCallMetadata" in mods else _load_meta(mods, "SearchCorpusToolCallMetadata"),
        schema=_schema(mods, "SEARCH_CORPUS_SCHEMA", "search_corpus"),
        reranker=reranker,
        display_limit=retrieval.search_display_limit,
        search_limit=retrieval.search_limit,
        token_counter=counter,
        capability_log=capability_log,
    )
    grep = LocalGrepCorpusTool(
        store=store,
        metadata_cls=_load_meta(mods, "GrepCorpusToolCallMetadata"),
        schema=_schema(mods, "GREP_CORPUS_SCHEMA", "grep_corpus"),
        token_counter=counter,
    )
    read = LocalReadDocumentTool(
        store=store,
        schema=_schema(mods, "READ_DOCUMENT_SCHEMA", "read_document"),
        reranker=reranker,
        token_counter=counter,
        max_tokens=retrieval.read_max_tokens,
    )
    grep.capability_log = capability_log
    read.capability_log = capability_log
    toolset = mods["ToolSet"](name="local_bm25_toolset")
    bound_search = _bind_tool(mods, search)
    toolset.tools[search.tool_schema.name] = bound_search
    toolset.tools[grep.tool_schema.name] = _bind_tool(mods, grep)
    toolset.tools[read.tool_schema.name] = _bind_tool(mods, read)
    toolset.add_tool(mods["PruneChunksTool"]())
    wrapped = IdMappedDataset(dataset, store.id_map) if dataset is not None else None
    return LocalToolPack(
        toolset=toolset,
        search_tool=bound_search,
        store=store,
        backend=backend,
        verifier_client=verifier,
        reranker=reranker,
        capability_log=capability_log,
        dataset_wrapper=wrapped,
    )


def _bind_tool(mods: Mapping[str, Any], impl: Any) -> Any:
    """Wrap a local backend tool as the original pydantic Tool so ToolSet accepts it."""
    Tool = mods.get("Tool")
    if Tool is None:
        return impl

    class BoundLocalTool(Tool):
        def __init__(self, inner: Any):
            super().__init__(tool_schema=inner.tool_schema)
            object.__setattr__(self, "_inner", inner)

        def __call__(self, params, overrides=None):
            return self._inner(params, overrides)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

    return BoundLocalTool(impl)


def _load_meta(mods: Mapping[str, Any], name: str) -> Any:
    if name in mods:
        return mods[name]
    from harness.tools import GrepCorpusToolCallMetadata, SearchCorpusToolCallMetadata  # type: ignore[import-not-found]

    return {"SearchCorpusToolCallMetadata": SearchCorpusToolCallMetadata, "GrepCorpusToolCallMetadata": GrepCorpusToolCallMetadata}[name]


def _schema(mods: Mapping[str, Any], const_name: str, tool_name: str) -> Any:
    if const_name in mods:
        return mods[const_name]
    from harness import tools as htools  # type: ignore[import-not-found]

    return getattr(htools, const_name)
