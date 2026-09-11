"""Pinned original retrieval / auxiliary-model conditions (E05 / L01)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping

RETRIEVAL_UPSTREAM = "upstream"
RETRIEVAL_LOCAL_BM25 = "local_bm25"
RETRIEVAL_LOCAL_HYBRID = "local_hybrid"
RETRIEVAL_SUBSTITUTE_BM25 = "substitute_bm25"

PROFILE_UPSTREAM_CHROMA = "upstream_chroma"
PROFILE_LOCAL_BM25 = "upstream_core_local_bm25"

DEFAULT_RERANKER = "baseten"
# Local upstream_api eval: verifier is always a separate harness-1 vLLM, not the actor.
DEFAULT_VERIFY_MODEL = "harness-1-verifier"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
SEARCH_DISPLAY_LIMIT = 10
SEARCH_KNN_LIMIT = 25
SEARCH_LIMIT = 50
SNIPPET_MAX_CHARS = 2048
READ_MAX_TOKENS = 4096
SENTENCE_COMPRESS_K = 4
AUTO_POPULATE_TOP_K = 8


@dataclass(frozen=True)
class RetrievalConfig:
    backend: str = RETRIEVAL_UPSTREAM
    dataset: str = "browsecompplus"
    collection_split: str = "test"
    reranker: str = DEFAULT_RERANKER
    verify_model: str = DEFAULT_VERIFY_MODEL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    search_display_limit: int = SEARCH_DISPLAY_LIMIT
    search_knn_limit: int = SEARCH_KNN_LIMIT
    search_limit: int = SEARCH_LIMIT
    snippet_max_chars: int = SNIPPET_MAX_CHARS
    read_max_tokens: int = READ_MAX_TOKENS
    sentence_compress_k: int = SENTENCE_COMPRESS_K
    auto_populate_top_k: int = AUTO_POPULATE_TOP_K
    chroma_available: bool | None = None
    reranker_available: bool | None = None
    verifier_available: bool | None = None
    notes: tuple[str, ...] = ()
    index_path: str = ""
    corpus_path: str = ""
    docstore_path: str = ""
    id_map_path: str = ""
    corpus_manifest: str = ""
    corpus_version: str = ""
    verify_base_url: str = ""
    reranker_base_url: str = ""
    reranker_model: str = ""
    offline: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "RetrievalConfig":
        payload = dict(data or {})
        notes = payload.get("notes") or ()
        if isinstance(notes, list):
            payload["notes"] = tuple(notes)
        if "dataset" in payload and payload["dataset"]:
            payload["dataset"] = canonical_dataset_name(str(payload["dataset"]))
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in payload.items() if k in known})

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["notes"] = list(self.notes)
        payload["eval_profile"] = self.eval_profile()
        return payload

    def eval_profile(self) -> str:
        if self.backend == RETRIEVAL_LOCAL_BM25:
            return PROFILE_LOCAL_BM25
        if self.backend == RETRIEVAL_UPSTREAM:
            return PROFILE_UPSTREAM_CHROMA
        return str(self.backend)

    def assert_official_baseline(self) -> None:
        if self.backend != RETRIEVAL_UPSTREAM:
            raise RuntimeError(
                "Chroma official baseline requires retrieval backend "
                f"{RETRIEVAL_UPSTREAM!r}; got {self.backend!r}. "
                f"For a local Lucene index use {RETRIEVAL_LOCAL_BM25}."
            )

    def assert_local_bm25(self) -> None:
        if self.backend != RETRIEVAL_LOCAL_BM25:
            raise RuntimeError(f"local_bm25 config required; got {self.backend!r}")
        if not self.index_path:
            raise RuntimeError("local_bm25 requires --index-path (Lucene/Pyserini)")
        if not self.corpus_path and not self.docstore_path:
            raise RuntimeError("local_bm25 requires --corpus-path or --docstore-path with full text")

    def assert_ready(self) -> None:
        if self.backend == RETRIEVAL_UPSTREAM:
            self.assert_official_baseline()
            return
        if self.backend == RETRIEVAL_LOCAL_BM25:
            self.assert_local_bm25()
            return
        if self.backend == RETRIEVAL_LOCAL_HYBRID:
            raise RuntimeError("local_hybrid is reserved and not implemented in this revision")
        if self.backend == RETRIEVAL_SUBSTITUTE_BM25:
            raise RuntimeError(
                "substitute_bm25 is retired and is not an eval backend. "
                "Use --retrieval-backend local_bm25 with a Lucene index; "
                "do not re-enable TRIM local_search_env."
            )
        raise RuntimeError(f"unknown retrieval backend {self.backend!r}")


def _opt_path(args: Any, name: str) -> str:
    value = getattr(args, name, None)
    return str(value) if value else ""


_DATASET_ALIASES = {
    "browsecomp_plus": "browsecompplus",
    "bc+": "browsecompplus",
    "BC+": "browsecompplus",
}


def canonical_dataset_name(name: str) -> str:
    raw = str(name or "").strip() or "browsecompplus"
    return _DATASET_ALIASES.get(raw, raw)


def retrieval_from_args(args: Any) -> RetrievalConfig:
    backend = str(getattr(args, "retrieval_backend", None) or RETRIEVAL_UPSTREAM)
    reranker = str(getattr(args, "reranker", None) or DEFAULT_RERANKER)
    notes: list[str] = []
    if backend == RETRIEVAL_LOCAL_BM25:
        notes.append("upstream_core_local_bm25; not a paper-hybrid Chroma reproduction")
    if backend == RETRIEVAL_SUBSTITUTE_BM25:
        notes.append("retired name; use local_bm25")
    return RetrievalConfig(
        backend=backend,
        dataset=canonical_dataset_name(getattr(args, "upstream_dataset", None) or "browsecompplus"),
        collection_split=str(getattr(args, "collection_split", None) or "test"),
        reranker=reranker,
        verify_model=str(getattr(args, "verify_model", None) or DEFAULT_VERIFY_MODEL),
        notes=tuple(notes),
        index_path=_opt_path(args, "index_path"),
        corpus_path=_opt_path(args, "corpus_path"),
        docstore_path=_opt_path(args, "docstore_path"),
        id_map_path=_opt_path(args, "id_map_path"),
        corpus_manifest=_opt_path(args, "corpus_manifest"),
        corpus_version=str(getattr(args, "corpus_version", None) or ""),
        verify_base_url=str(getattr(args, "verify_base_url", None) or ""),
        reranker_base_url=str(getattr(args, "reranker_base_url", None) or ""),
        reranker_model=str(getattr(args, "reranker_model", None) or ""),
        offline=bool(getattr(args, "offline", False)),
    )
