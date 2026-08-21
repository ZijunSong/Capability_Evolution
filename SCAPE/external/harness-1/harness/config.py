"""Runtime configuration helpers for the Search Agent project."""

from __future__ import annotations

import hashlib
import logging
import os
import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import anthropic
try:
    from baseten_performance_client import PerformanceClient
except ModuleNotFoundError:  # pragma: no cover - optional deployment dependency
    PerformanceClient = None
try:
    import pysqlite3  # type: ignore
    sys.modules["sqlite3"] = pysqlite3
except Exception:
    pass
import chromadb
import structlog
try:
    import tinker
except ModuleNotFoundError:  # local HF/PEFT evaluator does not require the hosted SDK
    tinker = SimpleNamespace()
from openai import OpenAI
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_ENV_FILES = (
    str(REPO_ROOT / ".env.local"),
    str(REPO_ROOT / ".env"),
)


def init_logging(
    app_level: int = logging.INFO,
    *,
    lib_level: int = logging.WARNING,
    colors: bool = True,
    pad_event: bool = True,
    pad_level: bool = False,
) -> None:
    """Configure structured logging without lowering library log thresholds."""

    logging.basicConfig(level=lib_level, format="%(message)s")
    structlog.configure_once(
        processors=[
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(
                colors=colors, pad_event=pad_event, pad_level=pad_level
            ),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(app_level),
        cache_logger_on_first_use=True,
    )


def _hash_embedding(text: str, *, dim: int = 128) -> list[float]:
    vec = [0.0] * dim
    tokens = [tok for tok in text.lower().split() if tok]
    if not tokens:
        return vec
    for tok in tokens[:2048]:
        h = int(hashlib.sha256(tok.encode("utf-8")).hexdigest()[:8], 16)
        vec[h % dim] += 1.0
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


class _LocalEmbeddingData:
    def __init__(self, embedding: list[float]) -> None:
        self.embedding = embedding


class _LocalEmbeddingResponse:
    def __init__(self, data: list[_LocalEmbeddingData]) -> None:
        self.data = data


class _LocalOpenAIEmbeddings:
    def __init__(self, dim: int = 128) -> None:
        self.dim = dim

    def create(self, model: str, input, encoding_format: str = "float") -> _LocalEmbeddingResponse:
        items = input if isinstance(input, list) else [input]
        data = [_LocalEmbeddingData(_hash_embedding(str(text), dim=self.dim)) for text in items]
        return _LocalEmbeddingResponse(data)


class _LocalOpenAIClient:
    def __init__(self, dim: int = 128) -> None:
        self.embeddings = _LocalOpenAIEmbeddings(dim=dim)


class _LocalChromaCollection:
    def __init__(self, collection: chromadb.Collection | None, records: list[dict[str, str]] | None = None) -> None:
        self._collection = collection
        self._records = records or []

    def search(self, search):
        # Local Chroma PersistentClient does not implement the Cloud Search DSL used
        # by Harness-1. In SCAPE local mode, prefer the qrel-aligned JSONL corpus
        # exported beside the Chroma build and return the same result shape.
        limit = int(os.environ.get("SCAPE_LOCAL_CHROMA_SEARCH_LIMIT", "50"))
        if self._records:
            records = self._records[:limit]
            return {
                "ids": [[r["id"] for r in records]],
                "documents": [[r["text"] for r in records]],
                "metadatas": [[{"source": r.get("source", r["id"])} for r in records]],
            }
        if self._collection is None:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]]}
        result = self._collection.get(limit=limit, include=["documents", "metadatas"])
        return {
            "ids": [result.get("ids", [])],
            "documents": [result.get("documents", [])],
            "metadatas": [result.get("metadatas", [])],
        }

    def get(self, **kwargs):
        if self._collection is None:
            return {"ids": [], "documents": [], "metadatas": []}
        return self._collection.get(**kwargs)

    def count(self):
        if self._records:
            return len(self._records)
        return 0 if self._collection is None else self._collection.count()


class _LocalChromaClient:
    def __init__(self, *, chroma_path: Path, collection_name: str) -> None:
        corpus_path = os.environ.get("SCAPE_RETRIEVAL_CORPUS")
        self._records: list[dict[str, str]] = []
        if corpus_path and Path(corpus_path).is_file():
            with Path(corpus_path).open(encoding="utf-8") as f:
                for line in f:
                    obj = __import__("json").loads(line)
                    self._records.append({
                        "id": str(obj.get("id") or obj.get("source")),
                        "source": str(obj.get("source") or obj.get("id")),
                        "text": str(obj.get("text") or ""),
                    })
        self._client = None if self._records else chromadb.PersistentClient(path=str(chroma_path))
        self._collection_name = collection_name
        self._collection_cache: dict[str, _LocalChromaCollection] = {}

    def get_collection(self, name: str):
        if name not in self._collection_cache:
            collection = None if self._client is None else self._client.get_collection(self._collection_name)
            self._collection_cache[name] = _LocalChromaCollection(collection, self._records)
        return self._collection_cache[name]


class Config(BaseSettings):
    """Runtime configuration loaded from environment variables or .env files."""

    model_config = SettingsConfigDict(
        env_file=DEFAULT_ENV_FILES,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: SecretStr = SecretStr("dummy")
    anthropic_api_key: SecretStr = SecretStr("dummy")
    chroma_api_key: SecretStr = SecretStr("dummy")
    chroma_database: str = "dummy"
    huggingface_token: SecretStr = SecretStr("dummy")
    tinker_api_key: SecretStr = SecretStr("dummy")
    browsecompplus_qrels_gold_path: str = ""
    browsecompplus_qrels_evidence_path: str = ""
    browsecompplus_queries_path: str = ""
    browsecompplus_answers_path: str = ""
    moonshot_api_key: SecretStr = SecretStr("dummy")
    baseten_api_key: SecretStr = SecretStr("dummy")
    baseten_model_url: str = ""
    jina_api_key: SecretStr = SecretStr("dummy")
    contextual_api_key: SecretStr = SecretStr("dummy")

    def get_chroma_client(self) -> chromadb.ClientAPI:
        chroma_path = os.environ.get("SCAPE_CHROMA_PATH") or os.environ.get("HARNESS1_CHROMA_PATH")
        if chroma_path:
            return _LocalChromaClient(
                chroma_path=Path(chroma_path),
                collection_name=os.environ.get("SCAPE_CHROMA_COLLECTION", "scape_browsecompplus_local_test"),
            )
        return chromadb.CloudClient(
            api_key=self.chroma_api_key.get_secret_value(),
            database=self.chroma_database,
        )

    def get_openai_client(self) -> OpenAI:
        if os.environ.get("SCAPE_LOCAL_OPENAI_EMBEDDINGS", "1") == "1":
            return _LocalOpenAIClient()
        return OpenAI(api_key=self.openai_api_key.get_secret_value())

    def get_anthropic_client(self) -> anthropic.Anthropic:
        return anthropic.Anthropic(api_key=self.anthropic_api_key.get_secret_value())

    def get_moonshot_client(self) -> OpenAI:
        return OpenAI(
            api_key=self.moonshot_api_key.get_secret_value(),
            base_url="https://api.moonshot.ai/v1",
        )

    def get_tinker_service_client(self) -> tinker.ServiceClient:
        return tinker.ServiceClient(api_key=self.tinker_api_key.get_secret_value())

    def get_baseten_client(self) -> PerformanceClient:
        if PerformanceClient is None:
            raise ModuleNotFoundError("baseten_performance_client is not installed")
        return PerformanceClient(
            base_url=self.baseten_model_url,
            api_key=self.baseten_api_key.get_secret_value(),
        )


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Return a cached settings instance."""

    config = Config()  # type: ignore[call-arg]
    init_logging()
    if config.huggingface_token:
        # Populate this here since HF libraries are cumbersome to configure otherwise
        os.environ["HF_TOKEN"] = config.huggingface_token.get_secret_value()
    if config.tinker_api_key:
        os.environ["TINKER_API_KEY"] = config.tinker_api_key.get_secret_value()
    if config.jina_api_key:
        os.environ["CHROMA_JINA_API_KEY"] = config.jina_api_key.get_secret_value()
    return config
