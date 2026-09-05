"""Optional BrowseComp-Plus retrieval backends for official four-cell eval."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import json
import os
import queue
import shutil
import sys
import threading

from trim.eval.official_query_pool import default_bcp_root

_PROBE_QUERIES = ("history", "company", "science", "government", "university")


class _PyseriniThread:
    """Serialize all Lucene/JNI work onto one long-lived thread.

    Pyjnius + LuceneSearcher are not safe to construct on a short-lived
    worker (eval used to init Pyserini in a daemon thread overlapping vLLM
    start). After that thread exits, ``search()`` returns empty hits without
    raising, so official recall and the episode pool both stay at 0.
    """

    _shared: "_PyseriniThread | None" = None
    _shared_lock = threading.Lock()

    def __init__(self) -> None:
        self._jobs: queue.Queue[tuple[Callable[[], Any], list[Any], list[BaseException], threading.Event] | None] = queue.Queue()
        self._thread = threading.Thread(target=self._loop, name="trim-pyserini-jni", daemon=True)
        self._started = threading.Event()

    @classmethod
    def shared(cls) -> "_PyseriniThread":
        with cls._shared_lock:
            if cls._shared is None:
                cls._shared = cls()
                cls._shared.start()
            return cls._shared

    def start(self) -> None:
        self._thread.start()
        if not self._started.wait(timeout=30):
            raise RuntimeError("Pyserini JNI thread failed to start")

    def _loop(self) -> None:
        self._started.set()
        while True:
            item = self._jobs.get()
            if item is None:
                return
            fn, out, err, done = item
            try:
                out.append(fn())
            except BaseException as exc:  # noqa: BLE001
                err.append(exc)
            done.set()

    def call(self, fn: Callable[[], Any], *, timeout: float = 120.0) -> Any:
        if not self._thread.is_alive():
            raise RuntimeError("Pyserini JNI thread is not running")
        out: list[Any] = []
        err: list[BaseException] = []
        done = threading.Event()
        self._jobs.put((fn, out, err, done))
        if not done.wait(timeout=timeout):
            raise TimeoutError("Pyserini JNI call timed out")
        if err:
            raise err[0]
        return out[0] if out else None


@dataclass
class SearchHit:
    docid: str
    text: str
    score: float = 0.0


class RetrievalBackend:
    name = "none"

    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        del query, k
        return []

    def get_doc(self, docid: str) -> str | None:
        del docid
        return None


def lucene_stored_text(raw: Any) -> str:
    """Unwrap Pyserini ``storeRaw`` JSON (``id`` / ``contents``) into document text."""
    if raw is None:
        return ""
    if isinstance(raw, dict):
        return str(raw.get("contents") or raw.get("text") or raw.get("document_text") or "")
    text = str(raw)
    stripped = text.strip()
    if not stripped.startswith("{"):
        return text
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return text
    if isinstance(payload, dict):
        return str(payload.get("contents") or payload.get("text") or payload.get("document_text") or text)
    return text


class PyseriniBackend(RetrievalBackend):
    name = "pyserini_lucene"

    def __init__(self, index_dir: Path):
        _configure_java_runtime()
        # pyserini.search.lucene imports the OpenAI encoder stack at module load.
        os.environ.setdefault("OPENAI_API_KEY", "sk-pyserini-local")
        index = str(index_dir)
        self._jni = _PyseriniThread.shared()

        def _construct():
            _configure_java_runtime()
            os.environ.setdefault("OPENAI_API_KEY", "sk-pyserini-local")
            from pyserini.search.lucene import LuceneSearcher

            return LuceneSearcher(index)

        self._searcher = self._jni.call(_construct, timeout=300.0)

    def num_docs(self) -> int:
        def _n() -> int:
            return int(getattr(self._searcher, "num_docs", 0) or 0)

        return int(self._jni.call(_n) or 0)

    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        q = str(query)
        kk = int(k)

        def _search() -> list[SearchHit]:
            hits: list[SearchHit] = []
            for hit in self._searcher.search(q, kk):
                raw = ""
                try:
                    raw = self._searcher.doc(hit.docid).raw()
                except Exception:
                    raw = getattr(hit, "raw", "") or ""
                hits.append(
                    SearchHit(
                        str(hit.docid),
                        lucene_stored_text(raw),
                        float(getattr(hit, "score", 0.0) or 0.0),
                    )
                )
            return hits

        return list(self._jni.call(_search, timeout=120.0) or [])

    def get_doc(self, docid: str) -> str | None:
        did = str(docid)

        def _get() -> str | None:
            try:
                doc = self._searcher.doc(did)
            except Exception:
                return None
            if doc is None:
                return None
            return lucene_stored_text(doc.raw() or "")

        return self._jni.call(_get, timeout=60.0)


class LocalJsonlBackend(RetrievalBackend):
    name = "local_corpus_token_overlap"

    def __init__(self, path: Path):
        import json
        import re

        self._docs: list[tuple[str, str]] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                did = str(row.get("id") or row.get("docid") or row.get("source") or "")
                text = str(row.get("text") or row.get("contents") or row.get("content") or "")
                if did and text:
                    self._docs.append((did, text))
        self._tok = lambda s: {x for x in re.findall(r"[a-z0-9]{3,}", (s or "").lower())}

    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        q = self._tok(query)
        scored: list[tuple[int, str, str]] = []
        for did, text in self._docs:
            overlap = len(q & self._tok(text))
            if overlap:
                scored.append((overlap, did, text))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [SearchHit(did, text, float(score)) for score, did, text in scored[:k]]

    def get_doc(self, docid: str) -> str | None:
        for did, text in self._docs:
            if did == str(docid):
                return text
        return None


def _apply_java_bin(java_bin: Path) -> bool:
    if not java_bin.is_file():
        return False
    home = java_bin.parent.parent
    os.environ["JAVA_HOME"] = str(home)
    os.environ["PATH"] = f"{java_bin.parent}:{os.environ.get('PATH', '')}"
    return True


def _configure_java_runtime() -> None:
    """Make a JDK visible before Pyserini imports jnius."""
    existing = os.environ.get("JAVA_HOME")
    if existing:
        java = Path(existing) / "bin" / "java"
        if java.is_file():
            os.environ["PATH"] = f"{java.parent}:{os.environ.get('PATH', '')}"
            return
    candidates: list[Path] = []
    for root in ("/opt/scape-jdk21", "/opt/jdk21"):
        candidates.append(Path(root) / "usr/lib/jvm/java-21-openjdk-amd64/bin/java")
    for prefix in (os.environ.get("CONDA_PREFIX"), sys.prefix, "/data/ppnm/miniconda3/envs/bishop"):
        if prefix:
            candidates.append(Path(prefix) / "lib/jvm/bin/java")
            candidates.append(Path(prefix) / "bin/java")
    which = shutil.which("java")
    if which:
        candidates.append(Path(which))
    for java in candidates:
        if _apply_java_bin(java):
            return


def assert_retrieval_ready(searcher: RetrievalBackend, *, formal: bool = False) -> None:
    """Fail fast if a live backend cannot actually retrieve documents."""
    if searcher is None or searcher.name == "none":
        if formal:
            raise RuntimeError("formal retrieval backend is none")
        return
    if searcher.name != "pyserini_lucene":
        return
    n_docs = getattr(searcher, "num_docs", None)
    if callable(n_docs):
        n_docs = n_docs()
    if n_docs is not None and int(n_docs) <= 0:
        raise RuntimeError(f"{searcher.name} index reports {n_docs} documents")
    hits: list[SearchHit] = []
    for query in _PROBE_QUERIES:
        hits = list(searcher.search(query, 3) or [])
        if hits:
            break
    if not hits:
        raise RuntimeError(
            f"{searcher.name} probe search returned 0 hits. "
            "Lucene/JNI is not usable; do not construct LuceneSearcher on a "
            "short-lived thread (search() will silently return empty)."
        )


def open_retrieval(
    bcp_root: Path | None = None, *, formal: bool = False
) -> RetrievalBackend:
    root = bcp_root or default_bcp_root()
    if root is None:
        if formal:
            raise RuntimeError("formal retrieval requires a BrowseComp-Plus root")
        return RetrievalBackend()
    index = root / "indexes" / "bm25"
    if index.is_dir():
        try:
            _configure_java_runtime()
            backend = PyseriniBackend(index)
            assert_retrieval_ready(backend, formal=formal)
            return backend
        except Exception as exc:
            if formal:
                raise RuntimeError(
                    f"official Pyserini Lucene retrieval unavailable for {index}"
                ) from exc
    elif formal:
        raise RuntimeError(f"official Pyserini Lucene index missing: {index}")
    corpus = root / "data" / "browsecomp_plus_decrypted.jsonl"
    if corpus.is_file():
        return LocalJsonlBackend(corpus)
    if formal:
        raise RuntimeError(f"BrowseComp-Plus corpus unavailable under {root}")
    return RetrievalBackend()


def norm_doc(docid: str) -> str:
    return str(docid).split("_", 1)[0]


def evidence_recall(retrieved: list[str], evidence_docids: list[str]) -> float:
    gold = {norm_doc(x) for x in evidence_docids}
    if not gold:
        return 0.0
    got = {norm_doc(x) for x in retrieved}
    return len(got & gold) / len(gold)


def hits_to_doc_store(hits: list[SearchHit]) -> dict[str, Any]:
    return {h.docid: {"id": h.docid, "text": h.text[:4000], "score": h.score} for h in hits}
