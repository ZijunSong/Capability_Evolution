from __future__ import annotations

import threading

import pytest

from trim.eval.browsecomp_retrieval import (
    RetrievalBackend,
    SearchHit,
    _PyseriniThread,
    assert_retrieval_ready,
)


def test_pyserini_thread_runs_on_stable_worker():
    box = _PyseriniThread()
    box.start()
    ids: set[int] = set()

    def grab() -> str:
        ids.add(threading.get_ident())
        return threading.current_thread().name

    assert box.call(grab) == "trim-pyserini-jni"
    assert box.call(grab) == "trim-pyserini-jni"
    assert ids == {box._thread.ident}


def test_pyserini_thread_propagates_errors():
    box = _PyseriniThread()
    box.start()

    def boom() -> None:
        raise ValueError("jni-fail")

    with pytest.raises(ValueError, match="jni-fail"):
        box.call(boom)


def test_assert_retrieval_ready_rejects_empty_lucene():
    class EmptyLucene(RetrievalBackend):
        name = "pyserini_lucene"

        def num_docs(self) -> int:
            return 12

        def search(self, query: str, k: int = 5) -> list[SearchHit]:
            del query, k
            return []

    with pytest.raises(RuntimeError, match="probe search returned 0 hits"):
        assert_retrieval_ready(EmptyLucene(), formal=True)


def test_assert_retrieval_ready_accepts_live_hits():
    class Live(RetrievalBackend):
        name = "pyserini_lucene"

        def num_docs(self) -> int:
            return 8

        def search(self, query: str, k: int = 5) -> list[SearchHit]:
            del query
            return [SearchHit("d1", "history of the company", 1.0)][:k]

    assert_retrieval_ready(Live(), formal=True)


def test_open_retrieval_formal_lucene_returns_hits():
    from trim.eval.browsecomp_retrieval import open_retrieval
    from trim.eval.official_query_pool import default_bcp_root

    root = default_bcp_root()
    if root is None or not (root / "indexes" / "bm25").is_dir():
        pytest.skip("BrowseComp-Plus BM25 index not available")
    searcher = open_retrieval(formal=True)
    assert searcher.name == "pyserini_lucene"
    hits = searcher.search("history", 5)
    assert hits
    assert hits[0].docid


def test_open_retrieval_from_short_lived_thread_still_hits():
    from trim.eval.browsecomp_retrieval import open_retrieval
    from trim.eval.official_query_pool import default_bcp_root

    root = default_bcp_root()
    if root is None or not (root / "indexes" / "bm25").is_dir():
        pytest.skip("BrowseComp-Plus BM25 index not available")
    holder: dict = {}
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            holder["searcher"] = open_retrieval(formal=True)
        except BaseException as exc:
            errors.append(exc)

    t = threading.Thread(target=worker, name="trim-eval-prep-sim", daemon=True)
    t.start()
    t.join(timeout=180.0)
    assert not t.is_alive()
    assert errors == []
    hits = holder["searcher"].search("history", 5)
    assert hits
    assert hits[0].docid
