from __future__ import annotations

from scape.eval.browsecomp_retrieval import SearchHit
from scape.eval.local_search_env import execute_tool, new_state


class _LiveSearcher:
    name = "pyserini_lucene"

    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        del query
        return [
            SearchHit("gold", "Apple FY2023 10-K was filed on November 3, 2023.", 4.2),
            SearchHit("noise", "unrelated weather notes", 0.1),
        ][:k]


def test_execute_tool_live_search_writes_index_hits_into_pool():
    st = new_state("When was the Apple FY2023 10-K filed?", {})
    st, obs, ok = execute_tool(
        st,
        "search_corpus",
        {"query": "Apple FY2023 10-K"},
        searcher=_LiveSearcher(),
        search_k=10,
    )
    assert ok is True
    assert "gold" in st["pool"]
    assert "gold" in st["doc_store"]
    assert "Apple FY2023" in obs
    assert st["n_search_calls"] == 1


def test_execute_tool_without_searcher_ranks_seeded_store_only():
    st = new_state(
        "capital",
        {
            "d1": {"id": "d1", "text": "Paris is the capital of France."},
            "d2": {"id": "d2", "text": "unrelated sports scores"},
        },
    )
    st, _obs, ok = execute_tool(st, "search_corpus", {"query": "capital of France"})
    assert ok is True
    assert "d1" in st["pool"]
    assert set(st["pool"]) <= {"d1", "d2"}
