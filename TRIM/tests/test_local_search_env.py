from __future__ import annotations

from trim.eval.browsecomp_retrieval import SearchHit
from trim.eval.local_search_env import execute_tool, new_state


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


def test_execute_tool_empty_query_falls_back_to_question():
    st = new_state(
        "capital of France",
        {"d1": {"id": "d1", "text": "Paris is the capital of France."}},
    )
    st, _obs, ok = execute_tool(st, "search_corpus", {})
    assert ok is True
    assert "d1" in st["pool"]


def test_execute_tool_query_alias_q():
    st = new_state("unused", {})
    st, _obs, ok = execute_tool(
        st,
        "search_corpus",
        {"q": "Apple FY2023 10-K"},
        searcher=_LiveSearcher(),
        search_k=10,
    )
    assert ok is True
    assert "gold" in st["pool"]


class _EmptyLive:
    name = "pyserini_lucene"

    def search(self, query: str, k: int = 5):
        del query, k
        return []


def test_execute_tool_live_empty_falls_back_to_seeded_store():
    st = new_state(
        "capital of France",
        {
            "d1": {"id": "d1", "text": "Paris is the capital of France."},
            "d2": {"id": "d2", "text": "unrelated sports scores"},
        },
    )
    st, obs, ok = execute_tool(
        st,
        "search_corpus",
        {"query": "capital of France"},
        searcher=_EmptyLive(),
        search_k=10,
    )
    assert ok is True
    assert "d1" in st["pool"]
    assert "Paris" in obs


def test_wm_text_shows_recent_pool_and_older_ids():
    from trim.eval.local_search_env import wm_text

    store = {f"d{i}": {"id": f"d{i}", "text": f"body {i}", "score": float(i)} for i in range(20)}
    st = new_state("q", store)
    st["pool"] = dict(store)
    st["curated"] = {"d0": store["d0"]}
    text = wm_text(st, auto_on=False)
    assert "d19" in text
    assert "[ ] d19:" in text
    assert "Earlier uncurated" in text or "d1" in text
    assert "(empty -- use curate" not in text


def test_search_observation_asks_to_curate():
    st = new_state("q", {"d1": {"id": "d1", "text": "Paris is the capital of France."}})
    st, obs, ok = execute_tool(st, "search_corpus", {"query": "capital of France"})
    assert ok is True
    assert "ACTION REQUIRED" in obs
    assert "curate" in obs.lower()


def test_curate_accepts_scalar_remove_and_add_ids():
    st = new_state(
        "q",
        {"d1": {"id": "d1", "text": "keep me"}, "d2": {"id": "d2", "text": "drop me"}},
    )
    st["pool"] = dict(st["doc_store"])
    st, _obs, ok = execute_tool(st, "curate", {"add_ids": "d1", "remove_ids": 2})
    assert ok is True
    assert "d1" in st["curated"]
    st, _obs, ok = execute_tool(st, "curate", {"remove_ids": "d1"})
    assert ok is True
    assert "d1" not in st["curated"]
