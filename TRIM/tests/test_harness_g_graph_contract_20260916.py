from pathlib import Path
import json
import subprocess
import sys

from trim.eval.harness_g_contract import (
    is_corpus_scope,
    is_formal_harness_g_eval,
    require_graph_path,
    validate_loaded_graph,
)
from trim.eval.harness_g_env import execute_tool, new_state
from trim.eval.harness_g_graph import build_graph_from_documents, save_graph_index
from trim.eval.harness1_metrics import episode_quality_metrics
from trim.adapters.harness_profiles import full_mask_for, zero_mask_for

_TRIM = Path(__file__).resolve().parents[1]
_PY = sys.executable


def _zero():
    return zero_mask_for("Harness-G")


class _Hit:
    def __init__(self, docid: str, text: str):
        self.docid = docid
        self.text = text
        self.score = 1.0


class _InitOnlySearcher:
    name = "fake_bm25"

    def __init__(self, store: dict):
        self.store = store

    def search(self, query: str, k: int = 10):
        del query, k
        return [_Hit("noise", self.store["noise"]["text"]), _Hit("bridge", self.store["bridge"]["text"])]


def _corpus():
    return {
        "noise": {"text": "Unrelated weather notes from Lyon."},
        "bridge": {"text": "Alice Smith met Bob Jones at the archive."},
        "target": {"text": "Bob Jones recorded that the treaty was signed in 1842."},
    }


def test_official_eval_requires_graph_path():
    assert is_formal_harness_g_eval(harness="Harness-G", benchmark="bcplus_test_50", smoke=False)
    assert not is_formal_harness_g_eval(harness="Harness-G", benchmark="bcplus_test_50", smoke=True)
    try:
        require_graph_path("", required=True)
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "graph-index-path" in str(exc)


def test_episode_scope_rejected_for_official_graph(tmp_path: Path):
    graph = build_graph_from_documents(_corpus(), scope="episode_doc_store")
    path = tmp_path / "episode.pkl"
    save_graph_index(graph, path)
    from trim.eval.harness_g_graph import load_graph_index

    loaded = load_graph_index(path)
    try:
        validate_loaded_graph(loaded, required=True, path=str(path))
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "non-corpus" in str(exc)


def test_corpus_scope_accepted(tmp_path: Path):
    graph = build_graph_from_documents(_corpus(), scope="corpus")
    path = tmp_path / "corpus.pkl"
    save_graph_index(graph, path)
    from trim.eval.harness_g_graph import load_graph_index

    loaded = load_graph_index(path)
    meta = validate_loaded_graph(loaded, required=True, path=str(path))
    assert is_corpus_scope(meta["graph_scope"])
    assert meta["graph_fingerprint"]
    assert meta["graph_num_docs"] == 3


def test_deterministic_graph_lookup_reaches_gold_without_llm():
    corpus = _corpus()
    graph = build_graph_from_documents(corpus, scope="corpus")
    st = new_state(
        "Who is Alice Smith?",
        {"noise": corpus["noise"], "bridge": corpus["bridge"]},
        harness_mask=_zero(),
        graph_index=graph,
        graph_index_path="/tmp/corpus.pkl",
    )
    assert st["graph_enabled"] is True
    assert st["graph_scope"] == "corpus"
    searcher = _InitOnlySearcher(corpus)
    st, _, ok = execute_tool(st, "init", {}, searcher=searcher, search_k=2)
    assert ok is True
    assert not any(str(sid).startswith("target:") for sid in st["visible_sids"])
    # LOOKUP Alice then Bob without selecting the gold page first.
    alice = next(a.get("eid") for a in st["action_map"].values() if a.get("eid") in {"e:alice", "e:alice_smith"})
    st, _, ok = execute_tool(st, "lookup", {"eid": alice}, searcher=searcher)
    assert ok is True
    bob = next(
        a.get("eid")
        for a in st["action_map"].values()
        if str(a.get("eid") or "") in {"e:bob", "e:bob_jones"}
    )
    st, _, ok = execute_tool(st, "lookup", {"eid": bob}, searcher=searcher)
    assert ok is True
    effects = st.get("runtime_effects") or {}
    assert int(effects.get("graph_lookup_calls") or 0) > 0
    assert int(effects.get("graph_new_docs") or 0) > 0
    gold_sids = [sid for sid in st["visible_sids"] if str(sid).startswith("target:")]
    assert gold_sids, st["visible_sids"]
    st, _, ok = execute_tool(st, "select", {"sid": gold_sids[0]})
    assert ok is True
    stats = episode_quality_metrics(
        st,
        {"gold_docids": ["target"], "evidence_docids": ["target", "bridge"]},
        tool_names=["init", "lookup", "lookup", "select"],
        valids=[True, True, True, True],
        reward=0.0,
    )
    assert "target" in (st.get("selected_docids") or [])
    assert "target" in (st.get("observed_docids") or [])
    assert stats["recall"] > 0
    assert stats["graph_enabled"] is True
    assert int(effects.get("lexical_lookup_calls") or 0) == 0


def test_metric_auditor_matches_episode_metrics(tmp_path: Path):
    from trim.eval.harness_g_contract import audit_trace_metrics

    selected = {"86987", "88633"}
    gold = {"86987", "99136"}
    row = {
        "query_id": "1127",
        "selected_docids": list(selected),
        "observed_docids": list(selected),
        "gold_docids": list(gold),
        "evidence_docids": list(gold),
        "recall": 0.5,
        "trajectory_recall": 0.5,
    }
    assert audit_trace_metrics(row) == []
    row["recall"] = 1.0
    assert audit_trace_metrics(row)


def test_zero_and_all_share_the_same_corpus_graph():
    corpus = _corpus()
    graph = build_graph_from_documents(corpus, scope="corpus")
    store = {"noise": corpus["noise"], "bridge": corpus["bridge"]}
    zero = new_state("Who is Alice Smith?", store, harness_mask=_zero(), graph_index=graph)
    full = new_state("Who is Alice Smith?", store, harness_mask=full_mask_for("Harness-G"), graph_index=graph)
    assert zero["graph_scope"] == full["graph_scope"] == "corpus"
    assert zero["graph_fingerprint"] == full["graph_fingerprint"]
    assert zero["graph_enabled"] is True
    assert full["graph_enabled"] is True


def test_official_shells_require_graph_index_path():
    for name in (
        "run_bcplus_test50_harness_g_gpu04567_eval.sh",
        "run_bcplus_test50_harness_g_gpu34567_parallel.sh",
    ):
        text = (_TRIM / "scripts" / name).read_text(encoding="utf-8")
        assert "GRAPH_INDEX_PATH:?" in text
        assert '${GRAPH_INDEX_PATH:+--graph-index-path' not in text
        assert '--graph-index-path "${GRAPH_INDEX_PATH}"' in text


def test_validate_official_run_and_audit_scripts(tmp_path: Path):
    from trim.eval.harness_g_official import validate_official_run

    corpus = _corpus()
    graph = build_graph_from_documents(corpus, scope="corpus")
    graph_path = tmp_path / "corpus.pkl"
    save_graph_index(graph, graph_path)
    fp = graph.content_fingerprint()
    run = tmp_path / "run"
    run.mkdir()
    (run / "LAUNCH.json").write_text(
        json.dumps(
            {
                "graph_index_path": str(graph_path),
                "graph_scope": "corpus",
                "graph_fingerprint": fp,
                "graph_required": True,
                "graph_enabled": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run / "PER_QUERY.jsonl").write_text(
        json.dumps(
            {
                "query_id": "q1",
                "gold_docids": ["target"],
                "selected_docids": ["target"],
                "observed_docids": ["bridge", "target"],
                "evidence_docids": ["target", "bridge"],
                "recall": 1.0,
                "trajectory_recall": 1.0,
                "graph_enabled": True,
                "graph_scope": "corpus",
                "graph_fingerprint": fp,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = validate_official_run(run)
    assert result["ok"] is True, result

    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "query": "Who is Alice Smith?",
                "gold_docids": ["target"],
                "evidence_docids": ["target", "bridge"],
                "initial_docids": ["noise", "bridge"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    static = subprocess.run(
        [
            _PY,
            str(_TRIM / "scripts" / "audit_harness_g_graph.py"),
            "--graph-index-path",
            str(graph_path),
            "--queries-jsonl",
            str(queries),
            "--out",
            str(tmp_path / "static.json"),
        ],
        cwd=str(_TRIM),
        capture_output=True,
        text=True,
        check=False,
    )
    assert static.returncode == 0, static.stdout + static.stderr
    static_payload = json.loads((tmp_path / "static.json").read_text(encoding="utf-8"))
    assert static_payload["gold_doc_coverage"] == 1.0
    assert static_payload["corpus_scope_ok"] is True

    reach = subprocess.run(
        [
            _PY,
            str(_TRIM / "scripts" / "audit_harness_g_graph_reachability.py"),
            "--graph-index-path",
            str(graph_path),
            "--queries-jsonl",
            str(queries),
            "--skip-retrieval",
            "--max-hops",
            "2",
            "--out",
            str(tmp_path / "reach.json"),
        ],
        cwd=str(_TRIM),
        capture_output=True,
        text=True,
        check=False,
    )
    assert reach.returncode == 0, reach.stdout + reach.stderr
    reach_payload = json.loads((tmp_path / "reach.json").read_text(encoding="utf-8"))["summary"]
    assert reach_payload["initial_recall"] == 0.0
    assert reach_payload["oracle_beats_initial"] is True
    assert reach_payload["initial_miss_graph_reachable"] >= 1

    metrics = subprocess.run(
        [
            _PY,
            str(_TRIM / "scripts" / "audit_harness_g_metrics.py"),
            "--per-query",
            str(run / "PER_QUERY.jsonl"),
        ],
        cwd=str(_TRIM),
        capture_output=True,
        text=True,
        check=False,
    )
    assert metrics.returncode == 0, metrics.stdout + metrics.stderr
    assert "PASS" in metrics.stdout

    validator = subprocess.run(
        [
            _PY,
            str(_TRIM / "scripts" / "validate_harness_g_official_run.py"),
            "--run-dir",
            str(run),
        ],
        cwd=str(_TRIM),
        capture_output=True,
        text=True,
        check=False,
    )
    assert validator.returncode == 0, validator.stdout + validator.stderr
    assert "OFFICIAL_RUN_VALID" in validator.stdout


def test_init_ranks_only_retrieved_docs():
    corpus = _corpus()
    graph = build_graph_from_documents(corpus, scope="corpus")
    ranked = graph.rank_init_sids("treaty signed 1842", topk=6, doc_order=["noise", "bridge"])
    assert ranked
    assert all(not str(sid).startswith("target:") for sid in ranked)


def test_corpus_graph_not_mutated_by_snippets():
    corpus = _corpus()
    graph = build_graph_from_documents(corpus, scope="corpus")
    fp = graph.content_fingerprint()
    n_sents = len(graph.sentences)
    noise_text = graph.docs["noise"]["text"]
    st = new_state(
        "Who is Alice Smith?",
        {"noise": {"text": "SHORT SNIPPET THAT IS DIFFERENT"}},
        harness_mask=_zero(),
        graph_index=graph,
    )
    assert st["graph"] is graph
    assert graph.content_fingerprint() == fp
    assert len(graph.sentences) == n_sents
    assert graph.docs["noise"]["text"] == noise_text
