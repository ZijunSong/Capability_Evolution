"""Harness-G zero-rl and all-trim use the graph env, not the Harness-1 probe."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from trim.adapters.components import all_component_ids, zero_mask
from trim.cli.launch import parse_train_args
from trim.eval.harness_g_env import execute_tool, new_state, wm_text
from trim.eval.harness_g_graph import build_graph_from_documents, save_graph_index
from trim.eval.runtime_effect_audit import audit_train_runtime_or_raise
from trim.training.four_cell_runtime import terminal_reward_breakdown
from trim.training.harness_g_train import load_train_graph
from trim.training.rl_opd_types import (
    OPD_LOSS_PROJECTED_GAP,
    TRAINING_MODE_RL,
    TRAINING_MODE_SCAPE_SEED,
)


def _parse(method: str, component: str):
    return parse_train_args(
        [
            "--harness",
            "Harness-G",
            "--train_method",
            method,
            "--component",
            component,
            "--out",
            f"/tmp/hg-{method}-{component}",
        ]
    )


def test_zero_rl_and_all_trim_cli_contract():
    rl_args, rl_spec = _parse("rl", "zero")
    trim_args, trim_spec = _parse("trim", "all")
    assert rl_spec.harness == "Harness-G"
    assert rl_spec.training_mode == TRAINING_MODE_RL
    assert rl_spec.components == ()
    assert rl_args.train_data == "bcplus_train_664"
    assert rl_args.retrieval_backend == "local_bm25"
    assert rl_args.n_queries == 664
    assert str(rl_args.graph_index_path or "").endswith("harness_g_corpus_graph.pkl")

    assert trim_spec.training_mode == TRAINING_MODE_SCAPE_SEED
    assert set(trim_spec.components) == set(all_component_ids("Harness-G"))
    assert trim_args.opd_loss == OPD_LOSS_PROJECTED_GAP
    assert float(trim_args.lambda_opd) == 0.01
    assert int(trim_args.opd_states_per_trajectory) == -1
    assert trim_args.train_data == "bcplus_train_664"
    student = zero_mask("Harness-G")
    assert all(v is False for v in student.values())

    explicit, spec = parse_train_args(
        [
            "--harness",
            "Harness-G",
            "--train_method",
            "rl",
            "--component",
            "zero",
            "--train-data",
            "sec",
            "--out",
            "/tmp/hg-explicit-sec",
        ]
    )
    assert explicit.train_data == "sec"
    assert spec.harness == "Harness-G"


def test_audit_zero_rl_skips_projection_and_all_trim_projects(tmp_path):
    rl_args, _rl_spec = _parse("rl", "zero")
    rl_args.lambda_opd = 0.0
    rl_audit = audit_train_runtime_or_raise(rl_args, out=tmp_path / "zero_rl")
    assert rl_audit["pass"] is True
    assert rl_audit["skipped"] is False
    assert rl_audit["opd_projection"] is None
    assert int(rl_audit["student_n_on"]) == 0
    assert int(rl_audit["teacher_n_on"]) == 0
    assert (tmp_path / "zero_rl" / "RUNTIME_EFFECT_AUDIT.json").is_file()

    trim_args, _trim_spec = _parse("trim", "all")
    trim_audit = audit_train_runtime_or_raise(trim_args, out=tmp_path / "all_trim")
    assert trim_audit["pass"] is True
    assert int(trim_audit["student_n_on"]) == 0
    assert int(trim_audit["teacher_n_on"]) == len(all_component_ids("Harness-G"))
    assert int(trim_audit["opd_n_projected_steps"] or 0) >= 1
    assert (trim_audit["teacher_mask"] or {})["hybrid_init_retrieve"] is True
    assert (trim_audit["student_mask"] or {})["hybrid_init_retrieve"] is False


def test_load_train_graph_fail_closed_and_smoke(monkeypatch, tmp_path):
    monkeypatch.setattr("trim.training.harness_g_train.default_graph_path", lambda: None)
    missing = SimpleNamespace(harness="Harness-G", component="zero", smoke=False, graph_index_path=None)
    with pytest.raises(SystemExit, match="corpus graph"):
        load_train_graph(missing)
    smoke = SimpleNamespace(harness="Harness-G", component="zero", smoke=True, graph_index_path=None)
    assert load_train_graph(smoke) is None
    h1 = SimpleNamespace(harness="Harness-1", component="all", smoke=False, graph_index_path=None)
    assert load_train_graph(h1) is None

    store = {"d1": {"id": "d1", "text": "Alice Smith visited Paris."}}
    graph = build_graph_from_documents(store, scope="corpus")
    path = tmp_path / "tiny.pkl"
    save_graph_index(graph, path)
    loaded = load_train_graph(
        SimpleNamespace(
            harness="Harness-G",
            component="answer_with",
            smoke=False,
            graph_index_path=str(path),
        )
    )
    assert loaded.scope == "corpus"
    assert str(loaded.source_path) == str(path)


def test_selected_gold_doc_is_the_rl_task_reward():
    mask = zero_mask("Harness-G")
    store = {"gold": {"id": "gold", "text": "Alice Smith visited Paris in 2019 for the treaty."}}
    st = new_state("Alice Smith Paris", store, harness_mask=mask)
    st, _obs, ok = execute_tool(st, "init", {})
    assert ok
    select = next(
        action
        for action in (st.get("action_map") or {}).values()
        if str(action.get("type") or "").upper() == "SELECT" and action.get("sid")
    )
    st, _obs, ok = execute_tool(st, "select", {"sid": select["sid"]})
    assert ok
    parts = terminal_reward_breakdown(
        st,
        query="Alice Smith Paris",
        gold_ids=["gold"],
        valids=[True, True],
        actions=[
            {"name": "init", "arguments": {}},
            {"name": "select", "arguments": {"sid": select["sid"]}},
        ],
    )
    assert parts["task_recall"] == 1.0
    assert parts["total"] > 0.1
    text = wm_text(st)
    assert "Harness-G Working Memory" in text
    assert "search_corpus" not in text
