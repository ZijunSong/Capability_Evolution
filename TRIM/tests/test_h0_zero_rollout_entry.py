import importlib.util
from pathlib import Path

import pytest


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_h0_zero_rollout.py"
    spec = importlib.util.spec_from_file_location("run_h0_zero_rollout", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_geometry_is_32_cards_and_eval_stops():
    mod = _load()
    assert mod.world_size(mod.DEFAULT_NNODES, mod.DEFAULT_REPLICAS_PER_NODE) == 32
    argv = mod.build_eval_argv(
        model_name="/models/base",
        out=Path("/tmp/node0"),
        query_ids=Path("/tmp/ids.txt"),
        replicas_per_node=8,
        tensor_parallel_size=1,
        extra=["--max-turns", "40"],
    )
    assert argv[argv.index("--benchmark") + 1] == "bcplus_full"
    assert argv[argv.index("--component") + 1] == "zero"
    assert argv[argv.index("--evaluation-path") + 1] == "legacy_local"
    assert "--train_method" not in argv
    assert argv[argv.index("--tp") + 1] == "8"


def test_nodes_cover_each_query_once():
    mod = _load()
    rows = [{"query_id": f"q{i}"} for i in range(32)]
    seen = []
    for rank in range(4):
        owned = mod.node_rows(rows, nnodes=4, replicas_per_node=8, node_rank=rank)
        assert len(owned) == 8
        seen.extend(row["query_id"] for row in owned)
    assert seen == [f"q{i}" for i in range(32)]


def test_component_override_is_rejected():
    mod = _load()
    with pytest.raises(SystemExit):
        mod._parse(["--model_name", "m", "--out", "/tmp/o", "--component", "all"])


def test_merge_reports_incomplete_without_starting_another_stage(tmp_path: Path):
    mod = _load()
    node = tmp_path / "node0" / "harness"
    node.mkdir(parents=True)
    (node / "PER_QUERY.jsonl").write_text(
        '{"query_id":"q0","final_answer_recall":0.5,"recall":0.5}\n',
        encoding="utf-8",
    )
    summary = mod.merge_zero_rollout([tmp_path / "node0"], expected_ids=["q0", "q1"])
    assert summary["component"] == "zero"
    assert summary["benchmark"] == "bcplus_full"
    assert summary["stop_after"] == "eval"
    assert summary["next_stage"] is None
    assert summary["execution_complete"] is False
    assert summary["final_answer_recall"] == 0.5
