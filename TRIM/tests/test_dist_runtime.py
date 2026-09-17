from __future__ import annotations

import os
from pathlib import Path

import pytest

from trim.cli.launch import parse_train_args
from trim.training.dist_runtime import (
    DistLaunchConfig,
    format_launch_help,
    gather_sharded_objects,
    interleave_round_robin,
    needs_torchrun,
    parse_dist_argv,
    parse_host_list,
    per_node_launch_commands,
    pin_local_cuda_device,
    reset_dist_info_for_tests,
    shard_for_rank,
    torchrun_cmd,
    training_backend_from_argv,
    under_torchrun,
    visible_cuda_count,
)


def setup_function():
    reset_dist_info_for_tests()


def test_shard_and_interleave_restore_order():
    rows = [{"query_id": f"q{i}"} for i in range(10)]
    shards = [shard_for_rank(rows, rank=r, world_size=4) for r in range(4)]
    assert [len(s) for s in shards] == [3, 3, 2, 2]
    assert [row["query_id"] for row in shards[0]] == ["q0", "q4", "q8"]
    merged = interleave_round_robin(shards)
    assert merged == rows


def test_shard_world_one_keeps_all():
    rows = [1, 2, 3]
    assert shard_for_rank(rows, rank=0, world_size=1) == rows


def test_parse_dist_argv_leaves_train_flags():
    cfg, rest = parse_dist_argv(
        [
            "--nnodes",
            "4",
            "--nproc-per-node",
            "1",
            "--node-rank",
            "2",
            "--master-addr",
            "10.0.0.1",
            "--rollout-replicas",
            "8",
            "--harness",
            "Harness-1",
            "--train_method",
            "trim",
            "--component",
            "all",
            "--max-turns",
            "40",
            "--out",
            "/tmp/dist-out",
        ]
    )
    assert cfg.nnodes == 4
    assert cfg.nproc_per_node == 1
    assert cfg.node_rank == 2
    assert cfg.master_addr == "10.0.0.1"
    assert cfg.rollout_replicas == 8
    assert "--max-turns" in rest
    assert rest[rest.index("--max-turns") + 1] == "40"
    args, spec = parse_train_args(rest)
    assert args.max_turns == 40
    assert spec.train_method == "trim"


def test_parse_train_keeps_max_turns_40(tmp_path: Path):
    args, _spec = parse_train_args(
        [
            "--train_method",
            "trim",
            "--component",
            "all",
            "--max-turns",
            "40",
            "--out",
            str(tmp_path / "out"),
            "--validate-only",
        ]
    )
    assert args.max_turns == 40


def test_torchrun_cmd_multinode_includes_rdzv():
    cfg = DistLaunchConfig(nnodes=4, nproc_per_node=1, master_addr="host0", master_port=29500, rollout_replicas=8)
    cmd = torchrun_cmd(
        script="/data/ppnm/Capability_Evolution/TRIM/scripts/run_train_multinode.py",
        train_argv=["--train_method", "trim", "--component", "all"],
        cfg=cfg,
        node_rank=1,
    )
    assert cmd[:3] == [cmd[0], "-m", "torch.distributed.run"]
    assert "--nnodes" in cmd and "4" in cmd
    assert "--node_rank" in cmd and "1" in cmd
    assert "--rdzv_endpoint" in cmd
    assert cmd[cmd.index("--rdzv_endpoint") + 1] == "host0:29500"
    assert "--standalone" not in cmd
    assert "--already-worker" in cmd
    assert "--rollout-replicas" in cmd
    assert cmd[cmd.index("--rollout-replicas") + 1] == "8"


def test_torchrun_cmd_single_node_uses_standalone():
    cfg = DistLaunchConfig(nnodes=1, nproc_per_node=8)
    cmd = torchrun_cmd(script="run_train_multinode.py", train_argv=["--component", "zero"], cfg=cfg, node_rank=0)
    assert "--standalone" in cmd
    assert "--node_rank" not in cmd


def test_needs_torchrun():
    assert needs_torchrun(DistLaunchConfig(nnodes=4, already_worker=False)) is True
    assert needs_torchrun(DistLaunchConfig(nnodes=1, nproc_per_node=8, already_worker=False)) is True
    assert needs_torchrun(DistLaunchConfig(nnodes=1, nproc_per_node=1, already_worker=False)) is False
    assert needs_torchrun(DistLaunchConfig(nnodes=4, already_worker=True)) is False


def test_missing_node_rank_prints_per_host_commands():
    cfg = DistLaunchConfig(nnodes=2, hosts=["a", "b"], master_addr="a")
    rows = per_node_launch_commands(script="run_train_multinode.py", train_argv=["--component", "all"], cfg=cfg)
    text = format_launch_help(rows)
    assert "node_rank=0" in text
    assert "node_rank=1" in text
    assert "host=a" in text


def test_parse_host_list():
    assert parse_host_list("h0, h1,h2") == ["h0", "h1", "h2"]
    assert parse_host_list("") == []


def test_pin_local_cuda_device_single_proc(monkeypatch):
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    payload = pin_local_cuda_device()
    assert payload["pinned"] is False
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"


def test_pin_local_cuda_device_nproc8(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    payload = pin_local_cuda_device()
    assert payload["pinned"] is True
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"


def test_visible_cuda_count_and_backend_argv(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    assert visible_cuda_count() == 8
    assert training_backend_from_argv(["--training-backend", "verl"]) == "verl"
    assert training_backend_from_argv(["--foo"]) == "hf_debug"


def test_gather_world_one_is_identity(tmp_path: Path):
    groups = [{"query_id": "q0"}]
    assert gather_sharded_objects(groups, shard_dir=tmp_path, tag="step0") == groups


def test_multinode_entry_requires_node_rank_when_nnodes_gt_1():
    import importlib.util
    import sys

    path = Path(__file__).resolve().parents[1] / "scripts" / "run_train_multinode.py"
    spec = importlib.util.spec_from_file_location("run_train_multinode_entry", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules["run_train_multinode_entry"] = module
    spec.loader.exec_module(module)
    with pytest.raises(SystemExit) as exc:
        module.main(
            [
                "--nnodes",
                "4",
                "--rollout-replicas",
                "8",
                "--train_method",
                "trim",
                "--component",
                "all",
                "--max-turns",
                "40",
            ]
        )
    msg = str(exc.value)
    assert "node_rank=0" in msg
    assert "torch.distributed.run" in msg
    assert "--max-turns" in msg
    assert "40" in msg
