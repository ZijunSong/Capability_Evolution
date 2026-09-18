from __future__ import annotations

import os
from pathlib import Path

import pytest

from trim.cli.launch import parse_train_args
from trim.training.dist_runtime import (
    TRIM_CUDA_PARENT_VISIBLE_ENV,
    TRIM_CUDA_PHYSICAL_ENV,
    TRIM_CUDA_PINNED_ENV,
    TRIM_TRAIN_WORKER_ENV,
    DistLaunchConfig,
    format_launch_help,
    gather_sharded_objects,
    isolate_inference_child_env,
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
    monkeypatch.delenv(TRIM_CUDA_PINNED_ENV, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    payload = pin_local_cuda_device()
    assert payload["pinned"] is False
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"


def test_pin_local_cuda_device_nproc8(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    monkeypatch.delenv(TRIM_CUDA_PINNED_ENV, raising=False)
    monkeypatch.delenv(TRIM_CUDA_PARENT_VISIBLE_ENV, raising=False)
    monkeypatch.delenv(TRIM_CUDA_PHYSICAL_ENV, raising=False)
    payload = pin_local_cuda_device()
    assert payload["pinned"] is True
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"
    again = pin_local_cuda_device()
    assert again["pinned"] is True
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"
    assert again["physical_id"] == "3"


def test_pin_local_cuda_device_repeat_rank1_does_not_index_error(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    monkeypatch.delenv(TRIM_CUDA_PINNED_ENV, raising=False)
    first = pin_local_cuda_device()
    assert first["cuda_visible_devices"] == "1"
    second = pin_local_cuda_device()
    assert second["cuda_visible_devices"] == "1"


def test_pin_rejects_bare_single_visible_device(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.delenv(TRIM_CUDA_PINNED_ENV, raising=False)
    with pytest.raises(RuntimeError, match="single-element"):
        pin_local_cuda_device()


def test_leftover_rank_world_size_does_not_count_as_worker(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    monkeypatch.delenv("TORCHELASTIC_RUN_ID", raising=False)
    monkeypatch.delenv(TRIM_TRAIN_WORKER_ENV, raising=False)
    monkeypatch.delenv("PET_NPROC_PER_NODE", raising=False)
    assert under_torchrun() is False
    cfg, _rest = parse_dist_argv(["--nproc-per-node", "8", "--training-backend", "verl"])
    assert cfg.already_worker is False
    assert cfg.expected_world_size == 8
    assert needs_torchrun(cfg) is True


def test_cloudml_pytorchjob_injection_does_not_skip_torchrun(monkeypatch):
    """CloudML init-pytorch injects RANK/WORLD_SIZE=1 and PET_NPROC_PER_NODE=auto."""
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setenv("PET_NPROC_PER_NODE", "auto")
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "cloudml-injected")
    monkeypatch.delenv(TRIM_TRAIN_WORKER_ENV, raising=False)
    assert under_torchrun() is False
    cfg, _rest = parse_dist_argv(["--nproc-per-node", "8", "--training-backend", "verl"])
    assert cfg.nproc_per_node == 8
    assert cfg.already_worker is False
    assert needs_torchrun(cfg) is True


def test_plan_a_world_size_gt_1_or_real_pet_nproc_is_worker(monkeypatch):
    monkeypatch.delenv(TRIM_TRAIN_WORKER_ENV, raising=False)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("PET_NPROC_PER_NODE", "auto")
    assert under_torchrun() is True
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("PET_NPROC_PER_NODE", "8")
    assert under_torchrun() is True


def test_complete_torchrun_env_is_worker(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "elastic-test")
    monkeypatch.delenv(TRIM_TRAIN_WORKER_ENV, raising=False)
    monkeypatch.delenv("PET_NPROC_PER_NODE", raising=False)
    assert under_torchrun() is True


def test_isolate_inference_child_env_drops_rank_keeps_cuda(monkeypatch):
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "elastic-test")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setenv(TRIM_TRAIN_WORKER_ENV, "1")
    child = isolate_inference_child_env()
    assert child["CUDA_VISIBLE_DEVICES"] == "3"
    assert "RANK" not in child
    assert "WORLD_SIZE" not in child
    assert TRIM_TRAIN_WORKER_ENV not in child


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
