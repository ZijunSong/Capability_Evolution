"""Scheme A launcher for the three SFT entries.

Four coordinator ranks, eight rollout replicas per node, tensor parallel 1:
32 cards. Rank 0 is the only gradient rank. This module does not import
``TRIM_run_train`` or the old train parser.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from trim.training.dist_runtime import (
    DIST_BACKEND_ENV,
    DistLaunchConfig,
    format_launch_help,
    init_dist_if_needed,
    needs_torchrun,
    parse_dist_argv,
    per_node_launch_commands,
    pin_local_cuda_device,
    torchrun_cmd,
    under_torchrun,
)
from trim.training.sft_ablation.cli import parse_sft_args
from trim.training.sft_ablation.geometry import (
    EXPERIMENT_MASTER_PORT,
    OFFICIAL_NNODES,
    OFFICIAL_NPROC_PER_NODE,
    OFFICIAL_ROLLOUT_REPLICAS,
    flag_present,
    planned_visible_ids,
    replica_gpu_plan,
    world_cards,
)
from trim.training.sft_ablation.io_util import json_dumps

_TRIM = Path(__file__).resolve().parents[2]


def _forward_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    parts = [str(_TRIM)]
    scape = _TRIM.parent / "SCAPE-EasyOPD"
    if scape.is_dir():
        parts.append(str(scape))
    if env.get("PYTHONPATH"):
        parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def apply_official_defaults(cfg: DistLaunchConfig, argv: list[str], experiment: str) -> DistLaunchConfig:
    """Outer-launcher defaults: 4 nodes x 8 replicas. Workers must not call this."""
    if not flag_present(argv, "--nnodes"):
        cfg.nnodes = OFFICIAL_NNODES
    if not flag_present(argv, "--nproc-per-node", "--nproc_per_node"):
        cfg.nproc_per_node = OFFICIAL_NPROC_PER_NODE
    if not flag_present(argv, "--rollout-replicas"):
        cfg.rollout_replicas = OFFICIAL_ROLLOUT_REPLICAS
    if not flag_present(argv, "--rdzv-id"):
        cfg.rdzv_id = experiment.replace("_", "-")
    if not flag_present(argv, "--master-port", "--master_port") and not os.environ.get("MASTER_PORT"):
        cfg.master_port = EXPERIMENT_MASTER_PORT[experiment]
    if not flag_present(argv, "--expected-world-size"):
        cfg.expected_world_size = int(cfg.nnodes) * int(cfg.nproc_per_node)
    return cfg


def validate_dist(cfg: DistLaunchConfig, *, tensor_parallel: int, bind_gpus: bool) -> dict[str, Any]:
    if int(cfg.nproc_per_node) != OFFICIAL_NPROC_PER_NODE:
        raise SystemExit(
            "SFT v1 refuses nproc-per-node other than 1. "
            "32 cards means 4 coordinator processes and 8 rollout replicas per node."
        )
    if int(cfg.rollout_replicas) < 1 or int(tensor_parallel) < 1:
        raise SystemExit("rollout-replicas and tensor-parallel-size must be positive")
    cards = world_cards(cfg.nnodes, cfg.rollout_replicas, tensor_parallel)
    per_node = int(cfg.rollout_replicas) * int(tensor_parallel)
    if bind_gpus:
        visible = planned_visible_ids(cfg.rollout_replicas, tensor_parallel)
        gpu_plan = replica_gpu_plan(
            visible,
            replicas=int(cfg.rollout_replicas),
            tensor_parallel=int(tensor_parallel),
        )
    else:
        gpu_plan = [
            [str(index) for index in range(replica * int(tensor_parallel), (replica + 1) * int(tensor_parallel))]
            for replica in range(int(cfg.rollout_replicas))
        ]
    return {
        "nnodes": int(cfg.nnodes),
        "nproc_per_node": int(cfg.nproc_per_node),
        "rollout_replicas": int(cfg.rollout_replicas),
        "tensor_parallel_size": int(tensor_parallel),
        "coordinator_world_size": int(cfg.nnodes) * int(cfg.nproc_per_node),
        "gradient_ranks": 1,
        "world_cards": cards,
        "cards_per_node": per_node,
        "replica_gpus": gpu_plan,
        "gpu_binding": "enforced" if bind_gpus else "planned",
        "official_32_card_layout": cards == 32 and int(cfg.nproc_per_node) == 1,
    }


def dry_run_plan(experiment: str, args: Any, cfg: DistLaunchConfig) -> dict[str, Any]:
    from trim.training.sft_ablation.contracts import PROFILES

    profile = PROFILES[experiment]
    topology = validate_dist(cfg, tensor_parallel=int(args.tensor_parallel_size), bind_gpus=False)
    return {
        "event": "sft_dry_run",
        "experiment": experiment,
        "state_source": profile.state_source,
        "prompt_owner": profile.prompt_owner,
        "supervision_provider": profile.supervision_provider,
        "teacher_mask": "all",
        "student_mask": "zero",
        "eval_harness": profile.eval_harness,
        "loss": profile.loss,
        "rl_enabled": False,
        "gap_enabled": False,
        "gate_enabled": False,
        "reward_used_for_optimization": False,
        "stage": args.stage,
        "online_refresh": profile.online_refresh,
        "projection": profile.projection,
        "rdzv_id": cfg.rdzv_id,
        "master_port": int(cfg.master_port),
        **topology,
    }


def _exec_torchrun(script: Path, cfg: DistLaunchConfig, train_argv: list[str], *, node_rank: int) -> int:
    cmd = torchrun_cmd(script=script, train_argv=train_argv, cfg=cfg, node_rank=node_rank)
    env = _forward_env()
    env[DIST_BACKEND_ENV] = cfg.dist_backend
    env["MASTER_ADDR"] = str(cfg.master_addr)
    env["MASTER_PORT"] = str(int(cfg.master_port))
    print("[sft-dist] " + " ".join(cmd), flush=True)
    os.execvpe(cmd[0], cmd, env)
    return 1


def _ssh_launch(script: Path, cfg: DistLaunchConfig, train_argv: list[str]) -> int:
    env = _forward_env()
    rows = per_node_launch_commands(script=script, train_argv=train_argv, cfg=cfg)
    cwd = os.getcwd()
    procs = []
    prefix = (
        f"cd {shlex.quote(cwd)}"
        f" && export PYTHONPATH={shlex.quote(env['PYTHONPATH'])}"
        f" && export PYTHONUNBUFFERED=1"
        f" && export MASTER_ADDR={shlex.quote(str(cfg.master_addr))}"
        f" && export MASTER_PORT={shlex.quote(str(int(cfg.master_port)))}"
        f" && export {DIST_BACKEND_ENV}={shlex.quote(cfg.dist_backend)}"
    )
    cuda = str(os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if cuda:
        prefix += f" && export CUDA_VISIBLE_DEVICES={shlex.quote(cuda)}"
    for rank, host, cmd in rows:
        remote = prefix + " && " + " ".join(shlex.quote(part) for part in cmd)
        print(f"[sft-dist] ssh {host} node_rank={rank}", flush=True)
        procs.append((rank, host, subprocess.Popen(["ssh", "-o", "StrictHostKeyChecking=accept-new", host, remote])))
    rc = 0
    for rank, host, proc in procs:
        got = int(proc.wait())
        if got != 0:
            print(f"[sft-dist] host={host} node_rank={rank} exited {got}", flush=True)
            rc = got or rc
    return rc


def _worker(args: Any, cfg: DistLaunchConfig, worker: Callable[..., int], experiment: str) -> int:
    pin = pin_local_cuda_device()
    os.environ[DIST_BACKEND_ENV] = cfg.dist_backend
    os.environ["TRIM_ROLLOUT_REPLICAS"] = str(int(cfg.rollout_replicas))
    info = init_dist_if_needed(backend=cfg.dist_backend)
    topology = validate_dist(cfg, tensor_parallel=int(args.tensor_parallel_size), bind_gpus=True)
    print(
        json_dumps(
            {
                "event": "sft_worker",
                "experiment": experiment,
                "rank": info.rank,
                "world_size": info.world_size,
                "cuda_pin": pin,
                **topology,
            }
        ),
        flush=True,
    )
    try:
        code = int(worker(args, info))
    except Exception as exc:
        _publish_failure(Path(args.out), info.rank, exc)
        raise
    if code != 0:
        _publish_failure(Path(args.out), info.rank, RuntimeError(f"worker exit {code}"))
    return code


def _publish_failure(out: Path, rank: int, exc: BaseException) -> None:
    try:
        from trim.training.sft_ablation.io_util import atomic_write_json

        out.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            out / "FAILED.json",
            {"rank": int(rank), "error": f"{type(exc).__name__}: {exc}"},
        )
    except Exception:
        print(f"[sft-dist] rank={rank} failed: {exc}", flush=True)


def launch_sft(*, script: Path, experiment: str, worker: Callable[..., int], argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    cfg, rest = parse_dist_argv(raw)
    worker_proc = bool(cfg.already_worker or under_torchrun())
    if not worker_proc:
        cfg = apply_official_defaults(cfg, raw, experiment)
    if any(token in {"-h", "--help"} for token in raw):
        parse_sft_args(experiment, rest)
        return 0
    args = parse_sft_args(experiment, rest)
    if args.dry_run:
        if worker_proc:
            raise SystemExit("dry-run must finish in the launcher, before torchrun")
        print(json_dumps(dry_run_plan(experiment, args, cfg)), flush=True)
        return 0
    validate_dist(cfg, tensor_parallel=int(args.tensor_parallel_size), bind_gpus=True)
    if worker_proc:
        return _worker(args, cfg, worker, experiment)
    if cfg.ssh_launch:
        if not cfg.hosts or len(cfg.hosts) != int(cfg.nnodes):
            raise SystemExit("--ssh-launch requires one --hosts entry per node")
        return _ssh_launch(script, cfg, rest)
    if needs_torchrun(cfg):
        if int(cfg.nnodes) > 1 and cfg.node_rank is None:
            rows = per_node_launch_commands(script=script, train_argv=rest, cfg=cfg)
            raise SystemExit(
                format_launch_help(rows)
                + "Pass --node-rank on each machine, or --hosts ... --ssh-launch from node 0.\n"
                + "Official layout: 4 nodes x 8 rollout replicas x tp 1 = 32 cards, 1 gradient rank."
            )
        node_rank = 0 if cfg.node_rank is None else int(cfg.node_rank)
        return _exec_torchrun(script, cfg, rest, node_rank=node_rank)
    return _worker(args, cfg, worker, experiment)
