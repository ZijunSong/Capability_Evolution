#!/usr/bin/env python3
"""Multi-node / multi-GPU TRIM RL entry.

``scripts/run_train.py`` is the single-node Scheme A loop: one process owns
every visible GPU, starts vLLM (tensor-parallel), rolls out, stops vLLM, then
loads HF for the optimizer step. That cannot span machines.

This launcher starts **one Scheme A process per node** (default
``--nproc-per-node 1``). Each rank:

1. Rolls out a round-robin shard of the current query groups on its local GPUs.
2. Writes the shard to the shared ``--out`` directory.
3. Rank 0 concatenates shards and runs the existing HF CISPO/OPD step.
4. Rank 0 writes the LoRA adapter; every rank barriers and reloads it for the
   next on-policy rollout.

``--out`` must be a shared filesystem (NFS/GPFS/lustre). Model weights, the
SEC BM25 index, and ``--rl-data`` must also be visible on every node.

Qwen-4B (fits one GPU) — 4 nodes × 8 cards, one actor per card::

    # identical command on every node, only --node-rank changes
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    PYTHONPATH=TRIM:SCAPE-EasyOPD python TRIM/scripts/run_train_multinode.py \\
      --nnodes 4 --nproc-per-node 1 --node-rank $NODE_RANK \\
      --master-addr $MASTER_ADDR --master-port 29500 \\
      --rollout-replicas 8 --tensor-parallel-size 1 \\
      --harness Harness-1 --benchmark bcplus_full \\
      --model_name /mnt/songzijun/models/Qwen3-4B-Instruct-2507 \\
      --train_method trim --component all \\
      --train-env local_legacy --train-data sec \\
      --train-steps 16 --train-groups-per-step 32 --group-size 8 \\
      --train-micro-batch-size 4 --max-turns 40 --max-model-len 8192 \\
      --out "$OUT"

20B-class (needs tensor parallel) — one TP=8 vLLM per node, omit replicas::

    python TRIM/scripts/run_train_multinode.py \\
      --nnodes 4 --nproc-per-node 1 --node-rank $NODE_RANK \\
      --master-addr $MASTER_ADDR --tensor-parallel-size 8 \\
      ...same train flags...

Slurm (1 task per node)::

    srun --nodes=4 --ntasks-per-node=1 --gpus-per-node=8 \\
      python TRIM/scripts/run_train_multinode.py --nnodes 4 ...

Passwordless SSH from node 0::

    python TRIM/scripts/run_train_multinode.py \\
      --nnodes 4 --hosts host0,host1,host2,host3 --ssh-launch \\
      --master-addr host0 ...
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

_TRIM = Path(__file__).resolve().parents[1]
_SCRIPT = str(Path(__file__).resolve())
if str(_TRIM) not in sys.path:
    sys.path.insert(0, str(_TRIM))

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


def _forward_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    pythonpath = env.get("PYTHONPATH", "")
    parts = [str(_TRIM)]
    scape = _TRIM.parent / "SCAPE-EasyOPD"
    if scape.is_dir():
        parts.append(str(scape))
    if pythonpath:
        parts.append(pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def _exec_torchrun(cfg: DistLaunchConfig, train_argv: list[str], *, node_rank: int) -> int:
    cmd = torchrun_cmd(script=_SCRIPT, train_argv=train_argv, cfg=cfg, node_rank=node_rank)
    env = _forward_env()
    env[DIST_BACKEND_ENV] = cfg.dist_backend
    env["MASTER_ADDR"] = str(cfg.master_addr)
    env["MASTER_PORT"] = str(int(cfg.master_port))
    print("[dist-launch] " + " ".join(cmd), flush=True)
    os.execvpe(cmd[0], cmd, env)
    return 1


def _ssh_launch(cfg: DistLaunchConfig, train_argv: list[str]) -> int:
    env = _forward_env()
    env[DIST_BACKEND_ENV] = cfg.dist_backend
    rows = per_node_launch_commands(script=_SCRIPT, train_argv=train_argv, cfg=cfg)
    cwd = os.getcwd()
    procs: list[tuple[int, str, subprocess.Popen]] = []
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
        remote = prefix + " && " + " ".join(shlex.quote(x) for x in cmd)
        print(f"[dist-launch] ssh {host} node_rank={rank}", flush=True)
        procs.append(
            (
                rank,
                host,
                subprocess.Popen(["ssh", "-o", "StrictHostKeyChecking=accept-new", host, remote]),
            )
        )
    rc = 0
    for rank, host, proc in procs:
        got = int(proc.wait())
        if got != 0:
            print(f"[dist-launch] host={host} node_rank={rank} exited {got}", flush=True)
            rc = got or rc
    return rc


def _worker(train_argv: list[str], cfg: DistLaunchConfig) -> int:
    pin = pin_local_cuda_device()
    os.environ[DIST_BACKEND_ENV] = cfg.dist_backend
    os.environ["TRIM_ROLLOUT_REPLICAS"] = str(int(cfg.rollout_replicas))
    info = init_dist_if_needed(backend=cfg.dist_backend)
    print(
        json_dumps(
            {
                "event": "dist_worker",
                "rank": info.rank,
                "world_size": info.world_size,
                "local_rank": info.local_rank,
                "nnodes": cfg.nnodes,
                "nproc_per_node": cfg.nproc_per_node,
                "rollout_replicas": cfg.rollout_replicas,
                "cuda_pin": pin,
            }
        ),
        flush=True,
    )
    from trim.cli.launch import LaunchError, parse_train_args
    from trim.eval.offline_credentials import ensure_local_offline_credentials
    from trim.training.gpu_keepalive import acquire_keepalive, release_keepalive
    from TRIM_run_train import run_parsed_train

    ensure_local_offline_credentials()
    try:
        args, spec = parse_train_args(train_argv)
    except LaunchError as exc:
        raise SystemExit(str(exc)) from exc
    args.rollout_replicas = int(cfg.rollout_replicas)
    args.dist_nnodes = int(cfg.nnodes)
    args.dist_nproc_per_node = int(cfg.nproc_per_node)
    acquire_keepalive()
    try:
        return run_parsed_train(args, spec)
    finally:
        release_keepalive()


def json_dumps(payload: dict) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, default=str)


def _import_run_train():
    """Load scripts/run_train.py as a module without executing ``main``."""
    import importlib.util

    path = _TRIM / "scripts" / "run_train.py"
    spec = importlib.util.spec_from_file_location("TRIM_run_train", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["TRIM_run_train"] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    cfg, train_argv = parse_dist_argv(raw)
    if cfg.already_worker or under_torchrun():
        _import_run_train()
        return _worker(train_argv, cfg)

    if cfg.ssh_launch:
        if not cfg.hosts:
            raise SystemExit("--ssh-launch requires --hosts host0,host1,...")
        if len(cfg.hosts) != int(cfg.nnodes):
            raise SystemExit(f"--hosts has {len(cfg.hosts)} entries, --nnodes={cfg.nnodes}")
        return _ssh_launch(cfg, train_argv)

    if needs_torchrun(cfg):
        if int(cfg.nnodes) > 1 and cfg.node_rank is None:
            rows = per_node_launch_commands(script=_SCRIPT, train_argv=train_argv, cfg=cfg)
            raise SystemExit(
                format_launch_help(rows)
                + "Pass --node-rank on each machine, or --hosts ... --ssh-launch from node 0."
            )
        node_rank = 0 if cfg.node_rank is None else int(cfg.node_rank)
        return _exec_torchrun(cfg, train_argv, node_rank=node_rank)

    _import_run_train()
    return _worker(train_argv, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
