#!/usr/bin/env python3
"""Start vLLM, run factorized replay, stop vLLM."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.opd.vllm_server import start_vllm_server


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--port", type=int, default=8100)
    p.add_argument("--gpu", default="0")
    p.add_argument("--use-stage2-ranker", action="store_true")
    p.add_argument("--operation-from-oracle", action="store_true")
    p.add_argument("--checkpoint-from-oracle", action="store_true")
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    out_root = Path(os.environ.get("SCOPE_VLLM_OUT_ROOT", str(_REPO / "outputs/scope_round11")))
    log = out_root / "logs" / f"vllm_factorized_{args.port}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = start_vllm_server(
        model_path=args.model_path,
        port=args.port,
        tensor_parallel_size=1,
        served_model_name=Path(args.model_path).name,
        log_path=str(log),
    )
    pid_dir = out_root / "pids"
    pid_dir.mkdir(parents=True, exist_ok=True)
    pid_file = pid_dir / f"vllm_port_{args.port}.pid"
    pid_file.write_text(str(handle.process.pid), encoding="utf-8")
    cmd = [
        sys.executable,
        str(_REPO / "training/scope_round11/replay_factorized_vllm.py"),
        "--model-path",
        args.model_path,
        "--input",
        str(args.input),
        "--output",
        str(args.output),
        "--port",
        str(args.port),
    ]
    if args.use_stage2_ranker:
        cmd.append("--use-stage2-ranker")
    if args.operation_from_oracle:
        cmd.append("--operation-from-oracle")
    if args.checkpoint_from_oracle:
        cmd.append("--checkpoint-from-oracle")
    try:
        subprocess.run(cmd, check=True, cwd=_REPO)
    finally:
        handle.stop()
        pid_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
