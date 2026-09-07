#!/usr/bin/env python3
"""One-click Harness-1 SFT (Tinker cloud or local HuggingFace LoRA).

Materializes the public ``pat-jj/harness-1-train-data`` stage=sft trajectories
(899 GPT-5.4 v8d traces) into the JSON layout ``training/train_sft.py`` expects,
sets the official v8d flags, and runs SFT. A local checkpoint directory
(``--model-name /path/to/gpt-oss-20b``) uses PEFT LoRA in-process and does
**not** need ``TINKER_API_KEY``. Tinker ids such as ``openai/gpt-oss-20b``
still use the hosted recipe. Defaults match
``external/harness-1/training/launch_sft_training.sh``.

Example:
  PYTHONPATH=TRIM python TRIM/scripts/run_sft.py
  PYTHONPATH=TRIM python TRIM/scripts/run_sft.py --model-name openai/gpt-oss-20b
  PYTHONPATH=TRIM python TRIM/scripts/run_sft.py --model-name /mnt/songzijun/models/openai/gpt-oss-20b
  PYTHONPATH=TRIM python TRIM/scripts/run_sft.py --pack-only --pack TRIM/data/harness-1-sft-data.tar.gz
  PYTHONPATH=TRIM python TRIM/scripts/run_sft.py --smoke --dry-run
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_TRIM = Path(__file__).resolve().parents[1]
if str(_TRIM) not in sys.path:
    sys.path.insert(0, str(_TRIM))

from trim.cli.launch import LaunchError, parse_sft_args
from trim.training.sft_data import (
    DEFAULT_SFT_EXTRACTED,
    REPO_SFT_PACK,
    assert_train_sft_ready,
    materialize_sft_data_dir,
)
from trim.training.sft_runtime import (
    HARNESS1_ROOT,
    HARNESS1_SFT_V8D_ENV,
    resolve_sft_backend,
    run_harness1_train_sft,
    sft_subprocess_env,
    train_sft_argv,
)


def _dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_sft_args(argv)
    except LaunchError as exc:
        raise SystemExit(str(exc)) from exc

    args.out.mkdir(parents=True, exist_ok=True)
    backend = resolve_sft_backend(getattr(args, "backend", "auto"), args.model_name)
    keepalive = None
    if (
        backend == "hf"
        and not args.pack_only
        and not args.dry_run
        and not args.validate_only
    ):
        from trim.training.gpu_keepalive import acquire_keepalive

        keepalive = acquire_keepalive(dim=4096)
    try:
        return _main_body(args, backend)
    finally:
        if keepalive is not None:
            from trim.training.gpu_keepalive import release_keepalive

            release_keepalive()


def _main_body(args: Any, backend: str) -> int:
    default_tar = REPO_SFT_PACK
    write_pack = args.pack
    if write_pack is None and not args.n_trajectories and (args.pack_only or not default_tar.is_file()):
        write_pack = default_tar
    dest = args.out / "sft_data" if args.n_trajectories else DEFAULT_SFT_EXTRACTED
    data_dir, data_meta = materialize_sft_data_dir(
        args.sft_data,
        dest=dest,
        n_trajectories=args.n_trajectories,
        write_pack=write_pack,
        allow_hf=True,
    )
    n_json = assert_train_sft_ready(data_dir)
    argv_train = train_sft_argv(
        data_dir=data_dir,
        log_path=args.out,
        model_name=args.model_name,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        max_length=args.max_length,
        min_recall=args.min_recall,
        save_every=args.save_every,
        eval_every=args.eval_every,
        load_checkpoint_path=args.load_checkpoint_path,
        python=args.python,
    )
    if backend == "hf":
        launch = {
            "framework": "huggingface + peft LoRA packed DDP",
            "entrypoint": "trim.training.hf_sft.run_hf_sft",
            "cwd": None,
            "backend": backend,
            "model_name": args.model_name,
            "device_map": "ddp",
            "pack_length": int(getattr(args, "pack_length", 8192)),
            "micro_batch_size": int(getattr(args, "micro_batch_size", 1)),
            "gradient_checkpointing": not bool(getattr(args, "no_gradient_checkpointing", False)),
            "merge": bool(getattr(args, "merge", False)),
            "sft_data": str(args.sft_data),
            "data_dir": str(data_dir),
            "data_meta": data_meta,
            "n_trajectory_json": n_json,
            "out": str(args.out),
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "lora_rank": args.lora_rank,
            "max_length": args.max_length,
            "min_recall": args.min_recall,
            "save_every": args.save_every,
            "eval_every": args.eval_every,
            "load_checkpoint_path": args.load_checkpoint_path,
            "python": sys.executable,
            "v8d_env": dict(HARNESS1_SFT_V8D_ENV),
            "smoke": bool(args.smoke),
            "argv": None,
            "pack_only": bool(args.pack_only),
            "dry_run": bool(args.dry_run or args.validate_only),
            "requires_tinker_api_key": False,
        }
    else:
        launch = {
            "framework": "tinker + tinker_cookbook.supervised",
            "entrypoint": str(HARNESS1_ROOT / "training" / "train_sft.py"),
            "cwd": str(HARNESS1_ROOT),
            "backend": backend,
            "model_name": args.model_name,
            "sft_data": str(args.sft_data),
            "data_dir": str(data_dir),
            "data_meta": data_meta,
            "n_trajectory_json": n_json,
            "out": str(args.out),
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "lora_rank": args.lora_rank,
            "max_length": args.max_length,
            "min_recall": args.min_recall,
            "save_every": args.save_every,
            "eval_every": args.eval_every,
            "load_checkpoint_path": args.load_checkpoint_path,
            "python": argv_train[0] if argv_train else None,
            "v8d_env": dict(HARNESS1_SFT_V8D_ENV),
            "smoke": bool(args.smoke),
            "argv": argv_train,
            "pack_only": bool(args.pack_only),
            "dry_run": bool(args.dry_run or args.validate_only),
            "requires_tinker_api_key": True,
        }
    _dump(args.out / "LAUNCH.json", launch)
    print(json.dumps(launch, indent=2, default=str), flush=True)

    if args.pack_only:
        print(
            json.dumps(
                {
                    "ok": True,
                    "pack": data_meta.get("pack") or str(write_pack),
                    "data_dir": str(data_dir),
                    "n": n_json,
                },
                indent=2,
            ),
            flush=True,
        )
        return 0

    if args.dry_run or args.validate_only:
        print(
            json.dumps(
                {
                    "ok": True,
                    "dry_run": True,
                    "backend": backend,
                    "n_trajectory_json": n_json,
                    "requires_tinker_api_key": backend == "tinker",
                },
                indent=2,
            ),
            flush=True,
        )
        return 0

    if backend == "hf":
        from trim.training.hf_sft import run_hf_sft

        summary = run_hf_sft(
            model_name=args.model_name,
            data_dir=data_dir,
            out=args.out,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            lora_rank=args.lora_rank,
            max_length=args.max_length,
            min_recall=args.min_recall,
            save_every=args.save_every,
            eval_every=args.eval_every,
            load_checkpoint_path=args.load_checkpoint_path,
            device_map="ddp",
            merge=bool(getattr(args, "merge", False)),
            pack_length=int(getattr(args, "pack_length", 8192)),
            micro_batch_size=int(getattr(args, "micro_batch_size", 1)),
            gradient_checkpointing=not bool(getattr(args, "no_gradient_checkpointing", False)),
        )
        print(json.dumps(summary, indent=2, default=str), flush=True)
        return 0 if summary.get("ok") else 1

    env = sft_subprocess_env(require_tinker_key=True)
    result = run_harness1_train_sft(argv_train, env=env, check=False)
    summary = {
        "ok": result.returncode == 0,
        "backend": "tinker",
        "returncode": result.returncode,
        "out": str(args.out),
        "data_dir": str(data_dir),
        "model_name": args.model_name,
        "checkpoints": str(args.out / "checkpoints.jsonl"),
    }
    _dump(args.out / "SFT_SUMMARY.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
