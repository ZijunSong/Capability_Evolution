#!/usr/bin/env python3
"""One-click Harness-1 SFT via the upstream Tinker recipe.

Materializes the public ``pat-jj/harness-1-train-data`` stage=sft trajectories
(899 GPT-5.4 v8d traces) into the JSON layout ``training/train_sft.py`` expects,
sets the official v8d flags, and runs Harness-1 SFT on Tinker. Defaults match
``external/harness-1/training/launch_sft_training.sh`` (GPT-OSS-20B, 3 epochs,
LoRA r=32, lr=5e-6, batch=128, max_length=32768, min_recall=0.1).

Example:
  PYTHONPATH=TRIM python TRIM/scripts/run_sft.py
  PYTHONPATH=TRIM python TRIM/scripts/run_sft.py --model-name openai/gpt-oss-20b
  PYTHONPATH=TRIM python TRIM/scripts/run_sft.py --pack-only --pack TRIM/data/harness-1-sft-data.tar.gz
  PYTHONPATH=TRIM python TRIM/scripts/run_sft.py --smoke --dry-run
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

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
    launch = {
        "framework": "tinker + tinker_cookbook.supervised",
        "entrypoint": str(HARNESS1_ROOT / "training" / "train_sft.py"),
        "cwd": str(HARNESS1_ROOT),
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
        print(json.dumps({"ok": True, "dry_run": True, "n_trajectory_json": n_json}, indent=2), flush=True)
        return 0

    env = sft_subprocess_env(require_tinker_key=True)
    result = run_harness1_train_sft(argv_train, env=env, check=False)
    summary = {
        "ok": result.returncode == 0,
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
