"""Unified experiment launcher with dry-run / resume / seed."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import yaml

from experiments.common.config_diff import config_diff
from experiments.common.manifest import (
    build_run_manifest,
    finalize_run_manifest,
    write_run_manifest,
)
from experiments.common.registry import ExperimentRegistry
from experiments.common.spec import ExperimentSpec
from experiments.common.validation import maybe_write_done, validate_run_dir

_REPO = Path(__file__).resolve().parents[2]


RunnerFn = Callable[[ExperimentSpec, Path], dict[str, Any]]


def prepare_output_dir(spec: ExperimentSpec, repo: Path = _REPO) -> Path:
    out = spec.resolved_output_dir(repo)
    if out.exists() and (out / "DONE").exists() and not spec.resume:
        raise FileExistsError(
            f"output already DONE: {out} (pass resume=True to continue, never overwrite)"
        )
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)
    return out


def write_resolved_artifacts(
    spec: ExperimentSpec,
    out: Path,
    *,
    parent: ExperimentSpec | None = None,
) -> None:
    spec.to_yaml(out / "resolved_config.yaml")
    diff = config_diff(parent, spec) if parent is not None else {"n_changed": 0, "changed": {}}
    (out / "config_diff.json").write_text(
        json.dumps(diff, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def launch(
    spec: ExperimentSpec,
    runner: RunnerFn,
    *,
    command: list[str] | None = None,
    parent: ExperimentSpec | None = None,
    repo: Path = _REPO,
) -> dict[str, Any]:
    out = prepare_output_dir(spec, repo)
    write_resolved_artifacts(spec, out, parent=parent)
    cmd = command or [sys.executable, "-m", "experiments.common.launcher", "--experiment-id", spec.experiment_id]
    manifest = build_run_manifest(spec, command=cmd, repo_root=repo)
    write_run_manifest(out / "run_manifest.json", manifest)

    if spec.dry_run:
        summary = {
            "schema_version": "iclr_summary_v1",
            "experiment_id": spec.experiment_id,
            "status": "dry_run",
            "metrics": {},
            "n_queries": 0,
            "errors": [],
            "notes": "dry-run only; no training/eval executed",
        }
        (out / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (out / "predictions.jsonl").write_text("", encoding="utf-8")
        (out / "telemetry.jsonl").write_text("", encoding="utf-8")
        manifest = finalize_run_manifest(manifest, exit_code=0, output_dir=out)
        write_run_manifest(out / "run_manifest.json", manifest)
        maybe_write_done(out)
        return summary

    exit_code = 0
    error_summary = None
    summary: dict[str, Any]
    try:
        summary = runner(spec, out)
        if "schema_version" not in summary:
            summary = {
                "schema_version": "iclr_summary_v1",
                "experiment_id": spec.experiment_id,
                "status": "completed",
                **summary,
            }
        (out / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    except Exception as exc:  # noqa: BLE001
        exit_code = 1
        error_summary = f"{type(exc).__name__}: {exc}"
        summary = {
            "schema_version": "iclr_summary_v1",
            "experiment_id": spec.experiment_id,
            "status": "failed",
            "metrics": {},
            "n_queries": 0,
            "errors": [error_summary],
        }
        (out / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        raise
    finally:
        manifest = finalize_run_manifest(
            manifest, exit_code=exit_code, output_dir=out, error_summary=error_summary
        )
        write_run_manifest(out / "run_manifest.json", manifest)
        if exit_code == 0:
            errs = validate_run_dir(out)
            if not errs:
                maybe_write_done(out)
    return summary


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SCOPE ICLR experiment launcher")
    p.add_argument("--experiment-id", required=True)
    p.add_argument("--registry", default=str(_REPO / "experiments" / "registry.yaml"))
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--gpu", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke-query-limit", type=int, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    reg = ExperimentRegistry(args.registry)
    overrides: dict[str, Any] = {
        "dry_run": args.dry_run,
        "resume": args.resume,
    }
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.gpu is not None:
        overrides["gpu"] = args.gpu
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if args.smoke_query_limit is not None:
        overrides["smoke_query_limit"] = args.smoke_query_limit
    spec = reg.resolve(args.experiment_id, **overrides)

    # Lazy import runners to avoid circular imports
    from experiments.ablations.runners.dispatch import dispatch_ablation
    from experiments.baselines.runners.dispatch import dispatch_baseline

    kind = reg.get_raw(args.experiment_id).get("type", "ablation")
    runner = dispatch_baseline if kind == "baseline" else dispatch_ablation
    launch(spec, runner, command=sys.argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
