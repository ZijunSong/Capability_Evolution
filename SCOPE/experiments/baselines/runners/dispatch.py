"""Baseline experiment dispatch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.baselines.adapters import ADAPTERS
from experiments.baselines.adapters.scope_env_adapter import ScopeEnvAdapter
from experiments.common.spec import ExperimentSpec
from inference.scope.eval_common import dup_closed_loop_metrics


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run_internal_baseline(spec: ExperimentSpec, out: Path) -> dict[str, Any]:
    n = spec.smoke_query_limit or 2
    eps = []
    for i in range(n):
        eps.append(
            {
                "query_id": f"b_q{i}",
                "recall": 0.0,
                "trajectory_recall": 0.0,
                "final_answer_recall": 0.0,
                "reward": 0.0,
                "turns": 3,
                "tool_calls": 1,
                "dup_curate_rate": 0.0,
                "n_curated": 0,
                "errors": [],
                "valid_decision_count": 1,
                "decisions": [
                    {
                        "gold": "KEEP_EVIDENCE",
                        "pred": "KEEP_EVIDENCE",
                    }
                ],
                "runtime": spec.runtime_config,
                "method": spec.method,
            }
        )
    metrics = dup_closed_loop_metrics(eps)
    metrics["baseline_method"] = spec.method
    metrics["runtime_config"] = spec.runtime_config
    _write_jsonl(out / "predictions.jsonl", eps)
    _write_jsonl(out / "telemetry.jsonl", [{"event": "internal_baseline_smoke", "variant": spec.variant}])
    adapter = ScopeEnvAdapter()
    dry = adapter.dry_run(spec)
    (out / "adapter_dry_run.json").write_text(json.dumps(dry, indent=2) + "\n", encoding="utf-8")
    return {
        "metrics": metrics,
        "n_queries": n,
        "errors": [],
        "status": "smoke_pipeline",
        "notes": "internal baseline smoke — not a research conclusion",
    }


def run_external_baseline(spec: ExperimentSpec, out: Path) -> dict[str, Any]:
    name = (spec.method or "").upper()
    if name not in ADAPTERS:
        raise KeyError(f"no adapter for method={spec.method}")
    adapter = ADAPTERS[name]()
    # Ensure output dir exists for prepare_data
    out.mkdir(parents=True, exist_ok=True)
    # rewrite output_dir absolute for adapter
    spec.output_dir = str(out)
    dry = adapter.dry_run(spec)
    (out / "adapter_dry_run.json").write_text(json.dumps(dry, indent=2) + "\n", encoding="utf-8")
    _write_jsonl(out / "predictions.jsonl", [])
    _write_jsonl(out / "telemetry.jsonl", [{"event": "external_dry_run", "adapter": name}])
    return {
        "metrics": {"dry_run": True, "command": dry.get("command"), "env_exists": dry.get("environment", {}).get("exists")},
        "n_queries": dry.get("data", {}).get("n_queries", 0),
        "errors": dry.get("validation_errors", []),
        "status": "dry_run",
        "notes": f"{name} dry-run only; no formal training",
    }


def dispatch_baseline(spec: ExperimentSpec, out: Path) -> dict[str, Any]:
    method = (spec.method or "").lower()
    if method in {"seed", "opid", "sdar"}:
        result = run_external_baseline(spec, out)
    else:
        result = run_internal_baseline(spec, out)
    return {
        "schema_version": "iclr_summary_v1",
        "experiment_id": spec.experiment_id,
        "status": result.get("status", "completed"),
        "metrics": result.get("metrics", {}),
        "n_queries": int(result.get("n_queries", 0)),
        "errors": result.get("errors", []),
        "notes": result.get("notes", ""),
    }
