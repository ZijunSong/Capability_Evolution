"""SCOPE-internal baseline environment adapter (B0–B6)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.baselines.adapters.base import BaselineAdapter
from experiments.common.provenance import require_path
from experiments.common.spec import ExperimentSpec

_REPO = Path(__file__).resolve().parents[3]


class ScopeEnvAdapter(BaselineAdapter):
    name = "scope_env"

    def prepare_data(self, spec: ExperimentSpec) -> dict[str, Any]:
        manifest = spec.test_manifest or "artifacts/datasets/round2_audit_100q/query_manifest.json"
        path = require_path(_REPO / manifest, label="test_manifest")
        return {"manifest": str(path), "smoke_query_limit": spec.smoke_query_limit or 2}

    def prepare_environment(self, spec: ExperimentSpec) -> dict[str, Any]:
        runtime = require_path(_REPO / spec.runtime_config, label="runtime_config")
        return {
            "runtime_config": str(runtime),
            "retriever": spec.retriever,
            "max_turns": spec.max_turns,
            "max_tokens": spec.max_tokens,
            "temperature": spec.temperature,
            "base_model": spec.base_model,
            "checkpoint": spec.checkpoint,
        }

    def build_command(self, spec: ExperimentSpec) -> list[str]:
        out = spec.output_dir
        return [
            "python",
            "-m",
            "experiments.common.launcher",
            "--experiment-id",
            spec.experiment_id,
            "--seed",
            str(spec.seed),
            "--output-dir",
            out,
            "--smoke-query-limit",
            str(spec.smoke_query_limit or 2),
        ]

    def collect_outputs(self, spec: ExperimentSpec) -> dict[str, Any]:
        out = Path(spec.output_dir)
        if not out.is_absolute():
            out = _REPO / out
        return {
            "summary": str(out / "summary.json"),
            "predictions": str(out / "predictions.jsonl"),
            "done": (out / "DONE").exists(),
        }

    def normalize_metrics(self, spec: ExperimentSpec) -> dict[str, Any]:
        import json

        outs = self.collect_outputs(spec)
        p = Path(outs["summary"])
        if not p.exists():
            raise FileNotFoundError(f"summary missing: {p}")
        return json.loads(p.read_text(encoding="utf-8")).get("metrics", {})
