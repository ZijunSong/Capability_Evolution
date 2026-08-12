"""SDAR external baseline adapter (P1; dry-run)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.baselines.adapters.base import BaselineAdapter
from experiments.baselines.adapters.browsecomp_adapter import load_manifest, to_baseline_queries, write_baseline_queries
from experiments.common.provenance import MissingAssetError
from experiments.common.spec import ExperimentSpec

_REPO = Path(__file__).resolve().parents[3]
SDAR_ROOT = _REPO / "external" / "baselines" / "SDAR"


class SdarAdapter(BaselineAdapter):
    name = "SDAR"

    def prepare_data(self, spec: ExperimentSpec) -> dict[str, Any]:
        manifest = _REPO / (
            spec.test_manifest or "artifacts/datasets/round2_audit_100q/query_manifest.json"
        )
        if not manifest.exists():
            raise MissingAssetError(f"SDAR manifest missing: {manifest}")
        rows = to_baseline_queries(load_manifest(manifest), limit=spec.smoke_query_limit or 2)
        out = Path(spec.output_dir)
        if not out.is_absolute():
            out = _REPO / out
        qpath = out / "sdar_queries.json"
        write_baseline_queries(qpath, rows)
        return {"queries_path": str(qpath), "n_queries": len(rows)}

    def prepare_environment(self, spec: ExperimentSpec) -> dict[str, Any]:
        return {
            "repo": str(SDAR_ROOT),
            "exists": SDAR_ROOT.joinpath(".git").exists(),
            "env_file": str(_REPO / "experiments/baselines/envs/sdar.environment.yml"),
        }

    def build_command(self, spec: ExperimentSpec) -> list[str]:
        data = self.prepare_data(spec)
        return [
            "bash",
            "-lc",
            f"cd {SDAR_ROOT} && echo '[SDAR dry-run] {data['queries_path']} seed={spec.seed}'",
        ]

    def collect_outputs(self, spec: ExperimentSpec) -> dict[str, Any]:
        return {"output_dir": spec.output_dir}

    def normalize_metrics(self, spec: ExperimentSpec) -> dict[str, Any]:
        return {"adapter": self.name, "status": "not_run"}
