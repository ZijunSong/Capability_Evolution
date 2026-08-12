"""OPID external baseline adapter (dry-run first)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.baselines.adapters.base import BaselineAdapter
from experiments.baselines.adapters.browsecomp_adapter import load_manifest, to_baseline_queries, write_baseline_queries
from experiments.common.provenance import MissingAssetError
from experiments.common.spec import ExperimentSpec

_REPO = Path(__file__).resolve().parents[3]
OPID_ROOT = _REPO / "external" / "baselines" / "OPID"


class OpidAdapter(BaselineAdapter):
    name = "OPID"

    def prepare_data(self, spec: ExperimentSpec) -> dict[str, Any]:
        manifest = _REPO / (
            spec.test_manifest or "artifacts/datasets/round2_audit_100q/query_manifest.json"
        )
        if not manifest.exists():
            raise MissingAssetError(f"OPID manifest missing: {manifest}")
        rows = to_baseline_queries(load_manifest(manifest), limit=spec.smoke_query_limit or 2)
        out = Path(spec.output_dir)
        if not out.is_absolute():
            out = _REPO / out
        qpath = out / "opid_queries.json"
        write_baseline_queries(qpath, rows)
        return {"queries_path": str(qpath), "n_queries": len(rows)}

    def prepare_environment(self, spec: ExperimentSpec) -> dict[str, Any]:
        return {
            "repo": str(OPID_ROOT),
            "exists": OPID_ROOT.joinpath(".git").exists(),
            "env_file": str(_REPO / "experiments/baselines/envs/opid.environment.yml"),
            "note": "Isolated env required; do not mutate SCOPE torch/vLLM",
        }

    def build_command(self, spec: ExperimentSpec) -> list[str]:
        data = self.prepare_data(spec)
        return [
            "bash",
            "-lc",
            (
                f"cd {OPID_ROOT} && "
                f"echo '[OPID dry-run] would run on {data['queries_path']} "
                f"model={spec.base_model} seed={spec.seed}'"
            ),
        ]

    def collect_outputs(self, spec: ExperimentSpec) -> dict[str, Any]:
        out = Path(spec.output_dir)
        if not out.is_absolute():
            out = _REPO / out
        return {"output_dir": str(out)}

    def normalize_metrics(self, spec: ExperimentSpec) -> dict[str, Any]:
        return {"adapter": self.name, "status": "not_run"}
