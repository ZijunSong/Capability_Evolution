"""SEED external baseline adapter (dry-run first)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.baselines.adapters.base import BaselineAdapter
from experiments.baselines.adapters.browsecomp_adapter import load_manifest, to_baseline_queries, write_baseline_queries
from experiments.common.provenance import MissingAssetError
from experiments.common.spec import ExperimentSpec

_REPO = Path(__file__).resolve().parents[3]
SEED_ROOT = _REPO / "external" / "baselines" / "SEED"


class SeedAdapter(BaselineAdapter):
    name = "SEED"

    def prepare_data(self, spec: ExperimentSpec) -> dict[str, Any]:
        manifest = _REPO / (
            spec.test_manifest or "artifacts/datasets/round2_audit_100q/query_manifest.json"
        )
        if not manifest.exists():
            raise MissingAssetError(f"SEED manifest missing: {manifest}")
        rows = to_baseline_queries(load_manifest(manifest), limit=spec.smoke_query_limit or 2)
        out = Path(spec.output_dir)
        if not out.is_absolute():
            out = _REPO / out
        qpath = out / "seed_queries.json"
        write_baseline_queries(qpath, rows)
        return {"queries_path": str(qpath), "n_queries": len(rows)}

    def prepare_environment(self, spec: ExperimentSpec) -> dict[str, Any]:
        exists = SEED_ROOT.joinpath(".git").exists()
        head = None
        if exists:
            try:
                head = (SEED_ROOT / ".git" / "HEAD").read_text(encoding="utf-8").strip()
            except Exception:  # noqa: BLE001
                head = None
        return {
            "repo": str(SEED_ROOT),
            "exists": exists,
            "head_ref": head,
            "env_file": str(_REPO / "experiments/baselines/envs/seed.environment.yml"),
            "note": "Do not install into SCOPE bishop env; use isolated env",
        }

    def build_command(self, spec: ExperimentSpec) -> list[str]:
        # Placeholder command — actual entry depends on upstream README; dry-run only.
        data = self.prepare_data(spec)
        return [
            "bash",
            "-lc",
            (
                f"cd {SEED_ROOT} && "
                f"echo '[SEED dry-run] would train/eval on {data['queries_path']} "
                f"with model={spec.base_model} seed={spec.seed}'"
            ),
        ]

    def collect_outputs(self, spec: ExperimentSpec) -> dict[str, Any]:
        out = Path(spec.output_dir)
        if not out.is_absolute():
            out = _REPO / out
        return {"output_dir": str(out), "normalized_summary": str(out / "summary.json")}

    def normalize_metrics(self, spec: ExperimentSpec) -> dict[str, Any]:
        return {
            "adapter": self.name,
            "status": "not_run",
            "note": "metrics normalization pending real SEED smoke",
        }
