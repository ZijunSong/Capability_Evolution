"""Load and validate experiment registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from experiments.common.spec import ExperimentSpec

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = _REPO / "experiments" / "registry.yaml"


class ExperimentRegistry:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else DEFAULT_REGISTRY
        if not self.path.exists():
            raise FileNotFoundError(f"registry not found: {self.path}")
        with self.path.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        self.meta = raw.get("meta", {})
        self.defaults = raw.get("defaults", {})
        self.experiments: dict[str, dict[str, Any]] = {}
        for item in raw.get("experiments", []) or []:
            if "experiment_id" not in item:
                raise ValueError(f"registry entry missing experiment_id: {item}")
            eid = item["experiment_id"]
            if eid in self.experiments:
                raise ValueError(f"duplicate experiment_id: {eid}")
            self.experiments[eid] = item

    def ids(self) -> list[str]:
        return sorted(self.experiments)

    def get_raw(self, experiment_id: str) -> dict[str, Any]:
        if experiment_id not in self.experiments:
            raise KeyError(f"unknown experiment_id: {experiment_id}")
        return dict(self.experiments[experiment_id])

    def resolve(self, experiment_id: str, **overrides: Any) -> ExperimentSpec:
        raw = self.get_raw(experiment_id)
        merged = dict(self.defaults)
        merged.update(raw)
        merged.update({k: v for k, v in overrides.items() if v is not None})
        if "output_dir" not in merged or not merged["output_dir"]:
            group = merged.get("group", "ungrouped")
            variant = merged.get("variant", experiment_id)
            seed = merged.get("seed", 42)
            kind = merged.get("type", "ablation")
            root = "outputs/iclr_baselines" if kind == "baseline" else "outputs/iclr_ablations"
            merged["output_dir"] = f"{root}/{group}/{variant}/seed_{seed}"
        return ExperimentSpec.from_dict(merged)

    def by_group(self, group: str) -> list[str]:
        return [eid for eid, e in self.experiments.items() if e.get("group") == group]

    def validate(self) -> list[str]:
        """Return list of validation errors (empty if ok)."""
        errors: list[str] = []
        for eid, raw in self.experiments.items():
            try:
                self.resolve(eid)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{eid}: {exc}")
            cf = raw.get("changed_factor")
            if not cf:
                errors.append(f"{eid}: missing changed_factor")
            status = raw.get("status", "NOT_IMPLEMENTED")
            allowed = {
                "NOT_IMPLEMENTED",
                "IMPLEMENTED",
                "UNIT_TESTED",
                "SMOKE_PASSED",
                "READY_FOR_FULL_RUN",
                "RUNNING",
                "COMPLETED",
                "BLOCKED",
                "INVALID",
            }
            if status not in allowed:
                errors.append(f"{eid}: invalid status {status}")
        return errors
