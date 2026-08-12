"""BaselineAdapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from experiments.common.spec import ExperimentSpec


class BaselineAdapter(ABC):
    name: str = "base"

    @abstractmethod
    def prepare_data(self, spec: ExperimentSpec) -> dict[str, Any]:
        ...

    @abstractmethod
    def prepare_environment(self, spec: ExperimentSpec) -> dict[str, Any]:
        ...

    @abstractmethod
    def build_command(self, spec: ExperimentSpec) -> list[str]:
        ...

    def validate_command(self, spec: ExperimentSpec) -> list[str]:
        cmd = self.build_command(spec)
        errors = []
        if not cmd:
            errors.append("empty command")
        return errors

    def launch(self, spec: ExperimentSpec) -> dict[str, Any]:
        raise NotImplementedError("launch requires external env; use dry-run / smoke scripts")

    @abstractmethod
    def collect_outputs(self, spec: ExperimentSpec) -> dict[str, Any]:
        ...

    @abstractmethod
    def normalize_metrics(self, spec: ExperimentSpec) -> dict[str, Any]:
        ...

    def dry_run(self, spec: ExperimentSpec) -> dict[str, Any]:
        data = self.prepare_data(spec)
        env = self.prepare_environment(spec)
        cmd = self.build_command(spec)
        errors = self.validate_command(spec)
        return {
            "adapter": self.name,
            "data": data,
            "environment": env,
            "command": cmd,
            "validation_errors": errors,
            "dry_run": True,
        }
