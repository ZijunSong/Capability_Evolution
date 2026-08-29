from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any
import subprocess
import sys

from easyopd.registry import register_method

from .component_registry import audit_component, get_component_spec, list_component_specs
from .core import SCAPEComponentOPD
from .event_collection import collect_event_states, state_uid
from .harness1_bridge import GptOssHarmonyAdapter, Harness1Bridge, Qwen3NativeChatAdapter
from .scape_agent_loop import SCAPEAgentLoop
from .skip_to_anchor import ALIGN, SKIP, project_bridge_steps, project_events

__all__ = [
    "SCAPEAgentLoop",
    "SCAPEComponentOPD",
    "SCAPEComponentOPDMethod",
    "GptOssHarmonyAdapter",
    "Qwen3NativeChatAdapter",
    "Harness1Bridge",
    "ALIGN",
    "SKIP",
    "audit_component",
    "build_hooks",
    "get_component_spec",
    "list_component_specs",
    "project_bridge_steps",
    "project_events",
]


def _needs_hydra_plus(prefix: str) -> bool:
    custom_roots = ("method", "component", "easyopd", "student", "teacher", "distillation", "reference", "controls", "evaluation", "reward", "ray_kwargs", "seed")
    return prefix == "seed" or prefix.startswith(custom_roots) or ".override_config." in prefix or prefix.endswith(".override_config")


def _flatten_overrides(prefix: str, value: Any) -> list[str]:
    if isinstance(value, Mapping):
        items: list[str] = []
        for key, child in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            items.extend(_flatten_overrides(next_prefix, child))
        return items
    if isinstance(value, bool):
        rendered = "True" if value else "False"
    elif value is None:
        rendered = "null"
    elif isinstance(value, str):
        rendered = value
    else:
        rendered = repr(value)
    key = f"+{prefix}" if _needs_hydra_plus(prefix) else prefix
    return [f"{key}={rendered}"]


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


@register_method("scape_component_opd")
class SCAPEComponentOPDMethod:
    name = "scape_component_opd"
    description = "SCAPE Harness-component OPD with method-local component supervision contracts"
    paper_url = None
    code_url = None
    verl_modified_files: list[str] = []

    @classmethod
    def build_hooks(cls, config):
        from .hooks import build_hooks

        return build_hooks(config)

    @classmethod
    def train(
        cls,
        config: dict[str, Any],
        *,
        dry_run: bool = False,
        output_dir: str | None = None,
        extra_args: dict[str, Any] | None = None,
    ) -> Any:
        runtime_cfg = config.get("runtime", {}) if isinstance(config, dict) else {}
        launch = runtime_cfg.get("launch", {}) if isinstance(runtime_cfg, dict) else {}
        if not isinstance(launch, dict):
            launch = {}

        merged_config = _deep_merge(config, extra_args or {}) if isinstance(config, dict) else dict(extra_args or {})
        runtime_cfg = merged_config.get("runtime", {}) if isinstance(merged_config, dict) else {}
        launch = runtime_cfg.get("launch", {}) if isinstance(runtime_cfg, dict) else {}
        if not isinstance(launch, dict):
            launch = {}
        command = launch.get("command")
        python_bin = str(runtime_cfg.get("python_bin") or sys.executable) if isinstance(runtime_cfg, dict) else sys.executable
        if command:
            cmd = [str(x) for x in command] if isinstance(command, (list, tuple)) else [str(command)]
        else:
            cmd = [python_bin, "-m", "verl.trainer.main_ppo"]

        overrides: list[str] = []
        for key in (
            "method",
            "easyopd",
            "component",
            "student",
            "teacher",
            "distillation",
            "reference",
            "controls",
            "evaluation",
            "data",
            "actor_rollout_ref",
            "algorithm",
            "critic",
            "trainer",
            "reward",
            "ray_kwargs",
            "seed",
        ):
            section = merged_config.get(key) if isinstance(merged_config, dict) else None
            if isinstance(section, Mapping):
                overrides.extend(_flatten_overrides(key, section))
            elif section is not None and key == "seed":
                overrides.extend(_flatten_overrides(key, section))
        if output_dir:
            overrides.append(f"trainer.default_local_dir={output_dir}")
        cmd.extend(overrides)

        manifest = {
            "method": cls.name,
            "command": cmd,
            "dry_run": dry_run,
            "output_dir": output_dir,
            "paper_grade": False,
        }
        if dry_run:
            return manifest

        return subprocess.run(cmd, check=False)
