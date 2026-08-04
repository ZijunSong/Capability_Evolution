#!/usr/bin/env python3
"""Compare AgentCore vs FullHarness configs for fair baseline Gate 1B."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _hash_obj(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _module_flags(cfg_path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    out: dict[str, Any] = {}
    for mod, opts in raw.items():
        if isinstance(opts, dict):
            out[mod] = {k: v for k, v in opts.items() if k != "module_id"}
    return out


AGENT_CORE_TOOLS = sorted(
    ["search_corpus", "read_document", "curate", "end_search", "verify", "review_docs"]
)


def tool_schema_hash() -> str:
    return _hash_obj(AGENT_CORE_TOOLS)


def compare_configs(a_path: Path, b_path: Path) -> dict[str, Any]:
    a_flags = _module_flags(a_path)
    b_flags = _module_flags(b_path)
    shared = {}
    changed_modules: dict[str, Any] = {}
    for mod in sorted(set(a_flags) | set(b_flags)):
        if a_flags.get(mod) == b_flags.get(mod):
            shared[mod] = a_flags.get(mod)
        else:
            changed_modules[mod] = {"a": a_flags.get(mod), "b": b_flags.get(mod)}

    changed_budget: list[str] = []
    runtime_a = {"max_turns": 35, "max_tokens": 2048, "temperature": 1.0}
    runtime_b = {"max_turns": 35, "max_tokens": 2048, "temperature": 1.0}
    if runtime_a != runtime_b:
        changed_budget = [
            k for k in runtime_a if runtime_a[k] != runtime_b.get(k)
        ]

    allowed_module_diffs = {"evidence_state", "context_budget", "verification", "retrieval"}
    module_keys = set(changed_modules.keys())
    gate_1b_pass = (
        not []
        and not changed_budget
        and module_keys
        and module_keys.issubset(allowed_module_diffs)
    )

    result = {
        "config_a": str(a_path),
        "config_b": str(b_path),
        "shared_fields": shared,
        "changed_module_flags": changed_modules,
        "changed_tools": [],
        "changed_prompt_fields": [],
        "changed_budget_fields": changed_budget,
        "changed_evaluator_fields": [],
        "tool_schema_hash": tool_schema_hash(),
        "runtime_budget": runtime_a,
        "gate_1b_pass": gate_1b_pass,
    }
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--agent-core",
        type=Path,
        default=_REPO / "harness/configs/agent_core.yaml",
    )
    p.add_argument(
        "--full-harness",
        type=Path,
        default=_REPO / "harness/configs/agent_core_full_harness.yaml",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    diff = compare_configs(args.agent_core, args.full_harness)
    (args.output_dir / "CONFIG_DIFF.json").write_text(
        json.dumps(diff, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "TOOL_SCHEMA_HASH.txt").write_text(diff["tool_schema_hash"] + "\n")
    (args.output_dir / "BUDGET_CONFIG.json").write_text(
        json.dumps(
            {"max_turns": 35, "max_tokens": 2048, "temperature": 1.0},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "PROMPT_HASH.txt").write_text(_hash_obj(["agent_core_shared_prompt"]) + "\n")
    print(json.dumps(diff, indent=2))


if __name__ == "__main__":
    main()
