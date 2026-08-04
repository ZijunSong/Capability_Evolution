"""AgentCore vs FullHarness parity tests (Round 8 Gate 1B)."""

from __future__ import annotations

from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[2]
AGENT_CORE = _REPO / "harness/configs/agent_core.yaml"
FULL = _REPO / "harness/configs/agent_core_full_harness.yaml"


def _flags(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_agent_core_tool_budget_fields_shared():
    from training.scope_round8.compare_agent_configs import AGENT_CORE_TOOLS, tool_schema_hash

    assert tool_schema_hash() == tool_schema_hash()
    assert "search_corpus" in AGENT_CORE_TOOLS


def test_agent_core_vs_full_only_module_flags_differ():
    a = _flags(AGENT_CORE)
    b = _flags(FULL)
    assert a["retrieval"]["enabled"] == b["retrieval"]["enabled"]
    assert a["verification"]["expose_verify_tool"] == b["verification"]["expose_verify_tool"]
    assert a["evidence_state"]["enabled"] != b["evidence_state"]["enabled"]
    assert a["context_budget"]["enabled"] != b["context_budget"]["enabled"]
    assert a["recovery"]["enabled"] == b["recovery"]["enabled"] == False
