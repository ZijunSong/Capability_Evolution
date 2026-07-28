"""Tests for module fallback behavior."""

from __future__ import annotations

from harness.graph.execution_context import ExecutionContext
from harness.graph.registry import ModuleRegistry
from harness.harness_config import load_harness_config, config_path


class _StubWorkingMemory:
    def __init__(self, query: str = "test"):
        self.query = query
        self.curated_ids: list[str] = []
        self.doc_store: dict = {}

    def get_minimal_state(self) -> str:
        lines = [f'Query: "{self.query}"', f"Curated Set ({len(self.curated_ids)}):"]
        for doc_id in self.curated_ids:
            snippet = self.doc_store.get(doc_id, {}).get("snippet", "")
            lines.append(f"  [*] {doc_id}: {snippet}")
        return "\n".join(lines)

    def get_structured_state(self, **kwargs) -> str:
        return self.get_minimal_state() + "\nDocument Pool: ..."


def test_evidence_state_minimal_fallback():
    cfg = load_harness_config(config_path("modules_minimal.yaml"))
    registry = ModuleRegistry.from_config(cfg)
    wm = _StubWorkingMemory("test query")
    wm.curated_ids = ["doc_a"]
    wm.doc_store["doc_a"] = {"snippet": "hello", "full_text": "hello world"}
    ctx = ExecutionContext("ep1", "q1", working_memory=wm)

    node = registry.get("evidence_state").node("E9")
    assert node is not None
    result = node.execute("payload", ctx)
    assert "doc_a" in str(result.output)
    assert "Document Pool" not in str(result.output)


def test_verification_fallback_empty_render():
    cfg = load_harness_config(config_path("ablate_verification.yaml"))
    registry = ModuleRegistry.from_config(cfg)
    ctx = ExecutionContext("ep1", "q1", working_memory=_StubWorkingMemory("q"))
    node = registry.get("verification").node("V3")
    assert node is not None
    result = node.execute(None, ctx)
    assert result.output == "" or result.metadata.get("fallback_used")


def test_context_truncation_fallback():
    cfg = load_harness_config(config_path("modules_minimal.yaml"))
    registry = ModuleRegistry.from_config(cfg)
    ctx = ExecutionContext("ep1", "q1")
    ctx.artifacts["context_char_limit"] = 50
    node = registry.get("context_budget").node("C5")
    assert node is not None
    long_text = "x" * 200
    result = node.execute(long_text, ctx)
    assert len(str(result.output)) <= 50 + len("\n...(truncated)")
