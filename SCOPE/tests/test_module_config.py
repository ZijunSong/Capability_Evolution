"""Unit tests for module configuration."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from harness.harness_config import (
    apply_harness_config,
    config_path,
    from_legacy_env,
    load_harness_config,
)
from harness.graph.registry import ModuleRegistry


def test_load_modules_full():
    cfg = load_harness_config(config_path("modules_full.yaml"))
    assert cfg.evidence_state.enabled
    assert cfg.verification.enabled
    assert cfg.retrieval.required


def test_ablate_verification_disables_verify():
    cfg = load_harness_config(config_path("ablate_verification.yaml"))
    assert not cfg.verification.enabled
    assert not cfg.verification.options.get("expose_verify_tool")


def test_config_hash_stable():
    cfg = load_harness_config(config_path("modules_full.yaml"))
    assert cfg.config_hash() == cfg.config_hash()


def test_legacy_env_mapping(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("V8D_VERIFY_TOOL", "0")
    monkeypatch.setenv("ABLATE_VERIFY_UNAVAILABLE", "1")
    cfg = from_legacy_env()
    assert not cfg.verification.enabled


def test_registry_builds_all_modules():
    cfg = load_harness_config(config_path("modules_full.yaml"))
    registry = ModuleRegistry.from_config(cfg)
    assert set(registry.modules) == {
        "retrieval",
        "evidence_state",
        "verification",
        "context_budget",
        "recovery",
    }


def test_apply_harness_config_sets_env():
    cfg = load_harness_config(config_path("ablate_verification.yaml"))
    env = apply_harness_config(cfg)
    assert env["V8D_VERIFY_TOOL"] == "0"
    assert env["ABLATE_VERIFY_UNAVAILABLE"] == "1"
