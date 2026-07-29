"""Tests for H_min_v2 harness config."""

from __future__ import annotations

from pathlib import Path

import yaml

from harness.harness_config import load_harness_config


def test_hmin_v2_loadable():
    path = Path(__file__).resolve().parents[2] / "harness/configs/modules_minimal_v2.yaml"
    cfg = load_harness_config(path)
    assert cfg.retrieval.enabled is True
    assert cfg.evidence_state.enabled is False
    assert cfg.verification.enabled is True
    assert cfg.recovery.enabled is False


def test_hmin_v2_external_verifier_callable():
    path = Path(__file__).resolve().parents[2] / "harness/configs/modules_minimal_v2.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    ver = raw["verification"]
    assert ver["expose_verify_tool"] is True


def test_hmin_v2_no_harness_auto_verify_decision():
    path = Path(__file__).resolve().parents[2] / "harness/configs/modules_minimal_v2.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    ver = raw["verification"]
    assert ver.get("harness_auto_verify_decision") is False
    assert ver.get("verification_aware_curation") is False


def test_hmin_v2_cognitive_policies_disabled():
    path = Path(__file__).resolve().parents[2] / "harness/configs/modules_minimal_v2.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    ev = raw["evidence_state"]
    assert ev["enabled"] is False
    assert ev["content_dedup"] is False
    assert ev["subtractive_curation"] is False
    ctx = raw["context_budget"]
    assert ctx["deterministic_truncation"] is True
    assert ctx.get("stop_budget_hint") is False


def test_hmin_v2_hard_budget():
    path = Path(__file__).resolve().parents[2] / "harness/configs/modules_minimal_v2.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert raw["context_budget"]["deterministic_truncation"] is True
