"""Refactor guide §10.1 + §9: per-role panel caps and analysis fragments."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_final_test_role_has_no_cap():
    """confirm caps must not clip final_test: the role cap map gives it None."""
    from rsi.campaign.controller import CampaignController

    src = inspect.getsource(CampaignController._eval_panel)
    assert '"final_test": None' in src
    assert '"transfer_test": None' in src
    assert "role_caps.get(role)" in src


def test_campaign_eval_panel_payload_carries_telemetry():
    from rsi.campaign.controller import CampaignController

    src = inspect.getsource(CampaignController._eval_panel)
    for key in ("metric_name", "partial_test_pass_rate", "n_decisions", "component_errors"):
        assert key in src, f"panel payload missing {key}"
    assert 'trigger_stats' not in src or '"trigger_stats": {}' not in src


def test_evolve_search_accepts_analysis_runner_and_uses_fragments():
    from rsi.evolution.search import EvolutionSearch

    assert "analysis_runner" in inspect.signature(EvolutionSearch.__init__).parameters
    assert hasattr(EvolutionSearch, "_fragments_from_analysis")
    # both campaign and CLI entries wire it
    from rsi.campaign import controller as cc
    from rsi.evolution import service as es

    assert "analysis_runner=analysis_runner" in inspect.getsource(cc.CampaignController._stage_evolve)
    assert "analysis_runner=analysis_runner" in inspect.getsource(es.run_evolve)
