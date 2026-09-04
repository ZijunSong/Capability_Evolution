from __future__ import annotations

from trim.adapters.components import full_mask, minus_mask
from trim.training.tool_mask import legal_tool_names


def test_subtractive_curate_action_reachable_under_minus_mask():
    tools = legal_tool_names()
    assert "curate" in tools

    full = full_mask()
    minus = minus_mask("subtractive_curation")
    assert full["subtractive_curation"] is True
    assert minus["subtractive_curation"] is False
    assert set(full) == set(minus)

    # Component masks change rendered privilege, not the canonical action space.
    assert "curate" in legal_tool_names()
