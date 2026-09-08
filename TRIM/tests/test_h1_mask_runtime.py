"""Harness-1 mask must change live env behavior, not just LAUNCH.json."""

from __future__ import annotations

import pytest

from trim.adapters.components import full_mask, zero_mask
from trim.eval.h1_component_runtime import apply_auto_populate, legal_tool_set
from trim.eval.local_search_env import execute_tool, new_state, wm_text
from trim.eval.runtime_effect_audit import audit_mask_wiring, merge_audits


STORE = {
    "d1": {"id": "d1", "text": "Alice Smith visited Paris in 2019. Bob Jones signed the treaty.", "score": 0.9},
    "d2": {"id": "d2", "text": "Carol Adams founded the company later.", "score": 0.4},
    "d3": {"id": "d3", "text": "Alice Smith visited Paris in 2019. Bob Jones signed the treaty.", "score": 0.8},
}


def test_zero_mask_rejects_verify():
    st = new_state("Alice Smith", STORE, harness_mask=zero_mask())
    st, obs, ok = execute_tool(st, "verify", {"doc_ids": ["d1"], "claim": "Alice visited Paris"})
    assert ok is False
    assert "invalid tool" in obs
    assert "verify" not in legal_tool_set(zero_mask())


def test_full_mask_accepts_verify():
    st = new_state("Alice Smith", STORE, harness_mask=full_mask())
    st, _obs, ok = execute_tool(st, "search_corpus", {"query": "Alice Paris"})
    assert ok is True
    st, obs, ok = execute_tool(st, "verify", {"doc_ids": ["d1"], "claim": "Alice visited Paris"})
    assert ok is True
    assert "Verify claim" in obs
    assert (st.get("runtime_effects") or {}).get("verify_tool", 0) >= 1


def test_auto_populate_fires_only_when_mask_on():
    off = new_state("Alice Smith Paris", STORE, harness_mask=zero_mask())
    off, _obs, _ok = execute_tool(off, "search_corpus", {"query": "Alice Paris"})
    assert not off.get("auto_seed")
    assert off.get("curated") in ({}, None) or len(off.get("curated") or {}) == 0

    on = new_state("Alice Smith Paris", STORE, harness_mask=full_mask())
    on, obs, ok = execute_tool(on, "search_corpus", {"query": "Alice Paris"})
    assert ok is True
    assert on.get("auto_seed")
    assert len(on.get("curated") or {}) >= 1
    assert "[AUTO]" in obs
    assert apply_auto_populate is not None


def test_wm_text_follows_mask_not_hardcoded_auto_off():
    st = new_state("Alice Smith Paris", STORE, harness_mask=full_mask())
    st, _obs, _ok = execute_tool(st, "search_corpus", {"query": "Alice Paris"})
    text = wm_text(st)
    assert "auto_populate_first_search=ON" in text
    assert "mask=full" in text
    assert st.get("auto_seed") is not None
    assert str(st["auto_seed"][0]) in text or "auto-populated" in text

    zero = new_state("Alice Smith Paris", STORE, harness_mask=zero_mask())
    zero, _obs, _ok = execute_tool(zero, "search_corpus", {"query": "Alice Paris"})
    ztext = wm_text(zero)
    assert "auto_populate_first_search=OFF" in ztext
    assert "mask=zero" in ztext
    assert "importance=fair" not in ztext


def test_importance_and_graph_and_budget_are_mask_gated():
    on = new_state("Alice Smith Paris", STORE, harness_mask=full_mask())
    on, obs, _ok = execute_tool(on, "search_corpus", {"query": "Alice Paris"})
    assert "[Evidence Graph]" in obs
    assert "[Context:" in obs
    on["curated"] = {}
    on["auto_seed"] = None
    on["importance"] = {}
    on, _obs, _ok = execute_tool(
        on, "curate", {"add_ids": ["d1"], "importance": {"d1": "very_high"}}
    )
    assert on["importance"].get("d1") == "very_high"
    assert "importance=very_high" in wm_text(on)

    off = new_state("Alice Smith Paris", STORE, harness_mask=zero_mask())
    off, obs, _ok = execute_tool(off, "search_corpus", {"query": "Alice Paris"})
    assert "[Evidence Graph]" not in obs
    assert "[Context:" not in obs
    off, _obs, _ok = execute_tool(
        off, "curate", {"add_ids": ["d1"], "importance": {"d1": "very_high"}}
    )
    assert "d1" not in (off.get("importance") or {}) or off["importance"].get("d1") != "very_high"


def test_content_dedup_drops_near_duplicate_only_when_on():
    on = new_state("Alice Smith Paris", STORE, harness_mask=full_mask())
    on, _obs, _ok = execute_tool(on, "search_corpus", {"query": "Alice Paris"})
    assert on.get("dedup_dropped")
    off = new_state("Alice Smith Paris", STORE, harness_mask=zero_mask())
    off, _obs, _ok = execute_tool(off, "search_corpus", {"query": "Alice Paris"})
    assert not off.get("dedup_dropped")


def test_apply_auto_populate_has_a_live_call_site():
    import inspect

    from trim.eval import local_search_env

    src = inspect.getsource(local_search_env.execute_tool)
    assert "_apply_auto_populate" in src
    assert "auto_populate_first_search" in src


def test_batched_rollout_passes_mask_into_new_state():
    pytest.importorskip("torch")
    from trim.training.batched_env_rollout import _prepare_chunk_episodes
    from trim.training.four_cell_runtime import doc_store_for_row

    mask = full_mask()
    rows = [
        {
            "query_id": "q1",
            "query": "Alice Smith Paris",
            "gold_docids": ["d1"],
            "frozen_doc_store": STORE,
        }
    ]
    episodes = _prepare_chunk_episodes(
        rows,
        component_id="all",
        group_size=1,
        policy_version="eval",
        seed=0,
        harness_mask=mask,
        searcher=None,
        doc_store_k=8,
        doc_store_workers=1,
        new_state=lambda q, store: new_state(q, store, harness_mask=mask),
        doc_store_for_row=doc_store_for_row,
    )
    del doc_store_for_row
    assert episodes
    assert episodes[0].st.get("harness_mask") == mask
    assert episodes[0].harness_mask == mask


def test_wiring_probe_full_passes_and_zero_is_inert():
    full = audit_mask_wiring(full_mask())
    zero = audit_mask_wiring(zero_mask())
    assert full["pass"], full["failures"]
    assert zero["pass"], zero["failures"]
    merged = merge_audits(full)
    assert merged["pass"]
    assert merged["claim_usable_for_full_vs_zero"] is False
