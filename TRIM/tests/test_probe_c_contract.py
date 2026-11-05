"""Pair regression (§16.1) + contract validation for Probe C."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trim.adapters.role_masks import (
    add_pair_masks,
    student_mask_for_ids,
    teacher_mask_for_ids,
)
from trim.probes.probe_c_contract import (
    ContractError,
    UNRESOLVED_REVISION,
    load_config,
    make_add_pair,
    make_legacy_full_minus_pair,
)


def test_add_pair_teacher_enables_only_m():
    pair = make_add_pair(["sentence_compress"], git_sha="g1")
    # Student = H0: every advanced component off.
    assert all(v is False for v in pair.student_mask.values())
    # Teacher: exactly m on, everything else off.
    on = [k for k, v in pair.teacher_mask.items() if v]
    assert on == ["sentence_compress"]
    assert pair.diff_components() == ("sentence_compress",)


def test_add_pair_diff_set_equals_component_ids():
    pair = make_add_pair(
        ["sentence_compress", "verify_tool"], git_sha="g1"
    )
    assert set(pair.diff_components()) == {"sentence_compress", "verify_tool"}


def test_add_pair_composite_requires_explicit_dependencies():
    # Declaring a dependency outside component_ids is refused, not silently added.
    with pytest.raises(ContractError):
        make_add_pair(
            ["sentence_compress"],
            git_sha="g1",
            declared_dependencies=["verify_tool"],
        )
    pair = make_add_pair(
        ["sentence_compress", "verify_tool"],
        git_sha="g1",
        declared_dependencies=["verify_tool"],
    )
    assert set(pair.component_ids) == {"sentence_compress", "verify_tool"}


def test_add_pair_masks_helper_matches_spec():
    student, teacher = add_pair_masks(["verify_tool"])
    pair = make_add_pair(["verify_tool"], git_sha="g1")
    assert student == pair.student_mask
    assert teacher == pair.teacher_mask


def test_legacy_paths_unchanged():
    """Legacy default Full-minus-m / Full semantics must not change."""
    student = student_mask_for_ids(["sentence_compress"])
    teacher = teacher_mask_for_ids(["sentence_compress"])
    # Student: Full-minus-m (all defaults on except m).
    assert student["sentence_compress"] is False
    assert student["verify_tool"] is True
    # Teacher: Full with m on.
    assert teacher["sentence_compress"] is True
    legacy = make_legacy_full_minus_pair(["sentence_compress"], git_sha="g1")
    assert legacy.student_mask == student
    assert legacy.teacher_mask == teacher


def test_teacher_encoding_does_not_reopen_full_with_explicit_mask():
    """encode_aligned_teacher_prompt must honour an explicit Teacher mask."""
    import trim.training.four_cell_runtime as fcr

    pair = make_add_pair(["sentence_compress"], git_sha="g1")
    frozen_st = {
        "query": "q",
        "harness_mask": dict(pair.student_mask),
        "pool": {},
        "curated": {},
        "tool_history": [],
        "step": 0,
    }
    captured: dict = {}

    class _Enc:
        def build_first_turn_prompt_ids(self, query):
            captured["query"] = query
            return [1, 2, 3]

        def build_continuation_prompt_ids(self, query, actions_obs=None, wm_text=""):
            captured["query"] = query
            captured["wm_text"] = wm_text
            return [1, 2, 3, 4]

    fcr.encode_aligned_teacher_prompt(
        _Enc(),
        "q",
        frozen_st=frozen_st,
        frozen_acts=[],
        component_id="sentence_compress",
        teacher_mask=dict(pair.teacher_mask),
    )
    # Explicit mask wins; nothing re-derived as default Full.
    assert captured.get("query") == "q"

    # Wrong key set must raise, not silently patch.
    with pytest.raises(ValueError):
        fcr.encode_aligned_teacher_prompt(
            _Enc(),
            "q",
            frozen_st=frozen_st,
            frozen_acts=[],
            component_id="sentence_compress",
            teacher_mask={"only_one": True},
        )


def test_config_rejects_unresolved_revision(tmp_path: Path):
    cfg = {
        "schema_version": "probe_c_v3",
        "model_id": "Qwen/Qwen3.5-9B",
        "model_revision": UNRESOLVED_REVISION,
        "pair_mode": "add_from_h0",
        "components": ["sentence_compress"],
        "state_source": "real_h0_rollout",
        "projection_mode": "direct_only",
        "probe_objective": "frozen_projected_gap",
        "loss_scale": 1.0,
        "splits": {"frozen": True},
    }
    path = tmp_path / "c.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ContractError, match="unresolved"):
        load_config(path, stage="audit")


def test_config_rejects_unfrozen_split_and_wrong_objective(tmp_path: Path):
    base = {
        "schema_version": "probe_c_v3",
        "model_id": "Qwen/Qwen3.5-9B",
        "model_revision": "rev-1",
        "pair_mode": "add_from_h0",
        "components": ["sentence_compress"],
        "state_source": "real_h0_rollout",
        "projection_mode": "direct_only",
        "probe_objective": "frozen_projected_gap",
        "loss_scale": 1.0,
    }
    unfrozen = dict(base, splits={"frozen": False})
    p1 = tmp_path / "u.json"
    p1.write_text(json.dumps(unfrozen), encoding="utf-8")
    with pytest.raises(ContractError, match="not frozen"):
        load_config(p1)

    ce = dict(base, probe_objective="sr_opd_ce", splits={"frozen": True})
    p2 = tmp_path / "ce.json"
    p2.write_text(json.dumps(ce), encoding="utf-8")
    with pytest.raises(ContractError, match="probe_objective"):
        load_config(p2)

    lam = dict(base, loss_scale=0.01, splits={"frozen": True})
    p3 = tmp_path / "lam.json"
    p3.write_text(json.dumps(lam), encoding="utf-8")
    with pytest.raises(ContractError, match="loss_scale"):
        load_config(p3)


def test_config_rejects_fake_state_source(tmp_path: Path):
    cfg = {
        "schema_version": "probe_c_v3",
        "model_id": "m",
        "model_revision": "rev-1",
        "pair_mode": "add_from_h0",
        "components": ["sentence_compress"],
        "state_source": "fake_search_env",
        "projection_mode": "direct_only",
        "probe_objective": "frozen_projected_gap",
        "loss_scale": 1.0,
        "splits": {"frozen": True},
    }
    p = tmp_path / "f.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ContractError, match="state_source"):
        load_config(p)


def test_pair_hash_stable_and_revision_sensitive():
    a = make_add_pair(["sentence_compress"], git_sha="g1")
    b = make_add_pair(["sentence_compress"], git_sha="g1")
    c = make_add_pair(["sentence_compress"], git_sha="g2")
    assert a.resolved_pair_hash() == b.resolved_pair_hash()
    assert a.resolved_pair_hash() != c.resolved_pair_hash()
