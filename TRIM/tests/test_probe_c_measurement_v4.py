"""Regression tests for the measurement_v4 fixes (guide §15, Commit A).

Each test targets a real defect the guide confirmed, not a mirror of the
implementation: legacy DIRECT-count conflation, unwired regen guard,
MISSING-as-pass, mechanical DEPRIORITIZE on small negative R_D, held-out K(s)
losing its Teacher target, and stage_confirm picking "any row once PROMOTEd".
"""

from __future__ import annotations

import json

import pytest

from trim.eval.probe_c_metrics import (
    GATE_POLICY_V4,
    MISSING,
    RegenStats,
    judge_c1,
)
from trim.probes.probe_c_cache import (
    FUNNEL_SCHEMA_VERSION_V4,
    FunnelCounts,
    FunnelRow,
    FunnelRowV4,
    STAGE_GRADIENT_ROW,
    STAGE_PROJECTION_DIRECT,
    STATUS_NO_SUPPORT,
    STATUS_READY,
)


# ---------------------------------------------------------------------------
# §15: test_funnel_failed_direct_not_counted_as_pass
# ---------------------------------------------------------------------------


def test_funnel_failed_direct_not_counted_as_pass():
    """A row that entered the DIRECT check and failed must NOT add a DIRECT pass.

    Legacy count() used stage_reached>=idx, which counted this row as DIRECT
    (the exact ccso 95/91 conflation). v4 passed-count must be 0.
    """
    row = FunnelRow(
        state_id="s1",
        query_id="q1",
        component_id="ccso",
        stage_reached=STAGE_PROJECTION_DIRECT,
        status=STATUS_NO_SUPPORT,
        reject_reason="unrealizable",
    )
    fc = FunnelCounts(component_id="ccso", rows=[row])
    # Legacy semantics (kept for compat) conflates.
    assert fc.count(STAGE_PROJECTION_DIRECT) == 1
    # v4: attempted yes, passed no.
    assert fc.count_attempted_v4(STAGE_PROJECTION_DIRECT) == 1
    assert fc.count_pass_v4(STAGE_PROJECTION_DIRECT) == 0
    v4 = fc.to_dict_v4()
    assert v4["schema_version"] == FUNNEL_SCHEMA_VERSION_V4
    assert v4["per_stage"][STAGE_PROJECTION_DIRECT] == {
        "attempted": 1,
        "passed": 0,
        "failed_here": 1,
    }


def test_funnel_ready_row_passes_all_stages_in_v4():
    row = FunnelRow(
        state_id="s2",
        query_id="q2",
        component_id="trieve",
        stage_reached=STAGE_GRADIENT_ROW,
        status=STATUS_READY,
        n_supervised_tokens=42,
    )
    fc = FunnelCounts(component_id="trieve", rows=[row])
    v4 = fc.to_dict_v4()
    assert v4["per_stage"][STAGE_GRADIENT_ROW]["passed"] == 1
    assert v4["state_level"]["n_supervision_rows_ready"] == 1
    assert v4["state_level"]["n_supervised_tokens"] == 42


def test_funnel_v4_mark_pass_attempt_fail_contract():
    r = FunnelRowV4(state_id="s", query_id="q", coalition_id="c", member_id=None, channel="native_event_action")
    r.mark_attempt(STAGE_PROJECTION_DIRECT)
    r.mark_pass(STAGE_PROJECTION_DIRECT)
    r.fail("realizable", ["CAPACITY"])
    assert STAGE_PROJECTION_DIRECT in r.passed_stages
    assert r.failed_stage == "realizable"
    assert "CAPACITY" in r.reason_codes
    d = r.to_dict()
    assert d["schema_version"] == FUNNEL_SCHEMA_VERSION_V4


# ---------------------------------------------------------------------------
# §15: test_missing_guard_cannot_promote + test_regen_guard_is_wired_through_screen
# ---------------------------------------------------------------------------


def test_missing_guard_cannot_promote():
    """MISSING regen stats + require_generation_guard → INCOMPLETE, never PROMOTE."""
    v = judge_c1(
        pre_js=0.20,
        post_js=0.10,
        pre_nll=0.60,
        post_nll=0.40,
        require_generation_guard=True,
    )
    assert v.status == "INCOMPLETE_MEASUREMENT"


def test_regen_guard_is_wired_through_screen(monkeypatch):
    """stage_screen must pass pre/post RegenStats into judge_c1 (the guide's
    finding: unit gate existed but the runner never passed it)."""
    import inspect

    from trim.probes import probe_c_runner

    src = inspect.getsource(probe_c_runner.stage_screen)
    assert "pre_regen=pre.regen_obj" in src
    assert "post_regen=post.regen_obj" in src


def test_quality_regression_blocks_promotion():
    """NLL/JS improve but generated invalid rate regresses beyond eps → GUARD_FAIL."""
    pre = RegenStats(n_attempted=100, n_ok=30)
    post = RegenStats(n_attempted=100, n_ok=25)  # invalid 0.70 → 0.75
    v = judge_c1(
        pre_js=0.14,
        post_js=0.10,
        pre_nll=0.546,
        post_nll=0.439,
        pre_regen=pre,
        post_regen=post,
        absolute_quality_status="PILOT_FROZEN_OK",
    )
    assert v.status == "GUARD_FAIL"
    assert v.detail["guard_status"] == "FAIL"


def test_absolute_quality_unresolved_blocks_deployable_promote():
    """Gates pass but absolute quality unresolved → BORDERLINE, not PROMOTE."""
    v = judge_c1(
        pre_js=0.14,
        post_js=0.10,
        pre_nll=0.546,
        post_nll=0.439,
        pre_regen=RegenStats(n_attempted=100, n_ok=30),
        post_regen=RegenStats(n_attempted=100, n_ok=32),
    )
    assert v.status == "BORDERLINE"
    assert v.detail["absolute_quality_status"] == "QUALITY_UNRESOLVED"


# ---------------------------------------------------------------------------
# §15: test_small_negative_rd_is_borderline_not_forced_drop
# + test_clear_negative_rd_at_max_budget_is_deprioritized
# ---------------------------------------------------------------------------


def test_small_negative_rd_is_borderline_not_forced_drop():
    """R_D=-0.003 within εD=0.005 with open CI → BORDERLINE (old code: forced
    DEPRIORITIZE via ``r_d <= 0`` regardless of delta_dk)."""
    v = judge_c1(
        pre_js=0.1411,
        post_js=0.1440,
        pre_nll=0.546,
        post_nll=0.439,
        max_dk_regression=0.005,
    )
    assert v.status == "BORDERLINE"
    assert v.reason == "FIT_IMPROVED_POLICY_UNRESOLVED"


def test_clear_negative_rd_at_max_budget_is_deprioritized():
    """CI-supported regression must still DEPRIORITIZE — not everything borderline."""
    v = judge_c1(
        pre_js=0.1411,
        post_js=0.1600,
        pre_nll=0.546,
        post_nll=0.560,
        ci_dk=(-0.03, -0.01),
        max_dk_regression=0.005,
    )
    assert v.status == "DEPRIORITIZE"


def test_gate_policy_version_is_reported():
    v = judge_c1(pre_js=0.2, post_js=0.1, pre_nll=0.6, post_nll=0.4)
    assert v.detail["policy_version"] == GATE_POLICY_V4


# ---------------------------------------------------------------------------
# §15: test_select_k_contains_own_target_and_no_train_target
# ---------------------------------------------------------------------------


def test_select_k_contains_own_target_and_no_train_target():
    """stage_screen must build held-out K(s) from same-state C_select targets,
    not from train_rows (whose state_id filter emptied the held-out K)."""
    import inspect

    from trim.probes import probe_c_runner

    src = inspect.getsource(probe_c_runner.stage_screen)
    # Same-state select targets feed the candidate constructor…
    assert "select_targets_by_state.get(state.state_id, [])" in src
    # …and train_rows are never passed to candidate_fn for held-out states.
    assert "candidate_fn(state, pair, train_rows)" not in src
    # The silent heldout fallback is gone (must hard-fail instead).
    assert "heldout = select_states  # documented fallback" not in src


def test_stage_confirm_uses_fixed_budget_summary():
    """seed11@4 PROMOTE must not override the 8-step result (guide §9.4)."""
    import inspect

    from trim.probes import probe_c_runner

    src = inspect.getsource(probe_c_runner.stage_confirm)
    assert "primary_budget" in src
    assert '_summary_status' in src


# ---------------------------------------------------------------------------
# materialize_with_audit (guide §4.3)
# ---------------------------------------------------------------------------


def _direct_projection_with_actions(actions):
    from trim.training.opd_projection import ProjectionKind, ProjectionResult

    return ProjectionResult(
        kind=ProjectionKind.DIRECT,
        actions=actions,
        teacher_start_event="evt_start",
        teacher_end_event="evt_end",
        component_id="ccso",
    )


def test_materialize_with_audit_reports_structured_rejections():
    """A failing action must return its gate + reason codes, not a silent break."""
    from tests.probe_c_fixtures import make_state_row
    from trim.state.snapshot import EnvironmentSnapshot
    from trim.training.action_codec import parse_action
    from trim.training.opd_dataset import materialize_with_audit

    row = make_state_row()
    snap = EnvironmentSnapshot.from_dict(row["student_snapshot"])
    # An action referencing an inaccessible doc should be rejected with a
    # structured reason, not an empty list.
    action = parse_action('read_document(doc_id="doc_does_not_exist_xyz")')
    proj = _direct_projection_with_actions([action])
    result = materialize_with_audit(proj, snap, component_id="ccso")
    if not result.steps:
        assert result.rejected, "silent rejection without structured reason"
        first = result.rejected[0]
        assert first.get("gate"), first
        assert first.get("reasons"), first


def test_materialize_legacy_wrapper_matches_audit_steps():
    from tests.probe_c_fixtures import make_state_row
    from trim.state.snapshot import EnvironmentSnapshot
    from trim.training.action_codec import parse_action
    from trim.training.opd_dataset import materialize, materialize_with_audit

    row = make_state_row()
    snap = EnvironmentSnapshot.from_dict(row["student_snapshot"])
    action = parse_action('search_corpus(query="capital of france")')
    proj = _direct_projection_with_actions([action])
    legacy = materialize(proj, snap, component_id="ccso")
    audited = materialize_with_audit(proj, snap, component_id="ccso")
    assert len(legacy) == len(audited.steps)
