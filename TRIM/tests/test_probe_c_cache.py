"""Cache invalidation and C0 funnel behavior (§16.3)."""

from __future__ import annotations

import pytest

from trim.probes.probe_c_cache import (
    FrozenTeacherCache,
    FunnelCounts,
    FunnelRow,
    STATUS_INVALID_MEASUREMENT,
    STATUS_NO_SUPPORT,
    STATUS_NO_TRIGGER,
    STATUS_READY,
    STATUS_UNREGISTERED,
    TeacherBranchOutcome,
    TeacherTarget,
    build_real_teacher_cache,
    compute_cache_key,
)
from trim.probes.probe_c_contract import make_add_pair
from trim.probes.probe_c_data import validate_state_row
from tests.probe_c_fixtures import make_state_row


def _key(**over):
    base = dict(
        teacher_policy_checksum="pol",
        model_revision="rev",
        tokenizer_hash="tok",
        chat_template_hash="ct",
        env_content_hash="env",
        component_revision="crev",
        resolved_pair_hash="pair",
        teacher_prompt_ids_hash="ph",
        target_action_ids_hash="th",
        effective_schema_hash="sch",
        truncation_policy_hash="tr",
        enable_thinking=False,
    )
    base.update(over)
    return compute_cache_key(**base)


def test_cache_key_sensitive_to_revisions():
    assert _key() == _key()
    assert _key() != _key(teacher_policy_checksum="pol2")
    assert _key() != _key(model_revision="rev2")
    assert _key() != _key(tokenizer_hash="tok2")
    assert _key() != _key(component_revision="crev2")
    assert _key() != _key(resolved_pair_hash="pair2")
    assert _key() != _key(enable_thinking=True)


def test_frozen_teacher_cache_survives_student_update():
    cache = FrozenTeacherCache()
    key = _key()
    calls = {"n": 0}

    def recompute():
        calls["n"] += 1
        return TeacherTarget(
            cache_key=key,
            state_id="s1",
            query_id="q1",
            component_id="sentence_compress",
            target_action={"name": "curate", "arguments": {"add_ids": ["d1"]}},
            target_action_ids=[10, 11, 12],
            teacher_token_logprobs=[-0.1, -0.2, -0.3],
            weights=[1.0, 1.0, 1.0],
            student_prompt_ids_hash="ph",
        )

    first = cache.compute_or_fetch(key, recompute=recompute)
    # A "student update" changes only the Student; the frozen Teacher result
    # must be byte-identical on the next fetch.
    second = cache.compute_or_fetch(key, recompute=recompute)
    assert first.teacher_token_logprobs == second.teacher_token_logprobs
    assert calls["n"] == 1  # single compute, second call hit the cache
    assert cache.manifest()["stores_student_logp"] is False
    assert cache.manifest()["stores_full_vocab_logits"] is False


def test_key_change_forces_recompute():
    cache = FrozenTeacherCache()
    k1, k2 = _key(), _key(component_revision="crev2")
    cache.compute_or_fetch(
        k1,
        recompute=lambda: TeacherTarget(
            cache_key=k1,
            state_id="s",
            query_id="q",
            component_id="m",
            target_action={"name": "a", "arguments": {}},
            target_action_ids=[1],
            teacher_token_logprobs=[-0.5],
            weights=[1.0],
            student_prompt_ids_hash="ph",
        ),
    )
    misses_before = cache.misses
    assert cache.get(k2) is None  # miss, never a silent hit
    assert cache.misses == misses_before + 1


def test_target_token_length_strict():
    with pytest.raises(ValueError, match="length mismatch"):
        TeacherTarget(
            cache_key="k",
            state_id="s",
            query_id="q",
            component_id="m",
            target_action={"name": "a", "arguments": {}},
            target_action_ids=[1, 2],
            teacher_token_logprobs=[-0.5],  # mismatch
            weights=[1.0, 1.0],
            student_prompt_ids_hash="ph",
        )


def _row(stage, status, **kw):
    return FunnelRow(
        state_id=kw.get("state_id", "s"),
        query_id=kw.get("query_id", "q"),
        component_id="sentence_compress",
        stage_reached=stage,
        status=status,
        n_supervised_tokens=kw.get("n_supervised_tokens", 0),
    )


def test_funnel_status_machine():
    ready = FunnelCounts(
        component_id="m",
        rows=[
            _row("gradient_bearing_row", STATUS_READY, query_id="q1", n_supervised_tokens=5),
            _row("gradient_bearing_row", STATUS_READY, query_id="q2", n_supervised_tokens=3),
        ],
    )
    assert ready.status(min_covered_queries=2) == STATUS_READY

    no_trigger = FunnelCounts(
        component_id="m", rows=[_row("opportunity", STATUS_NO_TRIGGER)]
    )
    assert no_trigger.status() == STATUS_NO_TRIGGER

    no_support = FunnelCounts(
        component_id="m", rows=[_row("projection_direct", STATUS_NO_SUPPORT)]
    )
    assert no_support.status() == STATUS_NO_SUPPORT

    unreg = FunnelCounts(
        component_id="m", rows=[_row("opportunity", STATUS_UNREGISTERED)]
    )
    assert unreg.status() == STATUS_UNREGISTERED

    bad = FunnelCounts(
        component_id="m",
        rows=[_row("encoding_valid", STATUS_INVALID_MEASUREMENT)],
    )
    assert bad.status() == STATUS_INVALID_MEASUREMENT


def test_funnel_zero_denominator_is_na():
    empty = FunnelCounts(component_id="m", rows=[])
    ratios = empty.ratios()
    assert ratios["trigger_over_opportunity"] == "N/A"
    assert ratios["supervised_states_over_all_states"] == "N/A"


def test_funnel_low_coverage_is_not_ready():
    rows = [
        _row("gradient_bearing_row", STATUS_READY, query_id=f"q{i}", n_supervised_tokens=2)
        for i in range(3)
    ]
    counts = FunnelCounts(component_id="m", rows=rows)
    assert counts.status(min_covered_queries=16) == STATUS_NO_SUPPORT
    assert counts.support_ready(min_covered_queries=2) is True


def test_build_cache_funnel_no_trigger_and_ready():
    pair = make_add_pair(["sentence_compress"], git_sha="g1")
    state = validate_state_row(
        make_state_row(
            student_mask=pair.student_mask, teacher_mask=pair.teacher_mask
        ),
        pair=pair,
    )
    # No trigger.
    rows, funnel = build_real_teacher_cache(
        [state],
        pair,
        component_id="sentence_compress",
        executor=lambda s, p: TeacherBranchOutcome(
            triggered=False, source_type="skip_untriggered", reject_reason="not_triggered"
        ),
        encoder=lambda s, a: ([1, 2], [True, True]),
        teacher_logp_fn=lambda prompt, target: [-0.1] * len(target),
    )
    assert rows == []
    assert funnel.status() == STATUS_NO_TRIGGER

    # READY path with a real capability source and DIRECT projection.
    def executor(s, p):
        return TeacherBranchOutcome(
            triggered=True,
            source_type="capability_effect",
            events=[object()],
            projection_kind="direct",
            projected_steps=[
                type(
                    "Step",
                    (),
                    {"target_action": {"name": "curate", "arguments": {"add_ids": ["d1"]}}},
                )()
            ],
            realizable=True,
        )

    rows2, funnel2 = build_real_teacher_cache(
        [state],
        pair,
        component_id="sentence_compress",
        executor=executor,
        encoder=lambda s, a: ([10, 11], [True, True]),
        teacher_logp_fn=lambda prompt, target: [-0.1] * len(target),
    )
    assert len(rows2) == 1
    assert funnel2.status(min_covered_queries=1) == STATUS_READY
    assert rows2[0].target.teacher_token_logprobs == [-0.1, -0.1]
    # Funnel reaches the final stage.
    assert funnel2.rows[-1].stage_reached == "gradient_bearing_row"


def test_synthetic_source_is_not_real_support():
    pair = make_add_pair(["sentence_compress"], git_sha="g1")
    state = validate_state_row(
        make_state_row(
            student_mask=pair.student_mask, teacher_mask=pair.teacher_mask
        ),
        pair=pair,
    )
    rows, funnel = build_real_teacher_cache(
        [state],
        pair,
        component_id="sentence_compress",
        executor=lambda s, p: TeacherBranchOutcome(
            triggered=True,
            source_type="synthetic_heuristic",
            events=[object()],
            reject_reason="synthetic_heuristic_not_a_real_capability_source",
        ),
        encoder=lambda s, a: ([1], [True]),
        teacher_logp_fn=lambda prompt, target: [-0.1] * len(target),
    )
    assert rows == []
    assert funnel.rows[-1].status == STATUS_NO_SUPPORT
    assert funnel.rows[-1].stage_reached == "real_capability_source"
