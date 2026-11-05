"""Data authenticity and isolation (§16.2) for the strict Probe-C loader."""

from __future__ import annotations

import pytest

from trim.probes.probe_c_contract import make_add_pair
from trim.probes.probe_c_data import (
    StrictLoadError,
    assert_group_disjoint,
    env_content_hash,
    load_strict_real_h0_states,
    split_audit,
    validate_state_row,
)
from tests.probe_c_fixtures import make_state_row


def _pair():
    return make_add_pair(["sentence_compress"], git_sha="g1")


def _rows_for_pair(*, n: int = 2):
    pair = _pair()
    return [
        make_state_row(
            query_id=f"q{i}",
            student_mask=pair.student_mask,
            teacher_mask=pair.teacher_mask,
        )
        for i in range(n)
    ]


def test_real_rows_load():
    states = load_strict_real_h0_states(_rows_for_pair(n=3), pair=_pair())
    assert len(states) == 3
    assert states[0].source == "real_h0_rollout"


def test_fake_source_rejected():
    row = make_state_row(source="fake_search_env")
    with pytest.raises(StrictLoadError, match="FAKE_OR_WRONG_SOURCE"):
        validate_state_row(row)


def test_missing_query_text_rejected():
    row = make_state_row(query_text="")
    with pytest.raises(StrictLoadError, match="MISSING_QUERY_TEXT"):
        validate_state_row(row)
    # query_id back-fill is forbidden: a placeholder text is not a query.
    row2 = make_state_row(query_text="N/A")
    with pytest.raises(StrictLoadError, match="MISSING_QUERY_TEXT"):
        validate_state_row(row2)


def test_missing_prompt_and_schema_rejected():
    with pytest.raises(StrictLoadError, match="MISSING_VISIBLE_HISTORY"):
        validate_state_row(make_state_row(raw_messages=[]))
    with pytest.raises(StrictLoadError, match="MISSING_EFFECTIVE_PROMPT_IDS"):
        validate_state_row(make_state_row(effective_student_prompt_ids=[]))
    with pytest.raises(StrictLoadError, match="MISSING_TOOL_SCHEMA"):
        validate_state_row(make_state_row(effective_tool_schema={}))


def test_truncation_lost_target_rejected():
    with pytest.raises(StrictLoadError, match="TRUNCATION_LOST_TARGET"):
        validate_state_row(make_state_row(truncation_lost_target=True))


def test_cross_split_query_and_group_rejected():
    pair = _pair()
    rows = [
        make_state_row(
            query_id="q1",
            split="C_train",
            scenario_group_id="grp1",
            student_mask=pair.student_mask,
            teacher_mask=pair.teacher_mask,
        ),
        make_state_row(
            query_id="q1",
            split="C_select",
            scenario_group_id="grp1",
            student_mask=pair.student_mask,
            teacher_mask=pair.teacher_mask,
        ),
    ]
    with pytest.raises(StrictLoadError, match="CROSS_SPLIT"):
        load_strict_real_h0_states(rows, pair=pair)


def test_teacher_student_branch_mix_rejected():
    pair = _pair()
    row = make_state_row(
        student_mask=pair.student_mask,
        teacher_mask=pair.teacher_mask,
        information_provenance=[
            {"owner": "teacher_full", "source": "teacher_branch"}
        ],
    )
    with pytest.raises(StrictLoadError, match="TEACHER_STUDENT_BRANCH_MIX"):
        validate_state_row(row)

    row2 = make_state_row(
        student_mask=pair.student_mask,
        teacher_mask=pair.teacher_mask,
        student_snapshot={
            **make_state_row()["student_snapshot"],
            "metadata": {"owner": "teacher_full"},
        },
    )
    with pytest.raises(StrictLoadError, match="TEACHER_STUDENT_BRANCH_MIX"):
        validate_state_row(row2)


def test_gold_doc_backfill_rejected():
    pair = _pair()
    row = make_state_row(
        student_mask=pair.student_mask,
        teacher_mask=pair.teacher_mask,
        information_provenance=[{"source": "gold_doc_store"}],
    )
    with pytest.raises(StrictLoadError, match="GOLD_DOC_STORE_BACKFILL"):
        validate_state_row(row)


def test_observation_copied_into_multiple_docs_rejected():
    pair = _pair()
    same = "identical observation text used as two different documents " * 2
    row = make_state_row(
        student_mask=pair.student_mask,
        teacher_mask=pair.teacher_mask,
        docs=[
            {"id": "d1", "text": same},
            {"id": "d2", "text": same},
        ],
        raw_messages=[
            {"role": "user", "content": "what is the capital of france"},
            # fabrication signature: the *observation* itself reused as docs
            {"role": "tool", "content": same},
        ],
    )
    with pytest.raises(StrictLoadError, match="OBSERVATION_COPIED"):
        validate_state_row(row)


def test_genuine_corpus_duplicate_docs_load():
    """Real corpora (e.g. SEC boilerplate) may share text across doc ids."""
    pair = _pair()
    same = "genuine corpus boilerplate shared by two filings of the same filer " * 2
    row = make_state_row(
        student_mask=pair.student_mask,
        teacher_mask=pair.teacher_mask,
        docs=[
            {"id": "0001213900-25-098707_27", "text": same},
            {"id": "0001213900-25-098713_27", "text": same},
        ],
    )
    state = validate_state_row(row)
    assert state.extra["duplicate_doc_text_groups"] == 1


def test_mask_pair_mismatch_rejected():
    pair = _pair()
    row = make_state_row(
        student_mask=pair.teacher_mask,  # wrong side
        teacher_mask=pair.teacher_mask,
    )
    with pytest.raises(StrictLoadError, match="MASK_PAIR_MISMATCH"):
        validate_state_row(row, pair=pair)


def test_env_content_hash_ignores_mask_and_metadata():
    base = make_state_row()["student_snapshot"]
    a = env_content_hash(base)
    tweaked = dict(base)
    tweaked["harness_mask"] = {"sentence_compress": True}
    tweaked["metadata"] = {"owner": "student_reduced", "tag": "other"}
    assert env_content_hash(tweaked) == a
    # Real environment content change is a different hash.
    changed = dict(base)
    changed["working_memory"] = {
        **base["working_memory"],
        "curated_ids": [],
    }
    assert env_content_hash(changed) != a


def test_split_audit_reports_actual_counts():
    states = load_strict_real_h0_states(_rows_for_pair(n=3), pair=_pair())
    audit = split_audit(states)
    assert audit["group_disjoint"] is True
    assert audit["counts"]["C_train"]["n_states"] == 3
    assert audit["report_actual_counts"] is True


def test_group_disjoint_direct_call():
    pair = _pair()
    rows = [
        make_state_row(
            query_id="a",
            split="C_train",
            scenario_group_id="g",
            student_mask=pair.student_mask,
            teacher_mask=pair.teacher_mask,
        ),
        make_state_row(
            query_id="b",
            split="C_audit",
            scenario_group_id="g",
            student_mask=pair.student_mask,
            teacher_mask=pair.teacher_mask,
        ),
    ]
    states = [validate_state_row(r) for r in rows]
    with pytest.raises(StrictLoadError, match="CROSS_SPLIT_GROUP"):
        assert_group_disjoint(states)
