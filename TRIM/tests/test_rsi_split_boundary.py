import pytest

from trim.rsi.splits import QuestionRecord, SplitError, assign_roles, assert_disjoint_from_forbidden
from trim.training.four_cell_runtime import _restrict_resolved_queries


def test_roles_cannot_share_a_cluster():
    records = [QuestionRecord("q1", "family"), QuestionRecord("q2", "family")]
    with pytest.raises(SplitError, match="cluster"):
        assign_roles(records, roles={"work": ["q1"], "confirm_h_r000": ["q2"]})


def test_train_filter_refuses_the_final_test_ids():
    rows = [{"query_id": "keep"}, {"query_id": "secret"}]

    class Args:
        train_query_id_list = ["keep"]
        train_query_ids = None
        eval_query_id_list = []
        eval_query_ids = None
        final_test_query_ids = ["secret"]

    kept, _eval, _meta, _frozen = _restrict_resolved_queries(Args(), rows, [], {}, [])
    assert [row["query_id"] for row in kept] == ["keep"]

    class Overlap(Args):
        train_query_id_list = ["secret"]

    with pytest.raises(ValueError, match="forbidden"):
        _restrict_resolved_queries(Overlap(), rows, [], {}, [])


def test_weight_dev_and_test_are_separate():
    records = [
        QuestionRecord("w", "cw"),
        QuestionRecord("d", "cd"),
        QuestionRecord("t", "ct"),
    ]
    manifest = assign_roles(records, roles={"work": ["w"], "weight_dev": ["d"], "test": ["t"]})
    assert_disjoint_from_forbidden(manifest.ids("work"), manifest.ids("test"), role="work")
    assert set(manifest.ids("weight_dev")).isdisjoint(manifest.ids("test"))
