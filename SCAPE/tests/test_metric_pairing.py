from __future__ import annotations

import pytest

from scape.eval.paired_bootstrap import pair_by_query_id, paired_query_stats


def test_metric_pairing_by_query_id():
    rows_a = [
        {"query_id": "q1", "recall": 0.2},
        {"query_id": "q2", "recall": 0.4},
        {"query_id": "q3", "recall": 0.5},
    ]
    rows_b = [
        {"query_id": "q2", "recall": 0.5},
        {"query_id": "q1", "recall": 0.1},
        {"query_id": "q3", "recall": 0.5},
    ]
    a, b = pair_by_query_id(rows_a, rows_b)
    stats = paired_query_stats(a, b, metric="recall", n_boot=200, seed=0)
    assert stats["n"] == 3
    assert stats["win"] + stats["loss"] + stats["tie"] == 3
    # other - base: q1: -0.1, q2: +0.1, q3: 0
    assert stats["tie"] == 1
    assert abs(stats["mean_delta"] - 0.0) < 1e-9

    with pytest.raises(ValueError):
        paired_query_stats(a, {"q1": {"recall": 1.0}}, metric="recall")
