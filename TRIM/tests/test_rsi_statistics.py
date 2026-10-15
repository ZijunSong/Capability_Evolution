from trim.rsi.statistics import INSUFFICIENT, cluster_paired_delta


def test_seeds_inside_one_question_do_not_inflate_n():
    rows = []
    for seed in (11, 29):
        rows.append({"cluster_id": "c1", "query_id": "q1", "seed": seed, "j_base": 0.2, "j_other": 0.4})
        rows.append({"cluster_id": "c2", "query_id": "q2", "seed": seed, "j_base": 0.0, "j_other": 0.2})
    report = cluster_paired_delta(rows, n_boot=50, seed=1)
    assert report["n_clusters"] == 2
    assert report["n_rows"] == 4
    assert report["status"] == "ok"


def test_all_zero_deltas_do_not_claim_noninferiority():
    rows = [
        {"cluster_id": "c1", "query_id": "q1", "j_base": 0.2, "j_other": 0.2},
        {"cluster_id": "c2", "query_id": "q2", "j_base": 0.4, "j_other": 0.4},
    ]
    report = cluster_paired_delta(rows, n_boot=40, seed=1)
    assert report["status"] == INSUFFICIENT
    assert report["interval"] is None
    assert report["lcb"] is None


def test_missing_pairs_are_not_dropped_from_the_denominator():
    rows = [
        {"cluster_id": "c1", "query_id": "q1", "j_base": 0.0, "j_other": 0.2},
        {"cluster_id": "c2", "query_id": "q2", "j_base": None, "j_other": 0.2},
    ]
    report = cluster_paired_delta(rows, require_complete=True)
    assert report["status"] == INSUFFICIENT
    assert report["n_clusters"] == 0
    assert report["reason"] == "missing_pair"
