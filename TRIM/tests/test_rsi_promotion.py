from trim.rsi.promotion import FULL, PARTIAL, REJECT, evaluate_promotion


def _rows(quality: float, cost: float, clusters=("c1", "c2")):
    return [
        {"cluster_id": cid, "query_id": cid, "seed": 1, "quality": quality + (0.01 if cid == "c2" else 0.0), "cost": cost}
        for cid in clusters
    ]


def test_s2_gain_without_matching_s0_is_partial():
    cells = {
        "S0": _rows(0.50, 10),
        "S1": _rows(0.20, 4),
        "S2": _rows(0.30, 4),
        "S3": _rows(0.55, 10),
        "retention": [
            {"cluster_id": "h1", "query_id": "h1", "j_base": 0.40, "j_other": 0.41},
            {"cluster_id": "h2", "query_id": "h2", "j_base": 0.30, "j_other": 0.28},
        ],
    }
    report = evaluate_promotion(cells=cells, n_boot=80, seed=3)
    assert report["status"] == PARTIAL
    assert report["promotable"] is True
    assert report["ccr"] != "NA"


def test_s3_only_does_not_publish():
    cells = {
        "S0": _rows(0.40, 10),
        "S1": _rows(0.20, 4),
        "S2": _rows(0.20, 4),
        "S3": _rows(0.60, 10),
        "retention": [
            {"cluster_id": "h1", "query_id": "h1", "j_base": 0.40, "j_other": 0.40},
            {"cluster_id": "h2", "query_id": "h2", "j_base": 0.20, "j_other": 0.25},
        ],
    }
    report = evaluate_promotion(cells=cells, n_boot=40, seed=3)
    assert report["status"] == REJECT
    assert report["promotable"] is False
    assert report["rollback_to_parent"] is True


def test_full_retirement_requires_s2_and_lower_cost():
    cells = {
        "S0": _rows(0.30, 10),
        "S1": _rows(0.10, 8),
        "S2": _rows(0.34, 6),
        "S3": _rows(0.36, 9),
        "retention": [
            {"cluster_id": "h1", "query_id": "h1", "j_base": 0.20, "j_other": 0.22},
            {"cluster_id": "h2", "query_id": "h2", "j_base": 0.10, "j_other": 0.12},
        ],
    }
    report = evaluate_promotion(cells=cells, n_boot=80, seed=4)
    assert report["status"] == FULL
