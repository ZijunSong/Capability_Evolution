from __future__ import annotations

from scripts.run_h100_3_subtractive_audit import (
    apply_curate_action,
    build_synthetic_oracle,
    curated_evidence_recall,
    curated_ids,
    document_ids,
    snapshot_wm,
)


def test_subtractive_evaluator_nonzero_oracle():
    row = build_synthetic_oracle()
    gold = snapshot_wm(row)["gold_doc_ids"]
    before = curated_ids(row)
    assert curated_evidence_recall(before, gold) == 0.0

    after = apply_curate_action(
        before,
        document_ids(row),
        {"name": "curate", "arguments": {"add_ids": ["d_gold"], "remove_ids": ["d_noise"]}},
    )
    assert curated_evidence_recall(after, gold) == 1.0
