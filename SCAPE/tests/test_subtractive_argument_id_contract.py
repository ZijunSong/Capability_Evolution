from __future__ import annotations

from scripts.run_h100_3_subtractive_audit import build_synthetic_oracle, valid_argument_audit


def test_subtractive_argument_id_contract_accepts_only_current_doc_ids():
    row = build_synthetic_oracle()
    audit = valid_argument_audit(
        row,
        {"name": "curate", "arguments": {"add_ids": ["d_gold", "d_missing"], "remove_ids": ["d_noise", "d_other"]}},
    )
    assert audit["valid_add_ids"] == ["d_gold"]
    assert audit["valid_remove_ids"] == ["d_noise"]
    assert audit["invalid_add_ids"] == ["d_missing"]
    assert audit["invalid_remove_ids"] == ["d_other"]
