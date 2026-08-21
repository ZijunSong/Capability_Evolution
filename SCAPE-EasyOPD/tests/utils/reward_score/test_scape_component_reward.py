from __future__ import annotations

from verl.utils.reward_score import default_compute_score


def test_scape_component_opd_reward_defaults_to_zero() -> None:
    score = default_compute_score("scape_component_opd/evidence_graph", "response text", "ground truth")
    assert float(score) == 0.0


def test_scape_component_opd_sentence_compress_reward_defaults_to_zero() -> None:
    score = default_compute_score("scape_component_opd/sentence_compress", "response text", "ground truth")
    assert float(score) == 0.0
