"""A4 verification ablation tests."""

from __future__ import annotations

from experiments.ablations.builders.build_verification_ablation import apply_gates, flags_for_variant


def test_full_gate_rejects_visibility():
    flags = flags_for_variant("a4_full_gate")
    cands = [{"id": "1", "visibility_ok": False, "schema_ok": True, "executable": True, "mutation_ok": True, "verified": True, "route": "ENDORSE", "action": "KEEP_EVIDENCE"}]
    kept, tel = apply_gates(cands, flags, hard_realizer_check=lambda c: True)
    assert tel.visibility_violation == 1
    assert tel.rejected == 1
    assert kept == []


def test_no_visibility_gate_keeps():
    flags = flags_for_variant("a4_no_visibility_gate")
    cands = [{"id": "1", "visibility_ok": False, "schema_ok": True, "executable": True, "mutation_ok": True, "verified": True, "route": "ENDORSE", "action": "KEEP_EVIDENCE"}]
    kept, tel = apply_gates(cands, flags, hard_realizer_check=lambda c: True)
    assert tel.accepted == 1
    assert len(kept) == 1


def test_hard_realizer_never_disabled():
    flags = flags_for_variant("a4_no_verification")
    cands = [{"id": "1", "visibility_ok": False, "schema_ok": False, "executable": False, "mutation_ok": False, "verified": False, "route": "ENDORSE", "action": "DANGEROUS"}]
    kept, tel = apply_gates(cands, flags, hard_realizer_check=lambda c: c.get("action") != "DANGEROUS")
    assert tel.invalid_live_action == 1
    assert kept == []
