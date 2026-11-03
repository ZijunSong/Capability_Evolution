"""Candidate dedup fingerprints (audit doc §8.2)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsi.evolution.dedup import DedupCache, ast_digest, bank_content_digest, dedup_verdict


def test_exact_content_dup_detected():
    cache = DedupCache()
    src_a = "class C:\n    def before_model(self, v, a):\n        return None\n"
    d = bank_content_digest({"components/c.py": src_a})
    verdict, _ = dedup_verdict(bank_digest=d, ast_digests={"c": ast_digest(src_a)}, cache=cache)
    assert verdict == "new"
    cache.exact[d] = {"macro_score": 0.4}
    verdict, prior = dedup_verdict(bank_digest=d, ast_digests={"c": ast_digest(src_a)}, cache=cache)
    assert verdict == "exact_dup" and prior["macro_score"] == 0.4


def test_ast_near_dup_ignores_comments():
    cache = DedupCache()
    a = "class C:\n    def before_model(self, v, a):\n        # comment\n        return None\n"
    b = 'class C:\n    """docstring"""\n    def before_model(self, v, a):\n        return None  # trailing\n'
    da, db = ast_digest(a), ast_digest(b)
    assert da == db, "comment/docstring-only differences must be AST-equal"


def test_near_dup_returns_prior_bank():
    cache = DedupCache()
    src = "class C:\n    def before_model(self, v, a):\n        return 1\n"
    d1 = bank_content_digest({"components/c.py": src + "# v1\n"})
    cache.exact[d1] = {"macro_score": 0.5}
    cache.record(bank_digest=d1, ast_digests={"c": ast_digest(src)}, result={"macro_score": 0.5})
    d2 = bank_content_digest({"components/c.py": src + "# v2 reworded\n"})
    verdict, prior = dedup_verdict(
        bank_digest=d2, ast_digests={"c": ast_digest(src)}, cache=cache
    )
    assert verdict.startswith("near_dup:") and prior is not None
