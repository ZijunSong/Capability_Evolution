"""Manifest disjointness tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

MANIFEST_DIR = Path("/data/ppnm/Capability_Evolution/SCOPE/artifacts/datasets/scope_round14/manifests")
AUDIT = Path("/data/ppnm/Capability_Evolution/SCOPE/artifacts/datasets/round2_audit_100q/query_manifest.json")


def _load_qids(path: Path) -> set[str]:
  data = json.loads(path.read_text(encoding="utf-8"))
  return {str(x) for x in data["query_ids"]}


@pytest.fixture(scope="module")
def manifests_created():
  if not (MANIFEST_DIR / "R14_FRESH100.json").exists():
    pytest.skip("R14 manifests not built yet")
  return MANIFEST_DIR


def test_fresh100_disjoint_from_audit(manifests_created):
  fresh = _load_qids(manifests_created / "R14_FRESH100.json")
  audit = _load_qids(AUDIT)
  assert not (fresh & audit)


def test_splits_pairwise_disjoint(manifests_created):
  names = ["R14_FRESH100", "R14_SMOKE20", "R14_TRAIN_POOL"]
  sets = [_load_qids(manifests_created / f"{n}.json") for n in names]
  for i, a in enumerate(names):
    for j, b in enumerate(names):
      if i >= j:
        continue
      assert not (sets[i] & sets[j]), f"{a} overlaps {b}"


def test_fresh100_has_100_queries(manifests_created):
  fresh = _load_qids(manifests_created / "R14_FRESH100.json")
  assert len(fresh) == 100
