"""Manifest-authoritative task resolution (audit doc §10.5).

The frozen split manifest must — not a fresh directory enumeration — decide
the evaluation task set.  These tests pin:

1. authoritative mode resolves exactly the manifest's ids, in manifest order;
2. a role/round missing from the manifest is a hard ProtocolViolation (the
   split cannot silently grow);
3. non-authoritative configs keep the legacy provider enumeration path;
4. placeholder-instruction materialization is blocked, not substituted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsi.contracts.errors import ProtocolViolation
from rsi.evaluation.task_resolver import TaskResolver, install_materialize_task_id


class _FakeConfig:
    def __init__(self, suite: dict, tmp_root: Path, manifest_file: str | None):
        self._suite = suite
        self._root = tmp_root
        self._manifest = manifest_file

    def section(self, name: str) -> dict:
        assert name == "suite"
        return dict(self._suite)

    def resolve_path(self, p: str | None):
        if p is None:
            return None
        path = Path(p)
        return path if path.is_absolute() else (self._root / path)


class _FakeProvider:
    """Provider exposing only the internal _materialize (legacy shape)."""

    def __init__(self):
        self.materialized: list[str] = []

    def _materialize(self, tid: str) -> object:
        self.materialized.append(tid)

        class _T:
            task_id = f"toy:{tid}"
            instruction = f"do {tid}"

        return _T()


class _BrokenProvider(_FakeProvider):
    def _materialize(self, tid: str) -> object:
        raise RuntimeError(f"task data missing for {tid}")


class _PlaceholderProvider(_FakeProvider):
    def _materialize(self, tid: str) -> object:
        task = super()._materialize(tid)
        task.instruction = f"(appworld task {tid})"  # the forbidden placeholder
        return task


def _write_manifest(tmp_path: Path, *, roles: dict) -> str:
    payload = {
        "schema_version": "rsi_split_v1",
        "campaign_id": "test",
        "manifests": {
            "toy": {
                "manifest_id": "test:toy",
                "domain": "toy",
                "assignments": [
                    {"role": role, "task_ids": ids, "round_index": None}
                    for role, ids in roles.items()
                ],
                "groups": {},
            }
        },
    }
    # compute the digest the loader checks: reuse SplitManifest.digest()
    from rsi.contracts.task import SplitAssignment, SplitManifest

    manifest = SplitManifest(
        manifest_id="test:toy",
        domain="toy",
        assignments=tuple(
            SplitAssignment(role=r, task_ids=tuple(ids), round_index=None)
            for r, ids in roles.items()
        ),
        groups={},
    )
    payload["manifests"]["toy"]["digest"] = manifest.digest()
    out = tmp_path / "splits" / "test.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload), "utf-8")
    return str(out)


def test_authoritative_manifest_resolves_exact_ids(tmp_path):
    mp = _write_manifest(tmp_path, roles={"harness_select": ["t2", "t1"]})
    cfg = _FakeConfig(
        {"manifest_is_authoritative": True, "split_manifest": mp}, tmp_path, mp
    )
    provider = _FakeProvider()
    tasks = TaskResolver(cfg).resolve(
        install_materialize_task_id(provider), domain="toy", role="harness_select"
    )
    assert [t.task_id for t in tasks] == ["toy:t2", "toy:t1"]
    assert provider.materialized == ["t2", "t1"]


def test_missing_role_is_protocol_violation(tmp_path):
    mp = _write_manifest(tmp_path, roles={"harness_select": ["t1"]})
    cfg = _FakeConfig(
        {"manifest_is_authoritative": True, "split_manifest": mp}, tmp_path, mp
    )
    with pytest.raises(ProtocolViolation, match="no assignment for role"):
        TaskResolver(cfg).resolve(
            install_materialize_task_id(_FakeProvider()), domain="toy", role="final_test"
        )


def test_non_authoritative_keeps_provider_enumeration(tmp_path):
    mp = _write_manifest(tmp_path, roles={"harness_select": ["t1"]})
    cfg = _FakeConfig(
        {"split_manifest": mp}, tmp_path, mp  # manifest_is_authoritative absent
    )
    resolver = TaskResolver(cfg)
    assert resolver.authoritative is False


def test_placeholder_instruction_blocked(tmp_path):
    mp = _write_manifest(tmp_path, roles={"harness_select": ["t1"]})
    cfg = _FakeConfig(
        {"manifest_is_authoritative": True, "split_manifest": mp}, tmp_path, mp
    )
    provider = install_materialize_task_id(_PlaceholderProvider())
    with pytest.raises(ProtocolViolation, match="placeholder instruction"):
        TaskResolver(cfg).resolve(provider, domain="toy", role="harness_select")


def test_materialize_failure_propagates(tmp_path):
    mp = _write_manifest(tmp_path, roles={"harness_select": ["t1"]})
    cfg = _FakeConfig(
        {"manifest_is_authoritative": True, "split_manifest": mp}, tmp_path, mp
    )
    provider = install_materialize_task_id(_BrokenProvider())
    with pytest.raises(RuntimeError, match="task data missing"):
        TaskResolver(cfg).resolve(provider, domain="toy", role="harness_select")
