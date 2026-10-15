"""Cluster-level splits. A query id file is not a substitute for this manifest."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from trim.rsi.canonical import sha256_json
from trim.rsi.types import PanelManifest


class SplitError(ValueError):
    """The requested split crosses a cluster or reuses a confirmation panel."""


@dataclass
class QuestionRecord:
    query_id: str
    cluster_id: str
    layer: str = "current"


@dataclass
class SplitManifest:
    records: tuple[QuestionRecord, ...]
    roles: dict[str, tuple[str, ...]]
    dataset_manifest_hash: str
    consumed: dict[str, dict[str, Any]] = field(default_factory=dict)

    def ids(self, role: str) -> tuple[str, ...]:
        return tuple(self.roles.get(role) or ())

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": [record.__dict__ for record in self.records],
            "roles": {key: list(value) for key, value in self.roles.items()},
            "dataset_manifest_hash": self.dataset_manifest_hash,
            "consumed": self.consumed,
            "n_questions": len(self.records),
            "n_clusters": len({record.cluster_id for record in self.records}),
            "n_by_role": {key: len(value) for key, value in self.roles.items()},
        }


def assign_roles(
    records: Sequence[QuestionRecord],
    *,
    roles: Mapping[str, Sequence[str]],
) -> SplitManifest:
    by_id = {}
    clusters: dict[str, str] = {}
    for record in records:
        if record.query_id in by_id:
            raise SplitError(f"duplicate query id {record.query_id}")
        by_id[record.query_id] = record
        previous = clusters.get(record.query_id)
        if previous is not None and previous != record.cluster_id:
            raise SplitError(f"{record.query_id} belongs to two clusters")
        clusters[record.query_id] = record.cluster_id
    role_ids: dict[str, tuple[str, ...]] = {}
    cluster_owner: dict[str, str] = {}
    for role, ids in roles.items():
        seen: set[str] = set()
        kept: list[str] = []
        for query_id in ids:
            if query_id not in by_id:
                raise SplitError(f"unknown query id {query_id} in role {role}")
            if query_id in seen:
                raise SplitError(f"duplicate query id {query_id} in role {role}")
            seen.add(query_id)
            cluster = by_id[query_id].cluster_id
            owner = cluster_owner.get(cluster)
            if owner is not None and owner != role:
                raise SplitError(
                    f"cluster {cluster} is in both {owner} and {role}; confirmation and train must not share a cluster"
                )
            cluster_owner[cluster] = role
            kept.append(query_id)
        role_ids[role] = tuple(kept)
    payload = {
        "records": [record.__dict__ for record in records],
        "roles": {key: list(value) for key, value in role_ids.items()},
    }
    return SplitManifest(
        records=tuple(records),
        roles=role_ids,
        dataset_manifest_hash=sha256_json(payload),
    )


def reserve_round_panels(
    manifest: SplitManifest,
    round_id: str,
    *,
    confirm_h_role: str,
    confirm_w_role: str,
) -> PanelManifest:
    for role in (confirm_h_role, confirm_w_role):
        if role not in manifest.roles:
            raise SplitError(f"missing confirmation role {role}")
        if manifest.consumed.get(role, {}).get("consumed"):
            raise SplitError(f"confirmation panel {role} was already consumed")
    work = manifest.ids("work")
    select = manifest.ids("harness_select")
    weight = manifest.ids("weight_dev")
    panel = PanelManifest(
        round_id=round_id,
        work_ids=work,
        select_ids=select,
        weight_dev_ids=weight,
        confirm_h_ids=manifest.ids(confirm_h_role),
        confirm_w_ids=manifest.ids(confirm_w_role),
        cluster_ids={record.query_id: record.cluster_id for record in manifest.records},
        dataset_manifest_hash=manifest.dataset_manifest_hash,
    )
    return panel


def consume_confirmation(
    manifest: SplitManifest,
    *,
    role: str,
    round_id: str,
    candidate_id: str,
    policy_digest: str,
) -> None:
    prior = manifest.consumed.get(role)
    if prior and prior.get("consumed"):
        same = (
            prior.get("round_id") == round_id
            and prior.get("candidate_id") == candidate_id
            and prior.get("policy_digest") == policy_digest
        )
        if not same:
            raise SplitError(
                f"confirmation panel {role} is already bound to {prior.get('candidate_id')}; "
                "a second candidate cannot reuse it"
            )
        return
    manifest.consumed[role] = {
        "consumed": True,
        "round_id": round_id,
        "candidate_id": candidate_id,
        "policy_digest": policy_digest,
    }


def load_id_file(path: str | Path) -> list[str]:
    file = Path(path)
    if not file.is_file():
        raise SplitError(f"query id file does not exist: {file}")
    ids: list[str] = []
    seen: set[str] = set()
    for line in file.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if not item or item.startswith("#"):
            continue
        if item in seen:
            raise SplitError(f"duplicate query id {item} in {file}")
        seen.add(item)
        ids.append(item)
    if not ids:
        raise SplitError(f"query id file is empty: {file}")
    return ids


def filter_rows_by_ids(
    rows: Sequence[Mapping[str, Any]],
    ids: Sequence[str],
    *,
    role: str,
) -> list[dict[str, Any]]:
    if not ids:
        raise SplitError(f"{role} id list is empty")
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        qid = str(row.get("query_id") or "")
        if not qid:
            raise SplitError(f"{role} row is missing query_id")
        if qid in by_id:
            raise SplitError(f"duplicate query id {qid} in the {role} pool")
        by_id[qid] = dict(row)
    missing = [qid for qid in ids if qid not in by_id]
    if missing:
        raise SplitError(f"unknown query ids for {role}: {missing[:8]}")
    return [by_id[qid] for qid in ids]


def assert_disjoint_from_forbidden(
    ids: Iterable[str],
    forbidden: Iterable[str],
    *,
    role: str,
) -> None:
    overlap = sorted(set(ids) & set(forbidden))
    if overlap:
        raise SplitError(f"{role} overlaps a forbidden panel: {overlap[:8]}")


def summarize_split(manifest: SplitManifest) -> str:
    counts = defaultdict(int)
    for record in manifest.records:
        counts[record.cluster_id] += 1
    return (
        f"questions={len(manifest.records)} clusters={len(counts)} "
        + " ".join(f"{role}={len(ids)}" for role, ids in sorted(manifest.roles.items()))
    )


def write_split(path: str | Path, manifest: SplitManifest) -> None:
    Path(path).write_text(json.dumps(manifest.to_dict(), indent=2) + "\n", encoding="utf-8")
