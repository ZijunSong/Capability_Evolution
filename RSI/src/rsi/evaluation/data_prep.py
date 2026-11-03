"""Dataset preparation and validation.

Materializes frozen split manifests per domain with group-disjoint roles
(``test_split_group_disjointness``).  Task ids are audit references only; they
may never appear as conditional branches or answer banks in evolved code (the
candidate validator rejects id-list constants — see evolution.validaton).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rsi.config import CampaignConfig
from rsi.contracts.errors import NotImplementedCapability, ProtocolViolation
from rsi.contracts.task import SplitAssignment, SplitManifest
from rsi.environments.registry import global_registry

ROLE_LIST = (
    "work_train",
    "harness_select",
    "weight_dev",
    "confirm_h",
    "confirm_w",
    "final_test",
    "transfer_test",
)


def _freeze_sources() -> dict[str, Any]:
    """Record content digests of the frozen dataset artifacts (task book §11.1:
    freeze exact source/version at preprocessing)."""
    from rsi.contracts.common import digest_of_file

    sources: dict[str, Any] = {}
    scan = {
        "spreadsheetbench": [
            "/mnt/songzijun/datasets/spreadsheetbench/spreadsheetbench_912_v0.1.tar.gz",
            "/mnt/songzijun/datasets/spreadsheetbench/all_data_912_v0.1/dataset.json",
        ],
        "swebench": [
            "/mnt/songzijun/datasets/swebench/swebench_train.parquet",
            "/mnt/songzijun/datasets/swebench/swebench_dev.parquet",
            "/mnt/songzijun/datasets/swebench/verified_test.parquet",
            "/mnt/songzijun/datasets/swebench/lite_test.parquet",
        ],
        "appworld": [
            "/mnt/songzijun/datasets/appworld/data",
        ],
        "webshop": [
            "/mnt/songzijun/datasets/webshop/frozen_manifest.json",
            "/mnt/songzijun/datasets/webshop/frozen_goals.json",
            "/mnt/songzijun/datasets/webshop/data",
        ],
        "alfworld": [
            "/mnt/songzijun/datasets/alfworld/frozen_manifest.json",
            "/mnt/songzijun/datasets/alfworld/logic",
        ],
    }
    for domain, paths in scan.items():
        entries = []
        for p in paths:
            path = Path(p)
            if path.is_file():
                entries.append({"path": p, "sha256": digest_of_file(p), "size": path.stat().st_size})
            elif path.is_dir():
                entries.append({"path": p, "kind": "directory", "present": True})
            else:
                entries.append({"path": p, "present": False})
        sources[domain] = entries
    return sources


def prepare_data(cfg: CampaignConfig) -> dict[str, Any]:
    suite = cfg.section("suite")
    domains = list(suite.get("domains") or [])
    split_path = cfg.resolve_path(str(suite.get("split_manifest", "data/splits/split.json")))
    assert split_path is not None
    split_path.parent.mkdir(parents=True, exist_ok=True)
    registry = global_registry()
    out: dict[str, Any] = {"split_manifest": str(split_path), "domains": {}}
    manifests: dict[str, Any] = {}
    for domain in domains:
        bundle = registry.get(domain)
        ok, missing = registry.check_deps(domain)
        if not ok:
            out["domains"][domain] = {"skipped": f"missing deps: {list(missing)}"}
            continue
        try:
            provider = bundle.task_provider_factory()
            assignments = []
            for role in ROLE_LIST:
                if role in ("confirm_h", "confirm_w"):
                    # one confirm panel per round for max_rounds rounds
                    for k in range(cfg.max_rounds):
                        tasks = provider.public_tasks(role, round_index=k)
                        if tasks:
                            assignments.append(
                                SplitAssignment(role=role, task_ids=tuple(t.task_id for t in tasks), round_index=k)
                            )
                else:
                    tasks = provider.public_tasks(role)
                    if tasks:
                        assignments.append(SplitAssignment(role=role, task_ids=tuple(t.task_id for t in tasks)))
            manifest = SplitManifest(
                manifest_id=f"{cfg.campaign_id}:{domain}",
                domain=domain,
                assignments=tuple(assignments),
                groups=provider.groups(),
            )
            manifest.check_group_disjoint()
        except NotImplementedCapability as e:
            out["domains"][domain] = {"skipped": str(e)}
            continue
        manifests[domain] = manifest
        out["domains"][domain] = {
            "manifest_digest": manifest.digest(),
            "n_tasks": sum(len(a.task_ids) for a in manifest.assignments),
        }
    payload = {
        "schema_version": "rsi_split_v1",
        "campaign_id": cfg.campaign_id,
        "config_digest": cfg.digest(),
        "frozen_sources": _freeze_sources(),
        "manifests": {
            d: {
                "manifest_id": m.manifest_id,
                "domain": m.domain,
                "assignments": [
                    {"role": a.role, "task_ids": list(a.task_ids), "round_index": a.round_index}
                    for a in m.assignments
                ],
                "groups": {k: list(v) for k, v in sorted(m.groups.items())},
                "digest": m.digest(),
            }
            for d, m in sorted(manifests.items())
        },
    }
    split_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), "utf-8")
    out["frozen_sources"] = payload["frozen_sources"]
    out["ok"] = True
    return out


def load_split_manifests(cfg: CampaignConfig) -> dict[str, SplitManifest]:
    suite = cfg.section("suite")
    split_path = cfg.resolve_path(str(suite.get("split_manifest", "data/splits/split.json")))
    assert split_path is not None
    if not split_path.is_file():
        raise ProtocolViolation(
            f"split manifest not found: {split_path}; run `rsi.cli data prepare` first"
        )
    payload = json.loads(split_path.read_text("utf-8"))
    out: dict[str, SplitManifest] = {}
    for domain, m in dict(payload.get("manifests") or {}).items():
        manifest = SplitManifest(
            manifest_id=str(m["manifest_id"]),
            domain=str(m["domain"]),
            assignments=tuple(
                SplitAssignment(
                    role=str(a["role"]),
                    task_ids=tuple(a["task_ids"]),
                    round_index=a.get("round_index"),
                )
                for a in m["assignments"]
            ),
            groups={k: tuple(v) for k, v in dict(m["groups"]).items()},
        )
        if manifest.digest() != m.get("digest"):
            raise ProtocolViolation(f"split manifest digest mismatch for {domain}")
        out[str(domain)] = manifest
    return out


def validate_data(cfg: CampaignConfig) -> dict[str, Any]:
    try:
        manifests = load_split_manifests(cfg)
    except ProtocolViolation as e:
        return {"ok": False, "error": str(e)}
    report: dict[str, Any] = {"ok": True, "domains": {}}
    for domain, manifest in sorted(manifests.items()):
        try:
            manifest.check_group_disjoint()
            problems: list[str] = []
            # A task id may repeat across rounds of the SAME role (pilot_reused_dev),
            # but never across different roles.
            seen: dict[str, str] = {}
            for a in manifest.assignments:
                for tid in a.task_ids:
                    if tid in seen and seen[tid] != a.role:
                        problems.append(f"task {tid} in both {seen[tid]} and {a.role}")
                    seen[tid] = a.role
            if problems:
                report["ok"] = False
            report["domains"][domain] = {
                "ok": not problems,
                "digest": manifest.digest(),
                "n_groups": len(manifest.groups),
                "problems": problems,
            }
        except ValueError as e:
            report["ok"] = False
            report["domains"][domain] = {"ok": False, "error": str(e)}
    return report
