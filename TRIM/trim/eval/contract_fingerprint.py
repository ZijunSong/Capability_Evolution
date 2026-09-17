"""Frozen run-contract fingerprint for Harness-G four-cell eval."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping

TRIM_ROOT = Path(__file__).resolve().parents[2]

KEY_RELPATHS: tuple[str, ...] = (
    "trim/eval/harness_g_env.py",
    "trim/eval/harness_g_runtime.py",
    "trim/eval/harness_g_graph.py",
    "trim/eval/harmony_runtime.py",
    "trim/eval/harness1_metrics.py",
    "trim/eval/sr_opd_four_cell_eval.py",
    "trim/eval/eval_shard_worker.py",
    "trim/adapters/harness_g_components.py",
    "trim/training/parse_rollout_action.py",
    "trim/training/batched_env_rollout.py",
    "trim/training/four_cell_runtime.py",
    "scripts/run_eval.py",
    "scripts/run_bcplus_test50_harness_g_gpu34567_parallel.sh",
    "scripts/run_bcplus_test50_harness_g_gpu04567_eval.sh",
)


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_identity(root: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {"root": str(root), "commit": None, "dirty": None, "clean": False}
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        payload["commit"] = commit
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        payload["dirty"] = dirty
        payload["clean"] = not bool(dirty.strip())
    except (OSError, subprocess.CalledProcessError):
        payload["error"] = "git_unavailable"
    return payload


def collect_contract_fingerprint(
    *,
    model_path: str | None = None,
    tokenizer_name: str | None = None,
    graph_fingerprint: str | None = None,
    graph_scope: str | None = None,
    corpus_path: str | None = None,
    qrels_path: str | None = None,
    sampling: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for rel in KEY_RELPATHS:
        path = TRIM_ROOT / rel
        files[rel] = {
            "path": str(path.resolve()) if path.exists() else str(path),
            "sha256": _sha256_file(path),
            "exists": path.is_file(),
        }
    payload = {
        "trim_root": str(TRIM_ROOT),
        "git": _git_identity(TRIM_ROOT),
        "files": files,
        "model_path": str(model_path or "") or None,
        "tokenizer_name": str(tokenizer_name or "") or None,
        "graph_fingerprint": graph_fingerprint,
        "graph_scope": graph_scope,
        "corpus_path": corpus_path,
        "qrels_path": qrels_path,
        "sampling": dict(sampling or {}),
        "pid": os.getpid(),
        "cwd": os.getcwd(),
    }
    if extra:
        payload["extra"] = dict(extra)
    blob = json.dumps(
        {
            "files": {k: v.get("sha256") for k, v in files.items()},
            "git_commit": (payload["git"] or {}).get("commit"),
            "git_clean": (payload["git"] or {}).get("clean"),
            "graph_fingerprint": graph_fingerprint,
            "graph_scope": graph_scope,
            "sampling": dict(sampling or {}),
            "model_path": payload["model_path"],
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    payload["contract_sha256"] = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    return payload


def fingerprints_compatible(left: Mapping[str, Any] | None, right: Mapping[str, Any] | None) -> tuple[bool, str]:
    if not left or not right:
        return False, "missing contract fingerprint"
    lhash = str(left.get("contract_sha256") or "")
    rhash = str(right.get("contract_sha256") or "")
    if lhash and rhash and lhash == rhash:
        return True, "ok"
    mismatches: list[str] = []
    lfiles = (left.get("files") or {}) if isinstance(left.get("files"), dict) else {}
    rfiles = (right.get("files") or {}) if isinstance(right.get("files"), dict) else {}
    for rel in sorted(set(lfiles) | set(rfiles)):
        lsha = (lfiles.get(rel) or {}).get("sha256") if isinstance(lfiles.get(rel), dict) else None
        rsha = (rfiles.get(rel) or {}).get("sha256") if isinstance(rfiles.get(rel), dict) else None
        if lsha != rsha:
            mismatches.append(rel)
    if (left.get("graph_fingerprint") or None) != (right.get("graph_fingerprint") or None):
        mismatches.append("graph_fingerprint")
    if (left.get("graph_scope") or None) != (right.get("graph_scope") or None):
        mismatches.append("graph_scope")
    ls = dict(left.get("sampling") or {})
    rs = dict(right.get("sampling") or {})
    for key in sorted(set(ls) | set(rs)):
        if ls.get(key) != rs.get(key):
            mismatches.append(f"sampling.{key}")
    if (left.get("model_path") or None) != (right.get("model_path") or None):
        mismatches.append("model_path")
    if not mismatches:
        return False, f"contract_sha256 mismatch {lhash} vs {rhash}"
    return False, "incompatible contract: " + ", ".join(mismatches[:12])


def merge_fingerprints(items: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not items:
        raise RuntimeError("no contract fingerprints to merge")
    first = dict(items[0])
    for other in items[1:]:
        ok, reason = fingerprints_compatible(first, other)
        if not ok:
            raise RuntimeError(reason)
    return first
