"""Candidate materializer.

Writes the model's *actual* patch into an isolated candidate directory, applies
operation semantics (add/edit/compose/prune/replace), verifies the parent
digest and computes the resulting code/manifest digests.  Fixed template
substitution (e.g. always copying a stdlib/query_repair file) is forbidden:
``files[].content`` must land on disk byte-for-byte
(``test_real_code_materialization``).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from rsi.contracts.artifact import BankManifest, CandidatePatch, ComponentManifest
from rsi.contracts.common import digest_of, digest_of_text
from rsi.contracts.errors import CandidateInvalid
from rsi.harness.loader import LoadedBank, load_bank


@dataclass(frozen=True)
class MaterializedCandidate:
    candidate_id: str
    bank_dir: Path
    bank_digest: str
    parent_bank_digest: str
    operation: str
    hypothesis: str
    manifest: BankManifest


def _read_manifest(bank_root: Path) -> BankManifest:
    path = bank_root / "manifest.json"
    if not path.is_file():
        return BankManifest(bank_id="parent")
    return BankManifest.from_json(json.loads(path.read_text("utf-8")))


def _write_manifest(bank_root: Path, manifest: BankManifest) -> None:
    (bank_root / "manifest.json").write_text(
        json.dumps(manifest.to_json(), indent=2, sort_keys=True, ensure_ascii=False), "utf-8"
    )


def materialize_candidate(
    patch: CandidatePatch,
    *,
    parent_bank_root: Path,
    parent_bank_digest: str,
    out_root: Path,
    model_output_ref: str | None = None,
) -> MaterializedCandidate:
    if patch.parent_bank_digest != parent_bank_digest:
        raise CandidateInvalid(
            f"parent digest mismatch: patch={patch.parent_bank_digest} parent={parent_bank_digest}"
        )
    manifest = _read_manifest(parent_bank_root)
    files_now: dict[str, str] = {}
    for comp in manifest.components:
        for rel in comp.files:
            src = parent_bank_root / rel
            if src.is_file():
                files_now[rel.replace("\\", "/")] = src.read_text("utf-8")

    # --- operation semantics -----------------------------------------
    if patch.operation in ("add", "edit", "compose", "replace"):
        for f in patch.files:
            rel = f.path.replace("\\", "/")
            if rel.startswith("/") or ".." in rel.split("/"):
                raise CandidateInvalid(f"candidate file path escapes bank: {f.path}")
            if patch.operation == "add" and rel in files_now:
                raise CandidateInvalid(f"add operation but file exists: {rel}")
            if patch.operation in ("edit", "compose") and rel not in files_now and not rel.startswith("components/"):
                raise CandidateInvalid(f"edit/compose targets missing file: {rel}")
            files_now[rel] = f.content
    elif patch.operation == "prune":
        for rel in patch.delete_paths:
            files_now.pop(rel.replace("\\", "/"), None)
    else:
        raise CandidateInvalid(f"unknown operation {patch.operation}")

    # manifest updates
    new_components: list[ComponentManifest] = list(manifest.components)
    mp = dict(patch.manifest_patch)
    if mp:
        if "components" in mp:
            # Full replacement list of component manifests (structured update).
            new_components = [ComponentManifest.from_json(c) for c in mp["components"]]
        else:
            # Single-component patch entry (add/edit).
            entry = ComponentManifest.from_json(mp)
            new_components = [c for c in new_components if c.id != entry.id]
            if patch.operation != "prune":
                new_components.append(entry)
    if patch.operation == "prune":
        removed = set(patch.delete_paths)
        # If delete_paths only names files, drop components that reference them.
        new_components = [
            c
            for c in new_components
            if not (set(f.replace("\\", "/") for f in c.files) & removed)
            and c.id not in removed
            and f"components/{c.id}.py" not in removed
        ]

    new_manifest = BankManifest(
        bank_id=f"candidate:{digest_of({'p': parent_bank_digest, 'op': patch.operation, 'h': patch.hypothesis})[:12]}",
        components=tuple(new_components),
        hooks=manifest.hooks,
        tools=manifest.tools,
        skills=manifest.skills,
        parent_digest=parent_bank_digest,
        extra={**dict(manifest.extra), "operation": patch.operation, "hypothesis": patch.hypothesis},
    )

    # Drop files of removed components
    keep_files: set[str] = set()
    for c in new_components:
        keep_files.update(f.replace("\\", "/") for f in c.files)
    files_now = {k: v for k, v in files_now.items() if k in keep_files or k == "manifest.json"}

    # Candidate identity over the resolved source tree + manifest
    candidate_id = digest_of(
        {
            "parent": parent_bank_digest,
            "files": {k: digest_of_text(v) for k, v in sorted(files_now.items())},
            "manifest": new_manifest.digest(),
        }
    )
    bank_dir = out_root / candidate_id / "bank"
    if bank_dir.exists():
        shutil.rmtree(bank_dir)
    bank_dir.mkdir(parents=True)
    for rel, content in files_now.items():
        target = bank_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        # Byte-for-byte: the model's code is the executed code.
        target.write_text(content, "utf-8")
    _write_manifest(bank_dir, new_manifest)

    # Verify by loading (also computes the executable bank digest).
    loaded: LoadedBank = load_bank(bank_dir)
    return MaterializedCandidate(
        candidate_id=candidate_id,
        bank_dir=bank_dir,
        bank_digest=loaded.bank_digest,
        parent_bank_digest=parent_bank_digest,
        operation=patch.operation,
        hypothesis=patch.hypothesis,
        manifest=new_manifest,
    )
