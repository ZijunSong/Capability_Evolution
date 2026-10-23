"""Bank loading: manifest validation, dependency graph, deterministic ordering.

Loader invariants (task book §6.2):
- file paths must not escape the bank root
- dependency graph must be acyclic
- call names must not conflict
- hook order must be deterministic
- candidate-invalid problems raise before anything executes
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rsi.contracts.artifact import COMPONENT_KINDS, HOOK_NAMES, BankManifest, ComponentManifest
from rsi.contracts.common import digest_of_file, digest_of_text
from rsi.contracts.errors import CandidateInvalid


@dataclass(frozen=True)
class LoadedComponent:
    manifest: ComponentManifest
    instance: Any
    source_hash: str
    module_name: str


@dataclass(frozen=True)
class LoadedBank:
    manifest: BankManifest
    components: tuple[LoadedComponent, ...]
    bank_digest: str
    root: Path

    @property
    def is_empty(self) -> bool:
        return self.manifest.is_empty()


def _validate_path_safe(root: Path, rel: str) -> Path:
    p = (root / rel).resolve()
    try:
        p.relative_to(root.resolve())
    except ValueError as e:
        raise CandidateInvalid(f"bank file path escapes bank root: {rel!r}") from e
    return p


def validate_manifest(manifest: BankManifest, root: Path) -> None:
    ids = [c.id for c in manifest.components]
    if len(ids) != len(set(ids)):
        raise CandidateInvalid(f"duplicate component ids: {ids}")
    for c in manifest.components:
        if c.kind not in COMPONENT_KINDS:
            raise CandidateInvalid(f"component {c.id}: unknown kind {c.kind!r}")
        entry_mod, _, entry_cls = c.entrypoint.partition(":")
        if not entry_mod or not entry_cls:
            raise CandidateInvalid(f"component {c.id}: entrypoint must be 'module:Class'")
        if not c.files:
            raise CandidateInvalid(f"component {c.id}: files must be non-empty")
        mod_file = entry_mod.replace(".", "/") + ".py"
        if mod_file not in [f.replace("\\", "/") for f in c.files]:
            raise CandidateInvalid(
                f"component {c.id}: entrypoint module {entry_mod!r} not in files list"
            )
        for f in c.files:
            _validate_path_safe(root, f)
        for dep in c.depends_on:
            if dep not in ids:
                raise CandidateInvalid(f"component {c.id}: missing dependency {dep!r}")
        for conf in c.conflicts_with:
            if conf in ids:
                raise CandidateInvalid(f"component {c.id}: conflicts with loaded {conf!r}")
        if c.hook_order != int(c.hook_order):
            raise CandidateInvalid(f"component {c.id}: hook_order must be int")
    # acyclic dependency check
    by_id = {c.id: c for c in manifest.components}
    visiting: set[str] = set()
    done: set[str] = set()

    def visit(cid: str, stack: tuple[str, ...]) -> None:
        if cid in done:
            return
        if cid in visiting:
            raise CandidateInvalid(f"dependency cycle: {' -> '.join(stack + (cid,))}")
        visiting.add(cid)
        for dep in by_id[cid].depends_on:
            visit(dep, stack + (cid,))
        visiting.discard(cid)
        done.add(cid)

    for cid in by_id:
        visit(cid, ())
    # deterministic order
    orders = sorted((c.hook_order, c.id) for c in manifest.components)
    if orders != sorted(orders, key=lambda t: (t[0], t[1])):
        raise CandidateInvalid("hook order is not deterministic")


def _ast_audit(source: str, filename: str) -> list[str]:
    """Auxiliary audit only — NOT a security sandbox.  Real isolation is the
    candidate worker process/container (see rsi.runtime.sandbox)."""
    problems: list[str] = []
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as e:
        return [f"syntax error: {e}"]
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.attr, str):
            if node.attr in {"docker", "system"} and isinstance(node.value, ast.Name):
                if node.value.id in {"os", "socket"}:
                    problems.append(f"{filename}: suspicious {node.value.id}.{node.attr}")
    return problems


def load_bank(root: str | Path) -> LoadedBank:
    """Load and validate a bank directory (manifest.json + component sources)."""
    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise CandidateInvalid(f"bank manifest not found: {manifest_path}")
    import json

    manifest = BankManifest.from_json(json.loads(manifest_path.read_text("utf-8")))
    validate_manifest(manifest, root)

    loaded: list[LoadedComponent] = []
    for comp in sorted(manifest.components, key=lambda c: (c.hook_order, c.id)):
        module_file = comp.entrypoint.partition(":")[0].replace(".", "/") + ".py"
        path = _validate_path_safe(root, module_file)
        if not path.is_file():
            raise CandidateInvalid(f"component {comp.id}: missing module file {module_file}")
        source = path.read_text("utf-8")
        problems = _ast_audit(source, module_file)
        if problems:
            raise CandidateInvalid(f"component {comp.id}: AST audit: {problems}")
        # Isolated module namespace per bank to avoid cross-bank collisions.
        module_name = f"rsi_bank_{digest_of_text(str(root))[:8]}_{module_file[:-3].replace('/', '_')}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise CandidateInvalid(f"component {comp.id}: cannot import {module_file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as e:  # noqa: BLE001 - candidate code may raise anything
            raise CandidateInvalid(f"component {comp.id}: import failed: {e!r}") from e
        cls = getattr(module, comp.entrypoint.partition(":")[2], None)
        if cls is None:
            raise CandidateInvalid(f"component {comp.id}: entrypoint class not found")
        try:
            instance = cls()
        except Exception as e:  # noqa: BLE001
            raise CandidateInvalid(f"component {comp.id}: constructor failed: {e!r}") from e
        for hook in HOOK_NAMES:
            if not callable(getattr(instance, hook, None)):
                raise CandidateInvalid(f"component {comp.id}: missing hook {hook!r}")
        source_hash = digest_of_text(source)
        loaded.append(
            LoadedComponent(
                manifest=comp,
                instance=instance,
                source_hash=source_hash,
                module_name=module_name,
            )
        )
    bank_digest = manifest.digest()
    # Include source hashes so code-only edits change the bank digest.
    if loaded:
        from rsi.contracts.common import digest_of

        bank_digest = digest_of(
            {"manifest": manifest.digest(), "sources": {c.manifest.id: c.source_hash for c in loaded}}
        )
    return LoadedBank(manifest=manifest, components=tuple(loaded), bank_digest=bank_digest, root=root)


def load_empty_bank() -> LoadedBank:
    """The canonical empty bank used by ``--harness minimal``.

    Not the same code path as ``banks/empty`` on disk: this one must behave
    identically to an on-disk empty bank (test_empty_bank_identity).
    """
    root = Path(__file__).resolve().parents[3] / "banks" / "empty"
    if (root / "manifest.json").is_file():
        bank = load_bank(root)
        if not bank.is_empty:
            raise CandidateInvalid("banks/empty must contain no components/hooks/tools/skills")
        return bank
    # Fallback: pure in-memory empty bank (same digest as on-disk empty).
    manifest = BankManifest(bank_id="empty")
    return LoadedBank(manifest=manifest, components=(), bank_digest=manifest.digest(), root=root)


def load_bank_ref(ref: str) -> LoadedBank:
    """Resolve a bank reference.

    ``minimal`` -> the empty bank; a directory -> load_bank(dir); a champion
    JSON file -> load the bank directory it points at.
    """
    import json as _json

    if ref in ("minimal", "empty", "none"):
        return load_empty_bank()
    path = Path(ref)
    if path.is_dir():
        return load_bank(path)
    if path.is_file():
        payload = _json.loads(path.read_text("utf-8"))
        bank_dir = payload.get("bank_dir") or payload.get("bank_path")
        if not bank_dir:
            raise CandidateInvalid(f"bank ref file {ref} has no bank_dir field")
        return load_bank(Path(str(bank_dir)))
    raise CandidateInvalid(f"cannot resolve bank ref {ref!r}")
