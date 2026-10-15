"""Load a frozen bank from its own directory. Fail before any model starts."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from trim.harness_bank.types import (
    BANK_SCHEMA,
    COMPONENT_KINDS,
    DISTILL_ROUTES,
    HOOK_PHASES,
    BankSpec,
    ComponentSpec,
)
from trim.rsi.canonical import sha256_bytes, sha256_json, sha256_text

_HASH_SKIP_FIELDS = {"bank_sha256", "content_hash"}
_EDITABLE_SUFFIXES = {".py", ".md", ".json", ".txt"}


class BankLoadError(ValueError):
    """The bank is not safe to mount."""


def load_bank(path: str | Path, *, allow_privileged: bool | None = None) -> BankSpec:
    root = Path(path).resolve()
    if not root.is_dir():
        raise BankLoadError(f"bank root is not a directory: {root}")
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise BankLoadError(f"missing manifest.json under {root}")
    _reject_symlink_escape(root, manifest_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BankLoadError(f"manifest.json is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise BankLoadError("manifest.json must be an object")
    if manifest.get("schema_version") != BANK_SCHEMA:
        raise BankLoadError(
            f"schema_version must be {BANK_SCHEMA}, got {manifest.get('schema_version')!r}"
        )
    components = _parse_components(manifest, root)
    legacy = _parse_legacy_flags(manifest.get("legacy_flags") or {})
    active = [str(x) for x in (manifest.get("active_components") or [])]
    known = {spec.id for spec in components}
    missing = [cid for cid in active if cid not in known]
    if missing:
        raise BankLoadError(f"active_components missing from components: {missing}")
    extra_active = known - set(active)
    if extra_active:
        raise BankLoadError(
            "components not listed in active_components are inactive archives and "
            f"must not sit in the loaded component list: {sorted(extra_active)}"
        )
    order = _hook_order(components, manifest.get("hook_order"))
    caps = dict(manifest.get("caps") or {})
    privileged_ok = bool(caps.get("allow_privileged_tools", False))
    if allow_privileged is not None:
        privileged_ok = bool(allow_privileged)
    for spec in components:
        if spec.kind == "privileged_tool" and not privileged_ok:
            raise BankLoadError(
                f"{spec.id} is privileged_tool; the first bank runtime keeps that kind closed"
            )
        if spec.kind == "persistent_system" and spec.distill_route != "persistent_uninternalized":
            raise BankLoadError(
                f"{spec.id} is persistent_system and cannot claim full internalization"
            )
        _require_support_file(root, spec.guidance, label=f"{spec.id} guidance")
        _require_support_file(root, spec.recipe, label=f"{spec.id} recipe")
        _load_factory(root, spec)
    digest = bank_content_hash(root, manifest)
    return BankSpec(
        schema_version=BANK_SCHEMA,
        bank_id=str(manifest.get("bank_id") or root.name),
        legacy_flags=legacy,
        active_components=tuple(active),
        components=tuple(components),
        hook_order=tuple(order),
        caps=caps,
        base_contract_sha256=str(manifest.get("base_contract_sha256") or ""),
        policy_digest_at_search=str(manifest.get("policy_digest_at_search") or ""),
        root=str(root),
        bank_sha256=digest,
    )


def bank_content_hash(root: Path, manifest: dict[str, Any]) -> str:
    """Hash declarations plus editable sources. Skip self-hash, logs, and caches."""
    body = {k: v for k, v in manifest.items() if k not in _HASH_SKIP_FIELDS}
    files: dict[str, str] = {}
    for rel in _hashed_files(root):
        path = root / rel
        files[rel] = sha256_bytes(path.read_bytes())
    return sha256_json({"manifest": body, "files": files, "adapter": "trim_bank_loader_v1"})


def _hashed_files(root: Path) -> list[str]:
    rels: list[str] = []
    for folder in ("components", "guidance", "recipes"):
        base = root / folder
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            _reject_symlink_escape(root, path)
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            rel = path.relative_to(root).as_posix()
            if path.suffix.lower() not in _EDITABLE_SUFFIXES:
                raise BankLoadError(f"unexpected file in bank editable surface: {rel}")
            rels.append(rel)
    for name in ("adaptation.json", "provenance.json"):
        path = root / name
        if path.exists():
            _reject_symlink_escape(root, path)
            rels.append(name)
    return rels


def _parse_legacy_flags(raw: Any) -> dict[str, bool]:
    if not isinstance(raw, dict):
        raise BankLoadError("legacy_flags must be an object")
    from trim.adapters.components import zero_mask

    known = set(zero_mask("Harness-1"))
    out: dict[str, bool] = {}
    for key, value in raw.items():
        if key not in known:
            raise BankLoadError(f"legacy flag {key!r} is not a Harness-1 static component")
        if not isinstance(value, bool):
            raise BankLoadError(f"legacy flag {key} must be a boolean")
        out[str(key)] = value
    return out


def _parse_components(manifest: dict[str, Any], root: Path) -> list[ComponentSpec]:
    raw_items = manifest.get("components") or []
    if not isinstance(raw_items, list):
        raise BankLoadError("components must be a list")
    specs: list[ComponentSpec] = []
    seen: set[str] = set()
    for item in raw_items:
        if not isinstance(item, dict):
            raise BankLoadError("each component must be an object")
        cid = str(item.get("id") or "")
        if not cid or cid in seen:
            raise BankLoadError(f"invalid or duplicate component id {cid!r}")
        seen.add(cid)
        kind = str(item.get("kind") or "")
        if kind not in COMPONENT_KINDS:
            raise BankLoadError(f"{cid} kind {kind!r} is not supported")
        route = str(item.get("distill_route") or "native_anchor")
        if route not in DISTILL_ROUTES:
            raise BankLoadError(f"{cid} distill_route {route!r} is not supported")
        phase = str(item.get("phase") or "before_model")
        if phase not in HOOK_PHASES:
            raise BankLoadError(f"{cid} phase {phase!r} is not supported")
        factory = str(item.get("factory") or "")
        _check_factory_declaration(root, factory)
        specs.append(
            ComponentSpec(
                id=cid,
                kind=kind,
                factory=factory,
                depends_on=tuple(str(x) for x in (item.get("depends_on") or [])),
                conflicts_with=tuple(str(x) for x in (item.get("conflicts_with") or [])),
                phase=phase,
                priority=int(item.get("priority") or 100),
                state_scope=str(item.get("state_scope") or "episode"),
                reads=tuple(str(x) for x in (item.get("reads") or [])),
                writes=tuple(str(x) for x in (item.get("writes") or [])),
                distill_route=route,
                target_tools=tuple(str(x) for x in (item.get("target_tools") or [])),
                guidance=str(item.get("guidance") or ""),
                recipe=str(item.get("recipe") or ""),
                max_extra_llm_calls_per_episode=int(item.get("max_extra_llm_calls_per_episode") or 0),
                max_extra_tool_calls_per_episode=int(item.get("max_extra_tool_calls_per_episode") or 0),
            )
        )
    _check_dependencies(specs)
    return specs


def _check_dependencies(specs: list[ComponentSpec]) -> None:
    by_id = {spec.id: spec for spec in specs}
    for spec in specs:
        for dep in spec.depends_on:
            if dep not in by_id:
                raise BankLoadError(f"{spec.id} depends_on unknown component {dep}")
        for other in spec.conflicts_with:
            if other not in by_id:
                raise BankLoadError(f"{spec.id} conflicts with unknown component {other}")
            if other != spec.id:
                raise BankLoadError(f"{spec.id} conflicts with {other}")
        writers = [
            other.id
            for other in specs
            if other.id != spec.id and set(other.writes) & set(spec.writes)
        ]
        for writer in writers:
            other = by_id[writer]
            ordered = spec.id in other.depends_on or other.id in spec.depends_on
            if not ordered:
                overlap = sorted(set(other.writes) & set(spec.writes))
                raise BankLoadError(
                    f"{spec.id} and {writer} both write {overlap} without a dependency order"
                )
    _toposort(specs)


def _toposort(specs: list[ComponentSpec]) -> list[str]:
    by_id = {spec.id: spec for spec in specs}
    indeg = {spec.id: 0 for spec in specs}
    children: dict[str, list[str]] = {spec.id: [] for spec in specs}
    for spec in specs:
        for dep in spec.depends_on:
            children[dep].append(spec.id)
            indeg[spec.id] += 1
    ready = sorted([cid for cid, deg in indeg.items() if deg == 0])
    ordered: list[str] = []
    while ready:
        cid = ready.pop(0)
        ordered.append(cid)
        for child in sorted(children[cid]):
            indeg[child] -= 1
            if indeg[child] == 0:
                ready.append(child)
                ready.sort()
    if len(ordered) != len(specs):
        raise BankLoadError("component dependency cycle")
    return ordered


def _hook_order(specs: list[ComponentSpec], declared: Any) -> list[str]:
    natural = _toposort(specs)
    if not declared:
        return natural
    declared_ids = [str(x) for x in declared]
    if set(declared_ids) != {spec.id for spec in specs} or len(declared_ids) != len(specs):
        raise BankLoadError("hook_order must list each active component once")
    position = {cid: i for i, cid in enumerate(declared_ids)}
    for spec in specs:
        for dep in spec.depends_on:
            if position[dep] > position[spec.id]:
                raise BankLoadError(f"hook_order runs {spec.id} before dependency {dep}")
    return declared_ids


def _check_factory_declaration(root: Path, factory: str) -> Path:
    if ":" not in factory or factory.startswith("/") or ".." in factory.split(":"):
        raise BankLoadError(f"factory must be a bank-relative path:function, got {factory!r}")
    rel, _, fn = factory.partition(":")
    if not rel.endswith(".py") or not fn.isidentifier():
        raise BankLoadError(f"factory {factory!r} is not path.py:function")
    path = (root / rel).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise BankLoadError(f"factory path escapes the bank root: {factory}") from exc
    _reject_symlink_escape(root, path)
    if not path.is_file():
        raise BankLoadError(f"factory file missing: {rel}")
    return path


def _require_support_file(root: Path, rel: str, *, label: str) -> None:
    if not rel:
        return
    if rel.startswith("/") or ".." in Path(rel).parts:
        raise BankLoadError(f"{label} escapes the bank root: {rel}")
    path = (root / rel).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise BankLoadError(f"{label} escapes the bank root: {rel}") from exc
    _reject_symlink_escape(root, path)
    if not path.is_file():
        raise BankLoadError(f"{label} is missing: {rel}")


def _reject_symlink_escape(root: Path, path: Path) -> None:
    if path.is_symlink():
        target = path.resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise BankLoadError(f"symlink escapes the bank root: {path}") from exc
    current = path
    while current != root:
        if current.is_symlink():
            target = current.resolve()
            try:
                target.relative_to(root)
            except ValueError as exc:
                raise BankLoadError(f"symlink escapes the bank root: {current}") from exc
        if current.parent == current:
            break
        current = current.parent


def _load_factory(root: Path, spec: ComponentSpec) -> Any:
    path = _check_factory_declaration(root, spec.factory)
    text = path.read_text(encoding="utf-8")
    _scan_source(text, label=spec.factory)
    module_name = "trim_bank_" + sha256_text(str(path))[:16]
    module_spec = importlib.util.spec_from_file_location(module_name, path)
    if module_spec is None or module_spec.loader is None:
        raise BankLoadError(f"cannot import factory {spec.factory}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    try:
        module_spec.loader.exec_module(module)
    except Exception as exc:
        raise BankLoadError(f"factory {spec.factory} failed to import: {exc}") from exc
    fn_name = spec.factory.split(":", 1)[1]
    factory = getattr(module, fn_name, None)
    if not callable(factory):
        raise BankLoadError(f"factory {spec.factory} is not callable")
    return factory


def _scan_source(text: str, *, label: str) -> None:
    lowered = text.lower()
    banned = ("gold_doc_ids", "gold_ids", "answer_key", "gold_answer", "task_answer_table")
    for token in banned:
        if token in lowered:
            raise BankLoadError(f"{label} contains task-specific answer material {token}")


def instantiate_components(bank: BankSpec) -> list[Any]:
    root = Path(bank.root)
    objects = []
    for spec in _ordered_specs(bank):
        factory = _load_factory(root, spec)
        obj = factory(spec)
        obj.component_id = spec.id
        objects.append(obj)
    return objects


def _ordered_specs(bank: BankSpec) -> list[ComponentSpec]:
    by_id = {spec.id: spec for spec in bank.components}
    return [by_id[cid] for cid in bank.hook_order]
