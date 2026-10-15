"""Load the native-action recipe that travels with a bank component."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from trim.harness_bank.loader import BankLoadError
from trim.harness_bank.types import BankSpec, ComponentSpec

_ROUTES = {"explicit_map", "skip_to_anchor", "native_recipe", "native_anchor", "persistent_uninternalized"}


def load_recipe(bank: BankSpec, spec: ComponentSpec) -> dict[str, Any]:
    if not spec.recipe:
        raise BankLoadError(f"{spec.id} has no recipe; refusing heuristic_generic")
    path = Path(bank.root) / spec.recipe
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BankLoadError(f"cannot read recipe for {spec.id}: {exc}") from exc
    if not isinstance(payload, dict):
        raise BankLoadError(f"recipe for {spec.id} must be an object")
    route = str(payload.get("route") or spec.distill_route)
    if route not in _ROUTES:
        raise BankLoadError(f"{spec.id} recipe route {route!r} is unknown")
    if payload.get("component_id") not in {None, spec.id}:
        raise BankLoadError(f"recipe component_id does not match {spec.id}")
    payload = dict(payload)
    payload["route"] = route
    payload["component_id"] = spec.id
    payload.setdefault("target_tools", list(spec.target_tools))
    payload.setdefault("skip_if_unmapped", True)
    return payload


def load_adaptation(bank: BankSpec) -> dict[str, Any]:
    path = Path(bank.root) / "adaptation.json"
    if not path.is_file():
        return {"components": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BankLoadError("adaptation.json must be an object")
    return payload
