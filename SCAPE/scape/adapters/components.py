"""Canonical Harness-1 component taxonomy and env-flag masks.

SCAPE never mutates upstream source for component toggles. Masks are applied
via environment variable overrides around harness entrypoints.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

# Upstream V8D_* flag <-> SCAPE canonical id
COMPONENT_TAXONOMY: dict[str, dict[str, Any]] = {
    "subtractive_curation": {
        "upstream_flag": "V8D_SUBTRACTIVE_CURATION",
        "semantic_or_runtime": "semantic",
        "changes_context": True,
        "changes_state": True,
        "changes_execution": False,
        "default_enabled": True,
        "runtime_anchor": False,
    },
    "importance_tagging": {
        "upstream_flag": "V8D_IMPORTANCE_TAGGING",
        "semantic_or_runtime": "semantic",
        "changes_context": True,
        "changes_state": True,
        "changes_execution": False,
        "default_enabled": True,
        "runtime_anchor": False,
    },
    "auto_populate_first_search": {
        "upstream_flag": "V8D_AUTO_POPULATE_FIRST_SEARCH",
        "semantic_or_runtime": "semantic",
        "changes_context": True,
        "changes_state": True,
        "changes_execution": True,
        "default_enabled": True,
        "runtime_anchor": False,
    },
    "evidence_graph": {
        "upstream_flag": "V8D_EVIDENCE_GRAPH",
        "semantic_or_runtime": "semantic",
        "changes_context": True,
        "changes_state": True,
        "changes_execution": False,
        "default_enabled": True,
        "runtime_anchor": False,
    },
    "sentence_compress": {
        "upstream_flag": "V8D_SENTENCE_COMPRESS",
        "semantic_or_runtime": "semantic",
        "changes_context": True,
        "changes_state": False,
        "changes_execution": False,
        "default_enabled": True,
        "runtime_anchor": False,
    },
    "chunk_neighbors": {
        "upstream_flag": "V8D_CHUNK_NEIGHBORS",
        "semantic_or_runtime": "runtime",
        "changes_context": True,
        "changes_state": False,
        "changes_execution": True,
        "default_enabled": False,
        "runtime_anchor": False,
    },
    "content_dedup": {
        "upstream_flag": "V8D_CONTENT_DEDUP",
        "semantic_or_runtime": "hybrid",
        "changes_context": True,
        "changes_state": True,
        "changes_execution": True,
        "default_enabled": True,
        "runtime_anchor": True,  # cheap deterministic runtime check by default
    },
    "verify_tool": {
        "upstream_flag": "V8D_VERIFY_TOOL",
        "semantic_or_runtime": "semantic",
        "changes_context": True,
        "changes_state": True,
        "changes_execution": True,
        "default_enabled": True,
        "runtime_anchor": False,
    },
    "token_budget_marker": {
        "upstream_flag": "V8D_TOKEN_BUDGET_MARKER",
        "semantic_or_runtime": "runtime",
        "changes_context": True,
        "changes_state": False,
        "changes_execution": False,
        "default_enabled": True,
        "runtime_anchor": True,  # exact accounting / hard budget marker
    },
    "adaptive_rerank_instruction": {
        "upstream_flag": "V8D_ADAPTIVE_RERANK_INSTRUCTION",
        "semantic_or_runtime": "semantic",
        "changes_context": True,
        "changes_state": False,
        "changes_execution": False,
        "default_enabled": False,
        "runtime_anchor": False,
    },
}

# Non-component runtime anchors that must never be selected as full internalization targets.
RUNTIME_ANCHORS: frozenset[str] = frozenset(
    {
        "retrieval_executor",
        "exact_accounting",
        "persistent_store",
        "hard_budget_enforcement",
        "cheap_deterministic_runtime_checks",
        "content_dedup",
        "token_budget_marker",
    }
)


@dataclass(frozen=True)
class ComponentSpec:
    component_id: str
    upstream_flag: str
    enabled: bool
    semantic_or_runtime: str
    changes_context: bool
    changes_state: bool
    changes_execution: bool
    runtime_anchor: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def all_component_ids() -> list[str]:
    return list(COMPONENT_TAXONOMY.keys())


def flag_for(component_id: str) -> str:
    if component_id not in COMPONENT_TAXONOMY:
        raise KeyError(f"unknown component_id: {component_id}")
    return str(COMPONENT_TAXONOMY[component_id]["upstream_flag"])


def full_mask() -> dict[str, bool]:
    """Official-ish full operating point used by SCAPE (canonical ids)."""
    return {
        cid: bool(meta["default_enabled"]) for cid, meta in COMPONENT_TAXONOMY.items()
    }


def minus_mask(component_id: str, base: Mapping[str, bool] | None = None) -> dict[str, bool]:
    if component_id not in COMPONENT_TAXONOMY:
        raise KeyError(f"unknown component_id: {component_id}")
    mask = dict(base or full_mask())
    mask[component_id] = False
    return mask


def coalition_minus_mask(
    component_ids: Iterable[str],
    base: Mapping[str, bool] | None = None,
) -> dict[str, bool]:
    """H_-S mask: disable every component in coalition S."""
    mask = dict(base or full_mask())
    for component_id in component_ids:
        if component_id not in COMPONENT_TAXONOMY:
            raise KeyError(f"unknown component_id: {component_id}")
        mask[component_id] = False
    return mask


def mask_to_env(mask: Mapping[str, bool]) -> dict[str, str]:
    """Convert canonical mask -> upstream V8D_* env values ('0'/'1')."""
    env: dict[str, str] = {}
    for cid, enabled in mask.items():
        if cid not in COMPONENT_TAXONOMY:
            raise KeyError(f"unknown component_id in mask: {cid}")
        env[flag_for(cid)] = "1" if enabled else "0"
    # Ensure every known flag is present
    for cid in COMPONENT_TAXONOMY:
        flag = flag_for(cid)
        env.setdefault(flag, "1" if full_mask()[cid] else "0")
    return env


def env_to_mask(env: Mapping[str, str]) -> dict[str, bool]:
    mask: dict[str, bool] = {}
    for cid, meta in COMPONENT_TAXONOMY.items():
        flag = meta["upstream_flag"]
        raw = env.get(flag)
        if raw is None:
            mask[cid] = bool(meta["default_enabled"])
        else:
            mask[cid] = str(raw) == "1"
    return mask


def component_specs(mask: Mapping[str, bool] | None = None) -> list[ComponentSpec]:
    m = mask or full_mask()
    specs: list[ComponentSpec] = []
    for cid, meta in COMPONENT_TAXONOMY.items():
        specs.append(
            ComponentSpec(
                component_id=cid,
                upstream_flag=str(meta["upstream_flag"]),
                enabled=bool(m.get(cid, meta["default_enabled"])),
                semantic_or_runtime=str(meta["semantic_or_runtime"]),
                changes_context=bool(meta["changes_context"]),
                changes_state=bool(meta["changes_state"]),
                changes_execution=bool(meta["changes_execution"]),
                runtime_anchor=bool(meta["runtime_anchor"]),
            )
        )
    return specs


def assert_mask_diff_only(
    before: Mapping[str, bool],
    after: Mapping[str, bool],
    *,
    expected_changed: Iterable[str],
) -> None:
    """Raise if any flag outside expected_changed differs."""
    expected = set(expected_changed)
    for cid in set(before) | set(after):
        b = before.get(cid)
        a = after.get(cid)
        if b != a and cid not in expected:
            raise AssertionError(
                f"unexpected mask change for {cid}: {b} -> {a}; "
                f"expected only {sorted(expected)}"
            )
        if b == a and cid in expected:
            raise AssertionError(f"expected {cid} to change, but both={b}")


def clone_mask(mask: Mapping[str, bool]) -> dict[str, bool]:
    return deepcopy(dict(mask))
