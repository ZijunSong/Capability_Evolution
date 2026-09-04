"""Multi-harness registry: Harness-1 (v8d) and Harness-G (graph menu)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from trim.adapters import harness_g_components
from trim.adapters.components import COMPONENT_TAXONOMY as HARNESS1_TAXONOMY
from trim.adapters.components import RUNTIME_ANCHORS as HARNESS1_RUNTIME_ANCHORS

HARNESS_1 = "Harness-1"
HARNESS_G = "Harness-G"
DEFAULT_HARNESS = HARNESS_1
ALLOWED_HARNESSES = (HARNESS_1, HARNESS_G)

HARNESS_ALIASES: dict[str, str] = {
    "harness-1": HARNESS_1,
    "harness1": HARNESS_1,
    "harness_1": HARNESS_1,
    "h1": HARNESS_1,
    "harness-g": HARNESS_G,
    "harnessg": HARNESS_G,
    "harness_g": HARNESS_G,
    "hg": HARNESS_G,
}

HARNESS1_COMPONENT_ALIASES: dict[str, str] = {
    cid: cid for cid in HARNESS1_TAXONOMY
} | {
    "auto": "auto_populate_first_search",
    "auto_populate": "auto_populate_first_search",
    "graph": "evidence_graph",
    "compress": "sentence_compress",
    "sentence": "sentence_compress",
    "neighbors": "chunk_neighbors",
    "chunk": "chunk_neighbors",
    "dedup": "content_dedup",
    "verify": "verify_tool",
    "budget": "token_budget_marker",
    "token_budget": "token_budget_marker",
    "rerank": "adaptive_rerank_instruction",
    "adaptive_rerank": "adaptive_rerank_instruction",
    "subtractive": "subtractive_curation",
    "curation": "subtractive_curation",
    "importance": "importance_tagging",
    "tagging": "importance_tagging",
}


@dataclass(frozen=True)
class HarnessProfile:
    name: str
    taxonomy: dict[str, dict[str, Any]]
    component_aliases: dict[str, str]
    runtime_tools: tuple[str, ...]
    teacher_only_tools: tuple[str, ...]
    runtime_anchors: frozenset[str]
    env_flag_prefix: str
    default_model_candidates: tuple[Path, ...] = ()

    def component_ids(self) -> list[str]:
        return list(self.taxonomy.keys())


HARNESS1_RUNTIME_TOOLS = (
    "fan_out_search",
    "search_corpus",
    "grep_corpus",
    "read_document",
    "review_docs",
    "curate",
    "end_search",
)

PROFILES: dict[str, HarnessProfile] = {
    HARNESS_1: HarnessProfile(
        name=HARNESS_1,
        taxonomy=HARNESS1_TAXONOMY,
        component_aliases=HARNESS1_COMPONENT_ALIASES,
        runtime_tools=HARNESS1_RUNTIME_TOOLS,
        teacher_only_tools=("verify", "importance_tagging"),
        runtime_anchors=HARNESS1_RUNTIME_ANCHORS,
        env_flag_prefix="V8D_",
        default_model_candidates=(
            Path("/mnt/songzijun/models/pat-jj_harness-1-full/harness-1"),
            Path("/data/ppnm/models/pat-jj_harness-1-full/harness-1"),
        ),
    ),
    HARNESS_G: HarnessProfile(
        name=HARNESS_G,
        taxonomy=harness_g_components.COMPONENT_TAXONOMY,
        component_aliases={
            cid: cid for cid in harness_g_components.COMPONENT_TAXONOMY
        }
        | harness_g_components.COMPONENT_ALIASES,
        runtime_tools=harness_g_components.RUNTIME_TOOLS,
        teacher_only_tools=harness_g_components.TEACHER_ONLY_TOOLS,
        runtime_anchors=harness_g_components.RUNTIME_ANCHORS,
        env_flag_prefix="HARNESS_G_",
    ),
}


def _norm(value: str) -> str:
    return str(value or "").strip().lower().replace(" ", "")


def normalize_harness(value: str | None) -> str:
    if value is None or str(value).strip() == "":
        return DEFAULT_HARNESS
    key = _norm(str(value))
    if key in HARNESS_ALIASES:
        return HARNESS_ALIASES[key]
    for name in ALLOWED_HARNESSES:
        if _norm(name) == key:
            return name
    raise KeyError(f"unknown harness: {value!r}; allowed: {list(ALLOWED_HARNESSES)}")


def profile_for(harness: str | None = None) -> HarnessProfile:
    return PROFILES[normalize_harness(harness)]


def taxonomy_for(harness: str | None = None) -> dict[str, dict[str, Any]]:
    return profile_for(harness).taxonomy


def all_component_ids_for(harness: str | None = None) -> list[str]:
    return profile_for(harness).component_ids()


def aliases_for(harness: str | None = None) -> dict[str, str]:
    return profile_for(harness).component_aliases


def runtime_tools_for(harness: str | None = None) -> tuple[str, ...]:
    return profile_for(harness).runtime_tools


def teacher_only_tools_for(harness: str | None = None) -> tuple[str, ...]:
    return profile_for(harness).teacher_only_tools


def infer_harness_from_ids(component_ids: Iterable[str] | str | None) -> str:
    """Guess Harness-G only when every concrete id is G-only.

    Overlapping aliases such as ``neighbors`` / ``dedup`` stay Harness-1
    unless the caller passes an explicit ``--harness Harness-G``.
    ``all`` / ``zero`` are also Harness-1 unless the harness is explicit.
    """
    if component_ids is None:
        return DEFAULT_HARNESS
    if isinstance(component_ids, str):
        parts = [p.strip() for p in component_ids.replace(";", ",").split(",") if p.strip()]
    else:
        parts = [str(x).strip() for x in component_ids if str(x).strip()]
    tokens = [_norm(p) for p in parts]
    if not tokens or any(t in {"zero", "all"} for t in tokens):
        return DEFAULT_HARNESS
    g_ids = set(harness_g_components.COMPONENT_TAXONOMY)
    g_alias_map = {_norm(k): v for k, v in harness_g_components.COMPONENT_ALIASES.items()}
    h1_ids = set(HARNESS1_TAXONOMY)
    h1_alias_keys = {_norm(k) for k in HARNESS1_COMPONENT_ALIASES}
    resolved: list[str] = []
    for tok, raw in zip(tokens, parts):
        is_h1 = raw in h1_ids or tok in h1_alias_keys
        is_g = raw in g_ids or tok in g_alias_map
        if is_h1:
            return HARNESS_1
        if is_g:
            resolved.append(raw if raw in g_ids else g_alias_map[tok])
        else:
            return HARNESS_1
    if resolved and all(cid in g_ids for cid in resolved):
        return HARNESS_G
    return HARNESS_1


def is_harness_g(
    harness: str | None = None,
    *,
    component_ids: Iterable[str] | str | None = None,
    mask: Mapping[str, bool] | None = None,
) -> bool:
    if harness:
        try:
            return normalize_harness(harness) == HARNESS_G
        except KeyError:
            pass
    if mask and any(k in harness_g_components.COMPONENT_TAXONOMY for k in mask):
        return True
    return infer_harness_from_ids(component_ids) == HARNESS_G


def full_mask_for(harness: str | None = None) -> dict[str, bool]:
    tax = taxonomy_for(harness)
    return {cid: bool(meta["default_enabled"]) for cid, meta in tax.items()}


def zero_mask_for(harness: str | None = None) -> dict[str, bool]:
    return {cid: False for cid in taxonomy_for(harness)}


def minus_mask_for(
    component_id: str,
    base: Mapping[str, bool] | None = None,
    *,
    harness: str | None = None,
) -> dict[str, bool]:
    tax = taxonomy_for(harness)
    if component_id not in tax:
        raise KeyError(f"unknown component_id: {component_id}")
    mask = dict(base or full_mask_for(harness))
    mask[component_id] = False
    return mask


def coalition_minus_mask_for(
    component_ids: Iterable[str],
    base: Mapping[str, bool] | None = None,
    *,
    harness: str | None = None,
) -> dict[str, bool]:
    tax = taxonomy_for(harness)
    mask = dict(base or full_mask_for(harness))
    for component_id in component_ids:
        if component_id not in tax:
            raise KeyError(f"unknown component_id: {component_id}")
        mask[component_id] = False
    return mask


def mask_to_env_for(mask: Mapping[str, bool], *, harness: str | None = None) -> dict[str, str]:
    tax = taxonomy_for(harness)
    full = full_mask_for(harness)
    env: dict[str, str] = {}
    for cid, enabled in mask.items():
        if cid not in tax:
            raise KeyError(f"unknown component_id in mask: {cid}")
        env[str(tax[cid]["upstream_flag"])] = "1" if enabled else "0"
    for cid, meta in tax.items():
        env.setdefault(str(meta["upstream_flag"]), "1" if full[cid] else "0")
    return env


def env_to_mask_for(env: Mapping[str, str], *, harness: str | None = None) -> dict[str, bool]:
    tax = taxonomy_for(harness)
    mask: dict[str, bool] = {}
    for cid, meta in tax.items():
        raw = env.get(str(meta["upstream_flag"]))
        if raw is None:
            mask[cid] = bool(meta["default_enabled"])
        else:
            mask[cid] = str(raw) == "1"
    return mask
