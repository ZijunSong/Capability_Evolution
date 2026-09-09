"""V8D flag mapping for official eval and isolated subprocesses (E03).

Eval ``--component`` is the actual enabled set. It is independent of adapter
loading. Training Student / Teacher masks live in ``trim.cli.launch`` and
must not be reused here.
"""

from __future__ import annotations

import os
from typing import Iterable, Mapping, Sequence

from trim.adapters.components import all_component_ids, default_component_ids
from trim.adapters.harness_profiles import infer_harness_from_ids

EVALUATION_PATH_UPSTREAM_API = "upstream_api"
EVALUATION_PATH_LEGACY_LOCAL = "legacy_local"

V8D_COMPONENT_FLAGS: dict[str, str] = {
    "subtractive_curation": "V8D_SUBTRACTIVE_CURATION",
    "importance_tagging": "V8D_IMPORTANCE_TAGGING",
    "auto_populate_first_search": "V8D_AUTO_POPULATE_FIRST_SEARCH",
    "evidence_graph": "V8D_EVIDENCE_GRAPH",
    "sentence_compress": "V8D_SENTENCE_COMPRESS",
    "chunk_neighbors": "V8D_CHUNK_NEIGHBORS",
    "content_dedup": "V8D_CONTENT_DEDUP",
    "verify_tool": "V8D_VERIFY_TOOL",
    "token_budget_marker": "V8D_TOKEN_BUDGET_MARKER",
    "adaptive_rerank_instruction": "V8D_ADAPTIVE_RERANK_INSTRUCTION",
}

# Paper ablation ``all_harness_mechanisms_disabled`` also sets this. Official
# TRIM ``zero`` must not. Reproduce that paper cell under a separate name.
ABLATE_FLAGS_CLEARED_FOR_BASELINE: tuple[str, ...] = (
    "ABLATE_REVIEW_DOCS_UNAVAILABLE",
    "ABLATE_VERIFY_UNAVAILABLE",
)

FULL_ENV_EXTRAS: dict[str, str] = {
    "SENTENCE_COMPRESS_K": "4",
    "AUTO_POPULATE_TOP_K": "8",
}


def all_enabled_mask(harness: str | None = None) -> dict[str, bool]:
    """Ten v8d flags on (eval/train Teacher ``all``)."""
    return {cid: True for cid in all_component_ids(harness)}


def actual_eval_mask(
    component_ids: Sequence[str] | None,
    *,
    harness: str | None = None,
    preset: str | None = None,
) -> dict[str, bool]:
    """Eval mask: listed advanced components ON, remaining advanced components OFF.

    Does not take the Student complement. Does not OR onto default_enabled.
    Adapter / run-dir / model name must not change this result.
    """
    resolved = harness or infer_harness_from_ids(component_ids or [])
    known = list(all_component_ids(resolved))
    ids = list(component_ids or [])
    key = (preset or "").strip().lower()
    if key == "all" or (not key and ids == known):
        return {cid: True for cid in known}
    if key == "zero" or (not ids and key in {"", "zero"}):
        return {cid: False for cid in known}
    if key == "default":
        enabled = set(default_component_ids(resolved))
        return {cid: cid in enabled for cid in known}
    enabled = set(ids)
    return {cid: cid in enabled for cid in known}


def v8d_env_from_mask(mask: Mapping[str, bool], *, harness: str | None = None) -> dict[str, str]:
    resolved = harness or infer_harness_from_ids(list(mask.keys()))
    env: dict[str, str] = dict(FULL_ENV_EXTRAS)
    for cid in all_component_ids(resolved):
        flag = V8D_COMPONENT_FLAGS.get(cid)
        if flag is None:
            from trim.adapters.components import flag_for

            flag = flag_for(cid, harness=resolved)
        env[flag] = "1" if mask.get(cid) else "0"
    for name in ABLATE_FLAGS_CLEARED_FOR_BASELINE:
        env[name] = "0"
    return env


def cleared_ablate_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    env = dict(base or os.environ)
    for name in ABLATE_FLAGS_CLEARED_FOR_BASELINE:
        env.pop(name, None)
    return env


def subprocess_env_for_mask(
    mask: Mapping[str, bool],
    *,
    harness: str | None = None,
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Env for a worker that imports ultra_core *after* these values are set."""
    env = cleared_ablate_env(os.environ)
    for name in list(env):
        if name.startswith("ABLATE_") or name.startswith("V8D_"):
            env.pop(name, None)
    env.update(v8d_env_from_mask(mask, harness=harness))
    if extra:
        env.update({str(k): str(v) for k, v in extra.items()})
    return env


def mask_enabled_count(mask: Mapping[str, bool]) -> tuple[int, int]:
    values = list(mask.values())
    return sum(1 for v in values if v), len(values)


def describe_mask(mask: Mapping[str, bool]) -> dict[str, object]:
    on, total = mask_enabled_count(mask)
    return {
        "enabled_count": on,
        "total": total,
        "enabled": [cid for cid, bit in mask.items() if bit],
        "disabled": [cid for cid, bit in mask.items() if not bit],
        "mask": dict(mask),
    }
