"""Canonical state hashes for shadow purity / recovery fork checks."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)


def sha16(payload: str | bytes) -> str:
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def hash_working_memory_fields(
    *,
    curated_ids: list[str] | tuple[str, ...] = (),
    pool_ids: list[str] | tuple[str, ...] = (),
    search_history: list[str] | tuple[str, ...] = (),
    observation_ids: list[str] | tuple[str, ...] = (),
    turn_number: int = 0,
) -> str:
    return sha16(
        _canonical(
            {
                "curated": list(curated_ids),
                "pool": list(pool_ids),
                "history": list(search_history),
                "obs": list(observation_ids),
                "turn": int(turn_number),
            }
        )
    )


def hash_decision_state_core(state_dict: dict[str, Any]) -> str:
    """Hash student-visible DecisionState core (excludes rendered_context text)."""
    keys = (
        "episode_id",
        "task_id",
        "turn_id",
        "observation_ids",
        "observed_ids",
        "visible_document_ids",
        "pool_document_ids",
        "curated_document_ids",
        "remaining_turns",
        "remaining_search_calls",
        "remaining_open_calls",
        "token_budget_used",
        "token_budget_total",
        "last_action_type",
        "last_action_arguments",
        "wm_snapshot_hash",
    )
    payload = {k: state_dict.get(k) for k in keys if k in state_dict}
    # claims / verification as structured ids only
    claims = state_dict.get("evidence_claims") or []
    if claims:
        payload["claim_ids"] = [
            c.get("claim_id") if isinstance(c, dict) else getattr(c, "claim_id", None)
            for c in claims
        ]
    return sha16(_canonical(payload))


def env_purity_fingerprint(env: Any) -> dict[str, Any]:
    """Best-effort env fingerprint for shadow no-mutation audits."""
    fp: dict[str, Any] = {}
    for attr in ("_current_turn", "turn", "n_steps"):
        if hasattr(env, attr):
            try:
                val = getattr(env, attr)
                if callable(val):
                    continue
                fp[attr] = val
            except Exception:
                pass
    wm = getattr(env, "wm", None)
    if wm is not None:
        try:
            curated = list(getattr(wm, "curated_ids", []) or [])
            pool = list(getattr(wm, "pool_ids", []) or [])
            history = list(getattr(wm, "search_history", []) or [])
            lineage = getattr(wm, "observation_lineage", None) or []
            obs_ids = []
            for o in lineage:
                if isinstance(o, dict):
                    obs_ids.append(o.get("observation_id"))
                else:
                    obs_ids.append(getattr(o, "observation_id", None))
            fp["wm_hash"] = hash_working_memory_fields(
                curated_ids=curated,
                pool_ids=pool,
                search_history=history,
                observation_ids=[x for x in obs_ids if x],
                turn_number=int(fp.get("_current_turn") or fp.get("turn") or 0),
            )
            fp["n_curated"] = len(curated)
            fp["n_pool"] = len(pool)
            fp["n_obs"] = len(obs_ids)
            if hasattr(wm, "snapshot_hash"):
                fp["wm_snapshot_hash"] = wm.snapshot_hash()
        except Exception as exc:
            fp["wm_error"] = str(exc)
    # tool call counters if present
    for attr in ("_tool_call_count", "tool_call_count", "n_tool_calls"):
        if hasattr(env, attr):
            fp[attr] = getattr(env, attr)
    return fp


def fingerprints_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return sha16(_canonical(a)) == sha16(_canonical(b))
