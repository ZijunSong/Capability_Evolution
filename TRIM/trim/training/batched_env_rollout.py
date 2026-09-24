"""Turn-synchronous batched env rollouts.

All live episodes of a query×group batch share one generate_batch call
per turn so vLLM continuous batching sees hundreds of prompts at once.

Document stores are prepared in query micro-batches (default: enough
queries to fill ~256 live episodes). The first generate_batch is submitted
as soon as the first micro-batch is ready, and the next batch is prepared
on a background thread while the GPU is busy.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence
import time
import traceback

from trim.eval.browsecomp_retrieval import RetrievalBackend
from trim.eval.harness1_metrics import EpisodeTiming, episode_quality_metrics, timed_section, trace_fields
from trim.training.rl_opd_types import (
    COLLECTION_MODE_RL_OPD,
    CollectionNeeds,
    HybridRolloutGroup,
    StudentDecisionPoint,
    collection_needs,
)
from trim.training.vllm_hybrid import GenerateRequest, GenerateResult, cispo_row_from_generation

GenerateBatch = Callable[[Sequence[GenerateRequest]], list[GenerateResult]]

# Keep about this many live episodes in one vLLM generate_batch.
# With --group-size 8 that is 32 queries; eval (group_size=1) gets 256.
DEFAULT_TARGET_LIVE_EPISODES = 256
DEFAULT_DOC_STORE_WORKERS = 8


def resolved_query_batch_size(
    n_rows: int,
    group_size: int,
    query_batch_size: int | None,
) -> int:
    """How many queries to prepare before the first generate_batch call."""
    n_rows = max(0, int(n_rows))
    if query_batch_size is not None:
        n = int(query_batch_size)
        if n <= 0:
            return max(1, n_rows) if n_rows else 1
        return max(1, n)
    target = max(1, int(DEFAULT_TARGET_LIVE_EPISODES) // max(1, int(group_size)))
    if n_rows:
        return max(1, min(n_rows, target))
    return target


@dataclass
class LiveEpisode:
    row: dict[str, Any]
    rollout_idx: int
    seed: int
    st: dict[str, Any]
    component_id: str
    policy_version: str
    acts: list[tuple[Any, Any]] = field(default_factory=list)
    points: list[StudentDecisionPoint] = field(default_factory=list)
    rl_rows: list[dict[str, Any]] = field(default_factory=list)
    valids: list[bool] = field(default_factory=list)
    exec_oks: list[bool] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    names: list[str] = field(default_factory=list)
    pending_pre: Any = None
    pending_prefix: str = ""
    pending_pids: list[int] = field(default_factory=list)
    harness_mask: dict[str, bool] | None = None
    teacher_mode: bool = False
    timing: EpisodeTiming = field(default_factory=EpisodeTiming)
    eval_only: bool = False
    reasoning_effort: str | None = None
    turn_events: list[dict[str, Any]] = field(default_factory=list)
    collection_mode: str = COLLECTION_MODE_RL_OPD
    opd_loss: str = ""
    n_turns: int = 0
    pending_prompt_acts: list[tuple[Any, Any]] = field(default_factory=list)
    pending_wm_text: str = ""
    prompt_budget: dict[str, Any] = field(default_factory=dict)
    turn_diags: list[dict[str, Any]] = field(default_factory=list)


def _needs_for(ep_or_mode: Any, opd_loss: str | None = None) -> CollectionNeeds:
    if isinstance(ep_or_mode, str):
        mode = ep_or_mode
        loss = opd_loss
    else:
        mode = str(getattr(ep_or_mode, "collection_mode", None) or COLLECTION_MODE_RL_OPD)
        loss = opd_loss if opd_loss is not None else str(getattr(ep_or_mode, "opd_loss", "") or "")
    return collection_needs(collection_mode=mode, opd_loss=loss)


def _keep_snapshots(mode: str, opd_loss: str | None = None) -> bool:
    return _needs_for(mode, opd_loss).need_student_snapshot


def _keep_teacher_encode(mode: str, opd_loss: str | None = None) -> bool:
    """Live teacher-token encoding. Gap modes save recoverable refs instead."""
    return _needs_for(mode, opd_loss).need_debug_view


def _keep_teacher_context(mode: str, opd_loss: str | None = None) -> bool:
    return _needs_for(mode, opd_loss).need_teacher_context


def _keep_dual_view(mode: str, opd_loss: str | None = None) -> bool:
    return _needs_for(mode, opd_loss).need_debug_view


def _prompt_budget(enc) -> tuple[int, int]:
    max_model_len = getattr(enc, "max_model_len", None) if enc is not None else None
    max_new = getattr(enc, "max_new_tokens", None) if enc is not None else None
    if max_model_len in (None, 0):
        max_model_len = 8192
    if max_new in (None, 0):
        max_new = 2048
    return int(max_model_len), int(max_new)


def _build_prompt_ids(ep: LiveEpisode, enc) -> list[int]:
    from trim.adapters.harness_profiles import is_harness_g
    from trim.eval.harmony_runtime import build_continuation_prompt_ids, build_first_turn_prompt_ids
    from trim.training.upstream_train_env import is_upstream_state, wm_text_for_train_state

    query = str(ep.row["query"])
    max_model_len, max_new = _prompt_budget(enc)
    budget = max(1, max_model_len - max(1, max_new))
    trim_report = {
        "max_model_len": max_model_len,
        "max_new_tokens": max_new,
        "history_budget": None,
        "dropped_history": 0,
        "pre_len": None,
        "post_len": None,
        "dropped_category": [],
    }
    if is_harness_g(mask=ep.harness_mask, component_ids=ep.component_id):
        from trim.eval.harness_g_env import wm_text
        from trim.eval.harness_g_runtime import build_prompt_ids as build_g_prompt_ids
        from trim.eval.harmony_runtime import fit_prompt_ids_to_context

        from trim.eval.harmony_runtime import prompt_history_keep, recent_actions_obs

        keep = prompt_history_keep(enc, harness_mask=ep.harness_mask)
        acts = recent_actions_obs(list(ep.acts), keep=keep)
        trim_report["history_budget"] = keep
        ids = build_g_prompt_ids(
            query, wm_text(ep.st), enc, harness_mask=ep.harness_mask, actions_obs=acts,
            reasoning_effort=ep.reasoning_effort,
        )
        trim_report["pre_len"] = len(ids)
        dropped_hist = 0
        while len(ids) > budget and len(acts) > 1:
            acts = acts[1:]
            dropped_hist += 1
            ids = build_g_prompt_ids(
                query, wm_text(ep.st), enc, harness_mask=ep.harness_mask, actions_obs=acts,
                reasoning_effort=ep.reasoning_effort,
            )
        if dropped_hist:
            trim_report["dropped_category"].append("history")
            trim_report["dropped_history"] = dropped_hist
        if len(ids) > budget:
            ids = fit_prompt_ids_to_context(ids, max_model_len=max_model_len, max_new_tokens=max_new)
            trim_report["dropped_category"].append("token_tail")
        trim_report["post_len"] = len(ids)
        trim_report["prompt_truncated"] = bool(trim_report["dropped_category"]) or int(trim_report["post_len"] or 0) < int(
            trim_report["pre_len"] or 0
        )
        ep.pending_prompt_acts = list(acts)
        ep.pending_wm_text = str(wm_text(ep.st) or "")
        ep.prompt_budget = dict(trim_report)
        ep.st["prompt_budget"] = trim_report
        vis = list(ep.st.get("visible_sids") or [])
        ep.st["prompt_visible_sids"] = list(vis)
        return ids
    wm = wm_text_for_train_state(ep.st) if is_upstream_state(ep.st) else None
    if wm is None:
        from trim.eval.local_search_env import wm_text

        wm = wm_text(ep.st)
    from trim.eval.harmony_runtime import prompt_history_keep, recent_actions_obs

    keep = prompt_history_keep(enc, harness_mask=ep.harness_mask)
    acts = recent_actions_obs(list(ep.acts), keep=keep)

    def _encode(use_acts: list) -> list[int]:
        if enc is not None and hasattr(enc, "build_first_turn_prompt_ids"):
            if not use_acts:
                return list(enc.build_first_turn_prompt_ids(query))
            return list(enc.build_continuation_prompt_ids(query, actions_obs=use_acts, wm_text=wm))
        if not use_acts:
            return build_first_turn_prompt_ids(query, enc=enc)
        return build_continuation_prompt_ids(
            query,
            actions_obs=use_acts,
            wm_text=wm,
            enc=enc,
        )

    ids = _encode(acts)
    pre_len = len(ids)
    n_acts_before = len(acts)
    while len(ids) > budget and len(acts) > 1:
        acts = acts[1:]
        ids = _encode(acts)
    token_trimmed = False
    if len(ids) > budget:
        from trim.eval.harmony_runtime import fit_prompt_ids_to_context

        ids = fit_prompt_ids_to_context(ids, max_model_len=max_model_len, max_new_tokens=max_new)
        token_trimmed = True
    ep.pending_prompt_acts = list(acts)
    ep.pending_wm_text = str(wm or "")
    ep.prompt_budget = {
        "max_model_len": max_model_len,
        "max_new_tokens": max_new,
        "history_budget": keep,
        "pre_len": pre_len,
        "post_len": len(ids),
        "dropped_history": n_acts_before - len(acts),
        "prompt_truncated": bool(token_trimmed or len(acts) < n_acts_before),
    }
    return ids


def _teacher_wm_for_episode(ep: LiveEpisode) -> str:
    from trim.training.four_cell_runtime import freeze_train_state, teacher_mask_for
    from trim.training.upstream_train_env import is_upstream_state, wm_text_for_train_state

    teacher_st = freeze_train_state(ep.st)
    teacher_st["harness_mask"] = teacher_mask_for(ep.component_id)
    if is_upstream_state(teacher_st):
        return str(wm_text_for_train_state(teacher_st) or "")
    from trim.eval.local_search_env import wm_text as local_wm_text

    return str(local_wm_text(teacher_st) or "")


def _named_len(state: Mapping[str, Any], keys: Sequence[str]) -> int | None:
    for key in keys:
        val = state.get(key)
        if isinstance(val, (list, tuple, set, dict)):
            return len(val)
    return None


def _append_turn_diag(
    ep: LiveEpisode,
    *,
    action: Mapping[str, Any],
    valid: bool,
    executed_ok: bool,
    gen: GenerateResult,
    turn_id: int,
    qid: str,
    curated_before: int | None,
    pool_before: int | None,
    prompt_ids: Sequence[int],
) -> None:
    """Compact per-turn record. Full token IDs stay on the RL row and are sampled later."""
    name = str(action.get("name") or "")
    finish = str(getattr(gen, "finish_reason", "") or "")
    duplicate = False
    if len(ep.actions) >= 2:
        prev = ep.actions[-2]
        duplicate = str(prev.get("name") or "") == name and dict(prev.get("arguments") or {}) == dict(
            action.get("arguments") or {}
        )
    budget = dict(getattr(ep, "prompt_budget", None) or {})
    curated_after = _named_len(ep.st, ("curated", "curated_ids", "selected_docids"))
    pool_after = _named_len(ep.st, ("pool", "observed_docids", "documents"))
    if valid and name not in {"truncated", "unknown", ""}:
        parse_method = "tool_call"
        parse_error = ""
    elif finish == "length" or name == "truncated":
        parse_method = "length"
        parse_error = "generation_length"
    else:
        parse_method = "parse_failed"
        parse_error = name or "unparsed"
    ep.turn_diags.append(
        {
            "request_id": str(getattr(gen, "request_id", "") or ""),
            "query_id": qid,
            "episode_id": f"{qid}_r{ep.rollout_idx}",
            "turn_id": int(turn_id),
            "policy_version": str(ep.policy_version),
            "parse_method": parse_method,
            "parse_error": parse_error,
            "tool_name": name,
            "structurally_valid": bool(valid),
            "executed_ok": bool(executed_ok),
            "generation_finish_reason": finish,
            "episode_end_reason": str(ep.st.get("end_reason") or "") if ep.st.get("ended") else "",
            "new_doc_count": None if pool_before is None or pool_after is None else max(0, pool_after - pool_before),
            "curated_delta": None
            if curated_before is None or curated_after is None
            else int(curated_after - curated_before),
            "duplicate_action": bool(duplicate),
            "prompt_truncated": bool(budget.get("prompt_truncated")),
            "prompt_pre_len": budget.get("pre_len"),
            "prompt_post_len": budget.get("post_len"),
            "n_prompt_ids": len(list(prompt_ids)),
        }
    )


def _finalize_episode_diags(ep: LiveEpisode, *, max_turns: int) -> list[dict[str, Any]]:
    diags = [dict(item) for item in (ep.turn_diags or [])]
    if not diags:
        return []
    ended = bool(ep.st.get("ended"))
    if ended:
        end_reason = str(ep.st.get("end_reason") or "ended")
    elif len(ep.names) >= int(max_turns):
        end_reason = "turn_limit"
    else:
        end_reason = ""
    last_turn = diags[-1].get("turn_id")
    for item in diags:
        item["episode_end_reason"] = end_reason if item.get("turn_id") == last_turn else ""
        item["last_turn_id"] = last_turn
    return diags


def turn_audit_from_groups(groups: Sequence[Any]) -> dict[str, Any]:
    """Aggregate lightweight turn diagnostics and keep one sample of each kind."""
    counts = {
        "n_turns": 0,
        "n_episodes": 0,
        "n_tool_call": 0,
        "n_end_search": 0,
        "n_parse_failed": 0,
        "n_length": 0,
        "n_turn_limit_episodes": 0,
        "n_duplicate_action": 0,
        "n_prompt_truncated": 0,
        "n_ended": 0,
    }
    samples: dict[str, dict[str, Any]] = {}
    for group in groups:
        tg = getattr(group, "trajectory_group", None) or {}
        if not isinstance(tg, dict):
            continue
        diags = list(tg.get("turn_diags") or [])
        rows = list(tg.get("rl_rows") or [])
        by_turn: dict[tuple[str, int], dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            by_turn[(str(row.get("episode_id") or ""), int(row.get("turn_id") or 0))] = row
        seen_eps: set[str] = set()
        episode_end: dict[str, str] = {}
        for diag in diags:
            if not isinstance(diag, dict):
                continue
            counts["n_turns"] += 1
            episode_id = str(diag.get("episode_id") or "")
            if episode_id:
                seen_eps.add(episode_id)
                if diag.get("episode_end_reason"):
                    episode_end[episode_id] = str(diag.get("episode_end_reason") or "")
            if diag.get("duplicate_action"):
                counts["n_duplicate_action"] += 1
            if diag.get("prompt_truncated"):
                counts["n_prompt_truncated"] += 1
            kind = _turn_diag_kind(diag)
            if kind == "tool_call":
                counts["n_tool_call"] += 1
            elif kind == "end_search":
                counts["n_end_search"] += 1
            elif kind == "parse_failed":
                counts["n_parse_failed"] += 1
            elif kind == "length":
                counts["n_length"] += 1
            if kind and kind not in samples:
                row = by_turn.get((episode_id, int(diag.get("turn_id") or 0))) or {}
                samples[kind] = {
                    "kind": kind,
                    **diag,
                    "prompt_token_ids": list(row.get("effective_prompt_ids") or row.get("prompt_ids") or []),
                    "action_token_ids": list(row.get("action_ids") or row.get("target_ids") or []),
                }
        counts["n_episodes"] += len(seen_eps)
        for reason in episode_end.values():
            if reason == "turn_limit":
                counts["n_turn_limit_episodes"] += 1
            elif reason:
                counts["n_ended"] += 1
        if any(str(d.get("episode_end_reason") or "") == "turn_limit" for d in diags if isinstance(d, dict)):
            if "turn_limit" not in samples:
                last = next(d for d in reversed(diags) if isinstance(d, dict))
                episode_id = str(last.get("episode_id") or "")
                row = by_turn.get((episode_id, int(last.get("turn_id") or 0))) or {}
                samples["turn_limit"] = {
                    "kind": "turn_limit",
                    **last,
                    "prompt_token_ids": list(row.get("effective_prompt_ids") or row.get("prompt_ids") or []),
                    "action_token_ids": list(row.get("action_ids") or row.get("target_ids") or []),
                }
    return {"counts": counts, "samples": [samples[key] for key in ("tool_call", "end_search", "parse_failed", "length", "turn_limit") if key in samples]}


def _turn_diag_kind(diag: Mapping[str, Any]) -> str:
    finish = str(diag.get("generation_finish_reason") or "")
    name = str(diag.get("tool_name") or "")
    if finish == "length" or name == "truncated" or diag.get("parse_method") == "length":
        return "length"
    if not diag.get("structurally_valid"):
        return "parse_failed"
    end_reason = str(diag.get("episode_end_reason") or "")
    if name == "end_search" and diag.get("executed_ok") and end_reason not in {"", "turn_limit"}:
        return "end_search"
    if diag.get("structurally_valid") and diag.get("executed_ok") and name not in {"end_search", "truncated", "unknown", ""}:
        return "tool_call"
    return ""


def _apply_generation(
    ep: LiveEpisode,
    gen: GenerateResult,
    *,
    enc,
    searcher: RetrievalBackend | None,
    search_k: int = 10,
) -> None:
    from trim.eval.harmony_runtime import decode_ids, make_action, make_observation
    from trim.adapters.harness_profiles import is_harness_g
    from trim.training.parse_rollout_action import parse_generated_action
    from trim.training.four_cell_runtime import (
        encode_aligned_teacher_prompt,
        freeze_train_state,
        snap_from_state,
    )
    from trim.training.upstream_train_env import apply_train_action, is_upstream_state

    if is_harness_g(mask=ep.harness_mask, component_ids=ep.component_id):
        from trim.eval.harness_g_env import execute_tool
    else:
        from trim.eval.local_search_env import execute_tool

    qid = str(ep.row["query_id"])
    eval_only = bool(getattr(ep, "eval_only", False) or ep.policy_version == "eval")
    frozen_acts = list(ep.acts)
    teacher_prompt_ids: list[int] = []
    teacher_snapshot_hash = ""
    mode = str(getattr(ep, "collection_mode", None) or COLLECTION_MODE_RL_OPD)
    loss = str(getattr(ep, "opd_loss", "") or "")
    if (
        enc is not None
        and not ep.teacher_mode
        and not eval_only
        and _keep_teacher_encode(mode, loss)
    ):
        frozen_st = freeze_train_state(ep.st)
        teacher_prompt_ids, _ = encode_aligned_teacher_prompt(
            enc,
            str(ep.row["query"]),
            frozen_st=frozen_st,
            frozen_acts=frozen_acts,
            component_id=ep.component_id,
        )
        teacher_snapshot_hash = snap_from_state(
            qid, frozen_st, ep.component_id, harness_mask=ep.harness_mask
        ).content_hash()
    g_eval = is_harness_g(mask=ep.harness_mask, component_ids=ep.component_id)
    attempt: dict[str, Any] | None = None
    if g_eval:
        from trim.eval.harness_g_env import _evidence_sig, _menu_hash

        attempt = {
            "query_id": qid,
            "rollout_id": int(ep.rollout_idx),
            "turn_id": int(ep.n_turns),
            "attempt_id": f"{qid}:e{ep.rollout_idx}:t{ep.n_turns}",
            "raw_output": str(gen.text or "")[:4000],
            "finish_reason": str(getattr(gen, "finish_reason", "") or ""),
            "parse_ok": None,
            "schema_ok": None,
            "menu_ok": None,
            "execution_ok": None,
            "normalized_action": None,
            "pre_state_hash": _evidence_sig(ep.st),
            "pre_menu_hash": _menu_hash(ep.st),
            "error_stage": None,
            "error_type": None,
            "error_message": None,
            "traceback_ref": None,
            "post_state_hash": None,
            "state_committed": False,
            "termination_reason": None,
        }
    action = {"name": "unknown", "arguments": {}}
    valid = False
    curated_before = _named_len(ep.st, ("curated", "curated_ids", "selected_docids"))
    pool_before = _named_len(ep.st, ("pool", "observed_docids", "documents"))
    try:
        with timed_section(ep.timing, "parse"):
            action, valid = parse_generated_action(
                gen.text,
                gen.token_ids,
                enc,
                harness_mask=ep.harness_mask,
                teacher_mode=bool(ep.teacher_mode),
                action_map=ep.st.get("action_map"),
                finish_reason=str(getattr(gen, "finish_reason", "") or ""),
            )
    except Exception as exc:  # noqa: BLE001
        if not g_eval:
            raise
        tb = traceback.format_exc()
        valid = False
        action = {"name": "unknown", "arguments": {}}
        if attempt is not None:
            attempt["error_stage"] = "parse"
            attempt["error_type"] = type(exc).__name__
            attempt["error_message"] = str(exc)
            attempt["traceback_ref"] = tb[-4000:]
    if attempt is not None:
        attempt["parse_ok"] = bool(valid)
        attempt["schema_ok"] = bool(valid)
        attempt["normalized_action"] = {
            "name": action.get("name"),
            "arguments": dict(action.get("arguments") or {}),
        }
    ep.valids.append(valid)
    ep.actions.append(action)
    ep.names.append(str(action.get("name")))
    _ok = False
    obs = ""
    with timed_section(ep.timing, "harness"):
        try:
            if is_upstream_state(ep.st):
                ep.st, obs, _ok = apply_train_action(
                    ep.st,
                    action,
                    valid,
                    searcher=searcher,
                    search_k=search_k,
                    mods=ep.st.get("_upstream_mods"),
                    execute_local=execute_tool,
                )
            elif g_eval:
                if not valid:
                    from trim.eval.harness_g_env import record_nonexecution_failure

                    name = str(action.get("name") or "unknown")
                    if name == "truncated":
                        code, msg = "truncated_output", "generation hit length limit without a complete tool call."
                    else:
                        code, msg = "parse_failed", "could not parse an executable tool call from model output."
                    if attempt is not None:
                        attempt["parse_ok"] = False
                        attempt["schema_ok"] = False
                        attempt["error_stage"] = "parse"
                        attempt["error_type"] = code
                        attempt["error_message"] = msg
                    ep.st, obs, _ok = record_nonexecution_failure(
                        ep.st,
                        name,
                        dict(action.get("arguments") or {}),
                        code=code,
                        msg=msg,
                        parse_ok=False,
                        schema_ok=False,
                        error_class="protocol",
                    )
                else:
                    ep.st, obs, exec_ok = execute_tool(
                        ep.st,
                        action.get("name"),
                        action.get("arguments"),
                        searcher=searcher,
                        search_k=search_k,
                    )
                    _ok = bool(exec_ok)
            else:
                ep.st, obs, _ok = apply_train_action(
                    ep.st,
                    action,
                    valid,
                    searcher=searcher,
                    search_k=search_k,
                    execute_local=execute_tool,
                )
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            if g_eval:
                from trim.eval.harness_g_env import record_nonexecution_failure

                if attempt is not None:
                    attempt["error_stage"] = "execute"
                    attempt["error_type"] = type(exc).__name__
                    attempt["error_message"] = str(exc)
                    attempt["traceback_ref"] = tb[-4000:]
                ep.st, obs, _ok = record_nonexecution_failure(
                    ep.st,
                    str(action.get("name") or "unknown"),
                    dict(action.get("arguments") or {}),
                    code="infrastructure_failure",
                    msg=f"{type(exc).__name__}: {exc}",
                    parse_ok=bool(valid),
                    schema_ok=bool(valid),
                    error_class="infrastructure",
                    traceback_ref=tb[-4000:],
                )
            else:
                ep.st["invalid_tools"] = int(ep.st.get("invalid_tools") or 0) + 1
                obs = f"ERROR: tool failed ({type(exc).__name__})."
                _ok = False
        finally:
            if g_eval:
                from trim.eval.harness_g_env import _evidence_sig, _menu_hash

                hist = list(ep.st.get("tool_history") or [])
                last_hist = hist[-1] if hist else {}
                last_event = (list(ep.st.get("turn_events") or []) or [{}])[-1]
                sealed = dict(attempt or {})
                sealed["parse_ok"] = bool(
                    sealed.get("parse_ok")
                    if sealed.get("parse_ok") is not None
                    else last_hist.get("parse_ok")
                )
                sealed["schema_ok"] = bool(
                    sealed.get("schema_ok")
                    if sealed.get("schema_ok") is not None
                    else last_hist.get("schema_ok")
                )
                sealed["menu_ok"] = bool(
                    last_hist.get("menu_ok")
                    if last_hist.get("menu_ok") is not None
                    else last_hist.get("target_ok")
                )
                sealed["execution_ok"] = bool(_ok)
                sealed["post_state_hash"] = _evidence_sig(ep.st)
                sealed["post_menu_hash"] = _menu_hash(ep.st)
                sealed["state_committed"] = bool(_ok)
                sealed["termination_reason"] = ep.st.get("end_reason") if ep.st.get("ended") else None
                sealed["n_tokens"] = len(list(gen.token_ids))
                if last_hist.get("error_code") and not sealed.get("error_type"):
                    sealed["error_type"] = last_hist.get("error_code")
                if last_event.get("traceback_ref") and not sealed.get("traceback_ref"):
                    sealed["traceback_ref"] = last_event.get("traceback_ref")
                    sealed["error_class"] = last_event.get("error_class") or sealed.get("error_class")
                events = list(ep.st.get("attempt_events") or [])
                events.append(sealed)
                ep.st["attempt_events"] = events
                ep.turn_events.append(dict(sealed))
        if is_harness_g(mask=ep.harness_mask, component_ids=ep.component_id):
            from trim.eval.harness_g_runtime import make_protocol_feedback

            if valid:
                try:
                    attempted_name = str(action.get("name") or "unknown")
                    ep.acts.append(
                        (
                            make_action(attempted_name, action.get("arguments") or {}),
                            make_observation(obs),
                        )
                    )
                except Exception:
                    ep.acts.append((make_protocol_feedback(str(obs)), str(obs)))
            else:
                ep.acts.append((make_protocol_feedback(str(obs)), str(obs)))
        else:
            try:
                attempted_name = str(action.get("name") or "unknown")
                ep.acts.append(
                    (make_action(attempted_name, action.get("arguments") or {}), make_observation(obs))
                )
            except Exception:
                pass
        if ep.st.get("ended"):
            ep.timing.mark_finished()
        action_ids = list(gen.token_ids)
        effective_prompt_ids = list(getattr(gen, "effective_prompt_ids", None) or ep.pending_pids)
        prompt_text = ""
        if not eval_only and enc is not None and _keep_dual_view(mode, loss):
            try:
                prompt_text = decode_ids(enc, effective_prompt_ids)
            except Exception:
                prompt_text = ""
        prompt_text = prompt_text or ("" if eval_only else ep.pending_prefix)
        ep.exec_oks.append(bool(_ok))
        truncated = str(getattr(gen, "finish_reason", "") or "") == "length"
        post = None
        if (_keep_snapshots(mode, loss) or ep.teacher_mode) and not eval_only:
            post = snap_from_state(qid, ep.st, ep.component_id, harness_mask=ep.harness_mask)
    turn_id = int(ep.n_turns)
    ep.n_turns += 1
    if (_keep_snapshots(mode, loss) or ep.teacher_mode) and ep.pending_pre is not None and not eval_only:
        from trim.training.action_encoding import (
            prompt_visible_doc_ids_from_prompt,
            visible_doc_ids_from_snapshot,
        )

        accessible = visible_doc_ids_from_snapshot(ep.pending_pre)
        prompt_visible = prompt_visible_doc_ids_from_prompt(
            prompt_ids=effective_prompt_ids,
            accessible_ids=accessible,
            enc=enc,
        )
        recorded_visible = list(prompt_visible) if prompt_visible is not None else []
        ep.points.append(
            StudentDecisionPoint(
                episode_id=f"{qid}_r{ep.rollout_idx}",
                query_id=qid,
                rollout_idx=ep.rollout_idx,
                turn_id=turn_id,
                policy_version=ep.policy_version,
                pre_action_snapshot=ep.pending_pre,
                pre_action_snapshot_hash=ep.pending_pre.content_hash(),
                student_model_input=ep.pending_prefix,
                student_action_tokens=action_ids,
                student_action_text=gen.text,
                action_tool_names=[action.get("name") or ""],
                post_action_snapshot=post,
                reward=None,
                structurally_valid=valid,
                executed_ok=bool(_ok),
                student_prompt_token_ids=list(effective_prompt_ids),
                teacher_prompt_token_ids=list(teacher_prompt_ids),
                visible_doc_ids=recorded_visible,
                accessible_doc_ids=list(accessible),
                prompt_visible_doc_ids=prompt_visible,
                teacher_snapshot_hash=teacher_snapshot_hash,
                teacher_decision_turn=turn_id,
                history_end_turn=len(frozen_acts),
            )
        )
    rec = cispo_row_from_generation(
        query_id=qid,
        prompt_ids=effective_prompt_ids,
        prompt_text=prompt_text,
        gen=gen,
        policy_version=ep.policy_version,
        turn_id=turn_id,
        valid=valid and not truncated,
    )
    rec["episode_id"] = f"{qid}_r{ep.rollout_idx}"
    rec["rollout_idx"] = ep.rollout_idx
    if truncated:
        rec["truncated_generation"] = True
    if not eval_only:
        ep.rl_rows.append(rec)
    _append_turn_diag(
        ep,
        action=action,
        valid=valid,
        executed_ok=bool(_ok),
        gen=gen,
        turn_id=turn_id,
        qid=qid,
        curated_before=curated_before,
        pool_before=pool_before,
        prompt_ids=effective_prompt_ids,
    )


def _prepare_chunk_episodes(
    chunk: Sequence[dict[str, Any]],
    *,
    component_id: str,
    group_size: int,
    policy_version: str,
    seed: int,
    harness_mask: dict[str, bool] | None,
    searcher: RetrievalBackend | None,
    doc_store_k: int,
    doc_store_workers: int,
    new_state,
    doc_store_for_row,
    teacher_mode: bool = False,
    collection_mode: str = COLLECTION_MODE_RL_OPD,
    opd_loss: str = "",
) -> list[LiveEpisode]:
    workers = max(1, int(doc_store_workers or 1))
    if workers == 1 or len(chunk) <= 1:
        stores = [doc_store_for_row(row, searcher, k=doc_store_k) for row in chunk]
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(chunk))) as pool:
            stores = list(
                pool.map(lambda row: doc_store_for_row(row, searcher, k=doc_store_k), chunk)
            )
    episodes: list[LiveEpisode] = []
    for row, store in zip(chunk, stores):
        copied = dict(store)
        for g in range(group_size):
            episodes.append(
                LiveEpisode(
                    row=row,
                    rollout_idx=g,
                    seed=int(seed) + 17 * g,
                    st=new_state(str(row["query"]), dict(copied), str(row.get("query_id") or "")),
                    component_id=component_id,
                    policy_version=policy_version,
                    harness_mask=harness_mask,
                    teacher_mode=teacher_mode,
                    eval_only=str(policy_version) == "eval",
                    collection_mode=str(collection_mode or COLLECTION_MODE_RL_OPD),
                    opd_loss=str(opd_loss or ""),
                )
            )
    return episodes


def _run_episode_turns(
    episodes: list[LiveEpisode],
    generate_batch: GenerateBatch,
    *,
    component_id: str,
    max_turns: int,
    max_new: int,
    policy_version: str,
    enc,
    searcher: RetrievalBackend | None,
    teacher_mode: bool,
    temperature: float,
    search_k: int,
    snap_from_state,
    env_workers: int = 1,
) -> None:
    for turn in range(max_turns):
        live = [ep for ep in episodes if not ep.st.get("ended")]
        if not live:
            break
        reqs: list[GenerateRequest] = []
        generated: list[GenerateResult | None] = [None] * len(live)
        request_slots: list[int] = []
        apply_jobs: list[tuple[LiveEpisode, GenerateResult]] = []
        for i, ep in enumerate(live):
            with timed_section(ep.timing, "prompt"):
                pids = _build_prompt_ids(ep, enc)
            with timed_section(ep.timing, "snapshot"):
                mode = str(getattr(ep, "collection_mode", None) or COLLECTION_MODE_RL_OPD)
                loss = str(getattr(ep, "opd_loss", "") or "")
                if (_keep_snapshots(mode, loss) or ep.teacher_mode) and not ep.eval_only:
                    pre = snap_from_state(str(ep.row["query_id"]), ep.st, component_id, harness_mask=ep.harness_mask)
                    teacher_wm = None
                    if _keep_teacher_context(mode, loss) or _keep_teacher_encode(mode, loss):
                        teacher_wm = _teacher_wm_for_episode(ep)
                    from trim.training.opd_prompt_encoding import attach_prompt_context

                    attach_prompt_context(
                        pre,
                        acts=list(getattr(ep, "pending_prompt_acts", None) or []),
                        wm_text=str(getattr(ep, "pending_wm_text", "") or ""),
                        teacher_wm_text=teacher_wm,
                    )
                    ep.pending_pre = pre
                else:
                    ep.pending_pre = None
                if _keep_dual_view(mode, loss) and ep.pending_pre is not None:
                    from trim.training.opd_dataset import render_student_prompt

                    ep.pending_prefix = render_student_prompt(ep.pending_pre, component_id=component_id)
                else:
                    ep.pending_prefix = ""
                ep.pending_pids = pids
            if teacher_mode:
                from trim.training.action_codec import render_action
                from trim.training.four_cell_runtime import teacher_action_from_point
                from trim.training.rl_opd_types import StudentDecisionPoint

                point = StudentDecisionPoint(
                    episode_id=f"{ep.row['query_id']}_r{ep.rollout_idx}",
                    query_id=str(ep.row["query_id"]),
                    rollout_idx=ep.rollout_idx,
                    turn_id=int(ep.n_turns),
                    policy_version=policy_version,
                    pre_action_snapshot=pre,
                    pre_action_snapshot_hash=pre.content_hash(),
                    student_model_input=ep.pending_prefix,
                    student_action_tokens=[],
                    student_action_text="",
                    action_tool_names=[],
                    post_action_snapshot=pre,
                    reward=None,
                    structurally_valid=True,
                )
                action = teacher_action_from_point(point, component_id)
                text = render_action(action)
                token_ids = list(enc.encode(text))
                generated[i] = GenerateResult(
                    request_id=f"{ep.row['query_id']}:e{ep.rollout_idx}:t{int(ep.n_turns)}",
                    token_ids=token_ids,
                    token_logprobs=[0.0] * len(token_ids),
                    text=text,
                    logprob_old=0.0,
                    logprob_provenance="teacher_projected_action",
                )
            else:
                request_slots.append(i)
                reqs.append(
                    GenerateRequest(
                        request_id=f"{ep.row['query_id']}:e{ep.rollout_idx}:t{int(ep.n_turns)}",
                        prompt_token_ids=pids,
                        max_new_tokens=max_new,
                        temperature=temperature,
                        seed=ep.seed + 31 * turn,
                    )
                )
        if reqs:
            t_gen = time.perf_counter()
            gens = generate_batch(reqs)
            gen_dt = time.perf_counter() - t_gen
            share = gen_dt / max(1, len(reqs))
            if len(gens) != len(reqs):
                raise RuntimeError(f"generate_batch returned {len(gens)} for {len(reqs)} requests")
            for slot, gen in zip(request_slots, gens):
                generated[slot] = gen
                live[slot].timing.add_model(share)
        for ep, gen in zip(live, generated):
            if gen is None:
                raise RuntimeError("missing generation for live episode")
            apply_jobs.append((ep, gen))
        workers = max(1, int(env_workers or 1))
        if workers == 1 or len(apply_jobs) <= 1:
            for ep, gen in apply_jobs:
                _apply_generation(ep, gen, enc=enc, searcher=searcher, search_k=search_k)
        else:
            def apply_one(job: tuple[LiveEpisode, GenerateResult]) -> None:
                ep, gen = job
                _apply_generation(ep, gen, enc=enc, searcher=searcher, search_k=search_k)

            with ThreadPoolExecutor(max_workers=min(workers, len(apply_jobs))) as pool:
                list(pool.map(apply_one, apply_jobs))
    for ep in episodes:
        ep.timing.mark_finished()


def _groups_from_episodes(
    episodes: list[LiveEpisode],
    rows: Sequence[dict[str, Any]],
    *,
    policy_version: str,
    max_turns: int,
    terminal_reward,
    curated_recall,
) -> list[HybridRolloutGroup]:
    by_q: dict[str, list[LiveEpisode]] = {}
    for ep in episodes:
        by_q.setdefault(str(ep.row["query_id"]), []).append(ep)

    groups: list[HybridRolloutGroup] = []
    for row in rows:
        qid = str(row["query_id"])
        members = by_q.get(qid, [])
        points: list[StudentDecisionPoint] = []
        rl_rows: list[dict[str, Any]] = []
        rewards: list[float] = []
        tool_seqs: list[list[str]] = []
        gold_ids = [str(x) for x in (row.get("gold_docids") or row.get("evidence_docids") or [])]
        query = str(row["query"])
        episode_stats: list[dict[str, Any]] = []
        turn_diags: list[dict[str, Any]] = []
        for ep in members:
            from trim.training.four_cell_runtime import terminal_reward_breakdown

            parts = terminal_reward_breakdown(
                ep.st,
                query=query,
                gold_ids=gold_ids,
                valids=ep.valids,
                actions=ep.actions,
                exec_oks=ep.exec_oks,
            )
            reward = float(parts["total"])
            for point in ep.points:
                point.reward = reward
                point.reward_parts = dict(parts)
            for rec in ep.rl_rows:
                rec["reward"] = reward
            points.extend(ep.points)
            rl_rows.extend(ep.rl_rows)
            rewards.append(reward)
            tool_seqs.append(list(ep.names))
            ep.st["gold_recall"] = float(curated_recall(ep.st, gold_ids) or 0.0)
            quality = episode_quality_metrics(
                ep.st,
                row,
                tool_names=ep.names,
                valids=ep.valids,
                reward=reward,
                max_turns=max_turns,
                timing=ep.timing.snapshot(),
                actions=ep.actions,
            )
            quality.update(
                {
                    "names": list(ep.names),
                    "ended": bool(ep.st.get("ended")),
                    "n_turns": len(ep.names),
                    "n_tool_calls": int(ep.st.get("n_tool_calls") or 0),
                    "n_search_calls": int(ep.st.get("n_search_calls") or 0),
                    "search_query": quality.get("search_query") or query,
                    "reward_parts": dict(parts),
                    "task_recall": float(parts.get("task_recall") or 0.0),
                    "n_exec_ok": int(parts.get("n_exec_ok") or 0),
                    "n_structurally_valid": sum(1 for v in ep.valids if v),
                    "max_turns": int(max_turns),
                    "n_valids": len(ep.valids),
                }
            )
            episode_stats.append(quality)
            turn_diags.extend(_finalize_episode_diags(ep, max_turns=max_turns))
        from trim.training.hf_rl_opd_client import episode_relative_advantages

        adv = episode_relative_advantages(rl_rows)
        for rec, a in zip(rl_rows, adv):
            rec["advantage"] = a
        groups.append(
            HybridRolloutGroup(
                query_id=qid,
                policy_version=policy_version,
                trajectory_group={
                    "rl_rows": rl_rows,
                    "query": row.get("query"),
                    "tool_seqs": tool_seqs,
                    "episode_stats": episode_stats,
                    "turn_diags": turn_diags,
                },
                decision_points=points,
                terminal_rewards=rewards,
                metadata={
                    "n_rl_rows": len(rl_rows),
                    "reward_spread": (max(rewards) - min(rewards)) if rewards else 0.0,
                    "batched": True,
                    "collection_mode": str(getattr(members[0], "collection_mode", "") if members else ""),
                },
            )
        )
    return groups


def rollout_queries_batched(
    generate_batch: GenerateBatch,
    rows: Sequence[dict[str, Any]],
    *,
    component_id: str,
    group_size: int,
    max_turns: int,
    max_new: int,
    policy_version: str,
    seed: int,
    sample: bool,
    enc,
    searcher: RetrievalBackend | None = None,
    teacher_mode: bool = False,
    harness_mask: dict[str, bool] | None = None,
    temperature: float | None = None,
    search_k: int = 10,
    doc_store_k: int = 12,
    query_batch_size: int | None = None,
    doc_store_workers: int = DEFAULT_DOC_STORE_WORKERS,
    train_env: str = "local_legacy",
    train_session: Any | None = None,
    rollout_backend: str = "vllm",
    reasoning_effort: str | None = None,
    graph_index: Any | None = None,
    collection_mode: str = COLLECTION_MODE_RL_OPD,
    opd_loss: str = "",
) -> list[HybridRolloutGroup]:
    """Batch across queries and group members; step the env between turns.

    Document-store prep is chunked so the first vLLM generate_batch runs as
    soon as one micro-batch is ready. The next chunk is prepared on a
    background thread while the GPU rolls out the current chunk.
    """
    from trim.eval.local_search_env import curated_recall
    from trim.adapters.harness_profiles import is_harness_g
    from trim.training.four_cell_runtime import (
        doc_store_for_row,
        resolved_rollout_mask,
        snap_from_state,
        terminal_reward,
    )
    from trim.training.hf_rl_opd_client import episode_relative_advantages
    from trim.training.upstream_train_env import canonical_train_env, new_state_fn

    rows = list(rows)
    if not rows:
        return []
    harness_mask = resolved_rollout_mask(
        component_id, harness_mask=harness_mask, teacher_mode=teacher_mode
    )
    batch = resolved_query_batch_size(len(rows), group_size, query_batch_size)
    workers = max(1, int(doc_store_workers or 1))
    chunks = [rows[i : i + batch] for i in range(0, len(rows), batch)]
    temperature = 0.0 if not sample else float(temperature if temperature is not None else 1.0)
    g = is_harness_g(mask=harness_mask, component_ids=component_id)
    train_env = canonical_train_env(train_env) if not g else "local_legacy"
    new_state = new_state_fn(
        train_env=train_env,
        harness_mask=harness_mask,
        session=train_session,
        is_harness_g=g,
        graph_index=graph_index,
    )

    def prepare(chunk: Sequence[dict[str, Any]]) -> tuple[list[LiveEpisode], float]:
        t0 = time.perf_counter()
        episodes = _prepare_chunk_episodes(
            chunk,
            component_id=component_id,
            group_size=group_size,
            policy_version=policy_version,
            seed=seed,
            harness_mask=harness_mask,
            searcher=searcher,
            doc_store_k=doc_store_k,
            doc_store_workers=workers,
            new_state=new_state,
            doc_store_for_row=doc_store_for_row,
            teacher_mode=teacher_mode,
            collection_mode=collection_mode,
            opd_loss=opd_loss,
        )
        return episodes, time.perf_counter() - t0

    groups: list[HybridRolloutGroup] = []
    with ThreadPoolExecutor(max_workers=1) as prefetch:
        next_fut = prefetch.submit(prepare, chunks[0])
        for i, chunk in enumerate(chunks):
            episodes, prep_s = next_fut.result()
            if i + 1 < len(chunks):
                next_fut = prefetch.submit(prepare, chunks[i + 1])
            for ep in episodes:
                ep.reasoning_effort = reasoning_effort
                ep.eval_only = bool(ep.eval_only or policy_version == "eval")
            print(
                f"[rollout] chunk {i + 1}/{len(chunks)} queries={len(chunk)} "
                f"episodes={len(episodes)} prep={prep_s:.1f}s backend={rollout_backend}",
                flush=True,
            )
            _run_episode_turns(
                episodes,
                generate_batch,
                component_id=component_id,
                max_turns=max_turns,
                max_new=max_new,
                policy_version=policy_version,
                enc=enc,
                searcher=searcher,
                teacher_mode=teacher_mode,
                temperature=temperature,
                search_k=search_k,
                snap_from_state=snap_from_state,
                env_workers=workers,
            )
            groups.extend(
                _groups_from_episodes(
                    episodes,
                    chunk,
                    policy_version=policy_version,
                    max_turns=max_turns,
                    terminal_reward=terminal_reward,
                    curated_recall=curated_recall,
                )
            )
    return groups


def traces_from_groups(
    groups: Sequence[HybridRolloutGroup],
    rows: Sequence[dict[str, Any]],
    *,
    searcher: RetrievalBackend | None,
    leak_check_fn: Callable[[str], bool] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    from trim.eval.sr_opd_four_cell_eval import search_metrics

    by_q = {g.query_id: g for g in groups}
    traces: list[dict[str, Any]] = []
    leak = 0
    bm25_cache: dict[str, list[Any]] = {}
    for row in rows:
        group = by_q.get(str(row["query_id"]))
        stats = {}
        reward = 0.0
        if group is not None:
            ep_stats = list((group.trajectory_group or {}).get("episode_stats") or [])
            stats = ep_stats[0] if ep_stats else {}
            reward = float(stats.get("reward") or (group.terminal_rewards[0] if group.terminal_rewards else 0.0))
            prefix = ""
            if group.decision_points:
                prefix = str(group.decision_points[0].student_model_input or "")
            if leak_check_fn and leak_check_fn(prefix):
                leak += 1
            elif "compressed_teacher_view" in prefix or "VERIFY_RESULT_SECRET" in prefix:
                leak += 1
        search_q = str(row.get("query") or "")
        sm = (
            search_metrics(searcher, search_q, list(row.get("evidence_docids") or []), cache=bm25_cache)
            if searcher is not None
            else {}
        )
        if sm:
            sm = {
                **sm,
                "initial_bm25_recall_at_5": sm.get("evidence_recall_at_5"),
                "initial_bm25_recall_at_100": sm.get("evidence_recall_at_100"),
            }
        traces.append(
            {
                "query_id": row["query_id"],
                "tool_names": list(stats.get("names") or []),
                **trace_fields(stats),
                **sm,
            }
        )
    return traces, leak
