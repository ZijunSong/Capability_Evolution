"""Build HuggingFace SFT examples from Harness-1 ultra_v3 trajectories.

Replays WorkingMemory the same way as ``external/harness-1/training/train_sft.py``,
then tokenizes with the gpt-oss Harmony encoding. Output is
``{input_ids, n_context}`` so local LoRA training does not need Tinker.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence

from trim.training.sft_runtime import (
    HARNESS1_SFT_MAX_LENGTH,
    HARNESS1_SFT_MIN_RECALL,
    apply_sft_v8d_env,
    harness1_pythonpath,
)

HARMONY_STOP_TOKENS = {200002, 200012}  # <|return|>, <|call|>


def prepare_harness1_sft_imports() -> None:
    apply_sft_v8d_env()
    for item in harness1_pythonpath().split(os.pathsep):
        if item and item not in sys.path:
            sys.path.insert(0, item)


class _PlaceholderTool:
    def __init__(self, schema: Any) -> None:
        self.tool_schema = schema


def _tool_registry() -> dict[str, _PlaceholderTool]:
    from harness.tools import GREP_CORPUS_SCHEMA, READ_DOCUMENT_SCHEMA, SEARCH_CORPUS_SCHEMA
    from harness.ultra_core import (
        CURATE_SCHEMA,
        END_SEARCH_SCHEMA,
        FAN_OUT_SEARCH_SCHEMA,
        REVIEW_DOCS_SCHEMA,
        VERIFY_SCHEMA,
        V8D_VERIFY_TOOL,
    )

    registry = {
        "fan_out_search": _PlaceholderTool(FAN_OUT_SEARCH_SCHEMA),
        "search_corpus": _PlaceholderTool(SEARCH_CORPUS_SCHEMA),
        "grep_corpus": _PlaceholderTool(GREP_CORPUS_SCHEMA),
        "read_document": _PlaceholderTool(READ_DOCUMENT_SCHEMA),
        "review_docs": _PlaceholderTool(REVIEW_DOCS_SCHEMA),
        "curate": _PlaceholderTool(CURATE_SCHEMA),
        "end_search": _PlaceholderTool(END_SEARCH_SCHEMA),
    }
    if V8D_VERIFY_TOOL:
        registry["verify"] = _PlaceholderTool(VERIFY_SCHEMA)
    return registry


def _turn_to_action(turn: dict[str, Any], registry: dict[str, _PlaceholderTool]):
    from harness.trajectory import ActionBuilder

    tool_name = turn.get("tool_name", "")
    params = turn.get("params", {})
    reasoning = turn.get("reasoning", "")
    builder = ActionBuilder()
    if reasoning:
        builder.add_reasoning(reasoning)
    tool = registry.get(tool_name)
    if tool:
        builder.add_tool_call(tool=tool, params=params, source=f"functions.{tool_name}")
    return builder.build()


def _turn_to_observation(turn: dict[str, Any]):
    from harness.trajectory import ObservationBuilder

    obs_text = turn.get("observation", "")
    tool_name = turn.get("tool_name", "")
    builder = ObservationBuilder()
    builder.add_observation(obs_text, source=f"functions.{tool_name}")
    return builder.build()


def replay_trajectory(trajectory: dict[str, Any]):
    """Replay one ultra_v3 trajectory through WorkingMemory (same as train_sft.py)."""
    from harness.ultra_core import (
        V8D_AUTO_POPULATE_FIRST_SEARCH,
        V8D_IMPORTANCE_TAGGING,
        WorkingMemory,
        auto_populate_from_first_search,
        build_result_summary,
        parse_doc_ids_from_observation,
    )

    query_text = trajectory["query_text"]
    turn_history = trajectory["turn_history"]
    doc_store_data = trajectory.get("doc_store", {}) or {}
    normalize_ids = trajectory.get("normalize_ids", False)
    registry = _tool_registry()

    wm = WorkingMemory(query_text, normalize_ids=normalize_ids)
    actions: list[Any] = []
    observations: list[Any] = []
    wm_snapshots: list[str] = [wm.to_text()]
    result_summaries: list[str] = []
    tool_types_used: set[str] = set()
    turns_since_curate = 0
    first_search_done = False

    for turn in turn_history:
        tool_name = turn.get("tool_name", "")
        params = turn.get("params", {}) or {}
        obs_text = turn.get("observation", "")
        tool_types_used.add(tool_name)
        pool_size_before = wm.get_pool_size()

        if tool_name in ("fan_out_search", "search_corpus", "grep_corpus", "read_document"):
            doc_ids = parse_doc_ids_from_observation(obs_text)
            doc_texts = {}
            for did in doc_ids:
                if did in doc_store_data:
                    snippet = doc_store_data[did].get("snippet", "")
                    doc_texts[did] = snippet
            pool_before = wm.get_pool_size()
            wm.add_to_pool(doc_ids, doc_texts if doc_texts else None)
            num_new = wm.get_pool_size() - pool_before
            if tool_name == "fan_out_search":
                queries = params.get("queries", [])
                q_summary = "; ".join(str(q)[:30] for q in queries[:3])
                wm.add_search_record("fan_out", q_summary, len(doc_ids), num_new=num_new)
            elif tool_name == "search_corpus":
                wm.add_search_record(
                    "search", params.get("query", "")[:50], len(doc_ids), num_new=num_new
                )
            elif tool_name == "grep_corpus":
                wm.add_search_record(
                    "grep", params.get("pattern", "")[:50], len(doc_ids), num_new=num_new
                )
            elif tool_name == "read_document":
                wm.add_search_record("read", params.get("doc_id", ""), len(doc_ids), num_new=num_new)
            if (
                V8D_AUTO_POPULATE_FIRST_SEARCH
                and not first_search_done
                and tool_name in ("fan_out_search", "search_corpus")
                and doc_ids
            ):
                auto_populate_from_first_search(wm, doc_ids)
                first_search_done = True
        elif tool_name == "review_docs":
            doc_ids = params.get("doc_ids", [])
            wm.add_search_record("review", ", ".join(doc_ids[:3]), len(doc_ids))
        elif tool_name == "verify":
            v_doc_ids = params.get("doc_ids", []) or []
            claim = str(params.get("claim", ""))[:50]
            wm.add_search_record("verify", claim, len(v_doc_ids), num_new=0)
        elif tool_name == "curate":
            add_ids = params.get("add_ids", [])
            remove_ids = params.get("remove_ids", [])
            importance = params.get("importance") if V8D_IMPORTANCE_TAGGING else None
            if not isinstance(add_ids, list):
                add_ids = [add_ids] if add_ids else []
            if not isinstance(remove_ids, list):
                remove_ids = [remove_ids] if remove_ids else []
            wm.curate(add_ids, remove_ids, importance=importance)
            turns_since_curate = 0

        if tool_name != "curate" and tool_name != "end_search":
            turns_since_curate += 1

        summary = build_result_summary(
            obs_text=obs_text,
            tool_names=[tool_name],
            wm=wm,
            turns_since_curate=turns_since_curate,
            tool_types_used=tool_types_used,
            current_turn=len(actions) + 1,
            pool_size_before=pool_size_before,
        )
        result_summaries.append(summary)
        actions.append(_turn_to_action(turn, registry))
        observations.append(_turn_to_observation(turn))
        wm.advance_turn()
        wm_snapshots.append(wm.to_text())

    return actions, observations, wm_snapshots, result_summaries, wm


def _is_assistant_message(msg: Any) -> bool:
    author = getattr(msg, "author", None)
    role = getattr(author, "role", None) if author is not None else getattr(msg, "role", None)
    value = getattr(role, "value", None) or str(role or "")
    return "assistant" in value.lower()


def _as_conversation(messages: Sequence[Any], template: Any) -> Any:
    ctor = type(template)
    try:
        return ctor(messages=list(messages))
    except TypeError:
        return SimpleNamespace(messages=list(messages))


def _render_tokens(enc: Any, conversation: Any, *, training: bool) -> list[int]:
    if training and hasattr(enc, "render_conversation_for_training"):
        tokens = enc.render_conversation_for_training(conversation)
    else:
        tokens = enc.render_conversation(conversation)
    return [int(t) for t in list(tokens)]


def load_harmony_encoder(model_name: str | None = None) -> Any:
    """Prefer the local gpt-oss tokenizer; fall back to openai_harmony."""
    if model_name:
        from trim.eval.harmony_hf_encoding import TokenizerHarmonyEncoding

        return TokenizerHarmonyEncoding.from_pretrained(str(model_name))
    from openai_harmony import HarmonyEncodingName, load_harmony_encoding

    return load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


def examples_from_trajectory(
    trajectory: dict[str, Any],
    enc: Any,
    *,
    max_length: int = HARNESS1_SFT_MAX_LENGTH,
    min_recall: float = HARNESS1_SFT_MIN_RECALL,
) -> list[dict[str, Any]]:
    from harness.ultra_core import (
        RECENT_K,
        action_observation_to_messages,
        build_context,
        get_system_prompt,
    )

    query_text = trajectory["query_text"]
    turn_history = trajectory.get("turn_history") or []
    final_recall = float(trajectory.get("final_recall") or 0.0)
    if final_recall < min_recall or not turn_history:
        return []

    actions, observations, wm_snapshots, result_summaries, _wm = replay_trajectory(trajectory)
    system_prompt = get_system_prompt(query_text)
    out: list[dict[str, Any]] = []
    query_id = str(trajectory.get("query_id") or "")

    for t_idx in range(len(actions)):
        n_turns = t_idx
        if n_turns <= RECENT_K:
            wm_text = None
            recent_actions = actions[:t_idx]
            recent_obs = observations[:t_idx]
            recent_summaries = result_summaries[:t_idx]
        else:
            wm_boundary = n_turns - RECENT_K
            wm_text = wm_snapshots[wm_boundary]
            recent_actions = actions[wm_boundary:t_idx]
            recent_obs = observations[wm_boundary:t_idx]
            recent_summaries = result_summaries[wm_boundary:t_idx]

        context_conv = build_context(
            system_prompt, wm_text, recent_actions, recent_obs, recent_summaries
        )
        target_msgs = action_observation_to_messages(
            actions[t_idx], observations[t_idx], compress=False
        )
        action_only_msgs = []
        for msg in target_msgs:
            if _is_assistant_message(msg):
                action_only_msgs.append(msg)
            else:
                break
        if not action_only_msgs:
            continue

        context_messages = list(context_conv.messages)
        full_messages = context_messages + action_only_msgs
        context_conversation = _as_conversation(context_messages, context_conv)
        full_conversation = _as_conversation(full_messages, context_conv)
        try:
            context_tokens = _render_tokens(enc, context_conversation, training=False)
            full_tokens = _render_tokens(enc, full_conversation, training=True)
        except Exception as exc:  # noqa: BLE001
            print(
                json.dumps(
                    {
                        "event": "sft_tokenize_skip",
                        "query_id": query_id,
                        "turn": t_idx,
                        "error": str(exc)[:200],
                    }
                ),
                flush=True,
            )
            continue

        n_context = len(context_tokens)
        n_target = len(full_tokens) - n_context
        if n_target <= 0:
            continue
        if len(full_tokens) > max_length:
            continue
        if not any(t in HARMONY_STOP_TOKENS for t in full_tokens[-5:]):
            continue
        out.append(
            {
                "input_ids": full_tokens,
                "n_context": n_context,
                "query_id": query_id,
                "turn_idx": t_idx,
            }
        )
    return out


def load_trajectory_json_dir(data_dir: Path | str) -> list[dict[str, Any]]:
    trajectories: list[dict[str, Any]] = []
    for path in sorted(Path(data_dir).glob("*.json")):
        if path.name == "MANIFEST.json":
            continue
        try:
            trajectories.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            print(
                json.dumps({"event": "sft_load_skip", "file": str(path), "error": str(exc)[:200]}),
                flush=True,
            )
    return trajectories


def build_hf_sft_examples(
    data_dir: Path | str,
    *,
    enc: Any | None = None,
    model_name: str | None = None,
    max_length: int = HARNESS1_SFT_MAX_LENGTH,
    min_recall: float = HARNESS1_SFT_MIN_RECALL,
    progress_fn: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prepare_harness1_sft_imports()
    encoder = enc if enc is not None else load_harmony_encoder(model_name)
    trajectories = load_trajectory_json_dir(data_dir)
    examples: list[dict[str, Any]] = []
    n_kept_traj = 0
    t0 = time.time()
    n_traj = len(trajectories)
    log_every = max(1, n_traj // 20) if n_traj else 1
    for i, traj in enumerate(trajectories):
        built = examples_from_trajectory(
            traj, encoder, max_length=max_length, min_recall=min_recall
        )
        if built:
            n_kept_traj += 1
            examples.extend(built)
        if progress_fn and ((i + 1) % log_every == 0 or (i + 1) == n_traj):
            progress_fn(
                {
                    "event": "building_datums",
                    "progress": f"{i + 1}/{n_traj}",
                    "datums_so_far": len(examples),
                    "elapsed_s": round(time.time() - t0, 1),
                }
            )
    meta = {
        "n_trajectories": n_traj,
        "n_trajectories_kept": n_kept_traj,
        "n_examples": len(examples),
        "max_length": int(max_length),
        "min_recall": float(min_recall),
        "avg_per_traj": (len(examples) / max(n_kept_traj, 1)),
        "total_build_s": round(time.time() - t0, 1),
        "encoder": type(encoder).__name__,
    }
    return examples, meta
