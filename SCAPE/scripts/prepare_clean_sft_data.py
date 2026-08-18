#!/usr/bin/env python3
"""Audit + convert Harness-1 public SFT trajectories (no RL mix-in)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SCOPE = Path("/data/ppnm/Capability_Evolution/SCOPE")
OUT_DEFAULT = REPO / "outputs/0814_clean_mechanism"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _unwrap_traj(rec: dict[str, Any]) -> dict[str, Any]:
    """HF rows nest the official trajectory JSON in payload_json."""
    payload = rec.get("payload_json")
    if isinstance(payload, str) and payload.strip().startswith("{"):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            payload = None
    if isinstance(payload, dict):
        merged = {**rec, **payload}
        merged.pop("payload_json", None)
        if not merged.get("query_text"):
            merged["query_text"] = rec.get("query") or payload.get("query_text") or ""
        if not merged.get("dataset"):
            merged["dataset"] = rec.get("dataset_name") or payload.get("dataset_name")
        merged["stage"] = rec.get("stage") or payload.get("stage") or "sft"
        return merged
    if "turn_history" in rec and ("query_text" in rec or "query" in rec):
        out = dict(rec)
        out.setdefault("query_text", rec.get("query") or "")
        return out
    for key in ("trajectory", "content", "data", "record", "json"):
        val = rec.get(key)
        if isinstance(val, dict) and "turn_history" in val:
            return {**rec, **val}
        if isinstance(val, str) and val.strip().startswith("{"):
            try:
                obj = json.loads(val)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and "turn_history" in obj:
                return {**rec, **obj}
    return rec


def audit_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    from scape.training.clean_sft import CANONICAL_TOOLS, parse_tool_name

    n = len(records)
    stages = Counter(str(r.get("stage", "sft")) for r in records)
    n_rl = sum(1 for r in records if str(r.get("stage", "")).lower() == "rl")
    trajs = [_unwrap_traj(r) for r in records]
    n_with_turns = sum(1 for t in trajs if t.get("turn_history"))
    n_query = sum(1 for t in trajs if t.get("query_text") or t.get("query") or t.get("query_id"))
    recalls: list[float] = []
    tool_parse = 0
    tool_total = 0
    invalid = 0
    n_turns = 0
    tools = Counter()
    query_ids: list[str] = []
    datasets = Counter()
    schema_keys = Counter()
    for t in trajs:
        for k in t.keys():
            schema_keys[k] += 1
        qid = str(t.get("query_id") or t.get("qid") or t.get("task_id") or "")
        if qid:
            query_ids.append(qid)
        ds = str(t.get("dataset") or t.get("source_dataset") or t.get("split") or "unknown")
        datasets[ds] += 1
        rec = t.get("final_recall")
        if rec is None:
            rec = t.get("recall")
        try:
            recalls.append(float(rec))
        except (TypeError, ValueError):
            pass
        turns = t.get("turn_history") or t.get("turns") or []
        n_turns += len(turns)
        for turn in turns:
            tool_total += 1
            name = turn.get("tool_name") or parse_tool_name(json.dumps(turn, ensure_ascii=False))
            if name in CANONICAL_TOOLS:
                tool_parse += 1
                tools[name] += 1
            else:
                invalid += 1
                if name:
                    tools[f"other:{name}"] += 1
    parse_rate = tool_parse / max(1, tool_total)
    keep_min_recall = [r for r in recalls if r >= 0.1]
    return {
        "n_records": n,
        "stages": dict(stages),
        "n_rl_records_in_dump": n_rl,
        "n_with_turn_history": n_with_turns,
        "n_with_query": n_query,
        "n_turns": n_turns,
        "mean_turns": n_turns / max(1, n_with_turns),
        "tool_call_parse_rate": parse_rate,
        "invalid_tool_rate": invalid / max(1, tool_total),
        "tool_counts": dict(tools.most_common()),
        "n_query_ids": len(query_ids),
        "n_unique_query_ids": len(set(query_ids)),
        "datasets": dict(datasets),
        "schema_keys": dict(schema_keys.most_common()),
        "n_with_recall": len(recalls),
        "n_keep_min_recall_0.1": len(keep_min_recall),
        "mean_final_recall": (sum(recalls) / len(recalls)) if recalls else None,
        "rl_mixed": n_rl > 0,
        "contains_rl_only_records": n_rl > 0,
        "official_recipe": {
            "min_recall": 0.1,
            "epochs": 3,
            "lora_rank": 32,
            "lr": 5e-6,
            "source": "pat-jj/harness-1-train-data stage=sft",
        },
    }


def convert_with_harmony(
    trajs: list[dict[str, Any]],
    *,
    min_recall: float,
    max_length: int,
) -> list[dict[str, Any]]:
    if str(SCOPE) not in sys.path:
        sys.path.insert(0, str(SCOPE))
    from openai_harmony import (  # type: ignore
        Conversation,
        HarmonyEncodingName,
        Role,
        load_harmony_encoding,
    )
    from harness.ultra_core import (  # type: ignore
        RECENT_K,
        action_observation_to_messages,
        build_context,
        get_system_prompt,
    )
    from training.train_sft import _replay_trajectory  # type: ignore

    enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    examples: list[dict[str, Any]] = []
    skipped_recall = 0
    skipped_tok = 0
    for ti, raw in enumerate(trajs):
        traj = _unwrap_traj(raw)
        if str(traj.get("stage", "sft")).lower() == "rl":
            continue
        recall = traj.get("final_recall")
        try:
            recall_f = float(recall) if recall is not None else 1.0
        except (TypeError, ValueError):
            recall_f = 1.0
        if recall_f < min_recall:
            skipped_recall += 1
            continue
        query_text = traj.get("query_text") or traj.get("query") or ""
        turn_history = traj.get("turn_history") or []
        if not query_text or not turn_history:
            continue
        doc_store = traj.get("doc_store") or {}
        normalize_ids = bool(traj.get("normalize_ids", False))
        try:
            actions, observations, wm_snapshots, result_summaries, _wm = _replay_trajectory(
                query_text, turn_history, doc_store, normalize_ids
            )
        except Exception:
            skipped_tok += 1
            continue
        system_prompt = get_system_prompt(query_text)
        qid = str(traj.get("query_id") or traj.get("qid") or f"sft_{ti}")
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
            action_only = []
            for msg in target_msgs:
                if msg.author.role == Role.ASSISTANT:
                    action_only.append(msg)
                else:
                    break
            if not action_only:
                continue
            context_messages = list(context_conv.messages)
            full_messages = context_messages + action_only
            try:
                context_tokens = list(enc.render_conversation(Conversation(messages=context_messages)))
                full_tokens = list(
                    enc.render_conversation_for_training(Conversation(messages=full_messages))
                )
            except Exception:
                skipped_tok += 1
                continue
            n_context = len(context_tokens)
            if len(full_tokens) <= n_context:
                continue
            if len(full_tokens) > max_length:
                skipped_tok += 1
                continue
            resp_ids = full_tokens[n_context:]
            try:
                response_text = enc.decode_utf8(resp_ids)
            except Exception:
                response_text = ""
            tool_name = (turn_history[t_idx] or {}).get("tool_name")
            examples.append(
                {
                    "query_id": qid,
                    "turn_idx": t_idx,
                    "dataset": str(traj.get("dataset") or "unknown"),
                    "final_recall": recall_f,
                    "tool_name": tool_name,
                    "n_context": n_context,
                    "input_ids": [int(x) for x in full_tokens],
                    "response_text": response_text,
                    "source": "pat-jj/harness-1-train-data",
                    "stage": "sft",
                    "is_rl": False,
                    "legacy_scope_path_used": False,
                }
            )
    return examples


def convert_text_fallback(
    trajs: list[dict[str, Any]],
    *,
    min_recall: float,
) -> list[dict[str, Any]]:
    """Turn-level (prompt, action) pairs if Harmony replay is unavailable."""
    examples: list[dict[str, Any]] = []
    for ti, raw in enumerate(trajs):
        traj = _unwrap_traj(raw)
        if str(traj.get("stage", "sft")).lower() == "rl":
            continue
        try:
            recall_f = float(traj.get("final_recall") if traj.get("final_recall") is not None else 1.0)
        except (TypeError, ValueError):
            recall_f = 1.0
        if recall_f < min_recall:
            continue
        query_text = traj.get("query_text") or traj.get("query") or ""
        turns = traj.get("turn_history") or []
        if not query_text or not turns:
            continue
        qid = str(traj.get("query_id") or f"sft_{ti}")
        hist = []
        for t_idx, turn in enumerate(turns):
            name = turn.get("tool_name") or ""
            params = turn.get("params") or {}
            reasoning = (turn.get("reasoning") or "").strip()
            obs = turn.get("observation") or ""
            prompt = (
                "You are a retrieval subagent. Continue the Harness-1 tool trajectory.\n"
                f"Query: {query_text}\n"
                + ("\n".join(hist) + "\n" if hist else "")
                + "Emit the next tool call.\n"
            )
            resp_parts = []
            if reasoning:
                resp_parts.append(reasoning)
            resp_parts.append(f"to={name}")
            try:
                resp_parts.append(json.dumps(params, ensure_ascii=False))
            except TypeError:
                resp_parts.append(str(params))
            examples.append(
                {
                    "query_id": qid,
                    "turn_idx": t_idx,
                    "dataset": str(traj.get("dataset") or traj.get("dataset_name") or "unknown"),
                    "final_recall": recall_f,
                    "tool_name": name,
                    "prompt_text": prompt,
                    "response_text": "\n".join(resp_parts) + "\n",
                    "source": "pat-jj/harness-1-train-data",
                    "stage": "sft",
                    "is_rl": False,
                    "fallback": True,
                    "legacy_scope_path_used": False,
                }
            )
            hist.append(f"ASSISTANT: to={name} {json.dumps(params, ensure_ascii=False)[:400]}")
            hist.append(f"OBS: {str(obs)[:400]}")
            if len(hist) > 12:
                hist = hist[-12:]
    return examples


def write_audit_md(audit: dict[str, Any], path: Path) -> None:
    lines = [
        "# PUBLIC_SFT_AUDIT",
        "",
        "Source: `pat-jj/harness-1-train-data` **stage=sft only**.",
        "RL SEC records are audited for exclusion and **not** used in Clean SFT.",
        "",
        f"- records: {audit.get('n_records')}",
        f"- stages: `{audit.get('stages')}`",
        f"- RL records in dump: {audit.get('n_rl_records_in_dump')} (excluded)",
        f"- trajectories with turn_history: {audit.get('n_with_turn_history')}",
        f"- unique query/task ids: {audit.get('n_unique_query_ids')}",
        f"- turns: {audit.get('n_turns')} (mean {audit.get('mean_turns')})",
        f"- tool-call parse rate: {audit.get('tool_call_parse_rate')}",
        f"- invalid tool rate: {audit.get('invalid_tool_rate')}",
        f"- keep min_recall>=0.1: {audit.get('n_keep_min_recall_0.1')} / {audit.get('n_with_recall')}",
        f"- datasets: `{audit.get('datasets')}`",
        f"- schema keys: `{list((audit.get('schema_keys') or {}).keys())}`",
        "",
        "## Tool counts",
        "",
        "```json",
        json.dumps(audit.get("tool_counts"), indent=2),
        "```",
        "",
        "## Official recipe freeze",
        "",
        "```json",
        json.dumps(audit.get("official_recipe"), indent=2),
        "```",
        "",
        "synthetic/mock SFT: **not used**",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--raw-jsonl", type=Path, default=None)
    ap.add_argument("--min-recall", type=float, default=0.1)
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--skip-convert", action="store_true")
    args = ap.parse_args()
    out = args.out_dir
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    raw = args.raw_jsonl or (data / "hf_raw" / "sft_trajectories.jsonl")
    if not raw.is_file():
        print(json.dumps({"error": f"missing {raw}"}))
        return 1
    records = []
    with raw.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    audit = audit_records(records)
    audit["raw_path"] = str(raw)
    audit["raw_sha256"] = _sha256_file(raw)
    audit["raw_bytes"] = raw.stat().st_size
    (data / "PUBLIC_SFT_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    write_audit_md(audit, out / "PUBLIC_SFT_AUDIT.md")
    print("AUDIT", json.dumps({k: audit[k] for k in ("n_records", "tool_call_parse_rate", "n_rl_records_in_dump", "n_keep_min_recall_0.1")}))
    if args.skip_convert:
        return 0
    trajs = [_unwrap_traj(r) for r in records if str(r.get("stage", "sft")).lower() != "rl"]
    examples = convert_with_harmony(trajs, min_recall=args.min_recall, max_length=args.max_length)
    if not examples:
        print("HARMONY_CONVERT_EMPTY — using text fallback")
        examples = convert_text_fallback(trajs, min_recall=args.min_recall)
    train_path = data / "CLEAN_SFT_TRAIN.jsonl"
    from scape.training.clean_sft import write_jsonl

    n = write_jsonl(train_path, examples)
    meta = {
        "n_examples": n,
        "n_source_sft": len(trajs),
        "min_recall": args.min_recall,
        "max_length": args.max_length,
        "path": str(train_path),
        "legacy_scope_path_used": False,
        "LOCAL_COMPAT_ONLY": True,
        "used_rl": False,
    }
    (data / "CLEAN_SFT_CONVERT.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("CONVERT", json.dumps(meta))
    return 0 if n > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
