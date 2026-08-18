#!/usr/bin/env python3
"""Fresh on-policy AUTO first-search states on clean gpt-oss occupancy."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
SCOPE = Path("/data/ppnm/Capability_Evolution/SCOPE")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(SCOPE) not in sys.path:
    sys.path.insert(0, str(SCOPE))

from scape.eval.harmony_runtime import (
    build_continuation_prompt_ids,
    build_first_turn_prompt_ids,
    generate_tool_turn,
    load_harmony_enc,
    make_action,
    make_observation,
)
from scape.eval.local_search_env import apply_auto_populate, execute_tool, new_state, wm_text
from scape.training.clean_sft import load_causal_lm


def _unwrap(rec: dict[str, Any]) -> dict[str, Any]:
    payload = rec.get("payload_json")
    if isinstance(payload, str) and payload.strip().startswith("{"):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            payload = None
    if isinstance(payload, dict):
        merged = {**rec, **payload}
        merged.pop("payload_json", None)
        return merged
    return rec


def load_traj_index(raw_path: Path, keep_ids: set[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with raw_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            rec = _unwrap(json.loads(line))
            qid = str(rec.get("query_id") or rec.get("qid") or "")
            if qid not in keep_ids or qid in out:
                continue
            out[qid] = rec
            if len(out) >= len(keep_ids):
                break
    return out


def gold_ids(rec: dict[str, Any]) -> list[str]:
    g = rec.get("ground_truth_ids") or rec.get("document_ids_json") or []
    if isinstance(g, str):
        try:
            g = json.loads(g)
        except json.JSONDecodeError:
            g = []
    return [str(x) for x in g] if isinstance(g, list) else []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--base-model", default="/data/ppnm/models/gpt-oss-20b")
    ap.add_argument("--split-manifest", type=Path, required=True)
    ap.add_argument("--raw-jsonl", type=Path, required=True)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=4)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-states", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    args = ap.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    man = json.loads(args.split_manifest.read_text())
    qids = list(man.get("auto_query_ids") or [])
    shard = qids[args.shard_id :: args.n_shards]
    if args.max_states:
        shard = shard[: args.max_states]
    trajs = load_traj_index(args.raw_jsonl, set(shard))
    enc = load_harmony_enc()
    _, model = load_causal_lm(args.model_path, device_map=f"cuda:{args.gpu}", base_model=args.base_model)

    jsonl = out / "AUTO_CLEAN_RAW.shard.jsonl"
    if jsonl.exists():
        jsonl.unlink()
    n_ok = 0
    t0 = time.time()
    for i, qid in enumerate(shard):
        rec = trajs.get(qid)
        if not rec:
            continue
        qtext = str(rec.get("query_text") or rec.get("query") or "")
        store = rec.get("doc_store") or {}
        if not qtext or not store:
            continue
        st = new_state(qtext, store)
        # Student first action (reduced / no AUTO) — occupancy
        p0 = build_first_turn_prompt_ids(qtext, enc=enc)
        g0 = generate_tool_turn(model, p0, max_new_tokens=args.max_new_tokens, enc=enc)
        name0 = (g0["parsed"] or {}).get("tool_name")
        args0 = (g0["parsed"] or {}).get("arguments") or {}
        st1, obs1, ok1 = execute_tool(st, name0, args0)
        if not ok1 or name0 not in {"fan_out_search", "search_corpus", "grep_corpus"}:
            # still keep if search-like failed; skip empty
            if not st1.get("pool"):
                continue
        # Relevant window: after first search, before second search
        reduced = dict(st1)
        reduced["auto_seed"] = None
        full = apply_auto_populate(st1, top_k=8)
        act = make_action(name0 or "search_corpus", args0, reasoning=None)
        ob = make_observation(obs1)
        try:
            prompt_reduced_ids = build_continuation_prompt_ids(
                qtext, actions_obs=[(act, ob)], wm_text=wm_text(reduced, auto_on=False), enc=enc
            )
            prompt_full_ids = build_continuation_prompt_ids(
                qtext, actions_obs=[(act, ob)], wm_text=wm_text(full, auto_on=True), enc=enc
            )
        except Exception:
            continue
        g_s = generate_tool_turn(model, prompt_reduced_ids, max_new_tokens=args.max_new_tokens, enc=enc)
        g_t = generate_tool_turn(model, prompt_full_ids, max_new_tokens=args.max_new_tokens, enc=enc)
        prompt_reduced = enc.decode_utf8(prompt_reduced_ids)
        prompt_full = enc.decode_utf8(prompt_full_ids)
        teacher_text = g_t["text"]
        student_text = g_s["text"]
        snap_key = json.dumps(
            {"qid": qid, "step": st1.get("step"), "pool": sorted((st1.get("pool") or {}).keys())},
            sort_keys=True,
        )
        snap_hash = hashlib.sha256(snap_key.encode()).hexdigest()[:16]
        row = {
            "snapshot_hash": snap_hash,
            "query_id": qid,
            "step": st1.get("step"),
            "component_id": "auto_populate_first_search",
            "reduced_view": wm_text(reduced, auto_on=False),
            "full_structured_view": wm_text(full, auto_on=True),
            "prompt_reduced": prompt_reduced,
            "prompt_full": prompt_full,
            "student_action": g_s["parsed"],
            "teacher_action": g_t["parsed"],
            "response_text": teacher_text,
            "student_response_text": student_text,
            "auto_effect_active": bool(full.get("auto_seed")),
            "first_search_name": name0,
            "teacher_does_not_step_environment": True,
            "no_future_reward": True,
            "no_gold_in_privilege": True,
            "resampled_duplicate": False,
            "student_inference_privilege": False,
            "LOCAL_COMPAT_ONLY": True,
            "legacy_scope_path_used": False,
            "gold_ids": gold_ids(rec),
            "n_pool": len(st1.get("pool") or {}),
            "n_curated_full": len(full.get("curated") or {}),
            "n_curated_reduced": len(reduced.get("curated") or {}),
        }
        with jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        n_ok += 1
        (out / "progress.json").write_text(
            json.dumps(
                {
                    "shard": args.shard_id,
                    "i": i + 1,
                    "n": len(shard),
                    "kept": n_ok,
                    "elapsed_s": time.time() - t0,
                    "last_qid": qid,
                },
                indent=2,
            )
            + "\n"
        )
    summary = {
        "shard_id": args.shard_id,
        "n_assigned": len(shard),
        "n_kept": n_ok,
        "elapsed_s": time.time() - t0,
        "path": str(jsonl),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "DONE").write_text("ok\n")
    print(json.dumps(summary, indent=2))
    del model
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
