#!/usr/bin/env python3
"""Same-xi_t AUTO ON vs OFF fork/replay value on clean gpt-oss occupancy."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scape.eval.harmony_runtime import (
    build_continuation_prompt_ids,
    generate_tool_turn,
    load_harmony_enc,
    make_action,
    make_observation,
)
from scape.eval.local_search_env import apply_auto_populate, curated_recall, execute_tool, new_state, wm_text
from scape.training.clean_sft import load_causal_lm, load_jsonl


def _mean(xs: list[float]) -> float:
    return sum(xs) / max(1, len(xs))


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def _bootstrap_ci(xs: list[float], seed: int, n_boot: int = 400) -> tuple[float, float, float]:
    if not xs:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    means = []
    n = len(xs)
    for _ in range(n_boot):
        samp = [xs[rng.randrange(n)] for _ in range(n)]
        means.append(_mean(samp))
    means.sort()
    lo = means[int(0.025 * (n_boot - 1))]
    hi = means[int(0.975 * (n_boot - 1))]
    return _mean(xs), lo, hi


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


def load_doc_stores(raw_path: Path, keep_ids: set[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not raw_path.is_file() or not keep_ids:
        return out
    with raw_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = _unwrap(json.loads(line))
            qid = str(rec.get("query_id") or rec.get("qid") or "")
            if qid not in keep_ids or qid in out:
                continue
            store = rec.get("doc_store") or {}
            qtext = str(rec.get("query_text") or rec.get("query") or "")
            if store:
                out[qid] = {"doc_store": store, "query_text": qtext}
            if len(out) >= len(keep_ids):
                break
    return out


def query_text_of(row: dict[str, Any], stores: dict[str, dict[str, Any]]) -> str:
    qid = str(row.get("query_id") or "")
    if qid in stores and stores[qid].get("query_text"):
        return str(stores[qid]["query_text"])
    pr = row.get("prompt_reduced") or ""
    if "<query>" in pr:
        return pr.split("<query>", 1)[1].split("</query>", 1)[0].strip()
    return ""


def reconstruct_post_search(qtext: str, store: dict[str, Any], gold: list[str], first_name: str | None) -> dict[str, Any]:
    st = new_state(qtext, store)
    st["gold_ids"] = list(gold)
    name = first_name if first_name in {"fan_out_search", "search_corpus", "grep_corpus"} else "search_corpus"
    if name == "fan_out_search":
        args: dict[str, Any] = {"queries": [qtext]}
    elif name == "grep_corpus":
        args = {"pattern": qtext}
    else:
        args = {"query": qtext}
    st, _, _ = execute_tool(st, name, args)
    return st


def snapshot_roll(st: dict[str, Any], names: list[str | None]) -> dict[str, Any]:
    gold = st.get("gold_ids") or []
    rec = curated_recall(st, gold)
    legal = sum(
        1
        for n in names
        if n
        in {
            "fan_out_search",
            "search_corpus",
            "grep_corpus",
            "read_document",
            "review_docs",
            "curate",
            "verify",
            "end_search",
        }
    )
    return {
        "names": names,
        "recall": rec,
        "legal_rate": legal / max(1, len(names)),
        "ended": bool(st.get("ended")),
        "n_steps": len(names),
        "n_curated": len(st.get("curated") or {}),
    }


def rollout_from(
    model,
    enc,
    query: str,
    state: dict[str, Any],
    *,
    auto_on: bool,
    k_steps: int,
    max_new_tokens: int,
    seed_offset: int,
) -> dict[str, Any]:
    st = copy.deepcopy(state)
    acts: list[tuple[Any, Any]] = []
    names: list[str | None] = []
    for step in range(k_steps):
        try:
            ids = build_continuation_prompt_ids(
                query,
                actions_obs=acts,
                wm_text=wm_text(st, auto_on=auto_on and step == 0),
                enc=enc,
            )
        except Exception:
            break
        gen = generate_tool_turn(model, ids, max_new_tokens=max_new_tokens, enc=enc)
        p = gen["parsed"] or {}
        name = p.get("tool_name")
        args = p.get("arguments") or {}
        names.append(name)
        st, obs, _ok = execute_tool(st, name, args)
        try:
            acts.append((make_action(name or "search_corpus", args), make_observation(obs)))
        except Exception:
            break
        if st.get("ended"):
            break
    out = snapshot_roll(st, names)
    out["seed_offset"] = seed_offset
    return out


def score_row(teacher: dict[str, Any], student: dict[str, Any]) -> float:
    # Prefer gold recall; otherwise legal-tool + curated size proxy (explicitly not final-answer).
    if teacher.get("recall") is not None and student.get("recall") is not None:
        return float(teacher["recall"]) - float(student["recall"])
    return (
        0.5 * (float(teacher["legal_rate"]) - float(student["legal_rate"]))
        + 0.05 * (float(teacher["n_curated"]) - float(student["n_curated"]))
    )


def act_fields(act: Any) -> tuple[str | None, dict[str, Any]]:
    if not isinstance(act, dict):
        return None, {}
    return act.get("tool_name"), (act.get("arguments") or {})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--base-model", default="/data/ppnm/models/gpt-oss-20b")
    ap.add_argument("--states-jsonl", type=Path, required=True)
    ap.add_argument(
        "--raw-jsonl",
        type=Path,
        default=Path("/data/ppnm/Capability_Evolution/SCAPE/outputs/0814_clean_mechanism/data/hf_raw/sft_trajectories.jsonl"),
    )
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n-seeds", type=int, default=2)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=4)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-states", type=int, default=0)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(args.states_jsonl)
    rows = rows[args.shard_id :: args.n_shards]
    if args.max_states:
        rows = rows[: args.max_states]
    qids = {str(r.get("query_id") or "") for r in rows}
    qids.discard("")
    stores = load_doc_stores(args.raw_jsonl, qids)
    enc = load_harmony_enc()
    _, model = load_causal_lm(args.model_path, device_map=f"cuda:{args.gpu}", base_model=args.base_model)
    per = out / "AUTO_CLEAN_VALUE_PER_STATE.shard.jsonl"
    if per.exists():
        per.unlink()
    values: list[float] = []
    active_vals: list[float] = []
    n_store = 0
    t0 = time.time()
    for i, row in enumerate(rows):
        qid = str(row.get("query_id") or "")
        qtext = query_text_of(row, stores)
        gold = [str(x) for x in (row.get("gold_ids") or [])]
        t_name, t_args = act_fields(row.get("teacher_action"))
        s_name, s_args = act_fields(row.get("student_action"))
        store = (stores.get(qid) or {}).get("doc_store") or {}
        if store:
            n_store += 1
        seed_vals = []
        extra = max(0, args.k - 1)
        for sd in range(args.n_seeds):
            try:
                st_base = reconstruct_post_search(qtext, store, gold, row.get("first_search_name"))
                st_t = apply_auto_populate(copy.deepcopy(st_base), top_k=8)
                st_s = copy.deepcopy(st_base)
                st_t, _, _ = execute_tool(st_t, t_name, t_args)
                st_s, _, _ = execute_tool(st_s, s_name, s_args)
                if extra and qtext:
                    t_roll = rollout_from(
                        model,
                        enc,
                        qtext,
                        st_t,
                        auto_on=False,
                        k_steps=extra,
                        max_new_tokens=args.max_new_tokens,
                        seed_offset=sd,
                    )
                    s_roll = rollout_from(
                        model,
                        enc,
                        qtext,
                        st_s,
                        auto_on=False,
                        k_steps=extra,
                        max_new_tokens=args.max_new_tokens,
                        seed_offset=sd + 17,
                    )
                else:
                    t_roll = snapshot_roll(st_t, [t_name])
                    s_roll = snapshot_roll(st_s, [s_name])
                seed_vals.append(score_row(t_roll, s_roll))
            except Exception as exc:  # noqa: BLE001
                seed_vals.append(0.0)
                with (out / "errors.log").open("a", encoding="utf-8") as ef:
                    ef.write(f"i={i} qid={qid} sd={sd} {type(exc).__name__}: {exc}\n")
        val = _mean(seed_vals)
        values.append(val)
        if row.get("auto_effect_active"):
            active_vals.append(val)
        rec = {
            "snapshot_hash": row.get("snapshot_hash"),
            "query_id": qid,
            "auto_effect_active": row.get("auto_effect_active"),
            "value": val,
            "seed_values": seed_vals,
            "k": args.k,
            "teacher_tool": t_name,
            "student_tool": s_name,
            "has_doc_store": bool(store),
            "n_gold": len(gold),
        }
        with per.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        (out / "progress.json").write_text(
            json.dumps(
                {
                    "i": i + 1,
                    "n": len(rows),
                    "mean_value": _mean(values),
                    "elapsed_s": time.time() - t0,
                    "n_with_doc_store": n_store,
                },
                indent=2,
            )
            + "\n"
        )
    mean, lo, hi = _bootstrap_ci(values, seed=8170 + args.k)
    a_mean, a_lo, a_hi = _bootstrap_ci(active_vals, seed=8171 + args.k) if active_vals else (0.0, 0.0, 0.0)
    summary = {
        "k": args.k,
        "n": len(values),
        "n_with_doc_store": n_store,
        "mean_value": mean,
        "ci95_low": lo,
        "ci95_high": hi,
        "effect_active_n": len(active_vals),
        "effect_active_mean": a_mean,
        "effect_active_ci95_low": a_lo,
        "effect_active_ci95_high": a_hi,
        "replay_noise_proxy": 0.0,
        "shard_id": args.shard_id,
        "elapsed_s": time.time() - t0,
        "metric": "teacher_minus_student_continuation (recall if gold else legal/curated proxy)",
        "LOCAL_COMPAT_ONLY": True,
    }
    # replay noise = within-state seed disagreement (greedy → ~0)
    seed_stds = []
    with per.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            seed_stds.append(_std([float(x) for x in (rec.get("seed_values") or [])]))
    summary["replay_noise_proxy"] = _mean(seed_stds)
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
