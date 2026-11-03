"""Probe C2 closed-loop confirmation (§10): full H0 episodes on C_select.

Conditions (deployment is H0 unless noted):

    no_update    θ0                  on H0      (baseline J)
    runtime_m    θ0                  on H_m     (Probe-A style external gain)
    harnessopd_m θ_m^8 (trim+m)      on H0      (component guidance)
    rl_only      θ_RL^8              on H0      (generic training gain)
    same_view    θ_same^8 (placebo)  on H0      (mechanism without privilege)

G_m^8 = J(harnessopd_m) - J(no_update); E_m^8 = J(harnessopd_m) - J(rl_only),
paired by query with bootstrap CI. J = final-answer-recall per full episode.

    # one condition (per GPU worker in the eval entry)
    python -m trim.probes.probe_c_c2_eval run --model-path /tmp/qwen35-9b \
        --states-out /path/collection_out --condition harnessopd_m \
        --adapter /path/checkpoints/scape_seed/step_00008/adapter --mask h0 --out OUT/c2
    # aggregate
    python -m trim.probes.probe_c_c2_eval compare --out OUT/c2
    # allowlist of training pool ids minus C_select/C_audit (C2 train exclusivity)
    python -m trim.probes.probe_c_c2_eval make-train-ids --states-out ... --out train_ids.txt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

_TRIM = Path(__file__).resolve().parents[2]
if str(_TRIM) not in sys.path:
    sys.path.insert(0, str(_TRIM))

from trim.state.snapshot import stable_hash  # noqa: E402

CONDITIONS = ("no_update", "runtime_m", "harnessopd_m", "rl_only", "same_view")
C2_SCHEMA = "probe_c_c2_v1"


def _load_assignment(states_out: Path) -> dict[str, Any]:
    path = states_out / "SPLIT_ASSIGNMENT.json"
    if not path.is_file():
        raise RuntimeError(f"missing {path}; collection must be complete")
    return json.loads(path.read_text(encoding="utf-8"))


def make_train_ids(states_out: Path, out_path: Path) -> Path:
    """Allowlist = full SEC pool minus C_select/C_audit (C2 train/eval exclusivity)."""
    from trim.eval.sec_corpus import (
        default_sec_corpus_root,
        default_sec_rl_data,
        load_sec_rl_queries,
    )

    payload = _load_assignment(states_out)
    assignment: dict[str, str] = payload["assignment"]
    rows, _src = load_sec_rl_queries(
        default_sec_rl_data(), n_queries=None, query_file=None,
        corpus_root=default_sec_corpus_root(),
    )
    forbidden = {q for q, s in assignment.items() if s in {"C_select", "C_audit"}}
    keep = [str(r["query_id"]) for r in rows if str(r["query_id"]) not in forbidden]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(keep) + "\n", encoding="utf-8")
    print(
        f"[probe_c_c2] train ids: keep={len(keep)} excluded_eval={len(forbidden)} -> {out_path}"
    )
    return out_path


def _h_m_mask() -> dict[str, bool]:
    from trim.adapters.role_masks import eval_mask_for_ids

    return eval_mask_for_ids(
        ["auto_populate_first_search", "chunk_neighbors"], harness="Harness-1"
    )


def run_condition(args: argparse.Namespace) -> int:
    from trim.adapters.components import zero_mask
    from trim.eval.model_tokenizer import load_model_encoding
    from trim.eval.sec_corpus import (
        attach_sec_doc_stores,
        default_sec_corpus_root,
        default_sec_rl_data,
        load_sec_rl_queries,
    )
    from trim.training.batched_env_rollout import rollout_queries_batched
    from trim.training.vllm_hybrid import VLLMGenerateClient

    condition = str(args.condition)
    if condition not in CONDITIONS:
        raise RuntimeError(f"unknown condition {condition!r}; want {CONDITIONS}")
    states_out = Path(args.states_out)
    payload = _load_assignment(states_out)
    assignment: dict[str, str] = payload["assignment"]
    split = str(args.split)
    rows, _src = load_sec_rl_queries(
        default_sec_rl_data(), n_queries=None, query_file=None,
        corpus_root=default_sec_corpus_root(),
    )
    attach_sec_doc_stores(rows, corpus_root=default_sec_corpus_root())
    panel = [r for r in rows if assignment.get(str(r["query_id"])) == split]
    print(f"[probe_c_c2] condition={condition} split={split} n_queries={len(panel)}")
    if not panel:
        raise RuntimeError(f"empty panel for split={split}")

    mask = _h_m_mask() if args.mask == "hm" else zero_mask("Harness-1")
    enc = load_model_encoding(str(args.model_path), max_model_len=int(args.max_model_len))
    # Live SEC BM25 searcher is REQUIRED: doc_store_for_row(None) yields an empty
    # pool and every episode silently scores recall=0 (empty environment).
    from trim.training.four_cell_runtime import open_train_retrieval

    searcher = open_train_retrieval(
        argparse.Namespace(train_data="sec", smoke=False, sec_corpus_root=None),
        train_rows=panel,
    )
    session_dir = Path(args.out) / "vllm_sessions" / f"{condition}_{args.mask}"
    client = VLLMGenerateClient(
        model_path=str(args.model_path),
        session_dir=session_dir,
        tensor_parallel_size=1,
        max_model_len=int(args.max_model_len),
        lora_path=str(args.adapter) if args.adapter else None,
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        enforce_eager=True,
        startup_timeout_s=3600.0,
        generate_timeout_s=3600.0,
    )
    client.start()
    try:
        groups = rollout_queries_batched(
            client.generate_batch,
            panel,
            component_id="zero",
            group_size=1,
            max_turns=int(args.max_turns),
            max_new=int(args.max_new),
            policy_version=f"c2:{condition}",
            seed=int(args.seed),
            sample=False,
            enc=enc,
            searcher=searcher,
            teacher_mode=False,
            harness_mask=mask,
            search_k=int(args.search_k),
            doc_store_k=int(args.doc_store_k),
            train_env="local_legacy",
            rollout_backend="vllm",
            collection_mode="rl",
            opd_loss="",
            rl_reward="final-answer-recall",
            enable_rl=True,
        )
    finally:
        client.close()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_out: list[dict[str, Any]] = []
    for group in groups:
        stats = ((group.trajectory_group or {}).get("episode_stats") or [{}])[0]
        rows_out.append(
            {
                "schema": C2_SCHEMA,
                "condition": condition,
                "deploy_mask": args.mask,
                "adapter": str(args.adapter or ""),
                "query_id": str(group.query_id),
                "reward": float((group.terminal_rewards or [0.0])[0]),
                "task_recall": float(stats.get("task_recall") or 0.0),
                "n_turns": int(stats.get("n_turns") or 0),
                "n_tool_calls": int(stats.get("n_tool_calls") or 0),
                "n_structurally_valid": int(stats.get("n_structurally_valid") or 0),
                "n_valids": int(stats.get("n_valids") or 0),
                "ended": bool(stats.get("ended")),
            }
        )
    path = out / f"PER_QUERY_{condition}_{args.mask}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows_out:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    mean_reward = sum(r["reward"] for r in rows_out) / len(rows_out)
    print(
        f"[probe_c_c2] {condition}/{args.mask}: n={len(rows_out)} "
        f"J(final-answer-recall)={mean_reward:.4f} -> {path}"
    )
    return 0


def compare(args: argparse.Namespace) -> int:
    from trim.eval.paired_bootstrap import paired_query_stats

    out = Path(args.out)

    def _load(cond: str, mask: str) -> dict[str, dict[str, Any]]:
        path = out / f"PER_QUERY_{cond}_{mask}.jsonl"
        if not path.is_file():
            return {}
        return {
            json.loads(l)["query_id"]: json.loads(l)
            for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip()
        }

    panels = {
        "no_update": _load("no_update", "h0"),
        "runtime_m": _load("runtime_m", "hm"),
        "harnessopd_m": _load("harnessopd_m", "h0"),
        "rl_only": _load("rl_only", "h0"),
        "same_view": _load("same_view", "h0"),
    }
    summary: dict[str, Any] = {
        "schema": C2_SCHEMA,
        "means": {
            c: (
                sum(r["reward"] for r in rows.values()) / len(rows)
                if rows
                else "N/A"
            )
            for c, rows in panels.items()
        },
        "n_queries": {c: len(rows) for c, rows in panels.items()},
        "cost": {
            c: {
                "mean_turns": (
                    sum(r["n_turns"] for r in rows.values()) / len(rows) if rows else "N/A"
                ),
                "mean_tool_calls": (
                    sum(r["n_tool_calls"] for r in rows.values()) / len(rows)
                    if rows
                    else "N/A"
                ),
                "parse_valid_rate": (
                    sum(r["n_structurally_valid"] for r in rows.values())
                    / max(1, sum(r["n_valids"] for r in rows.values()))
                    if rows
                    else "N/A"
                ),
            }
            for c, rows in panels.items()
        },
    }

    def _pair(a: str, b: str, name: str) -> None:
        pa, pb = panels.get(a) or {}, panels.get(b) or {}
        if not pa or not pb:
            summary[name] = {"note": "missing condition panel"}
            return
        stats = paired_query_stats(
            {q: {"recall": r["reward"]} for q, r in pa.items()},
            {q: {"recall": r["reward"]} for q, r in pb.items()},
            metric="recall",
            n_boot=2000,
            seed=42,
        )
        summary[name] = {
            "mean_delta": stats["mean_delta"],
            "ci95": stats["bootstrap_ci_95"],
            "win": stats["win"],
            "loss": stats["loss"],
            "tie": stats["tie"],
            "n": stats["n"],
            "definition": f"J({b}) - J({a})",
        }

    _pair("no_update", "harnessopd_m", "G_m_8")
    _pair("rl_only", "harnessopd_m", "E_m_8")
    _pair("no_update", "runtime_m", "runtime_gain")
    _pair("no_update", "rl_only", "rl_gain")
    _pair("same_view", "harnessopd_m", "privilege_gain")
    summary["note"] = (
        "G_m^8 = J(theta_m^8, H0) - J(theta0, H0); E_m^8 = J(HarnessOPD) - J(RL-only). "
        "C2 restarts from theta0 (C1 weights not carried). Paired by query, 2000 bootstraps."
    )
    (out / "PROBE_C_C2.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True)[:4000])
    print(f"[probe_c_c2] wrote {out / 'PROBE_C_C2.json'}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="probe_c_c2_eval")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("--model-path", required=True)
    p_run.add_argument("--states-out", required=True)
    p_run.add_argument("--condition", required=True, choices=list(CONDITIONS))
    p_run.add_argument("--adapter", default="")
    p_run.add_argument("--mask", choices=["h0", "hm"], default="h0")
    p_run.add_argument("--split", default="C_select")
    p_run.add_argument("--out", required=True)
    p_run.add_argument("--max-turns", type=int, default=40)
    p_run.add_argument("--max-new", type=int, default=2048)
    p_run.add_argument("--max-model-len", type=int, default=8192)
    p_run.add_argument("--search-k", type=int, default=10)
    p_run.add_argument("--doc-store-k", type=int, default=12)
    p_run.add_argument("--seed", type=int, default=20261031)
    p_run.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("--out", required=True)
    p_ids = sub.add_parser("make-train-ids")
    p_ids.add_argument("--states-out", required=True)
    p_ids.add_argument("--out", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.cmd == "run":
        return run_condition(args)
    if args.cmd == "compare":
        return compare(args)
    make_train_ids(Path(args.states_out), Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
