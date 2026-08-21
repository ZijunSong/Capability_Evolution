#!/usr/bin/env python3
"""Corrected content_dedup same-state K4/K8 reward fork.

Contract:
  - same xi_t rows from corrected high-redundancy content_dedup collection
  - Teacher/Full branch applies the dedup-on canonical projected action
  - Student/Reduced branch keeps content_dedup off and acts on the unfiltered pool
  - both continuations use the same reduced policy; no full-harness takeover

This is a deterministic Harness-1 state fork over recorded real_harness1 rows. It
uses the recorded dedup-on pool delta and search-result ids rather than
fabricating duplicate triggers.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATES = ROOT.parent / "SCAPE-EasyOPD/outputs/component_sweep_0818/h100_2/content_dedup_corrected_high_redundancy_v3/TRAIN_STATES_5K.jsonl"
DEFAULT_OUT = ROOT / "outputs/0820_content_dedup_corrected_reward_fork"


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256sums(root: Path) -> None:
    lines = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root)}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def stable_float(key: str) -> float:
    return int(hashlib.sha256(key.encode()).hexdigest()[:13], 16) / float(16**13 - 1)


def canonical_doc_id(doc_id: str) -> str:
    """Map corrected synthetic duplicate ids to the dedup survivor id.

    The corrected collector emits 24 docs per duplicate-heavy search turn. The
    real dedup hook keeps two survivors per turn: idx 000 and idx 008, which is
    also reflected in event_payload.pool_ids_teacher_post. We derive the mapping
    from the id suffix instead of text so the audit is reproducible from TRAIN_STATES_5K.
    """
    s = str(doc_id)
    if "-t" not in s or "-" not in s:
        return s
    try:
        prefix, idx_s = s.rsplit("-", 1)
        idx = int(idx_s)
    except ValueError:
        return s
    survivor = 0 if idx < 8 else 8
    return f"{prefix}-{survivor:03d}"


def unique_preserve(xs: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(x) for x in xs if str(x)))


def action_ids(action: dict[str, Any]) -> tuple[list[str], list[str]]:
    args = action.get("arguments") or {}
    return [str(x) for x in args.get("add_ids") or []], [str(x) for x in args.get("remove_ids") or []]


@dataclass
class ForkState:
    query_id: str
    state_uid: str
    pool_ids: list[str]
    curated_ids: list[str]
    search_result_ids: list[str]
    branch: str
    history: list[dict[str, Any]] = field(default_factory=list)
    cost: int = 0

    def clone(self, branch: str) -> "ForkState":
        return ForkState(
            query_id=self.query_id,
            state_uid=self.state_uid,
            pool_ids=list(self.pool_ids),
            curated_ids=list(self.curated_ids),
            search_result_ids=list(self.search_result_ids),
            branch=branch,
            history=list(self.history),
            cost=self.cost,
        )

    def execute(self, action: dict[str, Any], *, phase: str) -> None:
        name = str(action.get("name") or "end_search")
        add, remove = action_ids(action)
        if name == "curate":
            remove_set = set(remove)
            self.curated_ids = [x for x in self.curated_ids if x not in remove_set]
            for did in add:
                if did in self.pool_ids and did not in self.curated_ids:
                    self.curated_ids.append(did)
            self.cost += 1
        elif name == "read_document":
            self.cost += 1
        elif name in {"review_docs", "search_corpus", "fan_out_search", "grep_corpus"}:
            self.cost += 1
        elif name == "end_search":
            self.cost += 0
        else:
            self.cost += 1
        self.history.append({"phase": phase, "action": {"name": name, "arguments": action.get("arguments") or {}}, "metrics": self.metrics()})

    def reduced_policy_action(self, *, step: int) -> dict[str, Any]:
        """Reduced continuation policy with no dedup/canonical privilege.

        It greedily curates currently visible uncurated docs in pool order. This
        intentionally does not canonicalize duplicate ids, matching the component-off
        information boundary. The same policy is used after the first fork action
        in both branches.
        """
        candidates = [x for x in self.pool_ids if x not in self.curated_ids]
        if not candidates:
            return {"name": "end_search", "arguments": {}}
        # Alternate widths to avoid a degenerate one-step-only result while staying deterministic.
        width = 2 if step % 2 == 0 else 1
        return {"name": "curate", "arguments": {"add_ids": candidates[:width], "remove_ids": []}}

    def metrics(self) -> dict[str, float]:
        curated = unique_preserve(self.curated_ids)
        canonical_curated = [canonical_doc_id(x) for x in curated]
        unique_canon = set(canonical_curated)
        duplicate_curated = max(0, len(curated) - len(unique_canon))
        pool_canon = [canonical_doc_id(x) for x in self.pool_ids]
        pool_redundancy = max(0, len(self.pool_ids) - len(set(pool_canon))) / max(1, len(self.pool_ids))
        curated_redundancy = duplicate_curated / max(1, len(curated))
        coverage = len(unique_canon) / max(1, len(set(pool_canon)))
        # Reward favors canonical evidence coverage and penalizes redundant curation,
        # redundant pool burden, and tool cost. The scale follows prior SCAPE fork
        # objectives: higher is better, and T-S is the reported component utility.
        objective = 0.70 * coverage - 0.20 * curated_redundancy - 0.05 * pool_redundancy - 0.01 * self.cost
        return {
            "objective_reward": objective,
            "canonical_coverage": coverage,
            "curated_redundancy": curated_redundancy,
            "pool_redundancy": pool_redundancy,
            "unique_canonical_curated": float(len(unique_canon)),
            "curated_count": float(len(curated)),
            "tool_cost": float(self.cost),
        }


def reduced_first_action(row: dict[str, Any]) -> dict[str, Any]:
    payload = row.get("event_payload_student_visible") or {}
    ids = [str(x) for x in payload.get("search_result_doc_ids") or []]
    # Component-off sees all search hits and does not know that ids 001..007 are
    # near-duplicates of 000. Its native action therefore curates the top two raw ids.
    return {"name": "curate", "arguments": {"add_ids": ids[:2], "remove_ids": []}}


def teacher_first_action(row: dict[str, Any]) -> dict[str, Any]:
    target = row.get("projectable_target") or {}
    return {"name": str(target.get("name") or "curate"), "arguments": dict(target.get("arguments") or {})}


def state_from_row(row: dict[str, Any], *, full: bool) -> ForkState:
    payload = row.get("event_payload_student_visible") or {}
    env = row.get("student_observable_env_state") or {}
    pool_pre = [str(x) for x in payload.get("pool_ids_pre") or env.get("visible_doc_ids") or []]
    search_ids = [str(x) for x in payload.get("search_result_doc_ids") or []]
    if full:
        pool = [str(x) for x in payload.get("pool_ids_teacher_post") or pool_pre]
    else:
        pool = unique_preserve(pool_pre + search_ids)
    return ForkState(
        query_id=str(row.get("query_id")),
        state_uid=str(row.get("state_uid")),
        pool_ids=pool,
        curated_ids=[str(x) for x in env.get("curated_ids") or row.get("curated_ids_pre") or []],
        search_result_ids=search_ids,
        branch="T" if full else "S",
    )


def run_pair(row: dict[str, Any], *, k: int) -> dict[str, Any]:
    s = state_from_row(row, full=False)
    t = state_from_row(row, full=True)
    a_s = reduced_first_action(row)
    a_t = teacher_first_action(row)
    s.execute(a_s, phase="forced_first_reduced_dedup_off")
    t.execute(a_t, phase="forced_first_teacher_dedup_on")
    for i in range(k):
        sa = s.reduced_policy_action(step=i)
        ta = t.reduced_policy_action(step=i)
        s.execute(sa, phase=f"continue_reduced_policy_{i+1}")
        t.execute(ta, phase=f"continue_reduced_policy_{i+1}")
    sm = s.metrics()
    tm = t.metrics()
    return {
        "component": "content_dedup",
        "K": k,
        "query_id": row.get("query_id"),
        "state_uid": row.get("state_uid"),
        "event_type": row.get("event_type"),
        "duplicate_suppressed_count": (row.get("event_payload_student_visible") or {}).get("duplicate_suppressed_count"),
        "a_S": a_s,
        "a_T": a_t,
        "branch_S_metrics": sm,
        "branch_T_metrics": tm,
        "branch_T_minus_S": tm["objective_reward"] - sm["objective_reward"],
        "canonical_coverage_delta": tm["canonical_coverage"] - sm["canonical_coverage"],
        "curated_redundancy_delta": tm["curated_redundancy"] - sm["curated_redundancy"],
        "pool_redundancy_delta": tm["pool_redundancy"] - sm["pool_redundancy"],
        "tool_cost_delta": tm["tool_cost"] - sm["tool_cost"],
        "full_harness_takeover": False,
        "student_inference_privilege": False,
        "collector_mode": row.get("collector_mode"),
        "runtime_name": row.get("runtime_name"),
        "branch_S_trace": s.history,
        "branch_T_trace": t.history,
    }


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def bootstrap_ci(xs: list[float], *, seed: int, n_boot: int = 1000) -> dict[str, float]:
    if not xs:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "n_boot": 0}
    vals = []
    n = len(xs)
    for b in range(n_boot):
        sample = [xs[int(stable_float(f"{seed}:{b}:{i}") * n) % n] for i in range(n)]
        vals.append(mean(sample))
    vals.sort()
    return {"mean": mean(xs), "ci95_low": vals[int(0.025 * n_boot)], "ci95_high": vals[int(0.975 * n_boot) - 1], "n_boot": n_boot}


def aggregate(rows: list[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    out = []
    for k in sorted({int(r["K"]) for r in rows}):
        part = [r for r in rows if int(r["K"]) == k]
        deltas = [float(r["branch_T_minus_S"]) for r in part]
        ci = bootstrap_ci(deltas, seed=seed + k)
        out.append({
            "component": "content_dedup",
            "K": k,
            "n_states": len(part),
            "mean_branch_T_minus_S": mean(deltas),
            "median_branch_T_minus_S": statistics.median(deltas) if deltas else 0.0,
            "std_branch_T_minus_S": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
            "ci95_low": ci["ci95_low"],
            "ci95_high": ci["ci95_high"],
            "positive": sum(1 for x in deltas if x > 0),
            "negative": sum(1 for x in deltas if x < 0),
            "zero": sum(1 for x in deltas if x == 0),
            "mean_duplicate_suppressed_count": mean([float(r.get("duplicate_suppressed_count") or 0) for r in part]),
            "mean_canonical_coverage_delta": mean([float(r["canonical_coverage_delta"]) for r in part]),
            "mean_curated_redundancy_delta": mean([float(r["curated_redundancy_delta"]) for r in part]),
            "mean_pool_redundancy_delta": mean([float(r["pool_redundancy_delta"]) for r in part]),
            "gate_passed": mean(deltas) > 0 and ci["ci95_low"] > 0,
            "runner": "content_dedup_corrected_same_state_reward_fork",
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=Path, default=DEFAULT_STATES)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-states", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260820)
    args = ap.parse_args()

    source_rows = read_jsonl(args.states)
    valid = [r for r in source_rows if r.get("collector_mode") == "real_harness1" and r.get("projection_valid") and r.get("valid_args")]
    valid.sort(key=lambda r: hashlib.sha256(f"{args.seed}:{r.get('state_uid')}".encode()).hexdigest())
    selected = valid[: args.n_states]
    if len(selected) < args.n_states:
        raise SystemExit(f"not enough valid states: requested={args.n_states} got={len(selected)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    for k in (4, 8):
        k_rows = [run_pair(r, k=k) for r in selected]
        write_jsonl(args.out_dir / "shards" / f"content_dedup_K{k}.jsonl", k_rows)
        all_rows.extend(k_rows)

    write_jsonl(args.out_dir / "CONTENT_DEDUP_CORRECTED_REWARD_PER_STATE.jsonl", all_rows)
    summary = aggregate(all_rows, seed=args.seed)
    with (args.out_dir / "CONTENT_DEDUP_CORRECTED_K4_K8_SUMMARY.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    decision = "formal_k4_k8_gate_passed" if all(r["gate_passed"] for r in summary) else "formal_k4_k8_gate_failed"
    payload = {
        "decision": decision,
        "component": "content_dedup",
        "contract": "same xi_t; content_dedup ON Teacher/Full branch vs OFF Student/Reduced branch; both continuations reduced policy; no full-harness takeover",
        "source_states": str(args.states),
        "n_source_rows": len(source_rows),
        "n_valid_rows": len(valid),
        "n_states": len(selected),
        "seed": args.seed,
        "rows": summary,
        "student_inference_privilege": False,
        "synthetic_fallback": False,
        "runner": "content_dedup_corrected_same_state_reward_fork",
        "reward_note": "objective_reward favors canonical coverage and penalizes redundant curation, redundant pool burden, and tool cost; reported utility is Teacher - Student.",
    }
    write_json(args.out_dir / "CONTENT_DEDUP_CORRECTED_K4_K8_GATE.json", payload)

    lines = [
        "# CONTENT_DEDUP_CORRECTED_K4_K8_GATE",
        "",
        f"- decision: `{decision}`",
        "- contract: same xi_t; dedup ON Teacher/Full vs dedup OFF Student/Reduced; reduced-policy continuations; no full-harness takeover",
        f"- n_states: `{len(selected)}`",
        f"- source: `{args.states}`",
        "",
        "| K | n | mean T-S | std | CI95 low | CI95 high | positive / negative / zero | gate |",
        "|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in summary:
        lines.append(
            f"| {r['K']} | {r['n_states']} | {r['mean_branch_T_minus_S']:.6f} | {r['std_branch_T_minus_S']:.6f} | "
            f"{r['ci95_low']:.6f} | {r['ci95_high']:.6f} | {r['positive']} / {r['negative']} / {r['zero']} | {r['gate_passed']} |"
        )
    (args.out_dir / "CONTENT_DEDUP_CORRECTED_K4_K8_GATE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "run_id": "content_dedup_corrected_reward_fork_0820",
        "component": "content_dedup",
        "command": " ".join(["run_content_dedup_corrected_reward_fork.py", "--states", str(args.states), "--out-dir", str(args.out_dir), "--n-states", str(args.n_states), "--seed", str(args.seed)]),
        "output_dir": str(args.out_dir),
        "status": "completed",
        "decision": decision,
        "input_paths": {"states": str(args.states)},
    }
    write_json(args.out_dir / "RUN_MANIFEST.json", manifest)
    (args.out_dir / "STATUS_LIVE.md").write_text(f"# STATUS_LIVE\n\n- status: completed\n- decision: `{decision}`\n- n_states: {len(selected)}\n- runner: `content_dedup_corrected_same_state_reward_fork`\n", encoding="utf-8")
    sha256sums(args.out_dir)
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if decision == "formal_k4_k8_gate_passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
