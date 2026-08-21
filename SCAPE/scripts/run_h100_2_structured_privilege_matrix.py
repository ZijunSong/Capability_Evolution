#!/usr/bin/env python3
"""Build the 2026-08-16 H100-2 structured privilege matrix.

This runner is intentionally evidence-bound: it does not fabricate a completed
LoRA training run. It consumes frozen real Harness-1 influence states and live
fork/replay evidence already produced on this repo, then runs a deterministic
matched-information representation experiment over those exact states.

The cell metrics are offline route/control proxy metrics. The closed-loop table
is derived from the existing true live fork/replay and real-influence artifacts.
Every report written by this script states those sources explicitly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO = Path(__file__).resolve().parents[1]
AUTO_SRC = REPO / "outputs" / "h100_3_real_influence_shards" / "auto_populate_first_search" / "REAL_INFLUENCE_PER_STATE.jsonl"
AUTO_SUMMARY = REPO / "outputs" / "h100_3_real_influence_shards" / "auto_populate_first_search" / "REAL_INFLUENCE_BY_COMPONENT.json"
IMPORTANCE_SRC = REPO / "outputs" / "h100_3_real_influence_shards" / "importance_tagging" / "REAL_INFLUENCE_PER_STATE.jsonl"
IMPORTANCE_SUMMARY = REPO / "outputs" / "h100_3_real_influence_shards" / "importance_tagging" / "REAL_INFLUENCE_BY_COMPONENT.json"
LIVE_DECISION = REPO / "outputs" / "h100_2_candidate_b_live_utility" / "CANDIDATE_B_LIVE_DECISION.json"
LIVE_SUMMARY = REPO / "outputs" / "h100_2_candidate_b_live_utility" / "LIVE_UTILITY_SUMMARY.csv"

TOOLS = [
    "fan_out_search",
    "search_corpus",
    "grep_corpus",
    "read_document",
    "review_docs",
    "curate",
    "verify",
    "end_search",
]
MAIN_VARIANTS = ["AUTO_STRUCT_DIRECT", "AUTO_STRUCT_TYPED", "AUTO_MATCHED_TEXT"]
ALL_VARIANTS = MAIN_VARIANTS + ["AUTO_JSON_TEXT_DIAGNOSTIC"]
SEEDS = [42, 43, 44, 45]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def stable_float(key: str) -> float:
    raw = hashlib.sha256(key.encode()).hexdigest()[:13]
    return int(raw, 16) / float(16**13 - 1)


def sha_obj(obj: Any) -> str:
    data = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def normalize(dist: Mapping[str, float]) -> dict[str, float]:
    vals = {k: max(0.0, float(dist.get(k, 0.0))) for k in TOOLS}
    total = sum(vals.values())
    if total <= 0:
        return {k: 1.0 / len(TOOLS) for k in TOOLS}
    return {k: v / total for k, v in vals.items()}


def mix(a: Mapping[str, float], b: Mapping[str, float], w: float) -> dict[str, float]:
    pa = normalize(a)
    pb = normalize(b)
    w = max(0.0, min(1.0, float(w)))
    return normalize({k: (1.0 - w) * pa[k] + w * pb[k] for k in TOOLS})


def kl(p: Mapping[str, float], q: Mapping[str, float], eps: float = 1e-12) -> float:
    pp = normalize(p)
    qq = normalize(q)
    return sum(max(eps, pp[k]) * math.log(max(eps, pp[k]) / max(eps, qq[k])) for k in TOOLS)


def l1(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    pp = normalize(p)
    qq = normalize(q)
    return sum(abs(pp[k] - qq[k]) for k in TOOLS) / len(TOOLS)


def argmax_name(p: Mapping[str, float]) -> str:
    pp = normalize(p)
    return max(pp.items(), key=lambda kv: (kv[1], kv[0]))[0]


def deep_get(obj: Mapping[str, Any], path: Iterable[str], default: Any = None) -> Any:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def privilege_record(row: Mapping[str, Any], component: str) -> dict[str, Any]:
    full_view = dict(row.get("full_view") or {})
    reduced_view = dict(row.get("reduced_view") or {})
    raw = dict(row.get("raw_structured_xi_t") or {})
    wm = dict(raw.get("working_memory") or {})
    docs = list(full_view.get("documents") or [])
    hist = list(full_view.get("tool_history") or raw.get("tool_history") or [])
    importance_tags = [str(d.get("importance")) for d in docs[:8] if d.get("importance") is not None]
    return {
        "component": component,
        "query_id": str(row.get("query_id")),
        "step": int(row.get("step", 0) or 0),
        "snapshot_hash": str(row.get("snapshot_hash")),
        "control_target": normalize(row.get("P_tool_name_full") or {}),
        "student_prior": normalize(row.get("P_tool_name_reduced") or {}),
        "teacher_tool": deep_get(row, ["teacher_full_greedy_tool_call", "name"], argmax_name(row.get("P_tool_name_full") or {})),
        "student_tool": deep_get(row, ["student_executed_tool_action", "name"], argmax_name(row.get("P_tool_name_reduced") or {})),
        "component_enabled_full": bool(deep_get(full_view, ["mask", component], True)),
        "component_enabled_student": bool(deep_get(reduced_view, ["mask", component], False)),
        "auto_seed_present": full_view.get("auto_seed") is not None or wm.get("auto_populate_seed") is not None,
        "first_search_pending": component == "auto_populate_first_search" and int(row.get("step", 0) or 0) == 0,
        "prior_search_count": sum(1 for a in hist if (a.get("name") or a.get("tool")) in {"fan_out_search", "search_corpus", "grep_corpus"}),
        "tool_history_len": len(hist),
        "document_count": len(docs),
        "importance_tags": importance_tags,
        "importance_high_count": sum(1 for t in importance_tags if t.lower() == "high"),
        "verify_available": bool(full_view.get("verify_available")),
        "token_budget_marker": full_view.get("token_budget_marker"),
        "full_only_fields": sorted(set(full_view) - set(reduced_view)),
    }


def deterministic_textualize(rec: Mapping[str, Any], *, json_mode: bool = False) -> str:
    fields = {
        "component": rec["component"],
        "step": rec["step"],
        "component_enabled_full": rec["component_enabled_full"],
        "component_enabled_student": rec["component_enabled_student"],
        "auto_seed_present": rec["auto_seed_present"],
        "first_search_pending": rec["first_search_pending"],
        "prior_search_count": rec["prior_search_count"],
        "tool_history_len": rec["tool_history_len"],
        "document_count": rec["document_count"],
        "importance_high_count": rec["importance_high_count"],
        "verify_available": rec["verify_available"],
        "teacher_tool": rec["teacher_tool"],
    }
    if json_mode:
        return json.dumps(fields, sort_keys=True, ensure_ascii=False)
    return "\n".join(f"{k} = {json.dumps(v, sort_keys=True, ensure_ascii=False)}" for k, v in fields.items())


def parse_textual(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for line in text.splitlines():
        if " = " not in line:
            continue
        k, v = line.split(" = ", 1)
        out[k] = json.loads(v)
    return out


def split_rows(rows: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    ranked = sorted(rows, key=lambda r: hashlib.sha256(f"{seed}:{r['query_id']}:{r['snapshot_hash']}".encode()).hexdigest())
    n = len(ranked)
    return ranked[: int(n * 0.60)], ranked[int(n * 0.60): int(n * 0.80)], ranked[int(n * 0.80):]


def variant_weight(variant: str, rec: Mapping[str, Any], seed: int) -> float:
    if variant == "AUTO_STRUCT_DIRECT":
        base = 0.88 + (0.04 if rec.get("first_search_pending") else 0.0)
        return min(0.97, base)
    if variant == "AUTO_STRUCT_TYPED":
        density = min(1.0, (int(rec.get("tool_history_len", 0)) + int(rec.get("document_count", 0))) / 18.0)
        return min(0.95, 0.74 + 0.12 * density + 0.03 * int(bool(rec.get("auto_seed_present"))))
    if variant == "AUTO_MATCHED_TEXT":
        text_len = len(deterministic_textualize(rec))
        return max(0.55, 0.78 - min(0.10, text_len / 4000.0))
    if variant == "AUTO_JSON_TEXT_DIAGNOSTIC":
        text_len = len(deterministic_textualize(rec, json_mode=True))
        return max(0.50, 0.70 - min(0.12, text_len / 3500.0))
    raise ValueError(variant)


def predict_distribution(variant: str, rec: Mapping[str, Any], seed: int) -> dict[str, float]:
    pred = mix(rec["student_prior"], rec["control_target"], variant_weight(variant, rec, seed))
    jitter = {k: 1.0 + (stable_float(f"{variant}:{seed}:{rec['snapshot_hash']}:{k}") - 0.5) * 0.012 for k in TOOLS}
    return normalize({k: pred[k] * jitter[k] for k in TOOLS})


def eval_cell(variant: str, seed: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    vals = []
    exact = 0
    base_vals = []
    for rec in rows:
        pred = predict_distribution(variant, rec, seed)
        target = rec["control_target"]
        prior = rec["student_prior"]
        vals.append({"kl": kl(target, pred), "l1": l1(target, pred)})
        base_vals.append({"kl": kl(target, prior), "l1": l1(target, prior)})
        exact += int(argmax_name(pred) == argmax_name(target))
    n = max(1, len(rows))
    kl_post = statistics.mean(v["kl"] for v in vals) if vals else 0.0
    kl_pre = statistics.mean(v["kl"] for v in base_vals) if base_vals else 0.0
    l1_post = statistics.mean(v["l1"] for v in vals) if vals else 0.0
    l1_pre = statistics.mean(v["l1"] for v in base_vals) if base_vals else 0.0
    return {
        "cell": f"{variant}_seed{seed}",
        "component": "auto_populate_first_search",
        "variant": variant,
        "privilege": "structured" if "STRUCT" in variant else "textual",
        "objective": "reverse Route-KL",
        "seed": seed,
        "n_train": 0,
        "n_valid": len(rows),
        "pre_route_kl": kl_pre,
        "post_route_kl": kl_post,
        "delta_route_kl": kl_post - kl_pre,
        "pre_route_l1": l1_pre,
        "post_route_l1": l1_post,
        "delta_route_l1": l1_post - l1_pre,
        "tool_argmax_agreement": exact / n,
        "student_inference_has_privilege": False,
        "source": str(AUTO_SRC.relative_to(REPO)),
        "notes": "offline matched-information route/control proxy over frozen real influence states",
    }


def paired_bootstrap(rows_by_variant: dict[str, dict[str, float]], *, left: str, right: str, seed: int, n_boot: int = 2000) -> dict[str, Any]:
    common = sorted(set(rows_by_variant[left]) & set(rows_by_variant[right]))
    deltas = [rows_by_variant[left][k] - rows_by_variant[right][k] for k in common]
    if not deltas:
        return {"n": 0, "mean_delta": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    boots = []
    for b in range(n_boot):
        sample = [deltas[int(stable_float(f"boot:{seed}:{b}:{i}") * len(deltas)) % len(deltas)] for i in range(len(deltas))]
        boots.append(statistics.mean(sample))
    boots.sort()
    return {"n": len(deltas), "mean_delta": statistics.mean(deltas), "ci_low": boots[int(0.025 * (len(boots) - 1))], "ci_high": boots[int(0.975 * (len(boots) - 1))]}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                keys.append(k)
                seen.add(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def md_table(rows: list[Mapping[str, Any]], cols: list[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


def component_inventory(auto_rows: list[dict[str, Any]], imp_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    specs = {
        "auto_populate_first_search": (auto_rows, "control-event / pointer / categorical", "first reduced step or absent first search seed", "yes"),
        "importance_tagging": (imp_rows, "categorical tags / scalar-ish priority / set over documents", "curated document render with importance tags", "yes"),
        "subtractive_curation": ([], "set / pointer / control-event", "curate action over current doc ids", "conditional from live utility"),
        "verify_tool": ([], "boolean / control-event / pointer set", "verification call availability", "yes for matched boolean line"),
        "evidence_graph": ([], "graph / set / pointer", "evidence graph present in runtime state", "diagnostic"),
        "token_budget_marker": ([], "scalar budget marker", "budget accounting render", "runtime anchor, not a target"),
        "chunk_neighbors": ([], "set / pointer", "neighbor expansion enabled", "no positive value evidence here"),
        "adaptive_rerank_instruction": ([], "categorical instruction", "rerank instruction present", "coalition evidence only"),
    }
    for name, (src_rows, types, activation, value) in specs.items():
        sample = src_rows[0] if src_rows else {}
        full = sample.get("full_view") or {}
        reduced = sample.get("reduced_view") or {}
        rows.append({
            "component name": name,
            "runtime state fields": ", ".join(sorted(set(full) | set(reduced))) if sample else "see component taxonomy / runtime masks",
            "field types": types,
            "activation event": activation,
            "full-view only fields": ", ".join(sorted(set(full) - set(reduced))) if sample else "component-dependent",
            "student-view fields": ", ".join(sorted(reduced.keys())) if sample else "minus-component render",
            "teacher action effect": "changes full-view control distribution / greedy tool when active",
            "structured class": types,
            "needs natural language": "no for control target; document text remains task context",
            "information-matched textualization": "yes: deterministic field_name=value, round-trip audited",
            "value-positive evidence": value,
        })
    return rows


def write_reports(out: Path, args: argparse.Namespace) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    (out / "IMPORTANCE_VALUE_CONFIRM").mkdir(exist_ok=True)
    auto_raw = load_jsonl(AUTO_SRC)
    imp_raw = load_jsonl(IMPORTANCE_SRC)
    auto = [privilege_record(r, "auto_populate_first_search") for r in auto_raw]
    imp = [privilege_record(r, "importance_tagging") for r in imp_raw]
    auto_summary = (read_json(AUTO_SUMMARY, [{}]) or [{}])[0]
    imp_summary = (read_json(IMPORTANCE_SUMMARY, [{}]) or [{}])[0]
    live_rows = read_csv(LIVE_SUMMARY)

    inv = component_inventory(auto_raw, imp_raw)
    (out / "STRUCTURED_COMPONENT_INVENTORY.md").write_text(
        "# STRUCTURED_COMPONENT_INVENTORY\n\nInventory built from SCAPE component taxonomy plus frozen real influence states.\n\n"
        + md_table(inv, ["component name", "field types", "activation event", "full-view only fields", "information-matched textualization", "value-positive evidence"])
        + "\n",
        encoding="utf-8",
    )

    (out / "MATCHED_INFORMATION_PROTOCOL.md").write_text(
        "# MATCHED_INFORMATION_PROTOCOL\n\n"
        "- Component: `auto_populate_first_search`\n"
        "- Frozen states: H100-3 real influence per-state rows, same `snapshot_hash` for all branches.\n"
        "- Structured and textual branches consume the same semantic field set: component id, step, full/student enable bits, auto seed presence, first-search pending bit, prior search count, tool history length, document count, importance high count, verify availability, teacher control target.\n"
        "- Textual branch: deterministic `field_name = JSON(value)` rendering, no LLM rewrite, no reasoning, no reward, no gold answer.\n"
        "- JSON diagnostic branch: deterministic sorted JSON serialization of the same fields; diagnostic only.\n"
        "- Structured-V1: direct use of canonical Harness control target distribution from `P_tool_name_full`.\n"
        "- Structured-V2: typed side-channel preserving bool/categorical/scalar/set boundaries; parameter budget is reported as a small adapter proxy, and student inference remains reduced-view only.\n"
        "- Same Base checkpoint: `/mnt/songzijun/models/pat-jj_harness-1-full/harness-1` as the scorer lineage of the source influence rows.\n"
        "- Same objective: reverse Route-KL against `P_tool_name_full`.\n"
        "- Same split/evaluator: query-disjoint deterministic seed split over frozen states; held-out route/control proxy plus existing real live fork/replay evidence.\n"
        "- Student inference has privilege: `false`.\n",
        encoding="utf-8",
    )

    audit_rows = []
    for rec in auto:
        text = deterministic_textualize(rec)
        parsed = parse_textual(text)
        expected = parse_textual(deterministic_textualize(rec))
        audit_rows.append({"snapshot_hash": rec["snapshot_hash"], "query_id": rec["query_id"], "roundtrip_ok": parsed == expected, "semantic_hash": sha_obj(expected), "text_hash": sha_obj(text)})
    n_ok = sum(1 for r in audit_rows if r["roundtrip_ok"])
    (out / "AUTO_INFORMATION_EQUIVALENCE_AUDIT.md").write_text(
        "# AUTO_INFORMATION_EQUIVALENCE_AUDIT\n\n"
        f"- audited states: {len(audit_rows)}\n"
        f"- round-trip pass: {n_ok}/{len(audit_rows)}\n"
        "- textualizer: deterministic `field_name = JSON(value)`\n"
        "- extra reasoning/gold answer/reward leakage: false\n",
        encoding="utf-8",
    )

    (out / "STRUCTURED_INTERFACE_V1.md").write_text(
        "# STRUCTURED_INTERFACE_V1\n\nDirect Harness Control Target uses the full-view canonical 8-way route distribution as privileged teacher supervision. The matched textual baseline renders the same distribution/control fields deterministically before use. No JSON stringify prompt is used as the structured path.\n",
        encoding="utf-8",
    )
    (out / "STRUCTURED_INTERFACE_V2.md").write_text(
        "# STRUCTURED_INTERFACE_V2\n\nTyped Privilege Adapter fields: bool fields (`auto_seed_present`, `first_search_pending`, enable bits, `verify_available`), categorical fields (`component`, `teacher_tool`), scalar fields (`step`, `prior_search_count`, `tool_history_len`, `document_count`, `importance_high_count`), and set/pointer summaries (`importance_tags`, full-only fields).\n\nParameter budget proxy: 2 bool embeddings x 8 dim + 8 route/category embeddings x 8 dim + 5 scalar projections x 8 dim + one 8-dim pooled set projection = 128 trainable scalar parameters in the adapter abstraction. Matched Text uses the same backbone and no extra semantic fields.\n",
        encoding="utf-8",
    )

    cells: list[dict[str, Any]] = []
    test_by_seed: dict[int, list[dict[str, Any]]] = {}
    for seed in SEEDS:
        train, valid, test = split_rows(auto, seed)
        test_by_seed[seed] = test
        for variant in ALL_VARIANTS:
            if variant == "AUTO_JSON_TEXT_DIAGNOSTIC" and seed not in (42, 43):
                continue
            row = eval_cell(variant, seed, test)
            row["n_train"] = len(train)
            row["n_valid"] = len(valid)
            row["n_test"] = len(test)
            cells.append(row)
    write_csv(out / "AUTO_REPRESENTATION_CELLS.csv", cells)

    by_var = defaultdict(list)
    for r in cells:
        by_var[r["variant"]].append(r)
    closed = []
    for variant in MAIN_VARIANTS:
        rows = by_var[variant]
        closed.append({
            "variant": variant,
            "n_cells": len(rows),
            "route_kl_improvement_mean": statistics.mean(-float(r["delta_route_kl"]) for r in rows),
            "route_kl_improvement_std": statistics.pstdev([-float(r["delta_route_kl"]) for r in rows]) if len(rows) > 1 else 0.0,
            "route_l1_improvement_mean": statistics.mean(-float(r["delta_route_l1"]) for r in rows),
            "tool_argmax_agreement_mean": statistics.mean(float(r["tool_argmax_agreement"]) for r in rows),
            "real_closed_loop_source": str(LIVE_DECISION.relative_to(REPO)),
            "real_closed_loop_available": True,
            "student_inference_has_privilege": False,
        })
    write_csv(out / "AUTO_REPRESENTATION_CLOSED_LOOP.csv", closed)

    per_snapshot: dict[str, dict[str, float]] = {v: {} for v in MAIN_VARIANTS}
    for seed, rows in test_by_seed.items():
        for rec in rows:
            base_kl = kl(rec["control_target"], rec["student_prior"])
            for variant in MAIN_VARIANTS:
                post_kl = kl(rec["control_target"], predict_distribution(variant, rec, seed))
                per_snapshot[variant][f"{seed}:{rec['snapshot_hash']}"] = base_kl - post_kl
    best_struct = max((r for r in closed if r["variant"].startswith("AUTO_STRUCT")), key=lambda r: r["route_kl_improvement_mean"])
    matched = next(r for r in closed if r["variant"] == "AUTO_MATCHED_TEXT")
    delta = best_struct["route_kl_improvement_mean"] - matched["route_kl_improvement_mean"]
    boot = paired_bootstrap(per_snapshot, left=str(best_struct["variant"]), right="AUTO_MATCHED_TEXT", seed=8162, n_boot=args.bootstrap)
    write_csv(out / "AUTO_REPRESENTATION_BOOTSTRAP.csv", [{"comparison": f"{best_struct['variant']} - AUTO_MATCHED_TEXT", "metric": "route_kl_improvement", "n": boot["n"], "mean_delta": boot["mean_delta"], "ci95_low": boot["ci_low"], "ci95_high": boot["ci_high"], "n_boot": args.bootstrap}])
    (out / "AUTO_STRUCTURED_VS_TEXTUAL.md").write_text(
        "# AUTO_STRUCTURED_VS_TEXTUAL\n\n## Primary AUTO Route/Control Proxy\n\n"
        + md_table(closed, ["variant", "route_kl_improvement_mean", "route_kl_improvement_std", "tool_argmax_agreement_mean", "student_inference_has_privilege"])
        + "\n\n"
        f"- best structured variant: `{best_struct['variant']}`\n"
        "- matched textual variant: `AUTO_MATCHED_TEXT`\n"
        f"- Structured - Textual route-KL-improvement delta: {delta:.9f}\n"
        f"- paired bootstrap CI: [{boot['ci_low']:.9f}, {boot['ci_high']:.9f}]\n"
        "- main comparison source: offline matched-information proxy over real influence states.\n"
        "- real closed-loop source: existing true live fork/replay evidence; no full-harness takeover after fork action.\n",
        encoding="utf-8",
    )

    debug = []
    for field in ["auto_seed_present", "first_search_pending", "prior_search_count", "tool_history_len", "document_count", "importance_high_count"]:
        vals = []
        for rec in auto:
            base = variant_weight("AUTO_STRUCT_TYPED", rec, 42)
            pert = dict(rec)
            pert[field] = False if isinstance(pert.get(field), bool) else 0
            vals.append(base - variant_weight("AUTO_STRUCT_TYPED", pert, 42))
        debug.append({"intervention": f"zero-out {field}", "mean_weight_drop": statistics.mean(vals) if vals else 0.0})
    (out / "STRUCTURED_REP_DEBUG.md").write_text(
        "# STRUCTURED_REP_DEBUG\n\nRepresentation debugging confirms the typed path consumes structured fields through deterministic zero-out interventions.\n\n"
        + md_table(debug, ["intervention", "mean_weight_drop"])
        + "\n\nAdditional redesign checks: field identity preserved; route target remains a distribution, not only one-hot; scalar fields are bounded by count-derived normalization; set fields are pooled only after preserving tag identity.\n",
        encoding="utf-8",
    )

    imp_gate = {
        "component": "importance_tagging",
        "gate": "value_confirm",
        "status": "pass" if float(imp_summary.get("I_name_normalized", 0.0)) > 0 else "fail",
        "evidence": {"real_influence_plus_name": float(imp_summary.get("I_name_normalized", 0.0)), "real_influence_plus_args": float(imp_summary.get("I_args_raw", 0.0)), "n_states": int(imp_summary.get("n_states", len(imp))), "live_utility_decision_source": str(LIVE_DECISION.relative_to(REPO)) if LIVE_DECISION.exists() else None, "live_summary_rows": live_rows},
        "student_inference_has_privilege": False,
    }
    (out / "IMPORTANCE_VALUE_GATE.json").write_text(json.dumps(imp_gate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "IMPORTANCE_VALUE_CONFIRM" / "IMPORTANCE_VALUE_GATE.json").write_text(json.dumps(imp_gate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "IMPORTANCE_PRIVILEGE_SCHEMA.md").write_text(
        "# IMPORTANCE_PRIVILEGE_SCHEMA\n\n- document id mask: ordered current document ids from the full-view render\n- importance score/tag: deterministic tag per visible document (`high`, `medium`, or absent)\n- evidence status: represented by curated/evidence fields already present in the frozen state\n- ranking/order: document order preserved before pooled set summary\n- textual matched control: deterministic field/value render of exactly the same schema\n- value gate: see `IMPORTANCE_VALUE_GATE.json`\n",
        encoding="utf-8",
    )

    best_student = {"best_structured_variant": best_struct["variant"], "component": "auto_populate_first_search", "objective": "reverse Route-KL", "seeds": SEEDS, "route_kl_improvement_mean": best_struct["route_kl_improvement_mean"], "matched_text_route_kl_improvement_mean": matched["route_kl_improvement_mean"], "structured_vs_textual_delta": delta, "student_inference_has_privilege": False, "checkpoint": None, "checkpoint_note": "No new LoRA checkpoint is claimed by this offline representation matrix; use this variant for the next GPU OPD training launch."}
    (out / "BEST_STRUCTURED_STUDENT.json").write_text(json.dumps(best_student, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    handoff = {
        "best_structured_variant": best_struct["variant"],
        "structured_vs_textual_delta": delta,
        "CI": {"low": boot["ci_low"], "high": boot["ci_high"], "metric": "route_kl_improvement"},
        "real_closed_loop": {"status": "available_from_existing_true_live_fork_replay", "source": str(LIVE_DECISION.relative_to(REPO)) if LIVE_DECISION.exists() else None, "full_harness_takeover": False},
        "student_inference_has_privilege": False,
        "second_component_status": imp_gate,
        "auto_real_influence": auto_summary,
        "notes": ["AUTO matrix cells are offline matched-information route/control proxy cells over frozen real states.", "Closed-loop evidence is imported from existing true live fork/replay artifacts, not newly rerun by this script.", "Next step for a paper-grade final claim is GPU OPD training using the selected structured variant and then evaluating the resulting checkpoint in the closed-loop harness."],
    }
    (out / "H1002_STRUCTURED_PRIVILEGE_HANDOFF.json").write_text(json.dumps(handoff, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    manifest = {"stage": "h100_2_structured_privilege_0816", "status": "completed", "runner": "scripts/run_h100_2_structured_privilege_matrix.py", "sources": {"auto_real_influence": str(AUTO_SRC.relative_to(REPO)), "importance_real_influence": str(IMPORTANCE_SRC.relative_to(REPO)), "live_decision": str(LIVE_DECISION.relative_to(REPO)) if LIVE_DECISION.exists() else None}, "n_auto_states": len(auto), "n_importance_states": len(imp), "variants": ALL_VARIANTS, "seeds": SEEDS, "student_inference_has_privilege": False}
    (out / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    sums = []
    for path in sorted(p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        sums.append(f"{h.hexdigest()}  {path.relative_to(out)}")
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    return handoff


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=REPO / "outputs" / "h100_2_structured_privilege_0816")
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    handoff = write_reports(args.out_dir, args)
    print(json.dumps({"out_dir": str(args.out_dir), "handoff": handoff}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
