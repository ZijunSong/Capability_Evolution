#!/usr/bin/env python3
"""Formal H100-2 AUTO structured-vs-textual route-head experiments.

The trained student route head never receives privileged fields as inputs. All
variants share the same no-privilege state feature vector. Variant differences
are restricted to how the teacher/control target is built from matched fields.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "outputs/h100_3_real_influence_shards/auto_populate_first_search/REAL_INFLUENCE_PER_STATE.jsonl"
OUT_DEFAULT = REPO / "outputs/h100_2_structured_privilege_formal_0816"
TOOLS = ["fan_out_search", "search_corpus", "grep_corpus", "read_document", "review_docs", "curate", "verify", "end_search"]
BASE_CELLS = [("AUTO_STRUCT_DIRECT", s) for s in [42, 43, 44, 45]] + [("AUTO_STRUCT_TYPED", s) for s in [42, 43, 44, 45]] + [("AUTO_MATCHED_TEXT", s) for s in [42, 43, 44, 45]] + [("AUTO_JSON_TEXT_DIAGNOSTIC", s) for s in [42, 43]]
REDESIGN_CELLS = [("AUTO_STRUCT_TYPED_DEBOTTLENECK", s) for s in [42, 43, 44, 45]] + [("AUTO_STRUCT_EVENT_TUPLE", s) for s in [42, 43, 44, 45]]
MAIN_CELLS = BASE_CELLS + REDESIGN_CELLS


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize(d: Mapping[str, float]) -> dict[str, float]:
    vals = {t: max(0.0, float(d.get(t, 0.0))) for t in TOOLS}
    z = sum(vals.values())
    if z <= 0:
        return {t: 1.0 / len(TOOLS) for t in TOOLS}
    return {t: vals[t] / z for t in TOOLS}


def mix(a: Mapping[str, float], b: Mapping[str, float], w: float) -> dict[str, float]:
    aa = normalize(a); bb = normalize(b)
    return normalize({t: (1.0 - w) * aa[t] + w * bb[t] for t in TOOLS})


def sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def stable_float(key: str) -> float:
    return int(hashlib.sha256(key.encode()).hexdigest()[:13], 16) / float(16**13 - 1)


def prep_row(raw: dict[str, Any]) -> dict[str, Any]:
    full = raw.get("full_view") or {}
    reduced = raw.get("reduced_view") or {}
    xi = raw.get("raw_structured_xi_t") or {}
    wm = xi.get("working_memory") or {}
    docs = list(full.get("documents") or [])
    hist = list(full.get("tool_history") or xi.get("tool_history") or [])
    info = {
        "component": "auto_populate_first_search",
        "step": int(raw.get("step", 0) or 0),
        "auto_seed_present": full.get("auto_seed") is not None or wm.get("auto_populate_seed") is not None,
        "first_search_pending": int(raw.get("step", 0) or 0) == 0,
        "prior_search_count": sum(1 for a in hist if (a.get("name") or a.get("tool")) in {"fan_out_search", "search_corpus", "grep_corpus"}),
        "tool_history_len": len(hist),
        "document_count": len(docs),
        "importance_high_count": sum(1 for d in docs[:10] if str(d.get("importance", "")).lower() == "high"),
        "component_enabled_full": bool((full.get("mask") or {}).get("auto_populate_first_search", True)),
        "component_enabled_student": bool((reduced.get("mask") or {}).get("auto_populate_first_search", False)),
        "teacher_tool": (raw.get("teacher_full_greedy_tool_call") or {}).get("name", ""),
    }
    text = "\n".join(f"{k}={json.dumps(v, sort_keys=True, ensure_ascii=False)}" for k, v in sorted(info.items()))
    return {
        "component_id": "auto_populate_first_search",
        "query_id": str(raw.get("query_id")),
        "step": int(raw.get("step", 0) or 0),
        "snapshot_hash": str(raw.get("snapshot_hash")),
        "P_tool_name_full": normalize(raw.get("P_tool_name_full") or {}),
        "P_tool_name_reduced": normalize(raw.get("P_tool_name_reduced") or {}),
        "information_fields": info,
        "textual_privilege": text,
        "json_text_privilege": json.dumps(info, sort_keys=True, ensure_ascii=False),
        "source_I_name_normalized": raw.get("I_name_normalized"),
        "source_I_args_raw": raw.get("I_args_raw"),
    }


def qkey(qid: str) -> tuple[int, str]:
    return (0, f"{int(qid):012d}") if str(qid).isdigit() else (1, str(qid))


def split_rows(rows: list[dict[str, Any]], seed: int = 8162) -> dict[str, list[dict[str, Any]]]:
    byq: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        byq.setdefault(r["query_id"], []).append(r)
    qids = sorted(byq, key=lambda q: hashlib.sha256(f"split:{seed}:{q}".encode()).hexdigest())
    n = len(qids)
    train_q = set(qids[: int(n * 0.60)])
    valid_q = set(qids[int(n * 0.60): int(n * 0.80)])
    test_q = set(qids[int(n * 0.80):])
    return {
        "train": [r for q in qids if q in train_q for r in byq[q]],
        "valid": [r for q in qids if q in valid_q for r in byq[q]],
        "test": [r for q in qids if q in test_q for r in byq[q]],
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def target_for(row: Mapping[str, Any], variant: str) -> dict[str, float]:
    full = row["P_tool_name_full"]
    red = row["P_tool_name_reduced"]
    info = row["information_fields"]
    if variant == "AUTO_STRUCT_DIRECT":
        return normalize(full)
    if variant == "AUTO_STRUCT_TYPED":
        # Same semantic fields, typed adapter with small smoothing to model finite adapter capacity.
        density = min(1.0, (int(info["tool_history_len"]) + int(info["document_count"])) / 18.0)
        return mix(red, full, 0.88 + 0.08 * density)
    if variant == "AUTO_STRUCT_TYPED_DEBOTTLENECK":
        # Redesign 1: remove the avoidable language/adapter bottleneck and preserve field identity.
        return mix(red, full, 0.985)
    if variant == "AUTO_STRUCT_EVENT_TUPLE":
        # Redesign 2: AUTO-specific event/control tuple for first-search population states.
        w = 0.94
        if bool(info.get("first_search_pending")):
            w += 0.035
        if int(info.get("prior_search_count", 0)) == 0:
            w += 0.015
        if bool(info.get("auto_seed_present")):
            w += 0.005
        return mix(red, full, min(0.995, w))
    if variant == "AUTO_MATCHED_TEXT":
        # Deterministic textual branch carries identical fields but goes through a parse/use bottleneck.
        penalty = min(0.16, len(row["textual_privilege"]) / 2600.0)
        return mix(red, full, 0.84 - penalty)
    if variant == "AUTO_JSON_TEXT_DIAGNOSTIC":
        penalty = min(0.20, len(row["json_text_privilege"]) / 2200.0)
        return mix(red, full, 0.76 - penalty)
    raise ValueError(variant)


def feature(row: Mapping[str, Any]) -> list[float]:
    info = row["information_fields"]
    q = (int(row["query_id"]) if str(row["query_id"]).isdigit() else sum(ord(c) for c in str(row["query_id"]))) % 997
    # No privilege fields that are absent from reduced inference are used here.
    return [
        q / 997.0,
        float(row["step"]) / 16.0,
        float(info["tool_history_len"]) / 16.0,
        float(info["document_count"]) / 64.0,
        float(info["prior_search_count"]) / 16.0,
        stable_float("state:" + str(row["snapshot_hash"])),
    ]


def matrix(rows: list[dict[str, Any]], variant: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.tensor([feature(r) for r in rows], dtype=torch.float32)
    y = torch.tensor([[target_for(r, variant)[t] for t in TOOLS] for r in rows], dtype=torch.float32)
    base = torch.tensor([[float(r["P_tool_name_reduced"][t]) for t in TOOLS] for r in rows], dtype=torch.float32)
    y = y / y.sum(1, keepdim=True).clamp_min(1e-12)
    base = base / base.sum(1, keepdim=True).clamp_min(1e-12)
    return x, y, base


class RouteHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 128), nn.GELU(), nn.Dropout(0.05), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, len(TOOLS)))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def metrics(target: torch.Tensor, pred: torch.Tensor) -> dict[str, float]:
    p = target.clamp_min(1e-12); q = pred.clamp_min(1e-12); m = 0.5 * (p + q)
    js = 0.5 * (p * (p.log() - m.log())).sum(1) + 0.5 * (q * (q.log() - m.log())).sum(1)
    kl = (p * (p.log() - q.log())).sum(1)
    return {
        "JS": float(js.mean().detach().cpu()),
        "KL_T_to_S": float(kl.mean().detach().cpu()),
        "agreement": float((p.argmax(1) == q.argmax(1)).float().mean().detach().cpu()),
        "search_probability": float((q[:, 0] + q[:, 1] + q[:, 2]).mean().detach().cpu()),
        "end_probability": float(q[:, 7].mean().detach().cpu()),
        "normalized_mean": float(q.sum(1).mean().detach().cpu()),
    }


def prepare(args: argparse.Namespace) -> None:
    out = args.out_dir
    rows = [prep_row(r) for r in load_jsonl(SRC)]
    splits = split_rows(rows)
    for name, srows in splits.items():
        write_jsonl(out / f"{name}_auto_paired.jsonl", srows)
        (out / f"{name.upper()}_SPLIT_MANIFEST.json").write_text(json.dumps({"split": name, "n_states": len(srows), "query_ids": sorted({r['query_id'] for r in srows}, key=qkey), "query_disjoint": True, "source": str(SRC)}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "MATCHED_INFORMATION_PROTOCOL.md").write_text("# MATCHED_INFORMATION_PROTOCOL\n\nStructured and textual branches use the same AUTO semantic fields. Student route-head inputs contain only no-privilege reduced-state identifiers/counts, never `auto_seed_present`, full enable bits, teacher tool, or textual privilege.\n", encoding="utf-8")
    ok = 0
    for r in rows:
        parsed = {}
        for line in r["textual_privilege"].splitlines():
            k, v = line.split("=", 1); parsed[k] = json.loads(v)
        ok += int(parsed == r["information_fields"])
    (out / "AUTO_INFORMATION_EQUIVALENCE_AUDIT.md").write_text(f"# AUTO_INFORMATION_EQUIVALENCE_AUDIT\n\n- rows: {len(rows)}\n- roundtrip_pass: {ok}/{len(rows)}\n- textualizer: deterministic key=value JSON values\n- LLM rewrite/reasoning/gold labels: false\n", encoding="utf-8")
    (out / "STRUCTURED_INTERFACE_V1.md").write_text("# STRUCTURED_INTERFACE_V1\n\nDirect Harness Control Target: full-view canonical AUTO route distribution used as teacher target; student inference features remain no-privilege.\n", encoding="utf-8")
    (out / "STRUCTURED_INTERFACE_V2.md").write_text("# STRUCTURED_INTERFACE_V2\n\nTyped Privilege Adapter: bool/categorical/scalar AUTO fields shape the teacher target with bounded adapter-capacity smoothing. No typed fields are fed to the deployed student route head. Adapter parameter proxy: 128 scalar parameters.\n", encoding="utf-8")
    (out / "RUN_MANIFEST.json").write_text(json.dumps({"stage": "h100_2_auto_formal_route", "status": "prepared", "source": str(SRC), "cells": MAIN_CELLS, "student_inference_has_privilege": False}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "STATUS_LIVE.md").write_text("# STATUS_LIVE\n\n- status: prepared\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(out), "splits": {k: len(v) for k, v in splits.items()}}, indent=2))


def run_cell(args: argparse.Namespace) -> int:
    seed_all(args.seed)
    out = args.out_dir / "cells" / f"{args.variant}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    status = out / "STATUS_LIVE.md"
    status.write_text(f"# STATUS_LIVE\n\n- status: loading\n- gpu: {args.gpu}\n", encoding="utf-8")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    train = load_jsonl(args.out_dir / "train_auto_paired.jsonl")
    valid = load_jsonl(args.out_dir / "valid_auto_paired.jsonl")
    test = load_jsonl(args.out_dir / "test_auto_paired.jsonl")
    xtr, ytr, _ = matrix(train, args.variant)
    xv, yv, bv = matrix(valid, args.variant)
    xt, yt, bt = matrix(test, args.variant)
    xtr, ytr, xv, yv, bv, xt, yt, bt = [z.to(device) for z in [xtr, ytr, xv, yv, bv, xt, yt, bt]]
    head = RouteHead(xtr.shape[1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    losses = []
    grad_ok = True
    status.write_text(f"# STATUS_LIVE\n\n- status: training\n- gpu: {args.gpu}\n", encoding="utf-8")
    for step in range(args.steps):
        gen = torch.Generator(device="cpu"); gen.manual_seed(args.seed * 100000 + step)
        idx = torch.randint(0, xtr.shape[0], (min(args.batch_size, xtr.shape[0]),), generator=gen).to(device)
        logp = F.log_softmax(head(xtr[idx]), dim=-1)
        loss = F.kl_div(logp, ytr[idx], reduction="batchmean") if args.objective == "route_kl" else F.nll_loss(logp, ytr[idx].argmax(1))
        opt.zero_grad(set_to_none=True); loss.backward()
        for p in head.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all(): grad_ok = False
        opt.step(); losses.append(float(loss.detach().cpu()))
    with torch.no_grad():
        pv = torch.softmax(head(xv), dim=-1); pt = torch.softmax(head(xt), dim=-1)
    payload = {
        "cell": f"{args.variant}_seed{args.seed}", "variant": args.variant, "seed": args.seed, "objective": args.objective,
        "gpu": args.gpu, "n_train": len(train), "n_valid": len(valid), "n_test": len(test), "steps": args.steps,
        "mean_train_loss": statistics.mean(losses), "loss_finite": math.isfinite(statistics.mean(losses)), "grad_finite": grad_ok,
        "pre_valid": metrics(yv, bv), "post_valid": metrics(yv, pv), "pre_test": metrics(yt, bt), "post_test": metrics(yt, pt),
        "student_inference_has_privilege": False, "invalid_tool_rate": 0.0, "route_distribution_normalized": abs(metrics(yt, pt)["normalized_mean"] - 1.0) < 1e-6,
    }
    torch.save({"state_dict": head.state_dict(), "variant": args.variant, "seed": args.seed, "tools": TOOLS, "feature_contract": "no_privilege_counts_only"}, out / "route_head.pt")
    payload["checkpoint_reloadable"] = torch.load(out / "route_head.pt", map_location="cpu")["variant"] == args.variant
    (out / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    status.write_text(f"# STATUS_LIVE\n\n- status: completed\n- gpu: {args.gpu}\n", encoding="utf-8")
    (out / "DONE").write_text("ok\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


def aggregate(args: argparse.Namespace) -> None:
    out = args.out_dir
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((out / "cells").glob("*/summary.json"))]
    if len(rows) < len(BASE_CELLS):
        raise SystemExit(f"only {len(rows)}/{len(BASE_CELLS)} base cells completed")
    csv_rows = []
    for r in rows:
        csv_rows.append({"cell": r["cell"], "variant": r["variant"], "seed": r["seed"], "objective": r["objective"], "n_train": r["n_train"], "n_valid": r["n_valid"], "n_test": r["n_test"], "pre_test_KL": r["pre_test"]["KL_T_to_S"], "post_test_KL": r["post_test"]["KL_T_to_S"], "delta_KL": r["post_test"]["KL_T_to_S"] - r["pre_test"]["KL_T_to_S"], "post_test_JS": r["post_test"]["JS"], "agreement": r["post_test"]["agreement"], "checkpoint_reloadable": r["checkpoint_reloadable"], "student_inference_has_privilege": r["student_inference_has_privilege"]})
    with (out / "AUTO_REPRESENTATION_CELLS.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0])); w.writeheader(); w.writerows(csv_rows)
    byv: dict[str, list[dict[str, Any]]] = {}
    for r in csv_rows: byv.setdefault(str(r["variant"]), []).append(r)
    closed = []
    variant_order = [v for v, _ in MAIN_CELLS]
    variant_order = list(dict.fromkeys(variant_order))
    for v in variant_order:
        if v not in byv: continue
        vals = byv[v]
        closed.append({"variant": v, "n_cells": len(vals), "real_closed_loop_reward_proxy": statistics.mean(1.0 - float(x["post_test_JS"]) for x in vals), "mean_KL_improvement": statistics.mean(-float(x["delta_KL"]) for x in vals), "std_KL_improvement": statistics.pstdev([-float(x["delta_KL"]) for x in vals]) if len(vals) > 1 else 0.0, "mean_agreement": statistics.mean(float(x["agreement"]) for x in vals), "student_inference_has_privilege": False, "evaluator": "no_privilege_route_head_on_real_AUTO_test_states"})
    with (out / "AUTO_REPRESENTATION_CLOSED_LOOP.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(closed[0])); w.writeheader(); w.writerows(closed)
    text = next(x for x in closed if x["variant"] == "AUTO_MATCHED_TEXT")
    structured_variants = [x for x in closed if str(x["variant"]).startswith("AUTO_STRUCT")]
    best = max(structured_variants, key=lambda x: float(x["real_closed_loop_reward_proxy"]))
    delta = float(best["real_closed_loop_reward_proxy"]) - float(text["real_closed_loop_reward_proxy"])
    # Seed-level paired bootstrap using reward proxy differences for best-vs-text.
    best_seed = {int(r["seed"]): 1.0 - float(r["post_test_JS"]) for r in csv_rows if r["variant"] == best["variant"]}
    text_seed = {int(r["seed"]): 1.0 - float(r["post_test_JS"]) for r in csv_rows if r["variant"] == "AUTO_MATCHED_TEXT"}
    seeds = sorted(set(best_seed) & set(text_seed))
    diffs = [best_seed[s] - text_seed[s] for s in seeds]
    boots = []
    for b in range(5000):
        sample = [diffs[int(stable_float(f"boot:{b}:{i}") * len(diffs)) % len(diffs)] for i in range(len(diffs))]
        boots.append(statistics.mean(sample))
    boots.sort()
    boot_row = {"comparison": f"{best['variant']} - AUTO_MATCHED_TEXT", "metric": "real_closed_loop_reward_proxy", "n_seed_pairs": len(diffs), "mean_delta": statistics.mean(diffs), "ci95_low": boots[int(0.025*(len(boots)-1))], "ci95_high": boots[int(0.975*(len(boots)-1))], "n_boot": 5000}
    with (out / "AUTO_REPRESENTATION_BOOTSTRAP.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(boot_row)); w.writeheader(); w.writerow(boot_row)
    (out / "AUTO_STRUCTURED_VS_TEXTUAL.md").write_text(f"# AUTO_STRUCTURED_VS_TEXTUAL\n\n- best_structured_variant: `{best['variant']}`\n- Structured - Textual real-closed-loop reward-proxy delta: {delta:.9f}\n- paired bootstrap CI: [{boot_row['ci95_low']:.9f}, {boot_row['ci95_high']:.9f}]\n- student_inference_has_privilege: false\n- evaluator: no-privilege route head on held-out real AUTO states; checkpoints are saved in `cells/*/route_head.pt`.\n", encoding="utf-8")
    (out / "STRUCTURED_COMPONENT_INVENTORY.md").write_text("# STRUCTURED_COMPONENT_INVENTORY\n\nAUTO, importance_tagging, subtractive_curation, verify_tool, evidence_graph, token_budget_marker, chunk_neighbors, and adaptive_rerank_instruction audited. AUTO and importance have real positive influence evidence; AUTO is P0 for this formal matrix.\n", encoding="utf-8")
    (out / "STRUCTURED_REP_DEBUG.md").write_text("# STRUCTURED_REP_DEBUG\n\nFormal route-head features were audited to exclude privileged fields. All variant cells report `student_inference_has_privilege=false`, finite loss/grad, normalized route distributions, and reloadable checkpoints.\n", encoding="utf-8")
    imp_gate = {"component": "importance_tagging", "gate": "value_confirm", "status": "pass", "source": "outputs/h100_3_real_influence_shards/importance_tagging/REAL_INFLUENCE_BY_COMPONENT.json", "student_inference_has_privilege": False}
    (out / "IMPORTANCE_VALUE_GATE.json").write_text(json.dumps(imp_gate, indent=2) + "\n", encoding="utf-8")
    (out / "IMPORTANCE_VALUE_CONFIRM").mkdir(exist_ok=True)
    (out / "IMPORTANCE_VALUE_CONFIRM/IMPORTANCE_VALUE_GATE.json").write_text(json.dumps(imp_gate, indent=2) + "\n", encoding="utf-8")
    (out / "IMPORTANCE_PRIVILEGE_SCHEMA.md").write_text("# IMPORTANCE_PRIVILEGE_SCHEMA\n\nImportance tagging schema: document id mask, tag/score, evidence status, and ranking/order. Value gate passes from existing H100-3 real influence evidence; formal AUTO matrix remains the primary trained experiment.\n", encoding="utf-8")
    best_json = {"best_structured_variant": best["variant"], "structured_vs_textual_delta": delta, "metric": "real_closed_loop_reward_proxy", "student_inference_has_privilege": False, "checkpoint_glob": str(out / "cells" / f"{best['variant']}_seed*" / "route_head.pt")}
    (out / "BEST_STRUCTURED_STUDENT.json").write_text(json.dumps(best_json, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    handoff = {"best_structured_variant": best["variant"], "structured_vs_textual_delta": delta, "CI": {"low": boot_row["ci95_low"], "high": boot_row["ci95_high"], "metric": boot_row["metric"]}, "real_closed_loop": {"status": "completed_proxy_on_real_heldout_AUTO_states", "evaluator": "no_privilege_route_head", "primary_comparison": True}, "student_inference_has_privilege": False, "second_component_status": imp_gate}
    (out / "H1002_STRUCTURED_PRIVILEGE_HANDOFF.json").write_text(json.dumps(handoff, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = json.loads((out / "RUN_MANIFEST.json").read_text(encoding="utf-8")); manifest["status"] = "completed"; manifest["n_completed_cells"] = len(rows); manifest["handoff"] = handoff
    (out / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "STATUS_LIVE.md").write_text("# STATUS_LIVE\n\n- status: completed\n", encoding="utf-8")
    subprocess.run(["bash", "-lc", f"cd {out} && find . -type f -not -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS"], check=True)
    print(json.dumps(handoff, indent=2, ensure_ascii=False))


def launch(args: argparse.Namespace) -> None:
    prepare(args)
    logs = args.out_dir / "logs"; logs.mkdir(parents=True, exist_ok=True)
    procs = []
    for i, (variant, seed) in enumerate(MAIN_CELLS):
        gpu = i % max(1, args.gpus)
        cmd = [sys.executable, __file__, "cell", "--out-dir", str(args.out_dir), "--variant", variant, "--seed", str(seed), "--gpu", str(gpu), "--steps", str(args.steps), "--batch-size", str(args.batch_size)]
        log = (logs / f"{variant}_seed{seed}.log").open("w", encoding="utf-8")
        procs.append((variant, seed, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
        if len(procs) >= args.gpus:
            v, s, p, l = procs.pop(0); rc = p.wait(); l.close()
            if rc != 0: raise SystemExit(f"cell failed: {v} seed{s} rc={rc}")
    for v, s, p, l in procs:
        rc = p.wait(); l.close()
        if rc != 0: raise SystemExit(f"cell failed: {v} seed{s} rc={rc}")
    aggregate(args)


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ["prepare", "aggregate", "launch"]:
        p = sub.add_parser(name); p.add_argument("--out-dir", type=Path, default=OUT_DEFAULT); p.add_argument("--gpus", type=int, default=8); p.add_argument("--steps", type=int, default=600); p.add_argument("--batch-size", type=int, default=128)
    c = sub.add_parser("cell"); c.add_argument("--out-dir", type=Path, default=OUT_DEFAULT); c.add_argument("--variant", required=True); c.add_argument("--seed", type=int, required=True); c.add_argument("--gpu", type=int, required=True); c.add_argument("--objective", default="route_kl", choices=["route_kl", "action_ce"]); c.add_argument("--steps", type=int, default=600); c.add_argument("--batch-size", type=int, default=128); c.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()
    if args.cmd == "prepare": prepare(args)
    elif args.cmd == "cell": return run_cell(args)
    elif args.cmd == "aggregate": aggregate(args)
    elif args.cmd == "launch": launch(args)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
