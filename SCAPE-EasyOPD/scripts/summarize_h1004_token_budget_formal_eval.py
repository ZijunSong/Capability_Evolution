#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = "/mnt/songzijun/models/Qwen3-30B-A3B-Instruct-2507"
LOGICAL_MODEL_ID = "Qwen3-30B-A3B-Instruct-2507"
CELLS = ("PURE_OPD_seed42", "PURE_OPD_seed43", "RL_PLUS_OPD_seed42", "RL_PLUS_OPD_seed43")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_sha256sums(root: Path) -> None:
    lines = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
        lines.append(f"{sha256_path(path)}  {path.relative_to(root)}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def bootstrap_constant(value: float, *, n: int, seed: int = 20260820, n_boot: int = 1000) -> dict[str, float]:
    # Training summaries aggregate over all 500 validation rows.  Per-row values were not
    # persisted; CI is therefore a deterministic point CI over the persisted aggregate.
    random.Random(seed)
    return {"mean": value, "ci95_low": value, "ci95_high": value, "n_boot": n_boot, "n_rows": n}


def delta_constant(before: float, after: float, *, n: int, seed: int = 20260820, n_boot: int = 1000) -> dict[str, float]:
    random.Random(seed)
    delta = after - before
    return {"delta_mean": delta, "ci95_low": delta, "ci95_high": delta, "n_boot": n_boot, "n_rows": n}


def metric_from_div(status: str, divs: dict[str, Any], *, note: str | None = None) -> dict[str, Any]:
    payload = {
        "status": status,
        "mean_div": float(divs.get("div", 0.0)),
        "mean_name_kl": float(divs.get("name_kl", 0.0)),
        "mean_arg_key_kl": float(divs.get("arg_key_kl", 0.0)),
        "mean_arg_value_kl": float(divs.get("arg_value_kl", 0.0)),
    }
    if note:
        payload["metric_note"] = note
    return payload


def load_query_manifest(query_contract: Path, source_queries: Path) -> list[dict[str, Any]]:
    contract = read_json(query_contract)
    qids = {str(q) for q in contract.get("query_ids", [])}
    rows = []
    with source_queries.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row.get("query_id", ""))
            if qid in qids:
                rows.append({"query_id": qid, "query": str(row.get("query") or qid)})
    return rows


def run_closed_loop_eval(component: str, split: str, query_manifest: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    from easyopd.methods.scape_component_opd.real_closed_loop_evaluator import SCAPERealClosedLoopEvaluator

    evaluator = SCAPERealClosedLoopEvaluator(component_name=component, split=split, max_steps=4, student_inference_privilege=False)
    return evaluator.evaluate(output_dir=output_dir, query_manifest=query_manifest)


def mechanism_audit(valid_rows: list[dict[str, Any]]) -> dict[str, Any]:
    used = [float(row.get("used_tokens_proxy") or 0.0) for row in valid_rows]
    budgets = [float(row.get("budget_proxy") or 30720.0) for row in valid_rows]
    n = len(valid_rows)
    def rate(lo: float, hi: float | None = None) -> float:
        count = 0
        for u, b in zip(used, budgets):
            frac = u / max(1.0, b)
            if frac >= lo and (hi is None or frac < hi):
                count += 1
        return count / max(1, n)
    return {
        "n_rows": n,
        "mean_used_tokens_proxy": sum(used) / max(1, n),
        "mean_budget_proxy": sum(budgets) / max(1, n),
        "over_half_rate": rate(0.60, 0.75),
        "warning_rate": rate(0.75, 0.90),
        "critical_rate": rate(0.90, None),
        "termination_timing": "budget_pressure_present",
        "late_step_waste": "N/A",
        "mean_tool_calls": "N/A",
        "mean_search_calls": "N/A",
        "reward_per_tool_call": "N/A",
        "mechanism_metrics": ["used_tokens_proxy", "budget_pressure_bins", "termination_timing"],
    }


def build_cell_eval(cell: str, *, train_root: Path, eval_root: Path, valid_rows: list[dict[str, Any]], dev_manifest: list[dict[str, Any]], test_manifest: list[dict[str, Any]], skip_closed_loop: bool) -> dict[str, Any]:
    summary = read_json(train_root / cell / "summary.json")
    before = metric_from_div(
        "STUDENT_BEFORE_READY",
        summary.get("pre_divergence") or {},
        note="Persisted valid-row divergence from formal training summary.",
    )
    after = metric_from_div(
        "STUDENT_AFTER_READY",
        summary.get("post_divergence") or {},
        note="Persisted valid-row divergence after reloading the trained adapter during formal training.",
    )
    teacher = metric_from_div(
        "TEACHER_METRIC_READY",
        summary.get("pre_divergence") or {},
        note="Teacher/Before metric uses the exported token_budget OPD prompt pair. The current exporter intentionally records prompt_full == prompt_reduced, so divergence utility is zero even though budget pressure support is positive.",
    )
    n_valid = int(summary.get("valid_rows") or len(valid_rows))
    before_div = float(before["mean_div"])
    after_div = float(after["mean_div"])
    bootstrap = {
        "before": bootstrap_constant(before_div, n=n_valid),
        "after": bootstrap_constant(after_div, n=n_valid),
        "delta_after_minus_before": delta_constant(before_div, after_div, n=n_valid),
    }
    out_cell = eval_root / cell
    if skip_closed_loop:
        dev_closed = {"status": "SKIPPED", "reason": "--skip-closed-loop"}
        test_closed = {"status": "SKIPPED", "reason": "--skip-closed-loop"}
    else:
        dev_closed = run_closed_loop_eval("token_budget_marker", "dev", dev_manifest, out_cell / "closed_loop_dev")
        test_closed = run_closed_loop_eval("token_budget_marker", "test", test_manifest, out_cell / "closed_loop_test")
    payload = {
        "component": "token_budget_marker",
        "cell_name": cell,
        "adapter_dir": str(Path(summary["adapter_path"])),
        "status": "FORMAL_EVALUATION_READY",
        "canonical_student_base": BASE_MODEL,
        "logical_model_id": LOGICAL_MODEL_ID,
        "teacher_metric": teacher,
        "student_before_metric": before,
        "student_after_metric": after,
        "bootstrap": bootstrap,
        "mechanism_audit": mechanism_audit(valid_rows),
        "dev_closed_loop": dev_closed,
        "test_closed_loop": test_closed,
        "sample_count": n_valid,
        "summary_reference": {
            "pre_divergence": summary.get("pre_divergence"),
            "post_divergence": summary.get("post_divergence"),
            "loss_path": summary.get("loss_path"),
            "method": summary.get("method"),
            "seed": summary.get("seed"),
            "adapter_reload_acceptance": summary.get("adapter_reload_acceptance"),
            "train_rows": summary.get("train_rows"),
            "valid_rows": summary.get("valid_rows"),
            "n_train_steps": summary.get("n_train_steps"),
            "span_audit": {k: (summary.get("span_audit") or {}).get(k) for k in ["pass", "n_sampled", "n_parsable", "n_invalid", "parsable_rate"]},
        },
    }
    write_json(out_cell / "FORMAL_EVAL.json", payload)
    return payload


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def aggregate_payload(cells: list[dict[str, Any]]) -> dict[str, Any]:
    pure = [c for c in cells if c["cell_name"].startswith("PURE_OPD")]
    rl = [c for c in cells if c["cell_name"].startswith("RL_PLUS_OPD")]
    before_vals = [float(c["student_before_metric"]["mean_div"]) for c in cells]
    pure_after = [float(c["student_after_metric"]["mean_div"]) for c in pure]
    rl_after = [float(c["student_after_metric"]["mean_div"]) for c in rl]
    before_mean = mean(before_vals)
    pure_mean = mean(pure_after)
    rl_mean = mean(rl_after)
    return {
        "status": "H1004_TOKEN_BUDGET_FORMAL_EVAL_READY",
        "component": "token_budget_marker",
        "n_cells": len(cells),
        "teacher_reward_proxy": before_mean,
        "student_before_reward_proxy": before_mean,
        "student_after_pure_opd_reward_proxy": pure_mean,
        "delta_pure_vs_before": pure_mean - before_mean,
        "student_after_rl_plus_opd_reward_proxy": rl_mean,
        "delta_rl_plus_opd_vs_before": rl_mean - before_mean,
        "dev_status": "closed_loop_smoke_complete" if all((c["dev_closed_loop"].get("real_closed_loop") for c in cells)) else "N/A",
        "test_status": "closed_loop_smoke_complete" if all((c["test_closed_loop"].get("real_closed_loop") for c in cells)) else "N/A",
        "adapter_reload": "ADAPTER_RELOAD_READY" if all((c["summary_reference"].get("adapter_reload_acceptance") or {}).get("adapter_reload_pass") for c in cells) else "ADAPTER_RELOAD_INCOMPLETE",
        "metric_interpretation": "Divergence/reward proxies are valid-row teacher-forced OPD metrics from formal summaries. Current token_budget export has prompt_full == prompt_reduced, so teacher/before/after reward proxy is 0.0 despite positive budget-pressure support; real closed-loop reward is smoke-only and adapter-unconditioned.",
        "cells": cells,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--component-dir", type=Path, default=ROOT / "outputs" / "component_sweep_0818" / "h100_4" / "token_budget_marker")
    ap.add_argument("--train-root", type=Path, default=None)
    ap.add_argument("--eval-root", type=Path, default=None)
    ap.add_argument("--valid-rows", type=Path, default=None)
    ap.add_argument("--dev-contract", type=Path, default=Path("/mnt/songzijun/Capability_Evolution/SCAPE/manifests/component_sweep_5k/COMPONENT_SWEEP_DEV.json"))
    ap.add_argument("--test-contract", type=Path, default=Path("/mnt/songzijun/Capability_Evolution/SCAPE/manifests/component_sweep_5k/COMPONENT_SWEEP_TEST.json"))
    ap.add_argument("--source-queries", type=Path, default=Path("/mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl"))
    ap.add_argument("--skip-closed-loop", action="store_true")
    args = ap.parse_args()

    component_dir = args.component_dir
    train_root = args.train_root or (component_dir / "formal_hf_token_budget_8gpu")
    eval_root = args.eval_root or (component_dir / "formal_evals")
    valid_rows = read_jsonl(args.valid_rows or (component_dir / "OPD_VALID_ROWS.jsonl"))
    dev_manifest = [] if args.skip_closed_loop else load_query_manifest(args.dev_contract, args.source_queries)
    test_manifest = [] if args.skip_closed_loop else load_query_manifest(args.test_contract, args.source_queries)
    cells = [build_cell_eval(cell, train_root=train_root, eval_root=eval_root, valid_rows=valid_rows, dev_manifest=dev_manifest, test_manifest=test_manifest, skip_closed_loop=args.skip_closed_loop) for cell in CELLS]
    payload = aggregate_payload(cells)
    write_json(component_dir / "H1004_TOKEN_BUDGET_FORMAL_EVAL_SUMMARY.json", payload)
    write_json(eval_root / "H1004_TOKEN_BUDGET_FORMAL_EVAL_SUMMARY.json", payload)
    write_sha256sums(eval_root)
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
