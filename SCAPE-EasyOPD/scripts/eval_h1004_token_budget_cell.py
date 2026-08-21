#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import random
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
SCAPE_ROOT = Path("/mnt/songzijun/Capability_Evolution/SCAPE")
BASE_MODEL = os.environ.get("CANONICAL_STUDENT_BASE", "/mnt/songzijun/models/Qwen3-30B-A3B-Instruct-2507")
LOGICAL_MODEL_ID = os.environ.get("SCAPE_STUDENT_LOGICAL_MODEL", "Qwen3-30B-A3B-Instruct-2507")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def remap_lora_state_dict(raw_state: dict[str, Any]) -> dict[str, Any]:
    remapped: dict[str, Any] = {}
    for key, value in raw_state.items():
        if key.endswith(".lora_A.weight"):
            remapped[key.replace(".lora_A.weight", ".lora_A.default.weight")] = value
        elif key.endswith(".lora_B.weight"):
            remapped[key.replace(".lora_B.weight", ".lora_B.default.weight")] = value
        else:
            remapped[key] = value
    return remapped


def load_backend(*, adapter_dir: Path | None = None):
    import sys

    sys.path.insert(0, str(SCAPE_ROOT))
    from scape.training.hf_tool_opd import ScapeHFToolOPD

    backend = ScapeHFToolOPD(model_path=BASE_MODEL, device_map={"": 0}, use_lora=False)
    if adapter_dir is None:
        return backend

    adapter_config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    lora_cfg = LoraConfig(
        task_type=adapter_config.get("task_type", "CAUSAL_LM"),
        r=int(adapter_config.get("r", 8)),
        lora_alpha=int(adapter_config.get("lora_alpha", 16)),
        lora_dropout=float(adapter_config.get("lora_dropout", 0.05)),
        target_modules=list(adapter_config.get("target_modules") or ["q_proj", "k_proj", "v_proj", "o_proj"]),
        bias=adapter_config.get("bias", "none"),
    )
    base = backend.model
    try:
        model = PeftModel.from_pretrained(base, adapter_dir)
        reload_path = "peft_model_from_pretrained"
    except Exception:
        model = get_peft_model(base, lora_cfg)
        raw_state = load_file(str(adapter_dir / "adapter_model.safetensors"))
        missing, unexpected = model.load_state_dict(remap_lora_state_dict(raw_state), strict=False)
        bad_missing = [key for key in missing if "lora_" in key]
        bad_unexpected = [key for key in unexpected if "lora_" in key]
        if bad_missing or bad_unexpected:
            raise RuntimeError(f"manual adapter reload mismatch: missing={bad_missing[:8]} unexpected={bad_unexpected[:8]}")
        reload_path = "manual_safetensors_state_dict"
    backend.model = model
    backend._device = next(model.parameters()).device
    backend.reload_path = reload_path
    return backend


def bootstrap_mean(values: list[float], *, seed: int = 20260820, n_boot: int = 1000) -> dict[str, float]:
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(len(values))] for _ in range(len(values))]
        means.append(sum(sample) / max(1, len(sample)))
    means.sort()
    return {
        "mean": sum(values) / max(1, len(values)),
        "ci95_low": means[int(0.025 * (n_boot - 1))],
        "ci95_high": means[int(0.975 * (n_boot - 1))],
        "n_boot": n_boot,
    }


def paired_bootstrap_delta(before: list[float], after: list[float], *, seed: int = 20260820, n_boot: int = 1000) -> dict[str, float]:
    if len(before) != len(after):
        raise ValueError("paired bootstrap requires equal length")
    rng = random.Random(seed)
    deltas = [a - b for a, b in zip(before, after)]
    idxs = list(range(len(deltas)))
    means: list[float] = []
    for _ in range(n_boot):
        sample = [deltas[rng.choice(idxs)] for _ in idxs]
        means.append(sum(sample) / max(1, len(sample)))
    means.sort()
    return {
        "delta_mean": sum(deltas) / max(1, len(deltas)),
        "ci95_low": means[int(0.025 * (n_boot - 1))],
        "ci95_high": means[int(0.975 * (n_boot - 1))],
        "n_boot": n_boot,
    }


def evaluate_rows(backend: Any, rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    values: list[dict[str, float]] = []
    for row in rows:
        values.append(
            backend.score_divergence(
                prompt_reduced=row["prompt_reduced"],
                prompt_full=row["prompt_full"],
                response_text=row["response_text"],
                loss_path="full_response_kl",
            )
        )
    return values


def aggregate(values: list[dict[str, float]]) -> dict[str, float]:
    keys = values[0].keys()
    return {k: sum(float(v[k]) for v in values) / max(1, len(values)) for k in keys}


def load_query_manifest(query_contract: Path, source_queries: Path) -> list[dict[str, Any]]:
    contract = json.loads(query_contract.read_text(encoding="utf-8"))
    qids = {str(q) for q in contract.get("query_ids", [])}
    rows: list[dict[str, Any]] = []
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


def mechanism_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    used = [float(row.get("used_tokens_proxy") or 0.0) for row in rows]
    budgets = [float(row.get("budget_proxy") or 30720.0) for row in rows]
    critical = sum(1 for u, b in zip(used, budgets) if u / max(1.0, b) >= 0.90)
    warning = sum(1 for u, b in zip(used, budgets) if 0.75 <= u / max(1.0, b) < 0.90)
    over_half = sum(1 for u, b in zip(used, budgets) if 0.60 <= u / max(1.0, b) < 0.75)
    n = len(rows)
    return {
        "n_rows": n,
        "mean_used_tokens_proxy": sum(used) / max(1, n),
        "mean_budget_proxy": sum(budgets) / max(1, n),
        "critical_rate": critical / max(1, n),
        "warning_rate": warning / max(1, n),
        "over_half_rate": over_half / max(1, n),
        "termination_timing": "budget_pressure_present",
        "late_step_waste": "N/A",
        "reward_per_tool_call": "N/A",
        "mechanism_metrics": ["used_tokens_proxy", "budget_pressure_bins", "termination_timing"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--component", default="token_budget_marker")
    ap.add_argument("--cell-name", required=True)
    ap.add_argument("--cell-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--valid-rows", type=Path, default=Path("/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD/outputs/component_sweep_0818/h100_4/token_budget_marker/OPD_VALID_ROWS.jsonl"))
    ap.add_argument("--dev-contract", type=Path, default=Path("/mnt/songzijun/Capability_Evolution/SCAPE/manifests/component_sweep_5k/COMPONENT_SWEEP_DEV.json"))
    ap.add_argument("--test-contract", type=Path, default=Path("/mnt/songzijun/Capability_Evolution/SCAPE/manifests/component_sweep_5k/COMPONENT_SWEEP_TEST.json"))
    ap.add_argument("--source-queries", type=Path, default=Path("/mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl"))
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--skip-closed-loop", action="store_true")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    rows = load_rows(args.valid_rows)
    if not rows:
        raise SystemExit(f"no valid rows found at {args.valid_rows}")

    summary = json.loads((args.cell_dir / "summary.json").read_text(encoding="utf-8"))
    adapter_dir = Path(summary["adapter_path"])
    base_backend = load_backend()
    before_stats = evaluate_rows(base_backend, rows)
    before_mean = aggregate(before_stats)
    del base_backend
    gc.collect()
    torch.cuda.empty_cache()

    after_backend = load_backend(adapter_dir=adapter_dir)
    after_stats = evaluate_rows(after_backend, rows)
    after_mean = aggregate(after_stats)
    reload_path = getattr(after_backend, "reload_path", "unknown")
    del after_backend
    gc.collect()
    torch.cuda.empty_cache()

    teacher_metric = {
        "status": "TEACHER_METRIC_READY",
        "mean_div": before_mean["div"],
        "mean_name_kl": before_mean["name_kl"],
        "mean_arg_key_kl": before_mean["arg_key_kl"],
        "mean_arg_value_kl": before_mean["arg_value_kl"],
        "metric_note": "teacher/student divergence on token_budget rows; current exporter has prompt_full == prompt_reduced, so reward utility is diagnostic and expected to be zero unless exporter changes.",
    }
    student_before_metric = {
        "status": "STUDENT_BEFORE_READY",
        "mean_div": before_mean["div"],
        "mean_name_kl": before_mean["name_kl"],
        "mean_arg_key_kl": before_mean["arg_key_kl"],
        "mean_arg_value_kl": before_mean["arg_value_kl"],
    }
    student_after_metric = {
        "status": "STUDENT_AFTER_READY",
        "mean_div": after_mean["div"],
        "mean_name_kl": after_mean["name_kl"],
        "mean_arg_key_kl": after_mean["arg_key_kl"],
        "mean_arg_value_kl": after_mean["arg_value_kl"],
        "adapter_reload_path": reload_path,
    }

    bootstrap = {
        "before": bootstrap_mean([row["div"] for row in before_stats]),
        "after": bootstrap_mean([row["div"] for row in after_stats]),
        "delta_after_minus_before": paired_bootstrap_delta([row["div"] for row in before_stats], [row["div"] for row in after_stats]),
    }
    audit = mechanism_audit(rows)

    if args.skip_closed_loop:
        dev_closed_loop = {"status": "SKIPPED", "reason": "--skip-closed-loop"}
        test_closed_loop = {"status": "SKIPPED", "reason": "--skip-closed-loop"}
    else:
        dev_query_manifest = load_query_manifest(args.dev_contract, args.source_queries)
        test_query_manifest = load_query_manifest(args.test_contract, args.source_queries)
        dev_closed_loop = run_closed_loop_eval(args.component, "dev", dev_query_manifest, args.output_dir / args.cell_name / "closed_loop_dev")
        test_closed_loop = run_closed_loop_eval(args.component, "test", test_query_manifest, args.output_dir / args.cell_name / "closed_loop_test")

    payload = {
        "component": args.component,
        "cell_name": args.cell_name,
        "adapter_dir": str(adapter_dir),
        "status": "FORMAL_EVALUATION_READY",
        "canonical_student_base": BASE_MODEL,
        "logical_model_id": LOGICAL_MODEL_ID,
        "teacher_metric": teacher_metric,
        "student_before_metric": student_before_metric,
        "student_after_metric": student_after_metric,
        "bootstrap": bootstrap,
        "mechanism_audit": audit,
        "dev_closed_loop": dev_closed_loop,
        "test_closed_loop": test_closed_loop,
        "sample_count": len(rows),
        "summary_reference": {
            "pre_divergence": summary.get("pre_divergence"),
            "post_divergence": summary.get("post_divergence"),
            "loss_path": summary.get("loss_path"),
            "method": summary.get("method"),
            "seed": summary.get("seed"),
            "adapter_reload_acceptance": summary.get("adapter_reload_acceptance"),
        },
    }

    out = args.output_dir / args.cell_name / "FORMAL_EVAL.json"
    write_json(out, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
