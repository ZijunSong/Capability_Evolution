#!/usr/bin/env python3
"""Dup-SDI training-loop diagnosis with checkpoint/resume support."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.scope.dup_diagnostics import (
    analyze_route_target_distribution,
    audit_skip_curate_realization,
    filter_by_routes,
    load_jsonl,
    offline_capability_eval,
    sample_balanced_subset,
    write_json,
)
from training.scope.sdi_trainer import DupSDITrainer, SDITrainConfig

STEPS = (
    "distribution",
    "heldout_offline_eval",
    "overfit_sanity",
    "ablation_D1_correct_only",
    "ablation_D2_endorse_only",
    "ablation_D3_unified",
    "finalize",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "outputs/dup_sdi_round1/diagnosis",
    )
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=_REPO_ROOT / "artifacts/datasets/dup_sdi_round1",
    )
    p.add_argument(
        "--base-model",
        type=str,
        default="/data/ppnm/models/Qwen2.5-7B-Instruct",
    )
    p.add_argument(
        "--adapter",
        type=Path,
        default=_REPO_ROOT / "outputs/dup_sdi_round1",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--overfit-adapter",
        type=Path,
        default=None,
        help="Reuse existing overfit adapter instead of retraining",
    )
    p.add_argument("--only-step", type=str, default="", help="Run a single step name")
    return p.parse_args()


def _progress_path(out: Path) -> Path:
    return out / "progress.json"


def _load_progress(out: Path) -> dict[str, Any]:
    p = _progress_path(out)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    partial = out / "diagnosis_report.partial.json"
    if partial.exists():
        report = json.loads(partial.read_text(encoding="utf-8"))
        completed: list[str] = []
        if report.get("supervision_distribution_all"):
            completed.append("distribution")
        if report.get("heldout_offline_eval"):
            completed.append("heldout_offline_eval")
        if report.get("overfit_sanity"):
            completed.append("overfit_sanity")
        ablations = report.get("ablations") or {}
        for step, key in (
            ("ablation_D1_correct_only", "D1_correct_only"),
            ("ablation_D2_endorse_only", "D2_endorse_only"),
            ("ablation_D3_unified", "D3_unified"),
        ):
            if key in ablations:
                completed.append(step)
        if (out / "diagnosis_report.json").exists():
            completed.append("finalize")
        return {"completed_steps": completed, "report": report}
    return {"completed_steps": [], "report": {"schema_version": "scope.dup_diagnosis.v1"}}


def _save_progress(out: Path, progress: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_json(_progress_path(out), progress)
    write_json(out / "diagnosis_report.partial.json", progress["report"])


def _done(progress: dict[str, Any], step: str) -> bool:
    return step in progress.get("completed_steps", [])


def _mark(progress: dict[str, Any], step: str, out: Path) -> None:
    if step not in progress["completed_steps"]:
        progress["completed_steps"].append(step)
    _save_progress(out, progress)


def _run_offline_pair(
    valid_rows: list[dict], base_model: str, adapter: Path, device: str
) -> dict[str, Any]:
    base_trainer = DupSDITrainer(
        SDITrainConfig(
            model_path=base_model,
            output_dir=Path("/tmp/dup_diag_base"),
            device=device,
        )
    )
    base_result = offline_capability_eval(
        base_trainer, valid_rows, model_tag="base"
    ).to_dict()
    del base_trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    trained_trainer = DupSDITrainer(
        SDITrainConfig(
            model_path=base_model,
            adapter_path=str(adapter),
            eval_only=True,
            output_dir=Path("/tmp/dup_diag_trained"),
            device=device,
        )
    )
    trained_result = offline_capability_eval(
        trained_trainer, valid_rows, model_tag="dup_sdi_round1"
    ).to_dict()
    del trained_trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"base": base_result, "trained": trained_result}


def _train_variant(
    *,
    name: str,
    train_rows: list[dict],
    eval_rows: list[dict],
    out_dir: Path,
    base_model: str,
    device: str,
    num_epochs: int,
    learning_rate: float,
    kl_coef: float,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / f"{name}_train.jsonl"
    eval_path = out_dir / f"{name}_eval.jsonl"
    adapter_dir = out_dir / name

    if adapter_dir.exists() and (adapter_dir / "adapter_config.json").exists():
        print(f"[diag] Reuse trained adapter for {name}", flush=True)
    else:
        train_path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in train_rows) + "\n",
            encoding="utf-8",
        )
        eval_path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in eval_rows) + "\n",
            encoding="utf-8",
        )
        cfg = SDITrainConfig(
            model_path=base_model,
            output_dir=adapter_dir,
            kl_coef=kl_coef,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            batch_size=4,
            grad_accum=2,
            device=device,
        )
        trainer = DupSDITrainer(cfg)
        train_summary = trainer.train(train_path, eval_path)
        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    trainer = DupSDITrainer(
        SDITrainConfig(
            model_path=base_model,
            adapter_path=str(adapter_dir),
            eval_only=True,
            output_dir=Path(f"/tmp/dup_diag_{name}"),
            device=device,
        )
    )
    heldout = offline_capability_eval(trainer, eval_rows, model_tag=name).to_dict()
    train_eval = offline_capability_eval(
        trainer, train_rows, model_tag=f"{name}_train"
    ).to_dict()
    result = {
        "heldout_eval": heldout,
        "train_eval": train_eval,
        "adapter_dir": str(adapter_dir),
    }
    if (adapter_dir / "train_summary.json").exists():
        result["train_summary"] = json.loads(
            (adapter_dir / "train_summary.json").read_text(encoding="utf-8")
        )
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _finalize_report(report: dict[str, Any]) -> dict[str, Any]:
    held = report.get("heldout_offline_eval") or {}
    base_acc = float((held.get("base") or {}).get("target_action_accuracy", 0.0))
    tr_acc = float((held.get("trained") or {}).get("target_action_accuracy", 0.0))
    if tr_acc <= base_acc + 0.02:
        branch = "A_offline_not_improved"
    else:
        branch = "B_offline_improved_but_rollout_worse"
    dist = report.get("supervision_distribution_all") or {}
    route_tgt = dist.get("route_x_target_action") or {}
    report["diagnosis_branch"] = branch
    report["diagnosis_notes"] = {
        "branch": branch,
        "base_target_action_accuracy": base_acc,
        "trained_target_action_accuracy": tr_acc,
        "endorse_curate_add_n": (route_tgt.get("ENDORSE") or {}).get("curate_add", 0),
        "correct_curate_replace_n": (route_tgt.get("CORRECT") or {}).get(
            "curate_replace", 0
        ),
        "unified_ce_reinforces_curate_add_via_endorse": (route_tgt.get("ENDORSE") or {}).get(
            "curate_add", 0
        ),
    }
    return report


def _should_run(step: str, only_step: str) -> bool:
    return not only_step or only_step == step


def main() -> None:
    args = parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    progress = _load_progress(out) if args.resume else {
        "completed_steps": [],
        "report": {"schema_version": "scope.dup_diagnosis.v1"},
    }
    report = progress["report"]

    train_rows = load_jsonl(args.dataset_dir / "train.jsonl")
    valid_rows = load_jsonl(args.dataset_dir / "valid.jsonl")
    all_rows = train_rows + valid_rows

    # --- distribution (CPU) ---
    if _should_run("distribution", args.only_step) and not _done(progress, "distribution"):
        print("[diag] Step: distribution", flush=True)
        report["supervision_distribution_all"] = analyze_route_target_distribution(all_rows)
        report["supervision_distribution_valid"] = analyze_route_target_distribution(
            valid_rows
        )
        report["skip_curate_audit"] = audit_skip_curate_realization(all_rows)
        _mark(progress, "distribution", out)

    # --- held-out offline eval ---
    if _should_run("heldout_offline_eval", args.only_step) and not _done(
        progress, "heldout_offline_eval"
    ):
        print("[diag] Step: heldout_offline_eval", flush=True)
        report["heldout_offline_eval"] = _run_offline_pair(
            valid_rows, args.base_model, args.adapter, args.device
        )
        _mark(progress, "heldout_offline_eval", out)

    # --- overfit sanity ---
    if _should_run("overfit_sanity", args.only_step) and not _done(progress, "overfit_sanity"):
        print("[diag] Step: overfit_sanity", flush=True)
        overfit_rows = sample_balanced_subset(
            train_rows, n_endorse=32, n_correct=32, seed=7
        )
        (out / "overfit64.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in overfit_rows) + "\n",
            encoding="utf-8",
        )
        overfit_adapter = args.overfit_adapter
        if overfit_adapter is None:
            default = (
                _REPO_ROOT
                / "outputs/dup_sdi_round1/minimal_runtime_smoke20/trained/overfit64"
            )
            if default.exists():
                overfit_adapter = default
        if overfit_adapter and overfit_adapter.exists():
            trainer = DupSDITrainer(
                SDITrainConfig(
                    model_path=args.base_model,
                    adapter_path=str(overfit_adapter),
                    eval_only=True,
                    output_dir=Path("/tmp/dup_diag_overfit"),
                    device=args.device,
                )
            )
            report["overfit_sanity"] = {
                "adapter_dir": str(overfit_adapter),
                "train_eval": offline_capability_eval(
                    trainer, overfit_rows, model_tag="overfit64_train"
                ).to_dict(),
            }
            if (overfit_adapter / "train_summary.json").exists():
                report["overfit_sanity"]["train_summary"] = json.loads(
                    (overfit_adapter / "train_summary.json").read_text(encoding="utf-8")
                )
            del trainer
        else:
            report["overfit_sanity"] = _train_variant(
                name="overfit64",
                train_rows=overfit_rows,
                eval_rows=overfit_rows,
                out_dir=out,
                base_model=args.base_model,
                device=args.device,
                num_epochs=15,
                learning_rate=5e-5,
                kl_coef=0.0,
            )
        _mark(progress, "overfit_sanity", out)

    # --- ablations ---
    ablation_specs = [
        ("ablation_D1_correct_only", "D1_correct_only", {"CORRECT"}),
        ("ablation_D2_endorse_only", "D2_endorse_only", {"ENDORSE"}),
        ("ablation_D3_unified", "D3_unified", {"CORRECT", "ENDORSE"}),
    ]
    if "ablations" not in report:
        report["ablations"] = {}

    for step_name, variant_name, routes in ablation_specs:
        if not _should_run(step_name, args.only_step):
            continue
        if _done(progress, step_name):
            continue
        print(f"[diag] Step: {step_name}", flush=True)
        sub_train = filter_by_routes(train_rows, routes)
        sub_valid = filter_by_routes(valid_rows, routes)
        report["ablations"][variant_name] = _train_variant(
            name=variant_name,
            train_rows=sub_train,
            eval_rows=sub_valid,
            out_dir=out / "ablations",
            base_model=args.base_model,
            device=args.device,
            num_epochs=5,
            learning_rate=2e-5,
            kl_coef=0.01,
        )
        trainer = DupSDITrainer(
            SDITrainConfig(
                model_path=args.base_model,
                adapter_path=str(out / "ablations" / variant_name),
                eval_only=True,
                output_dir=Path(f"/tmp/dup_diag_{variant_name}_full"),
                device=args.device,
            )
        )
        report["ablations"][variant_name]["heldout_full_valid"] = (
            offline_capability_eval(trainer, valid_rows, model_tag=variant_name).to_dict()
        )
        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _mark(progress, step_name, out)

    # --- finalize ---
    if _should_run("finalize", args.only_step) and not _done(progress, "finalize"):
        report = _finalize_report(report)
        write_json(out / "diagnosis_report.json", report)
        _mark(progress, "finalize", out)
        print(json.dumps(report["diagnosis_notes"], indent=2, ensure_ascii=False))
        print(f"[diag] Wrote {out / 'diagnosis_report.json'}", flush=True)


if __name__ == "__main__":
    main()
