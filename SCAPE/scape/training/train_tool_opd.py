"""Tool-token OPD training entry (distillation-only V0).

Heavy model loops are optional. This module defines the SCAPE training contract
and a dry-run path used by unit tests / launchers before GPU jobs start.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from scape.common.manifest import build_run_manifest, finalize_run_manifest, write_run_manifest
from scape.common.status import write_status_live
from scape.training.harness_dropout import DropoutSchedule
from scape.training.teacher import FullViewTeacher, TeacherConfig
from scape.training.tool_opd import learnability_score, tool_opd_loss


def run_micro_distill(
    *,
    output_dir: Path,
    component_id: str,
    n_samples: int,
    seed: int,
    base_checkpoint: str,
    d_pre: float,
    synthetic_d_post: float | None = None,
    dry_run: bool = True,
    teacher_strategy: str = "ema",
    same_state_jsonl: Path | None = None,
    epochs: int = 1,
    opd_mode: str = "sr_opd",
    projection_max_events: int = 8,
    projection_max_macro_actions: int = 3,
    legacy_teacher_kl_weight: float = 0.0,
    projection_audit_path: Path | None = None,
) -> dict[str, Any]:
    """Train (or dry-run) one sample-size cell from a fixed base checkpoint.

    Important: 512 / 2k / 8k cells must each start from the same base_checkpoint,
    not continue from the previous cell's weights (unless a separate curriculum
    experiment is explicitly named).

    When dry_run=False, uses scape.training.hf_tool_opd (true tool-token KL).
    Never calls SCOPE train_opd.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    teacher = FullViewTeacher(
        config=TeacherConfig(strategy=teacher_strategy, strategy_lock_id="scape_teacher_v0_ema")
    )
    schedule = DropoutSchedule(
        target_components=[component_id],
        total_steps=max(1, n_samples // 8),
        mode="linear",
        seed=seed,
    )

    repo = Path(__file__).resolve().parents[2]
    manifest = build_run_manifest(
        run_id=f"L-{component_id}-n{n_samples}-s{seed}",
        stage="L",
        command=["python", "-m", "scape.training.train_tool_opd"],
        repo_root=repo,
        output_dir=output_dir,
        extra={
            "component_id": component_id,
            "n_samples": n_samples,
            "seed": seed,
            "base_checkpoint": base_checkpoint,
            "dry_run": dry_run,
            "legacy_scope_path_used": False,
            "opd_mode": opd_mode,
            "projection_schema_version": "scape_projection_v1",
            "loss_impl": (
                "scape.training.hf_tool_opd:sr_opd_ce"
                if opd_mode == "sr_opd"
                else "scape.training.hf_tool_opd:tool_token_kl"
            ),
            "projection_max_events": projection_max_events,
            "projection_max_macro_actions": projection_max_macro_actions,
            "legacy_teacher_kl_weight": legacy_teacher_kl_weight,
            **teacher.manifest_fields(),
            "dropout": schedule.to_dict(),
        },
    )
    write_run_manifest(output_dir / "RUN_MANIFEST.json", manifest)

    if not dry_run:
        from scape.collection.same_state import collect_same_state_dataset, load_same_state_jsonl
        from scape.training.hf_tool_opd import ScapeHFToolOPD, run_tool_opd_train

        if same_state_jsonl and Path(same_state_jsonl).is_file():
            rows = load_same_state_jsonl(Path(same_state_jsonl))[:n_samples]
        else:
            rows = collect_same_state_dataset(
                n_states=n_samples, component_id=component_id, seed=seed
            )
        eval_rows = rows[: max(1, min(32, len(rows) // 4))]
        backend = ScapeHFToolOPD(
            model_path=base_checkpoint,
            legacy_teacher_kl_weight=legacy_teacher_kl_weight,
        )
        loss_path = "sr_opd_ce" if opd_mode == "sr_opd" else "tool_token_kl"
        trained = run_tool_opd_train(
            backend, rows, eval_rows, loss_path=loss_path, epochs=epochs
        )
        summary = {
            "component_id": component_id,
            "n_samples": n_samples,
            "seed": seed,
            "base_checkpoint": base_checkpoint,
            "d_pre": trained["D_pre"],
            "d_post": trained["D_post"],
            "L_m": trained["L_m"],
            "mean_loss": trained["mean_train_loss"],
            "dry_run": False,
            "legacy_scope_path_used": False,
            "loss_impl": trained["loss_impl"],
            "teacher": teacher.manifest_fields(),
            **{k: trained[k] for k in trained if k.startswith(("name_", "arg_"))},
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        write_status_live(
            output_dir / "STATUS_LIVE.md",
            stage="L",
            run_id=manifest["run_id"],
            n_expected=1,
            n_finished=1,
            extra={"L_m": summary["L_m"]},
        )
        write_run_manifest(
            output_dir / "RUN_MANIFEST.json",
            finalize_run_manifest(manifest, exit_code=0, completed_shards=["main"]),
        )
        return summary

    # Dry-run scaffolding path
    losses = []
    for step in range(0, 8):
        mask = schedule.sample_mask(step)
        loss = tool_opd_loss(tool_token_kl=0.5 / (step + 1), anchor_kl=0.05)
        losses.append({"step": step, "loss": loss["loss"], "mask": mask})
        teacher.update_from_student({"step": float(step)})

    d_post = float(synthetic_d_post if synthetic_d_post is not None else max(0.05, d_pre * 0.6))
    summary = {
        "component_id": component_id,
        "n_samples": n_samples,
        "seed": seed,
        "base_checkpoint": base_checkpoint,
        "d_pre": d_pre,
        "d_post": d_post,
        "L_m": learnability_score(d_pre, d_post),
        "mean_loss": sum(x["loss"] for x in losses) / len(losses),
        "dry_run": dry_run,
        "legacy_scope_path_used": False,
        "opd_mode": opd_mode,
        "projection_schema_version": "scape_projection_v1",
        "teacher": teacher.manifest_fields(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_status_live(
        output_dir / "STATUS_LIVE.md",
        stage="L",
        run_id=manifest["run_id"],
        n_expected=1,
        n_finished=1,
        extra={"L_m": summary["L_m"]},
    )
    write_run_manifest(
        output_dir / "RUN_MANIFEST.json",
        finalize_run_manifest(manifest, exit_code=0, completed_shards=["main"]),
    )
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--component-id", required=True)
    ap.add_argument("--n-samples", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--base-checkpoint", required=True)
    ap.add_argument("--d-pre", type=float, default=1.0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true", default=False)
    ap.add_argument("--no-dry-run", action="store_true", default=False)
    ap.add_argument("--same-state-jsonl", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument(
        "--opd-mode",
        choices=("sr_opd", "legacy_same_action"),
        default="sr_opd",
        help="Formal path is sr_opd. legacy_same_action is regression only.",
    )
    ap.add_argument("--projection-max-events", type=int, default=8)
    ap.add_argument("--projection-max-macro-actions", type=int, default=3)
    ap.add_argument("--reject-nonrealizable", action="store_true", default=True)
    ap.add_argument("--projection-audit-path", type=Path, default=None)
    ap.add_argument("--legacy-teacher-kl-weight", type=float, default=0.0)
    args = ap.parse_args(argv)
    dry = True
    if args.no_dry_run:
        dry = False
    elif args.dry_run:
        dry = True
    summary = run_micro_distill(
        output_dir=args.out,
        component_id=args.component_id,
        n_samples=args.n_samples,
        seed=args.seed,
        base_checkpoint=args.base_checkpoint,
        d_pre=args.d_pre,
        dry_run=dry,
        same_state_jsonl=args.same_state_jsonl,
        epochs=args.epochs,
        opd_mode=args.opd_mode,
        projection_max_events=args.projection_max_events,
        projection_max_macro_actions=args.projection_max_macro_actions,
        legacy_teacher_kl_weight=args.legacy_teacher_kl_weight,
        projection_audit_path=args.projection_audit_path,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
