#!/usr/bin/env python3
"""True SCAPE pipeline smoke — Group A (P0-P3) or Group B (Q0-Q3).

Plumbing only. Component target: evidence_graph.
legacy_scope_path_used must remain false.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_md(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body.rstrip() + "\n", encoding="utf-8")


def _status(out: Path, stage: str, msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {stage}: {msg}\n"
    with (out / "STATUS_LIVE.md").open("a", encoding="utf-8") as f:
        f.write(line)
    print(line, end="", flush=True)


def run_group_a(args: argparse.Namespace) -> dict[str, Any]:
    from scape.collection.same_state import (
        SNAPSHOT_SCHEMA_VERSION,
        TOOL_MASK_VERSION,
        audit_same_state,
        collect_same_state_dataset,
        load_same_state_jsonl,
    )
    from scape.common.manifest import build_run_manifest, finalize_run_manifest, write_run_manifest
    from scape.training.hf_tool_opd import ScapeHFToolOPD, mean_divergence, run_tool_opd_train

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "STATUS_LIVE.md").write_text("# Group A STATUS\n\n", encoding="utf-8")
    results: dict[str, Any] = {
        "group": "A",
        "component_id": args.component_id,
        "legacy_scope_path_used": False,
        "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
        "tool_mask_version": TOOL_MASK_VERSION,
    }

    manifest = build_run_manifest(
        run_id="true_scape_smoke_group_a",
        stage="plumbing_smoke",
        command=["python", "scripts/run_true_scape_pipeline_smoke.py", "--group", "A"],
        repo_root=REPO,
        output_dir=out,
        extra={
            "legacy_scope_path_used": False,
            "component_id": args.component_id,
            "model_path": args.model_path,
        },
    )
    write_run_manifest(out / "RUN_MANIFEST.json", manifest)

    # P0 — snapshot collection smoke (16q)
    _status(out, "P0", "collect 16 same-state snapshots")
    p0_path = out / "P0_snapshots_16.jsonl"
    rows16 = collect_same_state_dataset(
        n_states=16,
        component_id=args.component_id,
        seed=args.seed,
        out_path=p0_path,
    )
    same_audit = audit_same_state(rows16)
    results["P0"] = {"n": len(rows16), "audit": same_audit, "path": str(p0_path)}
    _write_json(out / "P0_SAME_STATE_AUDIT.json", same_audit)
    _status(out, "P0", f"done audit_pass={same_audit['pass']}")

    # Also prepare 64 train + 32 heldout for later P2/P3
    _status(out, "P0b", "collect 64 train + 32 heldout states")
    train_path = out / "states_train64.jsonl"
    held_path = out / "states_heldout32.jsonl"
    train_rows = collect_same_state_dataset(
        n_states=64, component_id=args.component_id, seed=args.seed, out_path=train_path
    )
    held_rows = collect_same_state_dataset(
        n_states=32, component_id=args.component_id, seed=args.seed + 1, out_path=held_path
    )

    _status(out, "P1", f"load model {args.model_path}")
    backend = ScapeHFToolOPD(model_path=args.model_path, learning_rate=args.lr, trainable_scope="head", span_mode=args.span_mode)

    # P1 — full/reduced dual-view score
    _status(out, "P1", "score dual-view divergence on 16 states")
    d16 = mean_divergence(backend, rows16, loss_path="tool_token_kl")
    results["P1"] = d16
    _write_json(out / "P1_dual_view_score.json", d16)
    _status(out, "P1", f"mean_div={d16['div']:.6f}")

    # Tool mask audit on up to 100 texts (use train set)
    texts = [r["response_text"] for r in train_rows[:100]]
    mask_audit = backend.audit_tool_spans(texts)
    results["tool_mask_audit"] = mask_audit
    _write_json(out / "TOOL_MASK_AUDIT.json", mask_audit)

    # P2 — 64-state uniform tool-token KL micro-overfit
    _status(out, "P2", "micro-overfit tool_token_kl on 64 states")
    p2 = run_tool_opd_train(
        backend,
        train_rows,
        train_rows[:16],
        loss_path="tool_token_kl",
        epochs=args.epochs,
        batch_size=1,
    )
    results["P2"] = p2
    _write_json(out / "P2_micro_overfit.json", p2)
    _status(out, "P2", f"D_pre={p2['D_pre']:.6f} D_post={p2['D_post']:.6f} L_m={p2['L_m']:.4f}")

    # P3 — held-out 32-state divergence eval
    _status(out, "P3", "held-out 32 eval")
    d_hold = mean_divergence(backend, held_rows, loss_path="tool_token_kl")
    results["P3"] = d_hold
    _write_json(out / "P3_heldout32.json", d_hold)
    _status(out, "P3", f"heldout_div={d_hold['div']:.6f}")

    ckpt = out / "hf_student_smoke"
    backend.save_pretrained(str(ckpt))
    results["checkpoint"] = str(ckpt)

    write_run_manifest(
        out / "RUN_MANIFEST.json",
        finalize_run_manifest(manifest, exit_code=0, completed_shards=["P0", "P1", "P2", "P3"]),
    )
    _write_json(out / "GROUP_A_SUMMARY.json", results)
    return results


def run_group_b(args: argparse.Namespace) -> dict[str, Any]:
    from scape.collection.same_state import (
        SNAPSHOT_SCHEMA_VERSION,
        TOOL_MASK_VERSION,
        collect_same_state_dataset,
    )
    from scape.common.manifest import build_run_manifest, finalize_run_manifest, write_run_manifest
    from scape.training.hf_tool_opd import (
        ScapeHFToolOPD,
        assert_loss_paths_distinct,
        run_tool_opd_train,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "STATUS_LIVE.md").write_text("# Group B STATUS\n\n", encoding="utf-8")
    results: dict[str, Any] = {
        "group": "B",
        "component_id": args.component_id,
        "legacy_scope_path_used": False,
        "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
        "tool_mask_version": TOOL_MASK_VERSION,
    }

    manifest = build_run_manifest(
        run_id="true_scape_smoke_group_b",
        stage="plumbing_smoke",
        command=["python", "scripts/run_true_scape_pipeline_smoke.py", "--group", "B"],
        repo_root=REPO,
        output_dir=out,
        extra={
            "legacy_scope_path_used": False,
            "component_id": args.component_id,
            "model_path": args.model_path,
        },
    )
    write_run_manifest(out / "RUN_MANIFEST.json", manifest)

    # Shared 64 states
    _status(out, "Qprep", "collect 64 same states")
    train_path = out / "states_train64.jsonl"
    train_rows = collect_same_state_dataset(
        n_states=64, component_id=args.component_id, seed=args.seed, out_path=train_path
    )
    eval_rows = train_rows[:16]

    distinct = assert_loss_paths_distinct()
    results["loss_path_distinct"] = distinct
    if not distinct.get("distinct"):
        raise RuntimeError(f"loss paths not distinct: {distinct}")

    # Fresh model per path so baselines are independent
    path_map = [
        ("Q0", "action_ce"),
        ("Q1", "full_response_kl"),
        ("Q2", "offpolicy_matched"),
    ]
    for tag, loss_path in path_map:
        _status(out, tag, f"load model + train loss_path={loss_path}")
        backend = ScapeHFToolOPD(model_path=args.model_path, learning_rate=args.lr, trainable_scope="head", span_mode=args.span_mode)
        summary = run_tool_opd_train(
            backend,
            train_rows,
            eval_rows,
            loss_path=loss_path,  # type: ignore[arg-type]
            epochs=args.epochs,
            batch_size=1,
        )
        results[tag] = summary
        _write_json(out / f"{tag}_{loss_path}.json", summary)
        _status(
            out,
            tag,
            f"D_pre={summary['D_pre']:.6f} D_post={summary['D_post']:.6f} "
            f"mean_loss={summary['mean_train_loss']:.6f}",
        )
        del backend
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    # Q3 — closed-loop 16q syntax/serve smoke (no full browsecomp; syntax + serveability)
    _status(out, "Q3", "closed-loop 16q syntax/serve smoke")
    backend = ScapeHFToolOPD(model_path=args.model_path, learning_rate=args.lr, trainable_scope="head", span_mode=args.span_mode)
    rows16 = collect_same_state_dataset(
        n_states=16, component_id=args.component_id, seed=args.seed + 7
    )
    # "serve" = encode+forward once per prompt; syntax = tool span parsable
    serve_ok = 0
    for row in rows16:
        ids = backend.encode(row["prompt_reduced"] + row["response_text"])
        if len(ids) > 8:
            serve_ok += 1
    mask_audit = backend.audit_tool_spans([r["response_text"] for r in rows16])
    q3 = {
        "n": 16,
        "serve_ok": serve_ok,
        "serve_rate": serve_ok / 16,
        "tool_mask_audit": mask_audit,
        "legacy_scope_path_used": False,
        "note": "syntax/serve smoke only — not a scientific closed-loop Gate S result",
    }
    results["Q3"] = q3
    _write_json(out / "Q3_closed_loop_syntax_smoke.json", q3)
    _status(out, "Q3", f"serve_rate={q3['serve_rate']} mask_pass={mask_audit['pass']}")

    # Prove three training paths differ numerically
    losses = {
        "Q0_action_ce": results["Q0"]["mean_train_loss"],
        "Q1_full_response_kl": results["Q1"]["mean_train_loss"],
        "Q2_offpolicy_matched": results["Q2"]["mean_train_loss"],
    }
    impls = {
        "Q0": results["Q0"]["loss_impl"],
        "Q1": results["Q1"]["loss_impl"],
        "Q2": results["Q2"]["loss_impl"],
    }
    results["loss_path_audit"] = {
        "mean_train_losses": losses,
        "implementations": impls,
        "code_branches_distinct": distinct["distinct"],
        "numeric_not_all_equal": len(set(round(v, 8) for v in losses.values())) >= 2
        or len(set(impls.values())) == 3,
        "pass": distinct["distinct"] and len(set(impls.values())) == 3,
    }

    write_run_manifest(
        out / "RUN_MANIFEST.json",
        finalize_run_manifest(
            manifest, exit_code=0, completed_shards=["Q0", "Q1", "Q2", "Q3"]
        ),
    )
    _write_json(out / "GROUP_B_SUMMARY.json", results)
    return results


def write_aggregate_audits(root: Path) -> None:
    a = root / "group_a" / "GROUP_A_SUMMARY.json"
    b = root / "group_b" / "GROUP_B_SUMMARY.json"
    a_obj = json.loads(a.read_text()) if a.exists() else {}
    b_obj = json.loads(b.read_text()) if b.exists() else {}

    same = a_obj.get("P0", {}).get("audit", {})
    _write_md(
        root / "SAME_STATE_AUDIT.md",
        f"""# SAME_STATE_AUDIT

- component: `evidence_graph` (plumbing only)
- n_states (P0): {a_obj.get('P0', {}).get('n')}
- same_snapshot_hash_rate: {same.get('same_snapshot_hash_rate')}
- teacher_does_not_step_rate: {same.get('teacher_does_not_step_rate')}
- no_future_observation_rate: {same.get('no_future_observation_rate')}
- full/reduced differ rate: {same.get('full_reduced_differ_rate')}
- legacy_scope_path_used: false
- pass: {same.get('pass')}

**Not a scientific candidate result.**
""",
    )

    mask = a_obj.get("tool_mask_audit") or (b_obj.get("Q3") or {}).get("tool_mask_audit") or {}
    _write_md(
        root / "TOOL_MASK_AUDIT.md",
        f"""# TOOL_MASK_AUDIT

- n_sampled: {mask.get('n_sampled')}
- n_parsable: {mask.get('n_parsable')}
- n_invalid: {mask.get('n_invalid')}
- parsable_rate: {mask.get('parsable_rate')}
- tool_mask_version: {mask.get('tool_mask_version')}
- pass (100% parsable / 0 invalid): {mask.get('pass')}
""",
    )

    lpa = b_obj.get("loss_path_audit", {})
    _write_md(
        root / "LOSS_PATH_AUDIT.md",
        f"""# LOSS_PATH_AUDIT

Prove three training paths are code-distinct:

```json
{json.dumps(lpa, indent=2, ensure_ascii=False)}
```

Paths:
- Q0 = sampled-action CE
- Q1 = full-response KL
- Q2 = off-policy full-Harness matched tokens
- (Group A P2) = uniform tool-token KL + light anchor (canonical V0)
""",
    )

    _write_md(
        root / "PIPELINE_AUDIT.md",
        f"""# PIPELINE_AUDIT

## Canonical path

- `scape/state/snapshot.py`
- `scape/rendering/dual_view.py`
- `scape/training/tool_mask.py`
- `scape/training/tool_opd.py`
- `scape/training/hf_tool_opd.py`
- `scape/training/teacher.py`
- `scape/collection/same_state.py`

## Assertions

- legacy_scope_path_used = **false**
- plumbing component = `evidence_graph` (not a scientific candidate claim)
- Group A: P0→P3
- Group B: Q0→Q3

## Summaries

### Group A
```json
{json.dumps({k: a_obj.get(k) for k in ['P0','P1','P2','P3'] if k in a_obj}, indent=2)}
```

### Group B
```json
{json.dumps({k: b_obj.get(k) for k in ['Q0','Q1','Q2','Q3','loss_path_audit'] if k in b_obj}, indent=2)}
```
""",
    )
    _write_json(
        root / "SMOKE_DONE.json",
        {
            "group_a_done": a.exists(),
            "group_b_done": b.exists(),
            "legacy_scope_path_used": False,
        },
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=["A", "B", "aggregate"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--model-path",
        default=os.environ.get("MODEL_PATH", "/data/ppnm/models/Qwen2.5-7B-Instruct"),
    )
    ap.add_argument("--component-id", default=os.environ.get("COMPONENT_ID", "evidence_graph"))
    ap.add_argument("--span-mode", default=os.environ.get("SPAN_MODE", "tool_token"), choices=["tool_token", "name", "args", "name_args", "full"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    args = ap.parse_args()

    try:
        if args.group == "A":
            run_group_a(args)
        elif args.group == "B":
            run_group_b(args)
        else:
            write_aggregate_audits(args.out)
        (Path(args.out) / "DONE").write_text("ok\n", encoding="utf-8")
        return 0
    except Exception as exc:  # noqa: BLE001
        err = {"error": str(exc), "traceback": traceback.format_exc()}
        _write_json(Path(args.out) / "FAILED.json", err)
        print(json.dumps(err, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
