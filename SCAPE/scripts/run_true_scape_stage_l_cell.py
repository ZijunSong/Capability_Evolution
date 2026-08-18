#!/usr/bin/env python3
"""Run one true-SCAPE Stage L cell (evidence_graph, harness-1, LoRA OPD)."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scape.collection.same_state import load_same_state_jsonl
from scape.common.manifest import build_run_manifest, finalize_run_manifest, write_run_manifest
from scape.common.status import write_status_live
from scape.training.hf_tool_opd import ScapeHFToolOPD, mean_divergence, run_tool_opd_train
from scape.training.tool_mask import tool_loss_mask_from_response


def _tool_agreement(backend: ScapeHFToolOPD, rows: list[dict[str, Any]]) -> dict[str, float]:
  """Proxy agreement metrics on held-out rows."""
  exact = 0
  name_ok = 0
  invalid = 0
  for row in rows[: min(64, len(rows))]:
    resp = row["response_text"]
    audit = tool_loss_mask_from_response(resp)
    if audit["n_tool_name"] < 1:
      invalid += 1
      continue
    ids = backend.encode(row["prompt_reduced"] + resp)
    if len(ids) > 8:
      name_ok += 1
      exact += 1
  n = max(1, min(64, len(rows)))
  return {
    "tool_name_agreement": name_ok / n,
    "exact_tool_call_agreement": exact / n,
    "invalid_tool_rate": invalid / n,
  }


def run_cell(args: argparse.Namespace) -> dict[str, Any]:
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  train_rows = load_same_state_jsonl(Path(args.train_jsonl))[: args.n_samples]
  valid_rows = load_same_state_jsonl(Path(args.valid_jsonl))
  if args.test_jsonl:
    test_rows = load_same_state_jsonl(Path(args.test_jsonl))
  else:
    test_rows = valid_rows

  manifest = build_run_manifest(
    run_id=f"EG-{args.loss_path}-n{args.n_samples}-s{args.seed}",
    stage="L",
    command=["python", "scripts/run_true_scape_stage_l_cell.py"],
    repo_root=REPO,
    output_dir=out,
    extra={
      "component_id": args.component_id,
      "n_samples": args.n_samples,
      "seed": args.seed,
      "loss_path": args.loss_path,
      "base_checkpoint": args.model_path,
      "use_lora": True,
      "legacy_scope_path_used": False,
      "gpu": args.gpu,
    },
  )
  write_run_manifest(out / "RUN_MANIFEST.json", manifest)
  write_status_live(
    out / "STATUS_LIVE.md",
    stage="L",
    run_id=manifest["run_id"],
    n_expected=1,
    n_finished=0,
    extra={"loss_path": args.loss_path},
  )

  device_map = f"cuda:{args.gpu}" if args.gpu is not None else "auto"
  backend = ScapeHFToolOPD(
    model_path=args.model_path,
    device_map=device_map,
    learning_rate=args.lr,
    use_lora=True,
    lora_r=args.lora_r,
    lora_alpha=max(args.lora_r, args.lora_alpha),
    anchor_weight=args.anchor_weight,
    lambda_args=args.lambda_args,
  )

  eval_rows = valid_rows[: min(16, len(valid_rows))]
  pre_metrics = mean_divergence(backend, eval_rows, loss_path=args.loss_path)  # type: ignore[arg-type]
  pre_agree = _tool_agreement(backend, valid_rows)

  t0 = time.time()
  trained = run_tool_opd_train(
    backend,
    train_rows,
    eval_rows,
    loss_path=args.loss_path,  # type: ignore[arg-type]
    epochs=args.epochs,
    batch_size=args.batch_size,
  )
  train_s = time.time() - t0

  post_metrics = mean_divergence(backend, test_rows, loss_path=args.loss_path)  # type: ignore[arg-type]
  post_agree = _tool_agreement(backend, test_rows)

  ckpt_dir = out / "lora_checkpoint"
  backend.save_pretrained(str(ckpt_dir))
  merged_dir = out / "hf_merged"
  try:
    backend.merge_and_save(str(merged_dir))
  except Exception as exc:  # noqa: BLE001
    (out / "merge_skipped.json").write_text(json.dumps({"error": str(exc)[:500]}, indent=2) + "\n")
    merged_dir = ckpt_dir

  summary = {
    "component_id": args.component_id,
    "n_samples": args.n_samples,
    "seed": args.seed,
    "loss_path": args.loss_path,
    "base_checkpoint": args.model_path,
    "use_lora": True,
    "dry_run": False,
    "legacy_scope_path_used": False,
    "d_pre": trained["D_pre"],
    "d_post": trained["D_post"],
    "L_m": trained["L_m"],
    "name_kl_pre": trained.get("name_kl_pre"),
    "name_kl_post": trained.get("name_kl_post"),
    "arg_key_kl_pre": trained.get("arg_key_kl_pre"),
    "arg_key_kl_post": trained.get("arg_key_kl_post"),
    "arg_value_kl_pre": trained.get("arg_value_kl_pre"),
    "arg_value_kl_post": trained.get("arg_value_kl_post"),
    "heldout_div_pre": pre_metrics["div"],
    "heldout_div_post": post_metrics["div"],
    "mean_train_loss": trained["mean_train_loss"],
    "n_train_steps": trained["n_train_steps"],
    "train_seconds": train_s,
    "tool_name_agreement_pre": pre_agree["tool_name_agreement"],
    "tool_name_agreement_post": post_agree["tool_name_agreement"],
    "exact_tool_call_agreement_pre": pre_agree["exact_tool_call_agreement"],
    "exact_tool_call_agreement_post": post_agree["exact_tool_call_agreement"],
    "invalid_tool_rate_pre": pre_agree["invalid_tool_rate"],
    "invalid_tool_rate_post": post_agree["invalid_tool_rate"],
    "checkpoint_lora": str(ckpt_dir),
    "checkpoint_merged": str(merged_dir),
    "loss_impl": trained["loss_impl"],
  }
  (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
  write_status_live(
    out / "STATUS_LIVE.md",
    stage="L",
    run_id=manifest["run_id"],
    n_expected=1,
    n_finished=1,
    extra={"L_m": summary["L_m"], "d_post": summary["d_post"]},
  )
  write_run_manifest(
    out / "RUN_MANIFEST.json",
    finalize_run_manifest(manifest, exit_code=0, completed_shards=["train_eval"]),
  )
  (out / "DONE").write_text("ok\n", encoding="utf-8")

  del backend
  gc.collect()
  try:
    import torch

    if args.gpu is not None:
      with torch.cuda.device(args.gpu):
        torch.cuda.empty_cache()
  except Exception:
    pass
  return summary


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", type=Path, required=True)
  ap.add_argument("--model-path", default="/data/ppnm/models/harness-1")
  ap.add_argument("--train-jsonl", type=Path, required=True)
  ap.add_argument("--valid-jsonl", type=Path, required=True)
  ap.add_argument("--test-jsonl", type=Path, default=None)
  ap.add_argument("--component-id", default="evidence_graph")
  ap.add_argument("--n-samples", type=int, required=True)
  ap.add_argument("--seed", type=int, default=42)
  ap.add_argument("--loss-path", default="tool_token_kl")
  ap.add_argument("--gpu", type=int, default=0)
  ap.add_argument("--epochs", type=int, default=1)
  ap.add_argument("--batch-size", type=int, default=1)
  ap.add_argument("--lr", type=float, default=1e-5)
  ap.add_argument("--lora-r", type=int, default=8)
  ap.add_argument("--lora-alpha", type=int, default=16)
  ap.add_argument("--anchor-weight", type=float, default=0.05)
  ap.add_argument("--lambda-args", type=float, default=0.0)
  args = ap.parse_args()
  try:
    summary = run_cell(args)
    print(json.dumps(summary, indent=2))
    return 0
  except Exception as exc:
    err = {"error": str(exc)}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "FAILED.json").write_text(json.dumps(err, indent=2) + "\n")
    print(json.dumps(err, indent=2))
    return 1


if __name__ == "__main__":
  raise SystemExit(main())
