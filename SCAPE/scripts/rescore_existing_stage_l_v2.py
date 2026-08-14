#!/usr/bin/env python3
"""Rescore existing Stage-L checkpoints with Learnability V2 metrics."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from scape.collection.same_state import load_same_state_jsonl
from scape.eval.learnability_metrics_v2 import (
  LearnabilityMetricsV2,
  aggregate_rows_v2,
  learnability_improved,
  v2_gate_pass,
)
from scape.training.hf_tool_opd import ScapeHFToolOPD, LossPath

from run_learnability_historical_reeval import (
  build_checkpoint_inventory,
  filter_inventory,
  load_done_keys,
)


CSV_FIELDS = [
  "family", "checkpoint_id", "checkpoint_path", "valid_jsonl", "loss_path",
  "JS_name_pre", "JS_name_post", "CE_T_on_S_pre", "CE_T_on_S_post",
  "KL_name_pre", "KL_name_post", "KL_arg_key_pre", "KL_arg_key_post",
  "forward_KL_pre", "forward_KL_post", "signed_gap_pre", "signed_gap_post",
  "legacy_d_pre", "legacy_d_post", "legacy_L_m",
  "invalid_tool_rate_pre", "invalid_tool_rate_post",
  "v2_gate_pass", "v2_gate_reason",
  "n_valid",
]


def _load_legacy_summary(ck_path: Path, repo: Path) -> dict[str, Any]:
  """Find summary.json for a merged checkpoint."""
  ck = ck_path.resolve()
  base = Path("/data/ppnm/models/harness-1").resolve()
  if ck == base:
    return {}
  for summary in repo.rglob("summary.json"):
    try:
      data = json.loads(summary.read_text())
      merged = Path(data.get("checkpoint_merged", "")).resolve()
      if merged == ck:
        return data
    except (json.JSONDecodeError, OSError):
      continue
  return {}


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--repo", type=Path, default=REPO)
  ap.add_argument("--out-csv", type=Path, required=True)
  ap.add_argument("--teacher-path", default="/data/ppnm/models/harness-1")
  ap.add_argument("--families", nargs="*", default=None)
  ap.add_argument("--checkpoint-ids", nargs="*", default=None)
  ap.add_argument("--gpu", type=int, default=0)
  ap.add_argument("--max-valid", type=int, default=None)
  ap.add_argument("--skip-existing", action="store_true", default=True)
  ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
  args = ap.parse_args()

  out_csv = args.out_csv
  out_csv.parent.mkdir(parents=True, exist_ok=True)

  items = filter_inventory(build_checkpoint_inventory(args.repo), args.families)
  if args.checkpoint_ids:
    ids = set(args.checkpoint_ids)
    items = [x for x in items if x["checkpoint_id"] in ids]
  # Skip base-only rescoring entries (no post checkpoint)
  items = [x for x in items if x["checkpoint_id"] != "base"]

  if args.skip_existing:
    done = load_done_keys(out_csv)
    items = [x for x in items if (x["family"], x["checkpoint_id"]) not in done]

  if not items:
    print(json.dumps({"status": "skip", "reason": "nothing to rescore"}))
    return 0

  device_map = f"cuda:{args.gpu}"
  teacher = ScapeHFToolOPD(
    model_path=args.teacher_path,
    device_map=device_map,
    use_lora=False,
  )
  base_metrics_cache: dict[str, LearnabilityMetricsV2] = {}

  write_header = not out_csv.exists() or out_csv.stat().st_size == 0

  for item in items:
    ck_path = Path(item["checkpoint_path"])
    valid_rows = load_same_state_jsonl(Path(item["valid_jsonl"]))
    loss_path = item.get("loss_path", "tool_token_kl")

    valid_key = item["valid_jsonl"] + ":" + loss_path
    if valid_key not in base_metrics_cache:
      pre = aggregate_rows_v2(
        teacher, teacher, valid_rows,
        loss_path=loss_path,  # type: ignore[arg-type]
        max_rows=args.max_valid,
      )
      base_metrics_cache[valid_key] = pre
    else:
      pre = base_metrics_cache[valid_key]

    student = ScapeHFToolOPD(
      model_path=str(ck_path),
      device_map=device_map,
      use_lora=False,
    )
    post = aggregate_rows_v2(
      teacher, student, valid_rows,
      loss_path=loss_path,  # type: ignore[arg-type]
      max_rows=args.max_valid,
    )
    gate_ok, gate_reason = v2_gate_pass(pre, post)
    legacy = _load_legacy_summary(ck_path, args.repo)

    row = {
      "family": item["family"],
      "checkpoint_id": item["checkpoint_id"],
      "checkpoint_path": str(ck_path),
      "valid_jsonl": item["valid_jsonl"],
      "loss_path": loss_path,
      "JS_name_pre": pre.JS_name,
      "JS_name_post": post.JS_name,
      "CE_T_on_S_pre": pre.CE_T_on_S,
      "CE_T_on_S_post": post.CE_T_on_S,
      "KL_name_pre": pre.KL_name,
      "KL_name_post": post.KL_name,
      "KL_arg_key_pre": pre.KL_arg_key,
      "KL_arg_key_post": post.KL_arg_key,
      "forward_KL_pre": pre.forward_KL,
      "forward_KL_post": post.forward_KL,
      "signed_gap_pre": pre.signed_logprob_gap,
      "signed_gap_post": post.signed_logprob_gap,
      "legacy_d_pre": legacy.get("d_pre", ""),
      "legacy_d_post": legacy.get("d_post", ""),
      "legacy_L_m": legacy.get("L_m", ""),
      "invalid_tool_rate_pre": pre.invalid_tool_rate,
      "invalid_tool_rate_post": post.invalid_tool_rate,
      "v2_gate_pass": gate_ok,
      "v2_gate_reason": gate_reason,
      "n_valid": post.n_rows,
    }
    print(json.dumps(row))

    with out_csv.open("a", newline="", encoding="utf-8") as f:
      w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
      if write_header:
        w.writeheader()
        write_header = False
      w.writerow(row)

    del student
    gc.collect()
    import torch
    torch.cuda.empty_cache()

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
