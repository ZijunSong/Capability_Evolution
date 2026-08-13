#!/usr/bin/env python3
"""Controlled 64-state overfit test with canonical metrics."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scape.collection.same_state import load_same_state_jsonl
from scape.training.canonical_metrics import NUMERIC_FLOOR
from scape.training.hf_tool_opd import ScapeHFToolOPD


def _canonical_row_metrics(
    teacher: ScapeHFToolOPD,
    student: ScapeHFToolOPD,
    row: dict[str, Any],
) -> dict[str, float]:
    resp_ids = student.encode(row["response_text"])
    if not resp_ids:
        return {k: 0.0 for k in ["forward_KL", "reverse_KL", "JS", "signed_gap"]}
    red_ids = student.encode(row["prompt_reduced"])
    full_ids = teacher.encode(row["prompt_full"])
    with torch.no_grad():
        s_logits = student._response_position_logits(red_ids, resp_ids, require_grad=False)
        t_logits = teacher._response_position_logits(full_ids, resp_ids, require_grad=False)
    from scape.training.canonical_metrics import (
        aggregate_token_metrics,
        js_from_logits,
        kl_from_logits,
        signed_logprob_gap,
    )
    fwd = kl_from_logits(t_logits, s_logits, forward=True)
    rev = kl_from_logits(t_logits, s_logits, forward=False)
    js = js_from_logits(t_logits, s_logits)
    gap = signed_logprob_gap(t_logits, s_logits, resp_ids)
    spans = student.span_token_masks(row["response_text"], len(resp_ids))
    token_mask = student.response_token_mask(row["response_text"], loss_path="tool_token_kl")
    if len(token_mask) != len(resp_ids):
        token_mask = spans["tool"]
    m_tool = torch.tensor(token_mask, device=fwd.device, dtype=fwd.dtype)
    return aggregate_token_metrics(fwd, rev, js, gap, m_tool)


def mean_metrics(
    teacher: ScapeHFToolOPD,
    student: ScapeHFToolOPD,
    rows: list[dict[str, Any]],
) -> dict[str, float]:
    keys = ["forward_KL", "reverse_KL", "JS", "signed_gap"]
    acc = {k: 0.0 for k in keys}
    for row in rows:
        m = _canonical_row_metrics(teacher, student, row)
        for k in keys:
            acc[k] += m[k]
    n = max(1, len(rows))
    return {k: v / n for k, v in acc.items()}


def train_steps(
    backend: ScapeHFToolOPD,
    rows: list[dict[str, Any]],
    steps: int,
) -> None:
    for step in range(steps):
        for row in rows:
            backend.train_step([row], loss_path="tool_token_kl")


def shuffled_teacher_rows(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    prompts = [r["prompt_full"] for r in rows]
    rng.shuffle(prompts)
    out = []
    for i, row in enumerate(rows):
        new_row = dict(row)
        new_row["prompt_full"] = prompts[i]
        out.append(new_row)
    return out


def run_overfit_job(args: argparse.Namespace) -> list[dict[str, Any]]:
    train_rows = load_same_state_jsonl(Path(args.train_jsonl))[:64]
    eval_rows = train_rows  # frozen 64-state eval

    device_map = f"cuda:{args.gpu}"
    teacher = ScapeHFToolOPD(
        model_path=args.teacher_path,
        device_map=device_map,
        use_lora=False,
    )
    student = ScapeHFToolOPD(
        model_path=args.base_path,
        device_map=device_map,
        use_lora=True,
        learning_rate=args.lr,
    )

    if args.shuffled_teacher:
        eval_rows_train = shuffled_teacher_rows(train_rows, args.seed)
    else:
        eval_rows_train = train_rows

    results = []
    pre = mean_metrics(teacher, student, eval_rows)
    results.append({
        "job": args.job_name,
        "component": args.component,
        "shuffled_teacher": args.shuffled_teacher,
        "lr": args.lr,
        "step": 0,
        **pre,
    })

    prev_step = 0
    for step in args.step_schedule:
        delta = step - prev_step
        if delta > 0:
            train_steps(student, eval_rows_train, delta)
        post = mean_metrics(teacher, student, eval_rows)
        results.append({
            "job": args.job_name,
            "component": args.component,
            "shuffled_teacher": args.shuffled_teacher,
            "lr": args.lr,
            "step": step,
            **post,
        })
        prev_step = step

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    touch_done = out_dir / "DONE"
    touch_done.write_text(f"ok {time.time()}\n", encoding="utf-8")
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-name", required=True)
    ap.add_argument("--component", default="subtractive_curation")
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--teacher-path", default="/data/ppnm/models/harness-1")
    ap.add_argument("--base-path", default="/data/ppnm/models/harness-1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--shuffled-teacher", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--step-schedule", nargs="+", type=int, default=[1, 5, 20, 100])
    args = ap.parse_args()
    rows = run_overfit_job(args)
    for r in rows:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
