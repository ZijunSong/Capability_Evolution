#!/usr/bin/env python3
"""Plot RL training curves from per-step metrics.jsonl (TensorBoard-free fallback).

When the CloudML TensorBoard UI is unavailable (missing `tensorboard` package,
platform outage, or you just want offline plots), this script reads the
per-step JSONL that trainer_adapter.py writes alongside the TB events and
rebuilds the same 5 metric groups as static PNG panels.

Usage:
    python plot_metrics_jsonl.py <metrics.jsonl> [--out-dir <dir>] [--groups a,b,...]
    python plot_metrics_jsonl.py /mnt/songzijun/outputs/qwen3-4B-rl-recall-turn-punish-32gpu_20260928/rl/metrics.jsonl
    # -> writes <metrics.jsonl>.plots/*.png + dashboard.png (6-headline overview)

Each line in metrics.jsonl is one optimizer step (coordinator-only write).
Keys prefixed `tb_<group>_<key>` mirror the TensorBoard scalar groups exactly
(performance/ agent/ optimization/ learning_signal/ trim/ + legacy reward/
cispo/ tokens/ timing_s/ mem). This script groups them by prefix and plots
one PNG per group, x-axis = step.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path


# Metric groups -> (png filename, title, [keys]). Keys are the full jsonl keys.
# These mirror the trainer_adapter.py tb_<group>_<key> mirror block + a few
# flat legacy keys so a single run produces a complete dashboard.
GROUPS = [
    ("performance", "Performance (reward = recall - gated turn penalty)",
     ["tb_performance_reward_mean", "tb_performance_success_rate",
      "tb_performance_task_var", "tb_performance_turn_penalty_mean",
      "tb_performance_n_turns_mean"]),
    ("agent", "Agent behavior (rollout / trajectory)",
     ["tb_agent_turn_mean", "tb_agent_generated_tokens", "tb_agent_tool_calls",
      "tb_agent_invalid_tool_rate", "tb_agent_parse_valid_rate",
      "tb_agent_exec_ok_rate", "tb_agent_end_search_rate",
      "tb_agent_timeout_rate"]),
    ("optimization", "Optimization (CISPO policy gradient)",
     ["tb_optimization_policy_loss", "tb_optimization_entropy",
      "tb_optimization_approx_kl", "tb_optimization_grad_norm"]),
    ("learning_signal", "Learning signal (GRPO advantage spread)",
     ["tb_learning_signal_advantage_std",
      "tb_learning_signal_zero_variance_group_ratio",
      "tb_learning_signal_n_const_reward_groups"]),
    ("trim", "TRIM retrieval (selected vs gold)",
     ["tb_trim_selected_ratio", "tb_trim_n_curated_eps", "tb_trim_n_gold_eps"]),
    ("cispo", "CISPO clip stats",
     ["tb_cispo_ratio_clip_fraction", "tb_cispo_clip_low",
      "tb_cispo_clip_high", "tb_cispo_gate_mean"]),
    ("tokens", "Token throughput",
     ["tb_tokens_n_rl_tokens", "tb_tokens_n_microbatches",
      "tb_tokens_rl_padding_fraction", "tb_tokens_n_global_rl_microbatches"]),
    ("timing_s", "Timing (seconds per phase)",
     ["tb_timing_s_student_forward", "tb_timing_s_loss", "tb_timing_s_backward",
      "tb_timing_s_grad_sync", "tb_timing_s_optimizer", "tb_timing_s_update_wall"]),
    ("mem", "Memory",
     ["tb_mem_peak_mem_mb"]),
    ("counts", "Step counts / coverage",
     ["n_optimizer_steps", "n_rl_datums", "n_opd_datums",
      "tb_performance_n_groups", "tb_performance_n_episodes"]),
]

# 6-headline dashboard overview (one panel, 6 subplots) for a quick glance.
DASHBOARD_HEADLINE = [
    ("reward", "tb_performance_reward_mean"),
    ("success_rate", "tb_performance_success_rate"),
    ("turn_mean", "tb_agent_turn_mean"),
    ("entropy", "tb_optimization_entropy"),
    ("approx_kl", "tb_optimization_approx_kl"),
    ("grad_norm", "tb_optimization_grad_norm"),
]


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def to_series(records: list[dict], key: str) -> tuple[list, list]:
    xs, ys = [], []
    for rec in records:
        step = rec.get("step")
        if step is None:
            continue
        val = rec.get(key)
        if val is None:
            continue
        try:
            fval = float(val)
        except (TypeError, ValueError):
            continue
        if fval != fval:  # NaN
            continue
        xs.append(int(step))
        ys.append(fval)
    return xs, ys


def plot_group(records, fname, title, keys, out_dir):
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt

    n = len(keys)
    if n == 0:
        return
    cols = min(2, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 2.6 * rows), squeeze=False)
    for idx, key in enumerate(keys):
        ax = axes[idx // cols][idx % cols]
        xs, ys = to_series(records, key)
        if xs:
            ax.plot(xs, ys, marker=".", markersize=2, linewidth=1)
            ax.set_xlabel("step")
            # legend label = key without tb_ prefix for readability
            label = key[3:] if key.startswith("tb_") else key
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f"no data: {key}", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
    # hide unused axes
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = out_dir / fname
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"wrote {out} ({sum(1 for k in keys if to_series(records, k)[0])}/{len(keys)} series with data)")


def plot_dashboard(records, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(DASHBOARD_HEADLINE)
    cols = 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 2.4 * rows), squeeze=False)
    for idx, (label, key) in enumerate(DASHBOARD_HEADLINE):
        ax = axes[idx // cols][idx % cols]
        xs, ys = to_series(records, key)
        if xs:
            ax.plot(xs, ys, marker=".", markersize=2, linewidth=1, color="#1f77b4")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f"no data: {key}", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("step", fontsize=8)
        ax.tick_params(labelsize=7)
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].axis("off")
    fig.suptitle("RL training dashboard (headline metrics)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = out_dir / "dashboard.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"wrote {out} (6-headline overview)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("metrics_jsonl", help="path to <out>/<cell>/metrics.jsonl")
    ap.add_argument("--out-dir", default=None,
                    help="output dir (default: <metrics_jsonl>.plots)")
    ap.add_argument("--groups", default=None,
                    help="comma-separated subset of groups to plot (default: all). "
                         "e.g. performance,optimization")
    args = ap.parse_args()

    path = Path(args.metrics_jsonl)
    if not path.is_file():
        sys.exit(f"not a file: {path}")
    out_dir = Path(args.out_dir) if args.out_dir else path.parent / f"{path.name}.plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(path)
    if not records:
        sys.exit(f"no records parsed from {path}")
    print(f"loaded {len(records)} step records from {path}")

    # report reward_mode (should be recall_turn for this run)
    modes = defaultdict(int)
    for rec in records:
        modes[rec.get("tb_performance_reward_mode") or rec.get("reward_mode")] += 1
    if modes:
        print(f"reward_mode distribution: {dict(modes)}")

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        sys.exit("matplotlib not installed. pip install matplotlib then rerun.")

    if args.groups:
        wanted = {g.strip() for g in args.groups.split(",") if g.strip()}
        groups = [g for g in GROUPS if g[0] in wanted]
        if not groups:
            sys.exit(f"no matching groups for --groups {args.groups}. "
                     f"available: {[g[0] for g in GROUPS]}")
    else:
        groups = GROUPS

    for fname, title, keys in groups:
        plot_group(records, f"{fname}.png", title, keys, out_dir)
    plot_dashboard(records, out_dir)

    print(f"\nDone. {len(groups)} group panels + dashboard in {out_dir}/")


if __name__ == "__main__":
    main()
