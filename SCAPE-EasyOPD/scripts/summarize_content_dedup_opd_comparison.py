#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any


def bootstrap(values: list[float], seed: int, n_boot: int) -> dict[str, float]:
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    return {
        "mean": sum(values) / len(values),
        "ci95_low": means[int(0.025 * (n_boot - 1))],
        "ci95_high": means[int(0.975 * (n_boot - 1))],
        "n_boot": n_boot,
    }


def paired(before: dict[str, Any], after: dict[str, Any], metric: str, seed: int, n_boot: int) -> dict[str, Any]:
    before_by_uid = {row["state_uid"]: row for row in before["records"]}
    after_by_uid = {row["state_uid"]: row for row in after["records"]}
    state_uids = sorted(set(before_by_uid) & set(after_by_uid))
    deltas = [float(after_by_uid[uid][metric]) - float(before_by_uid[uid][metric]) for uid in state_uids]
    return {
        "n_pairs": len(deltas),
        "delta": bootstrap(deltas, seed, n_boot),
        "positive": sum(value > 0 for value in deltas),
        "negative": sum(value < 0 for value in deltas),
        "zero": sum(value == 0 for value in deltas),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=10000)
    args = parser.parse_args()
    names = ["TEACHER", "STUDENT_BEFORE", "PURE_OPD_seed42", "PURE_OPD_seed43", "RL_PLUS_OPD_seed42", "RL_PLUS_OPD_seed43"]
    cells = {name: json.loads((args.root / f"{name}.json").read_text(encoding="utf-8")) for name in names}
    expected_uids = {row["state_uid"] for row in cells["STUDENT_BEFORE"]["records"]}
    if len(expected_uids) != 500:
        raise RuntimeError(f"expected 500 unique paired rows, found {len(expected_uids)}")
    for name, payload in cells.items():
        summary = payload["summary"]
        uids = {row["state_uid"] for row in payload["records"]}
        if summary["n_rows"] != 500 or uids != expected_uids:
            raise RuntimeError(f"row pairing mismatch for {name}")
        if summary["student_inference_privilege"] is not False:
            raise RuntimeError(f"unexpected Student privilege for {name}")
        if name not in {"TEACHER", "STUDENT_BEFORE"} and summary["reload_path"] == "base_no_adapter":
            raise RuntimeError(f"adapter was not loaded for {name}")

    metrics = ["legal_action", "exact_projected_target"]
    result: dict[str, Any] = {
        "status": "CONTENT_DEDUP_ACTION_LEVEL_OPD_COMPARISON_COMPLETE",
        "component": "content_dedup",
        "n_frozen_valid_rows": 500,
        "metric_scope": "frozen_opd_valid_rows_action_level_internalization_diagnostic",
        "terminal_task_reward": None,
        "cells": {name: payload["summary"] for name, payload in cells.items()},
        "metrics": {},
    }
    before = cells["STUDENT_BEFORE"]
    for metric in metrics:
        metric_result: dict[str, Any] = {
            "teacher": sum(float(row[metric]) for row in cells["TEACHER"]["records"]) / 500,
            "before": sum(float(row[metric]) for row in before["records"]) / 500,
            "seed_cells": {},
            "methods": {},
        }
        for name in names[2:]:
            rate = sum(float(row[metric]) for row in cells[name]["records"]) / 500
            comparison = paired(before, cells[name], metric, 20260820 + len(metric) + len(name), args.n_boot)
            metric_result["seed_cells"][name] = {"rate": rate, "paired_vs_before": comparison}
        for method in ["PURE_OPD", "RL_PLUS_OPD"]:
            method_names = [f"{method}_seed42", f"{method}_seed43"]
            rates = [metric_result["seed_cells"][name]["rate"] for name in method_names]
            delta_means = [metric_result["seed_cells"][name]["paired_vs_before"]["delta"]["mean"] for name in method_names]
            merged_deltas = []
            for name in method_names:
                before_by_uid = {row["state_uid"]: row for row in before["records"]}
                after_by_uid = {row["state_uid"]: row for row in cells[name]["records"]}
                merged_deltas.extend(float(after_by_uid[uid][metric]) - float(before_by_uid[uid][metric]) for uid in sorted(expected_uids))
            metric_result["methods"][method] = {
                "rate_mean": statistics.mean(rates),
                "rate_sample_std": statistics.stdev(rates),
                "delta_mean": statistics.mean(delta_means),
                "delta_sample_std": statistics.stdev(delta_means),
                "paired_seed_row_bootstrap": bootstrap(merged_deltas, 20260820 + len(metric) + len(method), args.n_boot),
                "positive": sum(value > 0 for value in merged_deltas),
                "negative": sum(value < 0 for value in merged_deltas),
                "zero": sum(value == 0 for value in merged_deltas),
                "n_seed_rows": len(merged_deltas),
            }
        result["metrics"][metric] = metric_result

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
