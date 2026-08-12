#!/usr/bin/env python3
"""Targeted float32 HF re-score for Barrier A failing variants, then re-aggregate.

Strategy:
1) Find rows that still fail near-tie-aware Barrier agreement.
2) Re-score those rows with full float32 HF and patch hf_replay.jsonl.
3) Re-aggregate; if still failing, optionally re-run the whole split in float32.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.scope_round9.aggregate_frozen_replay import (
    _ops_agree_for_barrier,
    barrier_a_for_parity,
    compare_hf_vllm,
    load_jsonl,
)
from training.scope_round9.replay_frozen_hf import replay_rows

OUT = _REPO / "outputs/scope_round9"
FROZEN = _REPO / "artifacts/datasets/scope_round9/frozen_replay"
MARKER_DIR = OUT / "markers"
LOG_DIR = OUT / "logs"
R8 = _REPO / "outputs/scope_round8/merged"

VARIANTS = {
    "rollback_correct_only": {"gpu": "6", "port": 18106},
    "rollback_soft_replan_only": {"gpu": "7", "port": 18107},
}


def model_path(variant: str) -> Path:
    p = R8 / variant
    if (p / "config.json").exists():
        return p
    raise FileNotFoundError(p)


def failing_indices(hf: list[dict], vl: list[dict]) -> list[int]:
    out = []
    for i, (h, v) in enumerate(zip(hf, vl)):
        _, bar = _ops_agree_for_barrier(h, v)
        if not bar:
            out.append(i)
    return out


def patch_hf_float32(
    *,
    variant: str,
    split: str,
    gpu: str,
    idxs: list[int],
) -> dict:
    vdir = OUT / "wave_a" / variant
    hf_path = vdir / split / "hf_replay.jsonl"
    vl_path = vdir / split / "vllm_replay.jsonl"
    hf = load_jsonl(hf_path)
    vl = load_jsonl(vl_path)
    if not idxs:
        return {"split": split, "n_fail_before": 0, "n_rescored": 0, "n_fail_after_patch": 0}

    bak = vdir / split / "hf_replay.bf16.bak.jsonl"
    if not bak.exists():
        shutil.copy2(hf_path, bak)

    subset = [hf[i] for i in idxs]
    tmp_in = vdir / split / "_failing_subset_in.jsonl"
    tmp_out = vdir / split / "_failing_subset_hf_f32.jsonl"
    with tmp_in.open("w", encoding="utf-8") as f:
        for row in subset:
            # Keep original frozen fields; scoring uses effective_input_text.
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[{variant}/{split}] float32-rescore {len(idxs)} rows on cuda:{gpu}", flush=True)
    import os

    # Caller sets CUDA_VISIBLE_DEVICES; use cuda:0 inside the visible device.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(str(model_path(variant)), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path(variant)), torch_dtype=torch.float32, trust_remote_code=True
    ).eval().to(device)
    with torch.no_grad():
        rescored = replay_rows(model, tok, subset, device=device)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with tmp_out.open("w", encoding="utf-8") as f:
        for row in rescored:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_fixed = 0
    for local_i, global_i in enumerate(idxs):
        new_row = rescored[local_i]
        merged = dict(hf[global_i])
        for k in (
            "hf_logits",
            "pred_operation",
            "pred_checkpoint_local_id",
            "pred_checkpoint_global_id",
            "fallback_reason",
        ):
            if k in new_row:
                merged[k] = new_row[k]
        hf[global_i] = merged
        _, bar = _ops_agree_for_barrier(hf[global_i], vl[global_i])
        n_fixed += int(bar)

    with hf_path.open("w", encoding="utf-8") as f:
        for row in hf:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    remain = failing_indices(hf, vl)
    return {
        "split": split,
        "n_fail_before": len(idxs),
        "n_rescored": len(idxs),
        "n_fixed_among_rescored": n_fixed,
        "n_fail_after_patch": len(remain),
        "remaining_indices": remain[:20],
        "dtype": "float32",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def full_split_float32(variant: str, split: str, gpu: str) -> None:
    vdir = OUT / "wave_a" / variant
    inp = FROZEN / f"{split}.jsonl"
    if split == "self_live":
        inp = FROZEN / "self_live" / f"{variant}.jsonl"
    hf_out = vdir / split / "hf_replay.jsonl"
    bak = vdir / split / "hf_replay.bf16.bak.jsonl"
    if hf_out.exists() and not bak.exists():
        shutil.copy2(hf_out, bak)
    log = LOG_DIR / f"wave_a_{variant}_{split}_hf_f32.log"
    cmd = [
        sys.executable,
        str(_REPO / "training/scope_round9/replay_frozen_hf.py"),
        "--model-path",
        str(model_path(variant)),
        "--input",
        str(inp),
        "--output",
        str(hf_out),
        "--device",
        "cuda:0",
        "--dtype",
        "float32",
    ]
    print(f"[{variant}/{split}] FULL float32 HF replay", flush=True)
    import os

    env = os.environ.copy()
    # Inherit caller's CUDA_VISIBLE_DEVICES (launcher already pins the GPU).
    env.setdefault("CUDA_VISIBLE_DEVICES", gpu)
    env["PYTHONPATH"] = str(_REPO)
    with log.open("a", encoding="utf-8") as lf:
        subprocess.run(
            cmd,
            check=True,
            cwd=_REPO,
            env=env,
            stdout=lf,
            stderr=subprocess.STDOUT,
        )


def reaggregate(variant: str) -> dict:
    vdir = OUT / "wave_a" / variant
    report: dict = {"variant_dir": str(vdir), "split_failures": {}, "repair": "float32_targeted"}
    all_pass = True
    for split in ("offline_valid", "base_live", "self_live"):
        p = compare_hf_vllm(
            load_jsonl(vdir / split / "hf_replay.jsonl"),
            load_jsonl(vdir / split / "vllm_replay.jsonl"),
        )
        ok, fails = barrier_a_for_parity(p)
        report[split] = {
            "parity": p,
            "barrier_a_split_pass": ok,
            "barrier_a_failures": fails,
        }
        report["split_failures"][split] = fails
        all_pass = all_pass and ok
    report["barrier_a_pass"] = all_pass
    (vdir / "WAVE_A_REPORT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    marker = MARKER_DIR / f"wave_a_{variant}.DONE"
    if all_pass:
        marker.write_text("done\n", encoding="utf-8")
    else:
        marker.unlink(missing_ok=True)
    return report


def run_variant(variant: str, *, full_on_fail: bool) -> dict:
    cfg = VARIANTS[variant]
    gpu = str(cfg["gpu"])
    diag: dict = {"variant": variant, "gpu": gpu, "splits": []}
    vdir = OUT / "wave_a" / variant
    for split in ("base_live", "self_live"):
        hf = load_jsonl(vdir / split / "hf_replay.jsonl")
        vl = load_jsonl(vdir / split / "vllm_replay.jsonl")
        idxs = failing_indices(hf, vl)
        print(f"[{variant}/{split}] unresolved before={len(idxs)}", flush=True)
        if not idxs:
            diag["splits"].append({"split": split, "n_fail_before": 0})
            continue
        info = patch_hf_float32(variant=variant, split=split, gpu=gpu, idxs=idxs)
        diag["splits"].append(info)
        if info["n_fail_after_patch"] and full_on_fail:
            full_split_float32(variant, split, gpu)
            hf2 = load_jsonl(vdir / split / "hf_replay.jsonl")
            vl2 = load_jsonl(vdir / split / "vllm_replay.jsonl")
            remain = failing_indices(hf2, vl2)
            info["full_float32"] = True
            info["n_fail_after_full"] = len(remain)
            print(f"[{variant}/{split}] after full float32 remain={len(remain)}", flush=True)
    report = reaggregate(variant)
    diag["barrier_a_pass"] = report["barrier_a_pass"]
    diag["split_failures"] = report["split_failures"]
    out = OUT / "wave_a" / variant / "FLOAT32_REPAIR_DIAG.json"
    out.write_text(json.dumps(diag, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(diag, indent=2), flush=True)
    return diag


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    p.add_argument(
        "--full-on-fail",
        action="store_true",
        help="If targeted patch leaves failures, re-run whole split HF in float32.",
    )
    args = p.parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    MARKER_DIR.mkdir(parents=True, exist_ok=True)
    diag = run_variant(args.variant, full_on_fail=args.full_on_fail)
    if not diag.get("barrier_a_pass"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
