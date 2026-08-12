#!/usr/bin/env python3
"""Barrier 3: Binary calibration baseline on frozen O7 models.

Supports parallel per-seed scoring (--score-seed) and a final --aggregate pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.scope.operation_objectives import ScoreNorm
from training.scope.rollback_operation_objectives import score_rollback_prompt
from training.scope_round10.common import (
    BASE_MODEL,
    DATA,
    OUT,
    SEEDS,
    binary_operation,
    load_jsonl,
    write_json,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

R8_OUT = _REPO / "outputs/scope_round8"
CALIB_DIR = OUT / "calibration"
PER_SEED_DIR = CALIB_DIR / "per_seed"
SPLIT_DIR = DATA / "live_split"


def score_split(model, tokenizer, rows, device) -> list[dict]:
    out = []
    for row in rows:
        text = row.get("effective_input_text", "")
        s_cont, _, s_roll = score_rollback_prompt(
            model, tokenizer, text, device=device, norm=ScoreNorm.MEAN
        )
        margin = float(s_roll.item()) - float(s_cont.item())
        gold = binary_operation(row) or "CONTINUE"
        out.append({"margin": margin, "gold": gold})
    return out


def metrics_at_tau(scored: list[dict], tau: float) -> dict:
    tp_c = tp_r = fn_c = fn_r = 0
    for s in scored:
        pred = "ROLLBACK_TO" if s["margin"] >= tau else "CONTINUE"
        gold = s["gold"]
        if gold == "CONTINUE":
            if pred == "CONTINUE":
                tp_c += 1
            else:
                fn_c += 1
        else:
            if pred == "ROLLBACK_TO":
                tp_r += 1
            else:
                fn_r += 1
    cr = tp_c / max(tp_c + fn_c, 1)
    rr = tp_r / max(tp_r + fn_r, 1)
    bal = (cr + rr) / 2
    preds = ["ROLLBACK_TO" if s["margin"] >= tau else "CONTINUE" for s in scored]
    prior = {op: preds.count(op) / max(len(preds), 1) for op in ("CONTINUE", "ROLLBACK_TO")}
    return {
        "tau": tau,
        "balanced_accuracy": bal,
        "ContinueRecall": cr,
        "RollbackRecall": rr,
        "prediction_prior": prior,
        "n": len(scored),
    }


def best_tau(scored: list[dict], taus: list[float]) -> dict:
    best = None
    for tau in taus:
        m = metrics_at_tau(scored, tau)
        obj = min(m["ContinueRecall"], m["RollbackRecall"])
        if best is None or obj > best["objective"]:
            best = {**m, "objective": obj}
    return best or metrics_at_tau(scored, 0.0)


def model_path_for_seed(seed: int) -> Path:
    p = R8_OUT / f"merged/rollback_o7_seed{seed}"
    return p if (p / "config.json").exists() else Path(BASE_MODEL)


def score_one_seed(seed: int, gpu: str) -> None:
    device = torch.device(gpu if torch.cuda.is_available() else "cpu")
    calib_rows = load_jsonl(SPLIT_DIR / "live_calibration.jsonl")
    valid_rows = load_jsonl(SPLIT_DIR / "live_valid.jsonl")
    test_rows = load_jsonl(SPLIT_DIR / "live_test.jsonl")
    path = model_path_for_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    model.eval()
    result = {
        "seed": seed,
        "calibration": score_split(model, tokenizer, calib_rows, device),
        "live_valid_raw": score_split(model, tokenizer, valid_rows, device),
        "live_test_raw": score_split(model, tokenizer, test_rows, device),
    }
    del model
    torch.cuda.empty_cache()
    PER_SEED_DIR.mkdir(parents=True, exist_ok=True)
    write_json(PER_SEED_DIR / f"seed{seed}.json", result)
    print(f"scored seed {seed} -> {PER_SEED_DIR / f'seed{seed}.json'}")


def aggregate() -> dict:
    taus = [round(x * 0.05, 2) for x in range(-40, 41)]
    per_seed_calib: dict[int, list] = {}
    per_seed_valid: dict[int, list] = {}
    per_seed_test: dict[int, list] = {}
    for seed in SEEDS:
        path = PER_SEED_DIR / f"seed{seed}.json"
        if not path.exists():
            raise FileNotFoundError(f"missing {path}; run --score-seed first")
        data = json.loads(path.read_text(encoding="utf-8"))
        per_seed_calib[seed] = data["calibration"]
        per_seed_valid[seed] = data["live_valid_raw"]
        per_seed_test[seed] = data["live_test_raw"]

    pooled: list[dict] = []
    for seed in SEEDS:
        pooled.extend(per_seed_calib[seed])
    best = best_tau(pooled, taus)
    tau_shared = best["tau"]

    per_seed_eval = {}
    test_bals = []
    for seed in SEEDS:
        per_seed_eval[seed] = {
            "live_valid": metrics_at_tau(per_seed_valid[seed], tau_shared),
            "live_test": metrics_at_tau(per_seed_test[seed], tau_shared),
        }
        test_bals.append(per_seed_eval[seed]["live_test"]["balanced_accuracy"])

    seed_span = max(test_bals) - min(test_bals) if test_bals else 999
    all_pass = all(
        per_seed_eval[s]["live_test"]["balanced_accuracy"] >= 0.70
        and per_seed_eval[s]["live_test"]["ContinueRecall"] >= 0.70
        and per_seed_eval[s]["live_test"]["RollbackRecall"] >= 0.70
        for s in SEEDS
    ) and seed_span <= 0.05

    result = {
        "tau_shared": tau_shared,
        "calibration_selection": best,
        "per_seed_eval": per_seed_eval,
        "seed_span_balanced_accuracy": seed_span,
        "calibration_pass": all_pass,
        "ROUND10_PRIMARY_CAUSE": "class_prior_calibration" if all_pass else None,
    }
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    write_json(CALIB_DIR / "BINARY_CALIBRATION.json", result)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", default="cuda:0")
    p.add_argument("--score-seed", type=int, choices=SEEDS)
    p.add_argument("--aggregate", action="store_true")
    args = p.parse_args()

    if args.aggregate:
        print(json.dumps(aggregate(), indent=2))
        return
    if args.score_seed is not None:
        score_one_seed(args.score_seed, args.gpu)
        return

    # Legacy single-GPU path (all seeds sequential)
    for seed in SEEDS:
        score_one_seed(seed, args.gpu)
    print(json.dumps(aggregate(), indent=2))


if __name__ == "__main__":
    main()
