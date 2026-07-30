#!/usr/bin/env python3
"""Adapter vs merged HF parity + HF vs runtime parity (Round 6 Phase B)."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from openai import OpenAI

from training.scope.decision_config import DEFAULT_DECISION_CONFIG
from training.scope.vllm_operation_scorer import VllmOperationScorer
from training.scope_round6.common import OUT, VALID522, write_json
from training.scope_round6.scorer_utils import (
    input_hashes,
    load_adapter_model,
    load_merged_model,
    parity_predictions,
    score_samples_hf,
    scorer_paths_for_tag,
)
from training.scope_round6.state_sources import load_state_source


def adapter_merged_parity(tag: str, gpu: str, samples: list[dict]) -> dict:
    paths = scorer_paths_for_tag(tag)
    adapter_model, adapter_tok, dev = load_adapter_model(paths["adapter"], gpu)
    merged_model, merged_tok, _ = load_merged_model(paths["merged"], gpu)
    adapter_scored = score_samples_hf(adapter_model, adapter_tok, samples, dev)
    merged_scored = score_samples_hf(merged_model, merged_tok, samples, dev)
    pred_a = [r.prediction for r in adapter_scored]
    pred_m = [r.prediction for r in merged_scored]
    margin_a = [r.margin for r in adapter_scored]
    margin_m = [r.margin for r in merged_scored]
    margin_match = sum(1 for a, b in zip(margin_a, margin_m) if abs(a - b) < 1e-4)
    return {
        "prediction_parity": parity_predictions(pred_a, pred_m),
        "margin_exact_match_rate": margin_match / max(len(samples), 1),
        "n": len(samples),
    }


def hf_runtime_parity(
    tag: str,
    gpu: str,
    samples: list[dict],
    port: int,
) -> dict:
    paths = scorer_paths_for_tag(tag)
    model, tokenizer, dev = load_merged_model(paths["merged"], gpu)
    hf_scored = score_samples_hf(model, tokenizer, samples, dev)
    client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="EMPTY")
    vllm = VllmOperationScorer(client, "hmin-v2-rollout")
    runtime_preds = []
    hash_mismatches = 0
    for sample in samples:
        state_text = str(
            sample.get("student_state_text")
            or (sample.get("decision_state") or {}).get("rendered_context")
            or ""
        )
        hf_hashes = input_hashes(sample, tokenizer)
        r = vllm.score(state_text)
        pred = DEFAULT_DECISION_CONFIG.predict_from_scores(
            r.scores["KEEP_EVIDENCE"], r.scores["SKIP_DUPLICATE"]
        ).value
        runtime_preds.append(pred)
    hf_preds = [r.prediction for r in hf_scored]
    return {
        "prediction_parity": parity_predictions(hf_preds, runtime_preds),
        "hash_mismatch_count": hash_mismatches,
        "n": len(samples),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["adapter_merged", "hf_runtime"], required=True)
    p.add_argument("--scorer", default="o7_42")
    p.add_argument("--gpu", default="cuda:0")
    p.add_argument("--port", type=int, default=9600)
    p.add_argument("--n-states", type=int, default=512)
    p.add_argument("--state-source", default="valid522")
    p.add_argument("--output-dir", type=Path, default=OUT / "phase_b/parity")
    args = p.parse_args()

    samples = load_state_source(args.state_source)
    if len(samples) > args.n_states:
        rng = random.Random(42)
        samples = rng.sample(samples, args.n_states)

    if args.mode == "adapter_merged":
        report = adapter_merged_parity(args.scorer, args.gpu, samples)
    else:
        report = hf_runtime_parity(args.scorer, args.gpu, samples, args.port)

    out = args.output_dir / f"{args.mode}_{args.scorer}"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "parity.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
