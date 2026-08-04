#!/usr/bin/env python3
"""Exact vLLM scorer replay of live decision traces (Round 7)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from openai import OpenAI

from harness.capability.dup_operation import DupOperation
from training.scope.decision_config import DupDecisionConfig
from training.scope.decide_dup_operation import decide_dup_operation
from training.scope.vllm_operation_scorer import VllmOperationScorer
from training.scope_round7.common import OUT, load_jsonl, write_json


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace-dir", type=Path, required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--vllm-port", type=int, default=9204)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--manage-vllm", action="store_true", default=True)
    p.add_argument("--no-manage-vllm", action="store_false", dest="manage_vllm")
    args = p.parse_args()

    trace_path = args.trace_dir / "live_dup_decision_trace.jsonl"
    traces = load_jsonl(trace_path)
    out = args.output_dir or (OUT / "contract_trace/replay_vllm" / args.trace_dir.name)
    out.mkdir(parents=True, exist_ok=True)

    vllm_proc = None
    base_url = f"http://127.0.0.1:{args.vllm_port}/v1"
    if args.manage_vllm:
        log_path = out / "vllm_server.log"
        cmd = [
            "vllm", "serve", str(args.model_path),
            "--served-model-name", "hmin-v2-rollout",
            "--host", "127.0.0.1",
            "--port", str(args.vllm_port),
            "--tensor-parallel-size", "1",
            "--max-model-len", "32768",
            "--dtype", "bfloat16",
            "--disable-custom-all-reduce",
            "--enforce-eager",
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes",
        ]
        vllm_proc = subprocess.Popen(
            cmd, stdout=log_path.open("w"), stderr=subprocess.STDOUT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0")},
        )
        for _ in range(120):
            try:
                OpenAI(base_url=base_url, api_key="EMPTY").models.list()
                break
            except Exception:
                time.sleep(5)
        else:
            raise RuntimeError("vLLM server failed to start")

    client = OpenAI(base_url=base_url, api_key="EMPTY")
    scorer = VllmOperationScorer(
        client,
        "hmin-v2-rollout",
        decision_config=DupDecisionConfig(),
        model_path=str(args.model_path),
    )

    rows: list[dict] = []
    for tr in traces:
        prompt = tr.get("rendered_prompt") or ""
        if not prompt:
            sidecar = args.trace_dir / "prompt_sidecar" / f"{tr['rendered_prompt_sha256']}.txt"
            if sidecar.exists():
                prompt = sidecar.read_text(encoding="utf-8")
        result = scorer.score_prompt(prompt)
        sk = result.scores[DupOperation.KEEP_EVIDENCE.value]
        ss = result.scores[DupOperation.SKIP_DUPLICATE.value]
        decision = decide_dup_operation(
            score_keep=sk, score_skip=ss, threshold=float(tr.get("threshold", 0.0))
        )
        rows.append({
            "event_id": tr["event_id"],
            "score_keep_vllm": sk,
            "score_skip_vllm": ss,
            "margin_vllm": decision.margin,
            "operation_vllm": decision.predicted_operation.value,
            "score_keep_live": tr.get("score_keep"),
            "score_skip_live": tr.get("score_skip"),
            "margin_live": tr.get("margin"),
            "operation_live": tr.get("predicted_operation_pre_realizer"),
        })

    write_json(out / "vllm_replay.json", {"n": len(rows), "rows": rows})
    if vllm_proc is not None:
        vllm_proc.terminate()
        vllm_proc.wait(timeout=30)
    print(f"vLLM replay: {len(rows)} events -> {out}")


if __name__ == "__main__":
    main()
