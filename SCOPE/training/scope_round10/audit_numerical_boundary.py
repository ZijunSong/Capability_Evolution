#!/usr/bin/env python3
"""Summarize near-boundary residual flips after disable_replan (R10-P8)."""

from __future__ import annotations

import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
PHASE_A = _REPO / "outputs/scope_round10/phase_a"
OUT = PHASE_A / "audits"


def main() -> None:
    rows = []
    for seed in (42, 43, 44):
        for split in ("offline_valid", "base_live"):
            path = PHASE_A / f"seed{seed}" / split / "residual_mismatch.jsonl"
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    hs = float(r["hf_score_continue"]) - float(r["hf_score_rollback"])
                    vs = float(r["vllm_score_continue"]) - float(r["vllm_score_rollback"])
                    rows.append(
                        {
                            "seed": seed,
                            "split": split,
                            "event_id": r.get("event_id"),
                            "hf_signed_margin": hs,
                            "vllm_signed_margin": vs,
                            "hf_op": r.get("hf_operation"),
                            "vllm_op": r.get("vllm_operation"),
                            "near_0_25": abs(hs) < 0.25 or abs(vs) < 0.25,
                        }
                    )
    float32_rescued = 0
    float32_total = 0
    for seed in (42, 43, 44):
        for split in ("offline_valid", "base_live"):
            fp = PHASE_A / f"seed{seed}" / split / "float32_rescore.jsonl"
            if not fp.exists():
                continue
            with fp.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    float32_total += 1
                    if r.get("agree_vllm_after_float32"):
                        float32_rescued += 1
    summary = {
        "residual_n": len(rows),
        "near_0_25_n": sum(1 for r in rows if r["near_0_25"]),
        "sign_flip_n": sum(
            1 for r in rows if (r["hf_signed_margin"] > 0) != (r["vllm_signed_margin"] > 0)
        ),
        "float32_rescore_n": float32_total,
        "float32_rescued": float32_rescued,
        "float32_rescue_rate": float32_rescued / max(float32_total, 1),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "numerical_boundary_audit.json").write_text(
        json.dumps({"summary": summary, "residuals": rows}, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
