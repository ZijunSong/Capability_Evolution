#!/usr/bin/env python3
"""vLLM deterministic replay on frozen effective inputs (no re-wrap)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from openai import OpenAI

from training.scope.canonical_rollback_scorer import CanonicalRollbackOperationScorer
from training.scope.vllm_rollback_scorer import VllmRollbackScorer


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def replay_rows(
    scorer: VllmRollbackScorer,
    rows: list[dict],
) -> list[dict]:
    # Canonical single-backend contract (Round10 followup A2).
    canonical = CanonicalRollbackOperationScorer(
        scorer, threshold=0.0, disable_replan=True
    )
    return canonical.replay_rows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--port", type=int, default=8100)
    args = p.parse_args()

    client = OpenAI(api_key="EMPTY", base_url=f"http://127.0.0.1:{args.port}/v1")
    scorer = VllmRollbackScorer(
        client=client,
        model=Path(args.model_path).name,
        model_path=str(args.model_path),
    )
    rows = load_jsonl(args.input)
    replayed = replay_rows(scorer, rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in replayed:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"vLLM replay: {len(replayed)} rows -> {args.output}")


if __name__ == "__main__":
    main()
