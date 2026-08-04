#!/usr/bin/env python3
"""Archived state HF scorer audit (Round 7 GPU4-6)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.scope_round6.replay_closed_loop_states import main as _replay_main  # noqa: F401
from training.scope_round6.state_sources import load_b6_admission_states
from training.scope_round6.scorer_utils import load_merged_model, score_samples_hf, scorer_paths_for_tag
from training.scope_round6.metrics import aggregate_scored_rows
from training.scope_round7.common import OUT, SEEDS, VALID522, write_json


def audit_seed(seed: int, gpu: str, output_dir: Path) -> None:
    tag = f"o7_{seed}"
    paths = scorer_paths_for_tag(tag)
    model, tokenizer, dev = load_merged_model(paths["merged"], gpu)
    sources = {
        "valid522": str(VALID522),
        f"b6_o7_{seed}": f"o7_{seed}",
    }
    results = {}
    for name, src in sources.items():
        if name == "valid522":
            from training.scope_round6.common import load_jsonl
            from training.scope_round6.scorer_utils import score_samples_hf as score_hf
            samples = load_jsonl(VALID522)
            scored = score_hf(model, tokenizer, samples, dev)
        else:
            samples = load_b6_admission_states(src)
            scored = score_samples_hf(model, tokenizer, samples, dev)
        metrics = aggregate_scored_rows(scored)
        results[name] = metrics
    out = output_dir / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "archived_audit.json", results)
    print(f"Archived audit seed{seed}: {list(results.keys())}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--gpu", default="cuda:0")
    p.add_argument("--output-dir", type=Path, default=OUT / "contract_trace/replay_hf/archived")
    args = p.parse_args()
    audit_seed(args.seed, args.gpu, args.output_dir)


if __name__ == "__main__":
    main()
