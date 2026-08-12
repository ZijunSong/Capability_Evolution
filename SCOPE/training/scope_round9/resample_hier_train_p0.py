#!/usr/bin/env python3
"""P0: upsample CONTINUE in hier_sdi/train without touching holdout.

Keeps all ROLLBACK_TO rows; samples CONTINUE with replacement until
continue_frac ≈ target (default 0.75). Writes train.jsonl and a manifest;
backs up the previous train to train.before_p0.jsonl once.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_DIR = _REPO / "artifacts/datasets/scope_round9/hier_sdi"


def gold_op(row: dict) -> str:
    return str(
        (row.get("target_action") or {}).get("operation")
        or row.get("gold_operation")
        or row.get("operation")
        or "CONTINUE"
    )


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def upsample_continue(
    rows: list[dict], *, target_frac: float, seed: int
) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    cont = [r for r in rows if gold_op(r) == "CONTINUE"]
    roll = [r for r in rows if gold_op(r) == "ROLLBACK_TO"]
    other = [r for r in rows if gold_op(r) not in ("CONTINUE", "ROLLBACK_TO")]
    if not cont or not roll:
        raise SystemExit(f"need both CONTINUE and ROLLBACK rows; got C={len(cont)} R={len(roll)}")
    # Keep all ROLLBACK; choose n_cont so C/(C+R)=target_frac
    n_roll = len(roll)
    n_cont = int(round(target_frac / max(1.0 - target_frac, 1e-6) * n_roll))
    n_cont = max(n_cont, len(cont))
    sampled = rng.choices(cont, k=n_cont) if n_cont > len(cont) else rng.sample(cont, n_cont)
    out = sampled + list(roll) + list(other)
    rng.shuffle(out)
    prior = Counter(gold_op(r) for r in out)
    meta = {
        "n_before": len(rows),
        "n_after": len(out),
        "n_continue_src": len(cont),
        "n_rollback_kept": n_roll,
        "n_continue_after": prior.get("CONTINUE", 0),
        "continue_frac_after": prior.get("CONTINUE", 0) / max(len(out), 1),
        "target_frac": target_frac,
        "seed": seed,
        "prior_after": dict(prior),
    }
    return out, meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DIR)
    p.add_argument("--target-frac", type=float, default=0.75)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    train_path = args.data_dir / "train.jsonl"
    bak = args.data_dir / "train.before_p0.jsonl"
    if not bak.exists():
        shutil.copy2(train_path, bak)
    rows = load_jsonl(bak if bak.exists() else train_path)
    # Always resample from the pre-P0 backup to stay idempotent.
    out, meta = upsample_continue(rows, target_frac=args.target_frac, seed=args.seed)
    with train_path.open("w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta_path = args.data_dir / "TRAIN_P0_RESAMPLE.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
