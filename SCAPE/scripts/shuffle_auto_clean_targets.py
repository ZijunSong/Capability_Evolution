#!/usr/bin/env python3
"""Shuffle teacher targets on fixed AUTO_CLEAN training states (marginal-preserving)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-jsonl", type=Path, required=True)
    ap.add_argument("--out-jsonl", type=Path, required=True)
    ap.add_argument("--audit", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=817)
    args = ap.parse_args()
    rows = load_jsonl(args.in_jsonl)
    rng = random.Random(args.seed)
    targets = [
        {
            "response_text": r.get("response_text"),
            "teacher_action": r.get("teacher_action"),
            "prompt_full": r.get("prompt_full"),
        }
        for r in rows
    ]
    shuffled = list(targets)
    rng.shuffle(shuffled)
    n_fixed = 0
    out_rows = []
    for r, t in zip(rows, shuffled):
        nr = dict(r)
        orig = r.get("response_text")
        nr["response_text"] = t["response_text"]
        nr["teacher_action"] = t["teacher_action"]
        nr["prompt_full"] = t["prompt_full"]
        nr["shuffled_target"] = True
        nr["state_target_pairing_shuffled"] = True
        if orig == t["response_text"]:
            n_fixed += 1
        out_rows.append(nr)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    audit = {
        "n": len(rows),
        "n_fixed_points": n_fixed,
        "fixed_rate": n_fixed / max(1, len(rows)),
        "same_unique_states": True,
        "same_query_ids": True,
        "same_update_budget": True,
        "target_marginal_preserved": True,
        "seed": args.seed,
        "note": "prompt_reduced/state fixed; teacher response/full-view pairing shuffled.",
    }
    args.audit.write_text(json.dumps(audit, indent=2) + "\n")
    md = [
        "# AUTO_CLEAN_SHUFFLE_AUDIT",
        "",
        f"- n: {audit['n']}",
        f"- fixed points: {n_fixed} ({audit['fixed_rate']:.4f}) — target near 0",
        "- state fixed, teacher target marginal fixed, pairing shuffled",
        "",
    ]
    args.audit.with_suffix(".md").write_text("\n".join(md) + "\n")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
