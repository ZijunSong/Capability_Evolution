#!/usr/bin/env python3
"""5-query smoke dry-run: validate CLI flags without full GPU rollout."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
  sys.path.insert(0, str(_REPO))

MANIFEST = _REPO / "artifacts/datasets/scope_round14/manifests/R14_SMOKE20.json"
OUT = _REPO / "outputs/scope_round14/smoke5q"


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--gpu", type=int, default=0)
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--output-dir", type=Path, default=OUT)
  p.add_argument("--manifest", type=Path, default=MANIFEST)
  p.add_argument("--resume", action="store_true", default=False)
  args = p.parse_args()

  args.output_dir.mkdir(parents=True, exist_ok=True)
  results: dict = {"checks": []}

  # 1) retirement eval dry-run
  ret_out = args.output_dir / "retirement_dry"
  cmd = [
    sys.executable,
    str(_REPO / "training/scope_round14/run_module_retirement_eval.py"),
    "--capability",
    "duplicate_evidence",
    "--manifest",
    str(args.manifest),
    "--output-dir",
    str(ret_out),
    "--gpu",
    str(args.gpu),
    "--seed",
    str(args.seed),
    "--dry-run",
    "--resume",
  ]
  subprocess.run(cmd, check=True, cwd=_REPO)
  results["checks"].append({"retirement_dry_run": str(ret_out / "RETIREMENT_PLAN.json")})

  # 2) build_capability_evidence
  ev_out = args.output_dir / "evidence"
  subprocess.run(
    [
      sys.executable,
      str(_REPO / "training/scope_round14/build_capability_evidence.py"),
      "--capability",
      "duplicate_evidence",
      "--output-dir",
      str(ev_out),
      "--gpu",
      str(args.gpu),
      "--seed",
      str(args.seed),
      "--manifest",
      str(args.manifest),
    ],
    check=True,
    cwd=_REPO,
  )
  results["checks"].append({"evidence": str(ev_out / "DATASET_GATE.json")})

  # 3) train_local_decision dry-run on rollback_lite if present
  rb = _REPO / "artifacts/datasets/scope_round14/rollback_lite"
  if (rb / "train.jsonl").exists():
    tr_out = args.output_dir / "train_dry"
    subprocess.run(
      [
        sys.executable,
        str(_REPO / "training/scope_round14/train_local_decision.py"),
        "--capability",
        "rollback_lite",
        "--train",
        str(rb / "train.jsonl"),
        "--valid",
        str(rb / "valid.jsonl"),
        "--output-dir",
        str(tr_out),
        "--gpu",
        str(args.gpu),
        "--seed",
        str(args.seed),
        "--dry-run",
      ],
      check=True,
      cwd=_REPO,
    )
    results["checks"].append({"train_dry_run": str(tr_out / "TRAIN_PLAN.json")})

  results["pass"] = all(Path(list(c.values())[0]).exists() for c in results["checks"])
  out_path = args.output_dir / "SMOKE5Q.json"
  out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
  print(json.dumps(results, indent=2))
  if not results["pass"]:
    sys.exit(1)


if __name__ == "__main__":
  main()
