#!/usr/bin/env python3
"""Aggregate per-capability gate files into CAPABILITY_PORTFOLIO.md/json."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from training.scope_round14.typed_schema import ROUND14_CAPABILITIES


def load_gate(path: Path) -> dict | None:
  if not path.exists():
    return None
  return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument(
    "--inputs",
    nargs="*",
    type=Path,
    default=[],
    help="DATASET_GATE.json paths or parent dirs",
  )
  p.add_argument(
    "--output-dir",
    type=Path,
    default=Path("/data/ppnm/Capability_Evolution/SCOPE/outputs/scope_round14"),
  )
  args = p.parse_args()

  rows: list[dict] = []
  for cap in ROUND14_CAPABILITIES:
    gate_path = None
    for inp in args.inputs:
      if inp.is_dir():
        cand = inp / cap / "DATASET_GATE.json"
        if cand.exists():
          gate_path = cand
          break
        cand2 = inp / "DATASET_GATE.json"
        if cand2.exists():
          gate_path = cand2
          break
      elif inp.name == "DATASET_GATE.json" and cap in str(inp):
        gate_path = inp
        break
    if gate_path is None:
      default = args.output_dir / f"gpu_*_{cap}" / "DATASET_GATE.json"
      matches = sorted(args.output_dir.glob(f"**/{cap}/DATASET_GATE.json"))
      if matches:
        gate_path = matches[-1]
    g = load_gate(gate_path) if gate_path else None
    rows.append(
      {
        "capability": cap,
        "status": (g or {}).get("status", "UNRESOLVED"),
        "gate_a": (g or {}).get("gate_a_pass"),
        "gate_b": (g or {}).get("gate_b_pass"),
        "gate_c": (g or {}).get("gate_c_pass"),
        "gate_path": str(gate_path) if gate_path else None,
      }
    )

  out = args.output_dir
  out.mkdir(parents=True, exist_ok=True)
  payload = {
    "schema_version": "scope.round14.portfolio.v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "capabilities": rows,
  }
  json_path = out / "CAPABILITY_PORTFOLIO.json"
  json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

  md_lines = [
    "# Round14 Capability Portfolio",
    "",
    "| Capability | Status | Gate A | Gate B | Gate C |",
    "|---|---|---|---|---|",
  ]
  for r in rows:
    md_lines.append(
      f"| {r['capability']} | {r['status']} | {r['gate_a']} | {r['gate_b']} | {r['gate_c']} |"
    )
  md_path = out / "CAPABILITY_PORTFOLIO.md"
  md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
  print(json.dumps({"json": str(json_path), "md": str(md_path)}, indent=2))


if __name__ == "__main__":
  main()
