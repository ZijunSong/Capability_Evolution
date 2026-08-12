#!/usr/bin/env python3
"""Minimal Dup method ablation (B typed local + proxy baselines)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
  sys.path.insert(0, str(_REPO))

BASE_MODEL = "/data/ppnm/models/Qwen2.5-7B-Instruct"
O7_SEED42 = _REPO / "outputs/scope_round5/merged/o7_r64_seed42"
DUP_SDI = _REPO / "artifacts/datasets/dup_sdi_round3"


def git_commit() -> str:
  try:
    return subprocess.check_output(
      ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True
    ).strip()
  except Exception:
    return "unknown"


def load_json(path: Path) -> dict[str, Any] | None:
  if not path.exists():
    return None
  return json.loads(path.read_text(encoding="utf-8"))


def offline_proxy_metrics() -> dict[str, Any]:
  """Proxy: base vs O7 offline gap when full-trace SFT is too heavy for Round14."""
  return {
    "mode": "offline_proxy",
    "base_model": BASE_MODEL,
    "o7_checkpoint": str(O7_SEED42),
    "note": (
      "A_full_trace_sft deferred: would require full-trace CE on dup_sdi_round3. "
      "Proxy compares known O7 offline discriminative metrics vs base."
    ),
    "limitations": [
      "No fresh closed-loop for variant A in Round14 time budget",
      "B reuses GPU0 wave0 retirement when available",
    ],
  }


def count_infosafe(train: Path) -> dict[str, int]:
  if not train.exists():
    return {"total": 0, "info_safe": 0}
  total = info_safe = 0
  for line in train.open(encoding="utf-8"):
    if not line.strip():
      continue
    total += 1
    row = json.loads(line)
    if row.get("info_safe") is True or row.get("information_safe") is True:
      info_safe += 1
  return {"total": total, "info_safe": info_safe}


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument(
    "--output-dir",
    type=Path,
    default=_REPO / "outputs/scope_round14/gpu7_method_ablation",
  )
  p.add_argument(
    "--gpu0-anchor",
    type=Path,
    default=_REPO / "outputs/scope_round14/gpu0_dup_anchor",
  )
  args = p.parse_args()

  out = args.output_dir
  out.mkdir(parents=True, exist_ok=True)
  results: dict[str, Any] = {}

  # B: typed local O7 — reuse GPU0 retirement
  b_dir = out / "B_typed_local"
  b_dir.mkdir(parents=True, exist_ok=True)
  retire = load_json(args.gpu0_anchor / "RETIREMENT_EVAL.json") or load_json(
    args.gpu0_anchor / "DUP_RETIREMENT_GATE.json"
  )
  if retire:
    (b_dir / "RETIREMENT_EVAL_REUSE.json").write_text(
      json.dumps(retire, indent=2) + "\n", encoding="utf-8"
    )
  results["B_typed_local"] = {
    "status": "complete" if retire else "pending_gpu0",
    "retirement_reuse": str(args.gpu0_anchor),
    "o7_checkpoint": str(O7_SEED42),
  }

  # A: full-trace SFT proxy
  a_dir = out / "A_full_trace_sft"
  a_dir.mkdir(parents=True, exist_ok=True)
  proxy = offline_proxy_metrics()
  (a_dir / "ABLATION_PROXY.json").write_text(json.dumps(proxy, indent=2) + "\n", encoding="utf-8")
  results["A_full_trace_sft"] = proxy

  # C: local + info-safe filter
  c_dir = out / "C_local_infosafe_gate"
  c_dir.mkdir(parents=True, exist_ok=True)
  infosafe_stats = count_infosafe(DUP_SDI / "train.jsonl")
  c_plan = {
    "ablation": "C_local_infosafe_gate",
    "dup_sdi_round3": str(DUP_SDI),
    "infosafe_stats": infosafe_stats,
    "note": "Train filter: info_safe==True when field present; eval uses GPU0 retirement reuse",
    "retirement_reuse": str(args.gpu0_anchor),
  }
  (c_dir / "ABLATION_PLAN.json").write_text(json.dumps(c_plan, indent=2) + "\n", encoding="utf-8")
  results["C_local_infosafe_gate"] = c_plan

  # D: shadow dropout — placeholder pending GPU1-5 positives
  d_dir = out / "D_shadow_dropout"
  d_dir.mkdir(parents=True, exist_ok=True)
  d_plan = {
    "ablation": "D_shadow_dropout",
    "status": "planned",
    "note": "Second-round shadow dropout ablation waits for GPU1-5 positive local gates",
  }
  (d_dir / "ABLATION_PLAN.json").write_text(json.dumps(d_plan, indent=2) + "\n", encoding="utf-8")
  results["D_shadow_dropout"] = d_plan

  report_lines = [
    "# Dup method ablation (Round14)",
    "",
    f"- generated: {datetime.now(timezone.utc).isoformat()}",
    f"- git: {git_commit()}",
    "",
    "## B — typed local O7",
    f"- status: {results['B_typed_local']['status']}",
    f"- retirement reuse: `{args.gpu0_anchor}`",
    "",
    "## A — full-trace SFT",
    "- status: **proxy only** (see A_full_trace_sft/ABLATION_PROXY.json)",
    "",
    "## C — local + info-safe",
    f"- dup_sdi infosafe: {infosafe_stats}",
    "",
    "## D — shadow dropout",
    "- status: planned (blocked on GPU1-5)",
    "",
    "## Limitations",
    "- Variant A not trained end-to-end in Round14; offline proxy documents gap.",
    "- B/C retirement metrics depend on GPU0 wave0 completion.",
  ]
  (out / "METHOD_ABLATION_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
  (out / "METHOD_ABLATION.json").write_text(
    json.dumps(
      {
        "schema_version": "scope.round14.method_ablation.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
      },
      indent=2,
    )
    + "\n",
    encoding="utf-8",
  )
  print(json.dumps({"written": str(out / "METHOD_ABLATION_REPORT.md")}, indent=2))


if __name__ == "__main__":
  main()
