#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from easyopd.methods.scape_component_opd.h1004_post_sweep import (
    build_master_artifacts,
    discover_component_handoffs,
    run_reload_audit,
    write_final_handoff,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/mnt/songzijun/models/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--output-root", type=Path, default=Path("/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD/outputs/component_sweep_0818"))
    parser.add_argument("--mode", choices=["all"], default="all")
    parser.add_argument("--enable-token-budget-stress", action="store_true")
    args = parser.parse_args()

    output_root = args.output_root
    reload_result = run_reload_audit(model_path=args.model, output_dir=output_root / "h100_4" / "post_phase_u" / "qwen3_reload")
    discovery = discover_component_handoffs(output_root)
    master = build_master_artifacts(output_root)
    final = write_final_handoff(reload_result, discovery, master, output_dir=output_root / "h100_4" / "post_phase_u")
    payload = {
        "reload": reload_result.to_dict(),
        "discovery": discovery,
        "master": master,
        "final_handoff": str(final),
        "enable_token_budget_stress": bool(args.enable_token_budget_stress),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
