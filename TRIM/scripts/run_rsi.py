#!/usr/bin/env python3
"""Outer RSI campaign driver.

``preflight`` and ``plan`` do not train. ``run`` with ``launch.mode=manual_plan``
writes commands and stops before any optimizer step.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_TRIM = Path(__file__).resolve().parents[1]
if str(_TRIM) not in sys.path:
    sys.path.insert(0, str(_TRIM))

from trim.rsi.base_contract import assert_base_contract, build_base_contract, write_contract
from trim.rsi.config import load_config


def _path_from_config(config_path: Path, rel: str) -> Path:
    del config_path
    raw = Path(rel)
    if raw.is_absolute():
        return raw
    return (_TRIM / rel).resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TRIM fixed-H0 RSI campaign")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("preflight", "plan", "run", "report"), required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    report: dict = {
        "stage": args.stage,
        "campaign_id": config["campaign_id"],
        "executed_training": False,
        "updated": False,
        "ready_to_launch": False,
        "unimplemented": [
            "live proposer HTTP client",
            "multi-node GPU confirmation",
            "verl/fsdp2 bank OPD path",
        ],
        "backend": config["training"].get("backend"),
        "launch_mode": config["launch"].get("mode"),
    }
    missing = []
    contract_path = _path_from_config(args.config, config["base"]["contract"])
    if not contract_path.is_file():
        payload = build_base_contract(
            code_commit="UNBOUND",
            retrieval_backend=str(config["base"].get("retrieval_backend") or "local_bm25"),
            reward_protocol=str(config["training"]["rl_reward"]),
            metric_protocol=str(config["teacher_gate"]["metric"]),
        )
        payload["bindings"] = "template_unbound"
        write_contract(contract_path, payload)
        report["wrote_template_contract"] = str(contract_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    try:
        assert_base_contract(contract)
    except ValueError as exc:
        report["status"] = "PREFLIGHT_FAILED"
        report["stop_reason"] = str(exc)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 2
    for key in ("initial_manifest",):
        path = _path_from_config(args.config, config["policy"][key])
        if not path.is_file():
            missing.append(str(path))
    split_path = _path_from_config(args.config, config["data"]["split_manifest"])
    if not split_path.is_file():
        missing.append(str(split_path))
    if contract.get("bindings") == "template_unbound" or contract.get("code_commit") == "UNBOUND":
        missing.append("base contract is a template and is not bound to a corpus or commit")
    report["missing_bindings"] = missing
    report["ready_to_launch"] = not missing
    if args.stage == "preflight":
        report["status"] = "PREFLIGHT_OK" if not missing else "PREFLIGHT_UNBOUND"
        report["stop_reason"] = "" if not missing else "bindings_unresolved"
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    if args.stage == "plan":
        report["status"] = "PLAN_ONLY"
        report["stop_reason"] = "manual_plan_does_not_train"
        report["train_command_shape"] = (
            "python TRIM/scripts/run_train_multinode.py --train_method trim "
            "--component all --harness-bank $FROZEN_BANK --rl-reward "
            + str(config["training"]["rl_reward"])
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    if args.stage == "run":
        if config["launch"].get("mode") == "manual_plan" or missing:
            report["status"] = "AWAITING_MANUAL_LAUNCH"
            report["stop_reason"] = "manual_plan_does_not_execute_training"
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 0
        report["status"] = "RUN_NOT_WIRED"
        report["stop_reason"] = "automatic cluster launch is not configured"
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 2
    report["status"] = "REPORT_ONLY"
    report["stop_reason"] = "no campaign state was executed in this process"
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
