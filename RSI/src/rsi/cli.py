"""The single CLI entrypoint of RSI.

All business logic lives in package modules; this file only parses arguments
and dispatches.  Every command must run for real in a clean install
(``python -m rsi.cli ...``): plan artifacts are marked ``executed=false`` and
``campaign run`` exits non-zero when a required backend capability is missing
instead of returning a "successful" plan report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


def _load_cfg(args: argparse.Namespace):
    from rsi.config import load_config

    return load_config(args.config)


def cmd_doctor(args: argparse.Namespace) -> int:
    from rsi.doctor import run_doctor

    cfg = _load_cfg(args)
    report = run_doctor(cfg)
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if report.get("ok") else 1


def cmd_data_prepare(args: argparse.Namespace) -> int:
    from rsi.evaluation.data_prep import prepare_data

    cfg = _load_cfg(args)
    result = prepare_data(cfg)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


def cmd_data_validate(args: argparse.Namespace) -> int:
    from rsi.evaluation.data_prep import validate_data

    cfg = _load_cfg(args)
    result = validate_data(cfg)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if result.get("ok") else 1


def cmd_eval(args: argparse.Namespace) -> int:
    from rsi.evaluation.service import run_eval

    cfg = _load_cfg(args)
    result = run_eval(
        cfg,
        harness=args.harness,
        bank_ref=args.bank,
        split=args.split,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if result.get("ok") else 1


def cmd_evolve(args: argparse.Namespace) -> int:
    from rsi.evolution.service import run_evolve

    cfg = _load_cfg(args)
    result = run_evolve(cfg, initial_bank_ref=args.bank)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if result.get("ok") else 1


def cmd_probe_opd(args: argparse.Namespace) -> int:
    from rsi.opd.service import run_probe_opd

    cfg = _load_cfg(args)
    result = run_probe_opd(cfg, bank_ref=args.bank)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if result.get("ok") else 1


def cmd_train(args: argparse.Namespace) -> int:
    from rsi.training.service import run_train

    cfg = _load_cfg(args)
    result = run_train(cfg, bank_ref=args.bank)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if result.get("ok") else 1


def cmd_campaign_run(args: argparse.Namespace) -> int:
    from rsi.campaign.controller import CampaignController

    cfg = _load_cfg(args)
    controller = CampaignController.from_config(cfg, run_dir=Path(args.run_dir) if args.run_dir else None)
    result = controller.run()
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if result.get("ok") else 1


def cmd_campaign_resume(args: argparse.Namespace) -> int:
    from rsi.campaign.controller import CampaignController

    controller = CampaignController.resume(Path(args.run_dir))
    result = controller.run()
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if result.get("ok") else 1


def cmd_campaign_report(args: argparse.Namespace) -> int:
    from rsi.campaign.reports import build_report

    report = build_report(Path(args.run_dir))
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


def cmd_campaign_plan(args: argparse.Namespace) -> int:
    from rsi.campaign.planning import build_plan

    cfg = _load_cfg(args)
    plan = build_plan(cfg)
    plan["executed"] = False
    out = Path(args.out) if args.out else Path("runs") / f"{cfg.campaign_id}_plan.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=2, sort_keys=True, ensure_ascii=False), "utf-8")
    print(json.dumps({"plan_path": str(out), "executed": False}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rsi", description="RSI: Harness self-evolution + HarnessOPD internalization")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_config(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", required=True, help="campaign config YAML")

    p = sub.add_parser("doctor", help="check environment, deps, configs and identity")
    add_config(p)
    p.set_defaults(func=cmd_doctor)

    p_data = sub.add_parser("data", help="dataset preparation and validation")
    data_sub = p_data.add_subparsers(dest="data_command", required=True)
    p_prep = data_sub.add_parser("prepare", help="materialize frozen splits")
    add_config(p_prep)
    p_prep.set_defaults(func=cmd_data_prepare)
    p_val = data_sub.add_parser("validate", help="validate frozen splits and group disjointness")
    add_config(p_val)
    p_val.set_defaults(func=cmd_data_validate)

    p = sub.add_parser("eval", help="evaluate a policy under minimal/bank harness")
    add_config(p)
    p.add_argument("--harness", choices=["minimal", "bank"], default="minimal")
    p.add_argument("--bank", default=None, help="bank manifest/ref (required for --harness bank)")
    p.add_argument("--split", default="harness_select")
    p.add_argument("--seed", type=int, default=None)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("evolve", help="run harness code evolution")
    add_config(p)
    p.add_argument("--bank", default=None, help="initial bank ref (default: empty)")
    p.set_defaults(func=cmd_evolve)

    p = sub.add_parser("probe-opd", help="run the real distillability probe")
    add_config(p)
    p.add_argument("--bank", required=True, help="champion bank ref")
    p.set_defaults(func=cmd_probe_opd)

    p = sub.add_parser("train", help="run HarnessOPD + outcome RL internalization")
    add_config(p)
    p.add_argument("--bank", required=True, help="champion bank ref")
    p.set_defaults(func=cmd_train)

    p_campaign = sub.add_parser("campaign", help="multi-round campaign controller")
    campaign_sub = p_campaign.add_subparsers(dest="campaign_command", required=True)
    p_run = campaign_sub.add_parser("run", help="execute the full campaign")
    add_config(p_run)
    p_run.add_argument("--run-dir", default=None)
    p_run.set_defaults(func=cmd_campaign_run)
    p_resume = campaign_sub.add_parser("resume", help="resume an interrupted campaign")
    p_resume.add_argument("--run-dir", required=True)
    p_resume.set_defaults(func=cmd_campaign_resume)
    p_report = campaign_sub.add_parser("report", help="print the machine-readable report")
    p_report.add_argument("--run-dir", required=True)
    p_report.set_defaults(func=cmd_campaign_report)
    p_plan = campaign_sub.add_parser("plan", help="write a plan artifact (executed=false)")
    add_config(p_plan)
    p_plan.add_argument("--out", default=None)
    p_plan.set_defaults(func=cmd_campaign_plan)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
