#!/usr/bin/env python3
"""build_teacher_readiness (guide §11.2): generate the teacher_readiness
artifact ONLY from real confirmation evidence and identity digests.  Missing
evidence produces the exact deficiency list — never an implicit pass.

Usage:
    PYTHONPATH=src python scripts/build_teacher_readiness.py \
        --domain appworld --bank runs/c/evolve/champion.json \
        --confirm runs/c/confirm_h/confirm_h_r0.json \
        --policy /tmp/rsi_models/Qwen3.5-9B \
        --runtime-config configs/appworld_retry_v1.yaml \
        --out runs/c/teacher_readiness.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RSI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RSI_ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--confirm", required=True)
    ap.add_argument("--policy", default=None, help="model path or policy digest")
    ap.add_argument("--runtime-config", default=None)
    ap.add_argument("--alignment-report", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--status", default=None,
                    help="override the derived status (must still pass evidence checks)")
    args = ap.parse_args()

    from rsi.contracts.common import digest_of, digest_of_text
    from rsi.harness.loader import load_bank_ref
    from rsi.opd.readiness import TeacherReadiness, load_readiness

    problems: list[str] = []
    bank = load_bank_ref(args.bank)

    confirm_path = Path(args.confirm)
    confirm: dict = {}
    if confirm_path.is_file():
        confirm = json.loads(confirm_path.read_text("utf-8"))
    else:
        problems.append(f"confirm result missing: {confirm_path}")

    paired_delta = confirm.get("delta_h_macro_strict")
    if paired_delta is None:
        problems.append("confirm result carries no delta_h_macro_strict (pre-PR-03 confirm files cannot authorize training)")
    interval = confirm.get("paired_interval_strict")
    n_paired = sum(d.get("n_paired", 0) for d in (confirm.get("domains") or {}).values())
    if n_paired == 0:
        problems.append("confirm result has zero paired tasks")
    if isinstance(paired_delta, (int, float)) and paired_delta <= 0:
        # an honest negative confirm yields NO_CONFIRMED_GAIN, not READY
        problems.append(f"strict paired delta {paired_delta} <= 0 — NO_CONFIRMED_GAIN")

    # identities
    policy_digest = args.policy or ""
    if args.policy and Path(args.policy).is_dir():
        policy_digest = digest_of_text(f"hf_model:{args.policy}")
    if not policy_digest:
        problems.append("policy identity missing (pass --policy)")
    runtime_digest = ""
    env_digest = ""
    if args.runtime_config:
        import yaml

        raw = yaml.safe_load((RSI_ROOT / args.runtime_config).read_text("utf-8")) or {}
        runtime_digest = digest_of(raw.get("runtime") or {})
        domains = ((raw.get("suite") or {}).get("domains") or [])
        if domains and domains[0] == args.domain:
            try:
                from rsi.environments.registry import global_registry

                reg = global_registry()
                env_digest = digest_of(reg.get(domains[0]).adapter_factory().environment_identity())
            except Exception as e:  # noqa: BLE001
                problems.append(f"environment identity unresolvable: {e!r}")
        else:
            problems.append("runtime config domains do not cover the readiness domain")
    else:
        problems.append("runtime config missing (pass --runtime-config)")

    status = args.status or ("READY_FOR_PILOT" if not problems else "UNMEASURED")
    if args.status and problems:
        # an explicit status override cannot launder missing evidence
        problems.append(f"--status {args.status} requested but evidence is incomplete")

    art = TeacherReadiness(
        domain=args.domain,
        status=status,
        code_commit=_git_commit(),
        bank_digest=bank.bank_digest,
        policy_digest=policy_digest,
        runtime_contract_digest=runtime_digest,
        environment_digest=env_digest,
        metric_contract_digest=digest_of(confirm.get("metric_contract", "unversioned")),
        confirm_manifest_digest=digest_of_text(json.dumps(confirm.get("consumed", ""), sort_keys=True)),
        confirm_result_ref=str(confirm_path),
        paired_delta=(float(paired_delta) if isinstance(paired_delta, (int, float)) else None),
        paired_interval=(tuple(float(x) for x in interval) if interval else None),
        alignment_report_ref=args.alignment_report,
    )
    art.save(args.out)
    payload = {"ok": not problems, "status": status, "out": args.out, "problems": problems}
    print(json.dumps(payload, indent=1))
    return 0 if not problems else 1


def _git_commit() -> str:
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=RSI_ROOT, capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


if __name__ == "__main__":
    raise SystemExit(main())
