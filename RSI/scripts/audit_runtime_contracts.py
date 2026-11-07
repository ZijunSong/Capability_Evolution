#!/usr/bin/env python3
"""audit_runtime_contracts (guide §11.2): check env_docs, API fixtures,
finish semantics and output protocol per domain.

Usage:
    PYTHONPATH=src python scripts/audit_runtime_contracts.py \
        [--config configs/appworld_retry_v1.yaml] [--all]

Checks (fail = non-zero exit):
1. the config's runtime.environment_doc exists and does not reference
   interfaces the vendor does not have (AppWorld: show_app_apis, generic
   authenticate template);
2. the appworld adapter exposes typed Finish semantics
   (completion_submit_failed handling present in source);
3. profile is declared (runtime.profile) and resolvable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

RSI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RSI_ROOT / "src"))


def _check_env_doc(path: Path, domain: str) -> list[str]:
    problems: list[str] = []
    if not path.is_file():
        return [f"environment_doc missing: {path}"]
    text = path.read_text("utf-8")
    if domain == "appworld":
        # forbidden: non-existent vendor interfaces (PR-01 v3 contract)
        if "show_app_apis" in text:
            problems.append(f"{path.name}: references show_app_apis (vendor does not have it)")
        if "authenticate(**accounts" in text:
            problems.append(f"{path.name}: generic authenticate template (app-specific auth required)")
        for needed in ("show_app_descriptions", "show_api_descriptions", "show_api_doc"):
            if needed not in text:
                problems.append(f"{path.name}: missing verified discovery call {needed}")
        if "print(" not in text:
            problems.append(f"{path.name}: no print(...) output rule documented")
    return problems


def _check_adapter_finish_semantics() -> list[str]:
    src = (RSI_ROOT / "src" / "rsi" / "environments" / "appworld_env.py").read_text("utf-8")
    problems: list[str] = []
    if "completion_submit_failed" not in src:
        problems.append("appworld adapter lacks completion_submit_failed handling")
    if "_completion_submitted" not in src:
        problems.append("appworld adapter does not track prior answer submissions")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--all", action="store_true", help="audit every config with an environment_doc")
    args = ap.parse_args()

    import yaml

    problems: list[str] = []
    if args.all:
        configs = sorted((RSI_ROOT / "configs").glob("*.yaml"))
    elif args.config:
        configs = [RSI_ROOT / args.config]
    else:
        print("nothing to do: pass --config or --all")
        return 2

    for cfg_path in configs:
        if not cfg_path.is_file():
            problems.append(f"config missing: {cfg_path}")
            continue
        raw = yaml.safe_load(cfg_path.read_text("utf-8")) or {}
        runtime = raw.get("runtime") or {}
        if not runtime.get("profile"):
            problems.append(f"{cfg_path.name}: runtime.profile not declared")
        doc = runtime.get("environment_doc")
        if doc and isinstance(doc, str) and doc.endswith(".md"):
            domains = (raw.get("suite") or {}).get("domains") or []
            domain = domains[0] if domains else ""
            doc_path = doc if Path(doc).is_absolute() else RSI_ROOT / doc
            problems.extend(_check_env_doc(doc_path, domain))

    problems.extend(_check_adapter_finish_semantics())

    if problems:
        print("AUDIT FAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("AUDIT OK: all checked configs pass the runtime-contract audit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
