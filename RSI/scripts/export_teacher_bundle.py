#!/usr/bin/env python3
"""export_teacher_bundle (guide §11.2): export a bank's full audit bundle —
bank source, manifest, digests, the candidate_id -> bank_digest mapping and
the evaluation/confirm references that justify it.

Usage:
    PYTHONPATH=src python scripts/export_teacher_bundle.py \
        --bank runs/<campaign>/evolve/champion.json \
        --out runs/<campaign>/teacher_bundle
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

RSI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RSI_ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True, help="champion.json or bank manifest path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--confirm-result", default=None, help="confirm_h result json to bind")
    args = ap.parse_args()

    from rsi.harness.loader import load_bank_ref

    bank = load_bank_ref(args.bank)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. full bank source + manifest (guide §3.2 C: candidate audit needs the
    #    complete component package, not just digests)
    bank_src = out / "bank"
    if Path(bank.root).resolve() != bank_src.resolve():
        if bank_src.exists():
            shutil.rmtree(bank_src)
        shutil.copytree(bank.root, bank_src)

    bundle = {
        "bank_digest": bank.bank_digest,
        "bank_root_original": str(bank.root),
        "components": [
            {
                "id": c.manifest.id,
                "kind": c.manifest.kind,
                "files": list(c.manifest.files),
                "entrypoint": c.manifest.entrypoint,
                "hook_order": c.manifest.hook_order,
            }
            for c in bank.components
        ],
    }
    # 2. candidate identity mapping (guide §2.1: candidate_id and bank_digest
    #    are TWO identities — record the mapping instead of guessing)
    champ_path = Path(args.bank)
    if champ_path.is_file():
        raw = json.loads(champ_path.read_text("utf-8"))
        for key in ("candidate_id", "bank_dir", "hypothesis", "generation"):
            if key in raw:
                bundle[key] = raw[key]
    # 3. confirm evidence reference (unbound by default — the builder must
    #    NOT invent a confirmation that never ran)
    if args.confirm_result:
        cref = Path(args.confirm_result)
        if cref.is_file():
            shutil.copy2(cref, out / cref.name)
            bundle["confirm_result_ref"] = str(cref)
        else:
            bundle["confirm_result_ref"] = None
            bundle["confirm_result_error"] = f"missing: {cref}"
    else:
        bundle["confirm_result_ref"] = None

    (out / "bundle.json").write_text(
        json.dumps(bundle, indent=2, sort_keys=True, ensure_ascii=False), "utf-8"
    )
    print(json.dumps({"ok": True, "out": str(out), "bank_digest": bank.bank_digest}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
