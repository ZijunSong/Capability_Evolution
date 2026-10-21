#!/usr/bin/env python3
"""One-shot: seal the Teacher SFT legacy rows.jsonl into a v2 sealed dataset.

The accelerate_ddp CE backend (trim.training.sft_engine.fixed_runtime.run_fixed_sft)
calls load_sealed, which requires a sealed dataset directory (DATASET.json +
READY.json + SOURCE_CONTRACT.json + dataset.jsonl, with sha256 checks). The legacy
build_teacher_sft_rows path (_write_rows) only writes rows.jsonl + REJECTED.json,
so this script bridges the gap by calling seal_dataset on the already-collected
rows.

Usage (on a host that can import trim + load the GPT-OSS tokenizer):
  PYTHONPATH=/mnt/songzijun/Capability_Evolution/TRIM \
    python3 scripts/seal_teacher_sft_dataset.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_TRIM = Path(__file__).resolve().parents[1]
if str(_TRIM) not in sys.path:
    sys.path.insert(0, str(_TRIM))

from trim.training.sft_ablation.contracts import SFTBatchRow  # noqa: E402
from trim.training.sft_ablation.identity import tokenizer_digest  # noqa: E402
from trim.training.sft_engine.data import load_sealed, seal_dataset  # noqa: E402

# --- Inputs (collected by t-20261020164236-7rk9c, _action_of fix applied) ---
COLLECT_OUT = Path(
    "/mnt/songzijun/outputs/gpt-oss-20b-teacher-sft-bf16-32gpu_20261019_true_fix1"
)
ROWS_JSONL = COLLECT_OUT / "sft_rows" / "rows.jsonl"
CONTRACT_JSON = COLLECT_OUT / "CONTRACT.json"
SEALED_DIR = COLLECT_OUT / "sft_sealed"

MODEL_PATH = "/mnt/songzijun/models/openai/gpt-oss-20b-bf16-dequant"
EXPERIMENT = "teacher_sft"
VOCAB_SIZE = 201088  # gpt-oss-20b config.json
MAX_LENGTH = 8192  # --max-model-len used during collect
SEED = 42


def _read_rows(path: Path) -> list[SFTBatchRow]:
    rows: list[SFTBatchRow] = []
    n = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(SFTBatchRow.from_dict(json.loads(line)))
            n += 1
            if n % 20000 == 0:
                print(f"  read {n} rows...", flush=True)
    return rows


def main() -> int:
    print(f"sealing Teacher SFT dataset", flush=True)
    print(f"  rows:   {ROWS_JSONL}", flush=True)
    print(f"  source: {CONTRACT_JSON}", flush=True)
    print(f"  out:    {SEALED_DIR}", flush=True)
    if not ROWS_JSONL.is_file():
        raise FileNotFoundError(f"missing rows: {ROWS_JSONL}")
    if not CONTRACT_JSON.is_file():
        raise FileNotFoundError(f"missing contract: {CONTRACT_JSON}")

    # tokenizer_digest MUST match what run_fixed_sft recomputes at train time
    # (fixed_runtime.py: live_tokenizer = tokenizer_digest(model_name)); and it
    # must match the CONTRACT written during collect. Verify both.
    contract = json.loads(CONTRACT_JSON.read_text(encoding="utf-8"))
    contract_tok = str(contract.get("tokenizer_digest") or "")
    live_tok = tokenizer_digest(MODEL_PATH)
    print(f"  contract tokenizer_digest: {contract_tok}", flush=True)
    print(f"  live     tokenizer_digest: {live_tok}", flush=True)
    if contract_tok and contract_tok != live_tok:
        raise SystemExit(
            f"tokenizer_digest mismatch: contract={contract_tok} live={live_tok}"
        )

    print("reading rows.jsonl (streaming)...", flush=True)
    rows = _read_rows(ROWS_JSONL)
    print(f"  read {len(rows)} rows total", flush=True)
    if not rows:
        raise SystemExit("no rows to seal")

    print("calling seal_dataset...", flush=True)
    manifest = seal_dataset(
        rows,
        out_dir=SEALED_DIR,
        experiment=EXPERIMENT,
        source_contract_path=CONTRACT_JSON,
        tokenizer_digest=live_tok,
        vocab_size=VOCAB_SIZE,
        max_length=MAX_LENGTH,
        val_fraction=0.0,
        seed=SEED,
    )
    print(f"  sealed: n_rows={manifest.get('n_rows')}", flush=True)
    print(f"  package_digest={manifest.get('package_digest')}", flush=True)
    print(f"  rows_sha256={manifest.get('rows_sha256')[:16]}...", flush=True)

    # verify load_sealed round-trips (this is what run_fixed_sft will do)
    print("verifying load_sealed...", flush=True)
    ds = load_sealed(
        manifest_path=SEALED_DIR / "DATASET.json",
        rows_path=SEALED_DIR / "dataset.jsonl",
        expected_experiment=EXPERIMENT,
    )
    print(f"  load_sealed OK: experiment={ds.experiment}", flush=True)
    train_pop = ds.train_positions()
    print(f"  train_positions: {len(train_pop)}", flush=True)
    if not train_pop:
        raise SystemExit("sealed dataset has no training rows")
    print("DONE. sealed dataset ready for accelerate_ddp train.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
