"""Seal an existing SFT jsonl into a v2 dataset. Does not start a GPU job.

The input rows are copied as token ids. This script does not decode them,
apply a chat template, add EOS, or drop rows to make the file smaller.
"""

from pathlib import Path
import argparse
import sys

_TRIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TRIM))

from trim.training.sft_ablation.contracts import WEIGHT_POLICY
from trim.training.sft_ablation.identity import tokenizer_digest
from trim.training.sft_engine.data import iter_jsonl_rows, seal_dataset


def _vocab_size(model: str, override: int) -> int:
    if int(override) > 0:
        return int(override)
    path = Path(model) / "config.json"
    if not path.is_file():
        raise SystemExit("pass --vocab-size or a local --model-name directory with config.json")
    import json

    config = json.loads(path.read_text(encoding="utf-8"))
    if "vocab_size" in config:
        return int(config["vocab_size"])
    text = config.get("text_config") or {}
    if isinstance(text, dict) and "vocab_size" in text:
        return int(text["vocab_size"])
    raise SystemExit("config.json has no vocab_size; pass --vocab-size")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="prepare_sft_train_data", allow_abbrev=False)
    parser.add_argument("--experiment", required=True, choices=tuple(WEIGHT_POLICY))
    parser.add_argument("--rows", required=True, help="Existing rows.jsonl. Read only.")
    parser.add_argument("--source-contract", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--vocab-size", type=int, default=0)
    parser.add_argument("--tokenizer-digest", default="")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--val-fraction", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    digest = str(args.tokenizer_digest or tokenizer_digest(args.model_name))
    if digest in {"", "missing", "none"}:
        raise SystemExit("tokenizer digest is missing; pass --tokenizer-digest or a model directory with tokenizer files")
    rows = list(iter_jsonl_rows(Path(args.rows)))
    manifest = seal_dataset(
        rows,
        out_dir=Path(args.out),
        experiment=args.experiment,
        source_contract_path=Path(args.source_contract),
        tokenizer_digest=digest,
        vocab_size=_vocab_size(args.model_name, int(args.vocab_size)),
        max_length=int(args.max_length),
        val_fraction=float(args.val_fraction),
        seed=int(args.seed),
    )
    print(
        f"sealed {manifest['n_rows']} {args.experiment} rows "
        f"digest={manifest['package_digest'][:16]} arrow={bool(manifest.get('arrow_path'))}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        raise SystemExit(str(exc))
