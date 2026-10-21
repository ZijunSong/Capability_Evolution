"""Seal and read existing SFT rows without re-tokenizing them.

The CPU prepare step is the only place that parses the full JSONL. Training
ranks seek the rows selected for the current global batch.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from trim.training.sft_ablation.contracts import (
    PROFILES,
    WEIGHT_POLICY,
    SFTBatchRow,
    package_digest,
    supervised_weight,
)
from trim.training.sft_ablation.io_util import atomic_write_json, read_json, sha256_file

DATASET_SCHEMA = "sft_dataset_v2"


def iter_jsonl_rows(path: Path) -> Iterable[SFTBatchRow]:
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
            try:
                yield SFTBatchRow.from_dict(payload)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc


def percentile(values: Sequence[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(int(value) for value in values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * float(fraction)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(ordered[low])
    weight = rank - low
    return float(ordered[low]) * (1.0 - weight) + float(ordered[high]) * weight


def length_stats(rows: Sequence[SFTBatchRow]) -> dict[str, Any]:
    lengths = [len(row.prompt_ids) + len(row.response_ids) for row in rows]
    supervised = [sum(1 for bit in row.response_mask if bit) for row in rows]
    by_experiment: dict[str, int] = {}
    for row in rows:
        by_experiment[row.experiment] = by_experiment.get(row.experiment, 0) + 1
    return {
        "n_rows": len(rows),
        "p50": percentile(lengths, 0.50),
        "p90": percentile(lengths, 0.90),
        "p95": percentile(lengths, 0.95),
        "p99": percentile(lengths, 0.99),
        "max": max(lengths) if lengths else 0,
        "supervised_tokens": int(sum(supervised)),
        "weighted_token_mass": float(sum(supervised_weight(row) for row in rows)),
        "input_tokens": int(sum(lengths)),
        "n_rows_by_experiment": by_experiment,
        "n_queries": len({row.query_id for row in rows}),
    }


def split_query_ids(query_ids: Sequence[str], *, fraction: float, seed: int) -> tuple[list[str], list[str]]:
    """Hold out whole queries. Trajectories of one query never straddle the split."""
    unique = sorted(set(query_ids))
    if fraction < 0 or fraction >= 1:
        raise ValueError("val fraction must be in [0, 1)")
    if fraction == 0 or len(unique) < 2:
        return unique, []
    rng = random.Random(int(seed))
    order = list(unique)
    rng.shuffle(order)
    n_val = max(1, int(round(len(order) * float(fraction))))
    n_val = min(n_val, len(order) - 1)
    val = sorted(order[:n_val])
    train = sorted(order[n_val:])
    if set(val) & set(train):
        raise RuntimeError("query split overlap")
    return train, val


def check_row(
    row: SFTBatchRow,
    *,
    experiment: str,
    max_length: int,
    vocab_size: int,
    weight_policy: str,
) -> None:
    if row.experiment != experiment:
        raise ValueError(f"row {row.row_id} belongs to {row.experiment}, refused by {experiment}")
    if experiment not in PROFILES:
        raise ValueError(f"unknown experiment {experiment}")
    row.validate(max_length)
    if weight_policy == "uniform_one_v1" and float(row.weight) != 1.0:
        raise ValueError(f"row {row.row_id} weight {row.weight} is not uniform_one_v1")
    if weight_policy not in WEIGHT_POLICY.values():
        raise ValueError(f"unknown weight policy {weight_policy}")
    if int(vocab_size) < 1:
        raise ValueError("vocab_size is required")
    illegal = [token for token in row.prompt_ids + row.response_ids if token >= int(vocab_size)]
    if illegal:
        raise ValueError(f"row {row.row_id} has token ids outside vocab {vocab_size}")


def _write_arrow(path: Path, rows: Sequence[SFTBatchRow]) -> bool:
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
    except Exception:
        return False
    payload = [json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True) for row in rows]
    table = pa.table({"row_json": payload})
    with path.open("wb") as handle:
        with ipc.new_file(handle, table.schema) as writer:
            writer.write_table(table)
    return True


def seal_dataset(
    rows: Sequence[SFTBatchRow],
    *,
    out_dir: Path,
    experiment: str,
    source_contract_path: Path,
    tokenizer_digest: str,
    vocab_size: int,
    max_length: int,
    val_fraction: float = 0.0,
    seed: int = 0,
) -> dict[str, Any]:
    """Write a sealed dataset. ``READY.json`` is the last file."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not tokenizer_digest or tokenizer_digest in {"missing", "none"}:
        raise ValueError("refusing to seal a dataset without a tokenizer digest")
    policy = WEIGHT_POLICY[experiment]
    ordered = sorted(rows, key=lambda row: row.row_id)
    duplicates = []
    seen: set[str] = set()
    for row in ordered:
        check_row(
            row,
            experiment=experiment,
            max_length=max_length,
            vocab_size=vocab_size,
            weight_policy=policy,
        )
        if row.row_id in seen:
            duplicates.append(row.row_id)
        seen.add(row.row_id)
    if duplicates:
        atomic_write_json(out_dir / "DUPLICATES.json", {"row_ids": sorted(set(duplicates))})
        raise ValueError(
            f"{len(set(duplicates))} duplicate row_id values; left them in place and wrote DUPLICATES.json"
        )
    source_contract_path = Path(source_contract_path)
    source_payload = read_json(source_contract_path)
    source_experiment = source_payload.get("experiment") or (source_payload.get("immutable") or {}).get("experiment")
    if source_experiment and source_experiment != experiment:
        raise ValueError(f"source contract experiment {source_experiment} != {experiment}")
    source_bytes = source_contract_path.read_bytes()
    (out_dir / "SOURCE_CONTRACT.json").write_bytes(source_bytes)
    train_queries, val_queries = split_query_ids([row.query_id for row in ordered], fraction=val_fraction, seed=seed)
    train_set = set(train_queries)
    rows_path = out_dir / "dataset.jsonl"
    index: list[dict[str, Any]] = []
    hasher = hashlib.sha256()
    with rows_path.open("wb") as handle:
        for position, row in enumerate(ordered):
            blob = (json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
            offset = handle.tell()
            handle.write(blob)
            hasher.update(blob)
            index.append(
                {
                    "position": position,
                    "row_id": row.row_id,
                    "query_id": row.query_id,
                    "offset": offset,
                    "length": len(row.prompt_ids) + len(row.response_ids),
                    "supervised_tokens": int(sum(1 for bit in row.response_mask if bit)),
                    "weight": float(row.weight),
                    "train": row.query_id in train_set,
                }
            )
    file_sha = hasher.hexdigest()
    if file_sha != sha256_file(rows_path):
        raise RuntimeError("dataset.jsonl hash changed while sealing")
    arrow_path = out_dir / "dataset.arrow"
    wrote_arrow = _write_arrow(arrow_path, ordered)
    digest = package_digest(list(ordered))
    manifest = {
        "schema_version": DATASET_SCHEMA,
        "experiment": experiment,
        "weight_policy": policy,
        "tokenizer_digest": tokenizer_digest,
        "vocab_size": int(vocab_size),
        "max_length": int(max_length),
        "package_digest": digest,
        "rows_sha256": file_sha,
        "rows_path": rows_path.name,
        "arrow_path": arrow_path.name if wrote_arrow else None,
        "arrow_sha256": sha256_file(arrow_path) if wrote_arrow else None,
        "source_contract_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "n_rows": len(ordered),
        "stats": length_stats(ordered),
        "split": {
            "method": "none" if not val_queries else "query_id",
            "seed": int(seed),
            "val_fraction": float(val_fraction),
            "train_query_ids": train_queries,
            "val_query_ids": val_queries,
        },
        "row_index": index,
    }
    atomic_write_json(out_dir / "DATASET.json", manifest)
    atomic_write_json(
        out_dir / "READY.json",
        {
            "ready": True,
            "schema_version": DATASET_SCHEMA,
            "experiment": experiment,
            "package_digest": digest,
            "n_rows": len(ordered),
            "rows_sha256": file_sha,
        },
    )
    return manifest


@dataclass
class SealedDataset:
    root: Path
    manifest: dict[str, Any]
    rows_path: Path

    def __len__(self) -> int:
        return int(self.manifest["n_rows"])

    @property
    def experiment(self) -> str:
        return str(self.manifest["experiment"])

    @property
    def package_digest(self) -> str:
        return str(self.manifest["package_digest"])

    @property
    def tokenizer_digest(self) -> str:
        return str(self.manifest["tokenizer_digest"])

    def train_positions(self) -> list[int]:
        return [int(item["position"]) for item in self.manifest["row_index"] if item.get("train", True)]

    def get(self, position: int) -> SFTBatchRow:
        position = int(position)
        if self.rows_path.suffix == ".arrow":
            return self._arrow_row(position)
        item = self.manifest["row_index"][position]
        with self.rows_path.open("rb") as handle:
            handle.seek(int(item["offset"]))
            line = handle.readline()
        row = SFTBatchRow.from_dict(json.loads(line))
        if row.row_id != item["row_id"]:
            raise ValueError(f"row index mismatch at {position}: {row.row_id} != {item['row_id']}")
        return row

    def _arrow_row(self, position: int) -> SFTBatchRow:
        try:
            import pyarrow.ipc as ipc
        except Exception as exc:
            raise RuntimeError("reading dataset.arrow requires pyarrow") from exc
        table = getattr(self, "_arrow_table", None)
        if table is None:
            with self.rows_path.open("rb") as handle:
                table = ipc.open_file(handle).read_all()
            self._arrow_table = table
        payload = table.column("row_json")[position].as_py()
        row = SFTBatchRow.from_dict(json.loads(payload))
        expected = self.manifest["row_index"][position]["row_id"]
        if row.row_id != expected:
            raise ValueError(f"arrow row mismatch at {position}")
        return row


def load_sealed(
    *,
    manifest_path: Path,
    rows_path: Path | None = None,
    expected_experiment: str | None = None,
) -> SealedDataset:
    manifest_path = Path(manifest_path)
    root = manifest_path.parent
    ready_path = root / "READY.json"
    if not ready_path.is_file():
        raise FileNotFoundError(f"dataset is not sealed (missing {ready_path})")
    ready = read_json(ready_path)
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != DATASET_SCHEMA or ready.get("schema_version") != DATASET_SCHEMA:
        raise ValueError("unsupported dataset manifest")
    if ready.get("package_digest") != manifest.get("package_digest") or ready.get("rows_sha256") != manifest.get("rows_sha256"):
        raise ValueError("READY.json does not match DATASET.json")
    if expected_experiment and manifest.get("experiment") != expected_experiment:
        raise ValueError(
            f"refusing {manifest.get('experiment')} rows for {expected_experiment}"
        )
    chosen = Path(rows_path) if rows_path else root / str(manifest["rows_path"])
    if not chosen.is_file():
        raise FileNotFoundError(f"missing sealed rows: {chosen}")
    digest = sha256_file(chosen)
    if chosen.suffix == ".arrow":
        expected = manifest.get("arrow_sha256")
        if not expected or digest != expected:
            raise ValueError("dataset.arrow hash does not match the manifest")
    elif digest != manifest.get("rows_sha256"):
        raise ValueError("dataset rows hash does not match the manifest")
    source = root / "SOURCE_CONTRACT.json"
    if source.is_file() and sha256_file(source) != manifest.get("source_contract_sha256"):
        raise ValueError("SOURCE_CONTRACT.json hash does not match the manifest")
    return SealedDataset(root=root, manifest=manifest, rows_path=chosen)
