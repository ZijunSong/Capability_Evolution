"""BrowseComp-Plus query pools.

Canonical RL / RL+OPD split is the official BC+ 830 = 664 train + 166 test.
The 384-query eval pool remains available for older four-cell artifacts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SCOPE = REPO.parent / "SCOPE"
EASYOPD = REPO.parent / "SCAPE-EasyOPD"

CANDIDATE_BCP_ROOTS = (
    SCOPE / "external" / "BrowseComp-Plus",
    Path("/mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus"),
    Path("/data/ppnm/Capability_Evolution/SCOPE/external/BrowseComp-Plus"),
)
CANDIDATE_TRAIN_POOLS = (
    EASYOPD / "manifests" / "COMPONENT_SWEEP_TRAIN_POOL.json",
    Path("/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD/manifests/COMPONENT_SWEEP_TRAIN_POOL.json"),
)
CANDIDATE_EVAL_384 = (
    EASYOPD / "manifests" / "browsecomp_plus_eval_pool_384" / "query_manifest.json",
    Path("/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD/manifests/browsecomp_plus_eval_pool_384/query_manifest.json"),
)
CANDIDATE_SPLITS = (
    REPO / "manifests" / "browsecomp_plus_830" / "SPLIT.json",
    REPO.parent / "SCAPE" / "manifests" / "browsecomp_plus_830" / "SPLIT.json",
    SCOPE / "datagen" / "splits" / "browsecompplus_splits.json",
    REPO / "external" / "harness-1" / "datagen" / "splits" / "browsecompplus_splits.json",
    REPO.parent / "SCAPE" / "external" / "harness-1" / "datagen" / "splits" / "browsecompplus_splits.json",
    Path("/mnt/songzijun/Capability_Evolution/SCOPE/datagen/splits/browsecompplus_splits.json"),
)
CANDIDATE_BCPLUS_830 = (
    REPO / "manifests" / "browsecomp_plus_830",
    REPO.parent / "SCAPE" / "manifests" / "browsecomp_plus_830",
)
CANDIDATE_SENTENCE_STATES = (
    EASYOPD / "outputs" / "component_sweep_0818" / "h100_3_qwen3_faststart" / "sentence_compress" / "TRAIN_STATES_5K.jsonl",
    Path("/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD/outputs/component_sweep_0818/h100_3_qwen3_faststart/sentence_compress/TRAIN_STATES_5K.jsonl"),
    EASYOPD / "outputs" / "component_sweep_0818" / "h100_3_qwen3_faststart" / "sentence_compress" / "EVENT_ACTIVE_STATES_ALL.jsonl",
    Path("/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD/outputs/component_sweep_0818/h100_3_qwen3_faststart/sentence_compress/EVENT_ACTIVE_STATES_ALL.jsonl"),
)

OFFICIAL_384_COUNT = 384
BCPLUS_TOTAL = 830
BCPLUS_TRAIN = 664
BCPLUS_TEST = 166
SCORE_SPLIT_166 = "bcplus_test_166"
SCORE_SPLIT_830 = "bcplus_830"
SCORE_SPLIT_FULL = "bcplus_full"

_SCORE_SPLIT_TEST_ALIASES = frozenset(
    {
        SCORE_SPLIT_166,
        "bcplus_166",
        "test_166",
        "bcplus-test-166",
    }
)
_SCORE_SPLIT_FULL_ALIASES = frozenset(
    {
        SCORE_SPLIT_830,
        SCORE_SPLIT_FULL,
        "bcplus830",
        "all_pool",
        "bcplus-full",
        "bcplus-830",
    }
)


def canonical_score_split(value: str | None, *, default: str | None = None) -> str | None:
    """Map eval-pool aliases onto ``bcplus_test_166`` / ``bcplus_830``."""
    if value is None or str(value).strip() == "":
        return default
    key = str(value).strip().lower().replace("-", "_").replace(" ", "")
    if key in _SCORE_SPLIT_TEST_ALIASES:
        return SCORE_SPLIT_166
    if key in _SCORE_SPLIT_FULL_ALIASES:
        return SCORE_SPLIT_830
    return str(value)


def is_full_score_split(value: str | None) -> bool:
    return canonical_score_split(value) == SCORE_SPLIT_830


def score_split_for_benchmark(benchmark: str) -> str | None:
    """Split-specific ``--benchmark`` names imply an eval pool; ``BC+`` does not."""
    key = str(benchmark or "").strip()
    if key == SCORE_SPLIT_166:
        return SCORE_SPLIT_166
    if key in {SCORE_SPLIT_FULL, SCORE_SPLIT_830}:
        return SCORE_SPLIT_830
    return None


def first_existing(paths: tuple[Path, ...]) -> Path | None:
    for path in paths:
        if path.is_file() or path.is_dir():
            return path
    return None


def default_bcp_root() -> Path | None:
    return first_existing(CANDIDATE_BCP_ROOTS)


def default_eval_384_manifest() -> Path | None:
    return first_existing(CANDIDATE_EVAL_384)


def default_train_pool() -> Path | None:
    return first_existing(CANDIDATE_TRAIN_POOLS)


def default_split_file() -> Path | None:
    return first_existing(CANDIDATE_SPLITS)


def default_bcplus_830_dir() -> Path | None:
    return first_existing(CANDIDATE_BCPLUS_830)


def default_sentence_train_states() -> Path | None:
    return first_existing(CANDIDATE_SENTENCE_STATES)


def _unique_ids(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        qid = str(value)
        if not qid or qid in seen:
            continue
        seen.add(qid)
        out.append(qid)
    return out


def _load_split_payload(split_file: Path | None = None) -> tuple[dict[str, Any], Path]:
    path = split_file or default_split_file()
    if path is None:
        raise FileNotFoundError("BrowseComp+ split file not found")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if path.name == "SPLIT.json":
        parent = path.parent
        train_path = parent / str(payload.get("train_ids_file") or "train_query_ids.json")
        test_path = parent / str(payload.get("test_ids_file") or "test_query_ids.json")
        payload = dict(payload)
        if train_path.is_file():
            payload["train_query_ids"] = json.loads(train_path.read_text(encoding="utf-8"))
        if test_path.is_file():
            payload["test_query_ids"] = json.loads(test_path.read_text(encoding="utf-8"))
    return payload, path


def official_train_ids(split_file: Path | None = None) -> list[str]:
    payload, _path = _load_split_payload(split_file)
    ids = payload.get("train_query_ids") or []
    if not ids:
        ids = list(payload.get("sft_query_ids") or []) + list(payload.get("rl_query_ids") or [])
    return _unique_ids(ids)


def official_test_id_list(split_file: Path | None = None) -> list[str]:
    payload, _path = _load_split_payload(split_file)
    return _unique_ids(payload.get("test_query_ids") or [])


def official_test_ids(split_file: Path | None = None) -> set[str]:
    return set(official_test_id_list(split_file))


def tag_official_split(rows: list[dict[str, Any]], *, split_file: Path | None = None) -> list[dict[str, Any]]:
    test_ids = official_test_ids(split_file)
    tagged = []
    for row in rows:
        rec = dict(row)
        rec["official_split"] = "test" if rec["query_id"] in test_ids else "train"
        tagged.append(rec)
    return tagged


def official_test_subset(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("official_split") == "test"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_queries_tsv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            qid, query = line.rstrip("\n").split("\t", 1)
            out[str(qid)] = query
    return out


def read_qrels(path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) >= 4 and float(parts[3]) > 0:
                out.setdefault(str(parts[0]), []).append(str(parts[2]))
    return {qid: sorted(set(ids)) for qid, ids in out.items()}


def _as_record(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, str):
        return {"query_id": raw, "query": raw}
    if not isinstance(raw, dict):
        return None
    qid = str(raw.get("query_id") or raw.get("id") or "")
    if not qid:
        return None
    query = str(raw.get("query") or raw.get("question") or raw.get("query_text") or qid)
    rec = dict(raw)
    rec["query_id"] = qid
    rec["query"] = query
    rec["evidence_docids"] = _as_id_list(
        raw.get("evidence_docids")
        or raw.get("evidence_document_ids")
        or raw.get("evidence")
        or []
    )
    rec["gold_docids"] = _as_id_list(
        raw.get("gold_docids")
        or raw.get("gold_document_ids")
        or raw.get("golds")
        or []
    )
    return rec


def _as_id_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                import ast

                parsed = ast.literal_eval(text)
                if isinstance(parsed, (list, tuple, set)):
                    return [str(x) for x in parsed if str(x)]
            except Exception:
                pass
        return [text] if text else []
    if isinstance(value, (list, tuple, set)):
        return [str(x) for x in value if str(x)]
    return [str(value)]


def load_query_manifest(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows = payload
        else:
            rows = payload.get("queries") or payload.get("records") or []
            if not rows and payload.get("query_ids"):
                rows = [{"query_id": str(q), "query": str(q)} for q in payload["query_ids"]]
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in rows:
        rec = _as_record(raw)
        if rec is None or rec["query_id"] in seen:
            continue
        seen.add(rec["query_id"])
        out.append(rec)
    return out


def attach_bcp_fields(rows: list[dict[str, Any]], *, bcp_root: Path | None = None) -> list[dict[str, Any]]:
    root = bcp_root or default_bcp_root()
    if root is None:
        return rows
    qpath = root / "topics-qrels" / "queries.tsv"
    epath = root / "topics-qrels" / "qrel_evidence.txt"
    gpath = root / "topics-qrels" / "qrel_golds.txt"
    queries = read_queries_tsv(qpath) if qpath.is_file() else {}
    evidence = read_qrels(epath) if epath.is_file() else {}
    golds = read_qrels(gpath) if gpath.is_file() else {}
    attached: list[dict[str, Any]] = []
    for row in rows:
        rec = dict(row)
        qid = rec["query_id"]
        if qid in queries and (rec.get("query") in {"", qid} or rec.get("query") == rec.get("query_id")):
            rec["query"] = queries[qid]
        elif rec.get("query") == rec.get("query_id") and qid in queries:
            rec["query"] = queries[qid]
        rec.setdefault("evidence_docids", evidence.get(qid, []))
        rec.setdefault("gold_docids", golds.get(qid, []))
        if not rec.get("evidence_docids"):
            rec["evidence_docids"] = evidence.get(qid, [])
        if not rec.get("gold_docids"):
            rec["gold_docids"] = golds.get(qid, [])
        attached.append(rec)
    return attached


def _records_for_ids(ids: list[str], *, bcp_root: Path | None, official_split: str) -> list[dict[str, Any]]:
    rows = [{"query_id": qid, "query": qid, "official_split": official_split} for qid in ids]
    attached = attach_bcp_fields(rows, bcp_root=bcp_root)
    for rec in attached:
        rec["official_split"] = official_split
    return attached


def load_bcplus_830_split(
    *,
    split_file: Path | None = None,
    bcp_root: Path | None = None,
    n_train: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Load the canonical BC+ 830 = 664 train + 166 test split."""
    payload, path = _load_split_payload(split_file)
    train_ids = official_train_ids(path)
    test_ids = official_test_id_list(path)
    if len(train_ids) != BCPLUS_TRAIN or len(test_ids) != BCPLUS_TEST:
        raise RuntimeError(
            f"BC+ split must be {BCPLUS_TOTAL}={BCPLUS_TRAIN}+{BCPLUS_TEST}, "
            f"got train={len(train_ids)} test={len(test_ids)} from {path}"
        )
    overlap = sorted(set(train_ids) & set(test_ids))
    if overlap:
        raise RuntimeError(f"BC+ train/test overlap: {overlap[:8]}")
    train_rows = _records_for_ids(train_ids, bcp_root=bcp_root, official_split="train")
    test_rows = _records_for_ids(test_ids, bcp_root=bcp_root, official_split="test")
    used_train = list(train_rows)
    if n_train not in {None, 0} and int(n_train) < len(used_train):
        used_train = used_train[: int(n_train)]
    meta = {
        "path": str(path),
        "pool_contract": "browsecomp_plus_830",
        "split": f"{BCPLUS_TOTAL} = {BCPLUS_TRAIN} train + {BCPLUS_TEST} test",
        "query_count_total": BCPLUS_TOTAL,
        "train_available": BCPLUS_TRAIN,
        "test_available": BCPLUS_TEST,
        "query_count": len(used_train),
        "official_test_count": len(test_rows),
        "official_test_expected": BCPLUS_TEST,
        "using_full_train_split": len(used_train) == BCPLUS_TRAIN,
        "score_split": SCORE_SPLIT_166,
        "sha256": sha256_file(path),
        "source_counts": {
            "total_queries": payload.get("total_queries", BCPLUS_TOTAL),
            "train_queries": payload.get("train_queries", BCPLUS_TRAIN),
            "test_queries": payload.get("test_queries", BCPLUS_TEST),
        },
    }
    return used_train, test_rows, meta


def load_bcplus_830_full(
    *,
    split_file: Path | None = None,
    bcp_root: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """BC+ train ∪ test (830). Primary eval split for scape+rl."""
    train_rows, test_rows, split_meta = load_bcplus_830_split(
        split_file=split_file, bcp_root=bcp_root, n_train=None
    )
    rows = list(train_rows) + list(test_rows)
    meta = dict(split_meta)
    meta.update(
        {
            "query_count": len(rows),
            "score_split": SCORE_SPLIT_830,
            "eval_count": len(rows),
            "primary_eval": SCORE_SPLIT_830,
        }
    )
    return rows, meta


def load_official_384(*, manifest: Path | None = None, bcp_root: Path | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = manifest or default_eval_384_manifest()
    if path is None:
        raise FileNotFoundError("official 384 query_manifest.json not found")
    rows = tag_official_split(attach_bcp_fields(load_query_manifest(path), bcp_root=bcp_root))
    test_rows = official_test_subset(rows)
    meta = {
        "path": str(path),
        "query_count": len(rows),
        "official_384": len(rows) == OFFICIAL_384_COUNT,
        "official_test_count": len(test_rows),
        "official_test_expected": 76,
        "split_file": str(default_split_file()) if default_split_file() else None,
        "sha256": sha256_file(path),
        "pool_contract": "browsecomp_plus_eval_pool_384",
        "score_split": "official_test_76",
    }
    return rows, meta


def load_train_queries(
    *,
    manifest: Path | None = None,
    bcp_root: Path | None = None,
    n_queries: int | None = None,
    exclude_eval_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = manifest or default_train_pool()
    if path is None:
        raise FileNotFoundError("train query pool not found")
    rows = attach_bcp_fields(load_query_manifest(path), bcp_root=bcp_root)
    blocked = exclude_eval_ids or set()
    rows = [r for r in rows if r["query_id"] not in blocked]
    if n_queries is not None:
        rows = rows[: int(n_queries)]
    meta = {
        "path": str(path),
        "query_count": len(rows),
        "excluded_eval_ids": len(blocked),
        "sha256": sha256_file(path),
    }
    return rows, meta


def overlap_ids(train: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> list[str]:
    left = {r["query_id"] for r in train}
    right = {r["query_id"] for r in eval_rows}
    return sorted(left & right)
