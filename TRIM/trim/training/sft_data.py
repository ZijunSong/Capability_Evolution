"""Harness-1 public SFT trajectories: download, unwrap, pack, materialize.

Official source: ``pat-jj/harness-1-train-data`` ``stage=sft`` (899 GPT-5.4
v8d trajectories). ``training/train_sft.py`` expects a directory of
``ultra_v3_{dataset}_{query_id}.json`` files, so this module unwraps the
HuggingFace ``payload_json`` rows into that layout and optionally packs them
as ``harness-1-sft-data.tar.gz`` (same pattern as the SEC RL pack).
"""

from __future__ import annotations

import io
import json
import os
import re
import tarfile
from pathlib import Path
from typing import Any, Sequence

HF_SFT_REPO = "pat-jj/harness-1-train-data"
HF_SFT_SPLIT = "train"
SFT_STAGE = "sft"
SFT_PACK_NAME = "harness-1-sft-data"
EXPECTED_N_TRAJECTORIES = 899

DEFAULT_SFT_PACK = Path(
    os.environ.get("TRIM_SFT_DATA", "/data/ppnm/harness-1-sft-data.tar.gz")
)
DEFAULT_SFT_EXTRACTED = Path("/data/ppnm/harness-1-sft-data")
_SAFE_TOKEN = re.compile(r"[^A-Za-z0-9._-]+")

_LOCAL_JSONL_CANDIDATES = (
    Path("/data/ppnm/Capability_Evolution/SCAPE/outputs/0814_clean_mechanism/data/hf_raw/sft_trajectories.jsonl"),
)

TRAJECTORY_REQUIRED_KEYS = ("query_text", "turn_history")


def default_sft_pack() -> Path:
    env = os.environ.get("TRIM_SFT_DATA")
    if env:
        return Path(env)
    tar = Path("/data/ppnm/harness-1-sft-data.tar.gz")
    extracted = DEFAULT_SFT_EXTRACTED
    if tar.is_file():
        return tar
    if extracted.exists():
        return extracted
    return tar


def _parse_jsonish(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8")
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{") or text.startswith("["):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return value
    return value


def unwrap_sft_record(rec: Any) -> dict[str, Any] | None:
    """HF rows nest the official ultra_v3 trajectory in ``payload_json``."""
    if not isinstance(rec, dict):
        return None
    if str(rec.get("stage") or "").strip().lower() == "rl":
        return None
    payload = _parse_jsonish(rec.get("payload_json"))
    if isinstance(payload, dict):
        merged = {**rec, **payload}
        merged.pop("payload_json", None)
    else:
        merged = dict(rec)
        for key in ("trajectory", "content", "data", "record"):
            val = _parse_jsonish(merged.get(key))
            if isinstance(val, dict) and "turn_history" in val:
                merged = {**merged, **val}
                break
    stage = str(merged.get("stage") or rec.get("stage") or SFT_STAGE).strip().lower()
    if stage == "rl":
        return None
    if not merged.get("query_text"):
        merged["query_text"] = rec.get("query") or merged.get("query") or ""
    if not merged.get("dataset_name"):
        merged["dataset_name"] = rec.get("dataset_name") or merged.get("dataset") or "unknown"
    if not merged.get("query_id"):
        merged["query_id"] = rec.get("query_id") or merged.get("qid") or ""
    merged["stage"] = SFT_STAGE
    if not merged.get("query_text") or not merged.get("turn_history"):
        return None
    return merged


def trajectory_filename(traj: dict[str, Any], *, index: int | None = None) -> str:
    ds = _SAFE_TOKEN.sub("_", str(traj.get("dataset_name") or "unknown")).strip("._") or "unknown"
    qid = _SAFE_TOKEN.sub("_", str(traj.get("query_id") or "")).strip("._") or f"idx{index or 0}"
    name = f"ultra_v3_{ds}_{qid}.json"
    if index is not None and name == f"ultra_v3_{ds}_idx{index}.json":
        return name
    return name


def _read_jsonl_bytes(data: bytes) -> list[Any]:
    text = data.decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _load_jsonl_path(path: Path) -> list[Any]:
    rows: list[Any] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_sft_records_from_hf(
    *,
    repo: str = HF_SFT_REPO,
    split: str = HF_SFT_SPLIT,
) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(repo, split=split)
    rows: list[dict[str, Any]] = []
    for rec in ds:
        if str(rec.get("stage") or "").strip().lower() != SFT_STAGE:
            continue
        rows.append(dict(rec))
    return rows


def load_sft_trajectories(
    source: Path | None = None,
    *,
    n_trajectories: int | None = None,
    allow_hf: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load unwrapped ultra_v3 trajectories from tar / dir / jsonl / HuggingFace."""
    origin = Path(source) if source is not None else default_sft_pack()
    raw: list[Any] = []
    loc = str(origin)
    if origin.is_file() and origin.name.endswith(".tar.gz"):
        raw, loc = _load_rows_from_tar(origin)
    elif origin.is_dir():
        raw, loc = _load_rows_from_dir(origin)
    elif origin.is_file() and origin.suffix in {".jsonl", ".json"}:
        raw = _load_jsonl_path(origin) if origin.suffix == ".jsonl" else [json.loads(origin.read_text(encoding="utf-8"))]
        loc = str(origin)
    else:
        dumped = next((p for p in _LOCAL_JSONL_CANDIDATES if p.is_file()), None)
        if dumped is not None:
            raw = _load_jsonl_path(dumped)
            loc = str(dumped)
        elif allow_hf:
            raw = load_sft_records_from_hf()
            loc = f"hf://datasets/{HF_SFT_REPO}"
        else:
            raise FileNotFoundError(
                f"Harness-1 SFT data not found at {origin}. "
                "Pass --sft-data, place harness-1-sft-data.tar.gz, or download "
                f"{HF_SFT_REPO} (stage=sft)."
            )

    trajs: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    skipped_rl = 0
    for rec in raw:
        if isinstance(rec, dict) and str(rec.get("stage") or "").lower() == "rl":
            skipped_rl += 1
            continue
        traj = unwrap_sft_record(rec)
        if traj is None:
            continue
        key = (str(traj.get("dataset_name") or ""), str(traj.get("query_id") or ""))
        if key in seen:
            continue
        seen.add(key)
        trajs.append(traj)
    if not trajs:
        raise RuntimeError(f"SFT source {loc} produced zero stage=sft trajectories")
    used = list(trajs)
    if n_trajectories not in {None, 0} and int(n_trajectories) < len(used):
        used = used[: int(n_trajectories)]
    meta = {
        "path": loc,
        "source": str(origin),
        "repo": HF_SFT_REPO,
        "stage": SFT_STAGE,
        "n_trajectories": len(used),
        "n_trajectories_available": len(trajs),
        "n_rl_skipped": skipped_rl,
        "using_full_sft_split": n_trajectories in {None, 0} or int(n_trajectories) >= len(trajs),
        "expected_n": EXPECTED_N_TRAJECTORIES,
        "datasets": _dataset_counts(used),
    }
    return used, meta


def _dataset_counts(trajs: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for traj in trajs:
        name = str(traj.get("dataset_name") or "unknown")
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def _iter_dir_json_files(root: Path) -> list[Path]:
    traj_dir = root / "trajectories"
    search_roots = [traj_dir] if traj_dir.is_dir() else [root]
    found: list[Path] = []
    for base in search_roots:
        found.extend(sorted(p for p in base.glob("*.json") if p.name != "MANIFEST.json"))
        found.extend(sorted(base.glob("ultra_v3_*.json")))
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in found:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    jsonl = root / "sft_trajectories.jsonl"
    if jsonl.is_file() and not unique:
        return [jsonl]
    return unique


def _load_rows_from_dir(root: Path) -> tuple[list[Any], str]:
    files = _iter_dir_json_files(root)
    if not files:
        raise FileNotFoundError(f"no SFT trajectory JSON under {root}")
    if len(files) == 1 and files[0].suffix == ".jsonl":
        return _load_jsonl_path(files[0]), str(files[0])
    rows: list[Any] = []
    for path in files:
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return rows, str(root)


def _load_rows_from_tar(archive: Path) -> tuple[list[Any], str]:
    with tarfile.open(archive, "r:gz") as tf:
        members = [m for m in tf.getmembers() if m.isfile()]
        jsonl_members = [m for m in members if Path(m.name).name == "sft_trajectories.jsonl"]
        if jsonl_members:
            handle = tf.extractfile(jsonl_members[0])
            if handle is None:
                raise FileNotFoundError(f"{archive} jsonl member is empty")
            return _read_jsonl_bytes(handle.read()), f"{archive}::{jsonl_members[0].name}"
        rows: list[Any] = []
        loc = str(archive)
        for member in members:
            name = Path(member.name).name
            if not name.endswith(".json") or name == "MANIFEST.json":
                continue
            handle = tf.extractfile(member)
            if handle is None:
                continue
            try:
                rows.append(json.loads(handle.read().decode("utf-8")))
            except json.JSONDecodeError:
                continue
        if not rows:
            raise FileNotFoundError(f"{archive} has no SFT trajectory JSON files")
        return rows, loc


def write_trajectory_dir(
    trajs: Sequence[dict[str, Any]],
    dest: Path,
    *,
    manifest: dict[str, Any] | None = None,
) -> Path:
    """Write per-trajectory JSON files for ``training/train_sft.py --data-dir``."""
    traj_dir = dest / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    used_names: set[str] = set()
    written = 0
    for i, traj in enumerate(trajs):
        name = trajectory_filename(traj, index=i)
        if name in used_names:
            stem = Path(name).stem
            name = f"{stem}_{i}.json"
        used_names.add(name)
        (traj_dir / name).write_text(
            json.dumps(traj, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        written += 1
    payload = {
        "format": "harness-1 ultra_v3 per-trajectory JSON",
        "source": HF_SFT_REPO,
        "stage": SFT_STAGE,
        "n_trajectories": written,
        "train_sft_data_dir": "trajectories",
        "expected_n": EXPECTED_N_TRAJECTORIES,
        "datasets": _dataset_counts(trajs),
    }
    if manifest:
        payload.update(manifest)
    (dest / "MANIFEST.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return traj_dir


def pack_sft_tar(
    dest_tar: Path,
    trajs: Sequence[dict[str, Any]],
    *,
    arcname: str = SFT_PACK_NAME,
    manifest: dict[str, Any] | None = None,
) -> Path:
    """Write ``harness-1-sft-data.tar.gz`` with trajectory JSON + MANIFEST."""
    dest_tar = Path(dest_tar)
    dest_tar.parent.mkdir(parents=True, exist_ok=True)
    manifest_payload = {
        "format": "harness-1 ultra_v3 per-trajectory JSON",
        "source": HF_SFT_REPO,
        "stage": SFT_STAGE,
        "n_trajectories": len(trajs),
        "train_sft_data_dir": "trajectories",
        "expected_n": EXPECTED_N_TRAJECTORIES,
        "datasets": _dataset_counts(trajs),
    }
    if manifest:
        manifest_payload.update(manifest)
    used_names: set[str] = set()
    with tarfile.open(dest_tar, "w:gz") as tf:
        manifest_bytes = (json.dumps(manifest_payload, indent=2) + "\n").encode("utf-8")
        _add_tar_bytes(tf, f"{arcname}/MANIFEST.json", manifest_bytes)
        for i, traj in enumerate(trajs):
            name = trajectory_filename(traj, index=i)
            if name in used_names:
                name = f"{Path(name).stem}_{i}.json"
            used_names.add(name)
            data = (json.dumps(traj, ensure_ascii=False, default=str) + "\n").encode("utf-8")
            _add_tar_bytes(tf, f"{arcname}/trajectories/{name}", data)
    return dest_tar


def _add_tar_bytes(tf: tarfile.TarFile, arcname: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=arcname)
    info.size = len(data)
    tf.addfile(info, io.BytesIO(data))


def materialize_sft_data_dir(
    source: Path | None = None,
    *,
    dest: Path | None = None,
    n_trajectories: int | None = None,
    write_pack: Path | bool | None = None,
    allow_hf: bool = True,
) -> tuple[Path, dict[str, Any]]:
    """Return a directory of ``*.json`` trajectories ready for ``train_sft.py``.

    If ``source`` is already an extracted trajectory directory, it is reused.
    Otherwise trajectories are loaded, written under ``dest``, and optionally
    packed to ``harness-1-sft-data.tar.gz``.
    """
    origin = Path(source) if source is not None else default_sft_pack()
    ready = _existing_trajectory_dir(origin)
    pack_path: Path | None = None
    if write_pack is True:
        pack_path = Path("/data/ppnm/harness-1-sft-data.tar.gz")
    elif isinstance(write_pack, Path):
        pack_path = write_pack
    if ready is not None and n_trajectories in {None, 0}:
        files = sorted(p for p in ready.glob("*.json") if p.name != "MANIFEST.json")
        packed = None
        if pack_path is not None and not pack_path.is_file():
            trajs = [json.loads(p.read_text(encoding="utf-8")) for p in files]
            packed = str(pack_sft_tar(pack_path, trajs, manifest={"loaded_from": str(ready)}))
        meta = {
            "path": str(ready),
            "source": str(origin),
            "n_trajectories": len(files),
            "materialized": False,
            "pack": packed,
            "repo": HF_SFT_REPO,
            "stage": SFT_STAGE,
        }
        return ready, meta

    trajs, meta = load_sft_trajectories(origin if origin.exists() else None, n_trajectories=n_trajectories, allow_hf=allow_hf)
    out_root = Path(dest) if dest is not None else DEFAULT_SFT_EXTRACTED
    traj_dir = write_trajectory_dir(trajs, out_root, manifest={"loaded_from": meta.get("path")})
    packed = None
    if pack_path is not None:
        packed = str(pack_sft_tar(pack_path, trajs, manifest={"loaded_from": meta.get("path")}))
    meta = {
        **meta,
        "path": str(traj_dir),
        "extracted_root": str(out_root),
        "pack": packed,
        "materialized": True,
    }
    return traj_dir, meta


def _existing_trajectory_dir(origin: Path) -> Path | None:
    if origin.is_dir():
        traj_dir = origin / "trajectories"
        if traj_dir.is_dir() and any(traj_dir.glob("ultra_v3_*.json")):
            return traj_dir
        if any(origin.glob("ultra_v3_*.json")):
            return origin
        return None
    if origin.is_file() and origin.name.endswith(".tar.gz"):
        extracted = Path(str(origin)[: -len(".tar.gz")])
        if extracted.is_dir():
            return _existing_trajectory_dir(extracted)
        sibling = DEFAULT_SFT_EXTRACTED
        if sibling.is_dir() and origin == Path("/data/ppnm/harness-1-sft-data.tar.gz"):
            return _existing_trajectory_dir(sibling)
    return None


def assert_train_sft_ready(data_dir: Path) -> int:
    files = sorted(p for p in Path(data_dir).glob("*.json") if p.name != "MANIFEST.json")
    if not files:
        raise FileNotFoundError(f"{data_dir} has no trajectory JSON files")
    sample = json.loads(files[0].read_text(encoding="utf-8"))
    missing = [k for k in TRAJECTORY_REQUIRED_KEYS if k not in sample]
    if missing:
        raise ValueError(f"{files[0].name} missing {missing}; not a Harness-1 SFT trajectory")
    if not sample.get("turn_history"):
        raise ValueError(f"{files[0].name} has empty turn_history")
    return len(files)
