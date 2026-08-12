"""Run manifest: provenance + status for every experiment run."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.common.spec import ExperimentSpec

_REPO = Path(__file__).resolve().parents[2]


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dir_fingerprint(path: Path, max_files: int = 64) -> str | None:
    if not path.exists() or not path.is_dir():
        return None
    files = sorted(p for p in path.rglob("*") if p.is_file())[:max_files]
    h = hashlib.sha256()
    for p in files:
        rel = str(p.relative_to(path))
        h.update(rel.encode())
        h.update(b"\0")
        digest = _sha256_file(p)
        if digest:
            h.update(digest.encode())
        h.update(b"\0")
    return h.hexdigest()


def _git_info(repo: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()
        except Exception:  # noqa: BLE001
            return ""

    dirty = run("status", "--porcelain")
    dirty_hash = hashlib.sha256(dirty.encode()).hexdigest() if dirty else ""
    return {
        "branch": run("branch", "--show-current"),
        "head": run("rev-parse", "HEAD"),
        "dirty": bool(dirty),
        "dirty_diff_hash": dirty_hash,
    }


def _pkg_version(name: str) -> str | None:
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", None)
    except Exception:  # noqa: BLE001
        return None


def build_run_manifest(
    spec: ExperimentSpec,
    *,
    command: list[str] | str,
    repo_root: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    repo = repo_root or _REPO
    env_info: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "torch": _pkg_version("torch"),
        "transformers": _pkg_version("transformers"),
        "vllm": _pkg_version("vllm"),
    }
    try:
        import torch

        env_info["cuda"] = getattr(torch.version, "cuda", None)
        env_info["cuda_available"] = bool(torch.cuda.is_available())
        env_info["gpu_count"] = int(torch.cuda.device_count())
        if torch.cuda.is_available():
            env_info["gpu_names"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
    except Exception as exc:  # noqa: BLE001
        env_info["torch_error"] = str(exc)

    input_hashes: dict[str, str | None] = {}
    for key in ("train_manifest", "valid_manifest", "test_manifest", "runtime_config"):
        val = getattr(spec, key, None)
        if val:
            p = Path(val)
            if not p.is_absolute():
                p = repo / p
            input_hashes[key] = _sha256_file(p) if p.is_file() else _dir_fingerprint(p)

    ckpt_fp = None
    if spec.checkpoint:
        cp = Path(spec.checkpoint)
        if not cp.is_absolute():
            cp = repo / cp
        ckpt_fp = _sha256_file(cp) if cp.is_file() else _dir_fingerprint(cp)

    manifest = {
        "schema_version": "iclr_run_manifest_v1",
        "experiment_spec": spec.to_dict(),
        "git": _git_info(repo),
        "environment": env_info,
        "input_sha256": input_hashes,
        "checkpoint_fingerprint": ckpt_fp,
        "command": command if isinstance(command, str) else list(command),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "ended_at": None,
        "exit_code": None,
        "status": "running",
        "completed_shards": [],
        "error_summary": None,
        "output_files": [],
        "hostname": platform.node(),
        "pid": os.getpid(),
        "extra": extra or {},
    }
    return manifest


def finalize_run_manifest(
    manifest: dict[str, Any],
    *,
    exit_code: int,
    output_dir: Path,
    error_summary: str | None = None,
    completed_shards: list[str] | None = None,
) -> dict[str, Any]:
    out = dict(manifest)
    out["ended_at"] = datetime.now(timezone.utc).isoformat()
    out["exit_code"] = exit_code
    out["status"] = "completed" if exit_code == 0 else "failed"
    out["error_summary"] = error_summary
    if completed_shards is not None:
        out["completed_shards"] = completed_shards
    files = []
    if output_dir.exists():
        for p in sorted(output_dir.rglob("*")):
            if p.is_file() and p.name != "run_manifest.json":
                files.append(str(p.relative_to(output_dir)))
    out["output_files"] = files
    return out


def write_run_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_run_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
