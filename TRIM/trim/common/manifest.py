"""RUN_MANIFEST.json helpers (method-agnostic provenance)."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from trim.common.hashing import sha256_file


def _git_info(repo: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()
        except Exception:  # noqa: BLE001
            return ""

    dirty = run("status", "--porcelain")
    return {
        "branch": run("branch", "--show-current"),
        "head": run("rev-parse", "HEAD"),
        "dirty": bool(dirty),
    }


def _pkg_version(name: str) -> str | None:
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", None)
    except Exception:  # noqa: BLE001
        return None


def build_run_manifest(
    *,
    run_id: str,
    stage: str,
    command: list[str] | str,
    repo_root: Path,
    output_dir: Path,
    extra: Mapping[str, Any] | None = None,
    input_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
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

    input_sha = {}
    for k, v in (input_paths or {}).items():
        p = Path(v)
        input_sha[k] = sha256_file(p) if p.is_file() else None

    return {
        "schema_version": "scape_run_manifest_v1",
        "run_id": run_id,
        "stage": stage,
        "repo_root": str(repo_root.resolve()),
        "output_dir": str(output_dir.resolve()),
        "git": _git_info(repo_root),
        "environment": env_info,
        "input_sha256": input_sha,
        "command": command if isinstance(command, str) else list(command),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "ended_at": None,
        "exit_code": None,
        "status": "running",
        "completed_shards": [],
        "error_summary": None,
        "hostname": platform.node(),
        "pid": os.getpid(),
        "extra": dict(extra or {}),
    }


def finalize_run_manifest(
    manifest: dict[str, Any],
    *,
    exit_code: int,
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
    return out


def write_run_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(manifest), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_run_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
