"""Full or Zero Harness worker. Harness is imported only after the mask env is set."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

PROTECTED_PREFIXES = ("V8D_", "ABLATE_")


def _worker_command(job_path: Path, mask: Mapping[str, bool], extra_env: Mapping[str, str] | None) -> tuple[list[str], dict[str, str]]:
    from trim.training.dist_runtime import isolate_inference_child_env
    from trim.upstream_harness1.v8d_flags import subprocess_env_for_mask

    extra = {str(key): str(value) for key, value in dict(extra_env or {}).items()}
    blocked = [key for key in extra if key.startswith(PROTECTED_PREFIXES)]
    if blocked:
        raise ValueError(f"extra_env cannot override role flags: {blocked}")
    env = subprocess_env_for_mask(mask, harness="Harness-1", extra=extra)
    env = isolate_inference_child_env(env)
    return [sys.executable, "-m", "trim.training.sft_ablation.env_worker", "--job", str(job_path)], env


def start_env_worker(job_path: Path, *, mask: Mapping[str, bool], extra_env: Mapping[str, str] | None = None) -> subprocess.Popen:
    """Start one replica. The caller waits so several replicas can run together."""
    command, env = _worker_command(job_path, mask, extra_env)
    return subprocess.Popen(command, env=env, start_new_session=True)


def stop_env_worker(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        proc.wait(timeout=5)


def launch_env_worker(job_path: Path, *, mask: Mapping[str, bool], extra_env: Mapping[str, str] | None = None) -> None:
    proc = start_env_worker(job_path, mask=mask, extra_env=extra_env)
    code = proc.wait()
    if code != 0:
        raise subprocess.CalledProcessError(code, [sys.executable, "-m", "trim.training.sft_ablation.env_worker"])


def _tool_names(mask: Mapping[str, bool]) -> list[str]:
    from trim.training.tool_mask import legal_tool_names

    return legal_tool_names(harness_mask=mask)


def audit_loaded_mask(mask: Mapping[str, bool]) -> dict[str, Any]:
    """Import Harness in this process and compare frozen flags with the requested mask."""
    from trim.upstream_harness1.pin import ensure_harness1_on_path, HARNESS1_ROOT
    from trim.upstream_harness1.v8d_flags import V8D_COMPONENT_FLAGS

    ensure_harness1_on_path()
    # r11 ROOT-CAUSE FIX: `import training.train_rl` resolves `training` to
    # TRIM/trim/training/ (has __init__.py = real package) instead of
    # external/harness-1/training/ (no __init__.py = namespace pkg, where
    # train_rl.py lives). A real package ALWAYS shadows a namespace package
    # regardless of sys.path order. Fix: (1) create an empty __init__.py in
    # harness-1/training so it becomes a real package too, (2) insert harness-1
    # at sys.path[0] so it is found first, (3) drop any stale `training` from
    # sys.modules + invalidate caches. Then `import training.train_rl` finds
    # harness-1/training/train_rl.py. This is the ONLY layer that can fix the
    # r4-r11 ModuleNotFoundError chain (PYTHONPATH/.pth/insert(0) all failed
    # because real-pkg-shadows-namespace is a Python import rule, not a path issue).
    import importlib
    train_pkg = Path(HARNESS1_ROOT) / "training"
    train_init = train_pkg / "__init__.py"
    if not train_init.is_file():
        train_init.write_text("", encoding="utf-8")
    sys.path.insert(0, str(HARNESS1_ROOT))
    sys.modules.pop("training", None)
    importlib.invalidate_caches()
    import harness.ultra_core as ultra_core
    import training.train_rl as train_rl

    requested = {}
    ultra = {}
    train = {}
    for cid, flag in V8D_COMPONENT_FLAGS.items():
        requested[flag] = bool(mask.get(cid, False))
        if not hasattr(ultra_core, flag):
            raise RuntimeError(f"ultra_core missing {flag}")
        ultra[flag] = bool(getattr(ultra_core, flag))
        if ultra[flag] != requested[flag]:
            raise RuntimeError(f"mask mismatch {flag}: requested={requested[flag]} ultra={ultra[flag]}")
        if hasattr(train_rl, flag):
            train[flag] = bool(getattr(train_rl, flag))
            if train[flag] != requested[flag]:
                raise RuntimeError(f"mask mismatch {flag}: requested={requested[flag]} train_rl={train[flag]}")
        else:
            train[flag] = None
    names = _tool_names(mask)
    if bool(mask.get("verify_tool")) and "verify" not in names:
        raise RuntimeError("full mask missing verify")
    if not bool(mask.get("verify_tool")) and "verify" in names:
        raise RuntimeError("zero mask still exposes verify")
    return {
        "ok": True,
        "requested": requested,
        "ultra_core": ultra,
        "train_rl": train,
        "tool_names": names,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--job", required=True)
    args = parser.parse_args(argv)
    job = json.loads(Path(args.job).read_text(encoding="utf-8"))
    mask = dict(job["mask"])
    task = str(job.get("task") or "audit")
    try:
        audit = audit_loaded_mask(mask)
        if task == "audit":
            result = audit
        elif task == "collect":
            from trim.training.sft_ablation.teacher_archive import run_teacher_collection_job

            result = run_teacher_collection_job(job)
        elif task == "replay":
            from trim.training.projected_teacher_sft.runtime import run_replay_job

            result = run_replay_job(job)
        elif task == "rollout":
            from trim.training.student_onpolicy_sft.runtime import run_rollout_job

            result = run_rollout_job(job)
        else:
            raise ValueError(f"unknown env worker task: {task}")
    except Exception as exc:
        result = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "diag_sys_executable": sys.executable,
            "diag_sys_path": list(sys.path),
            "diag_pythonpath": os.environ.get("PYTHONPATH", ""),
            "diag_pythonnousersite": os.environ.get("PYTHONNOUSERSITE", ""),
            "diag_pythonpath_set": "PYTHONPATH" in os.environ,
            "diag_harness1_pth_exists": any("harness-1" in p for p in sys.path),
            "diag_site_packages_writable": os.path.isdir("/usr/local/lib/python3.12/dist-packages"),
            "diag_pth_file_exists": os.path.isfile("/usr/local/lib/python3.12/dist-packages/zzz_harness1.pth"),
        }
        Path(job["result"]).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return 1
    Path(job["result"]).parent.mkdir(parents=True, exist_ok=True)
    Path(job["result"]).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if result.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
