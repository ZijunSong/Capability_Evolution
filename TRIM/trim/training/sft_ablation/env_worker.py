"""Full or Zero Harness worker. Harness is imported only after the mask env is set."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

PROTECTED_PREFIXES = ("V8D_", "ABLATE_")


def launch_env_worker(job_path: Path, *, mask: Mapping[str, bool], extra_env: Mapping[str, str] | None = None) -> None:
    from trim.training.dist_runtime import isolate_inference_child_env
    from trim.upstream_harness1.v8d_flags import subprocess_env_for_mask

    extra = {str(key): str(value) for key, value in dict(extra_env or {}).items()}
    blocked = [key for key in extra if key.startswith(PROTECTED_PREFIXES)]
    if blocked:
        raise ValueError(f"extra_env cannot override role flags: {blocked}")
    env = subprocess_env_for_mask(mask, harness="Harness-1", extra=extra)
    env = isolate_inference_child_env(env)
    subprocess.run(
        [sys.executable, "-m", "trim.training.sft_ablation.env_worker", "--job", str(job_path)],
        env=env,
        check=True,
    )


def _tool_names(mask: Mapping[str, bool]) -> list[str]:
    from trim.training.tool_mask import legal_tool_names

    return legal_tool_names(harness_mask=mask)


def audit_loaded_mask(mask: Mapping[str, bool]) -> dict[str, Any]:
    """Import Harness in this process and compare frozen flags with the requested mask."""
    from trim.upstream_harness1.pin import ensure_harness1_on_path
    from trim.upstream_harness1.v8d_flags import V8D_COMPONENT_FLAGS

    ensure_harness1_on_path()
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
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        Path(job["result"]).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return 1
    Path(job["result"]).parent.mkdir(parents=True, exist_ok=True)
    Path(job["result"]).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if result.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
