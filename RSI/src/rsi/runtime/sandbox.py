"""Candidate execution isolation.

AST scanning is auxiliary auditing only — Python has no in-process security
sandbox.  Real isolation for candidate workers is a separate process (default)
or a container (``execution.candidate_isolation: container``) that does not
mount the host repo, scoring private data, the Docker socket or cloud
credentials; basic tools are reached only through the agent-facing task
sandbox.  Immutable boundaries guarantee the experiment cannot "change the
task or the score"; they are not an artificial capability limit.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from rsi.contracts.errors import InfraError, ProtocolViolation


@dataclass(frozen=True)
class SandboxSpec:
    isolation: str = "subprocess"  # subprocess | container
    max_wall_s: float = 60.0
    forbid_env: tuple[str, ...] = ("DOCKER_HOST", "AWS_SECRET_ACCESS_KEY", "CLOUDSDK_CONFIG")


def run_candidate_worker(
    entry_module: str,
    payload: Mapping[str, Any],
    *,
    cwd: Path,
    spec: SandboxSpec | None = None,
) -> dict[str, Any]:
    """Run a candidate-side unit of work in an isolated worker process.

    For ``isolation=container`` the caller must provide an image runner; this
    function refuses to pretend a subprocess is a container
    (``test_unimplemented_fails`` spirit).
    """
    spec = spec or SandboxSpec()
    if spec.isolation == "container":
        raise ProtocolViolation(
            "container isolation requires a configured image runner; refusing to "
            "run a plain subprocess while claiming container isolation"
        )
    script = (
        "import json,sys;"
        f"import {entry_module} as m;"
        "payload=json.loads(sys.stdin.read());"
        "print(json.dumps(m.main(payload), sort_keys=True))"
    )
    import os

    env = {k: v for k, v in os.environ.items() if k not in spec.forbid_env}
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            input=json.dumps(dict(payload)),
            capture_output=True,
            text=True,
            timeout=spec.max_wall_s,
            cwd=str(cwd),
            env=env,
        )
    except subprocess.TimeoutExpired as e:
        raise InfraError(f"candidate worker timed out after {spec.max_wall_s}s") from e
    if proc.returncode != 0:
        raise InfraError(f"candidate worker failed rc={proc.returncode}: {proc.stderr[-2000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])
