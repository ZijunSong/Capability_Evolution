"""SWE-bench adapter — repository read / modify / shell tasks.

Official source: https://github.com/SWE-bench/SWE-bench (package 5.0.2),
datasets frozen at ``data prepare``: ``princeton-nlp/SWE-bench`` (train/dev =
development pool), ``SWE-bench_Verified`` test (final panel),
``SWE-bench_Lite`` test (transfer panel).  Official metric: resolved rate.

Privacy boundary: gold ``patch`` and ``test_patch``/``FAIL_TO_PASS`` live only
in the evaluator side (``PrivateTaskRef``); the Actor sees the problem
statement, repo/base_commit and the workspace.  Training/dev instances are
family-deduplicated against the Verified/Lite test panels at data prepare.

Workspace: repo checked out at ``base_commit`` in an episode workdir (prepared
by ``data prepare``); snapshot/fork = workspace filesystem snapshot
(``supports_shadow_fork=True``).  Official scoring runs the SWE-bench
evaluation harness in per-instance docker images — the scorer refuses to run
when the pinned images/docker are unavailable (loud failure, no fallback).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from rsi.contracts.action import Action, Finish, ToolCall
from rsi.contracts.common import JSONValue, digest_of, digest_of_file
from rsi.contracts.errors import EnvironmentError_, NotImplementedCapability
from rsi.contracts.observation import (
    ArtifactRef,
    Observation,
    ObservationProvenance,
    ScoreRecord,
    SnapshotRef,
    StepResult,
    ToolSpec,
)
from rsi.contracts.task import PrivateTaskRef, PublicTask

SWEBENCH_VERSION = "swebench-5.0.2+data-pinned"

SWEBENCH_TOOL_SPEC: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="shell_exec",
        description="Run a shell command inside the repository working directory (5 min timeout).",
        parameters_schema={
            "type": "object",
            "properties": {"cmd": {"type": "string"}},
            "required": ["cmd"],
        },
        capabilities=("shell_exec", "repo_write"),
    ),
    ToolSpec(
        name="read_file",
        description="Read a UTF-8 text file from the repository working directory.",
        parameters_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        capabilities=("fs_read",),
    ),
    ToolSpec(
        name="write_file",
        description="Write UTF-8 text to a file in the repository working directory.",
        parameters_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
        capabilities=("fs_write",),
    ),
    ToolSpec(
        name="list_dir",
        description="List entries of a repository subdirectory.",
        parameters_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        capabilities=("fs_read",),
    ),
)


def _require_deps() -> Any:
    missing = []
    for mod in ("docker", "swebench"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        raise NotImplementedCapability(
            f"SWE-bench adapter requires optional packages {missing} "
            "(pip install -e '.[swebench]'). No fallback environment is used."
        )


def _safe_path(workdir: Path, rel: str) -> Path:
    p = (workdir / rel).resolve()
    try:
        p.relative_to(workdir.resolve())
    except ValueError as e:
        raise EnvironmentError_(f"path escapes workspace: {rel}") from e
    return p


class SWEBenchAdapter:
    """EnvironmentAdapter over a per-instance SWE-bench repo workspace."""

    name = "swebench"
    version = SWEBENCH_VERSION
    supports_shadow_fork = True

    def __init__(self, dataset_root: str | None = None, *, shell_timeout_s: int = 300):
        self._root = Path(dataset_root or os.environ.get("RSI_SWEBENCH_ROOT", "/mnt/songzijun/datasets/swebench"))
        self._shell_timeout_s = int(shell_timeout_s)
        self._workdir: Path | None = None
        self._task_id = ""
        self._snap_counter = 0
        self._started = False

    def check_ready(self) -> tuple[bool, str]:
        repos = self._root / "repos"
        if not repos.is_dir() or not any(repos.iterdir()):
            return False, "repo workspaces not prepared (run scripts/prepare_swebench_workspaces.py)"
        return True, "ok"

    # -- adapter protocol -------------------------------------------------
    def reset(self, public_task: PublicTask, seed: int) -> Observation:
        task_id = str(public_task.metadata.get("instance_id") or public_task.task_id)
        base_commit = str(public_task.metadata.get("base_commit", ""))
        repo_src = self._root / "repos" / task_id
        if not repo_src.is_dir():
            raise EnvironmentError_(
                f"workspace for {task_id} not prepared — run `rsi.cli data prepare` "
                "(git checkout at base_commit under datasets/swebench/repos/)"
            )
        self._task_id = task_id
        self._workdir = Path(tempfile.mkdtemp(prefix=f"rsi_swe_{task_id[:20]}_"))
        shutil.copytree(repo_src, self._workdir / "repo", symlinks=True)
        self._started = True
        return Observation(
            content=(
                f"Repository workspace ready at ./repo (base_commit={base_commit[:12]}). "
                f"Task: {public_task.instruction}"
            ),
            provenance=ObservationProvenance(source_tool="reset", source_ref=task_id),
        )

    def tool_catalog(self) -> tuple[ToolSpec, ...]:
        return SWEBENCH_TOOL_SPEC

    def step(self, action: Action) -> StepResult:
        self._require_started()
        if isinstance(action, Finish):
            return StepResult(
                observation=Observation(content="finished"),
                done=True,
                snapshot=self.snapshot(),
                cost={"tool_calls": 0},
            )
        assert isinstance(action, ToolCall)
        workdir = self._workdir / "repo"
        try:
            if action.name == "shell_exec":
                cmd = str(action.arguments.get("cmd", ""))
                proc = subprocess.run(
                    cmd, shell=True, cwd=str(workdir), capture_output=True, text=True,
                    timeout=self._shell_timeout_s,
                )
                out = (proc.stdout + ("\n[stderr]\n" + proc.stderr if proc.stderr else "")).strip()
                obs = Observation(
                    content=out[:20000] or "(no output)",
                    ok=proc.returncode == 0,
                    error_kind=None if proc.returncode == 0 else "tool_error",
                    provenance=ObservationProvenance(source_tool="shell_exec", source_ref=self._task_id),
                )
            elif action.name == "read_file":
                p = _safe_path(workdir, str(action.arguments.get("path", "")))
                obs = Observation(
                    content=p.read_text("utf-8")[:20000] if p.is_file() else f"no such file: {p.name}",
                    ok=p.is_file(),
                    error_kind=None if p.is_file() else "tool_error",
                    provenance=ObservationProvenance(source_tool="read_file", source_ref=str(p.name)),
                )
            elif action.name == "write_file":
                p = _safe_path(workdir, str(action.arguments.get("path", "")))
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(str(action.arguments.get("content", "")), "utf-8")
                obs = Observation(
                    content=f"wrote {p.name}",
                    provenance=ObservationProvenance(source_tool="write_file", source_ref=str(p.name)),
                )
            elif action.name == "list_dir":
                p = _safe_path(workdir, str(action.arguments.get("path", "")))
                names = sorted(x.name for x in p.iterdir())[:200] if p.is_dir() else []
                obs = Observation(
                    content=str(names),
                    provenance=ObservationProvenance(source_tool="list_dir", source_ref=str(p.name)),
                )
            else:
                obs = Observation(content=f"unknown tool {action.name!r}", ok=False, error_kind="model_invalid_action")
        except Exception as e:  # noqa: BLE001 - tool errors are observations
            obs = Observation(content=f"TOOL_ERROR: {e!r}", ok=False, error_kind="tool_error")
        return StepResult(observation=obs, done=False, snapshot=self.snapshot(), cost={"tool_calls": 1})

    def snapshot(self) -> SnapshotRef:
        self._require_started()
        self._snap_counter += 1
        tar_path = self._workdir / f"snap{self._snap_counter}.tar"
        with tarfile.open(tar_path, "w:gz") as tf:
            tf.add(self._workdir / "repo", arcname="repo")
        return SnapshotRef(ref_id=str(tar_path), digest=digest_of_file(str(tar_path)), kind="full")

    def fork(self, snapshot: SnapshotRef) -> "SWEBenchAdapter":
        child = SWEBenchAdapter(str(self._root), shell_timeout_s=self._shell_timeout_s)
        child._task_id = self._task_id
        child._workdir = Path(tempfile.mkdtemp(prefix="rsi_swe_fork_"))
        with tarfile.open(snapshot.ref_id, "r:gz") as tf:
            tf.extractall(child._workdir)
        child._started = True
        return child

    def state_digest(self) -> str:
        if not self._started:
            return digest_of({"unstarted": True})
        diff = self._git_diff() or ""
        return digest_of({"task": self._task_id, "n": self._snap_counter, "diff": digest_of(diff)})

    def _git_diff(self) -> str:
        """The actor's model_patch: tracked + untracked changes vs base_commit."""
        try:
            repo = str(self._workdir / "repo")
            # untracked files count as part of the patch (official scaffolds use
            # `git add -A && git diff HEAD`); plain `git diff` would drop them.
            subprocess.run(["git", "add", "-A"], cwd=repo, capture_output=True, timeout=120)
            proc = subprocess.run(
                ["git", "diff", "HEAD"], cwd=repo, capture_output=True, text=True, timeout=60
            )
            return proc.stdout
        except Exception:  # noqa: BLE001
            return ""

    def close(self) -> None:
        if self._workdir and self._workdir.exists():
            shutil.rmtree(self._workdir, ignore_errors=True)
        self._started = False

    def environment_identity(self) -> dict[str, JSONValue]:
        return {
            "name": self.name,
            "version": self.version,
            "dataset_root": str(self._root),
            "tool_schema_hash": digest_of([t.digest() for t in SWEBENCH_TOOL_SPEC]),
            "supports_shadow_fork": True,
            "synthetic": False,
        }

    def final_artifact(self) -> ArtifactRef:
        diff = self._git_diff() if self._started else ""
        path: str | None = None
        if self._started:
            # persist the patch: the scorer runs after close() removes the workdir
            f = tempfile.NamedTemporaryFile(
                "w", suffix=".diff", prefix=f"rsi_swe_{self._task_id[:40]}_", delete=False
            )
            f.write(diff)
            f.close()
            path = f.name
        return ArtifactRef(
            ref_id=f"swebench:{self._task_id}",
            kind="git_diff",
            path=path,
            digest=digest_of(diff),
        )

    def _require_started(self) -> None:
        if not self._started:
            raise EnvironmentError_("swebench adapter not reset")


class SWEBenchScorer:
    """Official SWE-bench evaluation (resolved rate) — docker-gated.

    Runs the pinned evaluation harness against the actor's git diff in the
    per-instance image.  When docker/images are unavailable this raises
    :class:`NotImplementedCapability` — it never substitutes a weaker check.
    """

    scorer_version = f"swebench-resolved-{SWEBENCH_VERSION}"

    def __init__(self, dataset_root: str | None = None):
        self._root = Path(dataset_root or os.environ.get("RSI_SWEBENCH_ROOT", "/mnt/songzijun/datasets/swebench"))
        self._private: dict[str, dict[str, Any]] = {}
        self._docker_checked = False

    def check_ready(self) -> tuple[bool, str]:
        """Capability probe used by panel-level domain skipping.

        Official scoring needs: docker daemon, the enriched spec table
        (image/eval_script/log_parser per instance), and the pre-fetched image
        tar cache (images are docker-load'ed on demand — Docker Hub is not
        reachable from the execution network).
        """
        try:
            self._require_docker()
        except NotImplementedCapability as e:
            return False, str(e)
        if not self._enriched_path().is_file():
            return False, f"enriched spec table missing: {self._enriched_path()}"
        tars = self._root / "images"
        if not tars.is_dir() or not any(tars.glob("*.tar")):
            return False, f"image tar cache missing: {tars} (run scripts/fetch_swebench_images_batch.py)"
        return True, "ok"

    def load_private(self, instance_id: str, record: dict[str, Any]) -> None:
        """Evaluator-side only: gold patch / test patch / test lists."""
        self._private[instance_id] = record

    def _require_docker(self) -> Any:
        try:
            import docker  # noqa: F401
            import swebench  # noqa: F401
        except ImportError as e:
            raise NotImplementedCapability(
                "official SWE-bench scoring requires docker+swebench packages"
            ) from e
        if not self._docker_checked:
            try:
                client = docker.from_env()
                client.ping()
            except Exception as e:  # noqa: BLE001
                raise NotImplementedCapability(
                    f"docker daemon unavailable for official SWE-bench scoring: {e!r}"
                ) from e
            self._docker_checked = True

    def _enriched_path(self) -> Path:
        return self._root / "enriched" / "swe_bench_test.parquet"

    def _enriched_row(self, instance_id: str) -> dict[str, Any]:
        """Official spec + gold fields for one instance (enriched HF dataset)."""
        cache = getattr(self, "_enr_cache", None)
        if cache is None:
            import pandas as pd

            df = pd.read_parquet(self._enriched_path())
            cache = {str(r["instance_id"]): r for r in df.to_dict("records")}
            self._enr_cache = cache
        if instance_id not in cache:
            raise NotImplementedCapability(
                f"no official eval spec for {instance_id} (outside test/Verified/Lite)"
            )
        return cache[instance_id]

    def _ensure_image(self, image_ref: str) -> None:
        """docker-load the pre-fetched tar if the daemon does not have the image."""
        import docker
        import subprocess

        client = docker.from_env()
        try:
            client.images.get(image_ref)
            return
        except Exception:  # noqa: BLE001
            pass
        folder = image_ref.split("/")[-1].split(":")[0]
        tar = self._root / "images" / f"{folder}.tar"
        if not tar.is_file():
            raise NotImplementedCapability(f"image tar missing for {image_ref}: {tar}")
        r = subprocess.run(["docker", "load", "-i", str(tar)], capture_output=True, text=True, timeout=1800)
        if r.returncode != 0:
            raise EnvironmentError_(f"docker load failed for {image_ref}: {(r.stderr or r.stdout)[-200:]}")

    def _prune_image(self, image_ref: str) -> None:
        if os.environ.get("RSI_SWEBENCH_PRUNE", "1") != "1":
            return
        try:
            import subprocess

            subprocess.run(["docker", "rmi", "-f", image_ref], capture_output=True, timeout=300)
        except Exception:  # noqa: BLE001
            pass

    def evaluate(self, final_artifact: ArtifactRef, private_task_ref: PrivateTaskRef) -> ScoreRecord:
        """Official SWE-bench resolved-rate scoring in the pinned per-instance image."""
        self._require_docker()
        task_id = private_task_ref.ref_id
        instance_id = task_id.split(":", 1)[-1]
        row = self._enriched_row(instance_id)
        model_patch = ""
        if final_artifact.path:
            model_patch = Path(final_artifact.path).read_text("utf-8")

        image_ref = str(row["image"])
        self._ensure_image(image_ref)
        try:
            from swebench.harness.constants import LOG_REPORT, RUN_EVALUATION_LOG_DIR
            from swebench.harness.run_evaluation import run_instances  # type: ignore

            pred = {
                instance_id: {
                    "instance_id": instance_id,
                    "model_name_or_path": "rsi_policy",
                    "model_patch": model_patch or "",
                }
            }
            run_id = os.environ.get("RSI_SWEBENCH_RUN_ID", "rsi_eval")
            timeout_s = int(os.environ.get("RSI_SWEBENCH_EVAL_TIMEOUT", "1800"))
            run_instances(
                predictions=pred,
                instances=[row],
                max_workers=1,
                run_id=run_id,
                timeout=timeout_s,
            )
            report_path = Path(RUN_EVALUATION_LOG_DIR) / run_id / "rsi_policy" / instance_id / LOG_REPORT
            if not report_path.is_file():
                raise EnvironmentError_(f"official harness wrote no report for {instance_id}")
            report = json.loads(report_path.read_text("utf-8"))[instance_id]
        finally:
            self._prune_image(image_ref)

        resolved = bool(report.get("resolved"))
        return ScoreRecord(
            task_id=task_id,
            domain="swebench",
            metric_name="resolved",
            value=1.0 if resolved else 0.0,
            success=resolved,
            details={
                "image": image_ref,
                "patch_exists": report.get("patch_exists"),
                "patch_successfully_applied": report.get("patch_successfully_applied"),
                "infra_failure": report.get("infra_failure", False),
            },
            scorer_version=self.scorer_version,
        )
