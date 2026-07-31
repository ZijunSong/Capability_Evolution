"""Bare on-policy rollout: tau ~ pi_theta(x), no Harness."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from training.opd._policy_backend import RolloutBackend
from training.opd.rollout_worker import QueryRecord


@dataclass
class BareTrajectory:
    """One bare sample tau_i ~ pi_theta(x_i)."""

    query_id: str
    query: str
    prompt_token_ids: list[int]
    response_token_ids: list[int]
    response_text: str
    mode: str = "bare"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def bare_messages(query: str) -> list[dict[str, str]]:
    """Minimal prompt: query only, no Harness scaffolding."""
    return [{"role": "user", "content": query}]


def load_completed_query_ids(jsonl_path: Path) -> set[str]:
    if not jsonl_path.exists():
        return set()
    done: set[str] = set()
    with jsonl_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(str(json.loads(line)["query_id"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def _build_trajectory(
    record: QueryRecord,
    rollout: RolloutBackend,
    *,
    max_new_tokens: int,
    temperature: float,
) -> BareTrajectory | None:
    result = rollout.rollout_chat(
        bare_messages(record.query),
        {"max_new_tokens": max_new_tokens, "temperature": temperature},
    )
    if not result.action_token_ids:
        return None
    return BareTrajectory(
        query_id=record.query_id,
        query=record.query,
        prompt_token_ids=result.prompt_token_ids,
        response_token_ids=result.action_token_ids,
        response_text=result.text,
        metadata={
            "rollout_backend": result.metadata.get("backend", "unknown"),
            **result.metadata,
        },
    )


async def run_bare_rollout_async(
    rollout: RolloutBackend,
    records: list[QueryRecord],
    *,
    max_new_tokens: int = 2048,
    temperature: float = 1.0,
    output_jsonl: Path | None = None,
    resume: bool = True,
    log_every: int = 10,
    parallel: int = 8,
) -> list[BareTrajectory]:
    """Concurrent bare rollout via asyncio + thread pool for sync HTTP clients."""
    parallel = max(1, int(parallel))
    trajectories: list[BareTrajectory] = []
    done_ids: set[str] = set()
    if output_jsonl is not None and resume:
        done_ids = load_completed_query_ids(output_jsonl)
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    pending = [r for r in records if r.query_id not in done_ids]
    if not pending:
        return trajectories

    fh = None
    if output_jsonl is not None:
        fh = output_jsonl.open("a", encoding="utf-8")

    sem = asyncio.Semaphore(parallel)
    write_lock = asyncio.Lock()
    completed = 0
    total = len(pending)

    try:

        async def _one(record: QueryRecord) -> BareTrajectory | None:
            nonlocal completed
            async with sem:
                traj = await asyncio.to_thread(
                    _build_trajectory,
                    record,
                    rollout,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
            if traj is None:
                return None
            async with write_lock:
                trajectories.append(traj)
                if fh is not None:
                    fh.write(json.dumps(traj.to_dict(), ensure_ascii=False) + "\n")
                    fh.flush()
                completed += 1
                if log_every > 0 and completed % log_every == 0:
                    print(f"[bare] progress {completed}/{total} (parallel={parallel})", flush=True)
            return traj

        await asyncio.gather(*[_one(r) for r in pending])
    finally:
        if fh is not None:
            fh.close()
    return trajectories


def run_bare_rollout(
    rollout: RolloutBackend,
    records: list[QueryRecord],
    *,
    max_new_tokens: int = 2048,
    temperature: float = 1.0,
    output_jsonl: Path | None = None,
    resume: bool = True,
    log_every: int = 10,
    parallel: int = 8,
) -> list[BareTrajectory]:
    """Bare on-policy rollout; ``parallel`` concurrent in-flight vLLM requests."""
    return asyncio.run(
        run_bare_rollout_async(
            rollout,
            records,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            output_jsonl=output_jsonl,
            resume=resume,
            log_every=log_every,
            parallel=parallel,
        )
    )


def save_bare_trajectories(
    trajectories: Iterable[BareTrajectory],
    output_dir: Path,
    *,
    manifest: dict[str, Any] | None = None,
    jsonl_name: str = "bare_rollouts.jsonl",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / jsonl_name
    if not jsonl_path.exists():
        with jsonl_path.open("w", encoding="utf-8") as fh:
            for traj in trajectories:
                fh.write(json.dumps(traj.to_dict(), ensure_ascii=False) + "\n")
    total = sum(1 for _ in open(jsonl_path, encoding="utf-8"))
    payload = {
        "mode": "bare",
        "n_trajectories": total,
        "output": str(jsonl_path),
        **(manifest or {}),
    }
    (output_dir / "bare_rollout_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return jsonl_path
