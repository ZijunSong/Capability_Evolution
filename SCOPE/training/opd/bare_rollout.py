"""Bare on-policy rollout: tau ~ pi_theta(x), no Harness."""

from __future__ import annotations

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


def run_bare_rollout(
    rollout: RolloutBackend,
    records: list[QueryRecord],
    *,
    max_new_tokens: int = 2048,
    temperature: float = 1.0,
    output_jsonl: Path | None = None,
    resume: bool = True,
    log_every: int = 10,
) -> list[BareTrajectory]:
    trajectories: list[BareTrajectory] = []
    done_ids: set[str] = set()
    if output_jsonl is not None and resume:
        done_ids = load_completed_query_ids(output_jsonl)
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    fh = None
    if output_jsonl is not None:
        fh = output_jsonl.open("a", encoding="utf-8")

    try:
        for idx, record in enumerate(records, start=1):
            if record.query_id in done_ids:
                continue
            result = rollout.rollout_chat(
                bare_messages(record.query),
                {"max_new_tokens": max_new_tokens, "temperature": temperature},
            )
            if not result.action_token_ids:
                continue
            traj = BareTrajectory(
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
            trajectories.append(traj)
            if fh is not None:
                fh.write(json.dumps(traj.to_dict(), ensure_ascii=False) + "\n")
                fh.flush()
            if log_every > 0 and idx % log_every == 0:
                print(f"[bare] progress {idx}/{len(records)}", flush=True)
    finally:
        if fh is not None:
            fh.close()
    return trajectories


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
