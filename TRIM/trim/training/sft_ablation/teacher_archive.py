"""Read-only Full teacher archive shared by Teacher SFT and Projected Teacher SFT."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from trim.training.opd_events import EventKind
from trim.training.sft_ablation.contracts import fingerprint

ARCHIVE_SCHEMA = "full_teacher_archive_v1"


def stable_event_id(trajectory_id: str, turn_id: int, index: int) -> str:
    return f"{trajectory_id}:t{int(turn_id)}:e{int(index)}"


def events_from_native_turn(
    *,
    trajectory_id: str,
    turn_id: int,
    action: Mapping[str, Any],
    observation: Mapping[str, Any] | None,
    pre_curated: Sequence[str],
    post_curated: Sequence[str],
    pre_state_hash: str,
    post_state_hash: str,
    prompt_ids: Sequence[int],
    executed_ok: bool,
    view_changed: bool = False,
    start_index: int = 0,
) -> list[dict[str, Any]]:
    """Split a native turn so a search is not stored as its auto-curate delta.

    A native curate keeps one action with the real curated delta and does not
    emit a second mutation that would execute the same delta again.
    """
    name = str(action.get("name") or "")
    arguments = dict(action.get("arguments") or {})
    before = [str(item) for item in pre_curated]
    after = [str(item) for item in post_curated]
    changed = set(before) != set(after)
    events: list[dict[str, Any]] = []
    index = int(start_index)

    def _event(kind: str, **extra: Any) -> dict[str, Any]:
        nonlocal index
        payload = {
            "event_id": stable_event_id(trajectory_id, turn_id, index),
            "event_index": index,
            "turn_id": int(turn_id),
            "kind": kind,
            "component": extra.pop("component", "model"),
            "pre_state_hash": pre_state_hash,
            "post_state_hash": post_state_hash,
            "prompt_ids": list(prompt_ids),
            "executed_ok": bool(executed_ok),
        }
        payload.update(extra)
        index += 1
        return payload

    state_delta: dict[str, Any] = {}
    if name == "curate":
        state_delta = {
            "curated_delta": {
                "add_ids": list(arguments.get("add_ids") or []),
                "remove_ids": list(arguments.get("remove_ids") or []),
            },
            "before_curated": before,
            "after_curated": after,
        }
    action_event = _event(
        EventKind.MODEL_ACTION.value,
        component="model",
        action_name=name,
        arguments=arguments,
        state_delta=state_delta,
        source_hook="native_model_action",
    )
    if name != "curate" and "curated_delta" in action_event["state_delta"]:
        raise RuntimeError("search/model action must not carry curated_delta")
    events.append(action_event)
    if observation is not None:
        events.append(
            _event(
                EventKind.TOOL_OBSERVATION.value,
                component="tool",
                observation=dict(observation),
                source_hook="tool_return",
            )
        )
    if name != "curate" and changed:
        events.append(
            _event(
                EventKind.HARNESS_MUTATION.value,
                component="auto_populate_first_search",
                state_delta={"before_curated": before, "after_curated": after, "curated_delta": _delta(before, after)},
                source_hook="_maybe_wrap_search_output",
            )
        )
    if view_changed:
        events.append(
            _event(
                EventKind.OBS_TRANSFORM.value,
                component="sentence_compress",
                state_delta={"view_changed": True},
                source_hook="obs_transform",
            )
        )
    return events


def _delta(before: Sequence[str], after: Sequence[str]) -> dict[str, list[str]]:
    prev, nxt = set(before), set(after)
    return {"add_ids": sorted(nxt - prev), "remove_ids": sorted(prev - nxt)}


def teacher_rollout_kwargs(
    *,
    policy_version: str,
    group_size: int,
    max_turns: int,
    max_new_tokens: int,
    seed: int,
    enc: Any,
    train_env: str,
    train_session: Any = None,
) -> dict[str, Any]:
    from trim.upstream_harness1.v8d_flags import all_enabled_mask

    return {
        "component_id": "all",
        "teacher_mode": False,
        "harness_mask": all_enabled_mask("Harness-1"),
        "collection_mode": "rl+opd",
        "opd_loss": "sr_opd_ce",
        "enable_rl": False,
        "policy_version": policy_version,
        "group_size": int(group_size),
        "max_turns": int(max_turns),
        "max_new": int(max_new_tokens),
        "seed": int(seed),
        "sample": True,
        "enc": enc,
        "train_env": train_env,
        "train_session": train_session,
    }


class FrozenTeacherSession:
    """Generate a fixed teacher archive. This object has no optimizer."""

    def __init__(self, generate_batch: Any, *, policy_version: str) -> None:
        self.generate_batch = generate_batch
        self.policy_version = policy_version
        self.optimizer_steps = 0
        self.calls: list[dict[str, Any]] = []

    def collect(self, rows: Sequence[dict[str, Any]], rollout_fn: Any, **kwargs: Any) -> dict[str, Any]:
        spec = teacher_rollout_kwargs(policy_version=self.policy_version, **kwargs)
        if spec["teacher_mode"] is not False:
            raise RuntimeError("teacher archive generation must set teacher_mode=False")
        if spec["enable_rl"]:
            raise RuntimeError("teacher archive generation must not enable RL")
        self.calls.append({"generate": self.generate_batch, **spec})
        groups = rollout_fn(self.generate_batch, list(rows), **spec)
        if self.optimizer_steps != 0:
            raise RuntimeError("teacher parameters were updated during archive generation")
        return archive_from_groups(groups, policy_version=self.policy_version, mask_name="all")


def archive_from_groups(groups: Sequence[Any], *, policy_version: str, mask_name: str) -> dict[str, Any]:
    trajectories = []
    for group in groups:
        query_id = str(getattr(group, "query_id", "") or "")
        points = list(getattr(group, "decision_points", []) or [])
        by_rollout: dict[int, list[Any]] = {}
        for point in points:
            by_rollout.setdefault(int(getattr(point, "rollout_idx", 0)), []).append(point)
        for rollout_idx, decisions in sorted(by_rollout.items()):
            trajectory_id = f"{query_id}:r{rollout_idx}"
            events: list[dict[str, Any]] = []
            stored = []
            for point in sorted(decisions, key=lambda item: int(item.turn_id)):
                action = _action_of(point)
                pre = list(getattr(point, "accessible_doc_ids", []) or [])
                post_snap = getattr(point, "post_action_snapshot", None)
                post = []
                if post_snap is not None:
                    from trim.training.opd_realizability import curated_ids_of

                    post = curated_ids_of(post_snap)
                turn_events = events_from_native_turn(
                    trajectory_id=trajectory_id,
                    turn_id=int(point.turn_id),
                    action=action,
                    observation=None,
                    pre_curated=pre,
                    post_curated=post,
                    pre_state_hash=str(point.pre_action_snapshot_hash),
                    post_state_hash="" if post_snap is None else post_snap.content_hash(),
                    prompt_ids=list(point.student_prompt_token_ids or []),
                    executed_ok=bool(point.executed_ok),
                    start_index=len(events),
                )
                events.extend(turn_events)
                stored.append(
                    {
                        "turn_id": int(point.turn_id),
                        "decision_id": str(point.decision_point_id),
                        "prompt_ids": list(point.student_prompt_token_ids or []),
                        "action": action,
                        "executed_ok": bool(point.executed_ok),
                        "structurally_valid": bool(point.structurally_valid),
                        "pre_state_hash": str(point.pre_action_snapshot_hash),
                        "reward": point.reward,
                    }
                )
            trajectories.append(
                {
                    "trajectory_id": trajectory_id,
                    "query_id": query_id,
                    "rollout_idx": int(rollout_idx),
                    "policy_version": policy_version,
                    "termination": "recorded",
                    "decisions": stored,
                    "events": events,
                }
            )
    archive = {
        "schema_version": ARCHIVE_SCHEMA,
        "policy_version": policy_version,
        "mask": mask_name,
        "teacher_mode": False,
        "trajectories": trajectories,
    }
    archive["digest"] = fingerprint(
        {"policy_version": policy_version, "trajectories": trajectories}
    )
    return archive


def _action_of(point: Any) -> dict[str, Any]:
    text = str(getattr(point, "student_action_text", "") or "")
    if text.strip():
        try:
            from trim.training.action_codec import parse_action

            return parse_action(text)
        except Exception:
            pass
    names = list(getattr(point, "action_tool_names", []) or [])
    if names:
        return {"name": str(names[0]), "arguments": {}}
    return {"name": "", "arguments": {}}


def seal_archive(archive: Mapping[str, Any], *, contract_digest: str) -> dict[str, Any]:
    if archive.get("schema_version") != ARCHIVE_SCHEMA:
        raise ValueError("teacher archive schema mismatch")
    if archive.get("teacher_mode") is not False:
        raise ValueError("teacher archive must record teacher_mode=False")
    sealed = dict(archive)
    sealed["contract_digest"] = contract_digest
    sealed["digest"] = fingerprint(
        {
            "policy_version": archive.get("policy_version"),
            "trajectories": archive.get("trajectories"),
        }
    )
    return sealed


def run_teacher_collection_job(job: Mapping[str, Any]) -> dict[str, Any]:
    """Real Full-mask generation inside an already-configured worker process."""
    from trim.eval.model_tokenizer import load_model_encoding
    from trim.training.batched_env_rollout import rollout_queries_batched
    from trim.training.vllm_hybrid import VLLMGenerateClient

    queries = list(job["queries"])
    policy_version = str(job["policy_version"])
    enc = load_model_encoding(
        str(job["model_name"]),
        max_model_len=int(job["max_model_len"]),
        max_new_tokens=int(job["max_new_tokens"]),
    )
    client = VLLMGenerateClient(
        model_path=str(job["model_name"]),
        session_dir=Path(job["session_dir"]),
        tensor_parallel_size=int(job.get("tensor_parallel_size") or 1),
        max_model_len=int(job["max_model_len"]),
        lora_path=str(job.get("adapter") or "") or None,
    )
    try:
        client.start()
        session = FrozenTeacherSession(client.generate_batch, policy_version=policy_version)
        archive = session.collect(
            queries,
            rollout_queries_batched,
            group_size=int(job["group_size"]),
            max_turns=int(job["max_turns"]),
            max_new_tokens=int(job["max_new_tokens"]),
            seed=int(job["seed"]),
            enc=enc,
            train_env=str(job["train_env"]),
        )
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
    return {"ok": True, "archive": archive}
