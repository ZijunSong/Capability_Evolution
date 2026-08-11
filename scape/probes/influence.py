"""Same-State Policy Influence probe.

State occupancy MUST come from Reduced Harness Student rollouts:
  xi_t ~ d^(pi, H_-m)

Full harness must not run an independent rollout for influence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from scape.rendering.dual_view import DualViewRenderer
from scape.state.snapshot import EnvironmentSnapshot
from scape.training.tool_opd import (
    disagreement_stats,
    js_divergence,
    normalize_probs,
    token_kl,
    tool_name_divergence,
)


PolicyFn = Callable[[Mapping[str, Any]], dict[str, Any]]


@dataclass
class InfluenceSample:
    snapshot_hash: str
    query_id: str
    step: int
    I_name: float
    I_args: float
    extras: dict[str, Any] = field(default_factory=dict)


def score_influence_on_snapshot(
    snapshot: EnvironmentSnapshot,
    *,
    component_id: str,
    student_policy: PolicyFn,
    teacher_policy: PolicyFn,
    renderer: DualViewRenderer | None = None,
    teacher_arg_logprobs: Sequence[float] | None = None,
    student_arg_logprobs: Sequence[float] | None = None,
) -> InfluenceSample:
    """Compute I_name / I_args on one student-owned snapshot."""
    rend = renderer or DualViewRenderer()
    before = rend.environment_steps
    dual = rend.render_pair(snapshot, component_id=component_id)
    student_out = student_policy(dual.student_view)
    teacher_out = teacher_policy(dual.full_view)
    after = rend.environment_steps
    if after != before:
        raise RuntimeError("influence probe must not step env via full teacher")

    s_probs = normalize_probs(student_out.get("tool_name_probs") or {})
    t_probs = normalize_probs(teacher_out.get("tool_name_probs") or {})
    name_stats = tool_name_divergence(s_probs, t_probs)

    if teacher_arg_logprobs is not None and student_arg_logprobs is not None:
        i_args = token_kl(student_arg_logprobs, teacher_arg_logprobs)
    else:
        # Fallback: disagreement-based proxy when token logprobs unavailable
        disc = disagreement_stats(
            student_out.get("decoded") or {"name": None, "arguments": {}},
            teacher_out.get("decoded") or {"name": None, "arguments": {}},
        )
        i_args = float(disc["argument_edit_distance"])

    disc = disagreement_stats(
        student_out.get("decoded") or {"name": None, "arguments": {}},
        teacher_out.get("decoded") or {"name": None, "arguments": {}},
    )
    extras = {
        **name_stats,
        **disc,
        "null_same_render_js": 0.0,
        "null_field_order_js": 0.0,
    }
    # Null controls: same render vs same render should be ~0
    same = dual.null_controls.get("same_render")
    if same is not None:
        extras["null_same_render_js"] = js_divergence(
            normalize_probs(student_policy(same).get("tool_name_probs") or {}),
            normalize_probs(student_policy(same).get("tool_name_probs") or {}),
        )
    ford = dual.null_controls.get("field_order_only")
    if ford is not None:
        extras["null_field_order_js"] = js_divergence(
            s_probs,
            normalize_probs(student_policy(ford).get("tool_name_probs") or {}),
        )

    return InfluenceSample(
        snapshot_hash=snapshot.content_hash(),
        query_id=snapshot.query_id,
        step=snapshot.step,
        I_name=float(name_stats["I_name_js"]),
        I_args=float(i_args),
        extras=extras,
    )


def aggregate_influence(samples: Sequence[InfluenceSample]) -> dict[str, Any]:
    if not samples:
        raise ValueError("empty influence samples")
    n = len(samples)
    mean_name = sum(s.I_name for s in samples) / n
    mean_args = sum(s.I_args for s in samples) / n
    null_same = sum(float(s.extras.get("null_same_render_js", 0.0)) for s in samples) / n
    null_order = sum(float(s.extras.get("null_field_order_js", 0.0)) for s in samples) / n
    return {
        "n": n,
        "I_name_mean": mean_name,
        "I_args_mean": mean_args,
        "null_same_render_mean": null_same,
        "null_field_order_mean": null_order,
        "above_null": mean_name > null_same + 1e-6 and mean_name > null_order,
    }


def assert_reduced_rollout_owns_state(
    snapshots: Sequence[EnvironmentSnapshot],
    *,
    component_id: str,
) -> None:
    """Every snapshot must have been collected under H_-m (component disabled)."""
    for snap in snapshots:
        if snap.harness_mask.get(component_id, True):
            raise AssertionError(
                f"snapshot {snap.query_id}@{snap.step} was not collected under H_-m "
                f"for {component_id}; reduced rollout must own state distribution"
            )
