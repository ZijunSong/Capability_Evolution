"""Probe C1 measurement: candidate-set JS, Teacher-target NLL, regen legality.

What these numbers are *not* (§8):
- ``D_K`` is the JS between *candidate-set conditional* distributions
  ``p^K(·|s)``, not the full tool-action distribution JS of Metric V2, and not
  a first-name-token full-vocabulary JS.
- ``NLL`` with one Teacher target per state is *Teacher-sample NLL* (a sampled
  estimate of the CE to the Teacher sampling distribution), not the full-
  distribution CE.
- Missing measurements stay missing: <2 distinct candidates → distance is
  ``None`` (缺测), never 0. Empty/truncated generations are failures, not drops.
- Main effects use absolute pre→post drops. ``(pre-post)/(pre+eps)`` ranking and
  signed logp gaps as "divergence" are forbidden.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Sequence

MISSING = "N/A"


class MetricsError(ValueError):
    pass


# ---------------------------------------------------------------------------
# candidate-set distributions
# ---------------------------------------------------------------------------


def sequence_logp_sum(
    logp_fn: Callable[[Sequence[int], Sequence[int]], Sequence[float]],
    *,
    prompt_ids: Sequence[int],
    target_ids: Sequence[int],
) -> float:
    """z_H(a;s) = Σ_j log π(y_j | c_H(s), y_<j) — model-native completion ids."""
    if not target_ids:
        raise MetricsError("empty candidate action sequence")
    logp = list(logp_fn(prompt_ids, target_ids))
    if len(logp) != len(target_ids):
        raise MetricsError(
            f"candidate logp length mismatch: {len(logp)} vs {len(target_ids)}"
        )
    return float(sum(float(x) for x in logp))


def length_normalized(z: float, n_tokens: int) -> float:
    """Sensitivity analysis only; must be labelled score-normalized."""
    if n_tokens <= 0:
        raise MetricsError("n_tokens must be > 0")
    return float(z) / float(n_tokens)


def candidate_probs(z_by_id: Mapping[str, float]) -> dict[str, float]:
    """softmax over the fixed candidate set K(s). Order-independent."""
    if len(z_by_id) < 2:
        raise MetricsError(
            "fewer than 2 distinct candidates: distance is MISSING, not 0"
        )
    ids = sorted(z_by_id)
    zs = [float(z_by_id[i]) for i in ids]
    m = max(zs)
    exps = [math.exp(z - m) for z in zs]
    total = sum(exps)
    return {i: e / total for i, e in zip(ids, exps)}


def js_divergence(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    """Jensen-Shannon divergence (base-e) over an aligned candidate set."""
    keys_p, keys_q = set(p), set(q)
    if keys_p != keys_q:
        raise MetricsError(
            f"candidate id mismatch for JS: {sorted(keys_p ^ keys_q)}"
        )
    keys = sorted(keys_p)
    eps = 1e-12

    def _norm(m: Mapping[str, float]) -> dict[str, float]:
        total = sum(max(float(m[k]), 0.0) for k in keys)
        if total <= 0:
            raise MetricsError("probability mass is zero; cannot normalize")
        return {k: max(float(m[k]), 0.0) / total for k in keys}

    pn, qn = _norm(p), _norm(q)

    def _kl(a: Mapping[str, float], b: Mapping[str, float]) -> float:
        return sum(a[k] * math.log((a[k] + eps) / (b[k] + eps)) for k in keys)

    mid = {k: 0.5 * (pn[k] + qn[k]) for k in keys}
    return 0.5 * _kl(pn, mid) + 0.5 * _kl(qn, mid)


def name_aggregate_probs(
    candidates: Mapping[str, Mapping[str, Any]],
    probs: Mapping[str, float],
) -> dict[str, float]:
    """Aggregate K-probabilities by tool name (optional name-level JS).

    Unequal candidate counts per tool name bias this aggregate; keep the
    caveat with any reported name JS.
    """
    out: dict[str, float] = {}
    for cid, p in probs.items():
        name = str((candidates.get(cid) or {}).get("name") or cid)
        out[name] = out.get(name, 0.0) + float(p)
    return out


# ---------------------------------------------------------------------------
# per-state / per-query records
# ---------------------------------------------------------------------------


@dataclass
class CandidateScore:
    state_id: str
    query_id: str
    candidate_ids: list[str]
    z_teacher: dict[str, float]
    z_student: dict[str, float]
    teacher_probs: dict[str, float]
    student_probs: dict[str, float]
    js: float | None  # None = MISSING (<2 candidates)
    name_js: float | None = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NllScore:
    state_id: str
    query_id: str
    nll: float | None  # None = MISSING
    n_tokens: int
    weight_sum: float
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_candidates(
    *,
    state_id: str,
    query_id: str,
    candidate_ids: Sequence[str],
    z_teacher: Mapping[str, float],
    z_student: Mapping[str, float],
    candidate_names: Mapping[str, str] | None = None,
) -> CandidateScore:
    """Compare real action distributions on one fixed candidate set K(s)."""
    ids = list(candidate_ids)
    if len(set(ids)) != len(ids):
        raise MetricsError("duplicate candidate ids in K(s)")
    if len(ids) < 2:
        return CandidateScore(
            state_id=state_id,
            query_id=query_id,
            candidate_ids=ids,
            z_teacher=dict(z_teacher),
            z_student=dict(z_student),
            teacher_probs={},
            student_probs={},
            js=None,
            name_js=None,
            note=MISSING + ": fewer than 2 distinct candidates",
        )
    tp = candidate_probs({i: float(z_teacher[i]) for i in ids})
    sp = candidate_probs({i: float(z_student[i]) for i in ids})
    js = js_divergence(tp, sp)
    name_js = None
    if candidate_names:
        name_js = js_divergence(
            name_aggregate_probs(
                {i: {"name": candidate_names.get(i, i)} for i in ids}, tp
            ),
            name_aggregate_probs(
                {i: {"name": candidate_names.get(i, i)} for i in ids}, sp
            ),
        )
    return CandidateScore(
        state_id=state_id,
        query_id=query_id,
        candidate_ids=ids,
        z_teacher=dict(z_teacher),
        z_student=dict(z_student),
        teacher_probs=tp,
        student_probs=sp,
        js=js,
        name_js=name_js,
    )


def score_teacher_nll(
    *,
    state_id: str,
    query_id: str,
    token_logprobs: Sequence[float],
    weights: Sequence[float],
) -> NllScore:
    """Masked mean NLL of a frozen legal target y* (Teacher-sample NLL)."""
    if not token_logprobs:
        return NllScore(
            state_id=state_id,
            query_id=query_id,
            nll=None,
            n_tokens=0,
            weight_sum=0.0,
            note=MISSING + ": empty target",
        )
    if len(token_logprobs) != len(weights):
        raise MetricsError("token_logprobs/weights length mismatch")
    wsum = float(sum(float(w) for w in weights))
    if wsum <= 0:
        return NllScore(
            state_id=state_id,
            query_id=query_id,
            nll=None,
            n_tokens=len(token_logprobs),
            weight_sum=0.0,
            note=MISSING + ": zero supervision weight",
        )
    total = sum(-float(lp) * float(w) for lp, w in zip(token_logprobs, weights))
    return NllScore(
        state_id=state_id,
        query_id=query_id,
        nll=float(total / wsum),
        n_tokens=len(token_logprobs),
        weight_sum=wsum,
    )


# ---------------------------------------------------------------------------
# aggregation: per-query first, then equal-weight across queries
# ---------------------------------------------------------------------------


def _per_query_mean(values: Sequence[float]) -> dict[str, float]:
    by_q: dict[str, list[float]] = {}
    for qid, v in values:
        by_q.setdefault(qid, []).append(float(v))
    return {q: sum(vs) / len(vs) for q, vs in by_q.items()}


def aggregate_js(scores: Sequence[CandidateScore]) -> dict[str, Any]:
    """E_s JS with query-level means, then equal-weight queries (macro)."""
    pairs = [(s.query_id, s.js) for s in scores if s.js is not None]
    n_missing = sum(1 for s in scores if s.js is None)
    if not pairs:
        return {
            "d_k": MISSING,
            "d_k_macro": MISSING,
            "d_k_micro": MISSING,
            "n_states_scored": 0,
            "n_states_missing": n_missing,
        }
    per_query = _per_query_mean(pairs)
    macro = sum(per_query.values()) / len(per_query)
    micro = sum(v for _, v in pairs) / len(pairs)
    return {
        "d_k": macro,
        "d_k_macro": macro,
        "d_k_micro": micro,
        "n_states_scored": len(pairs),
        "n_states_missing": n_missing,
        "per_query": per_query,
    }


def aggregate_nll(scores: Sequence[NllScore]) -> dict[str, Any]:
    """Teacher-sample NLL: micro mean over tokens and macro mean over queries."""
    valid = [s for s in scores if s.nll is not None]
    n_missing = len(scores) - len(valid)
    if not valid:
        return {
            "nll_macro": MISSING,
            "nll_micro": MISSING,
            "n_states_scored": 0,
            "n_states_missing": n_missing,
        }
    per_query = _per_query_mean([(s.query_id, s.nll) for s in valid])
    macro = sum(per_query.values()) / len(per_query)
    wsum = sum(s.weight_sum for s in valid)
    micro = (
        sum(s.nll * s.weight_sum for s in valid) / wsum if wsum > 0 else MISSING
    )
    return {
        "nll_macro": macro,
        "nll_micro": micro,
        "n_states_scored": len(valid),
        "n_states_missing": n_missing,
        "per_query": per_query,
    }


def improvement(pre: float | str, post: float | str) -> dict[str, Any]:
    """Absolute pre→post drop (positive = improvement). Relative ranking is banned."""
    if pre == MISSING or post == MISSING:
        return {"abs_drop": MISSING, "note": "missing measurement"}
    drop = float(pre) - float(post)
    return {
        "abs_drop": drop,
        "note": "absolute drop; do not rank on (pre-post)/(pre+eps)",
    }


# ---------------------------------------------------------------------------
# regeneration legality (§8.3)
# ---------------------------------------------------------------------------

FAIL_PARSE = "parse_error"
FAIL_SCHEMA = "schema_error"
FAIL_INACCESSIBLE_REF = "inaccessible_reference"
FAIL_TOOL_RUNTIME = "tool_runtime_failure"
FAIL_NOOP = "noop"
FAIL_TRUNCATION = "truncation"
FAIL_EMPTY = "empty_output"
FAILURE_KINDS = (
    FAIL_PARSE,
    FAIL_SCHEMA,
    FAIL_INACCESSIBLE_REF,
    FAIL_TOOL_RUNTIME,
    FAIL_NOOP,
    FAIL_TRUNCATION,
    FAIL_EMPTY,
)


@dataclass
class GenerationRecord:
    """One regeneration attempt. Attempts are never dropped from the denominator."""

    state_id: str
    query_id: str
    checkpoint: str  # pre | post@N
    ok: bool
    failure_kind: str = ""
    action: dict[str, Any] | None = None
    decode: str = "greedy"
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationContract:
    """Unified generation contract (guide §8.1): probe regeneration must use
    the SAME budget/stop/template as the collection rollout, or truncation
    masquerades as model failure. Restored from the collection manifest, never
    hard-coded per call site.

    - ``finish_reason`` distinguishes natural EOS from budget exhaustion; a
      budget-exhausted generation with an unclosed tool block is
      ``BUDGET_TRUNCATION`` (a measurement-class failure), NOT parse error.
    - ``raw_completion_ids`` / ``raw_text`` are always kept for audit.
    """

    max_new_tokens: int = 2048
    temperature: float = 0.0
    decode: str = "greedy"
    parser_revision: str = ""
    source: str = "collection_manifest"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


FINISH_EOS = "eos"
FINISH_BUDGET_TRUNCATION = "BUDGET_TRUNCATION"
FINISH_ERROR = "generate_error"


@dataclass
class RegenStats:
    n_attempted: int = 0
    n_ok: int = 0
    failures: dict[str, int] = field(default_factory=dict)

    @property
    def invalid_generated_rate(self) -> float | str:
        if self.n_attempted == 0:
            return MISSING
        return (self.n_attempted - self.n_ok) / self.n_attempted

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_attempted": self.n_attempted,
            "n_ok": self.n_ok,
            "invalid_generated_rate": self.invalid_generated_rate,
            "failures": dict(self.failures),
        }


def record_generation(record: GenerationRecord, stats: RegenStats) -> RegenStats:
    stats.n_attempted += 1
    if record.ok and not record.failure_kind:
        stats.n_ok += 1
        return stats
    kind = record.failure_kind or FAIL_PARSE
    if not record.ok and not record.failure_kind:
        kind = FAIL_EMPTY
    stats.failures[kind] = stats.failures.get(kind, 0) + 1
    return stats


def regen_stats(records: Sequence[GenerationRecord]) -> RegenStats:
    stats = RegenStats()
    for rec in records:
        record_generation(rec, stats)
    return stats


def invalid_rate_noninferior(
    pre: RegenStats,
    post: RegenStats,
    *,
    eps_invalid: float,
) -> dict[str, Any]:
    """Guard: generated invalid rate must not rise by more than eps_invalid."""
    pre_r, post_r = pre.invalid_generated_rate, post.invalid_generated_rate
    if pre_r == MISSING or post_r == MISSING:
        return {"pass": MISSING, "delta": MISSING, "note": "missing denominator"}
    delta = float(post_r) - float(pre_r)
    return {"pass": delta <= float(eps_invalid), "delta": delta, "eps": eps_invalid}


# ---------------------------------------------------------------------------
# gates and signal quality (§9)
# ---------------------------------------------------------------------------


@dataclass
class C1Verdict:
    status: str  # PROMOTE | BORDERLINE | DEPRIORITIZE | INVALID | LOW_SUPPORT | LOW_SIGNAL | GUARD_FAIL
    reason: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Gate policy v4: improvement thresholds (δN/δD) and degradation tolerance
# (εD) are separate axes; a small negative R_D within tolerance is
# BORDERLINE, never a mechanical DEPRIORITIZE (guide §9.1/§9.2).
GATE_POLICY_V4 = "probe_c_gate_v4"

# Absolute generation-quality status (guide §8.4): frozen in a pilot or left
# unresolved — never loosened to let an old result pass.
QUALITY_UNRESOLVED = "QUALITY_UNRESOLVED"


def judge_c1(
    *,
    pre_js: Any,
    post_js: Any,
    pre_nll: Any,
    post_nll: Any,
    pre_name_js: Any = MISSING,
    post_name_js: Any = MISSING,
    pre_regen: RegenStats | None = None,
    post_regen: RegenStats | None = None,
    null_js_noise_floor: float | None = None,
    delta_nll: float = 0.0,
    delta_dk: float = 0.0,
    max_dk_regression: float | None = None,
    eps_name_js: float = 0.02,
    eps_invalid: float = 0.02,
    support_status: str = "READY",
    seed_directions: Sequence[bool] | None = None,
    ci_nll: tuple[float, float] | None = None,
    ci_dk: tuple[float, float] | None = None,
    require_generation_guard: bool = False,
    absolute_quality_status: str = "",
    policy_version: str = GATE_POLICY_V4,
) -> C1Verdict:
    """Frozen pilot gates (policy ``probe_c_gate_v4``).

    Semantics split (guide §9):
    - ``delta_nll`` / ``delta_dk``: minimum *meaningful improvement* for PROMOTE.
    - ``max_dk_regression`` (εD): tolerable small regression; a small negative
      R_D with CI still open is BORDERLINE, not DEPRIORITIZE.
    - Regen guard is a strict three-state check: MISSING is a failure when
      ``require_generation_guard`` is set (it can never silently pass).
    - CI lower bounds, when provided, override the point estimates for PROMOTE.
    """
    if support_status in {"UNREGISTERED", "NO_TRIGGER", "NO_SUPPORT"}:
        return C1Verdict(support_status, "support funnel not ready", {"funnel": support_status, "policy_version": policy_version})
    if support_status == "INVALID_MEASUREMENT":
        return C1Verdict("INVALID", "measurement chain broken", {"funnel": support_status, "policy_version": policy_version})

    pre_js_v = None if pre_js == MISSING else float(pre_js)
    post_js_v = None if post_js == MISSING else float(post_js)
    pre_nll_v = None if pre_nll == MISSING else float(pre_nll)
    post_nll_v = None if post_nll == MISSING else float(post_nll)
    if pre_js_v is None or post_js_v is None or pre_nll_v is None or post_nll_v is None:
        return C1Verdict(
            "INVALID",
            "missing core measurement",
            {"pre_js": pre_js, "post_js": post_js, "policy_version": policy_version},
        )

    if null_js_noise_floor is not None and pre_js_v <= float(null_js_noise_floor):
        return C1Verdict(
            "LOW_SIGNAL",
            "pre D_K at/below null noise floor; relative improvement is meaningless",
            {"pre_js": pre_js_v, "null_js_noise_floor": null_js_noise_floor, "policy_version": policy_version},
        )

    r_n = pre_nll_v - post_nll_v
    r_d = pre_js_v - post_js_v
    eps_d = float(max_dk_regression) if max_dk_regression is not None else 0.0
    detail = {
        "R_N": r_n,
        "R_D": r_d,
        "delta_nll": delta_nll,
        "delta_dk": delta_dk,
        "max_dk_regression": eps_d,
        "policy_version": policy_version,
    }

    name_ok = True
    if pre_name_js != MISSING and post_name_js != MISSING:
        name_ok = (float(post_name_js) - float(pre_name_js)) <= float(eps_name_js)
        detail["name_js_delta"] = float(post_name_js) - float(pre_name_js)

    # Strict three-state guard: True / False / MISSING (never MISSING→pass).
    # No regen stats at all also counts as MISSING when the guard is required.
    invalid_ok: bool | str = True
    if pre_regen is not None and post_regen is not None:
        guard = invalid_rate_noninferior(pre_regen, post_regen, eps_invalid=eps_invalid)
        detail["invalid_guard"] = guard
        if guard.get("pass") is True:
            invalid_ok = True
        elif guard.get("pass") is False:
            invalid_ok = False
        else:
            invalid_ok = MISSING
    else:
        invalid_ok = MISSING
    guard_missing = invalid_ok == MISSING

    directions_ok = True
    if seed_directions:
        directions_ok = all(bool(x) for x in seed_directions)
        detail["seed_directions"] = list(seed_directions)

    # Point-estimate double improvement (both beyond their δ).
    double_ok = (r_n > float(delta_nll)) and (r_d > float(delta_dk))
    detail["double_improvement"] = double_ok

    # CI-based strictness when intervals are provided (guide §9.3 rule 4/5).
    if ci_nll is not None and ci_dk is not None:
        ci_pass = (float(ci_nll[0]) > float(delta_nll)) and (float(ci_dk[0]) > float(delta_dk))
        ci_degrades = float(ci_dk[1]) < -eps_d
        detail["ci"] = {
            "nll": [float(ci_nll[0]), float(ci_nll[1])],
            "dk": [float(ci_dk[0]), float(ci_dk[1])],
            "lower_bounds_pass": ci_pass,
            "dk_ci_excludes_regression": not ci_degrades,
        }
    else:
        ci_pass = double_ok
        ci_degrades = False
        detail["ci"] = MISSING

    # Guard status summary (three-state, surfaced in the report).
    guard_status = "PASS" if invalid_ok is True else ("FAIL" if invalid_ok is False else "MISSING")
    detail["guard_status"] = guard_status
    if absolute_quality_status:
        detail["absolute_quality_status"] = absolute_quality_status

    # Rule 1: guard wiring required but missing → incomplete, never PROMOTE.
    if require_generation_guard and guard_missing:
        return C1Verdict(
            "INCOMPLETE_MEASUREMENT",
            "generation guard MISSING while require_generation_guard=True",
            detail,
        )
    # Rule 2 (§9.3): clear legality/quality regression blocks auto-promotion.
    if invalid_ok is False or not name_ok:
        return C1Verdict(
            "GUARD_FAIL",
            "clear legality/quality regression; auto-promotion blocked",
            detail,
        )
    # Rule 4: PROMOTE needs δ-beyond improvement on both axes + all guards.
    # Absolute quality unresolved (or unspecified) can never yield a deployable
    # PROMOTE (guide §8.4): default is QUALITY_UNRESOLVED, frozen in a pilot.
    if ci_pass and double_ok and directions_ok and invalid_ok is True:
        if not absolute_quality_status or absolute_quality_status == QUALITY_UNRESOLVED:
            detail["absolute_quality_status"] = absolute_quality_status or QUALITY_UNRESOLVED
            return C1Verdict(
                "BORDERLINE",
                "meets gates but absolute generation quality unresolved "
                "(not deployable PROMOTE)",
                detail,
            )
        return C1Verdict("PROMOTE", "double improvement without legality regression", detail)
    # Guard-only-missing variant: gates pass on the axes, quality frozen, but
    # no regen stats were provided at all — also not a deployable PROMOTE.
    if (
        ci_pass
        and double_ok
        and directions_ok
        and guard_missing
        and (not absolute_quality_status or absolute_quality_status != QUALITY_UNRESOLVED)
    ):
        return C1Verdict(
            "PROMOTE",
            "double improvement (guard stats not collected this run)",
            detail,
        )
    # Rule 5: explicit, CI-supported regression → DEPRIORITIZE.
    if ci_degrades:
        return C1Verdict(
            "DEPRIORITIZE",
            "CI-supported D_K regression beyond tolerance",
            detail,
        )
    # Rule 3/6: NLL reliable but D_K unresolved → explicit subreason.
    if r_n > float(delta_nll) and not double_ok:
        return C1Verdict(
            "BORDERLINE",
            "FIT_IMPROVED_POLICY_UNRESOLVED",
            detail,
        )
    # Everything else is BORDERLINE (small negative R_D with open CI lands here).
    if r_n <= 0 and r_d <= 0:
        return C1Verdict(
            "DEPRIORITIZE",
            "no short-horizon improvement on either axis",
            detail,
        )
    return C1Verdict("BORDERLINE", "improvement below frozen thresholds or CI too wide", detail)


__all__ = [
    "C1Verdict",
    "CandidateScore",
    "FAILURE_KINDS",
    "FAIL_EMPTY",
    "FAIL_INACCESSIBLE_REF",
    "FAIL_NOOP",
    "FAIL_PARSE",
    "FAIL_SCHEMA",
    "FAIL_TOOL_RUNTIME",
    "FAIL_TRUNCATION",
    "GATE_POLICY_V4",
    "GenerationRecord",
    "MISSING",
    "MetricsError",
    "NllScore",
    "QUALITY_UNRESOLVED",
    "RegenStats",
    "aggregate_js",
    "aggregate_nll",
    "candidate_probs",
    "improvement",
    "invalid_rate_noninferior",
    "js_divergence",
    "judge_c1",
    "length_normalized",
    "name_aggregate_probs",
    "record_generation",
    "regen_stats",
    "score_candidates",
    "score_teacher_nll",
    "sequence_logp_sum",
]
