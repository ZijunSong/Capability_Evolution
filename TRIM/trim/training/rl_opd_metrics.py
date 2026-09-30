"""Hybrid step logging helpers. Reward and projection coverage stay separate."""

from __future__ import annotations

from typing import Any, Sequence

from trim.training.rl_opd_types import HybridStepMetrics, UPDATE_SKIPPED


def empty_hybrid_metrics(*, policy_version: str, lambda_opd: float) -> HybridStepMetrics:
    return HybridStepMetrics(
        update_type=UPDATE_SKIPPED,
        n_rl_datums=0,
        n_opd_datums=0,
        n_rl_tokens=0,
        n_opd_tokens=0,
        rl_loss_proxy=None,
        opd_nll=None,
        lambda_opd=float(lambda_opd),
        projection_coverage=0.0,
        reject_rate=0.0,
        policy_version=policy_version,
    )


def split_log_groups(metrics: HybridStepMetrics) -> dict[str, dict[str, Any]]:
    """Three log namespaces: rl / opd / hybrid."""
    return {
        "rl": {
            "native_loss_proxy": metrics.rl_loss_proxy,
            "num_datums": metrics.n_rl_datums,
            "num_loss_tokens": metrics.n_rl_tokens,
        },
        "opd": {
            "nll": metrics.opd_nll,
            "weighted_ce": metrics.opd_weighted_ce,
            "weighted_gap": metrics.opd_weighted_gap,
            "unweighted_nll": metrics.opd_unweighted_nll,
            "num_datums": metrics.n_opd_datums,
            "num_loss_tokens": metrics.n_opd_tokens,
            "projection_coverage": metrics.projection_coverage,
            "reject_rate": metrics.reject_rate,
        },
        "hybrid": {
            "lambda_opd": metrics.lambda_opd,
            "update_type": metrics.update_type,
            "optimizer_steps": metrics.n_optimizer_steps,
            "rl_fb_calls": metrics.n_rl_forward_backward,
            "opd_fb_calls": metrics.n_opd_forward_backward,
            "opd_to_rl_token_ratio": metrics.opd_to_rl_token_ratio,
        },
    }


def mean_reward(rewards: Sequence[float]) -> dict[str, float]:
    vals = [float(x) for x in rewards]
    if not vals:
        return {"reward_mean": 0.0, "reward_std": 0.0, "n": 0.0}
    mean = sum(vals) / len(vals)
    var = sum((x - mean) ** 2 for x in vals) / max(1, len(vals))
    return {"reward_mean": mean, "reward_std": var**0.5, "n": float(len(vals))}


def _corr(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0.0 or vy <= 0.0:
        return None
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return float(cov / (vx * vy) ** 0.5)


def reward_parts_group_stats(groups: Sequence[Any]) -> dict[str, Any]:
    """Record task/legal/shaping stats without changing the reward formula."""
    n_groups = 0
    n_task_zero_total_var = 0
    task_all: list[float] = []
    legal_all: list[float] = []
    shaping_all: list[float] = []
    parse_valid = 0
    n_valids = 0
    exec_ok = 0
    n_exec = 0
    n_end_search = 0
    n_timeout = 0
    n_episodes = 0
    n_const_groups = 0
    n_success = 0
    n_success_eps = 0
    n_tool_calls_all = 0
    group_reward_std_sum = 0.0
    group_reward_std_n = 0
    n_curated_eps = 0
    n_gold_eps = 0
    for group in groups:
        stats = list(((getattr(group, "trajectory_group", None) or {}).get("episode_stats") or []))
        parts = [dict(s.get("reward_parts") or {}) for s in stats if s.get("reward_parts")]
        if not parts:
            continue
        n_groups += 1
        task = [float(p.get("task") or 0.0) for p in parts]
        legal = [float(p.get("legal") or 0.0) for p in parts]
        shaping = [float(p.get("shaping") or 0.0) for p in parts]
        total = [float(p.get("total") or 0.0) for p in parts]
        turn_pen = [float(p.get("turn_penalty") or 0.0) for p in parts]
        n_turns_ep = [float(p.get("n_turns") or 0.0) for p in parts]
        task_all.extend(task)
        legal_all.extend(legal)
        shaping_all.extend(shaping)
        task_var = sum((x - (sum(task) / len(task))) ** 2 for x in task) / len(task)
        total_var = sum((x - (sum(total) / len(total))) ** 2 for x in total) / len(total)
        if task_var < 1e-12 and total_var > 1e-12:
            n_task_zero_total_var += 1
        # Group-internal reward std == the advantage normalization denominator
        # (episode_relative_advantages divides by exactly this std). Averaging
        # across groups gives the mean advantage-std for the step.
        if total_var < 1e-12:
            n_const_groups += 1
        group_reward_std_sum += (total_var ** 0.5)
        group_reward_std_n += 1
        for s in stats:
            n_episodes += 1
            names = list(s.get("names") or s.get("tool_names") or [])
            if "end_search" in names:
                n_end_search += 1
            if not s.get("ended") and int(s.get("n_turns") or 0) >= int(s.get("max_turns") or 0):
                n_timeout += 1
            n_valids += int(s.get("n_valids") or s.get("n_turns") or 0)
            parse_valid += int(s.get("n_structurally_valid") or 0)
            n_exec += int(s.get("n_valids") or s.get("n_turns") or 0)
            exec_ok += int(s.get("n_exec_ok") or 0)
            n_tool_calls_all += int(s.get("n_tool_calls") or s.get("n_turns") or 0)
            # success = recall > 0 (an episode found at least one gold doc).
            eps_recall = float(s.get("final_answer_recall") or s.get("task_recall") or 0.0)
            if eps_recall > 0:
                n_success_eps += 1
            n_curated_eps += int(s.get("n_curated") or 0)
            n_gold_eps += int(s.get("n_gold") or 0)
    def _mv(xs: list[float]) -> dict[str, float]:
        if not xs:
            return {"mean": 0.0, "var": 0.0, "n": 0.0}
        m = sum(xs) / len(xs)
        v = sum((x - m) ** 2 for x in xs) / len(xs)
        return {"mean": m, "var": v, "n": float(len(xs))}
    return {
        "n_groups": n_groups,
        "n_episodes": n_episodes,
        "task": _mv(task_all),
        "legal": _mv(legal_all),
        "shaping": _mv(shaping_all),
        "turn_penalty": _mv(turn_pen),
        "n_turns": _mv(n_turns_ep),
        "reward_mode": (parts[0].get("reward_mode") if parts else None),
        "corr_task_legal": _corr(task_all, legal_all),
        "corr_task_shaping": _corr(task_all, shaping_all),
        "frac_task_var_zero_total_var_nonzero": (
            (n_task_zero_total_var / n_groups) if n_groups else 0.0
        ),
        "parse_valid_rate": (parse_valid / max(1, n_valids)) if n_valids else None,
        "exec_ok_rate": (exec_ok / max(1, n_exec)) if n_exec else None,
        "end_search_rate": (n_end_search / max(1, n_episodes)) if n_episodes else None,
        "timeout_rate": (n_timeout / max(1, n_episodes)) if n_episodes else None,
        "reward_formula_unchanged": True,
        # New fields for the recall-only RL tensorboard.
        "n_const_reward_groups": n_const_groups,
        "zero_variance_group_ratio": (n_const_groups / n_groups) if n_groups else 0.0,
        "group_reward_std_mean": (group_reward_std_sum / group_reward_std_n) if group_reward_std_n else 0.0,
        "success_rate": (n_success_eps / n_episodes) if n_episodes else 0.0,
        "n_tool_calls_total": n_tool_calls_all,
        "n_curated_eps": n_curated_eps,
        "n_gold_eps": n_gold_eps,
        "selected_ratio": (n_curated_eps / n_gold_eps) if n_gold_eps else 0.0,
        "task_recall_mean": _mv(task_all)["mean"],
    }
