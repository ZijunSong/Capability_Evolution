#!/usr/bin/env python3
"""SCOPE training entry.

Round-1 main path (configs/scope/sdi_dup_premature.yaml):
  DecisionState → ArtifactV3 → verified routing → action-level SDI
  (no RL / recovery / adaptive weighting)

Legacy dual-mode OPD remains available via configs/scope/dual_mode.yaml
and training/opd_v2/ for ablation baselines.

Does NOT import harness.lifecycle or coevolution.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from harness.artifacts.schema import GuidanceMode
from harness.capability.action_space import CapabilityAction, CapabilityActionType
from harness.capability.selectors import RuleBasedCriticalStateSelector, SelectorConfig
from harness.capability.state import DecisionState
from harness.shadow.registry import build_default_registry
from training.opd_v2.dataset import load_transitions_jsonl, write_transitions_jsonl
from training.opd_v2.pipeline import build_transitions_for_step
from training.opd_v2.trainer import ScopeOPDConfig, ScopeOPDTrainer
from training.opd_v2.weighting import WeightingConfig
from training.scope_config import load_scope_config, scope_section


def _selector_from_cfg(scope: dict[str, Any]) -> RuleBasedCriticalStateSelector:
    sel = scope.get("selector") or {}
    mods = scope.get("modules") or {}
    return RuleBasedCriticalStateSelector(
        SelectorConfig(
            before_stop=bool(sel.get("before_stop", True)),
            after_curate=bool(sel.get("after_curate", True)),
            after_verify=bool(sel.get("after_verify", True)),
            after_review=bool(sel.get("after_review", True)),
            after_pool_growth=bool(sel.get("after_pool_growth", True)),
            repeated_query=bool(sel.get("repeated_query", False)),
            low_remaining_turns=bool(sel.get("low_remaining_turns", False)),
            evidence_enabled=bool(mods.get("evidence_state", True)),
            verification_enabled=bool(mods.get("verification", True)),
            budget_enabled=bool(mods.get("budget_control", False)),
        )
    )


def _registry_from_cfg(scope: dict[str, Any]):
    mods = scope.get("modules") or {}
    return build_default_registry(
        evidence_state=bool(mods.get("evidence_state", True)),
        verification=bool(mods.get("verification", True)),
        budget_control=bool(mods.get("budget_control", False)),
    )


def _trainer_from_cfg(scope: dict[str, Any]) -> ScopeOPDTrainer:
    opd = scope.get("opd") or {}
    guidance = scope.get("guidance") or {}
    wcfg_raw = opd.get("weighting") or {}
    wcfg = WeightingConfig(
        enabled=bool(opd.get("adaptive_weighting", False) or wcfg_raw.get("enabled", False)),
        mode=str(wcfg_raw.get("mode", "fixed")),
        ema_decay=float(wcfg_raw.get("ema_decay", 0.95)),
        min_scale=float(wcfg_raw.get("min_scale", 0.1)),
        max_scale=float(wcfg_raw.get("max_scale", 1.0)),
        update_every=int(wcfg_raw.get("update_every", 20)),
        lambda_0=float(opd.get("lambda_base", 0.01)),
    )
    cfg = ScopeOPDConfig(
        lambda_base=float(opd.get("lambda_base", 0.01)),
        beta=float(opd.get("beta", 5.0)),
        correct_scale=float(opd.get("correct_scale", 1.0)),
        endorse_enabled=bool(guidance.get("endorse", True)),
        correct_enabled=bool(guidance.get("correct", True)),
        weighting=wcfg,
    )
    return ScopeOPDTrainer(cfg)


def make_toy_decision_state(**overrides: Any) -> DecisionState:
    base = dict(
        episode_id="ep_toy",
        task_id="task_toy",
        turn_id=3,
        query="Who founded Acme Corp?",
        rendered_context="Query: Who founded Acme Corp?\nCurated: doc_a",
        action_history=(),
        observation_ids=("obs_t1_1", "obs_t2_1"),
        visible_document_ids=("doc_a", "doc_b"),
        pool_document_ids=("doc_a", "doc_b"),
        curated_document_ids=("doc_a",),
        evidence_claims=(),
        verification_records=(),
        remaining_turns=2,
        remaining_search_calls=None,
        token_budget_used=1000,
        token_budget_total=32768,
        last_action_type="search",
        repeated_query_score=0.2,
        wm_snapshot_hash="abc123",
    )
    base.update(overrides)
    return DecisionState(**base)


def _filter_by_guidance(
    transitions: list,
    scope: dict[str, Any],
) -> list:
    guidance = scope.get("guidance") or {}
    filtered = []
    for tr in transitions:
        if tr.mode == GuidanceMode.ENDORSE and not guidance.get("endorse", True):
            continue
        if tr.mode == GuidanceMode.CORRECT and not guidance.get("correct", True):
            continue
        filtered.append(tr)
    return filtered


def run_dry_run(scope: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    registry = _registry_from_cfg(scope)
    selector = _selector_from_cfg(scope)
    state = make_toy_decision_state()
    action = CapabilityAction(
        action_type=CapabilityActionType.STOP_AND_ANSWER,
        arguments={"reasoning": "done"},
    )
    transitions = build_transitions_for_step(
        state,
        action,
        registry=registry,
        selector=selector,
        final_reward=0.0,
        config=scope,
    )
    transitions = _filter_by_guidance(transitions, scope)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "toy_transitions.jsonl"
    write_transitions_jsonl(path, transitions)
    trainer = _trainer_from_cfg(scope)
    trainer.add_transitions(transitions)
    metrics = trainer.train_step_offline()
    metrics["n_transitions"] = float(len(transitions))
    (out_dir / "dry_run_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


def run_offline(
    scope: dict[str, Any],
    transitions_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    transitions = load_transitions_jsonl(transitions_path)
    if not transitions:
        # Build a small synthetic set if file empty/missing
        return run_dry_run(scope, out_dir)
    trainer = _trainer_from_cfg(scope)
    filtered = _filter_by_guidance(transitions, scope)
    trainer.add_transitions(filtered)
    metrics = trainer.train_step_offline()
    metrics["n_transitions"] = float(len(filtered))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "offline_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (out_dir / "module_stats.json").write_text(
        json.dumps(trainer.export_stats(), indent=2), encoding="utf-8"
    )
    return metrics


def run_online(
    scope: dict[str, Any],
    out_dir: Path,
    *,
    mock_rl_loss: float = 0.5,
) -> dict[str, Any]:
    """Online joint update skeleton.

    Full GPU rollout should call SlidingWindowSearchEnv.export_decision_state()
    each turn and feed build_transitions_for_step(). This entry verifies the
    joint loss path without requiring a live vLLM actor.
    """
    registry = _registry_from_cfg(scope)
    selector = _selector_from_cfg(scope)
    trainer = _trainer_from_cfg(scope)

    # Simulate one on-policy episode with two critical steps
    steps = [
        (
            make_toy_decision_state(turn_id=2, remaining_turns=5),
            CapabilityAction(
                action_type=CapabilityActionType.CURATE_DOCUMENT,
                arguments={"add_ids": ["doc_a"], "remove_ids": []},
            ),
        ),
        (
            make_toy_decision_state(turn_id=4, remaining_turns=1),
            CapabilityAction(
                action_type=CapabilityActionType.STOP_AND_ANSWER,
                arguments={"reasoning": "submit"},
            ),
        ),
    ]
    all_tr = []
    for state, action in steps:
        # Invariant: shadow must not change wm hash (we only use exported state)
        h0 = state.wm_snapshot_hash
        trs = build_transitions_for_step(
            state, action, registry=registry, selector=selector, config=scope
        )
        assert state.wm_snapshot_hash == h0
        all_tr.extend(trs)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_transitions_jsonl(out_dir / "online_transitions.jsonl", all_tr)
    trainer.add_transitions(all_tr)
    metrics = trainer.combine_with_rl(mock_rl_loss)
    metrics["n_transitions"] = float(len(all_tr))
    (out_dir / "online_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SCOPE dual-mode OPD trainer")
    p.add_argument(
        "--mode",
        choices=["dry-run", "offline", "online"],
        default="dry-run",
    )
    p.add_argument(
        "--config",
        type=str,
        default="configs/scope/dual_mode.yaml",
        help="SCOPE YAML config path",
    )
    p.add_argument(
        "--transitions",
        type=str,
        default="",
        help="JSONL path for offline mode",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="outputs/scope_train",
    )
    p.add_argument("--mock-rl-loss", type=float, default=0.5)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    cfg = load_scope_config(args.config)
    scope = scope_section(cfg)
    if not scope.get("enabled", True):
        print("scope.enabled=false — nothing to do (legacy RL path unchanged)")
        return 0

    out_dir = Path(args.out_dir)
    if args.mode == "dry-run":
        metrics = run_dry_run(scope, out_dir)
    elif args.mode == "offline":
        tr_path = Path(args.transitions) if args.transitions else out_dir / "transitions.jsonl"
        metrics = run_offline(scope, tr_path, out_dir)
    else:
        metrics = run_online(scope, out_dir, mock_rl_loss=args.mock_rl_loss)

    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
