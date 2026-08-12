"""Contract: HF and vLLM replay must both disable REPLAN in P0/R10 binary mode."""

from __future__ import annotations

import ast
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]


def _calls_disable_replan(path: Path) -> bool:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            name = getattr(fn, "attr", None) or getattr(fn, "id", None)
            # Direct decide_rollback_operation(...) or CanonicalRollbackOperationScorer(..., disable_replan=)
            if name not in {
                "decide_rollback_operation",
                "CanonicalRollbackOperationScorer",
            }:
                continue
            for kw in node.keywords:
                if kw.arg == "disable_replan":
                    return True
    return False


def test_hf_and_vllm_replay_pass_disable_replan() -> None:
    hf = _REPO / "training/scope_round9/replay_frozen_hf.py"
    vllm = _REPO / "training/scope_round9/replay_frozen_vllm.py"
    assert _calls_disable_replan(hf)
    assert _calls_disable_replan(vllm)


def test_closed_loop_runtime_passes_disable_replan() -> None:
    runtime = _REPO / "training/scope/rollback_operation_runtime.py"
    assert _calls_disable_replan(runtime)


def test_disable_replan_excludes_replan_argmax() -> None:
    from training.scope.decide_rollback_operation import decide_rollback_operation

    d = decide_rollback_operation(
        score_continue=-2.0,
        score_replan=-0.1,  # would win if allowed
        score_rollback=-1.5,
        threshold=0.0,
        disable_replan=True,
    )
    assert d.predicted_operation.value == "ROLLBACK_TO"
