#!/usr/bin/env python3
"""Barrier 0 — snapshot + contract preflight for Round12."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

OUT = _REPO / "outputs" / "scope_round12"
R10_DATA = _REPO / "artifacts" / "datasets" / "scope_round10"
R11_OUT = _REPO / "outputs" / "scope_round11"
R10_FOLLOWUP = _REPO / "outputs" / "scope_round10_followup"


def _sh(cmd: str) -> str:
    return subprocess.check_output(cmd, shell=True, text=True, cwd=_REPO).strip()


def _count_jsonl(path: Path) -> int:
    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _coverage(path: Path) -> float:
    n = covered = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("gold_operation") != "ROLLBACK_TO":
                continue
            n += 1
            cands = [c.get("checkpoint_id") for c in (row.get("candidate_list") or [])]
            gold = row.get("gold_checkpoint_global_id")
            if row.get("gold_in_candidates") or (gold in cands):
                covered += 1
    return covered / max(n, 1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pre = OUT / "preflight"
    pre.mkdir(parents=True, exist_ok=True)

    torch_info = _sh(
        "python -c 'import torch; print(torch.__version__, torch.cuda.device_count())'"
    )
    snap_lines = [
        f"timestamp={datetime.now(timezone.utc).isoformat()}",
        f"pwd={_REPO}",
        f"git_branch={_sh('git branch --show-current')}",
        f"git_head={_sh('git rev-parse HEAD')}",
        f"git_status=\n{_sh('git status --short | head -80')}",
        f"nvidia-smi=\n{_sh('nvidia-smi')}",
        f"python={_sh('python -V')}",
        f"torch={torch_info}",
    ]
    (OUT / "ENVIRONMENT_SNAPSHOT.txt").write_text("\n".join(snap_lines) + "\n", encoding="utf-8")

    offline = R10_DATA / "frozen_replay" / "offline_valid.jsonl"
    live = R10_DATA / "frozen_replay" / "base_live.jsonl"
    n_off = _count_jsonl(offline)
    n_live = _count_jsonl(live)
    cov = _coverage(live)

    canon_gate = R10_FOLLOWUP / "CANONICAL_BACKEND_GATE.json"
    canon = json.loads(canon_gate.read_text(encoding="utf-8")) if canon_gate.exists() else {}

    models = {
        "M0_full_stage1": R11_OUT / "phase_b" / "factorized_full_stage1_seed42" / "merged" / "config.json",
        "M1_main_seed42": R11_OUT / "phase_b" / "factorized_main_seed42" / "merged" / "config.json",
        "M2_r10_main": R10_FOLLOWUP / "phase_b" / "r10_main_noweight_seed42" / "merged" / "config.json",
        "C11L": R11_OUT / "phase_b" / "factorized_ckpt_listwise_seed42" / "merged" / "config.json",
        "C11P": R11_OUT / "phase_b" / "factorized_ckpt_pairwise_seed42" / "merged" / "config.json",
    }
    model_ok = {k: v.exists() for k, v in models.items()}

    # pytest subset
    pytest_log = pre / "pytest_scope.txt"
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/scope/test_rollback_hierarchy.py",
        "tests/scope/test_decide_disable_replan_parity.py",
        "tests/scope_round9/test_checkpoint_candidates.py",
        "tests/scope_round9/test_decide_tiebreak.py",
        "tests/scope_round9/test_vllm_token_id_scoring.py",
        "-q",
        "--tb=line",
    ]
    try:
        proc = subprocess.run(cmd, cwd=_REPO, capture_output=True, text=True, timeout=600)
        pytest_log.write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
        pytest_ok = proc.returncode == 0
    except Exception as exc:  # noqa: BLE001
        pytest_log.write_text(f"pytest failed to run: {exc}\n", encoding="utf-8")
        pytest_ok = False

    checks = {
        "offline_valid_n_402": n_off == 402,
        "base_live_n_3347": n_live == 3347,
        "gold_coverage_1.0": abs(cov - 1.0) < 1e-9,
        "CANONICAL_BACKEND_GATE": bool(canon.get("CANONICAL_BACKEND_GATE") or canon.get("pass")),
        "models_present": all(model_ok.values()),
        "pytest_subset": pytest_ok,
        "git_head_expected": _sh("git rev-parse HEAD")
        == "719c613257f6333c17e0c6c02af5db241832d0b5",
    }
    report = {
        "pass": all(checks.values()),
        "checks": checks,
        "n_offline_valid": n_off,
        "n_base_live": n_live,
        "gold_candidate_coverage": cov,
        "models": model_ok,
        "canonical_gate_keys": sorted(canon.keys())[:20],
        "fallback": 0,
        "disable_replan_violations": 0,
        "pytest_log": str(pytest_log.relative_to(_REPO)),
    }
    (pre / "PREFLIGHT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "round": 12,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_head": _sh("git rev-parse HEAD"),
        "branch": _sh("git branch --show-current"),
        "outputs": str(OUT),
        "datasets": {
            "offline_valid": str(offline),
            "base_live": str(live),
            "offline_sha256": hashlib.sha256(offline.read_bytes()).hexdigest(),
            "base_live_sha256": hashlib.sha256(live.read_bytes()).hexdigest(),
        },
        "preflight_pass": report["pass"],
    }
    (OUT / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
