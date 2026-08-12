#!/usr/bin/env python3
"""Aggregate S0/S1 (LOO) + S2/S3 (closed-loop) into FOUR_GRID + GATE_S_B."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

SCAPE = Path("/data/ppnm/Capability_Evolution/SCAPE")
OUT = SCAPE / "outputs/stage_s/B_verify_fourgrid"
LOO = SCAPE / "outputs/local_cal64_loo"
sys.path.insert(0, str(SCAPE))


def load_quality(job_dir: Path) -> dict[str, float]:
    p = job_dir / "harness_rollouts.jsonl"
    rows: dict[str, float] = {}
    if not p.exists():
        return rows
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("error") in (True, "True", 1) or (isinstance(r.get("error"), str) and r.get("error").strip()):
            continue
        q = str(r.get("query_id") or r.get("qid"))
        m = r.get("metrics") or r
        rows[q] = float(m.get("curated_recall") or m.get("recall") or m.get("harness_reward") or 0.0)
    return rows


def mean(d: dict[str, float], ids: list[str]) -> float:
    return sum(d[i] for i in ids) / len(ids) if ids else 0.0


def cost_proxy(job_dir: Path, default: float) -> float:
    p = job_dir / "harness_rollouts.jsonl"
    if not p.exists():
        return default
    costs = []
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        m = r.get("metrics") or r
        for k in ("n_tool_calls", "tool_calls", "n_turns", "turns"):
            if k in m and m[k] is not None:
                costs.append(float(m[k]))
                break
    return sum(costs) / len(costs) if costs else default


def ready(job: str) -> bool:
    d = OUT / job
    return (d / "DONE").exists() and (d / "harness_rollouts.jsonl").exists()


def aggregate() -> dict:
    from scape.eval.retirement import evaluate_gate_s

    s0 = load_quality(LOO / "full")
    s1 = load_quality(LOO / "minus_verify_tool")
    s2 = load_quality(OUT / "S2_trained_minus_verify")
    s3 = load_quality(OUT / "S3_trained_full")
    shared = sorted(set(s0) & set(s1) & set(s2) & set(s3))
    if len(shared) < 32:
        raise SystemExit(f"insufficient shared ids: {len(shared)}")
    grid = {
        "S0": {"quality": mean(s0, shared), "cost": cost_proxy(LOO / "full", 10.0), "label": "theta0+H_full", "source": "loo"},
        "S1": {"quality": mean(s1, shared), "cost": cost_proxy(LOO / "minus_verify_tool", 7.0), "label": "theta0+H_-verify", "source": "loo"},
        "S2": {"quality": mean(s2, shared), "cost": cost_proxy(OUT / "S2_trained_minus_verify", 7.0), "label": "theta'+H_-verify", "source": "closed_loop_L64_seed42_hf"},
        "S3": {"quality": mean(s3, shared), "cost": cost_proxy(OUT / "S3_trained_full", 10.0), "label": "theta'+H_full", "source": "closed_loop_L64_seed42_hf"},
        "n_shared": len(shared),
        "verdict": "CLOSED_LOOP",
        "h100_required": False,
        "student_ckpt": str(SCAPE / "outputs/stage_l/B_verify_opd_provisional/L64_seed42_hf/hf_model"),
        "note": "S0/S1 LOCAL_CAL64 LOO; S2/S3 closed-loop on OPD HF weights (provisional BM25+Qwen)",
    }
    # evaluate_gate_s expects nested quality/cost
    gs = evaluate_gate_s(
        {k: {"quality": grid[k]["quality"], "cost": grid[k]["cost"]} for k in ("S0", "S1", "S2", "S3")},
        non_inferior_tol=0.02,
        material_cost_reduction=0.05,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "FOUR_GRID.json").write_text(json.dumps(grid, indent=2) + "\n")
    (OUT / "GATE_S_B.json").write_text(json.dumps(gs, indent=2) + "\n")
    (OUT / "CLOSED_LOOP_DONE").write_text("ok\n")
    status = (
        f"# CLOSED_LOOP_STATUS — B verify four-grid (S2/S3)\n\n"
        f"- updated: {__import__('datetime').datetime.now().isoformat(timespec='seconds')}\n"
        f"- verdict: **CLOSED_LOOP**\n"
        f"- n_shared: {len(shared)}\n"
        f"- GATE_S: {gs.get('verdict')} pass={gs.get('pass')}\n"
        f"- student: L64_seed42_hf/hf_model\n"
    )
    (OUT / "CLOSED_LOOP_STATUS.md").write_text(status)
    print(json.dumps({"grid": grid, "gate_s": gs}, indent=2))
    return {"grid": grid, "gate_s": gs}


def main() -> None:
    deadline = time.time() + 12 * 3600
    while time.time() < deadline:
        if ready("S2_trained_minus_verify") and ready("S3_trained_full"):
            aggregate()
            return
        s2n = len((OUT / "S2_trained_minus_verify/harness_rollouts.jsonl").read_text().splitlines()) if (OUT / "S2_trained_minus_verify/harness_rollouts.jsonl").exists() else 0
        s3n = len((OUT / "S3_trained_full/harness_rollouts.jsonl").read_text().splitlines()) if (OUT / "S3_trained_full/harness_rollouts.jsonl").exists() else 0
        print(f"[wait] S2={s2n}/64 DONE={ready('S2_trained_minus_verify')} S3={s3n}/64 DONE={ready('S3_trained_full')}", flush=True)
        time.sleep(120)
    raise SystemExit("timeout waiting for S2/S3")


if __name__ == "__main__":
    main()
