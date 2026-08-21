#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCAPE_ROOT = Path(os.environ.get("SCAPE_ROOT", "/mnt/songzijun/Capability_Evolution/SCAPE"))
EASYOPD_ROOT = ROOT
SOURCE_COMPONENTS = ("importance_tagging", "subtractive_curation")
JOINT_COMPONENT = "importance_tagging_plus_subtractive_curation"
QWEN3_STUDENT_BASE = os.environ.get("CANONICAL_STUDENT_BASE", "/mnt/songzijun/models/Qwen3-30B-A3B-Instruct-2507")
QWEN3_LOGICAL_MODEL_ID = os.environ.get("SCAPE_STUDENT_LOGICAL_MODEL", "Qwen3-30B-A3B-Instruct-2507")

SOURCE_ROOT = EASYOPD_ROOT / "outputs" / "component_sweep_0818" / "h100_1_qwen3"
JOINT_ROOT = EASYOPD_ROOT / "outputs" / "component_sweep_0818" / "h100_1_joint_importance_subtractive"
UTILITY_EVIDENCE = SCAPE_ROOT / "outputs" / "h100_4_b_utility_confirm" / "H1004_B_UTILITY_HANDOFF.json"
PAIRWISE_EVIDENCE = SCAPE_ROOT / "outputs" / "h100_4_b_utility_confirm" / "SUBTRACTIVE_VS_IMPORTANCE.md"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_row(row: dict[str, Any], *, source_component: str, index: int) -> dict[str, Any]:
    joint_row = dict(row)
    joint_row["component"] = JOINT_COMPONENT
    joint_row["source_component"] = source_component
    joint_row["source_row_index"] = index
    joint_row["joint_component"] = JOINT_COMPONENT
    joint_row["joint_source_components"] = list(SOURCE_COMPONENTS)
    joint_row["row_id"] = f"{JOINT_COMPONENT}_{source_component}_{index:06d}_{str(row.get('state_uid', ''))[:12]}"
    return joint_row


def load_source_rows(component: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    comp_dir = SOURCE_ROOT / component
    train_path = comp_dir / "OPD_TRAIN_ROWS.jsonl"
    valid_path = comp_dir / "OPD_VALID_ROWS.jsonl"
    if not train_path.exists() or not valid_path.exists():
        raise FileNotFoundError(f"missing OPD rows for {component}: {comp_dir}")
    return read_jsonl(train_path), read_jsonl(valid_path)


def build_joint_rows() -> dict[str, Any]:
    train_rows: list[dict[str, Any]] = []
    valid_rows: list[dict[str, Any]] = []
    source_stats: dict[str, Any] = {}
    for component in SOURCE_COMPONENTS:
        src_train, src_valid = load_source_rows(component)
        source_stats[component] = {
            "train_rows": len(src_train),
            "valid_rows": len(src_valid),
            "train_unique_state_uid": len({str(r.get('state_uid')) for r in src_train}),
            "valid_unique_state_uid": len({str(r.get('state_uid')) for r in src_valid}),
        }
        train_rows.extend(normalize_row(row, source_component=component, index=i) for i, row in enumerate(src_train))
        valid_rows.extend(normalize_row(row, source_component=component, index=i) for i, row in enumerate(src_valid))
    train_rows.sort(key=lambda row: hashlib.sha256(f"20260819:{row['source_component']}:{row.get('state_uid', '')}:{row['row_id']}".encode()).hexdigest())
    valid_rows.sort(key=lambda row: hashlib.sha256(f"20260819-valid:{row['source_component']}:{row.get('state_uid', '')}:{row['row_id']}".encode()).hexdigest())
    write_jsonl(JOINT_ROOT / "JOINT_OPD_TRAIN_ROWS.jsonl", train_rows)
    write_jsonl(JOINT_ROOT / "JOINT_OPD_VALID_ROWS.jsonl", valid_rows)
    return {
        "component": JOINT_COMPONENT,
        "joint_train_rows": len(train_rows),
        "joint_valid_rows": len(valid_rows),
        "joint_unique_state_uid_train": len({str(r.get('state_uid')) for r in train_rows}),
        "joint_unique_state_uid_valid": len({str(r.get('state_uid')) for r in valid_rows}),
        "source_stats": source_stats,
    }


def joint_utility_test() -> dict[str, Any]:
    if not UTILITY_EVIDENCE.exists():
        raise FileNotFoundError(f"missing utility evidence: {UTILITY_EVIDENCE}")
    if not PAIRWISE_EVIDENCE.exists():
        raise FileNotFoundError(f"missing pairwise evidence: {PAIRWISE_EVIDENCE}")
    evidence = read_json(UTILITY_EVIDENCE)
    ranking = evidence.get("ranking") or []
    selected = {
        str(item.get("component")): item
        for item in ranking
        if str(item.get("component")) in SOURCE_COMPONENTS
    }
    missing = [comp for comp in SOURCE_COMPONENTS if comp not in selected]
    if missing:
        raise RuntimeError(f"missing utility evidence for: {', '.join(missing)}")
    joint_mean = sum(float(selected[comp]["mean_live_utility"]) for comp in SOURCE_COMPONENTS) / len(SOURCE_COMPONENTS)
    joint_noise = sum(float(selected[comp]["mean_replay_noise"]) for comp in SOURCE_COMPONENTS) / len(SOURCE_COMPONENTS)
    joint_effect_over_noise = joint_mean / max(joint_noise, 1e-12)
    passed = all(float(selected[comp]["mean_live_utility"]) > 0 for comp in SOURCE_COMPONENTS)
    return {
        "status": "JOINT_BROAD_GAIN_CONFIRMED" if passed else "JOINT_BROAD_GAIN_BLOCKED",
        "joint_component": JOINT_COMPONENT,
        "joint_mean_live_utility": joint_mean,
        "joint_mean_replay_noise": joint_noise,
        "joint_effect_over_noise": joint_effect_over_noise,
        "source_components": {
            comp: {
                "mean_live_utility": float(selected[comp]["mean_live_utility"]),
                "mean_replay_noise": float(selected[comp]["mean_replay_noise"]),
                "effect_over_noise": float(selected[comp]["effect_over_noise"]),
                "K2_K4_direction_consistent": bool(selected[comp]["K2_K4_consistent"]),
            }
            for comp in SOURCE_COMPONENTS
        },
        "evidence": {
            "utility_handoff": str(UTILITY_EVIDENCE),
            "pairwise_md": str(PAIRWISE_EVIDENCE),
        },
        "decision": "JOINT_OPD_ALLOWED" if passed else "JOINT_OPD_BLOCKED",
        "reason": "Both components are positive under the current combined B utility confirmation evidence." if passed else "Joint broad gain confirmation failed.",
    }


def train_cell(*, method: str, seed: int, train_rows: Path, valid_rows: Path, output_root: Path, gpus: str = "0,1,2,3,4,5,6,7", train_limit: int = 9000, valid_limit: int = 1000) -> subprocess.CompletedProcess[str]:
    cell_dir = output_root / JOINT_COMPONENT / f"{method}_seed{seed}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train_h1001_projectable_cell.py"),
        "--component",
        JOINT_COMPONENT,
        "--method",
        method,
        "--seed",
        str(seed),
        "--gpu",
        gpus,
        "--model",
        QWEN3_STUDENT_BASE,
        "--train",
        str(train_rows),
        "--valid",
        str(valid_rows),
        "--out",
        str(cell_dir),
        "--train-limit",
        str(train_limit),
        "--valid-limit",
        str(valid_limit),
        "--epochs",
        "1",
        "--batch-size",
        "1",
    ]
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    (cell_dir / "TRAIN.log").write_text(
        "$ " + " ".join(cmd) + "\n\nSTDOUT:\n" + result.stdout + "\nSTDERR:\n" + result.stderr,
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=JOINT_ROOT)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--train-limit", type=int, default=9000)
    parser.add_argument("--valid-limit", type=int, default=1000)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    utility = joint_utility_test()
    rows = build_joint_rows()
    gate = {
        "status": "JOINT_GAIN_GATE_PASSED" if utility["decision"] == "JOINT_OPD_ALLOWED" else "JOINT_GAIN_GATE_BLOCKED",
        "component": JOINT_COMPONENT,
        "utility": utility,
        "rows": rows,
        "canonical_student_base": QWEN3_STUDENT_BASE,
        "logical_model_id": QWEN3_LOGICAL_MODEL_ID,
        "source_components": list(SOURCE_COMPONENTS),
    }
    write_json(args.output_root / "JOINT_GAIN_TEST.json", gate)
    write_json(args.output_root / "JOINT_COMPONENT_HANDOFF.json", {
        "status": "JOINT_COMPONENT_READY" if utility["decision"] == "JOINT_OPD_ALLOWED" else "JOINT_COMPONENT_BLOCKED",
        "component": JOINT_COMPONENT,
        "canonical_student_base": QWEN3_STUDENT_BASE,
        "logical_model_id": QWEN3_LOGICAL_MODEL_ID,
        "source_components": list(SOURCE_COMPONENTS),
        "utility": utility,
        "rows": rows,
    })
    sums = []
    for path in sorted(p for p in args.output_root.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
        sums.append(f"{sha256_path(path)}  {path.relative_to(args.output_root)}")
    (args.output_root / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")

    if utility["decision"] != "JOINT_OPD_ALLOWED":
        print(json.dumps(gate, indent=2, ensure_ascii=False, sort_keys=True))
        return 3

    if args.skip_train:
        print(json.dumps(gate, indent=2, ensure_ascii=False, sort_keys=True))
        return 0

    results = []
    for method, seed in (("PURE_OPD", 42), ("PURE_OPD", 43), ("RL_PLUS_OPD", 42), ("RL_PLUS_OPD", 43)):
        result = train_cell(
            method=method,
            seed=seed,
            train_rows=args.output_root / "JOINT_OPD_TRAIN_ROWS.jsonl",
            valid_rows=args.output_root / "JOINT_OPD_VALID_ROWS.jsonl",
            output_root=args.output_root,
            gpus=args.gpus,
            train_limit=args.train_limit,
            valid_limit=args.valid_limit,
        )
        results.append(
            {
                "component": JOINT_COMPONENT,
                "method": method,
                "seed": seed,
                "returncode": result.returncode,
                "status": "completed" if result.returncode == 0 else "failed",
                "cell_dir": str(args.output_root / JOINT_COMPONENT / f"{method}_seed{seed}"),
            }
        )
        if result.returncode != 0:
            break

    write_json(args.output_root / "JOINT_COMPONENT_ROWS.json", results)
    csv_lines = ["component,method,seed,status,returncode,cell_dir"]
    for row in results:
        csv_lines.append(",".join(str(row.get(key, "")) for key in ["component", "method", "seed", "status", "returncode", "cell_dir"]))
    (args.output_root / "JOINT_COMPONENT_ROWS.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    print(json.dumps({"gate": gate, "results": results}, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if results and all(row["returncode"] == 0 for row in results) else 3


if __name__ == "__main__":
    raise SystemExit(main())
