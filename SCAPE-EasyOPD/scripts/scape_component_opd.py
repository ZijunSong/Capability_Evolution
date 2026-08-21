#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from easyopd import EasyOPD
from easyopd.methods.scape_component_opd.action_projection import project_curated_delta
from easyopd.methods.scape_component_opd.event_collection import collect_event_states, generate_real_harness_rollouts
from easyopd.methods.scape_component_opd.harness1_bridge import QWEN3_LOGICAL_MODEL_ID, QWEN3_STUDENT_BASE
from easyopd.methods.scape_component_opd.component_registry import audit_component, get_component_spec
from easyopd.methods.scape_component_opd.controls import assert_query_disjoint, query_disjoint_splits, shuffled_targets_preserve_marginal
from easyopd.methods.scape_component_opd.core import SCAPEComponentOPD
from easyopd.methods.scape_component_opd.diagnostics import write_event_support_csv, write_sha256sums
from easyopd.methods.scape_component_opd.scape_agent_loop import SCAPEAgentLoop
from easyopd.methods.scape_component_opd.real_closed_loop_evaluator import SCAPERealClosedLoopEvaluator
from easyopd.methods.scape_component_opd.state_snapshot import SCAPEStateSnapshot, assert_same_state_before_component_fork
from easyopd.methods.scape_component_opd.tool_span import require_parsable_tool_calls

DEFAULT_OUT = ROOT / "outputs" / "scape_easyopd"
DEFAULT_CONFIG = ROOT / "easyopd" / "config" / "scape_component_opd.yaml"
SCAPE_CONFIG_DIR = ROOT / "easyopd" / "config" / "scape"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def config_path_for(args: argparse.Namespace) -> str:
    if args.config:
        return str(args.config)
    component_path = SCAPE_CONFIG_DIR / f"{args.component}.yaml"
    if component_path.exists():
        return str(component_path)
    return str(DEFAULT_CONFIG)


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    method = args.component_method or "scape_component_opd"
    instance = EasyOPD.from_hparams(
        method,
        config_path=config_path_for(args),
        auto_resolve_data=False,
    )
    config = dict(instance.config)
    config.setdefault("method", {"name": method})
    config.setdefault("component", {"name": args.component})
    config.setdefault("runtime", {})
    config["runtime"]["output_dir"] = str(args.output_dir) if args.output_dir is not None else config["runtime"].get("output_dir")
    config.setdefault("actor_rollout_ref", {})
    config["actor_rollout_ref"].setdefault("model", {})
    config["actor_rollout_ref"]["model"]["lora_rank"] = 8
    config["actor_rollout_ref"]["model"]["lora_alpha"] = 16
    config["actor_rollout_ref"]["model"]["target_modules"] = "[\"q_proj\",\"k_proj\",\"v_proj\",\"o_proj\"]"
    return config


def sample_rows(component: str, seed: int) -> list[dict[str, Any]]:
    rows = []
    for i in range(4):
        pre = [f"d{i}_0"]
        post = [f"d{i}_0", f"d{i}_1"] if component == "auto_populate_first_search" else pre
        visible = [f"d{i}_0", f"d{i}_1", f"d{i}_2"]
        action, audit = project_curated_delta(curated_ids_pre=pre, curated_ids_post=post, visible_doc_ids=visible)
        rows.append(
            {
                "query_id": f"q{seed}_{i}",
                "component_event_active": action is not None,
                "projection_valid": bool(audit["projection_valid"]),
                "visibility_valid": bool(audit["visible_valid"]),
                "projected_action": action.to_dict() if action else None,
                "visible_doc_ids": visible,
                "curated_ids_pre": pre,
                "curated_ids_post": post,
                "terminal_reward": 1.0 if action else 0.0,
            }
        )
    return rows


def cmd_list(_args: argparse.Namespace) -> int:
    print(json.dumps(SCAPEComponentOPD.list_components(), indent=2, ensure_ascii=False))
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    payload = audit_component(args.component, event_support=args.event_support, student_has_tool=args.student_has_verify_tool)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not payload["can_train"] and not args.allow_refusal:
        return 2
    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    if args.mode == "formal":
        if args.runtime != "harness1":
            raise SystemExit("STOP_REAL_HARNESS_RUNTIME_UNAVAILABLE: formal collect requires --runtime harness1")
        if args.student_base != QWEN3_STUDENT_BASE:
            raise SystemExit(f"STOP_QWEN3_CHAT_CONTRACT_FAILED: canonical student base must be {QWEN3_STUDENT_BASE}")
        if not args.query_manifest:
            raise SystemExit("formal collect requires --query-manifest")
        out = args.collection_output_dir or (args.output_dir / "components" / args.component)
        rollout_manifest = args.rollout_manifest or (out / "ROLLOUT_MANIFEST.jsonl")
        rollout_generation = None
        if not rollout_manifest.exists():
            rollout_generation = generate_real_harness_rollouts(
                component=args.component,
                query_manifest=args.query_manifest,
                output_path=rollout_manifest,
                query_max=args.query_max,
                rollouts_max=args.rollouts_max,
                seed_base=args.selection_seed,
            )
        stats = collect_event_states(
            component=args.component,
            query_manifest=args.query_manifest,
            rollout_manifest=rollout_manifest,
            output_dir=out,
            query_min=args.query_min,
            query_max=args.query_max,
            rollouts_min=args.rollouts_min,
            rollouts_max=args.rollouts_max,
            target_unique_states=args.target_unique_event_states,
            selection_seed=args.selection_seed,
            require_real_harness=True,
        )
        if rollout_generation is not None:
            stats["rollout_generation"] = rollout_generation
            write_json(out / "DATA_STATS.json", stats)
        manifest = {
            "cmd": "collect",
            "component": args.component,
            "mode": args.mode,
            "runtime": args.runtime,
            "student_base": args.student_base,
            "logical_model_id": QWEN3_LOGICAL_MODEL_ID,
            "dry_run": False,
            "event_support": stats,
            "paper_grade": True,
            "synthetic_fallback": False,
        }
        write_json(out / "RUN_MANIFEST.json", manifest)
        print(json.dumps({"output_dir": str(out), "event_support": stats, "dry_run": False, "mode": args.mode}, indent=2, ensure_ascii=False))
        return 0 if stats["collection_status"] == "READY_5K" else 3

    out = args.output_dir / "components" / args.component / f"collect_seed{args.seed}"
    rows = sample_rows(args.component, args.seed)
    if args.component == "content_dedup":
        rows = [dict(r, component_event_active=False, projection_valid=False, visibility_valid=False, terminal_reward=0.0) for r in rows]
    metrics = {
        "n_queries": len({r["query_id"] for r in rows}),
        "n_states": len(rows),
        "n_event_active": sum(bool(r.get("component_event_active")) for r in rows),
        "event_rate": sum(bool(r.get("component_event_active")) for r in rows) / max(1, len(rows)),
        "n_projectable": sum(bool(r.get("projection_valid")) for r in rows),
        "n_valid_args": sum(bool(r.get("visibility_valid")) for r in rows),
        "n_terminal_reward": sum((r.get("terminal_reward") or 0) != 0 for r in rows),
    }
    if not args.dry_run:
        metrics = write_event_support_csv(out / "EVENT_SUPPORT.csv", rows)
        with (out / "TRANSITIONS.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        write_json(
            out / "RUN_MANIFEST.json",
            {"cmd": "collect", "component": args.component, "seed": args.seed, "dry_run": args.dry_run, "event_support": metrics, "paper_grade": False, "mode": args.mode, "synthetic_fallback": True},
        )
        write_sha256sums(out)
    print(json.dumps({"output_dir": str(out), "event_support": metrics, "dry_run": args.dry_run, "mode": args.mode}, indent=2, ensure_ascii=False))
    return 0


def train_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {
        "component": {"name": args.component},
        "seed": args.seed,
    }
    if args.python_bin:
        overrides.setdefault("runtime", {})["python_bin"] = args.python_bin
    aro = overrides.setdefault("actor_rollout_ref", {})
    actor = aro.setdefault("actor", {})
    actor.setdefault("ppo_mini_batch_size", args.train_batch_size)
    actor.setdefault("ppo_micro_batch_size_per_gpu", 1)
    actor.setdefault("optim", {}).setdefault("lr", 1e-6)
    ref = aro.setdefault("ref", {})
    ref.setdefault("log_prob_micro_batch_size_per_gpu", 1)
    rollout = aro.setdefault("rollout", {})
    rollout.setdefault("name", args.rollout_name)
    rollout.setdefault("tensor_model_parallel_size", args.rollout_tp)
    rollout.setdefault("gpu_memory_utilization", args.rollout_gpu_memory_utilization)
    rollout.setdefault("n", 1)
    rollout.setdefault("max_num_seqs", args.train_batch_size)
    rollout.setdefault("max_num_batched_tokens", args.max_prompt_length + args.max_response_length)
    rollout.setdefault("log_prob_micro_batch_size_per_gpu", 1)
    rollout.setdefault("enforce_eager", True)
    if args.student_model:
        model_cfg = overrides.setdefault("actor_rollout_ref", {}).setdefault("model", {})
        model_cfg["path"] = args.student_model
        model_cfg["trust_remote_code"] = True
        model_cfg.setdefault("override_config", {})["_attn_implementation"] = "eager"
        model_cfg.setdefault("lora_rank", 8)
        model_cfg.setdefault("lora_alpha", 16)
        # Qwen3-MoE router modules are not valid LoRA targets; keep the default
        # target set aligned with the successful pilot scripts.
        model_cfg.setdefault("target_modules", '["q_proj","k_proj","v_proj","o_proj"]')
    if args.teacher_model:
        dist = overrides.setdefault("distillation", {})
        dist["enabled"] = True
        dist["n_gpus_per_node"] = args.teacher_gpus
        dist["nnodes"] = 1
        dist.setdefault("teacher_models", {}).setdefault("teacher_model", {})["model_path"] = args.teacher_model
    if args.train_file or args.val_file:
        data = overrides.setdefault("data", {})
        if args.train_file:
            data["train_files"] = [args.train_file]
            if not args.val_file:
                candidate_val = str(Path(args.train_file).with_name("OPD_VALID_ROWS.parquet"))
                data["val_files"] = [candidate_val]
        if args.val_file:
            data["val_files"] = [args.val_file]
        data.setdefault("train_batch_size", args.train_batch_size)
        data.setdefault("max_prompt_length", args.max_prompt_length)
        data.setdefault("max_response_length", args.max_response_length)
        data.setdefault("prompt_key", args.prompt_key)
        data.setdefault("truncation", "right")
    overrides.setdefault("algorithm", {})["adv_estimator"] = args.adv_estimator
    if args.disable_critic:
        overrides.setdefault("critic", {})["enable"] = False
    trainer = overrides.setdefault("trainer", {})
    trainer.setdefault("n_gpus_per_node", args.gpus)
    trainer.setdefault("nnodes", 1)
    trainer.setdefault("total_training_steps", args.total_training_steps)
    trainer.setdefault("val_before_train", False)
    trainer.setdefault("save_freq", 999)
    trainer.setdefault("test_freq", 999)
    trainer.setdefault("project_name", "scape_easyopd")
    trainer.setdefault("experiment_name", f"{args.component}_seed{args.seed}")
    trainer["logger"] = ["console"]
    ray_init = overrides.setdefault("ray_kwargs", {}).setdefault("ray_init", {})
    ray_init["include_dashboard"] = False
    runtime_env = ray_init.setdefault("runtime_env", {}).setdefault("env_vars", {})
    runtime_env["PYTHONPATH"] = f"{ROOT}:{ROOT / 'verl'}:{runtime_env.get('PYTHONPATH', '')}"
    if args.reward_path:
        reward = overrides.setdefault("custom_reward_function", {})
        reward["path"] = args.reward_path
        reward["name"] = args.reward_name
    return overrides


def cmd_train(args: argparse.Namespace) -> int:
    config = load_config(args)
    audit = audit_component(args.component, event_support=args.event_support, student_has_tool=args.student_has_verify_tool)
    out = args.output_dir / "components" / args.component / f"train_seed{args.seed}"
    if not audit["can_train"]:
        write_json(out / "RUN_MANIFEST.json", {"cmd": "train", "component": args.component, "status": "refused", "decision_code": audit["decision_code"], "paper_grade": False})
        print(audit["decision_code"])
        return 2
    instance = EasyOPD.from_hparams("scape_component_opd", config_path=config_path_for(args), auto_resolve_data=False)
    extra = train_overrides(args)
    result = instance.train(dry_run=args.dry_run, output_dir=str(args.output_dir), extra_args=extra)
    write_json(out / "RUN_MANIFEST.json", {"cmd": "train", "component": args.component, "seed": args.seed, "dry_run": args.dry_run, "paper_grade": False, "config_keys": sorted(config.keys()), "overrides": extra})
    if args.dry_run:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if isinstance(result, subprocess.CompletedProcess):
        return result.returncode
    if hasattr(result, "returncode"):
        return int(result.returncode)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    out = args.output_dir / "acceptance" / f"eval_{args.component}_seed{args.seed}"
    evaluator = SCAPERealClosedLoopEvaluator(component_name=args.component, split=args.split, max_steps=4, student_inference_privilege=False)
    if args.dry_run:
        write_json(out / "RUN_MANIFEST.json", {"cmd": "eval", "component": args.component, "split": args.split, "status": "dry_run", "real_closed_loop": True, "paper_grade": False})
        print(json.dumps({"output_dir": str(out), "real_closed_loop": True, "dry_run": True}, indent=2))
        return 0
    handoff = evaluator.evaluate(output_dir=out)
    write_json(out / "RUN_MANIFEST.json", {"cmd": "eval", "component": args.component, "split": args.split, "status": "completed", "real_closed_loop": True, "paper_grade": True, "handoff": handoff})
    print(json.dumps({"output_dir": str(out), "real_closed_loop": True, "paper_grade": True, "handoff": handoff}, indent=2))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    loop = SCAPEAgentLoop(args.component, student_inference_privilege=False)
    snap = SCAPEStateSnapshot(query_id="q-smoke", turn_id=0, curated_ids=["d0"], documents=[{"id": "d0", "text": "x"}], component_masks={args.component: False})
    state_hashes = assert_same_state_before_component_fork(snap)
    require_parsable_tool_calls(['to=curate\n{"add_ids":["d1"],"remove_ids":[]}\n</tool_call>'])
    splits = query_disjoint_splits([f"q{i}" for i in range(20)], seed=args.seed)
    assert_query_disjoint(splits)
    shuffled_targets_preserve_marginal([{"projected_action": {"name": "curate", "arguments": {"add_ids": [str(i)], "remove_ids": []}}} for i in range(3)])
    out = args.output_dir / "acceptance" / f"run_{args.component}_seed{args.seed}"
    live = None
    if not args.dry_run:
        live = loop.run_live_search(query_id=f"q-live-{args.seed}", query="What evidence supports SCAPE component OPD?", output_dir=out)
    write_json(out / "RUN_MANIFEST.json", {"cmd": "run", "component": args.component, "dry_run": args.dry_run, "scape_runtime_available": loop.scape_runtime_available(), "live_multi_turn_search_pass": live is not None, "live_tool_calls": live.tool_calls if live else [], "state_hashes": state_hashes, "student_inference_privilege": False, "paper_grade": live is not None})
    write_sha256sums(out)
    print(json.dumps({"output_dir": str(out), "state_fork_pass": True, "live_multi_turn_search_pass": live is not None, "tool_calls": live.tool_calls if live else [], "paper_grade": live is not None}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--component", default="evidence_graph")
    common.add_argument("--component-method", default="scape_component_opd")
    common.add_argument("--config", default=None)
    common.add_argument("--dry-run", action="store_true")
    common.add_argument("--seed", type=int, default=8183)
    common.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    common.add_argument("--resume", action="store_true")
    sub.add_parser("list-components")
    a = sub.add_parser("audit", parents=[common]); a.add_argument("--event-support", type=int); a.add_argument("--student-has-verify-tool", action="store_true"); a.add_argument("--allow-refusal", action="store_true")
    c = sub.add_parser("collect", parents=[common])
    c.add_argument("--mode", choices=["formal", "smoke"], default="formal")
    c.add_argument("--runtime", default="harness1")
    c.add_argument("--student-base", default=QWEN3_STUDENT_BASE)
    c.add_argument("--query-manifest", "--query-pool", dest="query_manifest", type=Path, default=None)
    c.add_argument("--rollout-manifest", type=Path, default=None)
    c.add_argument("--collection-output-dir", type=Path, default=None)
    c.add_argument("--event-conditioned", action="store_true")
    c.add_argument("--query-min", type=int, default=1000)
    c.add_argument("--query-max", type=int, default=2000)
    c.add_argument("--rollouts-min", type=int, default=2)
    c.add_argument("--rollouts-max", type=int, default=4)
    c.add_argument("--target-unique-event-states", type=int, default=5000)
    c.add_argument("--selection-seed", type=int, default=20260818)
    t = sub.add_parser("train", parents=[common]); t.add_argument("--loss", default=None); t.add_argument("--event-support", type=int); t.add_argument("--student-has-verify-tool", action="store_true"); t.add_argument("--python-bin", default=None); t.add_argument("--student-model", default=QWEN3_STUDENT_BASE); t.add_argument("--teacher-model", default=QWEN3_STUDENT_BASE); t.add_argument("--train-file", default=None); t.add_argument("--val-file", default=None); t.add_argument("--prompt-key", default="prompt"); t.add_argument("--train-batch-size", type=int, default=1); t.add_argument("--max-prompt-length", type=int, default=512); t.add_argument("--max-response-length", type=int, default=128); t.add_argument("--gpus", type=int, default=8); t.add_argument("--teacher-gpus", type=int, default=1); t.add_argument("--total-training-steps", type=int, default=1); t.add_argument("--reward-path", default=str(ROOT / "examples" / "simple" / "reward.py")); t.add_argument("--reward-name", default="compute_score"); t.add_argument("--rollout-name", default="vllm"); t.add_argument("--rollout-tp", type=int, default=1); t.add_argument("--rollout-gpu-memory-utilization", type=float, default=0.3); t.add_argument("--adv-estimator", default="grpo"); t.add_argument("--disable-critic", action=argparse.BooleanOptionalAction, default=True)
    e = sub.add_parser("eval", parents=[common]); e.add_argument("--split", default="dev")
    r = sub.add_parser("run", parents=[common])
    args = parser.parse_args()
    if getattr(args, "component", None):
        get_component_spec(args.component)
    return {"list-components": cmd_list, "audit": cmd_audit, "collect": cmd_collect, "train": cmd_train, "eval": cmd_eval, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
