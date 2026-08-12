#!/usr/bin/env python3
"""Sequential OPD: vLLM rollout -> kill vLLM -> HF train + save_checkpoint (HF weights).

Mirrors smoke_opd_vllm_hf.py lifecycle, but supports BrowseComp-Plus L64 resolution
and SCAPE stage-L B verify defaults.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

_SCAPE = Path(__file__).resolve().parents[1]
_SCOPE = _SCAPE.parent / "SCOPE"
if str(_SCOPE) not in sys.path:
    sys.path.insert(0, str(_SCOPE))

import torch

from harness.harness_config import apply_harness_config, config_path, load_harness_config
from training.opd.hf_train_backend import HFTrainBackend, build_frozen_ref
from training.opd.rollout_worker import (
    RolloutConfig,
    load_query_records_from_json,
    resolve_query_records,
)
from training.opd.shadow_harness import ShadowHarness
from training.opd.trainer import OPDTrainer
from training.opd.transition_builder import build_transitions_from_rollout
from training.opd.vllm_rollout_backend import VLLMRolloutBackend
from training.opd.vllm_server import VLLMServerHandle, start_vllm_server


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sequential vLLM rollout + HF OPD with weight save"
    )
    p.add_argument("--model-path", default="/data/ppnm/models/Qwen2.5-7B-Instruct")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--limit", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset", default="browsecompplus")
    p.add_argument("--split", default="train")
    p.add_argument(
        "--queries-json",
        default=None,
        help="Optional JSON list; else resolve from dataset",
    )
    p.add_argument("--target-module", default="verification")
    p.add_argument(
        "--student-config", default=str(config_path("ablate_verification.yaml"))
    )
    p.add_argument(
        "--teacher-config", default=str(config_path("modules_full.yaml"))
    )
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--vllm-port", type=int, default=8772)
    p.add_argument("--tensor-parallel-size", type=int, default=4)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--offline-shadow", action="store_true", default=True)
    p.add_argument("--no-offline-shadow", action="store_false", dest="offline_shadow")
    p.add_argument("--skip-train", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    student_cfg = load_harness_config(args.student_config)
    teacher_cfg = load_harness_config(args.teacher_config)
    apply_harness_config(student_cfg)
    student_cfg.save_resolved(output_dir / "student_resolved_config.yaml")
    teacher_cfg.save_resolved(output_dir / "teacher_resolved_config.yaml")

    if args.queries_json:
        records = load_query_records_from_json(args.queries_json)[: args.limit]
    else:
        cfg = RolloutConfig(
            dataset=args.dataset,
            split=args.split,
            limit=args.limit,
            seed=args.seed,
            target_module=args.target_module,
        )
        records = resolve_query_records(cfg)

    query_payload = [{"query_id": r.query_id, "query": r.query} for r in records]
    (output_dir / "queries_l64.json").write_text(
        json.dumps(query_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "query_ids.json").write_text(
        json.dumps([r.query_id for r in records], indent=2), encoding="utf-8"
    )
    print(f"[1/6] Loaded {len(records)} queries (limit={args.limit}, seed={args.seed})")

    vllm_handle: VLLMServerHandle | None = None
    base_url = f"http://127.0.0.1:{args.vllm_port}/v1"
    try:
        print(f"[2/6] Starting vLLM TP={args.tensor_parallel_size} on {base_url} ...")
        vllm_handle = start_vllm_server(
            model_path=args.model_path,
            port=args.vllm_port,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            served_model_name="qwen",
            log_path=str(output_dir / "vllm_server.log"),
            enforce_eager=True,
            enable_auto_tool_choice=True,
            tool_call_parser="hermes",
        )
        base_url = vllm_handle.base_url
        print(f"[2/6] vLLM ready at {base_url}")

        rollout = VLLMRolloutBackend(
            base_url=base_url,
            model_name="qwen",
            tokenizer_path=args.model_path,
        )
        shadow = ShadowHarness(teacher_cfg, offline=args.offline_shadow)
        transitions = build_transitions_from_rollout(
            rollout,
            records,
            shadow,
            target_module=args.target_module,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        print(f"[3/6] vLLM rollout -> {len(transitions)} OPD transitions")

        manifest = {
            "mode": "vllm_browsecomp_sequential",
            "architecture": "vllm_rollout + hf_train",
            "model_path": args.model_path,
            "vllm_url": base_url,
            "n_queries": len(records),
            "n_transitions": len(transitions),
            "target_module": args.target_module,
            "seed": args.seed,
            "visible_gpus": torch.cuda.device_count(),
        }
        (output_dir / "rollout_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        print(json.dumps(manifest, indent=2))

        print("[4/6] Stopping vLLM to free GPUs for HF training ...")
        vllm_handle.stop()
        vllm_handle = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if args.skip_train:
            print("[5/6] Skipped HF training (--skip-train)")
            return

        visible = torch.cuda.device_count()
        print(f"[5/6] Loading HF TrainBackend on {visible} visible GPU(s) ...")
        student = HFTrainBackend(args.model_path, device_map="auto", trainable=True)
        teacher = build_frozen_ref(args.model_path, device_map="auto")
        trainer = OPDTrainer(student=student, teacher=teacher, output_dir=output_dir)
        trainer.add_transitions(transitions)
        for epoch in range(args.epochs):
            metrics = trainer.train_epoch(batch_size=args.batch_size)
            print(f"epoch={epoch} metrics={metrics}")
        ckpt = trainer.save_checkpoint("checkpoint.json")
        print(f"[6/6] Saved checkpoint to {ckpt}")
        hf_dir = output_dir / "hf_model"
        if hf_dir.is_dir():
            (output_dir / "DONE").write_text("ok\n", encoding="utf-8")
            print(f"[6/6] HF weights at {hf_dir}")
        else:
            raise SystemExit(f"hf_model missing under {output_dir}")
    finally:
        if vllm_handle is not None:
            vllm_handle.stop()


if __name__ == "__main__":
    main()
