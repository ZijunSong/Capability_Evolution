#!/usr/bin/env python3
"""Smoke test: vLLM rollout (GPU) -> HF TrainBackend OPD (GPU) — OPHSD-style split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from harness.harness_config import apply_harness_config, config_path, load_harness_config
from training.opd.hf_train_backend import HFTrainBackend, build_frozen_ref
from training.opd.rollout_worker import load_query_records_from_json
from training.opd.shadow_harness import ShadowHarness
from training.opd.trainer import OPDTrainer
from training.opd.transition_builder import build_transitions_from_rollout
from training.opd.vllm_rollout_backend import VLLMRolloutBackend
from training.opd.vllm_server import VLLMServerHandle, start_vllm_server


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="vLLM rollout + HF OPD smoke test")
    parser.add_argument(
        "--model-path",
        default="/data/ppnm/models/Qwen2.5-7B-Instruct",
    )
    parser.add_argument(
        "--queries-json",
        default=str(_REPO_ROOT / "tests/fixtures/browsecomp_sample_queries.json"),
    )
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--output-dir", default="outputs/smoke_opd_vllm_hf")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--vllm-port", type=int, default=8765)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument(
        "--vllm-url",
        default=None,
        help="Use existing vLLM server (skip subprocess launch)",
    )
    parser.add_argument("--manage-vllm", action="store_true", default=True)
    parser.add_argument("--no-manage-vllm", action="store_false", dest="manage_vllm")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--student-config", default=str(config_path("ablate_verification.yaml")))
    parser.add_argument("--teacher-config", default=str(config_path("modules_full.yaml")))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    student_cfg = load_harness_config(args.student_config)
    teacher_cfg = load_harness_config(args.teacher_config)
    apply_harness_config(student_cfg)
    student_cfg.save_resolved(output_dir / "student_resolved_config.yaml")
    teacher_cfg.save_resolved(output_dir / "teacher_resolved_config.yaml")

    records = load_query_records_from_json(args.queries_json)[: args.limit]
    print(f"[1/6] Loaded {len(records)} BrowseComp-style queries")

    vllm_handle: VLLMServerHandle | None = None
    base_url = args.vllm_url or f"http://127.0.0.1:{args.vllm_port}/v1"

    try:
        if args.vllm_url is None and args.manage_vllm:
            print(
                f"[2/6] Starting vLLM actor (TP={args.tensor_parallel_size}) on {base_url} ..."
            )
            vllm_handle = start_vllm_server(
                model_path=args.model_path,
                port=args.vllm_port,
                tensor_parallel_size=args.tensor_parallel_size,
                served_model_name="qwen",
                log_path=str(output_dir / "vllm_server.log"),
            )
            base_url = vllm_handle.base_url
            print(f"[2/6] vLLM ready at {base_url}")
        else:
            print(f"[2/6] Using existing vLLM at {base_url}")

        rollout = VLLMRolloutBackend(
            base_url=base_url,
            model_name="qwen",
            tokenizer_path=args.model_path,
        )
        shadow = ShadowHarness(teacher_cfg, offline=True)
        transitions = build_transitions_from_rollout(
            rollout,
            records,
            shadow,
            target_module="verification",
            max_new_tokens=args.max_new_tokens,
        )
        print(f"[3/6] vLLM rollout -> {len(transitions)} OPD transitions")
        for t in transitions:
            print(
                f"  - {t.query_id}: action_tokens={len(t.action_ids)} "
                f"preview={t.metadata.get('action_text', '')[:80]!r}"
            )

        manifest = {
            "architecture": "vllm_rollout + hf_train",
            "model_path": args.model_path,
            "vllm_url": base_url,
            "n_queries": len(records),
            "n_transitions": len(transitions),
            "visible_gpus": torch.cuda.device_count(),
        }
        (output_dir / "smoke_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        if vllm_handle is not None:
            print("[4/6] Stopping vLLM to free GPUs for HF training worker ...")
            vllm_handle.stop()
            vllm_handle = None
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            print("[4/6] Keeping external vLLM running (training may OOM if colocated)")

        if args.skip_train:
            print("[5/6] Skipped HF training (--skip-train)")
            return

        visible = torch.cuda.device_count()
        print(f"[5/6] Loading HF TrainBackend on {visible} visible GPU(s) ...")
        student = HFTrainBackend(args.model_path, device_map="auto", trainable=True)
        teacher = build_frozen_ref(args.model_path, device_map="auto")

        trainer = OPDTrainer(student=student, teacher=teacher, output_dir=output_dir)
        trainer.add_transitions(transitions)
        metrics = trainer.train_epoch(batch_size=len(transitions), weighting="uniform")
        print(f"[6/6] OPD train_step metrics: {metrics}")
        ckpt = trainer.save_checkpoint("smoke_checkpoint.json")
        print(f"Saved smoke artifact to {ckpt}")

    finally:
        if vllm_handle is not None:
            vllm_handle.stop()


if __name__ == "__main__":
    main()
