#!/usr/bin/env python3
"""OPD training: vLLM rollout + HF/FSDP training backend."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from harness.config import get_config
from harness.harness_config import apply_harness_config, config_path, load_harness_config
from training.opd.hf_train_backend import HFTrainBackend, build_frozen_ref
from training.opd.rollout_worker import (
    BrowseCompRolloutWorker,
    RolloutConfig,
    load_query_records_from_json,
)
from training.opd.shadow_harness import ShadowHarness
from training.opd.trainer import OPDTrainer
from training.opd.transition_builder import build_transitions_from_rollout
from training.opd.llm_factory import build_vllm_rollout_backend_from_env, llm_api_configured, llm_manifest_fields


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OPD: vLLM rollout + HF train")
    parser.add_argument("--model-path", required=True, help="HF model path for train + tokenizer")
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--vllm-model-name", default="qwen")
    parser.add_argument("--target-module", default="verification")
    parser.add_argument("--student-config", default=str(config_path("ablate_verification.yaml")))
    parser.add_argument("--teacher-config", default=str(config_path("modules_full.yaml")))
    parser.add_argument("--dataset", default="browsecompplus")
    parser.add_argument("--split", default="train", choices=["train", "test", "rl", "sft", "all"])
    parser.add_argument("--output-dir", default="outputs/opd_verification")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--offline-shadow", action="store_true")
    parser.add_argument("--mock-rollout", action="store_true")
    parser.add_argument("--checkpoint", default=None, help="Tinker checkpoint (legacy live env)")
    parser.add_argument("--queries-json", default=None)
    parser.add_argument("--query-id-file", default=None)
    parser.add_argument("--train", action="store_true", help="Run HF OPD train after rollout")
    parser.add_argument("--enable-fsdp", action="store_true")
    return parser.parse_args()


def _load_query_ids(path: str | None) -> list[str] | None:
    if not path:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [str(x) for x in data]


def main() -> None:
    args = parse_args()
    student_cfg = load_harness_config(args.student_config)
    teacher_cfg = load_harness_config(args.teacher_config)
    apply_harness_config(student_cfg)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    student_cfg.save_resolved(output_dir / "student_resolved_config.yaml")
    teacher_cfg.save_resolved(output_dir / "teacher_resolved_config.yaml")

    query_records = None
    if args.queries_json:
        query_records = load_query_records_from_json(args.queries_json)

    rollout_config = RolloutConfig(
        dataset=args.dataset,
        split=args.split,
        limit=args.limit,
        seed=args.seed,
        query_ids=_load_query_ids(args.query_id_file),
        query_records=query_records,
        target_module=args.target_module,
    )

    transitions = []
    mode = "vllm_browsecomp"

    if args.mock_rollout or (args.checkpoint is None and not args.queries_json):
        worker = BrowseCompRolloutWorker(
            rollout_config,
            student_config=student_cfg,
            teacher_config=teacher_cfg,
            offline_shadow=True,
        )
        transitions = worker.collect_mock_transitions()
        mode = "mock_browsecomp"
        query_ids = worker.resolve_query_ids()
    elif args.checkpoint:
        # Legacy Tinker + full Harness env path
        worker = BrowseCompRolloutWorker(
            rollout_config,
            student_config=student_cfg,
            teacher_config=teacher_cfg,
            offline_shadow=args.offline_shadow,
        )
        transitions, _ = worker.collect_transitions(
            checkpoint_path=args.checkpoint, mock=False
        )
        mode = "tinker_harness"
        query_ids = worker.resolve_query_ids()
    else:
        worker = BrowseCompRolloutWorker(rollout_config, teacher_config=teacher_cfg)
        records = worker.resolve_query_records()
        query_ids = [r.query_id for r in records]

        openai_client = None
        if not args.offline_shadow:
            try:
                openai_client = get_config().get_openai_client()
            except Exception:
                pass
        shadow = ShadowHarness(
            teacher_cfg, openai_client=openai_client, offline=args.offline_shadow
        )
        vllm_rollout = build_vllm_rollout_backend_from_env(
            tokenizer_path=args.model_path,
            base_url=args.vllm_url if not llm_api_configured() else None,
            model_name=args.vllm_model_name if not llm_api_configured() else None,
        )
        transitions = build_transitions_from_rollout(
            vllm_rollout,
            records,
            shadow,
            target_module=args.target_module,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

    (output_dir / "query_ids.json").write_text(
        json.dumps(query_ids, indent=2), encoding="utf-8"
    )
    manifest = {
        "mode": mode,
        "architecture": "vllm_rollout + hf_train" if mode == "vllm_browsecomp" else mode,
        "n_transitions": len(transitions),
        "vllm_url": args.vllm_url if not llm_api_configured() else None,
        **(llm_manifest_fields() if llm_api_configured() else {}),
        "model_path": args.model_path,
    }
    (output_dir / "rollout_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))

    if not args.train:
        return

    student = HFTrainBackend(args.model_path, trainable=True)
    teacher = build_frozen_ref(args.model_path)
    if args.enable_fsdp:
        student.enable_fsdp()
        teacher.enable_fsdp()

    trainer = OPDTrainer(student=student, teacher=teacher, output_dir=output_dir)
    trainer.add_transitions(transitions)
    for epoch in range(args.epochs):
        metrics = trainer.train_epoch(batch_size=8)
        print(f"epoch={epoch} metrics={metrics}")
    print(f"Saved checkpoint to {trainer.save_checkpoint()}")


if __name__ == "__main__":
    main()
