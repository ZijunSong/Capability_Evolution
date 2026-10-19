"""Build the immutable contract from CLI args and content digests."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any

from trim.adapters.components import zero_mask
from trim.training.sft_ablation.contracts import (
    LOSS_VERSION,
    WEIGHT_NORMALIZATION,
    WEIGHT_POLICY,
    fingerprint,
    immutable_identity,
    profile_record,
)
from trim.training.sft_ablation.geometry import OFFICIAL_NPROC_PER_NODE
from trim.training.sft_ablation.io_util import sha256_file, sha256_text, sha256_tree
from trim.upstream_harness1.pin import HARNESS1_ROOT
from trim.upstream_harness1.v8d_flags import all_enabled_mask


def _digest_path(path: str) -> str:
    target = Path(path) if path else None
    if target is None or not str(path):
        return "none"
    if not target.exists():
        return "missing"
    return sha256_tree(target)


def _mask_digest(mask: dict[str, bool]) -> str:
    return fingerprint(mask)


def tool_schema_digest(mask: dict[str, bool]) -> str:
    from trim.training.tool_mask import ACTION_ARG_SCHEMAS, legal_tool_names

    names = legal_tool_names(harness_mask=mask)
    payload = {name: sorted(ACTION_ARG_SCHEMAS.get(name, ())) for name in names}
    if mask.get("importance_tagging") or mask.get("verify_tool"):
        payload["curate"] = sorted(set(payload.get("curate", [])) | {"importance"})
    return fingerprint(payload)


def harness_source_digest() -> str:
    files = [
        HARNESS1_ROOT / "harness" / "ultra_core.py",
        HARNESS1_ROOT / "training" / "train_rl.py",
    ]
    rows = []
    for path in files:
        rows.append(f"{path.name}:{'missing' if not path.is_file() else sha256_file(path)}")
    return sha256_text("\n".join(rows))


def projector_digest() -> str:
    root = Path(__file__).resolve().parents[3]
    files = [
        root / "trim" / "training" / "opd_projection.py",
        root.parent / "SCAPE-EasyOPD" / "easyopd" / "methods" / "scape_component_opd" / "skip_to_anchor.py",
    ]
    rows = []
    for path in files:
        rows.append(f"{path.name}:{'missing' if not path.is_file() else sha256_file(path)}")
    return sha256_text("\n".join(rows))


def tokenizer_digest(model_path: str) -> str:
    root = Path(model_path) if model_path else None
    if root is None or not root.exists():
        return "missing"
    names = (
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
    )
    rows = [f"{name}:{sha256_file(root / name)}" for name in names if (root / name).is_file()]
    if not rows:
        return "missing"
    return sha256_text("\n".join(rows))


def build_contract(
    args: Namespace,
    *,
    query_order_digest: str,
    corpus_digest: str,
    retrieval_index_digest: str,
    teacher_archive_digest: str,
    nnodes: int,
    rollout_replicas: int,
    model_family: str = "pending",
    prompt_template_digest: str = "pending",
) -> dict[str, Any]:
    experiment = str(args.experiment)
    teacher_mask = all_enabled_mask("Harness-1")
    student_mask = zero_mask("Harness-1")
    teacher_model = str(getattr(args, "teacher_model", "") or args.model_name)
    teacher_adapter = str(getattr(args, "teacher_adapter", "") or args.init_adapter or "")
    scan = int(getattr(args, "max_anchor_scan_events", 8) or 8)
    sampling = "all_eligible_actions" if int(getattr(args, "states_per_trajectory", -1)) < 0 else "capped"
    record = {
        **profile_record(experiment),
        "weight_policy": WEIGHT_POLICY[experiment],
        "weight_normalization": WEIGHT_NORMALIZATION,
        "loss": LOSS_VERSION,
        "base_checkpoint_digest": _digest_path(args.model_name),
        "init_adapter_digest": _digest_path(args.init_adapter),
        "teacher_policy_digest": "none"
        if experiment == "student_onpolicy_sft"
        else fingerprint(
            {"model": _digest_path(teacher_model), "adapter": _digest_path(teacher_adapter)}
        ),
        "lora_target": str(args.lora_target),
        "lora_rank": int(args.lora_rank),
        "lora_alpha": int(args.lora_alpha),
        "lora_dropout": float(args.lora_dropout),
        "dtype": str(args.dtype),
        "query_order_digest": query_order_digest,
        "train_split": str(args.train_data),
        "corpus_digest": corpus_digest,
        "retrieval_index_digest": retrieval_index_digest,
        "teacher_archive_digest": teacher_archive_digest,
        "train_env": str(args.train_env),
        "teacher_mask_digest": _mask_digest(teacher_mask),
        "student_mask_digest": _mask_digest(student_mask),
        "tool_schema_digest": tool_schema_digest(teacher_mask if experiment == "teacher_sft" else student_mask),
        "harness_source_digest": harness_source_digest(),
        "verifier_config_version": sha256_text(str(args.verifier or "upstream-default"))[:16],
        "tokenizer_digest": tokenizer_digest(args.model_name),
        "model_family": model_family,
        "prompt_template_digest": prompt_template_digest,
        "max_model_len": int(args.max_model_len),
        "max_new_tokens": int(args.max_new_tokens),
        "action_only_supervision": True,
        "batch_size": int(args.batch_size),
        "micro_batch_size": int(args.micro_batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "optimizer": str(args.optimizer),
        "scheduler": str(args.scheduler),
        "seed": int(args.seed),
        "projector_digest": projector_digest(),
        "state_sampling_rule": sampling,
        "branch_selection": "trim_component_branches_v1"
        if experiment == "student_onpolicy_sft"
        else "full_model_trajectory_v1",
        "empty_supervision_policy": "count_round_without_optimizer_step",
        "max_anchor_scan_events": scan,
        "noop_policy": "skip_exact_noop_curate_v1",
        "group_size": int(args.group_size),
        "tensor_parallel_size": int(args.tensor_parallel_size),
        "rollout_replicas": int(rollout_replicas),
        "nnodes": int(nnodes),
        "nproc_per_node": OFFICIAL_NPROC_PER_NODE,
        "gradient_ranks": 1,
        "teacher_mask": teacher_mask,
        "student_mask": student_mask,
        "reward_used_for_optimization": False,
    }
    record["immutable"] = immutable_identity(record)
    return record
