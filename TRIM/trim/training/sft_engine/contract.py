"""v2 run contract. Data identity is separate from the launch topology."""

from __future__ import annotations

import importlib
import subprocess
from pathlib import Path
from typing import Any, Mapping

from trim.training.sft_ablation.contracts import LOSS_VERSION, fingerprint
from trim.training.sft_ablation.io_util import sha256_text

SCHEMA_VERSION = "sft_run_contract_v3"
# Old DDP runs sealed sft_run_contract_v2. Both schemas describe the same
# dotted fields below, so a v2 checkpoint can still exact-resume under the new
# code; the schema stamp itself is informational for that comparison.
KNOWN_SCHEMA_VERSIONS = frozenset({"sft_run_contract_v2", "sft_run_contract_v3"})

# Exact resume must agree on these dotted fields. A new out directory is a new run.
EXACT_RESUME_PATHS = (
    "experiment",
    "data.source_contract_digest",
    "data.dataset_digest",
    "data.tokenizer_digest",
    "data.action_only_supervision",
    "data.weight_policy",
    "optimization.loss",
    "optimization.backend",
    "optimization.global_batch_size",
    "optimization.micro_batch_size_per_gpu",
    "optimization.gradient_ranks",
    "optimization.gradient_accumulation_scaling",
    "optimization.learning_rate",
    "optimization.weight_decay",
    "optimization.grad_clip",
    "optimization.seed",
    "optimization.max_model_len",
    "optimization.logits_mode",
    "optimization.gradient_checkpointing",
    "model.init_adapter_digest",
    "model.base_path",
)


def _lookup(record: Mapping[str, Any], path: str) -> Any:
    cursor: Any = record
    for part in path.split("."):
        if not isinstance(cursor, Mapping) or part not in cursor:
            raise KeyError(path)
        cursor = cursor[part]
    return cursor


def require_exact_resume(saved: Mapping[str, Any], live: Mapping[str, Any]) -> None:
    left_schema = saved.get("schema_version")
    right_schema = live.get("schema_version")
    if left_schema not in KNOWN_SCHEMA_VERSIONS or right_schema not in KNOWN_SCHEMA_VERSIONS:
        raise ValueError("exact resume requires a known sft_run_contract schema on both sides")
    mismatch = []
    for path in EXACT_RESUME_PATHS:
        try:
            left = _lookup(saved, path)
            right = _lookup(live, path)
        except KeyError:
            mismatch.append(path)
            continue
        if left != right:
            mismatch.append(path)
    if mismatch:
        raise ValueError(f"exact resume refused: {mismatch}")


def environment_lock_digest() -> str:
    rows = []
    for name in ("torch", "transformers", "peft", "accelerate", "triton", "flash_attn"):
        try:
            module = importlib.import_module(name)
            rows.append(f"{name}=={getattr(module, '__version__', 'unknown')}")
        except Exception:
            rows.append(f"{name}==missing")
    return sha256_text("\n".join(rows))


def source_git_commit(start: Path | None = None) -> str:
    cursor = (start or Path(__file__)).resolve()
    if cursor.is_file():
        cursor = cursor.parent
    for candidate in (cursor, *cursor.parents):
        if (candidate / ".git").exists():
            try:
                return subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=candidate,
                    stderr=subprocess.DEVNULL,
                    text=True,
                ).strip()
            except Exception:
                return "unknown"
    return "unknown"


def build_v2_contract(
    args: Any,
    *,
    dataset_digest: str,
    source_contract_digest: str,
    tokenizer_digest: str,
    gradient_ranks: int,
    nnodes: int,
    nproc_per_node: int,
    init_adapter_digest: str,
    backend: str = "accelerate_ddp",
    adapter_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    experiment = str(args.experiment)
    from trim.training.sft_ablation.contracts import WEIGHT_POLICY

    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment": experiment,
        "data": {
            "source_contract_digest": source_contract_digest,
            "dataset_digest": dataset_digest,
            "tokenizer_digest": tokenizer_digest,
            "action_only_supervision": True,
            "weight_policy": WEIGHT_POLICY[experiment],
        },
        "optimization": {
            "loss": LOSS_VERSION,
            "backend": str(backend),
            "global_batch_size": int(args.global_batch_size),
            "micro_batch_size_per_gpu": int(args.micro_batch_size),
            "gradient_ranks": int(gradient_ranks),
            "gradient_accumulation_scaling": "manual_global_mass",
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "grad_clip": float(args.grad_clip),
            "optimizer": str(args.optimizer),
            "scheduler": str(args.scheduler),
            "seed": int(args.seed),
            "max_model_len": int(args.max_model_len),
            "max_new_tokens": int(args.max_new_tokens),
            "logits_mode": str(args.logits_mode),
            "gradient_checkpointing": str(args.gradient_checkpointing),
            "model_precision": str(args.model_precision),
            "rl_enabled": False,
            "gap_enabled": False,
            "gate_enabled": False,
        },
        "runtime": {
            "nnodes": int(nnodes),
            "nproc_per_node": int(nproc_per_node),
            "source_git_commit": source_git_commit(Path(__file__).resolve().parents[3]),
            "environment_lock_digest": environment_lock_digest(),
        },
        "model": {
            "base_path": str(args.model_name),
            "init_adapter_digest": init_adapter_digest,
            "lora_rank": int(args.lora_rank),
            "lora_alpha": int(args.lora_alpha),
            "lora_dropout": float(args.lora_dropout),
            "lora_target": str(args.lora_target),
            "attn_implementation": str(getattr(args, "attn_implementation", "") or ""),
        },
    }
    if adapter_config:
        # PEFT adapter_config.json is the fact; CLI defaults are only a hint.
        payload["model"]["adapter_config"] = dict(adapter_config)
    if str(backend) == "accelerate_fsdp2":
        payload["optimization"]["fsdp2"] = {
            "fsdp_version": 2,
            "wrap": "transformer_based_wrap",
            "reshard_after_forward": True,
            "state_dict_type": "SHARDED_STATE_DICT",
            "cpu_ram_efficient_loading": True,
            "cpu_offload": False,
            "activation_checkpointing_plugin": False,
            "mixed_precision_policy": "param_bf16_reduce_fp32",
        }
    payload["contract_digest"] = fingerprint(payload)
    return payload


def adapter_config_from(adapter_path: Any) -> dict[str, Any] | None:
    """Read the real PEFT configuration if the adapter directory provides one."""
    import json

    try:
        path = Path(str(adapter_path)) / "adapter_config.json"
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    keep = {}
    for key in ("r", "lora_alpha", "lora_dropout", "target_modules", "peft_type", "task_type", "bias"):
        if key in payload:
            keep[key] = payload[key]
    return keep or None
