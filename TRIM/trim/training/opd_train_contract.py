"""Method / OPD-loss / backend support matrix.

Unsupported combinations must fail before GPU training starts. The
distributed verl/FSDP2 actor is CE-only; projected/sampled gap stays on
hf_debug. Changing method and loss together is not an acceleration of the
same algorithm.
"""

from __future__ import annotations

from typing import Any

from trim.training.rl_opd_types import (
    OPD_LOSS_CE,
    OPD_LOSS_PROJECTED_GAP,
    OPD_LOSS_REVERSE_KL,
    OPD_LOSS_SAMPLED_GAP,
    TRAINING_MODE_PURE_OPD,
    TRAINING_MODE_RL,
    TRAINING_MODE_RL_OPD,
    TRAINING_MODE_SCAPE_RL,
    TRAINING_MODE_SCAPE_SEED,
    uses_seed_gap,
)

BACKEND_HF_DEBUG = "hf_debug"
BACKEND_VERL = "verl"

METHOD_ALIASES = {
    "rl": TRAINING_MODE_RL,
    "rl+opd": "rl+opd",
    "rl_opd": "rl+opd",
    "four_cell": "rl+opd",
    "opd": TRAINING_MODE_PURE_OPD,
    "pure_opd": TRAINING_MODE_PURE_OPD,
    "trim": TRAINING_MODE_SCAPE_SEED,
    "scape_seed": TRAINING_MODE_SCAPE_SEED,
    "scape+seed": TRAINING_MODE_SCAPE_SEED,
    "scape_rl": TRAINING_MODE_SCAPE_RL,
    "scape+rl": TRAINING_MODE_SCAPE_RL,
}

BACKEND_ALIASES = {
    "hf_debug": BACKEND_HF_DEBUG,
    "local_legacy": BACKEND_HF_DEBUG,
    "verl": BACKEND_VERL,
    "fsdp2": BACKEND_VERL,
    "torch_ddp_lora": BACKEND_VERL,
}

LOSS_ALIASES = {
    "": OPD_LOSS_CE,
    "none": OPD_LOSS_CE,
    OPD_LOSS_CE: OPD_LOSS_CE,
    OPD_LOSS_PROJECTED_GAP: OPD_LOSS_PROJECTED_GAP,
    OPD_LOSS_SAMPLED_GAP: OPD_LOSS_SAMPLED_GAP,
    OPD_LOSS_REVERSE_KL: OPD_LOSS_SAMPLED_GAP,
    "sr_opd_reverse_kl": OPD_LOSS_SAMPLED_GAP,
}

# Explicit allow-list. Missing entries are unsupported.
SUPPORTED_TRAIN_CONTRACTS: frozenset[tuple[str, str, str]] = frozenset(
    {
        (TRAINING_MODE_RL, OPD_LOSS_CE, BACKEND_HF_DEBUG),
        (TRAINING_MODE_RL, OPD_LOSS_CE, BACKEND_VERL),
        ("rl+opd", OPD_LOSS_CE, BACKEND_HF_DEBUG),
        ("rl+opd", OPD_LOSS_CE, BACKEND_VERL),
        (TRAINING_MODE_PURE_OPD, OPD_LOSS_CE, BACKEND_HF_DEBUG),
        (TRAINING_MODE_SCAPE_SEED, OPD_LOSS_PROJECTED_GAP, BACKEND_HF_DEBUG),
        (TRAINING_MODE_SCAPE_RL, OPD_LOSS_SAMPLED_GAP, BACKEND_HF_DEBUG),
        (TRAINING_MODE_SCAPE_RL, OPD_LOSS_CE, BACKEND_HF_DEBUG),
        ("rl+opd", OPD_LOSS_PROJECTED_GAP, BACKEND_HF_DEBUG),
        ("rl+opd", OPD_LOSS_SAMPLED_GAP, BACKEND_HF_DEBUG),
    }
)


def normalize_train_method(method: str | None) -> str:
    key = str(method or "").strip()
    lowered = key.lower().replace(" ", "")
    if lowered in METHOD_ALIASES:
        return METHOD_ALIASES[lowered]
    return key or TRAINING_MODE_RL


def normalize_backend(backend: str | None) -> str:
    key = str(backend or BACKEND_HF_DEBUG).strip().lower().replace("-", "_")
    return BACKEND_ALIASES.get(key, key)


def normalize_opd_loss(opd_loss: str | None, *, method: str | None = None) -> str:
    raw = str(opd_loss or "").strip()
    if raw in LOSS_ALIASES:
        return LOSS_ALIASES[raw]
    if not raw:
        norm_method = normalize_train_method(method)
        if norm_method == TRAINING_MODE_SCAPE_SEED:
            return OPD_LOSS_PROJECTED_GAP
        if norm_method == TRAINING_MODE_SCAPE_RL:
            return OPD_LOSS_SAMPLED_GAP
        return OPD_LOSS_CE
    return raw


def teacher_context_required(opd_loss: str | None) -> bool:
    return uses_seed_gap(str(opd_loss or ""))


def contract_supported(method: str, opd_loss: str, backend: str) -> bool:
    return (method, opd_loss, backend) in SUPPORTED_TRAIN_CONTRACTS


def describe_train_contract(method: str, opd_loss: str, backend: str) -> dict[str, Any]:
    need_teacher = teacher_context_required(opd_loss)
    actor = "ce" if not uses_seed_gap(opd_loss) else "gap"
    return {
        "method": method,
        "opd_loss": opd_loss,
        "backend": backend,
        "supported": contract_supported(method, opd_loss, backend),
        "teacher_context_required": need_teacher,
        "actor_objective": actor,
        "verl_ce_only": backend == BACKEND_VERL,
    }


def assert_train_contract(
    method: str | None,
    opd_loss: str | None = None,
    backend: str | None = None,
) -> dict[str, Any]:
    norm_method = normalize_train_method(method)
    norm_backend = normalize_backend(backend)
    norm_loss = normalize_opd_loss(opd_loss, method=norm_method)
    if norm_method == TRAINING_MODE_RL:
        norm_loss = OPD_LOSS_CE
    payload = describe_train_contract(norm_method, norm_loss, norm_backend)
    if payload["supported"]:
        return payload
    raise SystemExit(
        "unsupported train contract "
        f"method={norm_method!r} opd_loss={norm_loss!r} backend={norm_backend!r}. "
        "verl/fsdp2 is CE-only for rl and rl+opd; trim/scape+rl gap stays on hf_debug. "
        "Do not treat a method+loss change as the same algorithm accelerated."
    )


def weights_look_like_unnormalized_gap_mask(weights: list[float] | None) -> bool:
    """True when CE actor is handed a 0/1 gap mask instead of lambda-normalized weights."""
    vals = [float(w) for w in (weights or [])]
    nonzero = [w for w in vals if abs(w) > 1e-12]
    if len(nonzero) < 2:
        return False
    return all(abs(w - 1.0) < 1e-8 for w in nonzero)
