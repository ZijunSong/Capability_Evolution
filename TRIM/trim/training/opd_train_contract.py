"""Method / OPD-loss / backend support matrix.

Unsupported combinations must fail before GPU training starts. trim /
scape_seed uses sr_opd_projected_gap on hf_debug and the distributed
verl/FSDP2/DDP actor. rl+opd stays CE on those backends. Changing method
and loss together is not an acceleration of the same algorithm.
"""

from __future__ import annotations

from typing import Any

from trim.training.rl_opd_types import (
    OPD_LOSS_CE,
    OPD_LOSS_PROJECTED_GAP,
    OPD_LOSS_REVERSE_KL,
    OPD_LOSS_SAMPLED_GAP,
    PROJECTED_GAP_OBJECTIVE_VERSION,
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
    "verl_fsdp2": BACKEND_VERL,
    "torch_ddp_lora": BACKEND_VERL,
    "ddp": BACKEND_VERL,
    "ddp_lora": BACKEND_VERL,
}

LOSS_ALIASES = {
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
        (TRAINING_MODE_SCAPE_SEED, OPD_LOSS_PROJECTED_GAP, BACKEND_VERL),
        (TRAINING_MODE_SCAPE_RL, OPD_LOSS_SAMPLED_GAP, BACKEND_HF_DEBUG),
        (TRAINING_MODE_SCAPE_RL, OPD_LOSS_CE, BACKEND_HF_DEBUG),
        ("rl+opd", OPD_LOSS_PROJECTED_GAP, BACKEND_HF_DEBUG),
        ("rl+opd", OPD_LOSS_SAMPLED_GAP, BACKEND_HF_DEBUG),
    }
)

# trim/scape_seed is never CE or sampled-gap, even on hf_debug.
FORBIDDEN_METHOD_LOSSES: frozenset[tuple[str, str]] = frozenset(
    {
        (TRAINING_MODE_SCAPE_SEED, OPD_LOSS_CE),
        (TRAINING_MODE_SCAPE_SEED, OPD_LOSS_SAMPLED_GAP),
    }
)

CELL_FOR_METHOD = {
    TRAINING_MODE_RL: "rl",
    "rl+opd": "rl_opd",
    TRAINING_MODE_RL_OPD: "rl_opd",
    TRAINING_MODE_SCAPE_SEED: "scape_seed",
    TRAINING_MODE_SCAPE_RL: "scape_rl",
    TRAINING_MODE_PURE_OPD: "pure_opd",
}


def normalize_train_method(method: str | None) -> str:
    key = str(method or "").strip()
    lowered = key.lower().replace(" ", "")
    if lowered in METHOD_ALIASES:
        return METHOD_ALIASES[lowered]
    return key or TRAINING_MODE_RL


def normalize_backend(backend: str | None) -> str:
    key = str(backend or BACKEND_HF_DEBUG).strip().lower().replace("-", "_")
    return BACKEND_ALIASES.get(key, key)


def actor_wrap_for_backend(backend: str | None) -> str:
    key = str(backend or "").strip().lower().replace("-", "_")
    if key in {"torch_ddp_lora", "ddp", "ddp_lora"}:
        return "ddp"
    if normalize_backend(key) == BACKEND_HF_DEBUG:
        return "none"
    return "fsdp2"


def default_opd_loss_for_method(method: str | None) -> str:
    norm_method = normalize_train_method(method)
    if norm_method == TRAINING_MODE_SCAPE_SEED:
        return OPD_LOSS_PROJECTED_GAP
    if norm_method == TRAINING_MODE_SCAPE_RL:
        return OPD_LOSS_SAMPLED_GAP
    return OPD_LOSS_CE


def _loss_unspecified(opd_loss: str | None) -> bool:
    if opd_loss is None:
        return True
    raw = str(opd_loss).strip().lower()
    return raw in {"", "none", "null"}


def normalize_opd_loss(opd_loss: str | None, *, method: str | None = None) -> str:
    """Resolve loss after method default. Empty values are not CE aliases."""
    if _loss_unspecified(opd_loss):
        return default_opd_loss_for_method(method)
    raw = str(opd_loss).strip()
    if raw in LOSS_ALIASES:
        return LOSS_ALIASES[raw]
    return raw


def opd_loss_from_args(args: Any | None, *, method: str | None = None) -> str:
    """Read args.opd_loss without ``or 'sr_opd_ce'`` overriding trim defaults."""
    raw = getattr(args, "opd_loss", None) if args is not None else None
    use_method = method
    if use_method is None and args is not None:
        use_method = getattr(args, "train_method", None) or getattr(args, "training_mode", None)
    return normalize_opd_loss(raw, method=use_method)


def training_cell_for_method(method: str | None) -> str:
    norm = normalize_train_method(method)
    if norm not in CELL_FOR_METHOD:
        raise SystemExit(f"unsupported training cell for method={method!r} (normalized={norm!r})")
    return CELL_FOR_METHOD[norm]


def teacher_context_required(opd_loss: str | None) -> bool:
    return uses_seed_gap(str(opd_loss or ""))


def contract_supported(method: str, opd_loss: str, backend: str) -> bool:
    return (method, opd_loss, backend) in SUPPORTED_TRAIN_CONTRACTS


def describe_train_contract(
    method: str,
    opd_loss: str,
    backend: str,
    *,
    requested_backend: str | None = None,
    actor_wrap: str | None = None,
) -> dict[str, Any]:
    need_teacher = teacher_context_required(opd_loss)
    actor = "ce" if not uses_seed_gap(opd_loss) else "gap"
    requested = requested_backend or backend
    wrap = actor_wrap or actor_wrap_for_backend(requested)
    family = normalize_backend(requested)
    gap_on_dist = (
        method == TRAINING_MODE_SCAPE_SEED
        and opd_loss == OPD_LOSS_PROJECTED_GAP
        and family == BACKEND_VERL
    )
    return {
        "method": method,
        "opd_loss": opd_loss,
        "backend": requested,
        "backend_family": family,
        "actor_wrap": wrap,
        "supported": contract_supported(method, opd_loss, family),
        "teacher_context_required": need_teacher,
        "actor_objective": actor,
        "objective_version": PROJECTED_GAP_OBJECTIVE_VERSION if actor == "gap" else "ce_lambda_baked_v1",
        "verl_ce_only": family == BACKEND_VERL and actor == "ce",
        "distributed_gap_supported": gap_on_dist,
    }


def assert_train_contract(
    method: str | None,
    opd_loss: str | None = None,
    backend: str | None = None,
) -> dict[str, Any]:
    requested_backend = str(backend or BACKEND_HF_DEBUG)
    norm_method = normalize_train_method(method)
    norm_backend = normalize_backend(requested_backend)
    norm_loss = normalize_opd_loss(opd_loss, method=norm_method)
    if norm_method == TRAINING_MODE_RL:
        norm_loss = OPD_LOSS_CE
    if (norm_method, norm_loss) in FORBIDDEN_METHOD_LOSSES:
        raise SystemExit(
            "unsupported train contract "
            f"method={norm_method!r} opd_loss={norm_loss!r} backend={requested_backend!r}. "
            "trim/scape_seed requires sr_opd_projected_gap; CE and sampled-gap are not trim."
        )
    payload = describe_train_contract(
        norm_method,
        norm_loss,
        norm_backend,
        requested_backend=requested_backend,
        actor_wrap=actor_wrap_for_backend(requested_backend),
    )
    if payload["supported"]:
        return payload
    raise SystemExit(
        "unsupported train contract "
        f"method={norm_method!r} opd_loss={norm_loss!r} backend={requested_backend!r}. "
        "rl/rl+opd on verl/fsdp2/ddp stay CE; trim uses sr_opd_projected_gap on those "
        "backends; scape+rl gap stays on hf_debug. "
        "Do not treat a method+loss change as the same algorithm accelerated."
    )


def weights_look_like_unnormalized_gap_mask(weights: list[float] | None) -> bool:
    """True when CE actor is handed a 0/1 gap mask instead of lambda-normalized weights."""
    vals = [float(w) for w in (weights or [])]
    nonzero = [w for w in vals if abs(w) > 1e-12]
    if len(nonzero) < 2:
        return False
    return all(abs(w - 1.0) < 1e-8 for w in nonzero)
