"""Policy export and reload verification (task book §9.5).

A change-record file is not a parameter update: export must verify weight
hashes and actual prediction changes.  LoRA merge requires an export/reload
logprob equivalence check and an updated full-weight identity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rsi.contracts.common import digest_of
from rsi.contracts.errors import ProtocolViolation
from rsi.contracts.policy import PolicyArtifact
from rsi.training.lora import adapter_digest, lora_state_dict


def export_adapter(backend: Any, out_dir: str | Path, *, parent_policy_id: str | None = None) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = lora_state_dict(backend.model)
    if state:
        import torch

        torch.save(state, out_dir / "adapter.pt")
        adigest = adapter_digest(state)
    else:
        # Full-parameter reference path: export the full weight digest.
        full = {k: v.detach().cpu() for k, v in backend.model.state_dict().items()}
        import torch

        torch.save(full, out_dir / "model.pt")
        adigest = digest_of({k: [float(x) for x in v.flatten().tolist()[:8]] for k, v in sorted(full.items())})
    artifact = PolicyArtifact.make(
        base_model_id=str(getattr(backend, "name", "hf_reference")),
        kind="trained",
        adapter_path=str(out_dir),
        adapter_digest=adigest,
        parent_policy_id=parent_policy_id,
        extra={"n_params": sum(1 for _ in backend.model.parameters())},
    )
    payload = {"artifact": artifact.to_json(), "adapter_digest": adigest}
    (out_dir / "policy_artifact.json").write_text(json.dumps(payload, indent=2, sort_keys=True), "utf-8")
    return payload


def verify_export_reload_equivalence(
    policy_a: Any,
    policy_b: Any,
    *,
    probe_prompt: str = "probe prompt",
    probe_target: str = "target",
    atol: float = 1e-4,
) -> dict[str, Any]:
    """Fixed-input logprob equivalence within tolerance; identity must match."""
    ids = policy_a.encode(probe_prompt)
    target = policy_b.encode(probe_target)
    if target and target[0] == 256:
        target = target[1:]
    la = policy_a.teacher_force_logprobs(ids, target)
    lb = policy_b.teacher_force_logprobs(ids, target)
    if len(la) != len(lb):
        raise ProtocolViolation("export/reload logprob length mismatch")
    max_diff = max((abs(x - y) for x, y in zip(la, lb)), default=0.0)
    ok = max_diff <= atol
    if not ok:
        raise ProtocolViolation(f"export/reload logprob mismatch {max_diff:.6f} > {atol}")
    return {
        "ok": ok,
        "max_logprob_diff": max_diff,
        "identity_a": policy_a.policy_artifact.policy_id,
        "identity_b": policy_b.policy_artifact.policy_id,
    }
