"""Run identity, isolated episode outputs and derived RNG streams (PR-02,
guide §3.2).

Three fixes in one module:

A. **Episode output identity.**  A/B arms (and repeated candidates) of one
   run previously shared the same final-artifact paths (AppWorld keyed by
   ``rsi-<task>-<seed>``, OUT roots by task id), so the later arm silently
   overwrote the earlier one.  ``RunContext`` carries the run/phase/arm/bank/
   policy/replicate identity and derives a UNIQUE, colon-free output
   directory per episode; adapters write under it.  Task/environment RNG
   inputs are NOT changed by the identity — same task + same environment
   seed still reproduces the same initial world.

B. **Derived RNG streams.**  ``stable_seed`` derives independent per-decision
   seeds from stable string fields (never Python's salted ``hash``): actor /
   teacher / auxiliary / proposer / environment streams stay separate, so a
   teacher request inside one decision cannot consume the student's RNG and
   change its next sample (guide §3.2 B).

C. **Atomic writes.**  ``atomic_write_json`` uses tmp-file + rename and
   refuses to silently overwrite an existing artifact with different
   content.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from rsi.contracts.common import digest_of, digest_of_text

# RNG stream names (guide §3.2 B).  Student/teacher/aux/proposer never share
# a stream; two entries in this tuple always derive different seeds.
STREAMS = ("environment", "actor", "teacher", "auxiliary", "proposer")


def stable_seed(
    base_seed: int,
    *,
    task_id: str,
    replicate_id: int = 0,
    stream: str = "actor",
    decision_index: int = 0,
    attempt_index: int = 0,
) -> int:
    """Deterministic, process-independent seed derivation.

    Python's builtin ``hash()`` is salted per process — forbidden here.  The
    derivation hashes the stable fields with SHA-256 and folds the digest to
    a 63-bit non-negative int.
    """
    if stream not in STREAMS:
        raise ValueError(f"unknown RNG stream {stream!r} (known: {STREAMS})")
    payload = (
        f"{int(base_seed)}|{task_id}|{int(replicate_id)}|{stream}|"
        f"{int(decision_index)}|{int(attempt_index)}"
    )
    h = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


@dataclass(frozen=True)
class RunContext:
    """Identity of one episode's execution context (guide §3.2 A).

    ``episode_nonce`` is an independent output identity — it never enters the
    task RNG or the environment state; it only makes artifact paths unique so
    same-task A/B arms and candidate re-evaluations cannot overwrite each
    other.
    """

    run_id: str
    campaign_id: str
    phase: str
    arm_id: str
    bank_digest: str
    policy_digest: str
    replicate_id: int = 0
    episode_nonce: str = ""

    def __post_init__(self) -> None:
        if not self.run_id or not self.campaign_id:
            raise ValueError("RunContext requires run_id and campaign_id")

    def experiment_hash(self) -> str:
        """Colon-free stable hash of the execution identity (AppWorld
        experiment names must not contain ':')."""
        return digest_of(
            {
                "run": self.run_id,
                "campaign": self.campaign_id,
                "phase": self.phase,
                "arm": self.arm_id,
                "bank": self.bank_digest,
                "policy": self.policy_digest,
                "replicate": self.replicate_id,
                "nonce": self.episode_nonce,
            }
        )[:16]

    def episode_dir(self, runs_root: str | Path, episode_id: str) -> Path:
        """The isolated output directory for one episode (guide §3.2 A)."""
        safe_ep = str(episode_id).replace(":", "_").replace("/", "_")
        return (
            Path(runs_root)
            / self.run_id
            / self.phase
            / self.arm_id
            / (self.bank_digest[:12] or "nobank")
            / str(self.replicate_id)
            / safe_ep
        )


def atomic_write_json(path: str | Path, payload: Mapping[str, Any], *, overwrite_ok: bool = False) -> Path:
    """Tmp-file + rename JSON write; refuses to silently overwrite.

    If the target exists with the SAME content, the write is a no-op; with
    DIFFERENT content it raises unless ``overwrite_ok`` (callers that own the
    path set it explicitly).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
    if p.exists() and not overwrite_ok:
        existing = p.read_text("utf-8")
        if existing != text:
            raise FileExistsError(
                f"artifact identity collision: {p} exists with different content "
                "(same-path different-run overwrite is forbidden — give the run "
                "a unique RunContext/episode_nonce)"
            )
        return p
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, p)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return p
