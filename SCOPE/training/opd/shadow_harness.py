"""Shadow harness: compute privileged artifacts without changing student state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from harness.graph.execution_context import ExecutionContext
from harness.harness_config import HarnessConfig, load_harness_config
from harness.modules.verification import VerificationRecord
from harness.views.privileged_artifacts import PrivilegedArtifacts


@dataclass
class ShadowHarnessResult:
    artifacts: list[PrivilegedArtifacts] = field(default_factory=list)
    compute_cost: dict[str, float] = field(default_factory=dict)
    mode: str = "online"


class ShadowHarness:
    """Run teacher module on student prefix without mutating student WM."""

    def __init__(
        self,
        teacher_config: HarnessConfig | str,
        *,
        openai_client: Any = None,
        offline: bool = False,
    ) -> None:
        if isinstance(teacher_config, str):
            teacher_config = load_harness_config(teacher_config)
        self.teacher_config = teacher_config
        self.openai_client = openai_client
        self.offline = offline

    def run_verification_shadow(
        self,
        *,
        turn_id: int,
        claim: str,
        doc_ids: list[str],
        doc_texts: dict[str, str],
        student_wm: Any,
    ) -> ShadowHarnessResult:
        """Compute verification artifact from cached docs; does not touch student_wm."""
        if not self.teacher_config.verification.enabled:
            return ShadowHarnessResult(mode="disabled")

        if self.offline or self.openai_client is None:
            # Offline privileged annotation stub
            record = VerificationRecord(
                turn_id=turn_id,
                claim=claim,
                doc_ids=list(doc_ids),
                judgments={d: False for d in doc_ids},
                rationales={d: "offline_annotation" for d in doc_ids},
            )
            mode = "offline privileged annotation"
        else:
            from harness.ultra_core import exec_verify_claim

            raw = exec_verify_claim(self.openai_client, doc_texts, claim)
            record = VerificationRecord.from_verify_output(
                turn_id, claim, doc_ids, raw
            )
            mode = "online"

        artifact = PrivilegedArtifacts(
            module_id="verification",
            turn_id=turn_id,
            compact_text=record.compact_text(),
            structured_payload=record.to_dict(),
            provenance=[f"turn_{turn_id}"],
            future_leakage=False,
        )
        _ = student_wm  # explicitly not modified
        return ShadowHarnessResult(
            artifacts=[artifact],
            compute_cost={"verify_calls": 1.0},
            mode=mode,
        )


def build_shadow_context(episode_id: str, query_id: str) -> ExecutionContext:
    return ExecutionContext(episode_id=episode_id, query_id=query_id)
