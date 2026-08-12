"""Score rollback typed operations via running vLLM (no second full model load)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from harness.capability.rollback_operation import RollbackOperation
from training.scope.rollback_operation_objectives import format_rollback_operation_prompt


@dataclass
class RollbackScoreResult:
    scores: dict[str, float]
    predicted: RollbackOperation
    log_probs: dict[str, float]


@dataclass
class VllmRollbackScorer:
    client: OpenAI
    model: str
    model_path: str | None = None
    hint: str = ""
    _tokenizer: Any = field(default=None, init=False, repr=False)

    def _get_tokenizer(self) -> Any:
        if self._tokenizer is None and self.model_path:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
        return self._tokenizer

    def _completion_logprobs_from_token_ids(
        self, choice: Any, prompt_n: int, comp_n: int
    ) -> list[float]:
        """Extract completion token logprobs aligned to prompt_ids+comp_ids concat."""
        if not choice.logprobs or not choice.logprobs.token_logprobs:
            return []
        lps: list[float | None] = list(choice.logprobs.token_logprobs)
        out: list[float] = []
        for i in range(prompt_n, prompt_n + max(comp_n, 1)):
            if i < len(lps) and lps[i] is not None:
                out.append(float(lps[i]))
        if out:
            return out
        # Fallback: last N valid logprobs (should be rare with token-id prompts).
        valid = [float(lp) for lp in lps if lp is not None]
        return valid[-comp_n:] if valid else []

    def score_final_prompt(self, prompt: str) -> RollbackScoreResult:
        """Score a finalized effective prompt without re-wrapping.

        Uses HF-aligned tokenization: encode(prompt)+encode(verbalizer) as token
        ids, avoiding prompt+text merge artifacts that break string-echo scoring.
        """
        tok = self._get_tokenizer()
        if tok is None:
            raise RuntimeError(
                "VllmRollbackScorer.model_path is required for token-id parity scoring"
            )
        prompt_ids = tok.encode(prompt, add_special_tokens=False)
        scores: dict[str, float] = {}
        log_probs: dict[str, float] = {}
        for op in (
            RollbackOperation.CONTINUE,
            RollbackOperation.REPLAN,
            RollbackOperation.ROLLBACK_TO,
        ):
            comp_ids = tok.encode(op.value, add_special_tokens=False)
            if not comp_ids:
                scores[op.value] = -1e9
                log_probs[op.value] = -1e9
                continue
            input_ids = prompt_ids + comp_ids
            resp = self.client.completions.create(
                model=self.model,
                prompt=input_ids,
                max_tokens=0,
                echo=True,
                logprobs=1,
            )
            choice = resp.choices[0]
            tok_lps = self._completion_logprobs_from_token_ids(
                choice, len(prompt_ids), len(comp_ids)
            )
            mean_lp = sum(tok_lps) / max(len(tok_lps), 1) if tok_lps else -1e9
            scores[op.value] = mean_lp
            log_probs[op.value] = mean_lp * max(len(tok_lps), 1)
        from training.scope.decide_rollback_operation import decide_rollback_operation

        decision = decide_rollback_operation(
            score_continue=scores[RollbackOperation.CONTINUE.value],
            score_replan=scores[RollbackOperation.REPLAN.value],
            score_rollback=scores[RollbackOperation.ROLLBACK_TO.value],
            threshold=0.0,
            disable_replan=True,
        )
        return RollbackScoreResult(
            scores=scores, predicted=decision.predicted_operation, log_probs=log_probs
        )

    def score(
        self,
        decision_state_text: str,
        *,
        available_checkpoints: list[dict] | None = None,
        prompt_is_final: bool = False,
    ) -> RollbackScoreResult:
        if prompt_is_final:
            prompt = decision_state_text
        else:
            prompt = format_rollback_operation_prompt(
                decision_state_text,
                available_checkpoints=available_checkpoints,
                hint=self.hint,
            )
        return self.score_final_prompt(prompt)
