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

    def _completion_token_logprobs(
        self, choice: Any, prompt: str, verbalizer: str
    ) -> list[float]:
        if not choice.logprobs or not choice.logprobs.token_logprobs:
            return []
        lps: list[float | None] = list(choice.logprobs.token_logprobs)
        tok = self._get_tokenizer()
        if tok is not None:
            prompt_n = len(tok.encode(prompt, add_special_tokens=False))
            comp_n = max(len(tok.encode(verbalizer, add_special_tokens=False)), 1)
            suffix = [lp for i, lp in enumerate(lps) if lp is not None and i >= prompt_n]
            if suffix:
                return suffix[:comp_n]
        comp_n = 1
        if tok is not None:
            comp_n = max(len(tok.encode(verbalizer, add_special_tokens=False)), 1)
        valid = [lp for lp in lps if lp is not None]
        return valid[-comp_n:] if valid else []

    def score(
        self,
        decision_state_text: str,
        *,
        available_checkpoints: list[dict] | None = None,
    ) -> RollbackScoreResult:
        prompt = format_rollback_operation_prompt(
            decision_state_text,
            available_checkpoints=available_checkpoints,
            hint=self.hint,
        )
        scores: dict[str, float] = {}
        log_probs: dict[str, float] = {}
        for op in (
            RollbackOperation.CONTINUE,
            RollbackOperation.REPLAN,
            RollbackOperation.ROLLBACK_TO,
        ):
            completion = prompt + op.value
            resp = self.client.completions.create(
                model=self.model,
                prompt=completion,
                max_tokens=0,
                echo=True,
                logprobs=1,
            )
            choice = resp.choices[0]
            tok_lps = self._completion_token_logprobs(choice, prompt, op.value)
            mean_lp = sum(tok_lps) / max(len(tok_lps), 1) if tok_lps else -1e9
            scores[op.value] = mean_lp
            log_probs[op.value] = mean_lp * max(len(tok_lps), 1)
        logits = [
            scores[RollbackOperation.CONTINUE.value],
            scores[RollbackOperation.REPLAN.value],
            scores[RollbackOperation.ROLLBACK_TO.value],
        ]
        idx = max(range(3), key=lambda i: logits[i])
        predicted = [
            RollbackOperation.CONTINUE,
            RollbackOperation.REPLAN,
            RollbackOperation.ROLLBACK_TO,
        ][idx]
        return RollbackScoreResult(
            scores=scores, predicted=predicted, log_probs=log_probs
        )
