"""Score KEEP/SKIP via running vLLM OpenAI API (no second model load)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from harness.capability.dup_operation import DupOperation
from training.scope.operation_scorer import VERBALIZERS, OperationScoreResult
from training.scope.prompting import format_operation_prompt


@dataclass
class VllmOperationScorer:
    client: OpenAI
    model: str

    def score(self, decision_state_text: str) -> OperationScoreResult:
        prompt = format_operation_prompt(decision_state_text)
        scores: dict[str, float] = {}
        log_probs: dict[str, float] = {}
        for op in VERBALIZERS:
            completion = prompt + op.value
            resp = self.client.completions.create(
                model=self.model,
                prompt=completion,
                max_tokens=0,
                echo=True,
                logprobs=1,
            )
            choice = resp.choices[0]
            tok_lps = []
            if choice.logprobs and choice.logprobs.token_logprobs:
                prompt_len = len(prompt)
                # Approximate: score only completion suffix tokens
                for i, lp in enumerate(choice.logprobs.token_logprobs):
                    if lp is None:
                        continue
                    # tokens after prompt contribute
                    if i * 4 >= prompt_len:  # rough char-based split fallback
                        tok_lps.append(lp)
                if not tok_lps:
                    tok_lps = [x for x in choice.logprobs.token_logprobs if x is not None]
            mean_lp = sum(tok_lps) / max(len(tok_lps), 1) if tok_lps else -1e9
            scores[op.value] = mean_lp
            log_probs[op.value] = mean_lp * max(len(tok_lps), 1)
        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        return OperationScoreResult(
            scores=scores, predicted=DupOperation(best), log_probs=log_probs
        )
