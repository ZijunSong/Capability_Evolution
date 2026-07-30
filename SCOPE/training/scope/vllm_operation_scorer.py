"""Score KEEP/SKIP via running vLLM OpenAI API (no second model load)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from harness.capability.dup_operation import DupOperation
from training.scope.decision_config import DupDecisionConfig, DEFAULT_DECISION_CONFIG
from training.scope.operation_scorer import VERBALIZERS, OperationScoreResult
from training.scope.prompting import format_operation_prompt


@dataclass
class VllmOperationScorer:
    client: OpenAI
    model: str
    decision_config: DupDecisionConfig = DEFAULT_DECISION_CONFIG

    def score(self, decision_state_text: str) -> OperationScoreResult:
        prompt = format_operation_prompt(decision_state_text)
        scores: dict[str, float] = {}
        log_probs: dict[str, float] = {}
        prompt_token_count = len(prompt.split())  # fallback only
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
            tok_lps: list[float] = []
            if choice.logprobs and choice.logprobs.token_logprobs:
                text_offset = choice.logprobs.text_offset or []
                # Score only completion suffix tokens (after prompt boundary)
                if text_offset:
                    for i, lp in enumerate(choice.logprobs.token_logprobs):
                        if lp is None:
                            continue
                        offset = text_offset[i] if i < len(text_offset) else 0
                        if offset >= len(prompt):
                            tok_lps.append(lp)
                else:
                    # Fallback: use last N tokens matching verbalizer length
                    comp_len = max(len(op.value.split()), 1)
                    valid = [x for x in choice.logprobs.token_logprobs if x is not None]
                    tok_lps = valid[-comp_len:] if valid else []
            mean_lp = sum(tok_lps) / max(len(tok_lps), 1) if tok_lps else -1e9
            scores[op.value] = mean_lp
            log_probs[op.value] = mean_lp * max(len(tok_lps), 1)
        sk = scores[DupOperation.KEEP_EVIDENCE.value]
        ss = scores[DupOperation.SKIP_DUPLICATE.value]
        predicted = self.decision_config.predict_from_scores(sk, ss)
        return OperationScoreResult(
            scores=scores, predicted=predicted, log_probs=log_probs
        )
