"""HF-based policy provider (reference implementation).

Loads a real ``transformers`` causal LM (a local model path, or an offline
tiny randomly-initialized model for tests) behind the PolicyProvider protocol.
The tokenizer is a self-contained byte-level tokenizer so that everything works
offline with stable digests.

This is the inference side only; training backends live in ``rsi.training``.
"""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

from rsi.contracts.common import digest_of_text
from rsi.contracts.errors import NotImplementedCapability
from rsi.contracts.policy import PolicyArtifact, SamplingParams
from rsi.providers.base import SampleResult

BOS_ID = 256
EOS_ID = 257
VOCAB_SIZE = 258
TOKENIZER_DIGEST = "rsi_byte_tokenizer_v1"
CHAT_TEMPLATE_DIGEST = "rsi_render_v1"


def _require_torch_transformers() -> tuple[Any, Any]:
    try:
        import torch
        import transformers
    except ImportError as e:  # pragma: no cover
        raise NotImplementedCapability(
            "HFPolicy requires torch and transformers (pip install -e '.[train]')"
        ) from e
    return torch, transformers


def tune_torch_threads(max_threads: int = 8) -> int:
    """Avoid CPU oversubscription thrash on many-core hosts.

    Torch defaults to one thread per core; on large shared hosts that is far
    slower than a bounded pool for small models (measured 6.5s vs 0.1s per
    forward on a 44-thread host).  Applies only when the current setting is
    above the cap.
    """
    import os

    import torch

    current = torch.get_num_threads()
    cap = min(max_threads, os.cpu_count() or max_threads)
    if current > cap:
        torch.set_num_threads(cap)
    return torch.get_num_threads()


class ByteTokenizer:
    """Deterministic byte-level tokenizer with BOS/EOS specials."""

    vocab_size = VOCAB_SIZE

    def encode(self, text: str, *, add_bos: bool = True, add_eos: bool = False) -> list[int]:
        ids = list(text.encode("utf-8"))
        if add_bos:
            ids = [BOS_ID] + ids
        if add_eos:
            ids = ids + [EOS_ID]
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        data = bytes(i for i in ids if 0 <= i < 256)
        return data.decode("utf-8", errors="replace")

    @property
    def digest(self) -> str:
        return TOKENIZER_DIGEST


def build_tiny_model(hidden: int = 64, layers: int = 2, heads: int = 2) -> Any:
    """Offline tiny causal LM (random init) for CPU tests and smoke runs."""
    torch, transformers = _require_torch_transformers()
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=hidden,
        intermediate_size=hidden * 4,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=heads,
        max_position_embeddings=2048,
        rms_norm_eps=1e-5,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg)
    model.eval()
    return model


class HFPolicy:
    """PolicyProvider over a HF causal LM + ByteTokenizer."""

    def __init__(
        self,
        model: Any | None = None,
        *,
        model_path: str | None = None,
        base_model_id: str = "tiny-hf",
        device: str = "cpu",
    ):
        torch, _ = _require_torch_transformers()
        tune_torch_threads()
        if model is None and model_path is not None:
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
            base_model_id = model_path
        if model is None:
            model = build_tiny_model()
        self.model = model.to(device)
        self.device = device
        self.tokenizer = ByteTokenizer()
        self.policy_artifact = PolicyArtifact.make(base_model_id, kind="base")
        self.tokenizer_digest = self.tokenizer.digest
        self.chat_template_digest = CHAT_TEMPLATE_DIGEST
        self._torch = torch

    # -- PolicyProvider ----------------------------------------------------
    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_bos=True)

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(ids)

    def sample(self, prompt_ids: Sequence[int], params: SamplingParams) -> SampleResult:
        torch = self._torch
        ids = list(prompt_ids)
        logprobs: list[float] = []
        generated: list[int] = []
        self.model.eval()
        with torch.no_grad():
            for _ in range(params.max_new_tokens):
                inp = torch.tensor([ids], dtype=torch.long, device=self.device)
                out = self.model(inp)
                logits = out.logits[0, -1, :256]  # bytes only
                if params.temperature <= 0:
                    nxt = int(torch.argmax(logits).item())
                    lp = float(torch.log_softmax(logits.float(), dim=-1)[nxt].item())
                else:
                    probs = torch.softmax(logits.float() / params.temperature, dim=-1)
                    nxt = int(torch.multinomial(probs, 1, generator=None).item())
                    lp = float(torch.log(probs[nxt] + 1e-12).item())
                ids.append(nxt)
                generated.append(nxt)
                logprobs.append(lp)
                text = self.tokenizer.decode(generated)
                if len(text.strip()) > 0 and text.strip().endswith("}"):
                    break
        return SampleResult(
            text=self.tokenizer.decode(generated),
            token_ids=tuple(generated),
            logprobs=tuple(logprobs),
            finish_reason="stop",
        )

    def teacher_force_logprobs(self, prompt_ids: Sequence[int], target_ids: Sequence[int]) -> list[float]:
        torch = self._torch
        if len(target_ids) == 0:
            return []
        full = list(prompt_ids) + list(target_ids)
        inp = torch.tensor([full], dtype=torch.long, device=self.device)
        self.model.eval()
        with torch.no_grad():
            out = self.model(inp)
            logp = torch.log_softmax(out.logits[0].float(), dim=-1)
        # logprob of each target token given the prefix before it
        res: list[float] = []
        offset = len(prompt_ids)
        for i, tid in enumerate(target_ids):
            res.append(float(logp[offset + i - 1, tid].item()) if offset + i - 1 >= 0 else 0.0)
        return res

    def freeze_for_batch(self) -> PolicyArtifact:
        return self.policy_artifact

    # -- helpers ----------------------------------------------------------
    def adapter_state_dict(self) -> dict[str, Any]:
        """Full parameter state (the HF reference path trains all weights)."""
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.model.load_state_dict(state)

    def parameters(self):
        return self.model.parameters()

    def reload_and_verify(self, other: "HFPolicy", probe_text: str = "probe") -> float:
        """Logprob equivalence after export/reload (max abs diff)."""
        torch = self._torch
        ids = self.encode(probe_text)
        target = self.encode(" out", add_bos=False)
        a = self.teacher_force_logprobs(ids, target)
        b = other.teacher_force_logprobs(ids, target)
        return max((abs(x - y) for x, y in zip(a, b)), default=0.0)
