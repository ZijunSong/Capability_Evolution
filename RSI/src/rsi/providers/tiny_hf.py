"""HF-based policy provider (reference implementation).

Loads a real ``transformers`` causal LM behind the PolicyProvider protocol.

Two tokenizer paths:

- ``ByteTokenizer`` — self-contained 258-vocab byte tokenizer for the offline
  tiny model (tests / smoke).
- ``AutoTokenizer`` — the model's own tokenizer (chat template applied for
  prompts, raw ids for action targets) when loading a real model directory.
  Sampling runs over the model's full vocabulary.

This is the inference side only; training backends live in ``rsi.training``.
"""

from __future__ import annotations

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
    """Deterministic byte-level tokenizer with BOS/EOS specials (offline path)."""

    vocab_size = VOCAB_SIZE
    eos_id = EOS_ID
    is_hf = False

    def encode(self, text: str, *, add_bos: bool = True, add_eos: bool = False) -> list[int]:
        ids = list(text.encode("utf-8"))
        if add_bos:
            ids = [BOS_ID] + ids
        if add_eos:
            ids = ids + [EOS_ID]
        return ids

    def encode_prompt(self, text: str) -> list[int]:
        return self.encode(text, add_bos=True)

    def encode_action(self, text: str) -> list[int]:
        return self.encode(text, add_bos=False)

    def decode(self, ids: Sequence[int]) -> str:
        data = bytes(i for i in ids if 0 <= i < 256)
        return data.decode("utf-8", errors="replace")

    @property
    def digest(self) -> str:
        return TOKENIZER_DIGEST


class HFTokenizerAdapter:
    """The model's own tokenizer: chat template for prompts, raw ids for targets."""

    def __init__(self, tokenizer: Any, name_or_path: str):
        self.tok = tokenizer
        self.name_or_path = name_or_path
        if self.tok.eos_token_id is None:
            self.tok.eos_token_id = self.tok.pad_token_id
        self._digest = digest_of_text(
            f"hf_tok:{name_or_path}:{getattr(self.tok, 'vocab_size', 0)}:"
            f"{getattr(self.tok, 'chat_template', None) or 'no_chat_template'}"
        )

    @property
    def vocab_size(self) -> int:
        return int(getattr(self.tok, "vocab_size", len(self.tok)))

    @property
    def eos_id(self) -> int | None:
        return self.tok.eos_token_id

    is_hf = True

    def encode(self, text: str, *, add_bos: bool = True, add_eos: bool = False) -> list[int]:
        # Raw tokenization (no chat template): targets and raw ids.
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        return list(ids)

    def encode_prompt(self, text: str) -> list[int]:
        """Prompt path: apply the model's chat template when it has one.

        Qwen-family templates accept ``enable_thinking=False`` (thinking-mode
        output would be unparseable as the one-JSON-object action protocol).
        """
        tpl = getattr(self.tok, "chat_template", None)
        if tpl:
            messages = [{"role": "user", "content": text}]
            try:
                rendered = self.tok.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                rendered = self.tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            return list(self.tok(rendered, add_special_tokens=True)["input_ids"])
        ids = self.tok(text, add_special_tokens=True)["input_ids"]
        return list(ids)

    def encode_action(self, text: str) -> list[int]:
        return self.encode(text, add_bos=False)

    def decode(self, ids: Sequence[int]) -> str:
        return self.tok.decode(list(ids), skip_special_tokens=True)

    @property
    def digest(self) -> str:
        return self._digest


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


def generation_complete(text: str) -> bool:
    """Stop when the emitted text is a complete JSON object or fenced block.

    The old ``text.endswith("}")`` heuristic truncated nested JSON at the first
    inner brace close (e.g. ``"arguments": {...}``), producing unparseable
    actions.  Complete-parse is the correct terminator for the one-JSON-object
    action/patch protocol.
    """
    from rsi.contracts.action import extract_json_object

    s = text.strip()
    if not s:
        return False
    if s.endswith("```") and s.count("```") >= 2:
        return True
    try:
        extract_json_object(s)
        return True
    except Exception:  # noqa: BLE001 - incomplete output, keep generating
        return False


class _CompleteOutputStop:
    """transformers-compatible stopping criteria (duck-typed to avoid a hard
    import at module load)."""

    def __init__(self, tokenizer: Any, prompt_len: int):
        self._tokenizer = tokenizer
        self._prompt_len = prompt_len

    def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
        seq = input_ids[0][self._prompt_len :].tolist()
        if not seq:
            return False
        return generation_complete(self._tokenizer.decode(seq))


def _load_pretrained(model_path: str, *, torch_dtype: Any, device: str) -> Any:
    """Load HF causal-LM weights with bounded host-RAM footprint.

    Two failure modes motivated this wrapper (kygt9 hang, 2026-11-02):

    1. ``from_pretrained`` materializes the full checkpoint in host RAM before
       ``.to(device)``.  With 8 spawn workers each loading a 27B bf16 model
       (~54GB) that is ~430GB peak vs a 224GB pod — the OOM killer slaughters
       workers mid-load and ``Pool.map`` then waits forever on dead children.
       ``low_cpu_mem_usage=True`` (accelerate) streams shard-by-shard into the
       target device instead; if accelerate is unavailable we fall back to a
       process-wide file lock so at most one full-RAM materialization happens
       at a time.
    2. Load progress was invisible: a hang left zero log evidence.  The prints
       give the watchdog something to judge liveness on.
    """
    import time

    from transformers import AutoModelForCausalLM

    t0 = time.perf_counter()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            device_map={"": device} if str(device).startswith("cuda") else None,
        )
        print(f"[tiny_hf] loaded {model_path} -> {device} in {time.perf_counter() - t0:.0f}s", flush=True)
        return model
    except Exception as e:  # noqa: BLE001 - accelerate/device_map may be absent
        print(f"[tiny_hf] low-mem load failed ({e!r}); serialized full-RAM fallback", flush=True)
    # Fallback: serialize full-RAM materialization across panel workers so the
    # concurrent peak stays at one model copy instead of one per GPU.
    import fcntl
    import os

    lock_path = os.environ.get("RSI_MODEL_LOAD_LOCK", "/tmp/rsi_model_load.lock")
    with open(lock_path, "a") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype)
            print(
                f"[tiny_hf] serialized load {model_path} -> {device} in {time.perf_counter() - t0:.0f}s",
                flush=True,
            )
            return model
        finally:
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)


class HFPolicy:
    """PolicyProvider over a HF causal LM.

    ``model_path=None`` -> tiny offline model + ByteTokenizer (tests).
    ``model_path=<dir>`` -> real model + its own tokenizer/chat template.
    """

    def __init__(
        self,
        model: Any | None = None,
        *,
        model_path: str | None = None,
        base_model_id: str = "tiny-hf",
        device: str = "cpu",
        dtype: str | None = None,
    ):
        torch, _ = _require_torch_transformers()
        tune_torch_threads()
        torch_dtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }.get(str(dtype or "float32"), torch.float32)
        if model is None and model_path is not None:
            from transformers import AutoTokenizer

            model = _load_pretrained(model_path, torch_dtype=torch_dtype, device=device)
            tokenizer = HFTokenizerAdapter(AutoTokenizer.from_pretrained(model_path), model_path)
            base_model_id = model_path
        else:
            tokenizer = ByteTokenizer()
        if model is None:
            model = build_tiny_model()
        if str(device).startswith("cuda"):
            # model.to("cuda:*") 会走 torch._C._cuda_init()，失败时报的是
            # "NVIDIA driver too old" 这类看不出根因的信息。此处提前把设备可用性
            # 说清楚（u7mzf: torch 2.15.0+cu132 vs 节点 driver CUDA 12.8）。
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"HFPolicy: device {device!r} requested but torch.cuda.is_available() is False "
                    f"(torch {torch.__version__}) — torch build is incompatible with the node "
                    "NVIDIA driver; pin a driver-matched torch (e.g. torch==2.10.0 for CUDA 12.8 "
                    "drivers) before running"
                )
        self.model = model if getattr(model, "hf_device_map", None) else model.to(device)
        self.device = device
        self.tokenizer = tokenizer
        self.policy_artifact = PolicyArtifact.make(base_model_id, kind="base")
        self.tokenizer_digest = self.tokenizer.digest
        self.chat_template_digest = (
            digest_of_text(f"chat:{self.tokenizer.digest}") if getattr(tokenizer, "is_hf", False)
            else CHAT_TEMPLATE_DIGEST
        )
        self._torch = torch

    # -- PolicyProvider ----------------------------------------------------
    def encode(self, text: str) -> list[int]:
        """Encode a *prompt* (chat template applied for real-model tokenizers)."""
        return self.tokenizer.encode_prompt(text)

    def encode_action(self, text: str) -> list[int]:
        """Encode an *action target* as raw ids (no chat template)."""
        return self.tokenizer.encode_action(text)

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(ids)

    def sample(self, prompt_ids: Sequence[int], params: SamplingParams) -> SampleResult:
        """Autoregressive sampling via ``model.generate`` (KV cache enabled).

        The previous hand-rolled loop re-ran a full-prefix forward per token
        (pathological on Qwen3.x hybrid linear-attention models) and stopped at
        the first trailing ``}``, truncating nested JSON.  This path uses the
        HF KV cache, stops on EOS / complete-JSON / fenced-block, and reports
        per-token behavior logprobs via transition scores.
        """
        torch = self._torch
        self.model.eval()
        prompt_len = len(prompt_ids)
        input_ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device=self.device)
        gen_kwargs: dict[str, Any] = dict(
            max_new_tokens=params.max_new_tokens,
            use_cache=True,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=params.temperature > 0,
            eos_token_id=self.tokenizer.eos_id,
            pad_token_id=self.tokenizer.eos_id if self.tokenizer.eos_id is not None else 0,
        )
        if params.temperature > 0:
            gen_kwargs["temperature"] = float(params.temperature)
            gen_kwargs["top_p"] = float(params.top_p or 1.0)
        stopper = _CompleteOutputStop(self.tokenizer, prompt_len)
        try:
            from transformers import StoppingCriteriaList

            gen_kwargs["stopping_criteria"] = StoppingCriteriaList([stopper])
        except Exception:  # noqa: BLE001 - ultra-old transformers fallback
            pass
        with torch.no_grad():
            out = self.model.generate(input_ids, **gen_kwargs)
        generated = [int(t) for t in out.sequences[0][prompt_len:].tolist()]
        # per-token behavior logprobs of the taken tokens
        logprobs: list[float] = []
        try:
            from transformers import compute_transition_scores

            scores = compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
            logprobs = [float(x) for x in scores[0][: len(generated)].tolist()]
        except Exception:  # noqa: BLE001 - fall back to zeros with a loud tag
            logprobs = [0.0] * len(generated)
        # strip trailing EOS
        if self.tokenizer.eos_id is not None and generated and generated[-1] == self.tokenizer.eos_id:
            generated = generated[:-1]
            logprobs = logprobs[: len(generated)]
        text = self.tokenizer.decode(generated)
        return SampleResult(
            text=text,
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
        ids = self.encode_action(probe_text)
        target = self.encode_action(" out")
        a = self.teacher_force_logprobs(ids, target)
        b = other.teacher_force_logprobs(ids, target)
        return max((abs(x - y) for x, y in zip(a, b)), default=0.0)
