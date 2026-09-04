"""Local HF LoRA Clean-SFT for gpt-oss-20b on Harness-1 public SFT.

FULL vs TOOL share the same examples / optimizer budget / LoRA rank / epochs;
only the loss mask differs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MaskMode = Literal["full", "tool", "format_aware"]

CANONICAL_TOOLS = (
    "fan_out_search",
    "search_corpus",
    "grep_corpus",
    "read_document",
    "review_docs",
    "curate",
    "verify",
    "end_search",
)

HARMONY_TOOL_RE = re.compile(
    r"to=(?:functions\.)?(?P<name>[A-Za-z_][A-Za-z0-9_]*)",
)
JSON_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_tool_name(text: str) -> str | None:
    m = HARMONY_TOOL_RE.search(text or "")
    if m:
        name = m.group("name")
        if name != "functions":
            return name
    for name in CANONICAL_TOOLS:
        if re.search(rf"\b{re.escape(name)}\b", text or ""):
            return name
    return None


def format_aware_char_mask(text: str) -> list[bool]:
    """True from assistant tool-action Harmony control tokens through <|call|>."""
    n = len(text or "")
    mask = [False] * n
    if n == 0:
        return mask
    idx = text.find(" to=functions.")
    if idx < 0:
        idx = text.find("to=functions.")
    if idx < 0:
        idx = text.find("<|channel|>commentary")
    if idx < 0:
        return tool_char_mask(text)
    start_tag = text.rfind("<|start|>assistant", 0, idx + 1)
    start = start_tag if start_tag >= 0 else idx
    end = text.find("<|call|>", start)
    end = (end + len("<|call|>")) if end >= 0 else n
    for i in range(start, min(end, n)):
        mask[i] = True
    if not any(mask):
        return tool_char_mask(text)
    return mask


def tool_char_mask(text: str) -> list[bool]:
    """True on tool-call / action spans (name + JSON args), False on prose/reasoning."""
    n = len(text)
    mask = [False] * n
    if n == 0:
        return mask
    for m in HARMONY_TOOL_RE.finditer(text):
        for i in range(m.start(), m.end()):
            mask[i] = True
    for m in JSON_OBJ_RE.finditer(text):
        # keep JSON that sits near a tool call
        window = text[max(0, m.start() - 80) : m.start()]
        if "to=" in window or "constrain" in window or "functions." in window:
            for i in range(m.start(), m.end()):
                mask[i] = True
    if not any(mask):
        # fallback: whole assistant text is treated as action if a legal tool is named
        if parse_tool_name(text) in CANONICAL_TOOLS:
            return [True] * n
    return mask


def align_char_mask(text: str, char_mask: Sequence[bool], offsets: Sequence[tuple[int, int]]) -> list[bool]:
    out: list[bool] = []
    for a, b in offsets:
        if a >= b or a >= len(char_mask):
            out.append(False)
            continue
        chunk = char_mask[a : min(b, len(char_mask))]
        out.append(any(chunk))
    return out


def _offset_mapping(tokenizer, text: str) -> list[tuple[int, int]]:
    try:
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        if "offset_mapping" in enc and enc["offset_mapping"] is not None:
            return [(int(a), int(b)) for a, b in enc["offset_mapping"]]
    except Exception:  # noqa: BLE001
        pass
    ids = tokenizer.encode(text, add_special_tokens=False)
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for tid in ids:
        piece = tokenizer.decode([tid], skip_special_tokens=False)
        idx = text.find(piece, cursor)
        if idx < 0:
            idx = cursor
            end = min(len(text), idx + max(1, len(piece)))
        else:
            end = idx + len(piece)
        offsets.append((idx, end))
        cursor = end
    return offsets


def infer_lora_targets(model: torch.nn.Module) -> list[str]:
    preferred = ["q_proj", "k_proj", "v_proj", "o_proj"]
    names = {n.split(".")[-1] for n, _ in model.named_modules()}
    hit = [m for m in preferred if m in names]
    if hit:
        return hit
    # gpt-oss fused variants
    alt = [m for m in ("qkv_proj", "o_proj", "sinks") if m in names]
    if alt:
        return alt
    linear = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and "lm_head" not in name and "embed" not in name:
            leaf = name.split(".")[-1]
            if leaf not in linear:
                linear.append(leaf)
    return linear[:8] or ["q_proj"]


@dataclass
class CleanSFTTrainer:
    model_path: str
    device_map: str | dict[str, int] = "auto"
    torch_dtype: torch.dtype = torch.bfloat16
    learning_rate: float = 5e-6
    lora_r: int = 32
    lora_alpha: int = 32
    max_length: int = 4096
    mask_mode: MaskMode = "full"
    adapter_path: str | None = None

    def __post_init__(self) -> None:
        tok_src = self.model_path
        adapter = Path(self.adapter_path or "") if self.adapter_path else Path(self.model_path)
        resume = (adapter / "adapter_config.json").is_file() if self.adapter_path else False
        self.tokenizer = AutoTokenizer.from_pretrained(
            tok_src if Path(tok_src, "tokenizer_config.json").exists() else self.model_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        load_kw: dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": self.torch_dtype,
            "device_map": self.device_map,
        }
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, attn_implementation="sdpa", **load_kw
            )
        except Exception:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, **load_kw
            )
        self.model.config.use_cache = False
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        from peft import LoraConfig, PeftModel, get_peft_model, TaskType

        if resume:
            self.model = PeftModel.from_pretrained(self.model, str(adapter), is_trainable=True)
            for name, p in self.model.named_parameters():
                if "lora_" in name:
                    p.requires_grad = True
            self.lora_targets = ["resumed_adapter"]
        else:
            targets = infer_lora_targets(self.model)
            cfg = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                target_modules=targets,
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            self.model = get_peft_model(self.model, cfg)
            self.lora_targets = targets
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.learning_rate,
        )
        self._device = next(self.model.parameters()).device

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=False)

    def labels_for_example(self, example: dict[str, Any]) -> tuple[list[int], list[int]]:
        """Return (input_ids, labels) with -100 on tokens that should not receive loss."""
        if "input_ids" in example and example["input_ids"]:
            ids = [int(x) for x in example["input_ids"]]
            n_ctx = int(example.get("n_context") or 0)
        else:
            prompt = example["prompt_text"]
            response = example["response_text"]
            p_ids = self.encode(prompt)
            r_ids = self.encode(response)
            ids = p_ids + r_ids
            n_ctx = len(p_ids)
        if len(ids) > self.max_length:
            overflow = len(ids) - self.max_length
            # keep the tail (includes the action target)
            ids = ids[overflow:]
            n_ctx = max(0, n_ctx - overflow)
        labels = [-100] * len(ids)
        resp_ids = ids[n_ctx:]
        if self.mask_mode == "full":
            for i in range(n_ctx, len(ids)):
                labels[i] = ids[i]
            return ids, labels
        resp_text = example.get("response_text") or self.decode(resp_ids)
        char_mask = (
            format_aware_char_mask(resp_text)
            if self.mask_mode == "format_aware"
            else tool_char_mask(resp_text)
        )
        offsets = _offset_mapping(self.tokenizer, resp_text)
        tok_mask = align_char_mask(resp_text, char_mask, offsets)
        if len(tok_mask) != len(resp_ids):
            # length drift: mark all response tokens that look like a tool action
            keep = parse_tool_name(resp_text) in CANONICAL_TOOLS
            tok_mask = [keep] * len(resp_ids)
        if not any(tok_mask) and parse_tool_name(resp_text) in CANONICAL_TOOLS:
            tok_mask = [True] * len(resp_ids)
        for j, keep in enumerate(tok_mask):
            if keep and n_ctx + j < len(ids):
                labels[n_ctx + j] = ids[n_ctx + j]
        return ids, labels

    def train_step(self, batch: Sequence[dict[str, Any]]) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        n_tok = 0
        for ex in batch:
            ids, labels = self.labels_for_example(ex)
            if not any(l != -100 for l in labels):
                continue
            inp = torch.tensor([ids], device=self._device)
            lab = torch.tensor([labels], device=self._device)
            out = self.model(input_ids=inp, labels=lab)
            loss = out.loss
            if loss is None or not torch.isfinite(loss):
                continue
            loss.backward()
            losses.append(float(loss.detach().item()))
            n_tok += sum(1 for l in labels if l != -100)
        if losses:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], 1.0
            )
            self.optimizer.step()
        self.model.eval()
        return {
            "loss": sum(losses) / max(1, len(losses)),
            "n": float(len(losses)),
            "n_loss_tokens": float(n_tok),
            "mask_mode": self.mask_mode,
        }

    @torch.no_grad()
    def generate(self, prompt: str, *, max_new_tokens: int = 256) -> str:
        self.model.eval()
        ids = self.encode(prompt)
        inp = torch.tensor([ids], device=self._device)
        gen = self.model.generate(
            inp,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        new_ids = gen[0, len(ids) :].tolist()
        return self.decode(new_ids)

    def save_pretrained(self, out_dir: str) -> None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(out_dir)
        self.tokenizer.save_pretrained(out_dir)

    def merge_and_save(self, out_dir: str) -> None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        merged = self.model.merge_and_unload()
        merged.save_pretrained(out_dir)
        self.tokenizer.save_pretrained(out_dir)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _is_adapter_dir(path: Path) -> bool:
    return (path / "adapter_config.json").is_file() and (
        (path / "adapter_model.safetensors").is_file()
        or (path / "adapter_model.bin").is_file()
    )


def _resolve_parent_adapter(path: Path, parent_adapter: str | None = None) -> str | None:
    """OPD LoRA is trained after merging a Clean-SFT adapter. Eval must reload that parent.

    Without this, Peft loads the OPD adapter onto raw gpt-oss and the tool channel collapses.
    """
    if parent_adapter:
        p = Path(parent_adapter)
        return str(p) if _is_adapter_dir(p) else None
    marker = path / "parent_adapter.json"
    if marker.is_file():
        try:
            blob = json.loads(marker.read_text(encoding="utf-8"))
            cand = blob.get("parent_adapter") or blob.get("path")
            if cand and _is_adapter_dir(Path(cand)):
                return str(Path(cand))
        except json.JSONDecodeError:
            pass
    summary = path.parent / "summary.json"
    if summary.is_file():
        try:
            blob = json.loads(summary.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            blob = {}
        for key in ("base_checkpoint", "parent_adapter", "sft_adapter", "clean_auto_base"):
            cand = blob.get(key)
            if not cand:
                continue
            p = Path(str(cand))
            if _is_adapter_dir(p) and p.resolve() != path.resolve():
                return str(p)
    return None


def load_causal_lm(
    model_path: str,
    *,
    device_map: str,
    base_model: str | None = None,
    parent_adapter: str | None = None,
):
    """Load a full HF model or a PEFT adapter sitting on gpt-oss-20b.

    If `model_path` is an OPD LoRA trained on a merged Clean-SFT student, the SFT
    adapter is merged first, then the OPD adapter is applied.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = Path(model_path)
    tok_src = str(path if (path / "tokenizer_config.json").exists() else (base_model or model_path))
    tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    adapter = _is_adapter_dir(path) or (path / "adapter_config.json").is_file()
    load_kw = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "device_map": device_map,
    }
    parent = _resolve_parent_adapter(path, parent_adapter) if adapter else None
    foundation = base_model or "/data/ppnm/models/gpt-oss-20b"
    if adapter:
        from peft import PeftModel

        model = AutoModelForCausalLM.from_pretrained(foundation, **load_kw)
        if parent:
            model = PeftModel.from_pretrained(model, parent)
            model = model.merge_and_unload()
        model = PeftModel.from_pretrained(model, str(path))
    else:
        model = AutoModelForCausalLM.from_pretrained(str(path), **load_kw)
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True
    return tokenizer, model


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n
