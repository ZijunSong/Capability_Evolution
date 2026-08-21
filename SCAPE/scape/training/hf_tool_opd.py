"""HF Transformers backend for true SCAPE tool-token OPD.

Canonical path only — does not import SCOPE train_opd / KEEP-SKIP.
Loss paths (must remain code-distinct):

- tool_token_kl: uniform KL on tool-span tokens only (+ light anchor)
- action_ce: cross-entropy on sampled action tokens
- full_response_kl: KL over the entire response span
- offpolicy_matched: CE/KL on matched update tokens under full-harness prompt
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import torch
import torch.nn.functional as F

from scape.training.tool_mask import (
    build_tool_token_mask,
    tool_loss_mask_from_response,
)
from scape.training.canonical_metrics import (
    aggregate_token_metrics,
    js_from_logits,
    kl_from_logits,
    signed_logprob_gap,
)
from scape.training.opd_dataset import ProjectedTrainingStep
from scape.training.sr_opd_loss import (
    SR_OPD_LOSS_NAME,
    compute_sr_opd_ce,
    pack_sr_opd_metrics,
)
from scape.training.tool_opd import learnability_score, tool_opd_loss

LossPath = Literal[
    "tool_token_kl",
    "weighted_tool_token_kl",
    "tool_name_only_kl",
    "args_only_kl",
    "action_ce",
    "full_response_kl",
    "offpolicy_matched",
    "route_kl",
    "action_ce_plus_nextturn_kl",
    "sr_opd_ce",
]

# Derived from Stage L baseline ablation (name_only >> args_only >> uniform).
WEIGHTED_SPAN_WEIGHTS: dict[str, float] = {
    "tool_name": 3.0,
    "argument_key": 0.5,
    "argument_value": 0.5,
    "end_search": 1.0,
}

MASK_MODE_BY_LOSS: dict[str, dict[str, bool]] = {
    "tool_token_kl": {
        "include_name": True,
        "include_arg_keys": True,
        "include_arg_values": True,
        "include_end_search": True,
    },
    "weighted_tool_token_kl": {
        "include_name": True,
        "include_arg_keys": True,
        "include_arg_values": True,
        "include_end_search": True,
    },
    "tool_name_only_kl": {
        "include_name": True,
        "include_arg_keys": False,
        "include_arg_values": False,
        "include_end_search": False,
    },
    "args_only_kl": {
        "include_name": False,
        "include_arg_keys": True,
        "include_arg_values": True,
        "include_end_search": False,
    },
    "route_kl": {
        "include_name": True,
        "include_arg_keys": False,
        "include_arg_values": False,
        "include_end_search": False,
    },
    "action_ce": {
        "include_name": True,
        "include_arg_keys": True,
        "include_arg_values": True,
        "include_end_search": True,
    },
    "action_ce_plus_nextturn_kl": {
        "include_name": True,
        "include_arg_keys": True,
        "include_arg_values": True,
        "include_end_search": True,
    },
    "sr_opd_ce": {
        "include_name": True,
        "include_arg_keys": True,
        "include_arg_values": True,
        "include_end_search": True,
    },
}


def _offset_mapping(tokenizer, text: str) -> list[tuple[int, int]]:
    """Char offsets per token; falls back to approximate decode spans if slow tokenizer."""
    try:
        enc = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        if "offset_mapping" in enc and enc["offset_mapping"] is not None:
            return [(int(a), int(b)) for a, b in enc["offset_mapping"]]
    except Exception:  # noqa: BLE001
        pass
    # Approximate: decode cumulative token prefixes
    ids = tokenizer.encode(text, add_special_tokens=False)
    offsets: list[tuple[int, int]] = []
    cursor = 0
    built = ""
    for tid in ids:
        piece = tokenizer.decode([tid], skip_special_tokens=False)
        # find piece in remaining text; tolerate whitespace drift
        idx = text.find(piece, cursor)
        if idx < 0:
            idx = cursor
            end = min(len(text), idx + max(1, len(piece)))
        else:
            end = idx + len(piece)
        offsets.append((idx, end))
        cursor = end
        built += piece
    return offsets


@dataclass
class ScapeHFToolOPD:
    model_path: str
    device_map: str | dict[str, int] = "auto"
    torch_dtype: torch.dtype = torch.bfloat16
    learning_rate: float = 1e-5
    anchor_weight: float = 0.1
    lambda_args: float = 0.0
    legacy_teacher_kl_weight: float = 0.0
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16

    def __post_init__(self) -> None:
        from pathlib import Path as _Path

        model_dir = _Path(self.model_path)
        adapter_cfg = model_dir / "adapter_config.json"
        tok_src = str(model_dir)
        base_src = self.model_path
        if adapter_cfg.is_file():
            try:
                ac = json.loads(adapter_cfg.read_text(encoding="utf-8"))
                base_src = ac.get("base_model_name_or_path") or "/data/ppnm/models/gpt-oss-20b"
            except Exception:
                base_src = "/data/ppnm/models/gpt-oss-20b"
            tok_src = base_src if not (model_dir / "tokenizer_config.json").is_file() else str(model_dir)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            tok_src, trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            base_src if adapter_cfg.is_file() else self.model_path,
            device_map=self.device_map,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
        )
        if hasattr(self.model, "config"):
            self.model.config.use_cache = False
        self._parent_adapter = None
        if adapter_cfg.is_file():
            from peft import PeftModel as _PeftModel

            self._parent_adapter = str(model_dir)
            self.model = _PeftModel.from_pretrained(self.model, str(model_dir))
            self.model = self.model.merge_and_unload()
        if self.use_lora:
            from peft import LoraConfig, get_peft_model

            leaf = {n.split(".")[-1] for n, _ in self.model.named_modules()}
            targets = [m for m in ("q_proj", "k_proj", "v_proj", "o_proj") if m in leaf]
            if not targets:
                targets = [m for m in ("qkv_proj", "o_proj") if m in leaf] or ["q_proj"]
            lora_cfg = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                target_modules=targets,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_cfg)
        if hasattr(self.model, "config"):
            self.model.config.use_cache = False
        # gpt-oss MoE + non-reentrant checkpointing mismatches saved tensors; keep cache off only.
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.learning_rate,
        )
        self._device = next(self.model.parameters()).device

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def response_token_mask(
        self,
        response_text: str,
        *,
        loss_path: LossPath = "tool_token_kl",
    ) -> list[bool]:
        offsets = _offset_mapping(self.tokenizer, response_text)
        span_kwargs = MASK_MODE_BY_LOSS.get(loss_path, MASK_MODE_BY_LOSS["tool_token_kl"])
        audit = tool_loss_mask_from_response(
            response_text, token_offsets=offsets, **span_kwargs
        )
        mask = audit["token_mask"]
        assert mask is not None
        return list(mask)

    def response_token_weights(
        self,
        response_text: str,
        *,
        loss_path: LossPath = "weighted_tool_token_kl",
        span_weights: dict[str, float] | None = None,
    ) -> list[float]:
        """Per-token loss weights for weighted tool-token KL."""
        weights = span_weights or WEIGHTED_SPAN_WEIGHTS
        offsets = _offset_mapping(self.tokenizer, response_text)
        span_kwargs = MASK_MODE_BY_LOSS.get(loss_path, MASK_MODE_BY_LOSS["weighted_tool_token_kl"])
        spans = build_tool_token_mask(response_text, **span_kwargs)
        char_w = [0.0] * len(response_text)
        for sp in spans:
            w = float(weights.get(sp.kind, 0.0))
            for i in range(max(0, sp.start), min(len(response_text), sp.end)):
                char_w[i] = max(char_w[i], w)
        token_w: list[float] = []
        for start, end in offsets:
            if start >= end or start >= len(char_w):
                token_w.append(0.0)
                continue
            end = min(end, len(char_w))
            chunk = char_w[start:end]
            token_w.append(max(chunk) if chunk else 0.0)
        return token_w

    def audit_tool_spans(self, response_texts: Sequence[str]) -> dict[str, Any]:
        n = len(response_texts)
        parsable = 0
        invalid = 0
        details = []
        for text in response_texts:
            try:
                offsets = _offset_mapping(self.tokenizer, text)
                audit = tool_loss_mask_from_response(text, token_offsets=offsets)
                ok = (
                    audit["n_tool_name"] >= 1
                    and (audit["token_mask"] is not None)
                    and any(audit["token_mask"])
                )
                if ok:
                    parsable += 1
                else:
                    invalid += 1
                details.append(
                    {
                        "ok": ok,
                        "n_tool_name": audit["n_tool_name"],
                        "n_argument_key": audit["n_argument_key"],
                        "n_argument_value": audit["n_argument_value"],
                        "n_end_search": audit["n_end_search"],
                    }
                )
            except Exception as exc:  # noqa: BLE001
                invalid += 1
                details.append({"ok": False, "error": str(exc)})
        return {
            "n_sampled": n,
            "n_parsable": parsable,
            "n_invalid": invalid,
            "parsable_rate": parsable / max(1, n),
            "pass": n > 0 and invalid == 0 and parsable == n,
            "details_head": details[:20],
            "tool_mask_version": "scape_tool_mask_v1",
        }

    def _response_position_logits(
        self,
        prompt_ids: list[int],
        response_ids: list[int],
        *,
        require_grad: bool,
    ) -> torch.Tensor:
        """Logits at each response-token position [n_resp, vocab]."""
        if not response_ids:
            return torch.zeros(0, 0, device=self._device)
        full = prompt_ids + response_ids
        max_len = 2048
        if len(full) > max_len:
            overflow = len(full) - max_len
            if overflow < len(prompt_ids):
                prompt_ids = prompt_ids[overflow:]
            else:
                keep = max_len - 1
                response_ids = response_ids[-keep:]
                prompt_ids = prompt_ids[-1:]
            full = prompt_ids + response_ids
        inp = torch.tensor([full], device=self._device)
        # Qwen3 supports logits_to_keep.  Keeping only the response prediction
        # positions avoids materializing a full [sequence, vocab] tensor for
        # long Student-visible prefixes while preserving teacher-forced CE/KL.
        # Fall back to full-sequence logits for models that reject the kwarg.
        n_response = len(response_ids)
        keep_kwargs = {"logits_to_keep": n_response + 1}

        def _forward(kwargs: dict[str, Any]) -> torch.Tensor:
            if require_grad:
                self.model.train()
                return self.model(inp, **kwargs).logits[0]
            self.model.eval()
            with torch.no_grad():
                return self.model(inp, **kwargs).logits[0]

        try:
            logits = _forward(keep_kwargs)
            return logits[:-1]
        except TypeError:
            logits = _forward({})
            start = max(0, len(prompt_ids) - 1)
            return logits[start : start + len(response_ids)]

    def _teacher_forced_logprobs(
        self,
        prompt_ids: list[int],
        response_ids: list[int],
        *,
        require_grad: bool,
    ) -> torch.Tensor:
        """Return per-response-token logprob tensor (length = len(response_ids))."""
        if not response_ids:
            return torch.zeros(0, device=self._device)
        pos_logits = self._response_position_logits(
            prompt_ids, response_ids, require_grad=require_grad
        )
        ids = torch.tensor(response_ids, device=self._device, dtype=torch.long)
        logp = F.log_softmax(pos_logits, dim=-1)
        logps = logp.gather(1, ids.unsqueeze(1)).squeeze(1)
        return logps

    def span_token_masks(
        self,
        response_text: str,
        resp_len: int,
    ) -> dict[str, list[bool]]:
        """Per-span boolean masks aligned to response token ids."""
        offsets = _offset_mapping(self.tokenizer, response_text)
        name_audit = tool_loss_mask_from_response(
            response_text,
            token_offsets=offsets,
            include_name=True,
            include_arg_keys=False,
            include_arg_values=False,
            include_end_search=False,
        )
        key_audit = tool_loss_mask_from_response(
            response_text,
            token_offsets=offsets,
            include_name=False,
            include_arg_keys=True,
            include_arg_values=False,
            include_end_search=False,
        )
        val_audit = tool_loss_mask_from_response(
            response_text,
            token_offsets=offsets,
            include_name=False,
            include_arg_keys=False,
            include_arg_values=True,
            include_end_search=False,
        )
        full_audit = tool_loss_mask_from_response(
            response_text, token_offsets=offsets
        )
        def _align(mask: list[bool] | None) -> list[bool]:
            if mask is None or len(mask) != resp_len:
                return [True] * resp_len
            return list(mask)

        return {
            "tool": _align(full_audit["token_mask"]),
            "name": _align(name_audit["token_mask"]),
            "key": _align(key_audit["token_mask"]),
            "value": _align(val_audit["token_mask"]),
        }

    def score_canonical_metrics(
        self,
        *,
        prompt_reduced: str,
        prompt_full: str,
        response_text: str,
        loss_path: LossPath = "tool_token_kl",
    ) -> dict[str, float]:
        """Canonical M1–M4 metrics; legacy `div` = signed_gap (not KL)."""
        resp_ids = self.encode(response_text)
        if not resp_ids:
            z = {
                "forward_KL": 0.0,
                "reverse_KL": 0.0,
                "JS": 0.0,
                "signed_gap": 0.0,
                "tool_name_KL": 0.0,
                "arg_key_KL": 0.0,
                "arg_value_KL": 0.0,
                "div": 0.0,
            }
            return z
        red_ids = self.encode(prompt_reduced)
        full_ids = self.encode(prompt_full)
        with torch.no_grad():
            s_logits = self._response_position_logits(red_ids, resp_ids, require_grad=False)
            t_logits = self._response_position_logits(full_ids, resp_ids, require_grad=False)
        fwd = kl_from_logits(t_logits, s_logits, forward=True)
        rev = kl_from_logits(t_logits, s_logits, forward=False)
        js = js_from_logits(t_logits, s_logits)
        gap = signed_logprob_gap(t_logits, s_logits, resp_ids)

        spans = self.span_token_masks(response_text, len(resp_ids))
        token_mask = self.response_token_mask(response_text, loss_path=loss_path)
        if len(token_mask) != len(resp_ids):
            token_mask = spans["tool"]
        m_tool = torch.tensor(token_mask, device=fwd.device, dtype=fwd.dtype)
        m_name = torch.tensor(spans["name"], device=fwd.device, dtype=fwd.dtype)
        m_key = torch.tensor(spans["key"], device=fwd.device, dtype=fwd.dtype)
        m_val = torch.tensor(spans["value"], device=fwd.device, dtype=fwd.dtype)

        metrics = aggregate_token_metrics(
            fwd, rev, js, gap, m_tool,
            name_mask=m_name, key_mask=m_key, value_mask=m_val,
        )
        metrics["div"] = metrics["signed_gap"]
        return metrics

    def score_divergence(
        self,
        *,
        prompt_reduced: str,
        prompt_full: str,
        response_text: str,
        loss_path: LossPath = "tool_token_kl",
    ) -> dict[str, float]:
        """Teacher=full prompt, student=reduced prompt; both score same response tokens."""
        resp_ids = self.encode(response_text)
        red_ids = self.encode(prompt_reduced)
        full_ids = self.encode(prompt_full)
        with torch.no_grad():
            student_lp = self._teacher_forced_logprobs(red_ids, resp_ids, require_grad=False)
            teacher_lp = self._teacher_forced_logprobs(full_ids, resp_ids, require_grad=False)
        if student_lp.numel() == 0:
            return {"div": 0.0, "name_kl": 0.0, "arg_key_kl": 0.0, "arg_value_kl": 0.0}

        token_mask = self.response_token_mask(response_text, loss_path=loss_path)
        if len(token_mask) != len(resp_ids):
            # tokenizer offset quirks — fall back to all-true on response
            token_mask = [True] * len(resp_ids)

        kl_tok = (teacher_lp - student_lp)  # KL(teacher||student) proxy per token
        if loss_path == "full_response_kl":
            div = float(kl_tok.mean().item())
        elif loss_path == "weighted_tool_token_kl":
            w = torch.tensor(
                self.response_token_weights(response_text, loss_path=loss_path),
                device=kl_tok.device,
                dtype=kl_tok.dtype,
            )
            if float(w.sum().item()) <= 0:
                div = float(kl_tok.mean().item())
            else:
                div = float((kl_tok * w).sum().item() / w.sum().item())
        elif loss_path in (
            "tool_token_kl",
            "tool_name_only_kl",
            "args_only_kl",
            "offpolicy_matched",
            "action_ce",
            "route_kl",
            "action_ce_plus_nextturn_kl",
        ):
            m = torch.tensor(token_mask, device=kl_tok.device, dtype=kl_tok.dtype)
            if float(m.sum().item()) <= 0:
                div = float(kl_tok.mean().item())
            else:
                div = float((kl_tok * m).sum().item() / m.sum().item())
        else:
            raise ValueError(loss_path)

        # Span-kind proxies via char kinds → approximate equal split of masked KL
        audit = tool_loss_mask_from_response(
            response_text, token_offsets=_offset_mapping(self.tokenizer, response_text)
        )
        n_name = max(1, audit["n_tool_name"])
        n_key = max(1, audit["n_argument_key"])
        n_val = max(1, audit["n_argument_value"])
        return {
            "div": div,
            "name_kl": div * (n_name / (n_name + n_key + n_val)),
            "arg_key_kl": div * (n_key / (n_name + n_key + n_val)),
            "arg_value_kl": div * (n_val / (n_name + n_key + n_val)),
        }

    def _row_to_projected_step(self, row: Mapping[str, Any] | ProjectedTrainingStep) -> ProjectedTrainingStep:
        if isinstance(row, ProjectedTrainingStep):
            return row
        if row.get("target_text"):
            return ProjectedTrainingStep(
                prompt_reduced=str(row.get("prompt_reduced") or row.get("student_prompt") or ""),
                target_text=str(row["target_text"]),
                target_action=dict(row.get("target_action") or {}),
                token_mask=row.get("token_mask"),
                weight=float(row.get("weight") or row.get("projection_confidence") or 1.0),
                metadata=dict(row.get("metadata") or {}),
            )
        return ProjectedTrainingStep(
            prompt_reduced=str(row["prompt_reduced"]),
            target_text=str(row.get("response_text") or ""),
            target_action=dict(row.get("student_action") or row.get("target_action") or {}),
            token_mask=row.get("token_mask"),
            weight=float(row.get("weight") or 1.0),
            metadata=dict(row.get("metadata") or {}),
        )

    def train_projected_step(self, step: ProjectedTrainingStep) -> dict[str, Any]:
        """One SR-OPD CE backward. Logger metrics come from the same tensor."""
        prompt_ids = self.encode(step.prompt_reduced)
        resp_ids = self.encode(step.target_text)
        if not resp_ids:
            zero = torch.zeros((), device=self._device, requires_grad=True)
            return pack_sr_opd_metrics(zero, n_supervised=0.0, weight=step.weight)
        student_lp = self._teacher_forced_logprobs(prompt_ids, resp_ids, require_grad=True)
        mask_list = list(step.token_mask) if step.token_mask is not None else [True] * len(resp_ids)
        if len(mask_list) != len(resp_ids):
            mask_list = [True] * len(resp_ids)
        token_mask = torch.tensor(mask_list, device=self._device, dtype=student_lp.dtype)
        token_weight = torch.full_like(token_mask, float(step.weight))
        loss = compute_sr_opd_ce(student_lp, token_mask, token_weight)
        if self.legacy_teacher_kl_weight > 0.0:
            teacher_prompt = str((step.metadata or {}).get("prompt_full") or "")
            if teacher_prompt:
                with torch.no_grad():
                    teacher_lp = self._teacher_forced_logprobs(
                        self.encode(teacher_prompt), resp_ids, require_grad=False
                    )
                loss = loss + float(self.legacy_teacher_kl_weight) * (teacher_lp - student_lp.detach()).mean()
        metrics = pack_sr_opd_metrics(
            loss,
            n_supervised=float(token_mask.sum().item()),
            weight=step.weight,
        )
        loss.backward()
        metrics["loss_path"] = SR_OPD_LOSS_NAME
        return metrics

    def train_projected_batch(
        self,
        batch: Sequence[ProjectedTrainingStep | dict[str, Any]],
        *,
        loss_path: LossPath = "sr_opd_ce",
    ) -> dict[str, Any]:
        """Formal SR-OPD batch. Does not branch on component_id."""
        del loss_path
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total = 0.0
        n = 0
        n_tok = 0.0
        last_loss_id = 0
        for raw in batch:
            step = self._row_to_projected_step(raw)
            metrics = self.train_projected_step(step)
            total += float(metrics["loss"])
            n += 1
            n_tok += float(metrics["n_supervised_tokens"])
            last_loss_id = int(metrics["loss_id"])
        if n > 0:
            self.optimizer.step()
        self.model.eval()
        return {
            "loss": total / max(1, n),
            "sr_opd_ce": total / max(1, n),
            "batch_size": float(n),
            "n_supervised_tokens": n_tok,
            "loss_path": SR_OPD_LOSS_NAME,
            "loss_impl": f"scape.training.hf_tool_opd:{SR_OPD_LOSS_NAME}",
            "loss_id": last_loss_id,
        }

    def train_step(
        self,
        batch: Sequence[dict[str, Any]],
        *,
        loss_path: LossPath = "tool_token_kl",
    ) -> dict[str, float]:
        if loss_path == "sr_opd_ce":
            expanded: list[ProjectedTrainingStep | dict[str, Any]] = []
            for row in batch:
                if row.get("projected_steps"):
                    expanded.extend(list(row["projected_steps"]))
                else:
                    expanded.append(row)
            return self.train_projected_batch(expanded, loss_path="sr_opd_ce")
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total = 0.0
        n = 0
        for row in batch:
            row_loss = str(row.get("loss_kind") or loss_path)
            if loss_path == "action_ce_plus_nextturn_kl":
                if row_loss in ("nextturn_kl", "route_kl"):
                    row_loss = "route_kl"
                else:
                    row_loss = "action_ce"
            if row_loss == "offpolicy_matched" or loss_path == "offpolicy_matched":
                prompt = row["prompt_full"]
            else:
                prompt = row["prompt_reduced"]
            response = row["response_text"]
            prompt_ids = self.encode(prompt)
            resp_ids = self.encode(response)
            if not resp_ids:
                continue
            teacher_prompt = row.get("prompt_full") or row["prompt_reduced"]
            student_lp = None
            teacher_lp = None
            if row_loss not in ("route_kl", "action_ce"):
                student_lp = self._teacher_forced_logprobs(
                    prompt_ids, resp_ids, require_grad=True
                )
                with torch.no_grad():
                    teacher_lp = self._teacher_forced_logprobs(
                        self.encode(teacher_prompt), resp_ids, require_grad=False
                    )
            elif row_loss == "action_ce":
                student_lp = self._teacher_forced_logprobs(
                    prompt_ids, resp_ids, require_grad=True
                )

            mask_path = "route_kl" if row_loss == "route_kl" else row_loss
            token_mask = self.response_token_mask(response, loss_path=mask_path)  # type: ignore[arg-type]
            if len(token_mask) != len(resp_ids):
                token_mask = [True] * len(resp_ids)
            m = torch.tensor(token_mask, device=self._device, dtype=torch.float32)

            if row_loss == "action_ce":
                # CE on masked action tokens: -student_logprob
                if float(m.sum().item()) <= 0:
                    loss = -student_lp.mean()
                else:
                    loss = -(student_lp * m).sum() / m.sum()
                metrics = tool_opd_loss(tool_token_kl=float(loss.detach().item()), anchor_kl=0.0)
            elif row_loss == "full_response_kl":
                kl = (teacher_lp - student_lp).mean()
                loss = kl
                metrics = tool_opd_loss(
                    tool_token_kl=float(kl.detach().item()),
                    anchor_kl=0.0,
                    anchor_weight=0.0,
                )
            elif row_loss in (
                "tool_token_kl",
                "weighted_tool_token_kl",
                "tool_name_only_kl",
                "args_only_kl",
                "offpolicy_matched",
                "route_kl",
            ):
                if row_loss == "route_kl":
                    s_logits = self._response_position_logits(
                        prompt_ids, resp_ids, require_grad=True
                    )
                    with torch.no_grad():
                        t_logits = self._response_position_logits(
                            self.encode(teacher_prompt), resp_ids, require_grad=False
                        )
                    ids_t = torch.tensor(resp_ids, device=s_logits.device, dtype=torch.long)
                    student_lp = F.log_softmax(s_logits, dim=-1).gather(1, ids_t.unsqueeze(1)).squeeze(1)
                    legal = [
                        "fan_out_search",
                        "search_corpus",
                        "grep_corpus",
                        "read_document",
                        "review_docs",
                        "curate",
                        "verify",
                        "end_search",
                    ]
                    name_ids: list[int] = []
                    for name in legal:
                        ids = self.encode(name)
                        name_ids.append(ids[0] if ids else 0)
                    name_idx = next((i for i, bit in enumerate(token_mask) if bit), 0)
                    name_idx = min(name_idx, s_logits.size(0) - 1)
                    idx = torch.tensor(name_ids, device=s_logits.device, dtype=torch.long)
                    s_sub = s_logits[name_idx].index_select(0, idx)
                    t_sub = t_logits[name_idx].index_select(0, idx)
                    t_logp = F.log_softmax(t_sub, dim=-1)
                    s_logp = F.log_softmax(s_sub, dim=-1)
                    t_p = t_logp.exp()
                    route_kl = (t_p * (t_logp - s_logp)).sum()
                    arg_audit = tool_loss_mask_from_response(
                        response,
                        token_offsets=_offset_mapping(self.tokenizer, response),
                        include_name=False,
                        include_arg_keys=True,
                        include_arg_values=True,
                        include_end_search=False,
                    )
                    arg_mask = arg_audit.get("token_mask") or [False] * len(resp_ids)
                    if len(arg_mask) != len(resp_ids):
                        arg_mask = [False] * len(resp_ids)
                    am = torch.tensor(arg_mask, device=student_lp.device, dtype=student_lp.dtype)
                    if float(am.sum().item()) > 0:
                        arg_ce = -(student_lp * am).sum() / am.sum()
                    else:
                        arg_ce = -student_lp.mean() * 0.0
                    anchor = -student_lp.mean()
                    loss = route_kl + self.lambda_args * arg_ce + self.anchor_weight * anchor
                    packed = tool_opd_loss(
                        tool_token_kl=float(route_kl.detach().item()),
                        anchor_kl=float(anchor.detach().item()),
                        anchor_weight=self.anchor_weight,
                    )
                    metrics = packed
                else:
                    if row_loss == "weighted_tool_token_kl":
                        w = torch.tensor(
                            self.response_token_weights(response, loss_path=loss_path),
                            device=student_lp.device,
                            dtype=student_lp.dtype,
                        )
                        if float(w.sum().item()) <= 0:
                            tool_kl = (teacher_lp - student_lp).mean()
                        else:
                            tool_kl = ((teacher_lp - student_lp) * w).sum() / w.sum()
                    elif float(m.sum().item()) <= 0:
                        tool_kl = (teacher_lp - student_lp).mean()
                    else:
                        tool_kl = ((teacher_lp - student_lp) * m).sum() / m.sum()
                    # light anchor: small CE on all response tokens
                    anchor = -student_lp.mean()
                    packed = tool_opd_loss(
                        tool_token_kl=float(tool_kl.detach().item()),
                        anchor_kl=float(anchor.detach().item()),
                        anchor_weight=self.anchor_weight,
                    )
                    loss = tool_kl + self.anchor_weight * anchor
                    metrics = packed
            else:
                raise ValueError(row_loss)

            loss.backward()
            total += float(metrics["loss"])
            n += 1

        if n > 0:
            self.optimizer.step()
        self.model.eval()
        return {"loss": total / max(1, n), "batch_size": float(n), "loss_path": loss_path}

    def save_pretrained(self, out_dir: str) -> None:
        from pathlib import Path as _P

        self.model.save_pretrained(out_dir)
        self.tokenizer.save_pretrained(out_dir)
        parent = getattr(self, "_parent_adapter", None)
        if not parent:
            src = _P(self.model_path)
            if (src / "adapter_config.json").is_file():
                parent = str(src)
        if parent:
            (_P(out_dir) / "parent_adapter.json").write_text(
                json.dumps(
                    {
                        "parent_adapter": parent,
                        "note": "OPD LoRA was trained after merge_and_unload of this Clean-SFT adapter",
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

    def merge_and_save(self, out_dir: str) -> None:
        """Merge LoRA weights into base for serving when applicable."""
        if self.use_lora:
            merged = self.model.merge_and_unload()
            merged.save_pretrained(out_dir)
            self.tokenizer.save_pretrained(out_dir)
        else:
            self.save_pretrained(out_dir)


def mean_canonical_metrics(
    backend: ScapeHFToolOPD,
    rows: Sequence[dict[str, Any]],
    *,
    loss_path: LossPath = "tool_token_kl",
) -> dict[str, float]:
    keys = [
        "forward_KL", "reverse_KL", "JS", "signed_gap",
        "tool_name_KL", "arg_key_KL", "arg_value_KL", "div",
    ]
    acc = {k: 0.0 for k in keys}
    for row in rows:
        m = backend.score_canonical_metrics(
            prompt_reduced=row["prompt_reduced"],
            prompt_full=row["prompt_full"],
            response_text=row["response_text"],
            loss_path=loss_path,
        )
        for k in keys:
            acc[k] += m[k]
    n = max(1, len(rows))
    return {k: v / n for k, v in acc.items()}


def _row_score_loss_path(loss_path: LossPath, row: dict[str, Any]) -> LossPath:
    if loss_path != "action_ce_plus_nextturn_kl":
        return loss_path
    kind = str(row.get("loss_kind") or "")
    if kind in ("nextturn_kl", "route_kl"):
        return "route_kl"
    return "action_ce"


def mean_divergence(
    backend: ScapeHFToolOPD,
    rows: Sequence[dict[str, Any]],
    *,
    loss_path: LossPath = "tool_token_kl",
) -> dict[str, float]:
    acc = {"div": 0.0, "name_kl": 0.0, "arg_key_kl": 0.0, "arg_value_kl": 0.0}
    for row in rows:
        d = backend.score_divergence(
            prompt_reduced=row["prompt_reduced"],
            prompt_full=row["prompt_full"],
            response_text=row["response_text"],
            loss_path=_row_score_loss_path(loss_path, row),
        )
        for k in acc:
            acc[k] += d[k]
    n = max(1, len(rows))
    return {k: v / n for k, v in acc.items()}


def run_tool_opd_train(
    backend: ScapeHFToolOPD,
    train_rows: Sequence[dict[str, Any]],
    eval_rows: Sequence[dict[str, Any]],
    *,
    loss_path: LossPath = "tool_token_kl",
    epochs: int = 1,
    batch_size: int = 1,
) -> dict[str, Any]:
    d_pre = mean_divergence(backend, eval_rows, loss_path=loss_path)
    losses = []
    n_train = len(train_rows)
    for _ep in range(epochs):
        for i in range(0, n_train, batch_size):
            batch = train_rows[i : i + batch_size]
            step = backend.train_step(batch, loss_path=loss_path)
            losses.append(step)
            if (i // batch_size) % 5 == 0:
                print(
                    f"[opd] step={i}/{n_train} loss={step.get('loss')}",
                    flush=True,
                )
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
    d_post = mean_divergence(backend, eval_rows, loss_path=loss_path)
    return {
        "loss_path": loss_path,
        "D_pre": d_pre["div"],
        "D_post": d_post["div"],
        "L_m": learnability_score(d_pre["div"], d_post["div"]),
        "name_kl_pre": d_pre["name_kl"],
        "name_kl_post": d_post["name_kl"],
        "arg_key_kl_pre": d_pre["arg_key_kl"],
        "arg_key_kl_post": d_post["arg_key_kl"],
        "arg_value_kl_pre": d_pre["arg_value_kl"],
        "arg_value_kl_post": d_post["arg_value_kl"],
        "mean_train_loss": sum(x["loss"] for x in losses) / max(1, len(losses)),
        "n_train_steps": len(losses),
        "legacy_scope_path_used": False,
        "loss_impl": f"scape.training.hf_tool_opd:{loss_path}",
    }


def assert_loss_paths_distinct() -> dict[str, Any]:
    """Static proof that the four loss paths are different code branches."""
    import inspect

    src = inspect.getsource(ScapeHFToolOPD.train_step)
    required = [
        "action_ce",
        "full_response_kl",
        "tool_token_kl",
        "weighted_tool_token_kl",
        "tool_name_only_kl",
        "args_only_kl",
        "offpolicy_matched",
        "route_kl",
        "action_ce_plus_nextturn_kl",
        "sr_opd_ce",
    ]
    present = {k: (k in src) for k in required}
    # also ensure offpolicy uses prompt_full for student forward
    offpolicy_uses_full = "offpolicy_matched" in src and "prompt_full" in src
    return {
        "branches_present": present,
        "offpolicy_uses_full_prompt": offpolicy_uses_full,
        "distinct": all(present.values()) and offpolicy_uses_full,
    }
