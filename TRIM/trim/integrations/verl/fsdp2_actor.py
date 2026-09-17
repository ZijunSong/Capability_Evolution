"""FSDP2 (fallback FSDP1) CISPO actor. One optimizer.step per rollout batch."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.distributed as dist

from trim.integrations.verl.joint_objective import cispo_clip_bounds, verl_cispo_clip_config
from trim.training.hf_rl_batch import (
    gather_response_logprobs,
    iter_length_microbatches,
    log_train,
    pack_left_pad_teacher_forced,
)


def _decoder_layer_classes(model: Any) -> list[type]:
    found: list[type] = []
    seen: set[type] = set()
    for mod in model.modules():
        cls = type(mod)
        if cls.__name__.endswith("DecoderLayer") and cls not in seen:
            seen.add(cls)
            found.append(cls)
    return found


def _wrap_fsdp2(model: Any, *, device: torch.device, world_size: int) -> Any:
    if world_size <= 1:
        return model.to(device)
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

        mesh = init_device_mesh("cuda", (world_size,))
        mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        for cls in _decoder_layer_classes(model):
            for mod in list(model.modules()):
                if type(mod) is cls:
                    fully_shard(mod, mesh=mesh, mp_policy=mp)
        fully_shard(model, mesh=mesh, mp_policy=mp)
        return model
    except Exception:
        from trim.training.hf_sft import _wrap_fsdp

        local_rank = int(os.environ.get("LOCAL_RANK") or 0)
        return _wrap_fsdp(model, local_rank=local_rank)


class FSDP2CispoActor:
    """LoRA actor owned by TRIM; FSDP2 owns sharding/optimizer step."""

    def __init__(
        self,
        *,
        model_path: str,
        adapter_dir: str | None = None,
        learning_rate: float = 1e-5,
        micro_batch_size: int = 4,
        max_full_tokens: int = 8192,
        lora_r: int = 8,
        lora_alpha: int = 16,
        optimizer_path: str | None = None,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self.micro_batch_size = max(1, int(micro_batch_size))
        self.max_full_tokens = int(max_full_tokens)
        clip = verl_cispo_clip_config()
        self.clip_low, self.clip_high = cispo_clip_bounds(**clip)
        self.model_path = str(model_path)
        self.adapter_dir = adapter_dir
        self.learning_rate = float(learning_rate)
        self.lora_r = int(lora_r)
        self.lora_alpha = int(lora_alpha)
        self.optimizer_path = optimizer_path
        self.model = None
        self.optimizer = None
        self.tokenizer = None
        self._load()

    def _load(self) -> None:
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok_src = self.model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        load_kw: dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
        }
        if self.world_size <= 1 and self.device.type == "cuda":
            load_kw["device_map"] = {"": str(self.device)}
            load_kw.pop("low_cpu_mem_usage", None)
        model = AutoModelForCausalLM.from_pretrained(self.model_path, **load_kw)
        if hasattr(model, "config"):
            model.config.use_cache = False
        if hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
        adapter = Path(self.adapter_dir or "")
        if adapter.is_dir() and (adapter / "adapter_config.json").is_file():
            model = PeftModel.from_pretrained(model, str(adapter), is_trainable=True)
        else:
            leaf = {n.split(".")[-1] for n, _ in model.named_modules()}
            targets = [m for m in ("q_proj", "k_proj", "v_proj", "o_proj") if m in leaf] or ["q_proj"]
            model = get_peft_model(
                model,
                LoraConfig(
                    r=self.lora_r,
                    lora_alpha=self.lora_alpha,
                    target_modules=targets,
                    lora_dropout=0.0,
                    bias="none",
                    task_type="CAUSAL_LM",
                ),
            )
        if self.rank == 0:
            try:
                model.print_trainable_parameters()
            except Exception:
                n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"[verl-fsdp2] trainable_params={n_train}", flush=True)
        if self.device.type == "cuda" and self.world_size > 1:
            model = model.to(self.device)
        self.model = _wrap_fsdp2(model, device=self.device, world_size=self.world_size)
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        )
        if self.optimizer_path and Path(self.optimizer_path).is_file():
            payload = torch.load(self.optimizer_path, map_location="cpu", weights_only=False)
            self.optimizer.load_state_dict(payload)

    def close(self) -> None:
        self.optimizer = None
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _forward_response_logprobs(
        self,
        pairs: Sequence[tuple[list[int], list[int]]],
        extra: Sequence[dict[str, Any]],
    ) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
        pad_id = int(self.tokenizer.pad_token_id or 0)
        packed = pack_left_pad_teacher_forced(
            pairs, pad_id=pad_id, device=self.device, max_full=self.max_full_tokens
        )
        kwargs: dict[str, Any] = {
            "input_ids": packed.input_ids,
            "attention_mask": packed.attention_mask,
            "use_cache": False,
        }
        keep = int(packed.max_resp) + 1
        try:
            out = self.model(**kwargs, logits_to_keep=keep)
        except TypeError:
            out = self.model(**kwargs)
        logits = out.logits
        if logits.shape[1] > keep:
            logits = logits[:, -keep:, :]
        logps = gather_response_logprobs(logits, packed.response_ids, max_resp=packed.max_resp)
        aligned = []
        for row, resp in zip(extra, packed.response_ids):
            item = dict(row)
            n = len(resp)
            item["action_ids"] = list(resp)
            raw_lp = list(item.get("token_logprobs") or [])
            if raw_lp:
                item["token_logprobs"] = raw_lp[:n]
                if len(item["token_logprobs"]) != n:
                    raise ValueError("behavior logprob length changed during pack")
            else:
                item["token_logprobs"] = []
            item["action_mask"] = list(item.get("action_mask") or [1] * n)[:n]
            aligned.append(item)
        return logps, aligned

    def update(
        self,
        rl_rows: Sequence[dict[str, Any]],
        opd_rows: Sequence[Any] | None = None,
        *,
        lambda_opd: float = 0.0,
    ) -> dict[str, Any]:
        """Accumulate CISPO (+ optional CE) then one optimizer.step. All ranks must enter."""
        assert self.model is not None and self.optimizer is not None
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        local = [dict(r) for r in rl_rows if list(r.get("action_ids") or r.get("response_ids") or [])]
        n_opt = 0
        metrics = {
            "n_rl_rows_local": len(local),
            "n_opd_rows_local": len(list(opd_rows or [])),
            "n_optimizer_steps": 0,
            "loss": 0.0,
        }
        token_count = 0
        for row in local:
            mask = list(row.get("action_mask") or [1] * len(row.get("action_ids") or []))
            token_count += sum(1 for m in mask if m)
        token_t = torch.tensor([float(token_count)], device=self.device, dtype=torch.float32)
        if self.world_size > 1:
            dist.all_reduce(token_t, op=dist.ReduceOp.SUM)
        global_tokens = max(1.0, float(token_t.item()))
        metrics["n_rl_tokens"] = int(global_tokens)
        has_local = 1 if (local or list(opd_rows or [])) else 0
        has_t = torch.tensor([has_local], device=self.device, dtype=torch.int32)
        if self.world_size > 1:
            dist.all_reduce(has_t, op=dist.ReduceOp.MAX)
        if int(has_t.item()) == 0:
            return metrics

        loss_sum = 0.0
        n_mb = 0
        ratio_hits = 0
        ratio_total = 0
        if not local:
            dummy = None
            for p in self.model.parameters():
                if p.requires_grad:
                    term = p.float().sum() * 0.0
                    dummy = term if dummy is None else dummy + term
            if dummy is not None:
                dummy.backward()
        else:
            for chunk in iter_length_microbatches(
                local,
                size=self.micro_batch_size,
                length_fn=lambda r: len(r.get("prompt_ids") or r.get("effective_prompt_ids") or [])
                + len(r.get("action_ids") or []),
            ):
                n_mb += 1
                pairs = []
                for row in chunk:
                    prompt = list(row.get("effective_prompt_ids") or row.get("prompt_ids") or [])
                    action = list(row.get("action_ids") or [])
                    if len(prompt) + len(action) > self.max_full_tokens:
                        raise ValueError(
                            f"sampled sequence {len(prompt)+len(action)} exceeds {self.max_full_tokens}"
                        )
                    pairs.append((prompt, action))
                logps, aligned = self._forward_response_logprobs(pairs, chunk)
                chunk_terms = []
                for row, new_lp in zip(aligned, logps):
                    if new_lp.numel() == 0:
                        continue
                    new_lp = new_lp.float()
                    old_lp = torch.tensor(row["token_logprobs"], device=self.device, dtype=torch.float32)
                    mask = torch.tensor(row["action_mask"], device=self.device, dtype=torch.float32)
                    if not torch.isfinite(new_lp).all() or not torch.isfinite(old_lp).all():
                        raise ValueError("non-finite logprobs in FSDP2 CISPO")
                    ratio = (new_lp - old_lp).exp()
                    if not torch.isfinite(ratio).all():
                        raise ValueError("non-finite CISPO ratio")
                    weight = ratio.clamp(min=self.clip_low, max=self.clip_high).detach()
                    ratio_hits += int((ratio != weight).sum().item())
                    ratio_total += int(mask.sum().item())
                    adv = float(row.get("advantage") or 0.0)
                    chunk_terms.append((-(weight * adv * new_lp * mask)).sum() / global_tokens)
                if chunk_terms:
                    loss = torch.stack(chunk_terms).sum()
                    if loss.requires_grad:
                        loss.backward()
                    loss_sum += float(loss.detach().item())
                    del loss
                del logps, chunk_terms
        if opd_rows and float(lambda_opd) > 0:
            from trim.training.tinker_opd_datum import TinkerOPDDatum

            opd_loss = self._opd_ce(list(opd_rows), lambda_already_baked=True)
            loss_sum += float(opd_loss)
            del TinkerOPDDatum
        params = [p for p in self.model.parameters() if p.requires_grad]
        clip_fn = getattr(self.model, "clip_grad_norm_", None)
        if callable(clip_fn):
            clip_fn(1.0)
        else:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
        self.optimizer.step()
        n_opt = 1
        self.optimizer.zero_grad(set_to_none=True)
        metrics.update(
            {
                "n_optimizer_steps": n_opt,
                "n_microbatches": n_mb,
                "loss": loss_sum,
                "ratio_clip_fraction": (ratio_hits / max(1, ratio_total)),
                "clip_low": self.clip_low,
                "clip_high": self.clip_high,
            }
        )
        log_train("fsdp2_cispo", **metrics)
        if n_opt not in {0, 1}:
            raise RuntimeError(f"n_optimizer_steps={n_opt}")
        return metrics

    def _opd_ce(self, datums: Sequence[Any], *, lambda_already_baked: bool) -> float:
        from trim.training.tinker_opd_datum import TinkerOPDDatum

        del lambda_already_baked
        prepared: list[tuple[list[int], list[int], list[float]]] = []
        for raw in datums:
            if isinstance(raw, TinkerOPDDatum):
                prompt = list(raw.prompt_token_ids)
                n_p = len(prompt)
                resp = list(raw.target_tokens[n_p:])
                weights = list(raw.weights[n_p:])
            else:
                prompt = list(raw.get("prompt_ids") or raw.get("effective_prompt_ids") or [])
                resp = list(raw.get("target_ids") or [])
                weights = list(raw.get("weights") or [1.0] * len(resp))
            if prompt and resp:
                prepared.append((prompt, resp, weights))
        if not prepared:
            return 0.0
        total = 0.0
        for chunk in iter_length_microbatches(
            prepared, size=self.micro_batch_size, length_fn=lambda r: len(r[0]) + len(r[1])
        ):
            logps, _ = self._forward_response_logprobs([(p, r) for p, r, _w in chunk], [{} for _ in chunk])
            terms = []
            for (_p, _r, weights), logp in zip(chunk, logps):
                w = torch.tensor(weights[: logp.numel()], device=self.device, dtype=torch.float32)
                terms.append(-(logp.float() * w).sum())
            if terms:
                loss = torch.stack(terms).sum()
                if loss.requires_grad:
                    loss.backward()
                total += float(loss.detach().item())
        return total

    def save_adapter(self, path: Path) -> None:
        path = Path(path)
        if self.rank == 0:
            path.mkdir(parents=True, exist_ok=True)
        if self.world_size > 1:
            dist.barrier()
        core = self.model
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            if isinstance(core, FSDP):
                from trim.training.hf_sft import _save_peft_checkpoint

                _save_peft_checkpoint(core, self.tokenizer, path, merge=False)
                if self.world_size > 1:
                    dist.barrier()
                return
        except Exception:
            pass
        state = None
        try:
            from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

            state = get_model_state_dict(
                self.model, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
            )
        except Exception:
            state = None
        while hasattr(core, "module"):
            core = core.module
        if self.rank == 0:
            saver = getattr(core, "save_pretrained", None)
            if saver is None:
                raise RuntimeError("cannot save LoRA adapter from wrapped model")
            if state is not None:
                lora_state = {
                    k: v for k, v in state.items()
                    if "lora_" in k or "modules_to_save" in k
                }
                saver(str(path), state_dict=lora_state or None)
            else:
                saver(str(path))
            self.tokenizer.save_pretrained(str(path))
        if self.world_size > 1:
            dist.barrier()
