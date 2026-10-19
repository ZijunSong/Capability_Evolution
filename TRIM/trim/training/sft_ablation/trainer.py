"""LoRA CE trainer. No RL, gap, or gate route."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from trim.training.sft_ablation.ce import ce_update
from trim.training.sft_ablation.contracts import SFTBatchRow


class CEUpdateResult:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.updated = bool(payload["updated"])
        self.loss_ce = payload.get("loss_ce")
        self.grad_norm = payload.get("grad_norm")
        self.n_rows = int(payload["n_rows"])
        self.weighted_tokens = float(payload["weighted_tokens"])
        self.payload = payload


class SFTTrainer:
    def __init__(self, model: Any, optimizer: Any, scheduler: Any, *, pad_id: int, max_length: int, micro_batch_size: int, grad_clip: float) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.pad_id = int(pad_id)
        self.max_length = int(max_length)
        self.micro_batch_size = int(micro_batch_size)
        self.grad_clip = float(grad_clip)

    def update(self, rows: Sequence[SFTBatchRow]) -> CEUpdateResult:
        payload = ce_update(
            self.model,
            self.optimizer,
            self.scheduler,
            rows,
            pad_id=self.pad_id,
            max_length=self.max_length,
            micro_batch_size=self.micro_batch_size,
            grad_clip=self.grad_clip,
        )
        return CEUpdateResult(payload)

    def adapter_digest(self) -> str:
        from trim.training.sft_ablation.io_util import sha256_text

        rows = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                rows.append(f"{name}:{param.detach().float().cpu().numpy().tobytes().hex()}")
        return sha256_text("\n".join(rows))


def build_hf_trainer(args: Any, *, device: str | None = None) -> SFTTrainer:
    """Load the student base and a fresh or initial LoRA. Resume is separate."""
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from trim.training.clean_sft import infer_lora_targets

    dtype = torch.bfloat16 if str(args.dtype) == "bfloat16" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    if hasattr(model, "config"):
        model.config.use_cache = False
    init = Path(str(args.init_adapter or ""))
    if init.is_dir() and (init / "adapter_config.json").is_file():
        model = PeftModel.from_pretrained(model, str(init), is_trainable=True)
    else:
        targets = infer_lora_targets(model) if args.lora_target == "auto" else [part.strip() for part in str(args.lora_target).split(",") if part.strip()]
        model = get_peft_model(
            model,
            LoraConfig(
                r=int(args.lora_rank),
                lora_alpha=int(args.lora_alpha),
                lora_dropout=float(args.lora_dropout),
                target_modules=targets,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            ),
        )
    if device:
        model.to(device)
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    return SFTTrainer(
        model,
        optimizer,
        scheduler,
        pad_id=int(tokenizer.pad_token_id),
        max_length=int(args.max_model_len),
        micro_batch_size=int(args.micro_batch_size),
        grad_clip=float(args.grad_clip),
    )
