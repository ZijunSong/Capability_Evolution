#!/usr/bin/env python3
"""Round13 Stage1 operation-only training with query-norm / hard / event-uniform variants."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from harness.capability.rollback_operation import RollbackOperation
from training.scope.rollback_operation_objectives import rollback_operation_loss
from training.scope_round11.stage1_views import build_stage1_view
from training.scope_round9.run_wave_b_train import merge_lora

DATA = _REPO / "artifacts/datasets/scope_round13/operation_sdi"
BASE_MODEL = "/data/ppnm/models/Qwen2.5-7B-Instruct"
OUT_DEFAULT = _REPO / "outputs/scope_round13/phase_b_stage1/training"

LossMode = Literal["query_norm", "event_uniform"]

VARIANTS: dict[str, dict] = {
    "r13_onpolicy_querynorm_seed42": {"seed": 42, "loss_mode": "query_norm", "hard_mult": 2.0},
    "r13_onpolicy_querynorm_seed43": {"seed": 43, "loss_mode": "query_norm", "hard_mult": 2.0},
    "r13_onpolicy_querynorm_seed44": {"seed": 44, "loss_mode": "query_norm", "hard_mult": 2.0},
    "r13_onpolicy_querynorm_nohard_seed42": {
        "seed": 42,
        "loss_mode": "query_norm",
        "hard_mult": 1.0,
    },
    "r13_onpolicy_eventuniform_seed42": {
        "seed": 42,
        "loss_mode": "event_uniform",
        "hard_mult": 1.0,
    },
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.open(encoding="utf-8"):
        if line.strip():
            rows.append(json.loads(line))
    return rows


@dataclass
class R13TrainConfig:
    model_path: str
    output_dir: Path
    seed: int = 42
    device: str = "cuda:0"
    learning_rate: float = 2e-5
    num_epochs: int = 3
    grad_accum: int = 16
    max_length: int = 1536
    lora_rank: int = 64
    lora_alpha: int = 128
    loss_mode: LossMode = "query_norm"
    hard_mult: float = 2.0
    disable_replan: bool = True


class R13Stage1Trainer:
    def __init__(self, cfg: R13TrainConfig) -> None:
        self.cfg = cfg
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path, torch_dtype=dtype, trust_remote_code=True
        )
        lora = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora)
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        # Required for gradient checkpointing under PEFT/LoRA.
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
        if hasattr(self.model, "config"):
            self.model.config.use_cache = False
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.train()
        self._step = 0

    def _text(self, sample: dict[str, Any]) -> str:
        return build_stage1_view(
            sample, self.tokenizer, "A0", max_length=self.cfg.max_length
        ).effective_input_text

    def train(self, train_rows: list[dict], valid_rows: list[dict] | None = None) -> dict:
        # Group by query for query-norm
        by_q: dict[str, list[dict]] = defaultdict(list)
        for r in train_rows:
            by_q[str(r.get("query_id"))].append(r)
        query_ids = list(by_q.keys())

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.learning_rate)
        if self.cfg.loss_mode == "query_norm":
            steps_per_epoch = len(query_ids)
        else:
            steps_per_epoch = len(train_rows)
        total_steps = max(1, steps_per_epoch * self.cfg.num_epochs // max(self.cfg.grad_accum, 1))
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=max(1, total_steps // 10), num_training_steps=total_steps
        )

        for _epoch in range(self.cfg.num_epochs):
            if self.cfg.loss_mode == "query_norm":
                random.shuffle(query_ids)
                units = query_ids
            else:
                units = list(range(len(train_rows)))
                random.shuffle(units)

            for unit in units:
                if self.cfg.loss_mode == "query_norm":
                    # Incremental per-event backward keeps peak activation memory O(1 event).
                    # Mathematically equivalent to mean(mult*loss)/mean_mult then backward.
                    events = by_q[unit]
                    n_ev = max(len(events), 1)
                    mean_mult = sum(
                        self.cfg.hard_mult if s.get("is_hard_event") else 1.0 for s in events
                    ) / n_ev
                    scale = 1.0 / (n_ev * max(mean_mult, 1e-6) * self.cfg.grad_accum)
                    for sample in events:
                        tgt = RollbackOperation(
                            str(
                                (sample.get("target_action") or {}).get("operation")
                                or sample.get("gold_operation")
                                or "CONTINUE"
                            )
                        )
                        s1 = self._text(sample)
                        loss = rollback_operation_loss(
                            self.model,
                            self.tokenizer,
                            s1,
                            tgt,
                            device=self.device,
                            prompt_is_final=True,
                            disable_replan=self.cfg.disable_replan,
                        )
                        mult = self.cfg.hard_mult if sample.get("is_hard_event") else 1.0
                        (mult * loss * scale).backward()
                else:
                    sample = train_rows[unit]
                    tgt = RollbackOperation(
                        str(
                            (sample.get("target_action") or {}).get("operation")
                            or sample.get("gold_operation")
                            or "CONTINUE"
                        )
                    )
                    s1 = self._text(sample)
                    loss = rollback_operation_loss(
                        self.model,
                        self.tokenizer,
                        s1,
                        tgt,
                        device=self.device,
                        prompt_is_final=True,
                        disable_replan=self.cfg.disable_replan,
                    )
                    (loss / self.cfg.grad_accum).backward()

                if (self._step + 1) % self.cfg.grad_accum == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                self._step += 1
                if self._step % 50 == 0:
                    print(
                        f"[train] variant_steps={self._step} epoch_units={len(units)}",
                        flush=True,
                    )

        self.model.save_pretrained(self.cfg.output_dir)
        self.tokenizer.save_pretrained(self.cfg.output_dir)
        return {
            "n_train": len(train_rows),
            "n_queries": len(query_ids),
            "n_valid": len(valid_rows or []),
            "loss_mode": self.cfg.loss_mode,
            "hard_mult": self.cfg.hard_mult,
            "steps": self._step,
        }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", required=True, choices=list(VARIANTS))
    p.add_argument("--gpu", default="cuda:0")
    p.add_argument("--out-root", type=Path, default=OUT_DEFAULT)
    p.add_argument("--force-retrain", action="store_true")
    args = p.parse_args()

    vk = VARIANTS[args.variant]
    out = args.out_root / args.variant
    out.mkdir(parents=True, exist_ok=True)
    if (
        not args.force_retrain
        and (out / "merged" / "config.json").exists()
        and (out / "train_only_report.json").exists()
    ):
        # Do not write DONE here — outer runner still needs VALID eval.
        print(f"[skip-train] {args.variant}")
        return

    train_rows = load_jsonl(DATA / "train.jsonl")
    valid_rows = load_jsonl(DATA / "valid.jsonl") if (DATA / "valid.jsonl").exists() else []
    t0 = time.time()
    cfg = R13TrainConfig(
        model_path=BASE_MODEL,
        output_dir=out / "lora",
        seed=int(vk["seed"]),
        device=args.gpu,
        loss_mode=vk["loss_mode"],
        hard_mult=float(vk["hard_mult"]),
    )
    report = R13Stage1Trainer(cfg).train(train_rows, valid_rows)
    # Temporarily point merge helper at this out dir layout
    merged = merge_lora(out)
    full = {
        "variant": args.variant,
        "train_path": str(DATA / "train.jsonl"),
        "stage1_view": "A0",
        "operation_only": True,
        "train_report": report,
        "merged_path": str(merged),
        "wall_clock_s": time.time() - t0,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True
        ).strip(),
        "seed": vk["seed"],
        "loss_mode": vk["loss_mode"],
        "hard_mult": vk["hard_mult"],
    }
    (out / "train_only_report.json").write_text(json.dumps(full, indent=2) + "\n")
    print(json.dumps({k: full[k] for k in full if k != "train_report"}, indent=2))


if __name__ == "__main__":
    main()
