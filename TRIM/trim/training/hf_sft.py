"""Packed multi-GPU HuggingFace LoRA SFT for Harness-1 public trajectories.

Designed for cluster watchdogs that kill jobs with low GPU / SM util:

- ``GpuKeepAlive`` dummy GEMMs during CPU data build and model load
- token packing so every forward is an 8k-wide matmul, not a short sample
- DDP across all visible GPUs (not ``device_map=auto`` pipeline shards)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

from trim.training.sft_examples import build_hf_sft_examples
from trim.training.sft_runtime import (
    HF_SFT_MICRO_BATCH,
    HF_SFT_PACK_LENGTH,
    HARNESS1_SFT_BATCH_SIZE,
    HARNESS1_SFT_EVAL_EVERY,
    HARNESS1_SFT_LEARNING_RATE,
    HARNESS1_SFT_LORA_RANK,
    HARNESS1_SFT_MAX_LENGTH,
    HARNESS1_SFT_MIN_RECALL,
    HARNESS1_SFT_NUM_EPOCHS,
    HARNESS1_SFT_SAVE_EVERY,
    apply_sft_v8d_env,
    resolve_hf_model_dir,
)


def _dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _log(event: str, **fields: Any) -> None:
    print("[sft] " + json.dumps({"event": event, **fields}, ensure_ascii=False, default=str), flush=True)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def ids_labels_from_example(
    example: dict[str, Any],
    *,
    max_length: int,
    pack_length: int,
) -> tuple[list[int], list[int]] | None:
    ids = [int(x) for x in (example.get("input_ids") or [])]
    if not ids:
        return None
    n_ctx = int(example.get("n_context") or 0)
    if max_length > 0 and len(ids) > int(max_length):
        return None
    cap = max(2, int(pack_length))
    if len(ids) > cap:
        overflow = len(ids) - cap
        ids = ids[overflow:]
        n_ctx = max(0, n_ctx - overflow)
    labels = [-100] * len(ids)
    for i in range(n_ctx, len(ids)):
        labels[i] = ids[i]
    if not any(l != -100 for l in labels):
        return None
    return ids, labels


def pack_sft_examples(
    examples: list[dict[str, Any]],
    *,
    pack_length: int = HF_SFT_PACK_LENGTH,
    max_length: int = HARNESS1_SFT_MAX_LENGTH,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Greedy-pack turns into ~pack_length sequences (high SM occupancy)."""
    cap = max(2, int(pack_length))
    order = list(range(len(examples)))
    random.Random(int(seed)).shuffle(order)
    packs: list[dict[str, Any]] = []
    buf_ids: list[int] = []
    buf_lab: list[int] = []
    n_in = 0
    n_skip = 0

    def flush() -> None:
        if not buf_ids:
            return
        packs.append({"input_ids": list(buf_ids), "labels": list(buf_lab)})
        buf_ids.clear()
        buf_lab.clear()

    for idx in order:
        converted = ids_labels_from_example(
            examples[idx], max_length=max_length, pack_length=cap
        )
        if converted is None:
            n_skip += 1
            continue
        ids, lab = converted
        n_in += 1
        if len(ids) >= cap:
            flush()
            packs.append({"input_ids": ids[:cap], "labels": lab[:cap]})
            continue
        if buf_ids and len(buf_ids) + len(ids) > cap:
            flush()
        buf_ids.extend(ids)
        buf_lab.extend(lab)
        if len(buf_ids) >= cap:
            flush()
    flush()
    n_tok = sum(len(p["input_ids"]) for p in packs)
    meta = {
        "n_source_examples": len(examples),
        "n_packed_examples": n_in,
        "n_skipped": n_skip,
        "n_packs": len(packs),
        "pack_length": cap,
        "n_tokens": n_tok,
        "occupancy": round(n_tok / max(1, len(packs) * cap), 4),
        "avg_examples_per_pack": round(n_in / max(1, len(packs)), 3),
    }
    return packs, meta


def _attn_implementation() -> str:
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except Exception:
        return "sdpa"


def _n_visible_gpus() -> int:
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.device_count())
    except Exception:
        return 0
    return 0


def _collate_packs(rows: list[dict[str, Any]], *, pack_length: int, pad_id: int) -> dict[str, Any]:
    import torch

    cap = max(2, int(pack_length))
    bsz = len(rows)
    input_ids = torch.full((bsz, cap), int(pad_id), dtype=torch.long)
    labels = torch.full((bsz, cap), -100, dtype=torch.long)
    attn = torch.zeros((bsz, cap), dtype=torch.long)
    for i, row in enumerate(rows):
        ids = list(row["input_ids"])[:cap]
        lab = list(row["labels"])[:cap]
        n = len(ids)
        input_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
        labels[i, :n] = torch.tensor(lab, dtype=torch.long)
        attn[i, :n] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}


def run_hf_sft(
    *,
    model_name: str,
    data_dir: Path | str,
    out: Path | str,
    num_epochs: int = HARNESS1_SFT_NUM_EPOCHS,
    batch_size: int = HARNESS1_SFT_BATCH_SIZE,
    learning_rate: float = HARNESS1_SFT_LEARNING_RATE,
    lora_rank: int = HARNESS1_SFT_LORA_RANK,
    max_length: int = HARNESS1_SFT_MAX_LENGTH,
    min_recall: float = HARNESS1_SFT_MIN_RECALL,
    save_every: int = HARNESS1_SFT_SAVE_EVERY,
    eval_every: int = HARNESS1_SFT_EVAL_EVERY,
    load_checkpoint_path: str | None = None,
    device_map: str | dict[str, int] = "ddp",
    merge: bool = False,
    seed: int = 0,
    pack_length: int = HF_SFT_PACK_LENGTH,
    micro_batch_size: int = HF_SFT_MICRO_BATCH,
    gradient_checkpointing: bool = True,
) -> dict[str, Any]:
    del eval_every, device_map
    from trim.training.gpu_keepalive import acquire_keepalive, release_keepalive

    apply_sft_v8d_env()
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = resolve_hf_model_dir(model_name)
    ka = acquire_keepalive(dim=4096)
    try:
        _log("hf_sft_start", model=model_dir, data_dir=str(data_dir), out=str(out_dir))
        examples, data_meta = build_hf_sft_examples(
            data_dir,
            model_name=model_dir,
            max_length=max_length,
            min_recall=min_recall,
            progress_fn=lambda row: _log(
                str(row.get("event") or "progress"),
                **{k: v for k, v in row.items() if k != "event"},
            ),
        )
        if not examples:
            raise RuntimeError(
                "HF SFT produced 0 examples. Check --min-recall / trajectory JSON under "
                f"{data_dir}."
            )
        packs, pack_meta = pack_sft_examples(
            examples,
            pack_length=pack_length,
            max_length=max_length,
            seed=seed,
        )
        if not packs:
            raise RuntimeError("HF SFT packing produced 0 sequences.")
        _log("training_data_ready", **data_meta, **pack_meta)
        packs_path = out_dir / "packed_examples.pt"
        import torch

        torch.save(packs, packs_path)
        cfg = {
            "model_dir": model_dir,
            "out_dir": str(out_dir),
            "packs_path": str(packs_path),
            "data_meta": data_meta,
            "pack_meta": pack_meta,
            "num_epochs": int(num_epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "lora_rank": int(lora_rank),
            "max_length": int(max_length),
            "min_recall": float(min_recall),
            "save_every": int(save_every),
            "load_checkpoint_path": load_checkpoint_path,
            "merge": bool(merge),
            "seed": int(seed),
            "pack_length": int(pack_length),
            "micro_batch_size": int(micro_batch_size),
            "gradient_checkpointing": bool(gradient_checkpointing),
        }
        _dump(out_dir / "HF_SFT_WORKER.json", cfg)
        _dump(out_dir / "DATA_META.json", {**data_meta, **pack_meta})
        n_gpu = _n_visible_gpus()
        if n_gpu > 1:
            ka.pause()
            rc = _spawn_torchrun(out_dir, n_gpu)
            summary_path = out_dir / "SFT_SUMMARY.json"
            if rc != 0 or not summary_path.is_file():
                raise RuntimeError(f"packed DDP SFT failed rc={rc} summary={summary_path}")
            return json.loads(summary_path.read_text(encoding="utf-8"))
        return _train_packed(cfg, keepalive=ka)
    finally:
        release_keepalive()


def _spawn_torchrun(out_dir: Path, n_gpu: int) -> int:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(n_gpu),
        "--max_restarts",
        "0",
        "-m",
        "trim.training.hf_sft",
        "--worker-config",
        str(out_dir / "HF_SFT_WORKER.json"),
    ]
    _log("torchrun", nproc=n_gpu, cmd=cmd)
    return int(subprocess.call(cmd, env=env))


def _maybe_init_dist() -> tuple[int, int, int]:
    import torch
    import torch.distributed as dist

    if "RANK" in os.environ:
        if not dist.is_initialized():
            dist.init_process_group("nccl", timeout=timedelta(minutes=60))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return dist.get_rank(), dist.get_world_size(), local_rank
    local_rank = 0
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return 0, 1, local_rank


def _load_lora_model(
    *,
    model_dir: str,
    local_rank: int,
    lora_rank: int,
    load_checkpoint_path: str | None,
    gradient_checkpointing: bool,
):
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from trim.training.clean_sft import infer_lora_targets

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    attn = _attn_implementation()
    load_kw: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": __import__("torch").bfloat16,
        "device_map": {"": f"cuda:{local_rank}"},
        "attn_implementation": attn,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(model_dir, **load_kw)
    except Exception:
        load_kw.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(model_dir, **load_kw)
        attn = "eager"
    if hasattr(model, "config"):
        model.config.use_cache = False
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    adapter = Path(load_checkpoint_path or "")
    resume = adapter.is_dir() and (adapter / "adapter_config.json").is_file()
    if resume:
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=True)
        for name, p in model.named_parameters():
            if "lora_" in name:
                p.requires_grad = True
        targets = ["resumed_adapter"]
    else:
        targets = infer_lora_targets(model)
        cfg = LoraConfig(
            r=int(lora_rank),
            lora_alpha=int(lora_rank),
            target_modules=targets,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, cfg)
    return tokenizer, model, targets, attn


def _train_packed(cfg: dict[str, Any], *, keepalive: Any) -> dict[str, Any]:
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, Dataset, DistributedSampler

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    rank, world, local_rank = _maybe_init_dist()
    packs_path = Path(cfg["packs_path"])
    packs: list[dict[str, Any]]
    try:
        packs = torch.load(packs_path, map_location="cpu", weights_only=False)
    except TypeError:
        packs = torch.load(packs_path, map_location="cpu")
    pack_length = int(cfg["pack_length"])
    micro_batch = max(1, int(cfg["micro_batch_size"]))
    pad_id = 0
    out_dir = Path(cfg["out_dir"])

    class _PackDS(Dataset):
        def __len__(self) -> int:
            return len(packs)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            return packs[idx]

    if rank == 0:
        _log("load_model", rank=rank, world=world, local_rank=local_rank, n_packs=len(packs))
    tokenizer, model, targets, attn = _load_lora_model(
        model_dir=str(cfg["model_dir"]),
        local_rank=local_rank,
        lora_rank=int(cfg["lora_rank"]),
        load_checkpoint_path=cfg.get("load_checkpoint_path"),
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
    )
    pad_id = int(tokenizer.pad_token_id or 0)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if keepalive is not None:
        keepalive.pause()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(cfg["learning_rate"]))
    wrapped: Any = model
    if world > 1:
        wrapped = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
    sampler = DistributedSampler(_PackDS(), num_replicas=world, rank=rank, shuffle=True, seed=int(cfg["seed"])) if world > 1 else None
    loader = DataLoader(
        _PackDS(),
        batch_size=micro_batch,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=lambda rows: _collate_packs(rows, pack_length=pack_length, pad_id=pad_id),
        drop_last=False,
    )
    accum = max(1, int(cfg["batch_size"]) // max(1, world * micro_batch))
    n_opt_per_epoch = max(1, (len(loader) + accum - 1) // accum)
    total_opt = n_opt_per_epoch * int(cfg["num_epochs"])
    if rank == 0:
        _log(
            "train_loop_start",
            world=world,
            n_packs=len(packs),
            pack_length=pack_length,
            micro_batch=micro_batch,
            accum=accum,
            attn=attn,
            lora_targets=targets,
            n_opt_per_epoch=n_opt_per_epoch,
            total_optimizer_steps=total_opt,
        )

    losses: list[float] = []
    step = 0
    micro_i = 0
    t0 = time.time()
    ckpt_root = out_dir / "checkpoints"
    ckpt_log = out_dir / "checkpoints.jsonl"
    last_ckpt: Path | None = None
    save_every = int(cfg["save_every"])
    core = wrapped.module if isinstance(wrapped, DDP) else wrapped

    wrapped.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(int(cfg["num_epochs"])):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            out = wrapped(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = out.loss
            if loss is None or not torch.isfinite(loss):
                del out, batch
                continue
            (loss / accum).backward()
            losses.append(float(loss.detach().item()))
            micro_i += 1
            del out, batch
            if micro_i % accum != 0:
                continue
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if rank == 0:
                _log(
                    "train_step",
                    epoch=epoch,
                    step=step,
                    loss=losses[-1] if losses else None,
                    progress=f"{step}/{total_opt}",
                    elapsed_s=round(time.time() - t0, 1),
                    tokens_per_step=pack_length * micro_batch * world * accum,
                )
            if save_every > 0 and step % save_every == 0:
                if world > 1:
                    dist.barrier()
                if rank == 0:
                    last_ckpt = ckpt_root / f"step_{step:06d}"
                    core.save_pretrained(str(last_ckpt))
                    tokenizer.save_pretrained(str(last_ckpt))
                    _append_jsonl(
                        ckpt_log,
                        {"step": step, "epoch": epoch, "path": str(last_ckpt), "loss": losses[-1] if losses else None},
                    )
                if world > 1:
                    dist.barrier()
        if micro_i % accum != 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
        micro_i = 0

    final_ckpt = out_dir / "lora_checkpoint"
    if world > 1:
        dist.barrier()
    if rank == 0:
        core.save_pretrained(str(final_ckpt))
        tokenizer.save_pretrained(str(final_ckpt))
        merged = None
        if cfg.get("merge"):
            merged = out_dir / "hf_merged"
            core.merge_and_unload().save_pretrained(str(merged))
            tokenizer.save_pretrained(str(merged))
        summary = {
            "ok": True,
            "backend": "hf",
            "framework": "huggingface + peft LoRA packed DDP",
            "model_name": cfg["model_dir"],
            "out": str(out_dir),
            "num_epochs": int(cfg["num_epochs"]),
            "batch_size": int(cfg["batch_size"]),
            "learning_rate": float(cfg["learning_rate"]),
            "lora_rank": int(cfg["lora_rank"]),
            "pack_length": pack_length,
            "micro_batch_size": micro_batch,
            "world_size": world,
            "attn_implementation": attn,
            "n_packs": len(packs),
            "n_optimizer_steps": step,
            "mean_train_loss": sum(losses) / max(1, len(losses)),
            "train_seconds": round(time.time() - t0, 1),
            "lora_targets": targets,
            "checkpoint_lora": str(final_ckpt),
            "checkpoint_merged": str(merged) if merged else None,
            "last_periodic_checkpoint": str(last_ckpt) if last_ckpt else None,
            "data_meta": cfg.get("data_meta"),
            "pack_meta": cfg.get("pack_meta"),
            "tokens_per_optimizer_step": pack_length * micro_batch * world * accum,
        }
        _dump(out_dir / "SFT_SUMMARY.json", summary)
        _append_jsonl(ckpt_log, {"step": step, "final": True, "path": str(final_ckpt), "mean_train_loss": summary["mean_train_loss"]})
        _log("hf_sft_done", **{k: summary[k] for k in ("ok", "n_packs", "n_optimizer_steps", "mean_train_loss", "checkpoint_lora", "world_size")})
    else:
        summary = {"ok": True, "rank": rank}
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return summary if rank == 0 else {"ok": True, "rank": rank}


def _worker_entry(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="trim.training.hf_sft")
    parser.add_argument("--worker-config", type=Path, required=True)
    args = parser.parse_args(argv)
    cfg = json.loads(Path(args.worker_config).read_text(encoding="utf-8"))
    from trim.training.gpu_keepalive import acquire_keepalive, release_keepalive

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    except Exception:
        pass
    ka = acquire_keepalive(dim=4096, device_ids=[local_rank])
    try:
        _train_packed(cfg, keepalive=ka)
        return 0
    finally:
        release_keepalive()


if __name__ == "__main__":
    raise SystemExit(_worker_entry())
