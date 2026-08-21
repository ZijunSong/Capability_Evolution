#!/usr/bin/env python3
"""Matched structured-vs-textual 8-way route distillation cell."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LEGAL = [
    "fan_out_search",
    "search_corpus",
    "grep_corpus",
    "read_document",
    "review_docs",
    "curate",
    "verify",
    "end_search",
]


def load_rows(path: str) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_feature(row: dict, privilege: str) -> list[float]:
    """No-privilege Student features plus representation-specific deterministic tokens.

    The Student runtime part uses only reduced-view/source route state. The privilege
    condition is used solely to train against the matched teacher route target.
    """
    reduced = [float(row["P_tool_name_reduced"][tool]) for tool in LEGAL]
    q = (int(row["query_id"]) if str(row["query_id"]).isdigit() else sum(ord(c) for c in row["query_id"])) % 997
    step = int(row["step"])
    h = int(row["snapshot_hash"][:12], 16) / float(16**12 - 1)
    info = 1.0 if row["information_fields"]["verify_available"] else 0.0
    if privilege == "structured":
        rep = [info, 0.0, len(json.dumps(row["information_fields"], sort_keys=True)) / 100.0]
    else:
        rep = [0.0, info, len(row["prompt_textual"]) / 1000.0]
    return reduced + [q / 997.0, step / 32.0, h] + rep


def matrix(rows: list[dict], privilege: str, target_key: str) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor([stable_feature(row, privilege) for row in rows], dtype=torch.float32)
    y = torch.tensor([[float(row[target_key][tool]) for tool in LEGAL] for row in rows], dtype=torch.float32)
    y = y / y.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return x, y


class RouteHead(nn.Module):
    def __init__(self, dim: int, n: int = 8):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 64), nn.GELU(), nn.Linear(64, n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def metric_dict(target: torch.Tensor, pred: torch.Tensor) -> dict[str, float]:
    p = target.clamp_min(1e-12)
    q = pred.clamp_min(1e-12)
    m = 0.5 * (p + q)
    js = 0.5 * (p * (p.log() - m.log())).sum(1) + 0.5 * (q * (q.log() - m.log())).sum(1)
    kl = (p * (p.log() - q.log())).sum(1)
    ce = -(p * q.log()).sum(1)
    arg = q.argmax(1)
    return {
        "JS": float(js.mean().detach().cpu()),
        "KL_T_to_S": float(kl.mean().detach().cpu()),
        "CE_teacher_to_student": float(ce.mean().detach().cpu()),
        "agreement": float((p.argmax(1) == arg).float().mean().detach().cpu()),
        "entropy": float((-(q * q.log()).sum(1)).mean().detach().cpu()),
        "verify_probability": float(q[:, 6].mean().detach().cpu()),
        "end_probability": float(q[:, 7].mean().detach().cpu()),
        "search_probability": float((q[:, 0] + q[:, 1] + q[:, 2]).mean().detach().cpu()),
        "invalid_tool_rate": 0.0,
        "normalized_mean": float(q.sum(1).mean().detach().cpu()),
    }


def train_head(x: torch.Tensor, target: torch.Tensor, seed: int, objective: str, steps: int) -> tuple[RouteHead, float, bool]:
    seed_all(seed)
    device = x.device
    head = RouteHead(x.shape[1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-3, weight_decay=1e-4)
    batch = min(64, x.shape[0])
    losses = []
    grad_finite = True
    for step in range(steps):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed + step * 7919)
        idx = torch.randint(0, x.shape[0], (batch,), generator=gen).to(device)
        logp = F.log_softmax(head(x[idx]), dim=-1)
        if objective == "route_kl":
            loss = F.kl_div(logp, target[idx], reduction="batchmean")
        else:
            loss = F.nll_loss(logp, target[idx].argmax(1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        for param in head.parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                grad_finite = False
        opt.step()
        losses.append(float(loss.detach().cpu()))
    return head, float(sum(losses) / len(losses)), grad_finite


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", required=True)
    parser.add_argument("--privilege", choices=["structured", "textual"], required=True)
    parser.add_argument("--objective", choices=["route_kl", "action_ce"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--valid", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--steps", type=int, default=240)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    status = out / "STATUS_LIVE.md"

    def write_status(value: str) -> None:
        status.write_text(
            f"# STATUS_LIVE\n\n- cell: {args.cell}\n- status: {value}\n- gpu: {args.gpu}\n- privilege: {args.privilege}\n- objective: {args.objective}\n",
            encoding="utf-8",
        )

    write_status("loading_data")
    seed_all(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    train = load_rows(args.train)
    valid = load_rows(args.valid)
    test = load_rows(args.test)
    x_train, teacher_train = matrix(train, args.privilege, "P_tool_name_full")
    x_valid, teacher_valid = matrix(valid, args.privilege, "P_tool_name_full")
    x_test, teacher_test = matrix(test, args.privilege, "P_tool_name_full")
    _, base_valid = matrix(valid, args.privilege, "P_tool_name_reduced")
    _, base_test = matrix(test, args.privilege, "P_tool_name_reduced")
    x_train = x_train.to(device)
    teacher_train = teacher_train.to(device)
    x_valid = x_valid.to(device)
    teacher_valid = teacher_valid.to(device)
    x_test = x_test.to(device)
    teacher_test = teacher_test.to(device)
    base_valid = base_valid.to(device)
    base_test = base_test.to(device)

    write_status("training_route_head")
    pre_valid = metric_dict(teacher_valid, base_valid)
    pre_test = metric_dict(teacher_test, base_test)
    head, mean_loss, grad_finite = train_head(x_train, teacher_train, args.seed, args.objective, args.steps)
    with torch.no_grad():
        post_valid_probs = torch.softmax(head(x_valid), dim=-1)
        post_test_probs = torch.softmax(head(x_test), dim=-1)
    post_valid = metric_dict(teacher_valid, post_valid_probs)
    post_test = metric_dict(teacher_test, post_test_probs)
    common_valid = metric_dict(teacher_valid, post_valid_probs)
    common_test = metric_dict(teacher_test, post_test_probs)

    ckpt = {
        "cell": args.cell,
        "seed": args.seed,
        "privilege": args.privilege,
        "objective": args.objective,
        "head": head.state_dict(),
        "legal_tools": LEGAL,
    }
    torch.save(ckpt, out / "route_head.pt")
    reloaded = torch.load(out / "route_head.pt", map_location=device)
    checkpoint_reloadable = reloaded.get("cell") == args.cell
    payload = {
        "cell": args.cell,
        "seed": args.seed,
        "privilege": args.privilege,
        "objective": args.objective,
        "gpu": args.gpu,
        "n_train": len(train),
        "n_valid": len(valid),
        "n_test": len(test),
        "steps": args.steps,
        "mean_train_loss": mean_loss,
        "loss_finite": math.isfinite(mean_loss),
        "grad_finite": bool(grad_finite),
        "pre_valid": pre_valid,
        "post_valid": post_valid,
        "pre_test": pre_test,
        "post_test": post_test,
        "common_reference_valid": common_valid,
        "common_reference_test": common_test,
        "route_distribution_normalized": abs(post_test["normalized_mean"] - 1.0) < 1e-6,
        "invalid_tool_rate": 0.0,
        "checkpoint_reloadable": checkpoint_reloadable,
        "python": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": str(device),
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_status("completed")
    (out / "DONE").write_text("ok\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
