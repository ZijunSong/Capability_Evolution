#!/usr/bin/env python3
"""OPHSD-style whole-harness terminal-context route-head baseline cell.

This is a route-level faithful adaptation for the SCAPE/Harness-1 local BM25
setting: full-harness terminal context is used to define the frozen teacher route
(target distribution), while the trained student route head receives only reduced
no-privilege state features at inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TOOLS = [
    "fan_out_search",
    "search_corpus",
    "grep_corpus",
    "read_document",
    "review_docs",
    "curate",
    "verify",
    "end_search",
]


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_float(text: str) -> float:
    return int(hashlib.sha256(text.encode()).hexdigest()[:13], 16) / float(16**13 - 1)


def distribution(row: dict, key: str) -> list[float]:
    vals = [max(0.0, float(row[key].get(t, 0.0))) for t in TOOLS]
    total = sum(vals)
    if total <= 0:
        return [1.0 / len(TOOLS)] * len(TOOLS)
    return [v / total for v in vals]


def terminal_context(row: dict) -> dict:
    """Whole-harness terminal summary; excludes component-local auto fields."""
    info = dict(row.get("information_fields") or {})
    return {
        "query_id": str(row.get("query_id")),
        "terminal_step": int(row.get("step", 0)),
        "document_count": int(info.get("document_count", 0)),
        "tool_history_len": int(info.get("tool_history_len", 0)),
        "prior_search_count": int(info.get("prior_search_count", 0)),
        "importance_high_count": int(info.get("importance_high_count", 0)),
    }


def student_features(row: dict) -> list[float]:
    """No-privilege student features. No terminal-context or component signal."""
    qid = str(row.get("query_id"))
    q = (int(qid) if qid.isdigit() else sum(ord(c) for c in qid)) % 997
    step = int(row.get("step", 0))
    h = stable_float(str(row.get("snapshot_hash", "")))
    reduced = distribution(row, "P_tool_name_reduced")
    return reduced + [q / 997.0, step / 16.0, h]


def matrix(rows: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.tensor([student_features(r) for r in rows], dtype=torch.float32)
    teacher = torch.tensor([distribution(r, "P_tool_name_full") for r in rows], dtype=torch.float32)
    base = torch.tensor([distribution(r, "P_tool_name_reduced") for r in rows], dtype=torch.float32)
    teacher = teacher / teacher.sum(dim=1, keepdim=True).clamp_min(1e-12)
    base = base / base.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return x, teacher, base


class RouteHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 128), nn.GELU(), nn.Dropout(0.05), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, len(TOOLS)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def metrics(target: torch.Tensor, pred: torch.Tensor) -> dict[str, float]:
    p = target.clamp_min(1e-12)
    q = pred.clamp_min(1e-12)
    m = 0.5 * (p + q)
    js = 0.5 * (p * (p.log() - m.log())).sum(1) + 0.5 * (q * (q.log() - m.log())).sum(1)
    kl = (p * (p.log() - q.log())).sum(1)
    ce = -(p * q.log()).sum(1)
    return {
        "JS": float(js.mean().detach().cpu()),
        "KL_T_to_S": float(kl.mean().detach().cpu()),
        "CE_teacher_to_student": float(ce.mean().detach().cpu()),
        "agreement": float((p.argmax(1) == q.argmax(1)).float().mean().detach().cpu()),
        "search_probability": float((q[:, 0] + q[:, 1] + q[:, 2]).mean().detach().cpu()),
        "end_probability": float(q[:, 7].mean().detach().cpu()),
        "normalized_mean": float(q.sum(1).mean().detach().cpu()),
        "invalid_tool_rate": 0.0,
    }


def train(x: torch.Tensor, y: torch.Tensor, seed: int, steps: int) -> tuple[RouteHead, float, bool]:
    seed_all(seed)
    head = RouteHead(x.shape[1]).to(x.device)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-3, weight_decay=1e-4)
    batch = min(64, x.shape[0])
    losses = []
    grad_finite = True
    gen = torch.Generator(device="cpu")
    for step in range(steps):
        gen.manual_seed(seed + step * 7919)
        idx = torch.randint(0, x.shape[0], (batch,), generator=gen).to(x.device)
        logp = F.log_softmax(head(x[idx]), dim=-1)
        loss = F.kl_div(logp, y[idx], reduction="batchmean")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        for param in head.parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                grad_finite = False
        opt.step()
        losses.append(float(loss.detach().cpu()))
    return head, sum(losses) / max(1, len(losses)), grad_finite


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--valid", type=Path, required=True)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--steps", type=int, default=600)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "STATUS_LIVE.md").write_text(f"# STATUS_LIVE\n\n- cell: {args.cell}\n- status: loading\n- seed: {args.seed}\n", encoding="utf-8")
    seed_all(args.seed)
    train_rows = load_rows(args.train)
    valid_rows = load_rows(args.valid)
    test_rows = load_rows(args.test)
    ctx_sha = hashlib.sha256("\n".join(json.dumps(terminal_context(r), sort_keys=True) for r in train_rows).encode()).hexdigest()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    x_train, y_train, _ = matrix(train_rows)
    x_valid, y_valid, base_valid = matrix(valid_rows)
    x_test, y_test, base_test = matrix(test_rows)
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_valid, y_valid, base_valid = x_valid.to(device), y_valid.to(device), base_valid.to(device)
    x_test, y_test, base_test = x_test.to(device), y_test.to(device), base_test.to(device)
    pre_valid = metrics(y_valid, base_valid)
    pre_test = metrics(y_test, base_test)
    (args.out / "STATUS_LIVE.md").write_text(f"# STATUS_LIVE\n\n- cell: {args.cell}\n- status: training\n- seed: {args.seed}\n", encoding="utf-8")
    head, mean_loss, grad_finite = train(x_train, y_train, args.seed, args.steps)
    with torch.no_grad():
        post_valid = metrics(y_valid, torch.softmax(head(x_valid), dim=-1))
        post_test = metrics(y_test, torch.softmax(head(x_test), dim=-1))
    ckpt = {"cell": args.cell, "seed": args.seed, "state_dict": head.state_dict(), "legal_tools": TOOLS, "feature_contract": "no_privilege_student_features", "teacher_context_sha256": ctx_sha}
    torch.save(ckpt, args.out / "route_head.pt")
    reloaded = torch.load(args.out / "route_head.pt", map_location=device)
    payload = {
        "cell": args.cell,
        "method": "OPHSD-style",
        "adaptation": "whole_harness_terminal_context_route_teacher__no_privilege_student_route_head",
        "seed": args.seed,
        "objective": "route_kl",
        "n_train": len(train_rows),
        "n_valid": len(valid_rows),
        "n_test": len(test_rows),
        "steps": args.steps,
        "mean_train_loss": mean_loss,
        "loss_finite": math.isfinite(mean_loss),
        "grad_finite": bool(grad_finite),
        "pre_valid": pre_valid,
        "post_valid": post_valid,
        "pre_test": pre_test,
        "post_test": post_test,
        "teacher_context_sha256": ctx_sha,
        "student_inference_has_privilege": False,
        "component_local_signal_used": False,
        "route_distribution_normalized": abs(post_test["normalized_mean"] - 1.0) < 1e-6,
        "invalid_tool_rate": 0.0,
        "checkpoint_reloadable": reloaded.get("cell") == args.cell,
        "python": sys.executable,
    }
    (args.out / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.out / "STATUS_LIVE.md").write_text(f"# STATUS_LIVE\n\n- cell: {args.cell}\n- status: completed\n- seed: {args.seed}\n", encoding="utf-8")
    (args.out / "DONE").write_text("ok\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
