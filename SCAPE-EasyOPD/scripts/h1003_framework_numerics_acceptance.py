#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "scape_easyopd" / "framework" / "FRAMEWORK_NUMERICS_ACCEPTANCE.json"

from easyopd.methods.scape_component_opd.losses.reverse_kl import reverse_kl_exact
from easyopd.methods.scape_component_opd.tool_span import require_parsable_tool_calls
from easyopd.methods.scape_component_opd.harness1_bridge import Qwen3NativeChatAdapter


def manual_reverse(student_logits, teacher_logits):
    sl = F.log_softmax(student_logits.float(), dim=-1)
    tl = F.log_softmax(teacher_logits.float(), dim=-1)
    return (sl.exp() * (sl - tl)).sum(dim=-1)


def main() -> int:
    checks = {}
    s = torch.tensor([[[0.2, -0.1, 0.4], [1.0, -0.2, 0.0]]], dtype=torch.float64, requires_grad=True)
    t = torch.tensor([[[0.0, 0.3, -0.2], [-0.5, 0.4, 0.1]]], dtype=torch.float64)
    mask = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    loss, metrics = reverse_kl_exact(s, t, mask)
    expected = manual_reverse(s, t)[0, 0]
    checks["exact_reverse_kl_bruteforce"] = bool(torch.allclose(loss, expected, atol=1e-8))
    loss.backward()
    eps = 1e-4
    idx = (0, 0, 1)
    sp = s.detach().clone(); sm = s.detach().clone()
    sp[idx] += eps; sm[idx] -= eps
    lp, _ = reverse_kl_exact(sp, t, mask)
    lm, _ = reverse_kl_exact(sm, t, mask)
    fd = (lp - lm) / (2 * eps)
    checks["gradient_finite_difference"] = bool(torch.allclose(s.grad[idx].float(), fd.float(), atol=1e-3))
    sb = s.detach().to(torch.bfloat16)
    tb = t.detach().to(torch.bfloat16)
    bf16_loss, _ = reverse_kl_exact(sb, tb, mask.to(torch.bfloat16))
    fp32_loss, _ = reverse_kl_exact(s.detach().float(), t.detach().float(), mask.float())
    checks["bf16_vs_fp32_sanity"] = bool(torch.isfinite(bf16_loss) and abs(float(bf16_loss) - float(fp32_loss)) < 0.05)
    checks["response_mask"] = bool(metrics.get("n_tokens") == 1.0)
    span = require_parsable_tool_calls(['to=curate\n{"add_ids":["d1"],"remove_ids":[]}\n</tool_call>'])
    checks["tool_span_mask"] = bool(span)
    adapter = Qwen3NativeChatAdapter()
    chat = adapter.tokenizer_consistency_check()
    checks["qwen3_chat_serialization_mask_alignment"] = bool(chat.get("n_cases") == 10 and chat.get("unique_digests") == 10)
    status = "FRAMEWORK_NUMERICS_READY" if all(checks.values()) else "STOP_FRAMEWORK_NUMERICS_REGRESSION"
    payload = {"status": status, "checks": checks, "reverse_kl_loss": float(loss.detach()), "finite_difference": float(fd.detach()), "grad_component": float(s.grad[idx].detach()), "bf16_loss": float(bf16_loss.detach()), "fp32_loss": float(fp32_loss.detach()), "qwen3_chat": chat}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if status == "FRAMEWORK_NUMERICS_READY" else 3


if __name__ == "__main__":
    raise SystemExit(main())
