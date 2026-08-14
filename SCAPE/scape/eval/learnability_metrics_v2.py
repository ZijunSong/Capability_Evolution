"""Learnability Measurement Contract V2 (SCAPE-0813-Next-H20).

M1  JS_name          — Jensen-Shannon on legal tool-name distribution, >= 0
M2  CE_T_on_S        — cross-entropy of student on teacher-forced tokens, >= 0
M3  KL_name/args     — forward KL(T||S) on name / arg spans, >= 0
M4  action_agreement — tool-name / exact-call / arg similarity / invalid_tool_rate
M_diag signed_logprob_gap — diagnostic only; NOT a divergence; NOT a gate metric
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from scape.training.canonical_metrics import NUMERIC_FLOOR, aggregate_token_metrics
from scape.training.hf_tool_opd import ScapeHFToolOPD, LossPath
from scape.training.tool_mask import legal_tool_names, tool_loss_mask_from_response
from scape.training.tool_opd import disagreement_stats, normalize_probs

LEGAL_TOOL_NAMES = tuple(legal_tool_names())


@dataclass
class LearnabilityMetricsV2:
    """Per-eval-set aggregate metrics."""

    JS_name: float = 0.0
    CE_T_on_S: float = 0.0
    KL_name: float = 0.0
    KL_arg_key: float = 0.0
    KL_arg_value: float = 0.0
    forward_KL: float = 0.0
    reverse_KL: float = 0.0
    JS_tool: float = 0.0
    signed_logprob_gap: float = 0.0
    tool_name_agreement: float = 0.0
    exact_tool_call_agreement: float = 0.0
    argument_key_agreement: float = 0.0
    invalid_tool_rate: float = 0.0
    n_rows: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "JS_name": self.JS_name,
            "CE_T_on_S": self.CE_T_on_S,
            "KL_name": self.KL_name,
            "KL_arg_key": self.KL_arg_key,
            "KL_arg_value": self.KL_arg_value,
            "forward_KL": self.forward_KL,
            "reverse_KL": self.reverse_KL,
            "JS_tool": self.JS_tool,
            "signed_logprob_gap": self.signed_logprob_gap,
            "tool_name_agreement": self.tool_name_agreement,
            "exact_tool_call_agreement": self.exact_tool_call_agreement,
            "argument_key_agreement": self.argument_key_agreement,
            "invalid_tool_rate": self.invalid_tool_rate,
            "n_rows": self.n_rows,
        }


def _tool_name_js_from_logits(
  teacher_logits: Any,
  student_logits: Any,
  legal: Sequence[str] = LEGAL_TOOL_NAMES,
) -> float:
  """M1: JS over legal tool-name token support (first name token per row proxy)."""
  import torch
  import torch.nn.functional as F
  from scape.training.canonical_metrics import js_from_logits

  if teacher_logits.numel() == 0:
    return 0.0
  # Use first position as tool-name span proxy when full name spans unavailable
  t = teacher_logits[0]
  s = student_logits[0]
  # Restrict to legal tool token ids is expensive; use full vocab JS at name position
  js = js_from_logits(t.unsqueeze(0), s.unsqueeze(0)).item()
  return max(NUMERIC_FLOOR, float(js))


def score_row_v2(
  teacher: ScapeHFToolOPD,
  student: ScapeHFToolOPD,
  row: Mapping[str, Any],
  *,
  loss_path: LossPath = "tool_token_kl",
) -> dict[str, float]:
  """Score one same-state row with V2 contract."""
  import torch
  from scape.training.canonical_metrics import (
    js_from_logits,
    kl_from_logits,
    signed_logprob_gap,
  )

  prompt_reduced = row["prompt_reduced"]
  prompt_full = row["prompt_full"]
  response_text = row["response_text"]
  resp_ids = student.encode(response_text)
  if not resp_ids:
    return {
      "JS_name": 0.0,
      "CE_T_on_S": 0.0,
      "KL_name": 0.0,
      "KL_arg_key": 0.0,
      "KL_arg_value": 0.0,
      "forward_KL": 0.0,
      "reverse_KL": 0.0,
      "JS_tool": 0.0,
      "signed_logprob_gap": 0.0,
      "tool_name_agreement": 1.0,
      "exact_tool_call_agreement": 1.0,
      "argument_key_agreement": 1.0,
      "invalid_tool_rate": 0.0,
    }

  red_ids = student.encode(prompt_reduced)
  full_ids = teacher.encode(prompt_full)
  with torch.no_grad():
    s_logits = student._response_position_logits(red_ids, resp_ids, require_grad=False)
    t_logits = teacher._response_position_logits(full_ids, resp_ids, require_grad=False)
    s_lp = student._teacher_forced_logprobs(red_ids, resp_ids, require_grad=False)

  fwd = kl_from_logits(t_logits, s_logits, forward=True)
  rev = kl_from_logits(t_logits, s_logits, forward=False)
  js = js_from_logits(t_logits, s_logits)
  gap = signed_logprob_gap(t_logits, s_logits, resp_ids)

  spans = student.span_token_masks(response_text, len(resp_ids))
  token_mask = student.response_token_mask(response_text, loss_path=loss_path)
  if len(token_mask) != len(resp_ids):
    token_mask = spans["tool"]
  m_tool = torch.tensor(token_mask, device=fwd.device, dtype=fwd.dtype)
  m_name = torch.tensor(spans["name"], device=fwd.device, dtype=fwd.dtype)
  m_key = torch.tensor(spans["key"], device=fwd.device, dtype=fwd.dtype)
  m_val = torch.tensor(spans["value"], device=fwd.device, dtype=fwd.dtype)

  agg = aggregate_token_metrics(
    fwd, rev, js, gap, m_tool,
    name_mask=m_name, key_mask=m_key, value_mask=m_val,
  )

  # M2: CE_T_on_S = -mean log p_S(teacher token)
  ce_mask = m_tool
  if float(ce_mask.sum().item()) > 0:
    ce = float((-s_lp * ce_mask).sum().item() / ce_mask.sum().item())
  else:
    ce = float(-s_lp.mean().item())

  # M1: JS on name spans (subset of tool JS)
  if float(m_name.sum().item()) > 0:
    js_name = masked_mean_js(t_logits, s_logits, m_name)
  else:
    js_name = agg["JS"]

  # M4: action agreement from student_action vs teacher (response is teacher action)
  student_action = row.get("student_action") or {}
  teacher_action = _parse_response_action(response_text)
  disagree = disagreement_stats(
    {"name": student_action.get("name"), "arguments": student_action.get("arguments") or {}},
    teacher_action,
  )
  s_args = dict(student_action.get("arguments") or {})
  t_args = dict(teacher_action.get("arguments") or {})
  key_agree = 1.0 if set(s_args.keys()) == set(t_args.keys()) else 0.0
  name_audit = tool_loss_mask_from_response(response_text)
  invalid = 1.0 if name_audit["n_tool_name"] < 1 else 0.0

  return {
    "JS_name": js_name,
    "CE_T_on_S": ce,
    "KL_name": agg["tool_name_KL"],
    "KL_arg_key": agg["arg_key_KL"],
    "KL_arg_value": agg["arg_value_KL"],
    "forward_KL": agg["forward_KL"],
    "reverse_KL": agg["reverse_KL"],
    "JS_tool": agg["JS"],
    "signed_logprob_gap": agg["signed_gap"],
    "tool_name_agreement": 1.0 - float(disagree["tool_name_disagreement"]),
    "exact_tool_call_agreement": 1.0 - float(disagree["exact_tool_call_disagreement"]),
    "argument_key_agreement": key_agree,
    "invalid_tool_rate": invalid,
  }


def masked_mean_js(
  t_logits: Any,
  s_logits: Any,
  mask: Any,
) -> float:
  from scape.training.canonical_metrics import js_from_logits, masked_mean

  js = js_from_logits(t_logits, s_logits)
  return masked_mean(js, mask)


def _parse_response_action(response_text: str) -> dict[str, Any]:
  """Parse first tool call from response text."""
  lines = response_text.strip().splitlines()
  if not lines:
    return {"name": None, "arguments": {}}
  name_line = lines[0]
  if name_line.startswith("to="):
    name = name_line.split("=", 1)[1].strip()
  else:
    name = name_line
  args: dict[str, Any] = {}
  for line in lines[1:]:
    line = line.strip()
    if line.startswith("{") and line.endswith("}"):
      try:
        args = json.loads(line)
      except json.JSONDecodeError:
        pass
      break
  return {"name": name, "arguments": args}


def aggregate_rows_v2(
  teacher: ScapeHFToolOPD,
  student: ScapeHFToolOPD,
  rows: Sequence[Mapping[str, Any]],
  *,
  loss_path: LossPath = "tool_token_kl",
  max_rows: int | None = None,
) -> LearnabilityMetricsV2:
  subset = list(rows[:max_rows] if max_rows else rows)
  if not subset:
    return LearnabilityMetricsV2()

  keys = [
    "JS_name", "CE_T_on_S", "KL_name", "KL_arg_key", "KL_arg_value",
    "forward_KL", "reverse_KL", "JS_tool", "signed_logprob_gap",
    "tool_name_agreement", "exact_tool_call_agreement",
    "argument_key_agreement", "invalid_tool_rate",
  ]
  acc = {k: 0.0 for k in keys}
  for row in subset:
    m = score_row_v2(teacher, student, row, loss_path=loss_path)
    for k in keys:
      acc[k] += m[k]
  n = len(subset)
  return LearnabilityMetricsV2(
    **{k: acc[k] / n for k in keys},
    n_rows=n,
  )


def learnability_improved(pre: LearnabilityMetricsV2, post: LearnabilityMetricsV2) -> dict[str, bool]:
  """V2 gate directions: lower divergence / CE is better."""
  return {
    "JS_name_improved": post.JS_name < pre.JS_name - 1e-6,
    "CE_improved": post.CE_T_on_S < pre.CE_T_on_S - 1e-6,
    "KL_name_improved": post.KL_name < pre.KL_name - 1e-6,
    "invalid_not_worse": post.invalid_tool_rate <= pre.invalid_tool_rate + 1e-6,
  }


def v2_gate_pass(pre: LearnabilityMetricsV2, post: LearnabilityMetricsV2) -> tuple[bool, str]:
  imp = learnability_improved(pre, post)
  if imp["JS_name_improved"] and imp["CE_improved"] and imp["invalid_not_worse"]:
    return True, "v2_js_ce_pass"
  if imp["JS_name_improved"] and imp["KL_name_improved"] and imp["invalid_not_worse"]:
    return True, "v2_js_kl_name_pass"
  reasons = []
  if not imp["JS_name_improved"]:
    reasons.append("JS_name_not_improved")
  if not imp["CE_improved"]:
    reasons.append("CE_not_improved")
  if not imp["invalid_not_worse"]:
    reasons.append("invalid_tool_rate_worse")
  return False, "+".join(reasons) or "no_improvement"


def tool_name_distribution_js(
  p_logits: Mapping[str, float],
  q_logits: Mapping[str, float],
) -> float:
  """Discrete JS over legal tool names from logit dicts."""
  from scape.training.tool_opd import js_divergence

  keys = list(LEGAL_TOOL_NAMES)
  p = normalize_probs({k: p_logits.get(k, 0.0) for k in keys}, as_logits=True)
  q = normalize_probs({k: q_logits.get(k, 0.0) for k in keys}, as_logits=True)
  return js_divergence(p, q)
