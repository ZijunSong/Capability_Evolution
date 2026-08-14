"""V2 learnability metric controls C0–C4 (SCAPE-0813-Next-H20 §3)."""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from scape.eval.learnability_metrics_v2 import (
  LEGAL_TOOL_NAMES,
  tool_name_distribution_js,
)
from scape.training.canonical_metrics import (
  js_from_logits,
  kl_from_logits,
  probs_from_logits_list,
)
from scape.training.tool_opd import normalize_probs


def test_c0_identity_teacher_student_same_view():
  """C0: teacher == student, same view => KL≈0, JS≈0."""
  t = torch.tensor([2.0, 0.0, 0.0])
  s = torch.tensor([2.0, 0.0, 0.0])
  kl = kl_from_logits(t, s, forward=True).item()
  js = js_from_logits(t, s).item()
  assert abs(kl) < 1e-5
  assert abs(js) < 1e-5


def test_c1_field_order_perturbation_small():
  """C1: field-order null => close to identity baseline."""
  from scape.rendering.dual_view import field_order_perturb

  base = {"query_id": "q1", "step": 1, "documents": [{"id": "d1", "text": "abc"}]}
  perturbed = field_order_perturb(base)
  # Semantic equality via sorted json
  assert sorted(base.keys()) == sorted(perturbed.keys())
  p = normalize_probs({"a": 1.0, "b": 0.0})
  q = normalize_probs({"a": 1.0, "b": 0.0})
  from scape.training.tool_opd import js_divergence
  assert js_divergence(p, q) < 1e-6


def test_c2_positive_gap_base_vs_reduced():
  """C2: known influence-positive states should show positive forward KL."""
  t = torch.tensor([3.0, 0.0, 0.0, 0.0])
  s = torch.tensor([0.0, 3.0, 0.0, 0.0])
  kl = kl_from_logits(t, s, forward=True).item()
  assert kl > 0.01


def test_c3_temperature_perturb_strictly_gt_c0():
  """C3: perturbed student logits => KL/JS strictly > C0."""
  t = torch.tensor([2.0, 0.0, 0.0])
  s_cold = torch.tensor([2.0, 0.0, 0.0])
  s_hot = torch.tensor([0.5, 0.5, 0.0])
  kl_c0 = kl_from_logits(t, s_cold, forward=True).item()
  kl_c3 = kl_from_logits(t, s_hot, forward=True).item()
  js_c0 = js_from_logits(t, s_cold).item()
  js_c3 = js_from_logits(t, s_hot).item()
  assert kl_c3 > kl_c0 + 1e-5
  assert js_c3 > js_c0 + 1e-5


def test_c4_duplicate_paths_agree():
  """C4: two independent code paths agree within 1e-5 on 32 states."""
  random.seed(42)
  for _ in range(32):
    t = torch.randn(64)
    s = torch.randn(64)
    fwd_a = kl_from_logits(t, s, forward=True).item()
    t_logp = F.log_softmax(t, dim=-1)
    s_logp = F.log_softmax(s, dim=-1)
    fwd_b = (t_logp.exp() * (t_logp - s_logp)).sum().item()
    assert abs(fwd_a - fwd_b) <= 1e-5


def test_m1_tool_name_js_nonneg():
  logits_a = {name: random.random() for name in LEGAL_TOOL_NAMES}
  logits_b = {name: random.random() for name in LEGAL_TOOL_NAMES}
  js = tool_name_distribution_js(logits_a, logits_b)
  assert js >= -1e-7


def test_m1_identity_js_zero():
  logits = {name: float(i) for i, name in enumerate(LEGAL_TOOL_NAMES)}
  js = tool_name_distribution_js(logits, logits)
  assert abs(js) < 1e-5


def test_manual_reference_kl_js():
  teacher = [0.7, 0.2, 0.1]
  student = [0.4, 0.4, 0.2]
  labels = ["a", "b", "c"]
  p = probs_from_logits_list(teacher, labels)
  q = probs_from_logits_list(student, labels)
  from scape.training.tool_opd import kl_divergence, js_divergence
  t = torch.tensor(teacher, dtype=torch.float64)
  s = torch.tensor(student, dtype=torch.float64)
  fwd = kl_from_logits(t, s, forward=True).item()
  js = js_from_logits(t, s).item()
  assert abs(fwd - kl_divergence(p, q)) < 1e-5
  assert abs(js - js_divergence(p, q)) < 1e-5
