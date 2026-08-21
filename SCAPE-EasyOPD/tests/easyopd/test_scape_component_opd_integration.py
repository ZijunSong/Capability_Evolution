from __future__ import annotations

import torch

from easyopd.hook_dispatch import HookDispatcher
from easyopd.registry import auto_discover, list_methods


def test_scape_component_opd_registered_and_enabled():
    auto_discover()
    assert "scape_component_opd" in list_methods()
    dispatcher = HookDispatcher.from_config(
        {
            "easyopd": {"method": {"name": "scape_component_opd"}},
            "component": {"name": "evidence_graph"},
            "distillation": {"loss": "reverse_kl"},
        }
    )
    assert dispatcher.enabled
    assert dispatcher.hooks.has_loss
    assert dispatcher.hooks.has_rollout
    assert dispatcher.hooks.has_teacher_sidecar


def test_scape_component_opd_compute_loss_returns_scalar():
    dispatcher = HookDispatcher.from_config(
        {
            "easyopd": {"method": {"name": "scape_component_opd"}},
            "component": {"name": "auto_populate_first_search"},
            "distillation": {"loss": "projected_action_ce"},
        }
    )
    student = torch.randn(2, 3, 5)
    teacher = torch.randn(2, 3, 5)
    mask = torch.ones(2, 3)
    loss, metrics = dispatcher.compute_loss(student_logits=student, teacher_logits=teacher, mask=mask)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert isinstance(metrics, dict)
