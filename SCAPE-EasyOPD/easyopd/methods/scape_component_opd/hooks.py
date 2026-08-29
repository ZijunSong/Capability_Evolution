from __future__ import annotations

from typing import Any

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - import-time fallback for CPU-only envs
    class _TorchFallback:
        Tensor = Any

    torch = _TorchFallback()  # type: ignore[assignment]

from easyopd.hooks import Config, LossContext, Metrics, LossHook, RolloutHook, RewardHook, TeacherSidecarHook, AlignmentHook, Batch, MethodHooks

from .controls import assert_query_disjoint, query_disjoint_splits
from .scape_agent_loop import SCAPEAgentLoop
from .state_snapshot import SCAPEStateSnapshot, assert_same_state_before_component_fork
from .teacher_sidecar import SCAPETeacherSidecar
from .tool_span import require_parsable_tool_calls


class SCAPELossHook:
    def compute_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, mask: torch.Tensor, config: Config, **kwargs: Any):
        from .losses import alpha_jsd, forward_kl_exact, masked_action_ce, projected_action_ce, reverse_kl_exact

        loss_name = config.get("distillation", {}).get("loss", "projected_action_ce") if isinstance(config, dict) else getattr(getattr(config, "distillation", {}), "loss", "projected_action_ce")
        if loss_name == "forward_kl":
            return forward_kl_exact(student_logits, teacher_logits, mask)
        if loss_name == "reverse_kl":
            return reverse_kl_exact(student_logits, teacher_logits, mask)
        if loss_name == "jsd":
            return alpha_jsd(student_logits, teacher_logits, mask)
        if loss_name == "action_ce":
            target = kwargs.get("target_token_ids")
            if target is None:
                target = teacher_logits.argmax(dim=-1)
            return masked_action_ce(student_logits, target, mask)
        if loss_name == "projected_action_ce":
            target = kwargs.get("target_token_ids")
            if target is None:
                target = teacher_logits.argmax(dim=-1)
            return projected_action_ce(student_logits, target, mask)
        target = kwargs.get("target_token_ids")
        if target is None:
            target = teacher_logits.argmax(dim=-1)
        return projected_action_ce(student_logits, target, mask)

    def compute_loss_with_context(self, context: LossContext):
        config = context.config or {}
        student_logits = context.student_log_probs
        teacher_logits = context.teacher_log_probs
        mask = context.response_mask
        if student_logits is None or teacher_logits is None or mask is None:
            return torch.tensor(0.0), {}
        return self.compute_loss(student_logits, teacher_logits, mask, config, **(context.extra_kwargs or {}))


class SCAPERolloutHook:
    def on_rollout_end(self, batch: Batch, config: Config, **kwargs: Any) -> Batch:
        if isinstance(batch, dict):
            row = dict(batch)
        else:
            row = dict(getattr(batch, "__dict__", {}))
        component = (config.get("component", {}) if isinstance(config, dict) else getattr(config, "component", {})) or {}
        component_name = component.get("name") if isinstance(component, dict) else getattr(component, "name", "evidence_graph")
        loop = SCAPEAgentLoop(str(component_name), student_inference_privilege=False)
        row["scape_available_tools"] = loop.available_tools(include_verify=bool(component_name == "verify_tool"))
        row["scape_teacher_does_not_step"] = True
        row["scape_state_fork"] = True
        row["scape_student_privilege"] = False
        return row


class SCAPETeacherSidecarHook:
    def teacher_forward(self, batch: Batch, teacher_model: Any, config: Config, **kwargs: Any) -> Any:
        component = (config.get("component", {}) if isinstance(config, dict) else getattr(config, "component", {})) or {}
        component_name = component.get("name") if isinstance(component, dict) else getattr(component, "name", "evidence_graph")
        sidecar = SCAPETeacherSidecar(str(component_name), mode=str((config or {}).get("teacher", {}).get("mode", "same_weights_privileged_view")) if isinstance(config, dict) else "same_weights_privileged_view")
        return sidecar.teacher_forward(batch, teacher_model, config=config)


class SCAPEAlignmentHook:
    def build_alignment(self, student_tokenizer: Any, teacher_tokenizer: Any, input_ids: torch.Tensor, config: Config, **kwargs: Any) -> Any:
        return {"student_tokenizer": str(type(student_tokenizer).__name__), "teacher_tokenizer": str(type(teacher_tokenizer).__name__), "input_len": int(input_ids.numel())}


class SCAPERewardHook:
    def compute_reward(self, batch: Batch, config: Config, **kwargs: Any) -> torch.Tensor:
        if isinstance(batch, dict):
            rewards = batch.get("rewards")
        else:
            rewards = getattr(batch, "rewards", None)
        if rewards is None:
            return torch.zeros(1)
        if isinstance(rewards, torch.Tensor):
            return rewards
        return torch.tensor(rewards)


def build_hooks(config: Any) -> MethodHooks:
    component = config.get("component", {}) if isinstance(config, dict) else getattr(config, "component", {})
    component_name = component.get("name") if isinstance(component, dict) else getattr(component, "name", "evidence_graph")
    hooks = {
        "loss_hook": SCAPELossHook(),
        "rollout_hook": SCAPERolloutHook(),
        "teacher_sidecar_hook": SCAPETeacherSidecarHook(),
    }
    if component_name in {"evidence_graph", "sentence_compress", "adaptive_rerank_instruction", "token_budget_marker"}:
        hooks["alignment_hook"] = SCAPEAlignmentHook()
    if component_name in {"verify_tool", "content_dedup"}:
        hooks["reward_hook"] = SCAPERewardHook()
    return MethodHooks(**hooks)
