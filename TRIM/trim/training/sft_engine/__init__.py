"""Accelerate DDP action-CE engine for the three SFT ablations.

Importing this package does not load a model, initialize a process group,
or touch the RL/OPD trainers.
"""

__all__ = [
    "action_ce_numerator",
    "plan_slots",
    "run_fixed_sft",
    "run_student_v2",
]
