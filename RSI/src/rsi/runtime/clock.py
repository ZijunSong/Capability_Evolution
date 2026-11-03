"""Monotonic wall-clock source immune to clock freezing.

AppWorld freezes ``time.time`` / ``time.perf_counter`` via freezegun when a
task environment is initialized (``AppWorld.initialize -> _set_datetime``),
replacing them with the task's simulated datetime so that app code observes a
deterministic clock.  That patch is process-global and active for the whole
episode, so *any* elapsed-time measurement built on ``time.perf_counter``
silently measures the simulated date instead of real time.

This burned the H0 eval twice (kygt9/1nvje/cmjpt, 2026-11-02): ``BudgetLedger``
samples ``perf_counter`` before ``env.reset`` and again at the first loop
check.  After the freeze the second reading jumps to the task epoch
(~1.6e9 s), so ``check_wall`` computes an "elapsed" of ~53 years and every
episode dies with ``budget_wall`` before the model is ever called — zero
decisions, zero tokens, and per-task scores that are pure initial-state noise
(identical across models).

``time.clock_gettime(CLOCK_MONOTONIC)`` is not patched by freezegun and reads
the same underlying monotonic clock as ``perf_counter`` did, so it is the drop-in
replacement for every *elapsed* measurement in RSI.
"""

from __future__ import annotations

import time


def monotonic_ms() -> float:
    """Real elapsed-time origin in milliseconds; freezegun does not patch it."""
    return time.clock_gettime(time.CLOCK_MONOTONIC) * 1000.0


def monotonic_s() -> float:
    """Real elapsed-time origin in seconds; freezegun does not patch it."""
    return time.clock_gettime(time.CLOCK_MONOTONIC)
