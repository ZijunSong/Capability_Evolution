"""Round 10 HF↔vLLM parity contract constants."""

from __future__ import annotations

# Min eps that clears residual HF↔vLLM flips on P0 offline+holdout after disable_replan
# (max |signed margin| over disagreeing events ≈ 0.787). Slightly rounded up.
NEAR_BOUNDARY_PREFER_CONTINUE_EPS = 0.79
