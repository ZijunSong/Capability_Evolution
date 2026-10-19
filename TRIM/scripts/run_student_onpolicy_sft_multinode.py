"""Student on-policy SFT. Official layout matches zero-rl / all-trim: 4x8 = 32 cards.

Each round rolls the current student out under Zero, labels with the current
TRIM branches, and applies one action-level CE update on rank 0.
"""

from pathlib import Path
import sys

_TRIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TRIM))

from trim.training.sft_ablation.dist_launch import launch_sft
from trim.training.student_onpolicy_sft.runtime import run

if __name__ == "__main__":
    raise SystemExit(launch_sft(
        script=Path(__file__).resolve(),
        experiment="student_onpolicy_sft", worker=run,
    ))
