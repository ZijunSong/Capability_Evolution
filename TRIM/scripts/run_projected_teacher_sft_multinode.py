"""Projected Teacher SFT. Official layout matches zero-rl / all-trim: 4x8 = 32 cards.

Rank 0 is the only gradient rank. Eight replicas per node each get their own GPU.
"""

from pathlib import Path
import sys

_TRIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TRIM))

from trim.training.sft_ablation.dist_launch import launch_sft
from trim.training.projected_teacher_sft.runtime import run

if __name__ == "__main__":
    raise SystemExit(launch_sft(
        script=Path(__file__).resolve(),
        experiment="projected_teacher_sft", worker=run,
    ))
