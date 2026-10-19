"""Teacher SFT. Official layout: 4 nodes x 8 rollout replicas = 32 cards.

    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    PYTHONPATH=TRIM:SCAPE-EasyOPD python TRIM/scripts/run_teacher_sft_multinode.py \\
      --nnodes 4 --nproc-per-node 1 --node-rank $NODE_RANK \\
      --master-addr $MASTER_ADDR --master-port 29611 \\
      --rollout-replicas 8 --tensor-parallel-size 1 \\
      --model_name /models/base-model --out /shared/trim_sft/teacher/seed42 ...
"""

from pathlib import Path
import sys

_TRIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TRIM))

from trim.training.sft_ablation.dist_launch import launch_sft
from trim.training.teacher_sft.runtime import run

if __name__ == "__main__":
    raise SystemExit(launch_sft(
        script=Path(__file__).resolve(), experiment="teacher_sft", worker=run,
    ))
