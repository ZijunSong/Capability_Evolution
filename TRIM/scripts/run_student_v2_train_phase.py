"""Student on-policy SFT v2 train-phase launcher (Accelerate DDP).

Drives ``run_student_v2`` (``trim.training.sft_engine.online_runtime``) which
consumes ONE sealed round via ``--resume-round`` and applies a single global
action-position CE update on every training rank (Accelerate DDP, all_reduce
gradient sync), then publishes the adapter/optimizer/scheduler and exits.

This is the phase_jobs layout: rollout and training do NOT share a GPU in one
job. A CloudML job launched here performs exactly ONE optimizer update from a
sealed targets directory, so it is the unit for:
  * the first-round recovery/validation run (one update + profile), and
  * each update step once the resident online loop is wired.

Launch with ``accelerate launch --multi_gpu --num_machines N --num_processes
W --mixed_precision bf16`` so the Accelerator process group is initialized by
accelerate (the entry unsets the CloudML-injected RANK/WORLD_SIZE that would
otherwise make Accelerate think this is a single-process run).
"""

from pathlib import Path
import sys

_TRIM = Path(__file__).resolve().parents[1]
if str(_TRIM) not in sys.path:
    sys.path.insert(0, str(_TRIM))

from trim.training.sft_ablation.cli import parse_sft_args
from trim.training.sft_engine.online_runtime import run_student_v2

if __name__ == "__main__":
    args = parse_sft_args("student_onpolicy_sft")
    raise SystemExit(run_student_v2(args))
