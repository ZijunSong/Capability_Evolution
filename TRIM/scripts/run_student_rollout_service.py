"""One persistent Student rollout worker. Bind it to GPUs the CE job does not use."""

from pathlib import Path
import sys

_TRIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TRIM))

from trim.training.student_onpolicy_sft.rollout_service import main

if __name__ == "__main__":
    raise SystemExit(main())
