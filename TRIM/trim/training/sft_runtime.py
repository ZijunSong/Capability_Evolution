"""Invoke Harness-1 ``training/train_sft.py`` with official v8d + Tinker defaults.

Flags and hyperparameters are copied from ``external/harness-1/training/launch_sft_training.sh``.
The subprocess cwd is the pinned Harness-1 tree so ``ultra_core`` / Tinker cookbook
imports resolve the same way as upstream.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

TRIM_ROOT = Path(__file__).resolve().parents[2]
HARNESS1_ROOT = TRIM_ROOT / "external" / "harness-1"
HARNESS1_TRAIN_SFT = HARNESS1_ROOT / "training" / "train_sft.py"
HARNESS1_COOKBOOK = HARNESS1_ROOT / "tinker-cookbook"

# Official GPT-OSS-20B SFT recipe (launch_sft_training.sh).
HARNESS1_SFT_MODEL_NAME = "openai/gpt-oss-20b"
HARNESS1_SFT_NUM_EPOCHS = 3
HARNESS1_SFT_BATCH_SIZE = 128
HARNESS1_SFT_LEARNING_RATE = 5e-6
HARNESS1_SFT_LORA_RANK = 32
HARNESS1_SFT_MAX_LENGTH = 32768
HARNESS1_SFT_MIN_RECALL = 0.1
HARNESS1_SFT_SAVE_EVERY = 50
HARNESS1_SFT_EVAL_EVERY = 50
HARNESS1_SFT_AUTO_POPULATE_TOP_K = "8"

# v8d flags MUST match SFT generation + RL (see launch_sft_training.sh).
HARNESS1_SFT_V8D_ENV: dict[str, str] = {
    "V8D_SUBTRACTIVE_CURATION": "1",
    "V8D_IMPORTANCE_TAGGING": "1",
    "V8D_AUTO_POPULATE_FIRST_SEARCH": "1",
    "V8D_EVIDENCE_GRAPH": "1",
    "V8D_SENTENCE_COMPRESS": "1",
    "V8D_CONTENT_DEDUP": "1",
    "V8D_VERIFY_TOOL": "1",
    "V8D_TOKEN_BUDGET_MARKER": "1",
    "V8D_ADAPTIVE_RERANK_INSTRUCTION": "1",
    "AUTO_POPULATE_TOP_K": HARNESS1_SFT_AUTO_POPULATE_TOP_K,
}

SMOKE_NUM_EPOCHS = 1
SMOKE_BATCH_SIZE = 4

_MODEL_ALIASES = {
    "gpt-oss-20b": HARNESS1_SFT_MODEL_NAME,
    "gptoss20b": HARNESS1_SFT_MODEL_NAME,
    "openai/gpt-oss-20b": HARNESS1_SFT_MODEL_NAME,
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "gptoss120b": "openai/gpt-oss-120b",
    "openai/gpt-oss-120b": "openai/gpt-oss-120b",
}


def canonical_sft_model_name(value: str | None) -> str:
    text = str(value or "").strip() or HARNESS1_SFT_MODEL_NAME
    key = text.lower().replace(" ", "").replace("_", "-")
    return _MODEL_ALIASES.get(key, text)


def _uv_candidates() -> list[Path]:
    found: list[Path] = []
    which = shutil.which("uv")
    if which:
        found.append(Path(which))
    found.extend(
        [
            Path.home() / ".local/bin/uv",
            Path("/data/ppnm/claw-runtimes/uv/uv"),
            Path("/data/ppnm/miniconda3/envs/malhtb-mafbench/bin/uv"),
        ]
    )
    out: list[Path] = []
    seen: set[Path] = set()
    for path in found:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in seen or not path.is_file():
            continue
        seen.add(resolved)
        out.append(path)
    return out


def _python_candidates(explicit: str | None = None) -> list[str]:
    found = [
        explicit,
        os.environ.get("TRIM_SFT_PYTHON"),
        os.environ.get("HARNESS1_PYTHON"),
        sys.executable,
        "/data/ppnm/miniconda3/envs/bishop/bin/python",
        str(HARNESS1_ROOT / ".venv" / "bin" / "python"),
    ]
    out: list[str] = []
    seen: set[str] = set()
    for item in found:
        if not item:
            continue
        path = Path(item)
        if not path.exists():
            continue
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def python_has_tinker(exe: str, *, env: Mapping[str, str] | None = None) -> bool:
    probe_env = os.environ.copy()
    if env:
        probe_env.update(dict(env))
    probe_env["PYTHONPATH"] = harness1_pythonpath(probe_env.get("PYTHONPATH"))
    try:
        result = subprocess.run(
            [exe, "-c", "import tinker, chz, openai_harmony"],
            capture_output=True,
            timeout=30,
            env=probe_env,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def resolve_sft_python(explicit: str | None = None) -> list[str]:
    """Pick a Python that can import Tinker, matching Harness-1 ``uv run python``."""
    if explicit:
        path = Path(explicit)
        if path.exists():
            return [str(path)]
        which = shutil.which(explicit)
        return [which or explicit]
    for exe in _python_candidates():
        if python_has_tinker(exe):
            return [exe]
    for uv in _uv_candidates():
        return [str(uv), "run", "--project", str(HARNESS1_ROOT), "python"]
    return [sys.executable]


def harness1_pythonpath(existing: str | None = None) -> str:
    parts = [str(HARNESS1_ROOT), str(HARNESS1_COOKBOOK)]
    extra = (existing or "").strip()
    if extra:
        parts.append(extra)
    seen: list[str] = []
    for item in parts:
        if item and item not in seen:
            seen.append(item)
    return os.pathsep.join(seen)


def load_dotenv(path: Path) -> dict[str, str]:
    """Minimal KEY=VALUE loader (no dependency on python-dotenv)."""
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            out[key] = value
    return out


def dotenv_paths() -> list[Path]:
    return [
        HARNESS1_ROOT / ".env.local",
        HARNESS1_ROOT / ".env",
        TRIM_ROOT / ".env.local",
        TRIM_ROOT / ".env",
        Path.home() / ".env.local",
    ]


def resolve_tinker_api_key(env: Mapping[str, str] | None = None) -> str:
    current = dict(env or os.environ)
    if current.get("TINKER_API_KEY"):
        return str(current["TINKER_API_KEY"])
    for path in dotenv_paths():
        loaded = load_dotenv(path)
        if loaded.get("TINKER_API_KEY"):
            return loaded["TINKER_API_KEY"]
    return ""


def sft_subprocess_env(
    *,
    extra: Mapping[str, str] | None = None,
    require_tinker_key: bool = True,
) -> dict[str, str]:
    env = os.environ.copy()
    for path in dotenv_paths():
        for key, value in load_dotenv(path).items():
            env.setdefault(key, value)
    env.update(HARNESS1_SFT_V8D_ENV)
    if extra:
        env.update({str(k): str(v) for k, v in extra.items()})
    env["PYTHONPATH"] = harness1_pythonpath(env.get("PYTHONPATH"))
    if require_tinker_key and not str(env.get("TINKER_API_KEY") or "").strip():
        raise RuntimeError(
            "TINKER_API_KEY is not set. Copy external/harness-1/.env.example "
            "to .env.local or export TINKER_API_KEY (Harness-1 SFT uses Tinker)."
        )
    return env


def train_sft_argv(
    *,
    data_dir: Path | str,
    log_path: Path | str,
    model_name: str = HARNESS1_SFT_MODEL_NAME,
    num_epochs: int = HARNESS1_SFT_NUM_EPOCHS,
    batch_size: int = HARNESS1_SFT_BATCH_SIZE,
    learning_rate: float = HARNESS1_SFT_LEARNING_RATE,
    lora_rank: int = HARNESS1_SFT_LORA_RANK,
    max_length: int = HARNESS1_SFT_MAX_LENGTH,
    min_recall: float = HARNESS1_SFT_MIN_RECALL,
    save_every: int = HARNESS1_SFT_SAVE_EVERY,
    eval_every: int = HARNESS1_SFT_EVAL_EVERY,
    load_checkpoint_path: str | None = None,
    python: str | Sequence[str] | None = None,
) -> list[str]:
    if isinstance(python, (list, tuple)):
        prefix = [str(x) for x in python]
    else:
        prefix = resolve_sft_python(python)
    cmd = [
        *prefix,
        str(HARNESS1_TRAIN_SFT),
        "--data-dir",
        str(data_dir),
        "--log-path",
        str(log_path),
        "--model-name",
        canonical_sft_model_name(model_name),
        "--num-epochs",
        str(int(num_epochs)),
        "--batch-size",
        str(int(batch_size)),
        "--learning-rate",
        str(learning_rate),
        "--lora-rank",
        str(int(lora_rank)),
        "--max-length",
        str(int(max_length)),
        "--min-recall",
        str(min_recall),
        "--save-every",
        str(int(save_every)),
        "--eval-every",
        str(int(eval_every)),
    ]
    if load_checkpoint_path:
        cmd.extend(["--load-checkpoint-path", str(load_checkpoint_path)])
    return cmd


def run_harness1_train_sft(
    argv: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    if not HARNESS1_TRAIN_SFT.is_file():
        raise FileNotFoundError(f"Harness-1 train_sft.py not found: {HARNESS1_TRAIN_SFT}")
    return subprocess.run(
        list(argv),
        cwd=str(cwd or HARNESS1_ROOT),
        env=dict(env or sft_subprocess_env()),
        check=check,
        text=True,
    )
