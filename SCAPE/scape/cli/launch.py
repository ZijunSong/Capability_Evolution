"""Shared CLI for one-click Harness-1 / BC+ train and eval launchers.

`--component` is a list of Harness-1 components to turn on for the experiment.
Training uses Teacher H_full with those flags enabled and Student H_-S with
them disabled. Eval without an adapter runs the base harness with those flags
on; eval with `--run-dir` / `--adapter` scores the internalized student.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from scape.adapters.components import all_component_ids, coalition_minus_mask, full_mask, zero_mask
from scape.eval.eval_defaults import (
    HARNESS1_EVAL_MAX_MODEL_LEN,
    HARNESS1_EVAL_MAX_NEW_TOKENS,
    HARNESS1_EVAL_MAX_TURNS,
    HARNESS1_EVAL_SEARCH_K,
    HARNESS1_EVAL_TEMPERATURE,
)
from scape.training.rl_opd_types import (
    OPD_LOSS_PROJECTED_GAP,
    OPD_LOSS_SAMPLED_GAP,
    SCAPE_RL_LAMBDA_OPD,
    SCAPE_RL_OPD_GATE_BETA,
    TRAINING_MODE_PURE_OPD,
    TRAINING_MODE_RL,
    TRAINING_MODE_RL_OPD,
    TRAINING_MODE_SCAPE_RL,
    TRAINING_MODE_SCAPE_SEED,
)
from scape.eval.official_query_pool import SCORE_SPLIT_166, SCORE_SPLIT_830
from scape.eval.sec_corpus import default_sec_corpus_root, default_sec_rl_data

SCAPE_ROOT = Path(__file__).resolve().parents[2]

ALLOWED_HARNESSES = ("Harness-1",)
ALLOWED_BENCHMARKS = ("BC+",)
ALLOWED_MODEL_NAMES = ("harness-1",)
ALLOWED_TRAIN_METHODS = ("opd", "rl+opd", "rl", "scape+rl", "scape+seed", "seed+opd")
CANONICAL_COMPONENTS = tuple(all_component_ids())

_HARNESS_ALIASES = {
    "harness-1": "Harness-1",
    "harness1": "Harness-1",
    "harness_1": "Harness-1",
}

_BENCHMARK_ALIASES = {
    "bc+": "BC+",
    "bcplus": "BC+",
    "browsecomp+": "BC+",
    "browsecomp_plus": "BC+",
    "browsecomp-plus": "BC+",
    "browsecompplus": "BC+",
}

_MODEL_ALIASES = {
    "harness-1": "harness-1",
    "harness1": "harness-1",
    "harness_1": "harness-1",
}

_TRAIN_METHOD_ALIASES = {
    "opd": "opd",
    "pure_opd": "opd",
    "pure-opd": "opd",
    "rl": "rl",
    "rl+opd": "rl+opd",
    "rl_opd": "rl+opd",
    "rl-opd": "rl+opd",
    "scape+rl": "scape+rl",
    "scape_rl": "scape+rl",
    "scape-rl": "scape+rl",
    "scape+seed": "scape+seed",
    "scape_seed": "scape+seed",
    "scape-seed": "scape+seed",
    "seed+opd": "scape+seed",
    "seed_opd": "scape+seed",
    "seed-opd": "scape+seed",
}

_COMPONENT_ALIASES = {
    cid: cid for cid in CANONICAL_COMPONENTS
} | {
    "auto": "auto_populate_first_search",
    "auto_populate": "auto_populate_first_search",
    "graph": "evidence_graph",
    "compress": "sentence_compress",
    "sentence": "sentence_compress",
    "neighbors": "chunk_neighbors",
    "chunk": "chunk_neighbors",
    "dedup": "content_dedup",
    "verify": "verify_tool",
    "budget": "token_budget_marker",
    "token_budget": "token_budget_marker",
    "rerank": "adaptive_rerank_instruction",
    "adaptive_rerank": "adaptive_rerank_instruction",
    "subtractive": "subtractive_curation",
    "curation": "subtractive_curation",
    "importance": "importance_tagging",
    "tagging": "importance_tagging",
}

DEFAULT_BASE_MODEL_CANDIDATES = (
    Path("/mnt/songzijun/models/pat-jj_harness-1-full/harness-1"),
    Path("/data/ppnm/models/pat-jj_harness-1-full/harness-1"),
)
DEFAULT_BASE_MODEL = DEFAULT_BASE_MODEL_CANDIDATES[0]

_TRAIN_METHOD_TO_MODE = {
    "opd": TRAINING_MODE_PURE_OPD,
    "rl": TRAINING_MODE_RL,
    "rl+opd": TRAINING_MODE_RL_OPD,
    "scape+rl": TRAINING_MODE_SCAPE_RL,
    "scape+seed": TRAINING_MODE_SCAPE_SEED,
}


class LaunchError(ValueError):
    """Invalid one-click launcher argument."""


@dataclass(frozen=True)
class LaunchSpec:
    harness: str
    benchmark: str
    model_name: str
    components: tuple[str, ...]
    train_method: str | None = None
    base_model: Path = DEFAULT_BASE_MODEL
    out: Path = field(default_factory=lambda: SCAPE_ROOT / "outputs")
    extra: dict = field(default_factory=dict)

    @property
    def coalition(self) -> str:
        return ",".join(self.components) if self.components else "zero"

    @property
    def zero_components(self) -> bool:
        return not self.components

    @property
    def training_mode(self) -> str | None:
        if self.train_method is None:
            return None
        return train_method_to_mode(self.train_method)


def _norm(value: str) -> str:
    return str(value or "").strip().lower().replace(" ", "")


def _pick(value: str, aliases: dict[str, str], allowed: Sequence[str], flag: str) -> str:
    key = _norm(value)
    resolved = aliases.get(key)
    if resolved is None:
        raise LaunchError(f"{flag}={value!r} is not supported; currently only {list(allowed)}")
    return resolved


def parse_component_tokens(raw: Iterable[str] | str | None) -> list[str]:
    """Split `--component a b` and `--component a,b` into tokens."""
    if raw is None:
        return []
    if isinstance(raw, str):
        parts = [raw]
    else:
        parts = [str(x) for x in raw]
    tokens: list[str] = []
    for part in parts:
        text = part.replace(";", ",")
        for tok in text.split(","):
            item = tok.strip()
            if item:
                tokens.append(item)
    return tokens


def canonical_component_ids(raw: Iterable[str] | str | None) -> list[str]:
    tokens = parse_component_tokens(raw)
    if not tokens:
        raise LaunchError(
            "--component is required; pass `zero`, `all`, or one or more of: "
            + ", ".join(CANONICAL_COMPONENTS)
        )
    if any(_norm(tok) == "zero" for tok in tokens):
        if len(tokens) != 1 or _norm(tokens[0]) != "zero":
            raise LaunchError("--component zero cannot be combined with other component ids")
        return []
    if len(tokens) == 1 and _norm(tokens[0]) == "all":
        return list(CANONICAL_COMPONENTS)
    seen: set[str] = set()
    ids: list[str] = []
    unknown: list[str] = []
    for tok in tokens:
        cid = _COMPONENT_ALIASES.get(_norm(tok))
        if cid is None:
            unknown.append(tok)
            continue
        if cid not in seen:
            seen.add(cid)
            ids.append(cid)
    if unknown:
        raise LaunchError(
            "unknown --component value(s): "
            + ", ".join(unknown)
            + "; allowed: zero, all, "
            + ", ".join(CANONICAL_COMPONENTS)
        )
    return ids


def coalition_slug(component_ids: Sequence[str]) -> str:
    return "+".join(component_ids) if component_ids else "zero"


def train_method_to_mode(method: str) -> str:
    key = _TRAIN_METHOD_ALIASES.get(_norm(method))
    if key is None:
        raise LaunchError(
            f"--train_method={method!r} is not supported; use {list(ALLOWED_TRAIN_METHODS)}"
        )
    return _TRAIN_METHOD_TO_MODE[key]


def resolve_model_path(model_name: str, explicit: str | Path | None = None) -> Path:
    if explicit:
        return Path(explicit)
    _pick(model_name, _MODEL_ALIASES, ALLOWED_MODEL_NAMES, "--model_name")
    for candidate in DEFAULT_BASE_MODEL_CANDIDATES:
        if candidate.exists():
            return candidate
    return DEFAULT_BASE_MODEL


def student_mask_for_ids(component_ids: Sequence[str]) -> dict[str, bool]:
    if not component_ids:
        return zero_mask()
    return coalition_minus_mask(component_ids)


def teacher_mask_for_ids(component_ids: Sequence[str]) -> dict[str, bool]:
    if not component_ids:
        return zero_mask()
    mask = full_mask()
    for cid in component_ids:
        mask[cid] = True
    return mask


def default_out_dir(
    *,
    kind: str,
    harness: str,
    benchmark: str,
    model_name: str,
    components: Sequence[str],
    train_method: str | None = None,
) -> Path:
    parts = [
        kind,
        _norm(harness).replace("-", ""),
        _norm(benchmark).replace("+", "plus"),
        _norm(model_name),
    ]
    if train_method:
        parts.append(_norm(train_method).replace("+", "_"))
    parts.append(coalition_slug(components))
    return SCAPE_ROOT / "outputs" / "_".join(parts)


def discover_adapter_map(run_dir: Path | None) -> dict[str, str]:
    """Find saved LoRA adapters under a training `--out` directory."""
    if run_dir is None:
        return {}
    root = Path(run_dir)
    candidates = [
        root / "seed42" / "adapters",
        root / "adapters",
    ]
    for seed_dir in sorted(root.glob("seed*")):
        adapters = seed_dir / "adapters"
        if adapters not in candidates:
            candidates.append(adapters)
    mapping: dict[str, str] = {}
    preferred = ("scape_seed", "scape_rl", "rl_opd", "pure_opd", "rl", "before", "theta0")
    for adapters in candidates:
        if not adapters.is_dir():
            continue
        found: dict[str, str] = {}
        for cell in preferred:
            path = adapters / cell
            if (path / "adapter_model.safetensors").is_file() or (path / "adapter_config.json").is_file():
                found[cell] = str(path)
        for path in sorted(adapters.iterdir()):
            if path.is_dir() and path.name not in found:
                if (path / "adapter_model.safetensors").is_file() or (path / "adapter_config.json").is_file():
                    found[path.name] = str(path)
        if found:
            mapping = found
            break
    adapter_json = root / "adapter_map.json"
    if adapter_json.is_file():
        import json

        payload = json.loads(adapter_json.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            mapping.update({str(k): str(v) for k, v in payload.items() if v})
    return mapping


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--harness",
        default="Harness-1",
        help="Search harness. Currently only Harness-1.",
    )
    parser.add_argument(
        "--benchmark",
        default="BC+",
        help="Evaluation benchmark. Currently only BC+. scape+rl eval uses the full 830 (664+166).",
    )
    parser.add_argument(
        "--model_name",
        "--model-name",
        dest="model_name",
        default="harness-1",
        help="Base model name. Currently only harness-1.",
    )
    parser.add_argument(
        "--component",
        nargs="+",
        required=True,
        metavar="ID",
        help=(
            "Harness-1 components to enable. Space- or comma-separated list. "
            "Allowed: " + ", ".join(CANONICAL_COMPONENTS) + ". "
            "Pass `all` to enable every component, or `zero` to enable none."
        ),
    )
    parser.add_argument("--out", type=Path, default=None, help="Output directory.")
    parser.add_argument(
        "--base-model",
        default="",
        help="Override checkpoint path for --model_name harness-1.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-eval", type=int, default=None, help="Optional eval subset. Default is the full score split.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--rollout-backend", choices=("vllm", "hf"), default="vllm")
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    return parser


def add_train_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    add_common_args(parser)
    parser.add_argument(
        "--train_method",
        "--train-method",
        dest="train_method",
        required=True,
        choices=ALLOWED_TRAIN_METHODS,
        help="opd = PURE OPD (sr_opd_ce); rl+opd = CISPO + CE OPD; rl = CISPO only; scape+rl = CISPO + SEED on sampled actions; scape+seed = CISPO + projected actions + SEED-scale gap.",
    )
    parser.add_argument(
        "--n-queries",
        type=int,
        default=None,
        help="Train query cap. Default 664 for opd/rl+opd/rl; scape+rl uses every SEC RL query.",
    )
    parser.add_argument("--train-steps", type=int, default=8)
    parser.add_argument("--lambda-opd", type=float, default=None, help="OPD coefficient. Default 0.1; scape+rl and scape+seed use 0.01 (SEED).")
    parser.add_argument(
        "--opd-gate-beta",
        type=float,
        default=SCAPE_RL_OPD_GATE_BETA,
        help="SEED OPD gate β in σ(βΔ). Used by scape+rl and scape+seed.",
    )
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument(
        "--opd-states-per-trajectory",
        type=int,
        default=None,
        help="k decision points per trajectory. Default 3 for opd/rl+opd; -1 (all actions) for scape+rl and scape+seed.",
    )
    parser.add_argument(
        "--eval-max-turns",
        type=int,
        default=HARNESS1_EVAL_MAX_TURNS,
        help="Closed-loop eval horizon. Training rollouts keep --max-turns; Harness-1 eval uses 40.",
    )
    parser.add_argument(
        "--eval-max-new-tokens",
        type=int,
        default=HARNESS1_EVAL_MAX_NEW_TOKENS,
        help="Eval generation tokens per turn. Training keeps --max-new-tokens.",
    )
    parser.add_argument(
        "--eval-temperature",
        type=float,
        default=HARNESS1_EVAL_TEMPERATURE,
        help="Eval sampling temperature. Training rollouts stay greedy unless the collector samples.",
    )
    parser.add_argument(
        "--rl-data",
        type=Path,
        default=None,
        help="scape+rl query pack: tar.gz, extracted dir, or jsonl. Default /data/ppnm/harness-1-rl-data.tar.gz.",
    )
    parser.add_argument(
        "--sec-corpus-root",
        type=Path,
        default=None,
        help="scape+rl SEC retrieval corpus. Default /data/ppnm/harness-1-sec-corpus.",
    )
    parser.add_argument(
        "--query-manifest",
        type=Path,
        default=None,
        help="Optional query JSON/JSONL override. scape+rl still evaluates on BC+ 830.",
    )
    parser.add_argument(
        "--score-split",
        choices=(SCORE_SPLIT_166, SCORE_SPLIT_830),
        default=None,
        help="Eval query pool. Default bcplus_test_166; scape+rl uses bcplus_830.",
    )
    parser.add_argument("--sft-adapter", default="")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def add_eval_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    add_common_args(parser)
    parser.set_defaults(max_model_len=HARNESS1_EVAL_MAX_MODEL_LEN)
    parser.add_argument(
        "--max-turns",
        type=int,
        default=HARNESS1_EVAL_MAX_TURNS,
        help="Max tool turns per query. Harness-1 Table-2 eval uses 40.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=HARNESS1_EVAL_MAX_NEW_TOKENS,
        help="Max generation tokens per turn. Harness-1 eval uses 2048.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=HARNESS1_EVAL_TEMPERATURE,
        help="Sampling temperature. Harness-1 eval uses 1.0; 0 is greedy.",
    )
    parser.add_argument(
        "--search-k",
        type=int,
        default=HARNESS1_EVAL_SEARCH_K,
        help="Live BM25 hits per search call. Harness-1 SEARCH_DISPLAY_LIMIT is 10.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Training --out directory; adapters under seed*/adapters/ are loaded automatically.",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="Explicit LoRA adapter directory. Implies adapter (H_min) eval.",
    )
    parser.add_argument(
        "--adapter-map",
        type=Path,
        default=None,
        help="JSON {cell: adapter_dir}. Overrides --run-dir discovery.",
    )
    parser.add_argument(
        "--eval-mode",
        choices=("auto", "harness", "adapter"),
        default="auto",
        help="auto: adapter if --run-dir/--adapter is given, else harness-on.",
    )
    parser.add_argument(
        "--score-split",
        choices=(SCORE_SPLIT_166, SCORE_SPLIT_830),
        default=None,
        help="Eval query pool. Default bcplus_830 (664+166). Pass bcplus_test_166 for the 166-test subset.",
    )
    parser.add_argument("--audit-only", action="store_true")
    return parser


def _spec_from_ns(args: argparse.Namespace, *, train: bool) -> LaunchSpec:
    harness = _pick(args.harness, _HARNESS_ALIASES, ALLOWED_HARNESSES, "--harness")
    benchmark = _pick(args.benchmark, _BENCHMARK_ALIASES, ALLOWED_BENCHMARKS, "--benchmark")
    model_name = _pick(args.model_name, _MODEL_ALIASES, ALLOWED_MODEL_NAMES, "--model_name")
    components = tuple(canonical_component_ids(args.component))
    method = None
    if train:
        method = _TRAIN_METHOD_ALIASES.get(_norm(args.train_method))
        if method is None:
            raise LaunchError(f"--train_method={args.train_method!r} is not supported")
    base_model = resolve_model_path(model_name, args.base_model or None)
    out = args.out
    if out is None:
        out = default_out_dir(
            kind="train" if train else "eval",
            harness=harness,
            benchmark=benchmark,
            model_name=model_name,
            components=components,
            train_method=method,
        )
    return LaunchSpec(
        harness=harness,
        benchmark=benchmark,
        model_name=model_name,
        components=components,
        train_method=method,
        base_model=base_model,
        out=Path(out),
        extra=dict(vars(args)),
    )


def parse_train_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, LaunchSpec]:
    parser = argparse.ArgumentParser(
        description="One-click Harness-1 / BC+ training (opd | rl+opd | rl | scape+rl | scape+seed).",
    )
    add_train_args(parser)
    args = parser.parse_args(argv)
    spec = _spec_from_ns(args, train=True)
    args.harness = spec.harness
    args.benchmark = spec.benchmark
    args.model_name = spec.model_name
    args.component = spec.coalition
    args.component_ids = list(spec.components)
    args.train_method = spec.train_method
    args.training_mode = spec.training_mode
    if args.opd_states_per_trajectory is None:
        args.opd_states_per_trajectory = (
            -1 if spec.train_method in {"scape+rl", "scape+seed"} else 3
        )
    if spec.train_method == "scape+rl":
        args.opd_loss = OPD_LOSS_SAMPLED_GAP
    elif spec.train_method == "scape+seed":
        args.opd_loss = OPD_LOSS_PROJECTED_GAP
    else:
        args.opd_loss = "sr_opd_ce"
    if args.lambda_opd is None:
        args.lambda_opd = (
            SCAPE_RL_LAMBDA_OPD if spec.train_method in {"scape+rl", "scape+seed"} else 0.1
        )
    if getattr(args, "opd_gate_beta", None) is None:
        args.opd_gate_beta = SCAPE_RL_OPD_GATE_BETA
    if args.n_queries is None and spec.train_method != "scape+rl":
        args.n_queries = 664
    if getattr(args, "score_split", None) is None:
        args.score_split = SCORE_SPLIT_830 if spec.train_method == "scape+rl" else SCORE_SPLIT_166
    if getattr(args, "sec_corpus_root", None) is None:
        args.sec_corpus_root = default_sec_corpus_root()
    if getattr(args, "rl_data", None) is None:
        args.rl_data = default_sec_rl_data()
    args.base_model = str(spec.base_model)
    args.out = spec.out
    return args, spec


def parse_eval_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, LaunchSpec]:
    parser = argparse.ArgumentParser(
        description="One-click Harness-1 / BC+ closed-loop eval (default: full 830).",
    )
    add_eval_args(parser)
    args = parser.parse_args(argv)
    spec = _spec_from_ns(args, train=False)
    args.harness = spec.harness
    args.benchmark = spec.benchmark
    args.model_name = spec.model_name
    args.component = spec.coalition
    args.component_ids = list(spec.components)
    args.base_model = str(spec.base_model)
    args.out = spec.out
    if getattr(args, "score_split", None) is None:
        args.score_split = SCORE_SPLIT_830
    return args, spec
