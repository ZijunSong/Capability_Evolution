"""Qwen3.5-9B/27B runtime hooks for Probe C (collect/cache/screen wiring).

Everything that touches the model or the environment is injected here:

- ``verify_model_loader``: Qwen3_5ForConditionalGeneration + transformers>=5.6,
  no silent fallback to another Qwen.
- ``collect_states``: reads the strict rows produced by ``probe_c_collect``.
- ``teacher_logp_fn``: frozen θ0 (base, **no adapter**) Teacher scoring. The
  Student starts from base + a zero-init LoRA (peft lora_B=0), so θ0 is shared.
- ``student_backend_factory`` / ``student_logp_fn``: LoRA Student implementing
  the ``probe_c_train.StudentBackend`` protocol (gated projected-gap only).
- ``make_candidate_set``: fixed K(s) with native completion ids.
- ``make_teacher_prompt_fn``: the H_m branch-view Teacher prompt (explicit
  pair mask; never the default Full view).
- ``regenerate_fn``: fresh greedy generation + real parser/legality checks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from trim.eval.probe_c_metrics import (
    FAIL_EMPTY,
    FAIL_INACCESSIBLE_REF,
    FAIL_PARSE,
    FAIL_SCHEMA,
    FAIL_TRUNCATION,
    GenerationRecord,
)
from trim.probes.probe_c_contract import HarnessPairSpec, ProbeCConfig
from trim.probes.probe_c_data import ProbeCState
from trim.probes.probe_c_runner import RuntimeHooks
from trim.probes.probe_c_train import (
    frozen_projected_gap_numerator,
    scale_c1_loss,
)

MODEL_ARCH = "Qwen3_5ForConditionalGeneration"
MIN_TRANSFORMERS = (5, 6, 0)

_enc_cache: dict[str, Any] = {}
_teacher_cache: dict[str, Any] = {}
_student_cache: dict[tuple[str, int], Any] = {}


def _model_path(cfg: ProbeCConfig) -> str:
    path = str(cfg.raw.get("model_path") or "").strip()
    if not path:
        raise RuntimeError("config model_path is empty; resolve it before running")
    return path


# ---------------------------------------------------------------------------
# loader verification
# ---------------------------------------------------------------------------


def verify_model_loader(cfg: ProbeCConfig) -> dict[str, Any]:
    model_path = Path(_model_path(cfg))
    model_cfg = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    archs = [str(a) for a in (model_cfg.get("architectures") or [])]
    if MODEL_ARCH not in archs:
        return {
            "supported": False,
            "reason": f"architectures={archs}; require {MODEL_ARCH}",
        }
    try:
        import transformers

        version = tuple(
            int(x) for x in str(transformers.__version__).split(".")[:3] if x.isdigit()
        )
    except Exception as exc:  # pragma: no cover - env dependent
        return {"supported": False, "reason": f"transformers import failed: {exc}"}
    padded = version + (0,) * (3 - len(version))
    if padded < MIN_TRANSFORMERS:
        return {
            "supported": False,
            "reason": f"transformers {transformers.__version__} < 5.6 has no qwen3_5 "
            "registration (KeyError at actor init)",
        }
    return {
        "supported": True,
        "arch": MODEL_ARCH,
        "transformers": str(transformers.__version__),
        "model_path": str(model_path),
    }


# ---------------------------------------------------------------------------
# model handles
# ---------------------------------------------------------------------------


def load_encoder(cfg: ProbeCConfig) -> Any:
    key = _model_path(cfg)
    if key not in _enc_cache:
        from trim.eval.model_tokenizer import load_model_encoding

        _enc_cache[key] = load_model_encoding(key, max_model_len=int(cfg.raw.get("max_model_len") or 8192))
    return _enc_cache[key]


def load_teacher_backend(cfg: ProbeCConfig) -> Any:
    """Frozen θ0 Teacher: base model, NO adapter (θ0 has no initial adapter)."""
    key = _model_path(cfg)
    if key not in _teacher_cache:
        from trim.training.hf_tool_opd import ScapeHFToolOPD

        backend = ScapeHFToolOPD(
            model_path=key,
            base_model_override=key,
            device_map="cuda:0",
            learning_rate=0.0,
            use_lora=False,
        )
        backend.model.eval()
        for p in backend.model.parameters():
            p.requires_grad_(False)
        _teacher_cache[key] = backend
    return _teacher_cache[key]


def load_student_backend(cfg: ProbeCConfig, seed: int) -> Any:
    """Student θ0 = base + zero-init LoRA (peft lora_B=0 ⇒ logits == base)."""
    key = (_model_path(cfg), int(seed))
    if key not in _student_cache:
        import torch

        from trim.training.hf_tool_opd import ScapeHFToolOPD

        torch.manual_seed(int(seed))
        lora_r = int((cfg.raw.get("budget") or {}).get("lora_rank") or 16)
        lr = float((cfg.raw.get("budget") or {}).get("learning_rate") or 2e-5)
        backend = ScapeHFToolOPD(
            model_path=_model_path(cfg),
            base_model_override=_model_path(cfg),
            device_map="cuda:1" if _has_second_gpu() else "cuda:0",
            learning_rate=lr,
            use_lora=True,
            lora_r=lora_r,
            lora_alpha=lora_r * 2,
        )
        _student_cache[key] = ProbeCLoraStudentBackend(backend, grad_clip=float((cfg.raw.get("budget") or {}).get("grad_clip") or 1.0))
    return _student_cache[key]


def _has_second_gpu() -> bool:
    try:
        import torch

        return torch.cuda.device_count() >= 2
    except Exception:
        return False


def teacher_logp_fn(cfg: ProbeCConfig, pair: HarnessPairSpec):
    backend = load_teacher_backend(cfg)

    def _logp(prompt_ids: Sequence[int], target_ids: Sequence[int]) -> list[float]:
        if not target_ids:
            return []
        logp = backend._teacher_forced_logprobs(
            [int(t) for t in prompt_ids], [int(t) for t in target_ids], require_grad=False
        )
        return [float(x) for x in logp.detach().float().cpu().reshape(-1).tolist()]

    return _logp


def student_logp_fn(backend: Any):
    def _logp(prompt_ids: Sequence[int], target_ids: Sequence[int]) -> list[float]:
        return backend.token_logprobs(prompt_ids, target_ids)

    return _logp


class ProbeCLoraStudentBackend:
    """StudentBackend protocol over ScapeHFToolOPD (LoRA, gated projected-gap)."""

    def __init__(self, backend: Any, *, grad_clip: float = 1.0) -> None:
        import torch

        self.torch = torch
        self.backend = backend
        self.grad_clip = float(grad_clip)
        self._last_loss = 0.0

    # -- protocol -------------------------------------------------------
    def token_logprobs(
        self, prompt_ids: Sequence[int], target_ids: Sequence[int]
    ) -> list[float]:
        logp = self.backend._teacher_forced_logprobs(
            [int(t) for t in prompt_ids], [int(t) for t in target_ids], require_grad=False
        )
        return [float(x) for x in logp.detach().float().cpu().reshape(-1).tolist()]

    def train_microbatch(
        self,
        *,
        prompt_ids: Sequence[int],
        target_ids: Sequence[int],
        teacher_logprobs: Sequence[float],
        weights: Sequence[float],
        z_gap: float,
        gate_beta: float,
        loss_scale: float,
    ) -> dict[str, Any]:
        torch = self.torch
        student = self.backend._teacher_forced_logprobs(
            [int(t) for t in prompt_ids], [int(t) for t in target_ids], require_grad=True
        ).reshape(-1)
        teacher = torch.tensor(
            [float(x) for x in teacher_logprobs],
            dtype=student.dtype,
            device=student.device,
        )
        w = torch.tensor(
            [float(x) for x in weights], dtype=student.dtype, device=student.device
        )
        if teacher.numel() != student.numel():
            raise ValueError(
                f"target token length mismatch: student={student.numel()} "
                f"teacher={teacher.numel()}"
            )
        numer = frozen_projected_gap_numerator(
            student, teacher, w, gate_beta=gate_beta
        )
        loss = scale_c1_loss(
            numer, z_gap=z_gap, loss_scale=loss_scale, world_size=1
        )
        loss.backward()
        self._last_loss = float(loss.detach().item())
        return {
            "numerator": float(numer.detach().item()),
            "loss": self._last_loss,
        }

    def optimizer_step(self) -> dict[str, Any]:
        torch = self.torch
        params = [p for p in self.backend.model.parameters() if p.grad is not None]
        grad_norm = None
        if params and self.grad_clip > 0:
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(params, self.grad_clip).item()
            )
        self.backend.optimizer.step()
        self.backend.optimizer.zero_grad(set_to_none=True)
        return {"grad_norm": grad_norm, "loss": self._last_loss}

    def trainable_checksum(self) -> str:
        from trim.probes.probe_c_train import checksum_payload

        return checksum_payload(self.save_trainable())

    def save_trainable(self) -> dict[str, Any]:
        torch = self.torch
        named = {
            name: p.detach().cpu().clone()
            for name, p in self.backend.model.named_parameters()
            if p.requires_grad
        }
        payload: dict[str, Any] = {
            "model": named,
            "optimizer": self.backend.optimizer.state_dict(),
        }
        if torch.cuda.is_available():
            payload["cuda_rng"] = torch.cuda.get_rng_state_all()
        return payload

    def load_trainable(self, payload: Mapping[str, Any]) -> None:
        torch = self.torch
        named = dict(payload.get("model") or {})
        current = {
            name: p for name, p in self.backend.model.named_parameters() if p.requires_grad
        }
        if set(named) != set(current):
            raise RuntimeError(
                f"trainable param mismatch: saved={sorted(named)[:3]}... "
                f"current={sorted(current)[:3]}..."
            )
        with torch.no_grad():
            for name, p in current.items():
                p.copy_(named[name].to(p.device, dtype=p.dtype))
        self.backend.optimizer.load_state_dict(payload["optimizer"])
        if "cuda_rng" in payload and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
        self.backend.optimizer.zero_grad(set_to_none=True)


# ---------------------------------------------------------------------------
# prompts / candidates / regeneration
# ---------------------------------------------------------------------------


def make_teacher_prompt_fn(cfg: ProbeCConfig, pair: HarnessPairSpec):
    """Teacher-side prompt ids under the explicit pair.teacher_mask (H_m view)."""
    from trim.eval.local_search_env import wm_text as local_wm_text
    from trim.training.four_cell_runtime import encode_aligned_teacher_prompt
    from trim.training.opd_prompt_encoding import _snapshot_acts_and_wm

    enc = load_encoder(cfg)
    component_label = ",".join(pair.component_ids)

    def _st_from_snapshot(snap_dict: Mapping[str, Any], mask: Mapping[str, bool]) -> dict[str, Any]:
        wm = dict(snap_dict.get("working_memory") or {})
        docs = {
            str(d.get("id")): {"id": str(d.get("id")), "text": str(d.get("text") or "")}
            for d in (wm.get("documents") or [])
            if isinstance(d, dict) and d.get("id")
        }
        pool = dict(wm.get("pool") or docs)
        curated_ids = [str(x) for x in (wm.get("curated_ids") or [])]
        curated = {did: (pool.get(did) or docs.get(did) or {"id": did}) for did in curated_ids}
        return {
            "query": str(wm.get("query") or snap_dict.get("query_text") or ""),
            "pool": pool,
            "curated": curated,
            "importance": dict(wm.get("curated_importance") or {}),
            "step": int(wm.get("step") or snap_dict.get("step") or 0),
            "search_count": int(wm.get("search_count") or 0),
            "first_search_pending": bool(wm.get("first_search_pending")),
            "auto_seed": wm.get("auto_populate_seed"),
            "tool_history": list(wm.get("tool_history") or snap_dict.get("tool_history") or []),
            "harness_mask": dict(mask),
        }

    def _fn(state: ProbeCState, outcome: Any) -> list[int]:
        from trim.state.snapshot import EnvironmentSnapshot

        snap = EnvironmentSnapshot.from_dict(state.student_snapshot)
        acts, _saved_wm = _snapshot_acts_and_wm(snap)
        teacher_st = _st_from_snapshot(state.student_snapshot, pair.teacher_mask)
        teacher_wm = local_wm_text(teacher_st)
        ids, _ = encode_aligned_teacher_prompt(
            enc,
            state.query_text,
            frozen_st=teacher_st,
            frozen_acts=list(acts),
            component_id=component_label,
            teacher_wm_text=teacher_wm,
            teacher_mask=dict(pair.teacher_mask),
        )
        return list(ids)

    return _fn


def _encode_action(cfg: ProbeCConfig, action: Mapping[str, Any]) -> list[int]:
    from trim.training.action_codec import render_action
    from trim.training.action_encoding import encode_supervised_action

    enc = load_encoder(cfg)
    sup = encode_supervised_action(
        target_action=dict(action),
        target_text=render_action(dict(action)),
        encode=enc.encode,
        model_enc=enc,
    )
    return list(sup.target_ids)


def make_candidate_set(cfg: ProbeCConfig, candidate_limit: int = 8):
    """Fixed K(s): H0 base action + H_m projected action + same-tool variants."""

    from trim.training.tool_mask import legal_tool_names
    from trim.adapters.components import zero_mask

    legal = set(legal_tool_names(harness_mask=zero_mask("Harness-1")))

    def _fn(
        state: ProbeCState,
        pair: HarnessPairSpec,
        cache_rows: Sequence[Any] = (),
    ) -> list[tuple[str, dict[str, Any]]]:
        candidates: list[tuple[str, dict[str, Any]]] = []
        seen: set[str] = set()

        def _add(cid: str, action: Mapping[str, Any]) -> None:
            if str(action.get("name") or "") not in legal:
                return
            canon = json.dumps(
                {"name": action.get("name"), "arguments": action.get("arguments") or {}},
                sort_keys=True,
                ensure_ascii=False,
            )
            if canon in seen:
                return
            used = {c for c, _ in candidates}
            real_cid = cid
            n = 1
            while real_cid in used:
                real_cid = f"{cid}_{n}"
                n += 1
            seen.add(canon)
            try:
                ids = _encode_action(cfg, action)
            except Exception:
                return  # non-encodable candidates are excluded, not faked
            candidates.append(
                (
                    real_cid,
                    {
                        "name": str(action.get("name") or ""),
                        "arguments": dict(action.get("arguments") or {}),
                        "token_ids": ids,
                    },
                )
            )

        parse = dict(state.original_action_parse_result or {})
        if parse.get("action_text"):
            from trim.training.action_codec import parse_action

            try:
                base = parse_action(str(parse["action_text"]))
            except Exception:
                base = None
            if base:
                _add("base", {"name": base.get("name"), "arguments": base.get("arguments") or {}})
        for row in cache_rows or []:
            row_state = getattr(row, "state", None)
            row_sid = getattr(row_state, "state_id", None)
            if row_sid is None and isinstance(row, Mapping):
                row_sid = row.get("state_id")
            if row_sid is not None and row_sid != state.state_id:
                continue  # K(s) is per-state; never mix other states' targets
            target = getattr(row, "target", None)
            if target is None and isinstance(row, Mapping):
                target = row.get("target")
            if isinstance(target, Mapping):
                action = dict(target.get("target_action") or {})
            elif target is not None:
                action = dict(getattr(target, "target_action", {}) or {})
            else:
                action = {}
            if action:
                _add("proj", action)
        # Same-tool legal variants (never gold-derived).
        wm = dict(state.student_snapshot.get("working_memory") or {})
        visible = [str(x) for x in (wm.get("accessible_doc_ids") or [])]
        curated = {str(x) for x in (wm.get("curated_ids") or [])}
        query = state.query_text
        variants: list[dict[str, Any]] = [
            {"name": "search_corpus", "arguments": {"query": query}},
            {"name": "curate", "arguments": {"add_ids": [d for d in visible if d not in curated][:1], "remove_ids": []}},
        ]
        for did in visible[:3]:
            variants.append({"name": "read_document", "arguments": {"doc_id": did}})
        for i, action in enumerate(variants):
            if len(candidates) >= max(2, int(candidate_limit)):
                break
            _add(f"var{i}", action)
        return candidates[: max(2, int(candidate_limit))]

    return _fn


def regenerate_fn(backend: Any, state: ProbeCState, pair: HarnessPairSpec | None, *, checkpoint: str):
    """Fresh greedy generation + real parser/schema/visibility checks (§8.3)."""
    import torch

    from trim.adapters.components import zero_mask
    from trim.training.tool_mask import legal_tool_names

    inner = getattr(backend, "backend", backend)  # unwrap ProbeCLoraStudentBackend
    enc = load_encoder_from_backend(inner)
    legal = set(legal_tool_names(harness_mask=zero_mask("Harness-1")))
    prompt_ids = [int(t) for t in state.effective_student_prompt_ids]
    try:
        inp = torch.tensor([prompt_ids], device=inner._device)
        out = inner.model.generate(
            inp,
            do_sample=False,
            max_new_tokens=128,
            eos_token_id=list(enc.stop_token_ids or []),
            pad_token_id=(enc.stop_token_ids or [0])[0],
        )
        new_ids = out[0, inp.size(1) :].tolist()
    except Exception as exc:
        return GenerationRecord(
            state_id=state.state_id,
            query_id=state.query_id,
            checkpoint=checkpoint,
            ok=False,
            failure_kind="generate_error",
            detail={"error": f"generate_failed:{type(exc).__name__}:{exc}"},
        )
    text = enc.decode_tokens(new_ids)
    if not text.strip():
        return GenerationRecord(
            state_id=state.state_id,
            query_id=state.query_id,
            checkpoint=checkpoint,
            ok=False,
            failure_kind=FAIL_EMPTY,
        )
    try:
        parsed = enc.parse_tool_call(text, completion_ids=list(new_ids))
    except Exception:
        parsed = None
    if parsed is None or not getattr(parsed, "parsed", False) or not getattr(parsed, "tool_name", ""):
        return GenerationRecord(
            state_id=state.state_id,
            query_id=state.query_id,
            checkpoint=checkpoint,
            ok=False,
            failure_kind=FAIL_PARSE,
            detail={"text_head": text[:200]},
        )
    name = str(parsed.tool_name)
    if name not in legal:
        return GenerationRecord(
            state_id=state.state_id,
            query_id=state.query_id,
            checkpoint=checkpoint,
            ok=False,
            failure_kind=FAIL_SCHEMA,
            detail={"tool_name": name},
        )
    # Visibility: doc tools must reference docs visible at this decision.
    wm = dict(state.student_snapshot.get("working_memory") or {})
    visible = {str(x) for x in (wm.get("accessible_doc_ids") or [])}
    args = dict(getattr(parsed, "arguments", None) or {})
    doc_ref = str(args.get("doc_id") or "")
    add_ids = [str(x) for x in (args.get("add_ids") or [])]
    refs = ([doc_ref] if doc_ref else []) + add_ids
    for ref in refs:
        if ref and ref not in visible:
            return GenerationRecord(
                state_id=state.state_id,
                query_id=state.query_id,
                checkpoint=checkpoint,
                ok=False,
                failure_kind=FAIL_INACCESSIBLE_REF,
                detail={"ref": ref},
            )
    return GenerationRecord(
        state_id=state.state_id,
        query_id=state.query_id,
        checkpoint=checkpoint,
        ok=True,
        action={"name": name, "arguments": args},
        decode="greedy",
    )


def load_encoder_from_backend(backend: Any) -> Any:
    """ModelEncoding for a ScapeHFToolOPD handle (tokenizer family aware)."""
    key = str(getattr(backend, "model_path", "") or "")
    if key not in _enc_cache:
        from trim.eval.model_tokenizer import load_model_encoding

        _enc_cache[key] = load_model_encoding(key)
    return _enc_cache[key]


# ---------------------------------------------------------------------------
# collect + assemble
# ---------------------------------------------------------------------------


def collect_states(cfg: ProbeCConfig) -> Sequence[Mapping[str, Any]]:
    """Read strict rows from the shared real-H0 collection (probe_c_collect)."""
    src = Path(str(cfg.raw.get("state_source_dir") or "").strip() or "")
    if not src.is_dir():
        raise RuntimeError(
            f"state_source_dir {src} missing; run probe_c_collect prepare/worker/merge "
            "first (no synthetic fallback)"
        )
    path = src / "states" / "all_states.jsonl"
    if not path.is_file():
        path = src / "all_states.jsonl"
    if not path.is_file():
        raise RuntimeError(f"no merged states under {src}")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return rows


def build_hooks() -> RuntimeHooks:
    def _student_factory(cfg: ProbeCConfig, seed: int) -> Any:
        return load_student_backend(cfg, int(seed))

    def _candidate_factory(cfg: ProbeCConfig):
        return make_candidate_set(cfg, candidate_limit=int(cfg.candidate_limit))

    def _regen(backend: Any, state: ProbeCState, pair: HarnessPairSpec | None, *, checkpoint: str):
        return regenerate_fn(backend, state, pair, checkpoint=checkpoint)

    return RuntimeHooks(
        collect_states=collect_states,
        teacher_logp_fn=teacher_logp_fn,
        student_backend_factory=_student_factory,
        student_logp_fn=student_logp_fn,
        make_candidate_set=_candidate_factory,
        make_teacher_prompt_fn=make_teacher_prompt_fn,
        load_encoder=load_encoder,
        regenerate_fn=_regen,
        verify_model_loader=verify_model_loader,
    )


# ---------------------------------------------------------------------------
# config resolution (entry-time)
# ---------------------------------------------------------------------------


def resolve_config(
    base_path: str | Path,
    *,
    out_path: str | Path,
    model_path: str,
    state_source_dir: str = "",
    output_root: str = "",
    git_sha: str = "",
) -> Path:
    """Freeze model revision / paths / component revisions into a runnable config."""
    import copy

    from trim.probes.probe_c_collect import model_identity
    from trim.probes.probe_c_contract import component_revision_of

    raw = json.loads(Path(base_path).read_text(encoding="utf-8"))
    raw = copy.deepcopy(raw)
    model_path = str(model_path)
    identity = model_identity(Path(model_path))
    raw["model_path"] = model_path
    raw["model_revision"] = identity["model_revision"]
    raw["git_sha"] = str(git_sha or raw.get("git_sha") or "")
    if state_source_dir:
        raw["state_source_dir"] = str(state_source_dir)
    if output_root:
        raw["output_root"] = str(output_root)
    revs: dict[str, str] = {}
    members_map = dict(raw.get("coalition_members") or {})
    for key in list(raw.get("components") or []):
        members = members_map.get(key) or [key]
        revs[key] = component_revision_of(
            members[0], git_sha=raw["git_sha"], file_hashes={"members": ",".join(members)}
        )
    raw["component_revisions"] = revs
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(raw, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="probe_c_qwen35_hooks")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("resolve-config")
    p.add_argument("--base", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--state-source-dir", default="")
    p.add_argument("--output-root", default="")
    p.add_argument("--git-sha", default="")
    args = parser.parse_args(argv)
    path = resolve_config(
        args.base,
        out_path=args.out,
        model_path=args.model_path,
        state_source_dir=args.state_source_dir,
        output_root=args.output_root,
        git_sha=args.git_sha,
    )
    print(f"[probe_c_qwen35_hooks] resolved config -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ProbeCLoraStudentBackend",
    "build_hooks",
    "collect_states",
    "load_encoder",
    "load_student_backend",
    "load_teacher_backend",
    "make_candidate_set",
    "make_teacher_prompt_fn",
    "regenerate_fn",
    "student_logp_fn",
    "teacher_logp_fn",
    "verify_model_loader",
]
