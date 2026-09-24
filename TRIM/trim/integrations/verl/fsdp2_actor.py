"""TRIM-owned CISPO actor. FSDP2 or official DDP wrap; one optimizer.step per batch."""

from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.distributed as dist

from trim.integrations.verl.batch_adapter import RankBatchPlan, dummy_rl_row, plan_joint_sync_batches
from trim.integrations.verl.joint_objective import cispo_clip_bounds, verl_cispo_clip_config
from trim.training.hf_rl_batch import (
    gather_response_logprobs,
    log_train,
    pack_left_pad_teacher_forced,
)


def _decoder_layer_classes(model: Any) -> list[type]:
    found: list[type] = []
    seen: set[type] = set()
    for mod in model.modules():
        cls = type(mod)
        if cls.__name__.endswith("DecoderLayer") and cls not in seen:
            seen.add(cls)
            found.append(cls)
    return found


def _wrap_fsdp2(model: Any, *, device: torch.device, world_size: int) -> Any:
    if world_size <= 1:
        return model.to(device)
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    except Exception as exc:
        raise RuntimeError("FSDP2 imports failed; refusing silent FSDP1 fallback") from exc
    mesh = init_device_mesh("cuda", (world_size,))
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for cls in _decoder_layer_classes(model):
        for mod in list(model.modules()):
            if type(mod) is cls:
                fully_shard(mod, mesh=mesh, mp_policy=mp)
    fully_shard(model, mesh=mesh, mp_policy=mp)
    return model


def _wrap_ddp(model: Any, *, device: torch.device, world_size: int) -> Any:
    model = model.to(device)
    if world_size <= 1:
        return model
    from torch.nn.parallel import DistributedDataParallel as DDP

    device_ids = [0] if device.type == "cuda" else None
    return DDP(model, device_ids=device_ids, output_device=device_ids[0] if device_ids else None, find_unused_parameters=False)


def _set_fsdp_grad_sync(model: Any, enabled: bool) -> None:
    seen = False
    for mod in model.modules():
        fn = getattr(mod, "set_requires_gradient_sync", None)
        if callable(fn):
            fn(bool(enabled))
            seen = True
    if not seen:
        fn = getattr(model, "set_requires_gradient_sync", None)
        if callable(fn):
            fn(bool(enabled))


def actor_class_name(wrap: str) -> str:
    return "DDPLoraActor" if str(wrap) == "ddp" else "FSDP2CispoActor"


def _row_opd_loss(raw: Any) -> str:
    loss = getattr(raw, "opd_loss", None)
    if loss is None and isinstance(raw, dict):
        loss = raw.get("opd_loss") or raw.get("loss_id")
        meta = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
        loss = loss or (meta or {}).get("loss_id")
    return str(loss or "")


def unpack_gap_row(raw: Any, *, require_projected: bool = True) -> dict[str, Any]:
    """Student/teacher prefixes and a* tokens for projected-gap scoring."""
    from trim.training.tinker_opd_datum import TinkerOPDDatum

    dummy = bool(isinstance(raw, dict) and raw.get("is_dummy"))
    if isinstance(raw, TinkerOPDDatum):
        student = list(raw.prompt_token_ids)
        n_p = len(student)
        resp = list(raw.target_tokens[n_p:]) if len(raw.target_tokens) >= n_p else list(raw.target_tokens)
        teacher = list(raw.teacher_prompt_token_ids or [])
        weights = list(raw.weights[n_p:]) if len(raw.weights) >= n_p else list(raw.weights)
        meta = dict(raw.metadata or {})
        loss = str(raw.opd_loss or meta.get("loss_id") or "")
        lam = float(meta.get("lambda_opd") if meta.get("lambda_opd") is not None else 0.01)
        beta = float(meta.get("gate_beta") if meta.get("gate_beta") is not None else 5.0)
        projector_used = bool(meta.get("projector_used", True))
        sampled_action = bool(meta.get("sampled_action", False))
    elif isinstance(raw, dict):
        student = list(raw.get("prompt_ids") or raw.get("prompt_token_ids") or raw.get("effective_prompt_ids") or [])
        resp = list(raw.get("target_ids") or [])
        if not resp and raw.get("target_tokens"):
            tokens = list(raw["target_tokens"])
            resp = tokens[len(student) :] if len(tokens) >= len(student) else tokens
        teacher = list(raw.get("teacher_prompt_token_ids") or raw.get("teacher_prompt_ids") or [])
        weights = list(raw.get("weights") or [])
        if len(weights) > len(resp) and student:
            weights = weights[-len(resp) :] if resp else weights
        meta = dict(raw.get("metadata") or {})
        loss = _row_opd_loss(raw)
        lam = float(meta.get("lambda_opd") if meta.get("lambda_opd") is not None else raw.get("lambda_opd") or 0.01)
        beta = float(meta.get("gate_beta") if meta.get("gate_beta") is not None else raw.get("gate_beta") or 5.0)
        projector_used = bool(meta.get("projector_used", True))
        sampled_action = bool(meta.get("sampled_action", False))
    else:
        raise TypeError(f"unsupported OPD row type: {type(raw)!r}")
    if dummy:
        if not student:
            student = [1, 2, 3, 4]
        if not teacher:
            teacher = list(student)
        if not resp:
            resp = [1]
        if not weights:
            weights = [0.0] * len(resp)
        if not loss:
            from trim.training.rl_opd_types import OPD_LOSS_PROJECTED_GAP

            loss = OPD_LOSS_PROJECTED_GAP
    else:
        if not loss:
            raise ValueError("projected-gap datum missing loss_id; refuse CE fallback")
        if not student or not resp:
            raise ValueError("projected-gap datum missing student prompt or a* tokens")
        if not teacher:
            raise ValueError("projected-gap datum missing teacher_prompt_token_ids")
        if len(weights) != len(resp):
            raise ValueError(
                f"projected-gap weight/target length mismatch: weights={len(weights)} target={len(resp)}"
            )
        if require_projected and (sampled_action or not projector_used):
            raise ValueError(
                "distributed gap actor requires projector_used=True and sampled_action=False "
                f"(got projector_used={projector_used} sampled_action={sampled_action})"
            )
    return {
        "student_ids": [int(x) for x in student],
        "resp_ids": [int(x) for x in resp],
        "teacher_ids": [int(x) for x in teacher],
        "weights": [float(w) for w in weights],
        "dummy": dummy,
        "lambda_opd": lam,
        "gate_beta": beta,
        "opd_loss": loss,
        "projector_used": projector_used,
        "sampled_action": sampled_action,
    }


def detect_opd_objective(rows: Sequence[Any] | None) -> str:
    from trim.training.rl_opd_types import uses_seed_gap

    saw_gap = False
    saw_ce = False
    for raw in list(rows or []):
        loss = _row_opd_loss(raw)
        if not loss:
            continue
        if uses_seed_gap(loss):
            saw_gap = True
        else:
            saw_ce = True
    if saw_gap and saw_ce:
        raise ValueError("mixed CE and gap OPD rows in one optimizer update")
    if saw_gap:
        return "gap"
    return "ce"


class FSDP2CispoActor:
    """LoRA actor owned by TRIM. Wrap is official DDP or FSDP2; not native verl."""

    def __init__(
        self,
        *,
        model_path: str,
        adapter_dir: str | None = None,
        learning_rate: float = 1e-5,
        micro_batch_size: int = 4,
        max_full_tokens: int = 8192,
        lora_r: int = 8,
        lora_alpha: int = 16,
        optimizer_path: str | None = None,
        wrap: str = "fsdp2",
        heartbeat_every: int = 8,
        heartbeat_s: float = 30.0,
    ) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self.micro_batch_size = max(1, int(micro_batch_size))
        self.max_full_tokens = int(max_full_tokens)
        clip = verl_cispo_clip_config()
        self.clip_low, self.clip_high = cispo_clip_bounds(**clip)
        self.model_path = str(model_path)
        self.adapter_dir = adapter_dir
        self.learning_rate = float(learning_rate)
        self.lora_r = int(lora_r)
        self.lora_alpha = int(lora_alpha)
        self.optimizer_path = optimizer_path
        self.wrap = str(wrap or "fsdp2").lower().replace("-", "_")
        if self.wrap in {"torch_ddp_lora", "ddp_lora"}:
            self.wrap = "ddp"
        self.heartbeat_every = max(1, int(heartbeat_every))
        self.heartbeat_s = max(1.0, float(heartbeat_s))
        self.model = None
        self.optimizer = None
        self.tokenizer = None
        self.wrapper_type = "none"
        self._logged_logits_fallback = False
        self._logged_logits_shape = False
        self.phase_seconds = {
            "student_forward_s": 0.0,
            "loss_s": 0.0,
            "backward_s": 0.0,
            "grad_sync_s": 0.0,
        }
        self.runtime_info: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok_src = self.model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        load_kw: dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
        }
        if self.world_size <= 1 and self.device.type == "cuda":
            load_kw["device_map"] = {"": str(self.device)}
            load_kw.pop("low_cpu_mem_usage", None)
        model = AutoModelForCausalLM.from_pretrained(self.model_path, **load_kw)
        if hasattr(model, "config"):
            model.config.use_cache = False
        if hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
        adapter = Path(self.adapter_dir or "")
        if adapter.is_dir() and (adapter / "adapter_config.json").is_file():
            model = PeftModel.from_pretrained(model, str(adapter), is_trainable=True)
        else:
            leaf = {n.split(".")[-1] for n, _ in model.named_modules()}
            targets = [m for m in ("q_proj", "k_proj", "v_proj", "o_proj") if m in leaf] or ["q_proj"]
            model = get_peft_model(
                model,
                LoraConfig(
                    r=self.lora_r,
                    lora_alpha=self.lora_alpha,
                    target_modules=targets,
                    lora_dropout=0.0,
                    bias="none",
                    task_type="CAUSAL_LM",
                ),
            )
        if self.rank == 0:
            try:
                model.print_trainable_parameters()
            except Exception:
                n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"[verl-fsdp2] trainable_params={n_train}", flush=True)
        if self.device.type == "cuda":
            model = model.to(self.device)
        if self.wrap == "ddp":
            self.model = _wrap_ddp(model, device=self.device, world_size=self.world_size)
        elif self.wrap == "none":
            self.model = model.to(self.device)
        else:
            self.model = _wrap_fsdp2(model, device=self.device, world_size=self.world_size)
        self.wrapper_type = type(self.model).__name__
        if self.rank == 0:
            print(
                f"[actor] wrap={self.wrap} wrapper={self.wrapper_type} "
                f"world_size={self.world_size} device={self.device} native_verl=false",
                flush=True,
            )
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        )
        if self.optimizer_path:
            path = Path(self.optimizer_path)
            if not path.is_file():
                raise FileNotFoundError(f"optimizer state missing: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            self.optimizer.load_state_dict(payload)
            del payload
        self.runtime_info = self._collect_runtime_info()
        if self.rank == 0:
            import json

            brief = {
                "event": "actor_runtime",
                "wrap": self.wrap,
                "wrapper": self.wrapper_type,
                "requested_dtype": self.runtime_info.get("requested_dtype"),
                "expert_classes": self.runtime_info.get("expert_classes"),
                "attention_classes": self.runtime_info.get("attention_classes"),
                "attention_implementation": self.runtime_info.get("attention_implementation"),
                "quantizer_class": self.runtime_info.get("quantizer_class"),
                "quantization_fallback": self.runtime_info.get("quantization_fallback"),
                "parameter_dtypes": self.runtime_info.get("effective_parameter_dtypes"),
                "trainable_params": self.runtime_info.get("trainable_params"),
                "allocated_mb": self.runtime_info.get("allocated_mb"),
                "reserved_mb": self.runtime_info.get("reserved_mb"),
                "versions": self.runtime_info.get("versions"),
            }
            print(json.dumps(brief, default=str), flush=True)

    def _collect_runtime_info(self) -> dict[str, Any]:
        try:
            from trim.training.runtime_manifest import collect_actor_runtime

            return collect_actor_runtime(self, requested_dtype="bfloat16")
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}", "requested_dtype": "bfloat16"}

    def close(self) -> None:
        self.optimizer = None
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _forward_response_logprobs(
        self,
        pairs: Sequence[tuple[list[int], list[int]]],
        extra: Sequence[dict[str, Any]],
        *,
        time_bucket: str | None = "student_forward_s",
    ) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
        pad_id = int(self.tokenizer.pad_token_id or 0)
        packed = pack_left_pad_teacher_forced(
            pairs, pad_id=pad_id, device=self.device, max_full=self.max_full_tokens
        )
        kwargs: dict[str, Any] = {
            "input_ids": packed.input_ids,
            "attention_mask": packed.attention_mask,
            "use_cache": False,
        }
        keep = int(packed.max_resp) + 1
        t_fwd = time.perf_counter()
        try:
            out = self.model(**kwargs, logits_to_keep=keep)
        except TypeError:
            if not self._logged_logits_fallback:
                print(
                    f"[actor rank{self.rank}] logits_to_keep unsupported; "
                    "falling back to full-sequence logits",
                    flush=True,
                )
                self._logged_logits_fallback = True
            out = self.model(**kwargs)
        if time_bucket:
            self.phase_seconds[time_bucket] = float(self.phase_seconds.get(time_bucket) or 0.0) + (
                time.perf_counter() - t_fwd
            )
        logits = out.logits
        if not self._logged_logits_shape:
            print(
                f"[actor rank{self.rank}] logits_shape={tuple(logits.shape)} "
                f"logits_to_keep={keep} fallback={self._logged_logits_fallback}",
                flush=True,
            )
            self._logged_logits_shape = True
        if logits.shape[1] > keep:
            logits = logits[:, -keep:, :]
        t_lp = time.perf_counter()
        logps = gather_response_logprobs(logits, packed.response_ids, max_resp=packed.max_resp)
        if time_bucket:
            self.phase_seconds[time_bucket] = float(self.phase_seconds.get(time_bucket) or 0.0) + (
                time.perf_counter() - t_lp
            )
        aligned = []
        for row, resp in zip(extra, packed.response_ids):
            item = dict(row)
            n = len(resp)
            item["action_ids"] = list(resp)
            raw_lp = list(item.get("token_logprobs") or [])
            if raw_lp:
                item["token_logprobs"] = raw_lp[:n]
                if len(item["token_logprobs"]) != n:
                    raise ValueError("behavior logprob length changed during pack")
            else:
                item["token_logprobs"] = []
            item["action_mask"] = list(item.get("action_mask") or [1] * n)[:n]
            aligned.append(item)
        return logps, aligned

    def _sync_context(self, sync_now: bool):
        if sync_now or self.world_size <= 1:
            if self.wrap == "fsdp2":
                _set_fsdp_grad_sync(self.model, True)
            return nullcontext()
        if self.wrap == "ddp":
            no_sync = getattr(self.model, "no_sync", None)
            if callable(no_sync):
                return no_sync()
        if self.wrap == "fsdp2":
            _set_fsdp_grad_sync(self.model, False)
        return nullcontext()

    def _peak_mem_mb(self) -> float | None:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return None
        return round(float(torch.cuda.max_memory_allocated()) / (1024 * 1024), 1)

    def _heartbeat(
        self,
        *,
        done: int,
        total: int,
        stage: str,
        input_tokens: int,
        supervised_tokens: int,
        last_len: int,
        started: float,
    ) -> None:
        elapsed = max(1e-6, time.perf_counter() - started)
        payload = {
            "rank": self.rank,
            "wrap": self.wrap,
            "wrapper": self.wrapper_type,
            "done": done,
            "total": total,
            "stage": stage,
            "input_tokens": input_tokens,
            "supervised_tokens": supervised_tokens,
            "tokens_per_s": round(input_tokens / elapsed, 1),
            "last_batch_len": last_len,
            "peak_mem_mb": self._peak_mem_mb(),
            "elapsed_s": round(elapsed, 2),
        }
        log_train("actor_heartbeat", **payload)

    def _rl_numerator(
        self,
        chunk: Sequence[dict[str, Any]],
    ) -> tuple[torch.Tensor, int, int, int]:
        pairs = []
        extras = []
        for row in chunk:
            prompt = list(row.get("effective_prompt_ids") or row.get("prompt_ids") or dummy_rl_row()["prompt_ids"])
            action = list(row.get("action_ids") or [1])
            if len(prompt) + len(action) > self.max_full_tokens:
                raise ValueError(f"sampled sequence {len(prompt)+len(action)} exceeds {self.max_full_tokens}")
            pairs.append((prompt, action))
            extras.append(dict(row))
        logps, aligned = self._forward_response_logprobs(pairs, extras)
        terms: list[torch.Tensor] = []
        ratio_hits = 0
        ratio_total = 0
        input_tokens = sum(len(p) + len(a) for p, a in pairs)
        for row, new_lp in zip(aligned, logps):
            if new_lp.numel() == 0:
                continue
            new_lp = new_lp.float()
            old_raw = list(row.get("token_logprobs") or [])
            if not old_raw:
                continue
            old_lp = torch.tensor(old_raw[: new_lp.numel()], device=self.device, dtype=torch.float32)
            mask = torch.tensor(
                list(row.get("action_mask") or [1] * new_lp.numel())[: new_lp.numel()],
                device=self.device,
                dtype=torch.float32,
            )
            if new_lp.numel() != old_lp.numel():
                n = min(new_lp.numel(), old_lp.numel(), mask.numel())
                new_lp = new_lp[:n]
                old_lp = old_lp[:n]
                mask = mask[:n]
            if not torch.isfinite(new_lp).all() or not torch.isfinite(old_lp).all():
                raise ValueError("non-finite logprobs in CISPO")
            ratio = (new_lp - old_lp).exp()
            if not torch.isfinite(ratio).all():
                raise ValueError("non-finite CISPO ratio")
            weight = ratio.clamp(min=self.clip_low, max=self.clip_high).detach()
            ratio_hits += int((ratio != weight).sum().item())
            ratio_total += int(mask.sum().item())
            adv = float(row.get("advantage") or 0.0)
            terms.append((-(weight * adv * new_lp * mask)).sum())
        if terms:
            numer = torch.stack(terms).sum()
        else:
            dummy = next(p for p in self.model.parameters() if p.requires_grad)
            numer = dummy.float().sum() * 0.0
        return numer, ratio_hits, ratio_total, input_tokens

    def _teacher_forward_context(self):
        """Disable gradient reduction during teacher scoring. FSDP2 still all-gathers."""
        if self.wrap == "ddp":
            no_sync = getattr(self.model, "no_sync", None)
            if callable(no_sync):
                return no_sync()
        if self.wrap == "fsdp2":
            _set_fsdp_grad_sync(self.model, False)
        return nullcontext()

    def _score_teacher_chunk(self, chunk: Sequence[Any]) -> list[torch.Tensor]:
        pairs = []
        unpacked_rows = []
        for raw in chunk:
            row = unpack_gap_row(raw, require_projected=not bool(isinstance(raw, dict) and raw.get("is_dummy")))
            if len(row["teacher_ids"]) + len(row["resp_ids"]) > self.max_full_tokens:
                raise ValueError(
                    f"teacher sequence {len(row['teacher_ids'])+len(row['resp_ids'])} exceeds {self.max_full_tokens}"
                )
            pairs.append((row["teacher_ids"], row["resp_ids"]))
            unpacked_rows.append(row)
        with torch.no_grad():
            logps, _ = self._forward_response_logprobs(pairs, [{} for _ in pairs], time_bucket=None)
        out: list[torch.Tensor] = []
        for row, logp in zip(unpacked_rows, logps):
            lp = logp.detach().float().reshape(-1)
            if not row["dummy"] and lp.numel() != len(row["resp_ids"]):
                raise ValueError(
                    f"teacher a* logprob length mismatch: logp={lp.numel()} target={len(row['resp_ids'])}"
                )
            out.append(lp.cpu())
        return out

    def _projected_gap_numerator(
        self,
        chunk: Sequence[Any],
        teacher_lps: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, int, dict[str, float]]:
        from trim.training.sr_opd_loss import gated_action_gap_weighted_sum

        prepared = []
        for raw in chunk:
            dummy = bool(isinstance(raw, dict) and raw.get("is_dummy"))
            prepared.append(unpack_gap_row(raw, require_projected=not dummy))
        if len(prepared) != len(teacher_lps):
            raise ValueError("teacher logprob chunks drifted from student gap chunks")
        pairs = [(row["student_ids"], row["resp_ids"]) for row in prepared]
        for row in prepared:
            if len(row["student_ids"]) + len(row["resp_ids"]) > self.max_full_tokens:
                raise ValueError(
                    f"student sequence {len(row['student_ids'])+len(row['resp_ids'])} exceeds {self.max_full_tokens}"
                )
        logps, _ = self._forward_response_logprobs(pairs, [{} for _ in pairs])
        terms: list[torch.Tensor] = []
        input_tokens = 0
        gate_sum = 0.0
        n_tok = 0
        student_lp_sum = 0.0
        teacher_lp_sum = 0.0
        for row, student_lp, teacher_lp in zip(prepared, logps, teacher_lps):
            input_tokens += len(row["student_ids"]) + len(row["resp_ids"])
            s = student_lp.float().reshape(-1)
            t = teacher_lp.to(device=s.device, dtype=torch.float32).reshape(-1).detach()
            if row["dummy"]:
                terms.append(s.sum() * 0.0)
                continue
            w = torch.tensor(row["weights"], device=s.device, dtype=torch.float32)
            if s.numel() != t.numel() or s.numel() != w.numel():
                raise ValueError(
                    f"projected-gap student/teacher/weight length mismatch: "
                    f"student={s.numel()} teacher={t.numel()} weights={w.numel()}"
                )
            numer = gated_action_gap_weighted_sum(s, t, w, gate_beta=row["gate_beta"])
            terms.append(numer)
            with torch.no_grad():
                delta = (t - s).detach()
                gate = torch.sigmoid(delta * float(row["gate_beta"]))
                n_keep = int((w != 0).sum().item())
                gate_sum += float((gate * (w != 0).float()).sum().item())
                n_tok += n_keep
                student_lp_sum += float((s * (w != 0).float()).sum().item())
                teacher_lp_sum += float((t * (w != 0).float()).sum().item())
        if terms:
            numer = torch.stack(terms).sum()
        else:
            dummy_p = next(p for p in self.model.parameters() if p.requires_grad)
            numer = dummy_p.float().sum() * 0.0
        stats = {
            "gate_mean": (gate_sum / n_tok) if n_tok else 0.0,
            "student_logp": (student_lp_sum / n_tok) if n_tok else 0.0,
            "teacher_logp": (teacher_lp_sum / n_tok) if n_tok else 0.0,
            "n_weighted_tokens": float(n_tok),
        }
        return numer, input_tokens, stats

    def _opd_numerator(self, chunk: Sequence[Any]) -> tuple[torch.Tensor, int]:
        from trim.training.tinker_opd_datum import TinkerOPDDatum

        prepared: list[tuple[list[int], list[int], list[float], bool]] = []
        for raw in chunk:
            dummy = bool(isinstance(raw, dict) and raw.get("is_dummy"))
            loss = _row_opd_loss(raw)
            if loss:
                from trim.training.rl_opd_types import uses_seed_gap

                if uses_seed_gap(loss) and not dummy:
                    raise ValueError(
                        f"CE actor path refused gap datum opd_loss={loss!r}"
                    )
            if isinstance(raw, TinkerOPDDatum):
                prompt = list(raw.prompt_token_ids)
                n_p = len(prompt)
                resp = list(raw.target_tokens[n_p:])
                weights = list(raw.weights[n_p:])
            else:
                prompt = list(raw.get("prompt_ids") or raw.get("effective_prompt_ids") or [1, 2, 3, 4])
                resp = list(raw.get("target_ids") or [1])
                weights = list(raw.get("weights") or [0.0] * len(resp))
            if prompt and resp:
                prepared.append((prompt, resp, weights, dummy))
        if not prepared:
            dummy_p = next(p for p in self.model.parameters() if p.requires_grad)
            return dummy_p.float().sum() * 0.0, 0
        logps, _ = self._forward_response_logprobs(
            [(p, r) for p, r, _w, _d in prepared],
            [{} for _ in prepared],
        )
        terms: list[torch.Tensor] = []
        input_tokens = 0
        for (prompt, resp, weights, dummy), logp in zip(prepared, logps):
            input_tokens += len(prompt) + len(resp)
            if dummy:
                terms.append(logp.float().sum() * 0.0)
                continue
            from trim.training.opd_train_contract import weights_look_like_unnormalized_gap_mask

            if weights_look_like_unnormalized_gap_mask(list(weights)):
                raise ValueError(
                    "distributed CE actor refused 0/1 gap-style token weights; "
                    "CE weights must already include lambda / Z"
                )
            w = torch.tensor(weights[: logp.numel()], device=self.device, dtype=torch.float32)
            terms.append(-(logp.float() * w).sum())
        if terms:
            return torch.stack(terms).sum(), input_tokens
        dummy_p = next(p for p in self.model.parameters() if p.requires_grad)
        return dummy_p.float().sum() * 0.0, input_tokens

    def update(
        self,
        rl_rows: Sequence[dict[str, Any]],
        opd_rows: Sequence[Any] | None = None,
        *,
        lambda_opd: float = 0.0,
        plan: RankBatchPlan | None = None,
    ) -> dict[str, Any]:
        """Accumulate CISPO (+ CE or projected-gap) then one optimizer.step. All ranks must enter."""
        from trim.training.rl_opd_types import PROJECTED_GAP_OBJECTIVE_VERSION, uses_projected_seed, uses_seed_gap
        from trim.training.sr_opd_loss import scale_projected_gap_loss

        scheduled = plan or plan_joint_sync_batches(
            list(rl_rows),
            list(opd_rows or []),
            rank=self.rank,
            world_size=self.world_size,
            micro_batch_size=self.micro_batch_size,
        )
        inspect_rows: list[Any] = list(opd_rows or [])
        for chunk in scheduled.opd_chunks:
            inspect_rows.extend(chunk)
        objective = detect_opd_objective(inspect_rows)
        gap_mode = objective == "gap"
        if gap_mode:
            for raw in inspect_rows:
                if isinstance(raw, dict) and raw.get("is_dummy"):
                    continue
                loss = _row_opd_loss(raw)
                if loss and uses_seed_gap(loss) and not uses_projected_seed(loss):
                    raise ValueError(
                        f"distributed actor implements projected-gap only; refused {loss!r}"
                    )
        assert self.model is not None and self.optimizer is not None
        self.phase_seconds = {
            "student_forward_s": 0.0,
            "loss_s": 0.0,
            "backward_s": 0.0,
            "grad_sync_s": 0.0,
        }
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        steps = scheduled.steps()
        scale = float(self.world_size)
        z_rl = max(1.0, float(scheduled.global_rl_tokens))
        z_gap = float(scheduled.global_opd_weight) if gap_mode else 0.0
        metrics = {
            "n_rl_rows_local": sum(1 for chunk in scheduled.rl_chunks for r in chunk if not r.get("is_dummy")),
            "n_opd_rows_local": sum(len(chunk) for chunk in scheduled.opd_chunks),
            "n_optimizer_steps": 0,
            "loss": 0.0,
            "n_rl_tokens": int(scheduled.global_rl_tokens),
            "n_dummy_rl": int(scheduled.n_dummy_rl),
            "n_dummy_opd": int(scheduled.n_dummy_opd),
            "n_sync_rounds": int(scheduled.n_sync_rounds),
            "n_global_rl_microbatches": int(scheduled.n_global_rl_microbatches),
            "rl_padding_fraction": float(scheduled.rl_padding_fraction),
            "opd_padding_fraction": float(scheduled.opd_padding_fraction),
            "rl_real_tokens": int(scheduled.rl_real_tokens),
            "opd_real_tokens": int(scheduled.opd_real_tokens),
            "wrap": self.wrap,
            "wrapper": self.wrapper_type,
            "opd_objective": "projected_gap" if gap_mode else "ce",
            "objective_version": PROJECTED_GAP_OBJECTIVE_VERSION if gap_mode else "ce_lambda_baked_v1",
            "Z_gap": z_gap if gap_mode else None,
            "lambda_projected_gap": float(lambda_opd) if gap_mode else None,
        }
        if not steps:
            return metrics

        teacher_logps_by_chunk: list[list[torch.Tensor]] = []
        teacher_score_s = 0.0
        if gap_mode and scheduled.opd_chunks:
            t_teacher = time.perf_counter()
            ctx = self._teacher_forward_context()
            with ctx:
                for chunk in scheduled.opd_chunks:
                    teacher_logps_by_chunk.append(self._score_teacher_chunk(chunk))
            if self.wrap == "fsdp2":
                _set_fsdp_grad_sync(self.model, True)
            teacher_score_s = time.perf_counter() - t_teacher

        loss_sum = 0.0
        ratio_hits = 0
        ratio_total = 0
        input_tokens = 0
        supervised = 0
        gap_raw = 0.0
        gate_mean_acc = 0.0
        gate_n = 0
        started = time.perf_counter()
        last_hb = started
        last_len = 0
        opd_i = 0
        skipped_gap = gap_mode and z_gap <= 0.0
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        for i, (kind, chunk) in enumerate(steps):
            sync_now = i == len(steps) - 1
            last_len = sum(
                len(r.get("effective_prompt_ids") or r.get("prompt_ids") or [])
                + len(r.get("action_ids") or r.get("target_ids") or [])
                if isinstance(r, dict)
                else 0
                for r in chunk
            )
            ctx = self._sync_context(sync_now)
            with ctx:
                fwd_before = float(self.phase_seconds.get("student_forward_s") or 0.0)
                t_block = time.perf_counter()
                if kind == "rl":
                    numer, hits, tot, n_in = self._rl_numerator(chunk)
                    ratio_hits += hits
                    ratio_total += tot
                    supervised += tot
                    loss = scale * (numer / z_rl)
                elif gap_mode:
                    t_lps = teacher_logps_by_chunk[opd_i]
                    numer, n_in, gap_stats = self._projected_gap_numerator(chunk, t_lps)
                    if skipped_gap:
                        loss = numer * 0.0
                    else:
                        loss = scale_projected_gap_loss(
                            numer,
                            lambda_opd=float(lambda_opd),
                            z_gap=z_gap,
                            world_size=self.world_size,
                        )
                    gap_raw += float(numer.detach().item())
                    if gap_stats["n_weighted_tokens"]:
                        gate_mean_acc += gap_stats["gate_mean"] * gap_stats["n_weighted_tokens"]
                        gate_n += gap_stats["n_weighted_tokens"]
                    opd_i += 1
                else:
                    # Weights already include lambda; keep the original sum, only undo DDP mean.
                    numer, n_in = self._opd_numerator(chunk)
                    loss = scale * numer
                block_s = time.perf_counter() - t_block
                fwd_dt = float(self.phase_seconds.get("student_forward_s") or 0.0) - fwd_before
                self.phase_seconds["loss_s"] = float(self.phase_seconds.get("loss_s") or 0.0) + max(0.0, block_s - fwd_dt)
                t_backward = time.perf_counter()
                if loss.requires_grad:
                    loss.backward()
                backward_dt = time.perf_counter() - t_backward
                if sync_now and self.world_size > 1:
                    self.phase_seconds["grad_sync_s"] = float(self.phase_seconds.get("grad_sync_s") or 0.0) + backward_dt
                else:
                    self.phase_seconds["backward_s"] = float(self.phase_seconds.get("backward_s") or 0.0) + backward_dt
                loss_sum += float(loss.detach().item())
                del loss, numer
            input_tokens += n_in
            now = time.perf_counter()
            if (
                (i + 1) % self.heartbeat_every == 0
                or sync_now
                or (now - last_hb) >= self.heartbeat_s
            ):
                self._heartbeat(
                    done=i + 1,
                    total=len(steps),
                    stage=kind,
                    input_tokens=input_tokens,
                    supervised_tokens=supervised,
                    last_len=last_len,
                    started=started,
                )
                last_hb = now
        params = [p for p in self.model.parameters() if p.requires_grad]
        clip_fn = getattr(self.model, "clip_grad_norm_", None)
        if callable(clip_fn):
            clip_fn(1.0)
        else:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t_opt = time.perf_counter()
        has_rl = float(scheduled.global_rl_tokens) > 0
        has_gap = gap_mode and z_gap > 0
        has_ce = (not gap_mode) and float(scheduled.global_opd_weight) > 0
        has_signal = has_rl or has_gap or has_ce
        n_opt = 0
        if has_signal:
            self.optimizer.step()
            n_opt = 1
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        opt_s = time.perf_counter() - t_opt
        self.optimizer.zero_grad(set_to_none=True)
        metrics.update(
            {
                "n_optimizer_steps": n_opt,
                "skipped_empty_supervision": not has_signal,
                "skipped_gap_zero_Z": skipped_gap,
                "n_microbatches": len(steps),
                "loss": loss_sum,
                "projected_gap_raw": gap_raw if gap_mode else None,
                "gate_mean": (gate_mean_acc / gate_n) if gate_n else None,
                "teacher_score_s": round(teacher_score_s, 3) if gap_mode else None,
                "student_forward_s": round(float(self.phase_seconds.get("student_forward_s") or 0.0), 3),
                "loss_s": round(float(self.phase_seconds.get("loss_s") or 0.0), 3),
                "backward_s": round(float(self.phase_seconds.get("backward_s") or 0.0), 3),
                "grad_sync_s": round(float(self.phase_seconds.get("grad_sync_s") or 0.0), 3),
                "ratio_clip_fraction": (ratio_hits / max(1, ratio_total)),
                "clip_low": self.clip_low,
                "clip_high": self.clip_high,
                "update_wall_s": round(time.perf_counter() - started, 3),
                "optimizer_s": round(opt_s, 3),
                "peak_mem_mb": self._peak_mem_mb(),
            }
        )
        log_train(f"{self.wrap}_cispo", **metrics)
        if n_opt not in {0, 1}:
            raise RuntimeError(f"n_optimizer_steps={n_opt}")
        return metrics

    def _opd_ce(self, datums: Sequence[Any], *, lambda_already_baked: bool) -> float:
        del lambda_already_baked
        plan = plan_joint_sync_batches(
            [],
            list(datums),
            rank=self.rank,
            world_size=1,
            micro_batch_size=self.micro_batch_size,
        )
        total = 0.0
        for chunk in plan.opd_chunks:
            numer, _ = self._opd_numerator(chunk)
            if numer.requires_grad:
                numer.backward()
            total += float(numer.detach().item())
        return total

    def save_adapter(self, path: Path) -> None:
        path = Path(path)
        if self.wrap == "ddp":
            self._save_ddp_adapter(path)
            return
        if self.rank == 0:
            path.mkdir(parents=True, exist_ok=True)
        if self.world_size > 1:
            dist.barrier()
        core = self.model
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            if isinstance(core, FSDP):
                from trim.training.hf_sft import _save_peft_checkpoint

                _save_peft_checkpoint(core, self.tokenizer, path, merge=False)
                if self.world_size > 1:
                    dist.barrier()
                return
        except Exception:
            pass
        state = None
        try:
            from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

            state = get_model_state_dict(
                self.model, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
            )
        except Exception:
            state = None
        while hasattr(core, "module"):
            core = core.module
        if self.rank == 0:
            saver = getattr(core, "save_pretrained", None)
            if saver is None:
                raise RuntimeError("cannot save LoRA adapter from wrapped model")
            if state is not None:
                lora_state = {
                    k: v for k, v in state.items()
                    if "lora_" in k or "modules_to_save" in k
                }
                saver(str(path), state_dict=lora_state or None)
            else:
                saver(str(path))
            self.tokenizer.save_pretrained(str(path))
        if self.world_size > 1:
            dist.barrier()

    def _save_ddp_adapter(self, path: Path) -> None:
        """Rank 0 writes the PEFT adapter only. Other ranks wait and share failures.

        This path assumes q/k/v/o LoRA with ``bias=none`` and frozen embeddings.
        It is not a correct FSDP2 shard gather.
        """
        err: str | None = None
        if self.world_size > 1:
            dist.barrier()
        if self.rank == 0:
            try:
                path.mkdir(parents=True, exist_ok=True)
                core = self.model
                seen: set[int] = set()
                while hasattr(core, "module") and id(core) not in seen:
                    seen.add(id(core))
                    nxt = core.module
                    if nxt is None or nxt is core:
                        break
                    core = nxt
                if not hasattr(core, "peft_config"):
                    raise RuntimeError("DDP adapter save expected a PEFT model after unwrapping")
                try:
                    core.save_pretrained(
                        str(path),
                        safe_serialization=True,
                        save_embedding_layers=False,
                    )
                except TypeError:
                    core.save_pretrained(str(path), safe_serialization=True)
                if self.tokenizer is None:
                    raise RuntimeError("tokenizer missing during DDP adapter save")
                self.tokenizer.save_pretrained(str(path))
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
        if self.world_size > 1:
            from trim.training.dist_runtime import broadcast_object

            err = broadcast_object(err)
            dist.barrier()
        if err:
            raise RuntimeError(f"DDP LoRA adapter save failed: {err}")


class DDPLoraActor(FSDP2CispoActor):
    """Official PyTorch DDP LoRA actor. Same CISPO math as FSDP2CispoActor."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs["wrap"] = "ddp"
        super().__init__(**kwargs)
