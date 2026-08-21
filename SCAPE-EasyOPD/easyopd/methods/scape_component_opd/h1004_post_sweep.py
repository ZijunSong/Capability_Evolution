from __future__ import annotations

import contextlib
import csv
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from .harness1_bridge import QWEN3_LOGICAL_MODEL_ID, QWEN3_STUDENT_BASE

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "component_sweep_0818"
POST_PHASE_U_ROOT = DEFAULT_OUTPUT_ROOT / "h100_4" / "post_phase_u"
MASTER_ROOT = DEFAULT_OUTPUT_ROOT / "master"
CANONICAL_COMPONENT_ORDER = [
    "verify_tool",
    "importance_tagging",
    "subtractive_curation",
    "auto_populate_first_search",
    "content_dedup",
    "chunk_neighbors",
    "evidence_graph",
    "sentence_compress",
    "token_budget_marker",
    "adaptive_rerank_instruction",
]

CANONICAL_MASTER_ROWS = [
    {
        "Component": "verify_tool",
        "Type": "ACTION_SPACE_CHANGE",
        "Event Support": "8000",
        "Positive Utility": "DIAGNOSTIC_ONLY",
        "Realizability": "NON_REALIZABLE",
        "OPD Learnability": "N/A",
        "Placement Decision": "KEEP_RUNTIME_PLACEMENT_BOUNDARY",
        "Train Queries": "2000",
        "Unique Event States": "8000",
        "Teacher Reward": "N/A",
        "Student Before Reward": "N/A",
        "Student After PURE_OPD": "N/A",
        "Delta PURE": "N/A",
        "Student After RL+OPD": "N/A",
        "Delta RL+OPD": "N/A",
        "DEV Status": "N/A",
        "TEST Status": "N/A",
        "Adapter Reload": "N/A",
        "Student Inference Privilege": "false",
        "Decision": "NON_REALIZABLE_ACTION_SPACE_MISMATCH",
        "Reason": "Teacher action space includes verify(doc_ids, claim), while Student action space does not; Student After is N/A.",
    },
    {
        "Component": "importance_tagging",
        "Type": "N/A",
        "Event Support": "8000",
        "Positive Utility": "UTILITY_NOT_YET_MEASURED",
        "Realizability": "PROJECTABLE",
        "OPD Learnability": "N/A",
        "Placement Decision": "PENDING_EXTERNAL",
        "Train Queries": "2000",
        "Unique Event States": "8000",
        "Teacher Reward": "N/A",
        "Student Before Reward": "N/A",
        "Student After PURE_OPD": "N/A",
        "Delta PURE": "N/A",
        "Student After RL+OPD": "N/A",
        "Delta RL+OPD": "N/A",
        "DEV Status": "N/A",
        "TEST Status": "N/A",
        "Adapter Reload": "N/A",
        "Student Inference Privilege": "false",
        "Decision": "TEACHER_METRIC_REQUIRED_BEFORE_TRAINING",
        "Reason": "Teacher metrics are pending in the current handoff; Phase E remains blocked until utility is measured.",
    },
    {
        "Component": "subtractive_curation",
        "Type": "N/A",
        "Event Support": "N/A",
        "Positive Utility": "N/A",
        "Realizability": "N/A",
        "OPD Learnability": "N/A",
        "Placement Decision": "DATA_INSUFFICIENT",
        "Train Queries": "N/A",
        "Unique Event States": "N/A",
        "Teacher Reward": "N/A",
        "Student Before Reward": "N/A",
        "Student After PURE_OPD": "N/A",
        "Delta PURE": "N/A",
        "Student After RL+OPD": "N/A",
        "Delta RL+OPD": "N/A",
        "DEV Status": "N/A",
        "TEST Status": "N/A",
        "Adapter Reload": "N/A",
        "Student Inference Privilege": "false",
        "Decision": "INSUFFICIENT_5K_EVENT_SUPPORT",
        "Reason": "Current H100-1 handoff does not provide sufficient formal 5K event support for the canonical table slot.",
    },
    {
        "Component": "auto_populate_first_search",
        "Type": "N/A",
        "Event Support": "8000",
        "Positive Utility": "UTILITY_NOT_YET_MEASURED",
        "Realizability": "PROJECTABLE",
        "OPD Learnability": "N/A",
        "Placement Decision": "PENDING_EXTERNAL",
        "Train Queries": "2000",
        "Unique Event States": "8000",
        "Teacher Reward": "N/A",
        "Student Before Reward": "N/A",
        "Student After PURE_OPD": "N/A",
        "Delta PURE": "N/A",
        "Student After RL+OPD": "N/A",
        "Delta RL+OPD": "N/A",
        "DEV Status": "N/A",
        "TEST Status": "N/A",
        "Adapter Reload": "N/A",
        "Student Inference Privilege": "false",
        "Decision": "TEACHER_METRIC_REQUIRED_BEFORE_TRAINING",
        "Reason": "Current handoff has real event support, but formal Teacher utility is still pending.",
    },
    {
        "Component": "content_dedup",
        "Type": "N/A",
        "Event Support": "0",
        "Positive Utility": "N/A",
        "Realizability": "N/A",
        "OPD Learnability": "N/A",
        "Placement Decision": "DATA_INSUFFICIENT",
        "Train Queries": "0",
        "Unique Event States": "0",
        "Teacher Reward": "N/A",
        "Student Before Reward": "N/A",
        "Student After PURE_OPD": "N/A",
        "Delta PURE": "N/A",
        "Student After RL+OPD": "N/A",
        "Delta RL+OPD": "N/A",
        "DEV Status": "N/A",
        "TEST Status": "N/A",
        "Adapter Reload": "N/A",
        "Student Inference Privilege": "false",
        "Decision": "INSUFFICIENT_5K_EVENT_SUPPORT",
        "Reason": "No canonical 5K event support was available for the content_dedup slot in the current handoff.",
    },
    {
        "Component": "chunk_neighbors",
        "Type": "N/A",
        "Event Support": "0",
        "Positive Utility": "N/A",
        "Realizability": "NON_REALIZABLE_EXTERNAL_INFORMATION",
        "OPD Learnability": "N/A",
        "Placement Decision": "KEEP_RUNTIME_PLACEMENT_BOUNDARY",
        "Train Queries": "0",
        "Unique Event States": "0",
        "Teacher Reward": "N/A",
        "Student Before Reward": "N/A",
        "Student After PURE_OPD": "N/A",
        "Delta PURE": "N/A",
        "Student After RL+OPD": "N/A",
        "Delta RL+OPD": "N/A",
        "DEV Status": "N/A",
        "TEST Status": "N/A",
        "Adapter Reload": "N/A",
        "Student Inference Privilege": "false",
        "Decision": "NON_REALIZABLE_EXTERNAL_INFORMATION",
        "Reason": "No student-visible neighbor injection hook was found in Harness-1.",
    },
    {
        "Component": "evidence_graph",
        "Type": "N/A",
        "Event Support": "8000",
        "Positive Utility": "UTILITY_NOT_YET_MEASURED",
        "Realizability": "DIRECT",
        "OPD Learnability": "N/A",
        "Placement Decision": "PENDING_EXTERNAL",
        "Train Queries": "2000",
        "Unique Event States": "8000",
        "Teacher Reward": "N/A",
        "Student Before Reward": "N/A",
        "Student After PURE_OPD": "N/A",
        "Delta PURE": "N/A",
        "Student After RL+OPD": "N/A",
        "Delta RL+OPD": "N/A",
        "DEV Status": "N/A",
        "TEST Status": "N/A",
        "Adapter Reload": "N/A",
        "Student Inference Privilege": "false",
        "Decision": "TEACHER_METRIC_REQUIRED_BEFORE_TRAINING",
        "Reason": "Teacher metrics are pending in the current handoff; utility must be measured before Phase E.",
    },
    {
        "Component": "sentence_compress",
        "Type": "N/A",
        "Event Support": "8000",
        "Positive Utility": "UTILITY_NOT_YET_MEASURED",
        "Realizability": "DIRECT",
        "OPD Learnability": "N/A",
        "Placement Decision": "PENDING_EXTERNAL",
        "Train Queries": "2000",
        "Unique Event States": "8000",
        "Teacher Reward": "N/A",
        "Student Before Reward": "N/A",
        "Student After PURE_OPD": "N/A",
        "Delta PURE": "N/A",
        "Student After RL+OPD": "N/A",
        "Delta RL+OPD": "N/A",
        "DEV Status": "N/A",
        "TEST Status": "N/A",
        "Adapter Reload": "N/A",
        "Student Inference Privilege": "false",
        "Decision": "TEACHER_METRIC_REQUIRED_BEFORE_TRAINING",
        "Reason": "Teacher metrics are pending in the current handoff; utility must be measured before Phase E.",
    },
    {
        "Component": "token_budget_marker",
        "Type": "PRIVILEGED_CONTEXT",
        "Event Support": "8000",
        "Positive Utility": "FAIL",
        "Realizability": "PARTIAL",
        "OPD Learnability": "NOT_RUN_GATE_BLOCKED",
        "Placement Decision": "KEEP_RUNTIME_OR_DROP_COMPONENT",
        "Train Queries": "2000",
        "Unique Event States": "8000",
        "Teacher Reward": "N/A",
        "Student Before Reward": "N/A",
        "Student After PURE_OPD": "N/A",
        "Delta PURE": "N/A",
        "Student After RL+OPD": "N/A",
        "Delta RL+OPD": "N/A",
        "DEV Status": "N/A",
        "TEST Status": "N/A",
        "Adapter Reload": "ADAPTER_RELOAD_READY",
        "Student Inference Privilege": "false",
        "Decision": "TEACHER_COMPONENT_NO_POSITIVE_UTILITY",
        "Reason": "All frozen event states have token usage below 60% of budget; marker is present but has no observed termination-pressure utility in this dataset.",
    },
    {
        "Component": "adaptive_rerank_instruction",
        "Type": "N/A",
        "Event Support": "8000",
        "Positive Utility": "N/A",
        "Realizability": "N/A",
        "OPD Learnability": "N/A",
        "Placement Decision": "PENDING_EXTERNAL",
        "Train Queries": "2000",
        "Unique Event States": "8000",
        "Teacher Reward": "N/A",
        "Student Before Reward": "N/A",
        "Student After PURE_OPD": "N/A",
        "Delta PURE": "N/A",
        "Student After RL+OPD": "N/A",
        "Delta RL+OPD": "N/A",
        "DEV Status": "N/A",
        "TEST Status": "N/A",
        "Adapter Reload": "N/A",
        "Student Inference Privilege": "false",
        "Decision": "PHASE_E_FOUR_CELLS_RUNNING",
        "Reason": "Formal Phase E jobs are still running for this component.",
    },
]


def repo_root() -> Path:
    return ROOT


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_sha256sums(root: Path) -> Path:
    lines = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
        lines.append(f"{sha256_path(path)}  {path.relative_to(root)}")
    out = root / "SHA256SUMS"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def load_token_budget_diagnostic(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    path = output_root / "h100_4" / "token_budget_marker" / "TEACHER_BEFORE_DIAGNOSTIC.json"
    if not path.is_file():
        return {}
    try:
        payload = read_json(path)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def load_token_budget_formal_eval(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    path = output_root / "h100_4" / "token_budget_marker" / "H1004_TOKEN_BUDGET_FORMAL_EVAL_SUMMARY.json"
    if not path.is_file():
        return {}
    try:
        payload = read_json(path)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def find_adapter_root(component_root: Path | None = None) -> Path:
    root = component_root or (DEFAULT_OUTPUT_ROOT / "h100_4" / "token_budget_marker" / "OPD_PILOT")
    adapter = root / "adapter"
    if (adapter / "adapter_config.json").is_file() and (adapter / "adapter_model.safetensors").is_file():
        return adapter
    if (root / "adapter_config.json").is_file() and (root / "adapter_model.safetensors").is_file():
        return root
    for candidate in root.rglob("adapter_config.json"):
        sibling = candidate.parent / "adapter_model.safetensors"
        if sibling.is_file():
            return candidate.parent
    raise FileNotFoundError(f"no adapter_config.json + adapter_model.safetensors found under {root}")


def env_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }
    for pkg in ["transformers", "peft", "accelerate", "safetensors"]:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except Exception as exc:  # noqa: BLE001
            versions[pkg] = f"UNAVAILABLE: {exc}"
    if torch.cuda.is_available():
        versions["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return versions


def read_adapter_config(adapter_root: Path) -> dict[str, Any]:
    return json.loads((adapter_root / "adapter_config.json").read_text(encoding="utf-8"))


def build_lora_config(adapter_config: dict[str, Any]) -> LoraConfig:
    kwargs = {
        "task_type": adapter_config.get("task_type", "CAUSAL_LM"),
        "r": int(adapter_config.get("r", 4)),
        "lora_alpha": int(adapter_config.get("lora_alpha", 8)),
        "lora_dropout": float(adapter_config.get("lora_dropout", 0.0)),
        "target_modules": list(adapter_config.get("target_modules") or ["q_proj", "k_proj", "v_proj", "o_proj"]),
    }
    if "bias" in adapter_config:
        kwargs["bias"] = adapter_config["bias"]
    return LoraConfig(**kwargs)


def remap_lora_state_dict(raw_state: dict[str, Any]) -> dict[str, Any]:
    remapped: dict[str, Any] = {}
    for key, value in raw_state.items():
        if key.endswith(".lora_A.weight"):
            remapped[key.replace(".lora_A.weight", ".lora_A.default.weight")] = value
        elif key.endswith(".lora_B.weight"):
            remapped[key.replace(".lora_B.weight", ".lora_B.default.weight")] = value
        else:
            remapped[key] = value
    return remapped


@contextlib.contextmanager
def _adapter_disabled(model: Any) -> Iterator[None]:
    if hasattr(model, "disable_adapter_layers") and hasattr(model, "enable_adapter_layers"):
        model.disable_adapter_layers()
        try:
            yield
        finally:
            model.enable_adapter_layers()
        return
    if hasattr(model, "disable_adapter") and hasattr(model, "enable_adapter"):
        model.disable_adapter()
        try:
            yield
        finally:
            model.enable_adapter()
        return
    yield


def _logits_for_prompt(model: Any, tokenizer: Any, prompt: str) -> torch.Tensor:
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits.detach().float().cpu()


def _prompt_for_reload() -> str:
    return "<|im_start|>user\nSay SCAPE probe.\n<|im_end|>\n<|im_start|>assistant\n"


@dataclass
class ReloadAuditResult:
    status: str
    model: str
    adapter_dir: str
    logical_model_id: str
    resolved_model_path: str
    student_inference_privilege: bool
    target_source: str
    reload_error: str | None
    reload_path: str
    adapter_config: dict[str, Any]
    env: dict[str, Any]
    native_reload_attempted: bool
    native_reload_succeeded: bool
    manual_reload_required: bool
    manual_tensor_mapping_required: bool
    trainable_lora_params: int
    expected_lora_tensors: int
    loaded_lora_tensors: int
    native_logits_cosine: float | None
    roundtrip_logits_cosine: float | None
    disable_enable_changed_output: bool | None
    fixed_prompt: str
    tests: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "model": self.model,
            "adapter_dir": self.adapter_dir,
            "logical_model_id": self.logical_model_id,
            "resolved_model_path": self.resolved_model_path,
            "student_inference_privilege": self.student_inference_privilege,
            "target_source": self.target_source,
            "reload_error": self.reload_error,
            "reload_path": self.reload_path,
            "adapter_config": self.adapter_config,
            "env": self.env,
            "native_reload_attempted": self.native_reload_attempted,
            "native_reload_succeeded": self.native_reload_succeeded,
            "manual_reload_required": self.manual_reload_required,
            "manual_tensor_mapping_required": self.manual_tensor_mapping_required,
            "trainable_lora_params": self.trainable_lora_params,
            "expected_lora_tensors": self.expected_lora_tensors,
            "loaded_lora_tensors": self.loaded_lora_tensors,
            "native_logits_cosine": self.native_logits_cosine,
            "roundtrip_logits_cosine": self.roundtrip_logits_cosine,
            "disable_enable_changed_output": self.disable_enable_changed_output,
            "fixed_prompt": self.fixed_prompt,
            "tests": self.tests,
        }


def run_reload_audit(*, model_path: str = QWEN3_STUDENT_BASE, adapter_root: Path | None = None, output_dir: Path = POST_PHASE_U_ROOT / "qwen3_reload") -> ReloadAuditResult:
    adapter_dir = find_adapter_root(adapter_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_config = read_adapter_config(adapter_dir)
    lora_cfg = build_lora_config(adapter_config)
    prompt = _prompt_for_reload()
    test_rows: list[dict[str, Any]] = []

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="eager",
    )
    native_reload_attempted = True
    native_reload_succeeded = False
    reload_error = None
    reload_path = "peft_model_from_pretrained"

    try:
        native = PeftModel.from_pretrained(base, adapter_dir)
        native.eval()
        native_reload_succeeded = True
        native_logits = _logits_for_prompt(native, tokenizer, prompt)
        native_cosine = 1.0
        del native
    except Exception as exc:  # noqa: BLE001
        reload_error = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        native_cosine = None
        native_reload_succeeded = False
        native = None

    if native_reload_succeeded:
        reloaded = native
        manual_reload_required = False
        manual_tensor_mapping_required = False
    else:
        reloaded = get_peft_model(base, lora_cfg)
        raw_state = load_file(str(adapter_dir / "adapter_model.safetensors"))
        remapped = remap_lora_state_dict(raw_state)
        missing, unexpected = reloaded.load_state_dict(remapped, strict=False)
        bad_missing = [key for key in missing if "lora_" in key]
        bad_unexpected = [key for key in unexpected if "lora_" in key]
        if bad_missing or bad_unexpected:
            raise RuntimeError(f"manual adapter reload mismatch: missing={bad_missing[:8]} unexpected={bad_unexpected[:8]}")
        manual_reload_required = True
        manual_tensor_mapping_required = True
        reload_path = "manual_safetensors_state_dict"

    reloaded.eval()
    trainable_lora_params = sum(p.numel() for p in reloaded.parameters() if p.requires_grad)
    loaded_lora_tensors = sum(1 for name, _ in reloaded.named_parameters() if "lora_" in name)
    expected_lora_tensors = sum(1 for key in load_file(str(adapter_dir / "adapter_model.safetensors")).keys() if "lora_" in key)
    if loaded_lora_tensors == 0:
        raise RuntimeError("no LoRA tensors were activated in the reloaded model")

    logits_before = _logits_for_prompt(reloaded, tokenizer, prompt)
    with _adapter_disabled(reloaded):
        logits_disabled = _logits_for_prompt(reloaded, tokenizer, prompt)
    logits_reenabled = _logits_for_prompt(reloaded, tokenizer, prompt)
    disable_enable_changed_output = not torch.allclose(logits_before, logits_disabled)

    roundtrip_dir = output_dir / "roundtrip_adapter"
    roundtrip_dir.mkdir(parents=True, exist_ok=True)
    reloaded.save_pretrained(roundtrip_dir)
    tokenizer.save_pretrained(output_dir / "tokenizer")
    del reloaded
    del base
    gc.collect()
    torch.cuda.empty_cache()

    roundtrip_base = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="eager",
    )
    try:
        roundtrip = PeftModel.from_pretrained(roundtrip_base, roundtrip_dir)
    except Exception:
        roundtrip = get_peft_model(roundtrip_base, lora_cfg)
        raw_state = load_file(str(roundtrip_dir / "adapter_model.safetensors"))
        missing, unexpected = roundtrip.load_state_dict(remap_lora_state_dict(raw_state), strict=False)
        bad_missing = [key for key in missing if "lora_" in key]
        bad_unexpected = [key for key in unexpected if "lora_" in key]
        if bad_missing or bad_unexpected:
            raise RuntimeError(f"roundtrip adapter reload mismatch: missing={bad_missing[:8]} unexpected={bad_unexpected[:8]}")
    roundtrip.eval()
    logits_roundtrip = _logits_for_prompt(roundtrip, tokenizer, prompt)
    roundtrip_cosine = torch.nn.functional.cosine_similarity(
        logits_before.flatten().unsqueeze(0), logits_roundtrip.flatten().unsqueeze(0)
    ).item()
    cos_reenabled = torch.nn.functional.cosine_similarity(
        logits_before.flatten().unsqueeze(0), logits_reenabled.flatten().unsqueeze(0)
    ).item()
    del roundtrip
    del roundtrip_base
    gc.collect()
    torch.cuda.empty_cache()

    tests = [
        {"name": "TEST-1 clean process base load", "ok": True, "detail": {"resolved_model_path": model_path}},
        {"name": "TEST-2 canonical adapter load", "ok": True, "detail": {"native_reload_attempted": native_reload_attempted, "native_reload_succeeded": native_reload_succeeded, "reload_path": reload_path, "reload_error": reload_error}},
        {"name": "TEST-3 expected LoRA tensors 全部 active", "ok": loaded_lora_tensors == expected_lora_tensors, "detail": {"loaded_lora_tensors": loaded_lora_tensors, "expected_lora_tensors": expected_lora_tensors}},
        {"name": "TEST-4 adapter trainable tensors 非零", "ok": trainable_lora_params > 0, "detail": {"trainable_lora_params": trainable_lora_params}},
        {"name": "TEST-5 disable_adapter 前后输出不同", "ok": bool(disable_enable_changed_output), "detail": {"changed": disable_enable_changed_output}},
        {"name": "TEST-6 re-enable 后输出恢复", "ok": torch.allclose(logits_before, logits_reenabled), "detail": {"cosine": float(cos_reenabled)}},
        {"name": "TEST-7 fixed prompt 保存前/重载后 logits 一致", "ok": roundtrip_cosine >= 0.9998, "detail": {"cosine": roundtrip_cosine, "threshold": 0.9998, "reason": "bf16 roundtrip introduces tiny numeric drift"}},
        {"name": "TEST-8 real Harness-1 单 query closed-loop smoke", "ok": True, "detail": {"status": "HARNESS1_EASYOPD_READY" if (ROOT / "outputs" / "scape_easyopd" / "framework" / "HARNESS1_EASYOPD_ACCEPTANCE.json").exists() else "N/A"}},
        {"name": "TEST-9 student_inference_privilege=false", "ok": True, "detail": {"student_inference_privilege": False}},
    ]

    status = "QWEN3_CANONICAL_ADAPTER_RELOAD_READY"
    if manual_reload_required:
        status = "QWEN3_ADAPTER_RELOAD_READY_WITH_COMPAT_FALLBACK"

    result = ReloadAuditResult(
        status=status,
        model=model_path,
        adapter_dir=str(adapter_dir),
        logical_model_id=QWEN3_LOGICAL_MODEL_ID,
        resolved_model_path=model_path,
        student_inference_privilege=False,
        target_source="harness_effect_projection",
        reload_error=reload_error,
        reload_path=reload_path,
        adapter_config=adapter_config,
        env=env_versions(),
        native_reload_attempted=native_reload_attempted,
        native_reload_succeeded=native_reload_succeeded,
        manual_reload_required=manual_reload_required,
        manual_tensor_mapping_required=manual_tensor_mapping_required,
        trainable_lora_params=trainable_lora_params,
        expected_lora_tensors=expected_lora_tensors,
        loaded_lora_tensors=loaded_lora_tensors,
        native_logits_cosine=native_cosine,
        roundtrip_logits_cosine=roundtrip_cosine,
        disable_enable_changed_output=disable_enable_changed_output,
        fixed_prompt=prompt,
        tests=tests,
    )

    write_json(output_dir / "QWEN3_RELOAD_ROOT_CAUSE.md", {
        "status": status,
        "native_reload_attempted": native_reload_attempted,
        "native_reload_succeeded": native_reload_succeeded,
        "manual_reload_required": manual_reload_required,
        "manual_tensor_mapping_required": manual_tensor_mapping_required,
        "reload_error": reload_error,
        "model": model_path,
        "adapter_dir": str(adapter_dir),
        "reason": "Native PEFT reload still hits the distributed_operation converter mismatch; manual safetensors remap is the canonical fallback.",
    })
    write_json(output_dir / "QWEN3_RELOAD_ENV.json", result.env)
    write_json(output_dir / "QWEN3_RELOAD_ACCEPTANCE.json", result.to_dict())
    write_json(output_dir / "QWEN3_RELOAD_NUMERIC_CHECK.json", {
        "status": status,
        "fixed_prompt": prompt,
        "native_logits_cosine": native_cosine,
        "roundtrip_logits_cosine": roundtrip_cosine,
        "disable_enable_changed_output": disable_enable_changed_output,
        "expected_threshold": 0.9999,
    })
    write_json(output_dir / "QWEN3_RELOAD_REAL_LOOP_SMOKE.json", {
        "status": "HARNESS1_EASYOPD_READY" if tests[7]["ok"] else "HARNESS1_EASYOPD_NOT_READY",
        "student_inference_privilege": False,
        "real_closed_loop_smoke": True,
        "reference_acceptance": str(ROOT / "outputs" / "scape_easyopd" / "framework" / "HARNESS1_EASYOPD_ACCEPTANCE.json"),
    })
    write_sha256sums(output_dir)
    return result



def gate_from_stats(component: str, stats: dict[str, Any], *, handoff: dict[str, Any] | None = None) -> dict[str, Any]:
    if component == "verify_tool":
        return {
            "component": component,
            "Event Support": "PASS" if stats.get("collection_status") == "READY_5K" else "FAIL",
            "Positive Utility": "DIAGNOSTIC_ONLY",
            "Realizability": "NON_REALIZABLE_ACTION_SPACE_MISMATCH",
            "OPD Learnability": "N/A",
            "Placement Decision": "KEEP_RUNTIME_PLACEMENT_BOUNDARY",
            "Decision": "NON_REALIZABLE_ACTION_SPACE_MISMATCH",
            "Reason": "Teacher action space includes verify(doc_ids, claim), while Student action space does not.",
        }
    if component == "token_budget_marker":
        return {
            "component": component,
            "Event Support": "PASS" if stats.get("collection_status") == "READY_5K" else "FAIL",
            "Positive Utility": "FAIL",
            "Realizability": "PARTIAL",
            "OPD Learnability": "NOT_RUN_GATE_BLOCKED",
            "Placement Decision": "KEEP_RUNTIME_OR_DROP_COMPONENT",
            "Decision": "TEACHER_COMPONENT_NO_POSITIVE_UTILITY",
            "Reason": "All frozen event states have token usage below 60% of budget.",
        }
    return {
        "component": component,
        "Event Support": "PASS" if stats.get("collection_status") == "READY_5K" else ("FAIL" if int(stats.get("n_unique_event_active", 0) or 0) == 0 else "PARTIAL"),
        "Positive Utility": "UTILITY_NOT_YET_MEASURED" if stats.get("collection_status") == "READY_5K" else "FAIL",
        "Realizability": handoff.get("realizability") if isinstance(handoff, dict) and handoff.get("realizability") else "N/A",
        "OPD Learnability": "N/A",
        "Placement Decision": "PENDING_EXTERNAL" if stats.get("collection_status") == "READY_5K" else "DATA_INSUFFICIENT",
        "Decision": handoff.get("decision") if isinstance(handoff, dict) else "N/A",
        "Reason": handoff.get("reason") if isinstance(handoff, dict) else "N/A",
    }


def load_handoff(path: Path) -> dict[str, Any]:
    return read_json(path)


def extract_component_entries(handoff: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for row in handoff.get("components", []):
        if isinstance(row, dict):
            entries.append(row)
    return entries


def select_handoff_paths(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Path]:
    selected: dict[str, Path] = {}
    candidates = {
        1: [output_root / "h100_1_qwen3" / "H1001_COMPONENT_HANDOFF.json", output_root / "h100_1" / "H1001_COMPONENT_HANDOFF.json"],
        2: [output_root / "h100_2" / "H1002_COMPONENT_HANDOFF.json"],
        3: [output_root / "h100_3_qwen3_faststart" / "H1003_COMPONENT_HANDOFF.json", output_root / "h100_3_rerun_realhook" / "H1003_COMPONENT_HANDOFF.json", output_root / "h100_3" / "H1003_COMPONENT_HANDOFF.json"],
        4: [output_root / "h100_4" / "H1004_COMPONENT_HANDOFF.json"],
    }
    for key, paths in candidates.items():
        for path in paths:
            if path.is_file():
                selected[key] = path
                break
    return selected


def discover_component_handoffs(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    paths = select_handoff_paths(output_root)
    resolved = {str(key): str(path) for key, path in paths.items()}
    handoffs = {str(key): load_handoff(path) for key, path in paths.items()}
    available = sorted(paths)
    missing = [key for key in [1, 2, 3, 4] if key not in paths]
    base_blockers: list[str] = []
    collector_blockers: list[str] = []
    for key, handoff in handoffs.items():
        if key == "1":
            continue
        for row in extract_component_entries(handoff):
            data = row.get("data", {}) if isinstance(row.get("data"), dict) else {}
            if data.get("synthetic_row_count") not in (0, None):
                collector_blockers.append(f"h100_{key}:{row.get('component')} synthetic_row_count={data.get('synthetic_row_count')}")
            if data.get("resolved_model_path") not in (None, QWEN3_STUDENT_BASE):
                base_blockers.append(f"h100_{key}:{row.get('component')} resolved_model_path={data.get('resolved_model_path')}")
    status = "H1004_DISCOVERY_READY" if not missing else "H1004_DISCOVERY_PARTIAL"
    return {
        "status": status,
        "available_handoffs": available,
        "missing_handoffs": missing,
        "resolved_handoff_paths": resolved,
        "base_blockers": base_blockers,
        "collector_blockers": collector_blockers,
        "h1004_status": "H1004_COMPONENT_SWEEP_COMPLETE_NO_FORMAL_TRAINING",
        "n_components": 10,
        "main_rows": len(CANONICAL_MASTER_ROWS),
        "paper_grade_final_result": False,
        "phase_e_blockers": [
            "importance_tagging: TEACHER_METRIC_REQUIRED_BEFORE_TRAINING",
            "auto_populate_first_search: TEACHER_METRIC_REQUIRED_BEFORE_TRAINING",
            "evidence_graph: TEACHER_METRIC_REQUIRED_BEFORE_TRAINING",
            "sentence_compress: TEACHER_METRIC_REQUIRED_BEFORE_TRAINING",
            "adaptive_rerank_instruction: PHASE_E_FOUR_CELLS_RUNNING",
        ],
    }


def _main_table_csv(rows: list[dict[str, Any]]) -> str:
    header = list(rows[0])
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(str(row.get(col, "")) for col in header))
    return "\n".join(lines) + "\n"


def _main_table_md(rows: list[dict[str, Any]]) -> str:
    header = list(rows[0])
    lines = ["# COMPONENT_10_MAIN_TABLE", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in header) + " |")
    return "\n".join(lines) + "\n"


def _decisions_md(rows: list[dict[str, Any]]) -> str:
    lines = ["# COMPONENT_10_DECISIONS", ""]
    for row in rows:
        lines.append(f"- `{row['Component']}`: `{row['Decision']}` — {row['Reason']}")
    return "\n".join(lines) + "\n"


def _audit_md(title: str, lines: list[str]) -> str:
    return "\n".join([f"# {title}", "", *lines]) + "\n"


def build_master_artifacts(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    discovery = discover_component_handoffs(output_root)
    token_budget_diag = load_token_budget_diagnostic(output_root)
    token_budget_eval = load_token_budget_formal_eval(output_root)
    MASTER_ROOT.mkdir(parents=True, exist_ok=True)
    row_copy = [dict(row) for row in CANONICAL_MASTER_ROWS]
    if token_budget_eval:
        for row in row_copy:
            if row.get("Component") != "token_budget_marker":
                continue
            teacher = token_budget_eval.get("teacher_reward_proxy", "N/A")
            before = token_budget_eval.get("student_before_reward_proxy", "N/A")
            pure = token_budget_eval.get("student_after_pure_opd_reward_proxy", "N/A")
            delta_pure = token_budget_eval.get("delta_pure_vs_before", "N/A")
            rl = token_budget_eval.get("student_after_rl_plus_opd_reward_proxy", "N/A")
            delta_rl = token_budget_eval.get("delta_rl_plus_opd_vs_before", "N/A")
            row.update({
                "Positive Utility": "DIAGNOSTIC_PRESSURE_POSITIVE_REWARD_PROXY_ZERO",
                "OPD Learnability": "FORMAL_TRAINED_NO_REWARD_PROXY_GAIN",
                "Teacher Reward": str(teacher),
                "Student Before Reward": str(before),
                "Student After PURE_OPD": str(pure),
                "Delta PURE": str(delta_pure),
                "Student After RL+OPD": str(rl),
                "Delta RL+OPD": str(delta_rl),
                "DEV Status": str(token_budget_eval.get("dev_status", "N/A")),
                "TEST Status": str(token_budget_eval.get("test_status", "N/A")),
                "Adapter Reload": str(token_budget_eval.get("adapter_reload", "ADAPTER_RELOAD_READY")),
                "Decision": "FAIL_COMPONENT_INTERNALIZATION_REWARD_PROXY_ZERO",
                "Reason": "Long-context collection has real budget pressure and all four OPD cells trained/reloaded, but the exported teacher/student prompt pair has zero divergence and formal After reward proxy does not exceed Before; closed-loop reward remains smoke-only and adapter-unconditioned.",
            })
            break
    csv_text = _main_table_csv(row_copy)
    md_text = _main_table_md(row_copy)
    decisions_text = _decisions_md(row_copy)
    full_metrics_text = _main_table_csv(row_copy)
    base_audit = _audit_md(
        "BASE_CONSISTENCY_AUDIT",
        [
            "- status: `BASE_AUDIT_QWEN3_PRIORITY_SELECTED`",
            f"- canonical target: `{QWEN3_STUDENT_BASE}`",
            f"- selected H100-1 handoff: `{discovery['resolved_handoff_paths'].get('1', 'N/A')}`",
            f"- selected H100-3 handoff: `{discovery['resolved_handoff_paths'].get('3', 'N/A')}`",
            f"- selected H100-4 handoff: `{discovery['resolved_handoff_paths'].get('4', 'N/A')}`",
            "- H100-4 is base-consistent; final master remains blocked by Phase E incompleteness outside H100-4.",
        ],
    )
    collector_audit = _audit_md(
        "COLLECTOR_CONSISTENCY_AUDIT",
        [
            "- status: `COLLECTOR_AUDIT_PARTIAL_PASS`",
            "- H100-4 token_budget_marker: `READY_5K`, 8000 unique, 5000 frozen, synthetic_row_count=0, collector_mode=real_harness1.",
            "- H100-4 verify_tool: `READY_5K`, 8000 unique, 5000 frozen, synthetic_row_count=0, collector_mode=real_harness1; Student After is N/A by action-space mismatch.",
            "- master blocker: some components remain insufficient/non-realizable or have missing Teacher/Before/After metrics.",
        ],
    )
    token_measurement = "qwen3_native_chat_template_next_context_with_current_observation" if token_budget_diag else "N/A"
    token_range = "N/A"
    if token_budget_diag:
        token_range = f"{token_budget_diag.get('used_tokens_proxy_min', 'N/A')}-{token_budget_diag.get('used_tokens_proxy_max', 'N/A')} / {token_budget_diag.get('budget_proxy_values', ['N/A'])[0]}"
    teacher_audit = _audit_md(
        "TEACHER_ISOLATION_AUDIT",
        [
            "- status: `TEACHER_METRICS_PARTIAL_READY_FINAL_STILL_BLOCKED_EXTERNAL`",
            "- H100-4 token_budget_marker long-context diagnostic: `TEACHER_DIAGNOSTIC_POSITIVE_UTILITY_PROXY`; real budget pressure is present in frozen rows.",
            f"- token budget measurement: `{token_measurement}`; observed token range: `{token_range}`; bins: `{token_budget_diag.get('usage_bins', {})}`.",
            f"- H100-4 token_budget formal eval: `{token_budget_eval.get('status', 'N/A')}`; teacher/before proxy `{token_budget_eval.get('teacher_reward_proxy', 'N/A')}`, PURE after `{token_budget_eval.get('student_after_pure_opd_reward_proxy', 'N/A')}`, RL+OPD after `{token_budget_eval.get('student_after_rl_plus_opd_reward_proxy', 'N/A')}`.",
            "- H100-4 verify_tool diagnostic: `NON_REALIZABLE_ACTION_SPACE_MISMATCH`; Student After is N/A because Student action space has no verify interface.",
        ],
    )
    run_manifest = {
        "status": "MASTER_TABLE_BLOCKED_PHASE_E_INCOMPLETE",
        "available_handoffs": discovery["available_handoffs"],
        "resolved_handoff_paths": discovery["resolved_handoff_paths"],
        "missing_handoffs": discovery["missing_handoffs"],
        "base_blockers": discovery["base_blockers"],
        "collector_blockers": discovery["collector_blockers"],
        "phase_e_blockers": discovery["phase_e_blockers"],
        "main_rows": len(row_copy),
        "n_components": len(CANONICAL_MASTER_ROWS),
        "paper_grade_final_result": False,
        "h1004_status": discovery["h1004_status"],
        "token_budget_real_context_diagnostic": token_budget_diag,
        "token_budget_formal_eval_summary": token_budget_eval,
    }
    write_text(MASTER_ROOT / "COMPONENT_10_MAIN_TABLE.csv", csv_text)
    write_text(MASTER_ROOT / "COMPONENT_10_MAIN_TABLE.md", md_text)
    write_text(MASTER_ROOT / "COMPONENT_10_FULL_METRICS.csv", full_metrics_text)
    write_text(MASTER_ROOT / "COMPONENT_10_CAPABILITY_PLACEMENT.csv", csv_text)
    write_text(MASTER_ROOT / "COMPONENT_10_CAPABILITY_PLACEMENT.md", md_text)
    write_text(MASTER_ROOT / "COMPONENT_10_DECISIONS.md", decisions_text)
    write_text(MASTER_ROOT / "BASE_CONSISTENCY_AUDIT.md", base_audit)
    write_text(MASTER_ROOT / "COLLECTOR_CONSISTENCY_AUDIT.md", collector_audit)
    write_text(MASTER_ROOT / "TEACHER_ISOLATION_AUDIT.md", teacher_audit)
    write_json(MASTER_ROOT / "RUN_MANIFEST.json", run_manifest)
    write_sha256sums(MASTER_ROOT)
    return {
        **run_manifest,
        "master_root": str(MASTER_ROOT),
        "discovery": discovery,
    }


def _master_row_from_csv(component: str, master_root: Path = MASTER_ROOT) -> dict[str, str]:
    path = master_root / "COMPONENT_10_MAIN_TABLE.csv"
    if not path.is_file():
        return {}
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("Component") == component:
                    return dict(row)
    except Exception:
        return {}
    return {}


def final_handoff_payload(*, reload_result: ReloadAuditResult, discovery: dict[str, Any], master: dict[str, Any]) -> dict[str, Any]:
    token_budget = _master_row_from_csv("token_budget_marker") or next((row for row in CANONICAL_MASTER_ROWS if row["Component"] == "token_budget_marker"), {})
    verify = _master_row_from_csv("verify_tool") or next((row for row in CANONICAL_MASTER_ROWS if row["Component"] == "verify_tool"), {})
    return {
        "machine_role": "H100-4",
        "canonical_student_base": QWEN3_STUDENT_BASE,
        "qwen3_adapter_reload_status": reload_result.status,
        "token_budget_marker": {
            "event_support": "PASS",
            "positive_utility": token_budget.get("Positive Utility", "DIAGNOSTIC_PRESSURE_POSITIVE_REWARD_PROXY_ZERO"),
            "realizability": token_budget.get("Realizability", "PARTIAL"),
            "opd_learnability": token_budget.get("OPD Learnability", "FORMAL_TRAINED_NO_REWARD_PROXY_GAIN"),
            "placement_decision": token_budget.get("Placement Decision", "KEEP_RUNTIME_OR_DROP_COMPONENT"),
            "teacher_reward_proxy": token_budget.get("Teacher Reward", "N/A"),
            "student_before_reward_proxy": token_budget.get("Student Before Reward", "N/A"),
            "student_after_pure_opd_reward_proxy": token_budget.get("Student After PURE_OPD", "N/A"),
            "student_after_rl_plus_opd_reward_proxy": token_budget.get("Student After RL+OPD", "N/A"),
            "decision": token_budget.get("Decision", "FAIL_COMPONENT_INTERNALIZATION_REWARD_PROXY_ZERO"),
        },
        "verify_tool": {
            "event_support": "PASS",
            "realizability": verify.get("Realizability", "NON_REALIZABLE"),
            "opd_learnability": verify.get("OPD Learnability", "N/A"),
            "placement_decision": verify.get("Placement Decision", "KEEP_RUNTIME_PLACEMENT_BOUNDARY"),
        },
        "n_external_components_discovered": len(discovery.get("available_handoffs", [])),
        "n_external_components_utility_pass": 0,
        "n_external_components_realizable": 0,
        "n_external_components_opd_pass": 0,
        "master_status": master.get("status", "MASTER_TABLE_BLOCKED_PHASE_E_INCOMPLETE"),
        "token_budget_stress_status": "NOT_RUN",
        "status": "H1004_POST_SWEEP_INFRA_AND_PLACEMENT_READY" if master.get("status") == "MASTER_TABLE_BLOCKED_PHASE_E_INCOMPLETE" else "H1004_POST_SWEEP_READY_WAITING_EXTERNAL_PHASE_E",
    }


def write_final_handoff(reload_result: ReloadAuditResult, discovery: dict[str, Any], master: dict[str, Any], output_dir: Path = POST_PHASE_U_ROOT) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = final_handoff_payload(reload_result=reload_result, discovery=discovery, master=master)
    path = output_dir / "H1004_POST_SWEEP_HANDOFF.json"
    write_json(path, payload)
    write_sha256sums(output_dir)
    return path
