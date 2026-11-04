"""Proposer clients: turn an EvolutionPacket into CandidatePatch objects.

Real model requests only.  Parse failures are recorded and retried a limited
number of times; a fabricated candidate is never returned.  ``self`` mode uses
the current actor policy for code generation; ``assisted`` mode may use an
external stronger coding model and must be labeled as such.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from rsi.contracts.artifact import CandidateFile, CandidatePatch
from rsi.contracts.common import JSONValue
from rsi.contracts.errors import CandidateInvalid, InfraError
from rsi.evolution.packet import EvolutionPacket

PROPOSER_VERSION = "rsi_proposer_v1"

# ⚠️ 不要在这里维护独立的 JSON 定位正则。候选提取必须与停机判据
# ``rsi.contracts.action.extract_json_object`` 同源（见 parse_candidate_patch 的
# docstring）——曾各写一套，导致生成端判定完整、解析端判定无 JSON，全部拒绝。


def load_prompt_template(name: str = "proposer_v1.md") -> str:
    root = Path(__file__).resolve().parents[3] / "prompts"
    path = root / name
    if path.is_file():
        return path.read_text("utf-8")
    return _FALLBACK_PROMPT


_FALLBACK_PROMPT = """You evolve an agent harness component library.
Given the parent bank sources and real execution feedback, propose ONE candidate
patch as JSON with keys: parent_bank_digest, operation (add|edit|compose|prune|replace),
hypothesis, files ([{path, content}]), delete_paths, manifest_patch, alignment_hints,
evidence_refs, expected_cost_change, self_checks.
Write real Python code in files[].content. Do not reference dataset task ids in code.

EXACT importable API surface (from rsi.harness.component ONLY; no submodules,
never invent types like ActionLog/ToolResult or a "Component" base class —
the component protocol is HarnessComponent, which you do NOT need to import;
a component is any class defining the five hooks):
  EpisodeContext(episode_id, task, tool_catalog, private_state, rng_seed,
                 extra_llm_call_budget, config), PublicView(task, tool_catalog,
                 history, remaining_action_budget, extra), RuntimeAPI(call_tool,
                 call_model, get_state, set_state, budget, log), ViewPatch,
                 ProposalEffect, StatePatch, FinishEffect, HarnessComponent
  from rsi.contracts.action import ToolCall, Finish; from rsi.contracts.events import Event
  Event fields (attr access, NOT dict): event_id, order, component_id,
  event_kind, pre_state_digest, post_state_digest, action, observation
  (Observation: content/ok/error_kind/provenance/structured), visibility,
  effect_type, information_provenance, cost, data. event.get() is invalid.

  ctx.config carries {"max_actions","max_tool_calls","max_extra_llm_calls","max_wall_ms","rng_seed"}.

HARD REQUIREMENTS:
1. Every component class defines ALL FIVE hooks with EXACTLY these signatures:
   def initialize(self, ctx): ...
   def before_model(self, view, api): ...
   def after_proposal(self, action, view, api): ...
   def after_observation(self, event, api): ...
   def before_finish(self, action, view, api): ...
   (do not omit parameters; return no-op effect objects when idle)
2. manifest_patch.entrypoint class name must EXACTLY match the class defined
   in the source file.
3. Hooks must never raise on ordinary inputs.
"""


def render_proposer_prompt(
    packet: EvolutionPacket,
    template: str | None = None,
    retry_notes: Sequence[str] = (),
) -> str:
    tpl = template or load_prompt_template()
    payload = packet.to_json()
    notes = ""
    if retry_notes:
        joined = "\n".join(f"- {n}" for n in retry_notes)
        notes = (
            "\n\n=== PREVIOUS ATTEMPT ERRORS (your last patches were rejected; "
            "FIX THESE EXACTLY) ===\n" + joined + "\n"
        )
    # Keep parent sources inline: the proposer must read real parent code.
    #
    # ⚠️ 输出封套骨架必须**放在最后**（9fhn3/3b1bb 复盘）。模板里的 schema 说明在
    # 中段，之后紧跟一大坨 EvolutionPacket JSON——模型看到的"最近形状"是 packet，
    # 于是吐出压平的 {path,content,entrypoint} 而不是 {operation,files[],manifest_patch}
    # 封套，全部被拒。近因效应：越靠后越强，schema 必须在最末重述。
    skeleton = (
        "\n\n=== OUTPUT ENVELOPE (RESTATE: this is what you MUST emit) ===\n"
        "Top-level keys MUST be exactly these (this is a patch envelope, NOT a bare "
        "file object — do NOT emit only {\"path\",\"content\",\"entrypoint\"}):\n"
        "manifest_patch.entrypoint format is STRICTLY \"module:Class\" "
        "(e.g. \"components.my_component:MyComponent\") — never a bare class name.\n"
        "Emit PLAIN JSON only (no comments, no trailing commas).\n"
        "{\n"
        '  "parent_bank_digest": "<verbatim from packet>",\n'
        '  "operation": "add",\n'
        '  "hypothesis": "<mechanism + expected improvement>",\n'
        '  "files": [{"path": "components/my_component.py", "content": "<FULL python source>"}],\n'
        '  "delete_paths": [],\n'
        '  "manifest_patch": {\n'
        '    "id": "my_component",\n'
        '    "version": "1",\n'
        '    "files": ["components/my_component.py"],\n'
        '    "entrypoint": "components.my_component:MyComponent",\n'
        '    "depends_on": [], "conflicts_with": [], "hook_order": 100,\n'
        '    "required_capabilities": [], "state_schema": {}, "resource_caps": {},\n'
        '    "provenance": {}, "alignment_hints": {}, "kind": "other"\n'
        "  },\n"
        '  "alignment_hints": {}, "evidence_refs": [],\n'
        '  "expected_cost_change": {}, "self_checks": []\n'
        "}\n"
        "Respond with a single JSON object for the CandidatePatch now."
    )
    return (
        tpl
        + notes
        + "\n\n=== EVOLUTION PACKET (JSON) ===\n"
        + json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        + skeleton
    )


def _derive_entrypoint(path: str, content: str) -> str | None:
    """从 flat 文件对象确定性推导 ``module:Class`` entrypoint。

    module 由 path 推导（``components/xxx.py`` -> ``components.xxx``），
    class 取源码里第一个顶层 ``class`` 定义。两者缺一返回 None（维持拒绝）。
    注意 ``entrypoint module must be in files list``（loader.py:69）——files
    只有这一个文件，所以模块名必须落在该文件本身，不猜别的落点。
    """
    if not isinstance(path, str) or not path.endswith(".py"):
        return None
    mod = path[:-3].replace("/", ".").replace("\\", ".").strip(".")
    if not mod or " " in mod:
        return None
    for line in content.splitlines():
        s = line.strip()
        # 只认顶层 class（非缩进行）；缩进体内的 class 不是入口
        if line.startswith("class ") and ":" in s:
            name = s[len("class "):].split("(", 1)[0].split(":", 1)[0].strip()
            if name.isidentifier():
                return f"{mod}:{name}"
    return None


def _coerce_flat_file_envelope(obj: dict[str, Any]) -> dict[str, Any]:
    """把"压平的文件对象"提升成合法 patch 封套（窄口径，不造内容）。

    3b1bb 拿到的实证：模型吐 ``{parent_bank_digest, path, content, entrypoint}``
    ——把 ``files[]`` 项、``manifest_patch.entrypoint`` 与封套字段混成一层，
    缺 ``operation``，于是 ``CandidatePatch.from_json`` KeyError 全拒。

    只在**无歧义**时提升：``operation`` 缺失 且 同时有 ``path`` 与 ``content``。
    提升规则：
      * ``files`` = 模型给的 ``{path, content}`` 原样（**内容一字不造**）
      * ``operation`` = "add"（模型自己写了一个新文件，只有 add 说得通）
      * ``manifest_patch`` 仅在模型给了 ``entrypoint`` 时补；``id`` 由文件名推导
        （命名约定，不是语义捏造），``files`` 同上
      * ``hypothesis`` 缺则留空串（元数据缺省，非捏造结论）

    提升发生时返回值上带 ``_normalized`` 标记，调用方应记进 history 留痕。
    """
    if "operation" in obj:
        return obj
    if not (isinstance(obj.get("path"), str) and "content" in obj):
        return obj
    path = obj["path"]
    content = obj["content"]
    out: dict[str, Any] = {
        "parent_bank_digest": obj.get("parent_bank_digest"),
        "operation": "add",
        "hypothesis": obj.get("hypothesis", ""),
        "files": [{"path": path, "content": content}],
        "delete_paths": [],
        "manifest_patch": {},
        "alignment_hints": obj.get("alignment_hints") or {},
        "evidence_refs": obj.get("evidence_refs") or [],
        "expected_cost_change": obj.get("expected_cost_change") or {},
        "self_checks": obj.get("self_checks") or [],
    }
    entrypoint = obj.get("entrypoint")
    if not (isinstance(entrypoint, str) and entrypoint):
        # evimr 实证（2026-11-04）：27B 自提议稳定吐 {path, content} 且不带
        # entrypoint —— 18 次提议全拒。源码里有真实 class 时，机械提取类名
        # 不是造内容：组件会被真实注册执行（fh46b 的 no-op 防线针对的是
        # "落盘文件不进 manifest"，这里文件进 manifest 且可加载，防线语义保住）。
        # 源码里没有 class 才维持拒绝（真正无法注册的 no-op）。
        entrypoint = _derive_entrypoint(path, content)
    if isinstance(entrypoint, str) and entrypoint:
        stem = Path(path).stem
        out["manifest_patch"] = {
            "id": obj.get("id") or stem,
            "version": str(obj.get("version", "1")),
            "files": [path],
            "entrypoint": entrypoint,
            "depends_on": [],
            "conflicts_with": [],
            "hook_order": int(obj.get("hook_order", 100)),
            "required_capabilities": [],
            "state_schema": {},
            "resource_caps": {},
            "provenance": {},
            "alignment_hints": {},
            "kind": "other",
        }
    else:
        # ⚠️ 无 entrypoint 就不能抬——否则新文件不进 manifest，候选退化成
        # "库没变"的 no-op，分数必然等于 parent 原分，却仍被记 legal（fh46b 实证：
        # 1f8ac955 两次 ACCEPTED，磁盘只有 parent 的 context_compressor.py，
        # macro 0.3273 与 champion parent 完全相同）。老实拒绝，让模型补 entrypoint。
        raise CandidateInvalid(
            "flat-file patch missing entrypoint: a new component must declare "
            "manifest_patch.entrypoint as 'module:Class' (e.g. "
            "\"components.guard:Guard\"); refusing to emit a no-op candidate"
        )
    out["_normalized"] = "flat_file->envelope"
    return out


def parse_candidate_patch(text: str, *, parent_bank_digest: str) -> CandidatePatch:
    """Parse one CandidatePatch from model output.  Never fabricates.

    ⚠️ 用与 :func:`rsi.contracts.action.extract_json_object` **同一套**定位逻辑。
    2026-11-02 9fhn3 复盘：停机判据 ``generation_complete`` 用的是宽松版
    （容忍散文前缀/思考块/围栏，按括号深度配平），而这里曾用严格版
    （围栏非贪婪正则 + 必须整体首尾是 ``{``/``}``）。于是生成端判定"JSON 已完整、
    停止生成"，解析端却说"no JSON object"——18 次提议全废，
    ``evaluated_candidates: 0``、``CANDIDATES_REPEATEDLY_INVALID``。
    两者必须同源，否则生成/解析永远对不齐。
    """
    from rsi.contracts.action import ActionParseError, extract_json_object

    try:
        obj = extract_json_object(text)
    except ActionParseError as e:
        raise CandidateInvalid(f"proposer output contains no JSON object: {e}") from e
    except Exception as e:  # noqa: BLE001 —— 解析器自身出错也要留下可诊断的证据
        raise CandidateInvalid(f"proposer output unparsable: {e!r}") from e
    if not isinstance(obj, dict):
        raise CandidateInvalid("proposer JSON must be an object")
    obj.setdefault("parent_bank_digest", parent_bank_digest)
    if obj.get("parent_bank_digest") != parent_bank_digest:
        raise CandidateInvalid(
            f"candidate parent digest mismatch: {obj.get('parent_bank_digest')} != {parent_bank_digest}"
        )
    obj = _coerce_flat_file_envelope(obj)
    if obj.pop("_normalized", None):
        # 留痕：规范化只是把模型给的 path/content 抬进封套，内容一字未造；
        # 但 provenance 必须能看出这条候选经过了形态转换。
        print(f"[proposer] normalized flat-file envelope -> patch "
              f"(files={[f.get('path') for f in obj.get('files', [])]})",
              flush=True)
    # from_json 对缺字段的 JSON 会抛 KeyError/TypeError；必须收成 CandidateInvalid，
    # 否则一个畸形候选会把整个 search 打崩（而不是跳过该候选继续）。
    try:
        patch = CandidatePatch.from_json(obj)
    except CandidateInvalid:
        raise
    except Exception as e:  # noqa: BLE001
        raise CandidateInvalid(f"candidate patch malformed: {e!r}; keys={sorted(obj)[:12]}") from e
    if patch.operation != "prune" and not patch.files and not patch.manifest_patch:
        raise CandidateInvalid("candidate must contain real files or a manifest_patch")
    # add 必须注册组件，否则落盘文件不进 manifest = no-op 候选（分数恒等于 parent）。
    if patch.operation == "add" and patch.files and not (
        isinstance(patch.manifest_patch, Mapping) and patch.manifest_patch.get("entrypoint")
    ):
        raise CandidateInvalid(
            "operation=add with files requires manifest_patch.entrypoint "
            "('module:Class'); without it the component is never registered "
            "and the candidate is a silent no-op"
        )
    return patch


class Proposer(Protocol):
    mode: str  # self | assisted | fixture

    def propose(self, packet: EvolutionPacket, *, retry_notes: Sequence[str] = ()) -> CandidatePatch: ...


@dataclass
class APIProposer:
    """Assisted mode: external coding model over an OpenAI-compatible API."""

    client: Any
    max_retries: int = 2
    mode: str = "assisted"
    temperature: float = 0.7

    def propose(self, packet: EvolutionPacket, *, retry_notes: Sequence[str] = ()) -> CandidatePatch:
        from rsi.providers.openai_compat import APIMessage

        prompt = render_proposer_prompt(packet, retry_notes=retry_notes)
        last_err: BaseException | None = None
        for _attempt in range(self.max_retries + 1):
            try:
                out = self.client.chat(
                    [APIMessage(role="user", content=prompt)],
                    temperature=self.temperature,
                )
            except InfraError as e:
                last_err = e
                continue
            try:
                return parse_candidate_patch(out, parent_bank_digest=packet.parent_bank_digest)
            except CandidateInvalid as e:
                last_err = e
                continue
        raise CandidateInvalid(f"proposer failed after retries: {last_err!r}")


@dataclass
class SelfProposer:
    """Self mode: the current actor policy generates the candidate code.

    For HF policies generation goes through the same provider interface used
    for task rollouts (batch-synchronized proposer reload is the caller's job).
    ``max_new_tokens`` bounds the generation budget per attempt (a candidate
    patch is a JSON object, not a novel).
    """

    policy: Any
    max_retries: int = 2
    mode: str = "self"
    max_new_tokens: int = 1024

    def propose(self, packet: EvolutionPacket, *, retry_notes: Sequence[str] = ()) -> CandidatePatch:
        from rsi.contracts.policy import SamplingParams

        prompt = render_proposer_prompt(packet, retry_notes=retry_notes)
        prompt_ids = self.policy.encode(prompt)
        last_err: BaseException | None = None
        for _attempt in range(self.max_retries + 1):
            sample = self.policy.sample(
                prompt_ids, SamplingParams(temperature=0.7, max_new_tokens=self.max_new_tokens)
            )
            try:
                return parse_candidate_patch(sample.text, parent_bank_digest=packet.parent_bank_digest)
            except CandidateInvalid as e:
                last_err = e
                continue
        raise CandidateInvalid(f"self proposer failed after retries: {last_err!r}")


@dataclass
class FixtureProposer:
    """Test-only proposer emitting prebuilt patches.

    Fixed fixtures validate code paths (``test_real_code_materialization``)
    and are never research results.
    """

    patches: Sequence[CandidatePatch]
    mode: str = "fixture"
    _i: int = 0
    # recorded for tests: the retry_notes each propose() call received
    seen_retry_notes: list[tuple[str, ...]] = field(default_factory=list)

    def propose(self, packet: EvolutionPacket, *, retry_notes: Sequence[str] = ()) -> CandidatePatch:
        self.seen_retry_notes.append(tuple(retry_notes))
        if self._i >= len(self.patches):
            raise CandidateInvalid("fixture proposer exhausted")
        patch = self.patches[self._i]
        self._i += 1
        return patch
