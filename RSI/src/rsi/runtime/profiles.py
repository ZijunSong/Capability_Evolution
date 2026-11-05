"""Fixed H0 profiles (refactor guide §5): the non-evolvable behavioral base.

A profile is the fixed instruction text + environment-interface contract that
every arm of a campaign shares.  It is an IMMUTABLE, versioned object:

    fixed profile → environment base doc → task → tool schemas
    → real history → bank's appended guidance → next action

Bank components may add guidance AFTER the profile but never overwrite,
reorder, or claim its identity.  Profile text participates in prompt_hash, so
changing it (or the window) changes the protocol identity — old baselines are
never silently reused under a new profile.

Configs MUST declare ``runtime.profile`` explicitly; a missing profile id is
a load error (guide §5.2: avoid a typo'd key silently falling back to the old
base).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from rsi.contracts.common import JSONValue, digest_of
from rsi.contracts.errors import ProtocolViolation

PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"

# profile_id -> markdown file
PROFILE_FILES = {
    "react_zero_shot_v1": "h0_react_v1.md",
    "act_only_v1": "h0_act_v1.md",
}


@dataclass(frozen=True)
class H0Profile:
    """Immutable H0 behavioral base (guide §5.2)."""

    profile_id: str
    instruction_text: str
    environment_doc_text: str = ""
    renderer_version: str = "h0_react_render_v2"
    observation_visibility_policy: str = "raw_full_truncated_transport"

    def digest(self) -> str:
        return digest_of(
            {
                "profile_id": self.profile_id,
                "instruction_text": self.instruction_text,
                "environment_doc_text": self.environment_doc_text,
                "renderer_version": self.renderer_version,
                "observation_visibility_policy": self.observation_visibility_policy,
            }
        )

    def to_json(self) -> dict[str, JSONValue]:
        return {
            "profile_id": self.profile_id,
            "instruction_text": self.instruction_text,
            "environment_doc_text": self.environment_doc_text,
            "renderer_version": self.renderer_version,
            "observation_visibility_policy": self.observation_visibility_policy,
            "profile_digest": self.digest(),
        }


def load_profile(profile_id: str, *, environment_doc_text: str = "") -> H0Profile:
    """Load a fixed profile by id; unknown/missing ids are hard errors."""
    fname = PROFILE_FILES.get(profile_id)
    if fname is None:
        raise ProtocolViolation(
            f"unknown H0 profile {profile_id!r} (known: {sorted(PROFILE_FILES)}) — "
            "configs must declare runtime.profile explicitly; there is no silent "
            "fallback to the legacy base"
        )
    path = PROMPTS_DIR / fname
    if not path.is_file():
        raise ProtocolViolation(f"H0 profile file missing: {path}")
    text = path.read_text("utf-8")
    # strip the trailing HTML comment block (documentation, not prompt content)
    if "<!--" in text:
        text = text.split("<!--", 1)[0]
    return H0Profile(
        profile_id=profile_id,
        instruction_text=text.strip(),
        environment_doc_text=environment_doc_text,
    )


def profile_from_runtime_config(runtime_cfg: dict) -> H0Profile:
    """Build the profile from the ``runtime`` config section.

    ``runtime.profile`` is REQUIRED (guide §5.2); ``runtime.environment_doc``
    optionally supplies per-environment interface text bundled with the
    profile identity.
    """
    profile_id = runtime_cfg.get("profile")
    if not profile_id:
        raise ProtocolViolation(
            "runtime.profile is REQUIRED (react_zero_shot_v1 | act_only_v1) — "
            "a config without an explicit profile must not run: a missing key "
            "would silently pick the old base and corrupt protocol identity"
        )
    env_doc = str(runtime_cfg.get("environment_doc", "") or "")
    if env_doc and env_doc.endswith(".md") and not "\n" in env_doc:
        # The config declares a PATH (configs/env_docs/<domain>_v1.md), not
        # inline text. Before the fix the path string itself went into the
        # header — the doc content never reached the model (E0/E1 latent
        # defect: appworld had a 1.9KB doc "present" as one filename line).
        # Repo-relative paths resolve against CWD (entries cd to RSI_ROOT);
        # absolute paths pass through. A missing file is a hard error.
        from pathlib import Path as _Path

        _p = _Path(env_doc)
        if not _p.is_file():
            raise ProtocolViolation(
                f"runtime.environment_doc points to a missing file: {env_doc}"
            )
        env_doc = _p.read_text("utf-8")
    elif env_doc and "\n" not in env_doc and not env_doc.endswith(".md"):
        raise ProtocolViolation(
            f"runtime.environment_doc is a bare string that is neither a .md "
            f"path nor multi-line text: {env_doc[:60]!r} — refusing to render "
            "a likely-path-as-text into the header"
        )
    return load_profile(str(profile_id), environment_doc_text=env_doc)
