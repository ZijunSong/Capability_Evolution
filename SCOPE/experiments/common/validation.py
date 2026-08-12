"""Schema validation for experiment outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
SCHEMA_DIR = _REPO / "experiments" / "schemas"

EXPECTED_RUN_FILES = (
    "run_manifest.json",
    "resolved_config.yaml",
    "config_diff.json",
    "summary.json",
)


def _load_schema(name: str) -> dict[str, Any]:
    path = SCHEMA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"schema missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _check_required(obj: dict[str, Any], required: list[str], *, ctx: str) -> list[str]:
    return [f"{ctx}: missing {k}" for k in required if k not in obj]


def validate_against_schema(obj: dict[str, Any], schema_name: str) -> list[str]:
    """Minimal JSON-schema subset validator (required + type checks)."""
    schema = _load_schema(schema_name)
    errors = _check_required(obj, schema.get("required", []), ctx=schema_name)
    props = schema.get("properties", {})
    for key, prop in props.items():
        if key not in obj:
            continue
        expected = prop.get("type")
        val = obj[key]
        if expected == "object" and not isinstance(val, dict):
            errors.append(f"{schema_name}.{key}: expected object")
        elif expected == "array" and not isinstance(val, list):
            errors.append(f"{schema_name}.{key}: expected array")
        elif expected == "string" and not isinstance(val, str):
            errors.append(f"{schema_name}.{key}: expected string")
        elif expected == "number" and not isinstance(val, (int, float)):
            errors.append(f"{schema_name}.{key}: expected number")
        elif expected == "integer" and not isinstance(val, int):
            errors.append(f"{schema_name}.{key}: expected integer")
        elif expected == "boolean" and not isinstance(val, bool):
            errors.append(f"{schema_name}.{key}: expected boolean")
    return errors


def validate_run_dir(output_dir: Path, *, require_done: bool = False) -> list[str]:
    errors: list[str] = []
    if not output_dir.exists():
        return [f"output_dir missing: {output_dir}"]
    for name in EXPECTED_RUN_FILES:
        if not (output_dir / name).exists():
            errors.append(f"missing {name}")
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        errors.extend(validate_against_schema(summary, "summary.schema.json"))
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        errors.extend(validate_against_schema(manifest, "run_manifest.schema.json"))
    if require_done and not (output_dir / "DONE").exists():
        errors.append("DONE marker missing")
    if (output_dir / "DONE").exists() and errors:
        errors.append("DONE present but validation failed — marker is invalid")
    return errors


def maybe_write_done(output_dir: Path) -> bool:
    """Create DONE only if all expected files pass schema validation."""
    errors = validate_run_dir(output_dir, require_done=False)
    # DONE itself is not required yet
    errors = [e for e in errors if e != "DONE marker missing"]
    if errors:
        return False
    (output_dir / "DONE").write_text("ok\n", encoding="utf-8")
    return True
