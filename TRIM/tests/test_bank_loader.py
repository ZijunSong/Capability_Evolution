import json
from pathlib import Path

import pytest

from trim.harness_bank.loader import BankLoadError, load_bank
from tests.rsi_support import write_bank


def test_empty_bank_loads():
    loaded = load_bank(Path(__file__).resolve().parents[1] / "harness_banks" / "empty")
    assert loaded.active_components == ()
    assert loaded.bank_sha256


def test_missing_factory_file(tmp_path: Path):
    root = write_bank(tmp_path / "bank", component_ids=["evo.query_repair"])
    (root / "components" / "query_repair.py").unlink()
    with pytest.raises(BankLoadError, match="missing"):
        load_bank(root)


def test_path_escape_rejected(tmp_path: Path):
    root = write_bank(tmp_path / "bank", component_ids=["evo.query_repair"])
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["components"][0]["factory"] = "../../etc/passwd:create"
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BankLoadError):
        load_bank(root)


def test_dependency_cycle(tmp_path: Path):
    root = write_bank(tmp_path / "bank", component_ids=["evo.a", "evo.b"])
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["components"][0]["depends_on"] = ["evo.b"]
    manifest["components"][1]["depends_on"] = ["evo.a"]
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BankLoadError, match="cycle"):
        load_bank(root)


def test_conflict_rejected(tmp_path: Path):
    root = write_bank(tmp_path / "bank", component_ids=["evo.a", "evo.b"])
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["components"][0]["conflicts_with"] = ["evo.b"]
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BankLoadError, match="conflicts"):
        load_bank(root)


def test_hash_changes_when_source_changes(tmp_path: Path):
    root = write_bank(tmp_path / "bank", component_ids=["evo.query_repair"])
    first = load_bank(root).bank_sha256
    source = root / "components" / "query_repair.py"
    source.write_text(source.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")
    second = load_bank(root).bank_sha256
    assert first != second
