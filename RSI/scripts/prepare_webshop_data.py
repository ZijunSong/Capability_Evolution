"""Prepare the WebShop dataset root + frozen goal manifest (resumable).

Produces the official WebShop project tree under ``RSI_WEBSHOP_ROOT`` (default
``/mnt/songzijun/datasets/webshop``) — the official package resolves data and
search-index paths relative to itself, so the dataset root *is* the project
root (mirrors the official ``setup.sh`` layout):

    web_agent_site/        official package (copied from the frozen source tree)
    search_engine/         official index scripts + built Lucene indexes
    data/                  product / attribute files (official Google Drive ids)
    frozen_goals.json      official get_goals output under a pinned RNG seed
    frozen_manifest.json   source digests, panel pins, source commit

Then installs the tree editable (``pip install -e --no-deps``; upstream ships no
install metadata) and writes the two frozen files consumed by
``rsi.environments.webshop_env`` / ``WebShopTaskProvider``.

Why freeze goals: the official goal construction draws random price bounds
(``get_human_goals``) and item prices (``generate_product_prices``) from the
unseeded global RNG — a bare env is not deterministic across processes.  This
script materializes the goals ONCE under ``FREEZE_SEED`` in the official
post-shuffle order (``random.seed(233); shuffle``), and the adapter pins
``SimServer.goals``/``product_prices`` to the frozen copies so the official
``get_reward`` definition evaluates exactly the frozen benchmark.

Requires: network (Google Drive via gdown), Java (pyserini Lucene indexing),
``pip install gdown`` and the ``webshop`` extra deps.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_ROOT = Path("/mnt/songzijun/datasets/webshop")
DEFAULT_SRC = Path("/mnt/songzijun/datasets/webshop_src")

# Official data ids from the upstream setup.sh.
GDOWN_IDS = {
    "small": {
        "items_shuffle_1000.json": "1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib",
        "items_ins_v2_1000.json": "1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu",
    },
    "all": {
        "items_shuffle.json": "1A2whVgOO0euk5O13n2iYDM0bQRkkRduB",
        "items_ins_v2.json": "1s2j6NgHljiZzQNL3veZaAiyW_qDEgBNi",
    },
}
HUMAN_ATTR = ("items_human_ins.json", "14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O")

FREEZE_SEED = 20261023   # RNG seed for price/goal construction (our freeze pin)
OFFICIAL_SHUFFLE_SEED = 233  # SimServer reseeds here before shuffling goals
PANEL = {"final_test_count": 500, "transfer_test_count": 250}

INDEX_FOR_FLAVOR = {"small": ("resources_1k", "indexes_1k"), "all": ("resources", "indexes")}
ITEMS_FOR_FLAVOR = {"small": "data/items_shuffle_1000.json", "all": "data/items_shuffle.json"}


def sh(cmd: list[str], cwd: Path | None = None, timeout: int = 7200) -> None:
    print(f"$ {' '.join(cmd)[:120]}", flush=True)
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)[:120]}\n{(proc.stderr or proc.stdout)[-800:]}")


def copy_tree(src: Path, dst: Path, *, skip_names: set[str]) -> None:
    if dst.exists():
        print(f"keep existing {dst}")
        return
    shutil.copytree(
        src, dst,
        ignore=lambda _dir, names: {n for n in names if n in skip_names or n.startswith(".") or n == "__pycache__"},
    )
    print(f"copied {src} -> {dst}")


def download_data(root: Path, flavor: str, skip: bool) -> None:
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    wanted = dict(GDOWN_IDS[flavor])
    wanted[HUMAN_ATTR[0]] = HUMAN_ATTR[1]
    for fname, fid in wanted.items():
        dest = data / fname
        if dest.is_file() and dest.stat().st_size > 0:
            print(f"have {dest}")
            continue
        if skip:
            raise RuntimeError(f"--skip-download set but {dest} missing")
        import gdown

        print(f"downloading {fname} (id {fid}) ...", flush=True)
        gdown.download(id=fid, output=str(dest), quiet=False)
        if not dest.is_file() or dest.stat().st_size == 0:
            raise RuntimeError(f"gdown failed for {fname} (id {fid})")


def rewrite_utils_for_flavor(root: Path, flavor: str) -> None:
    """README-sanctioned edit: point DEFAULT_* at the full data files."""
    if flavor != "all":
        return
    utils = root / "web_agent_site" / "utils.py"
    text = utils.read_text("utf-8")
    text = text.replace("items_ins_v2_1000.json", "items_ins_v2.json")
    text = text.replace("items_shuffle_1000.json", "items_shuffle.json")
    utils.write_text(text, "utf-8")
    print(f"rewrote DEFAULT_* paths in {utils} for flavor=all")


def build_index(root: Path, flavor: str, skip: bool) -> None:
    resources, indexes = INDEX_FOR_FLAVOR[flavor]
    if (root / "search_engine" / indexes).is_dir() and any((root / "search_engine" / indexes).iterdir()):
        print(f"have index {indexes}")
        return
    if skip:
        raise RuntimeError(f"--skip-index set but {indexes} missing")
    # pyjnius cannot locate libjvm on some images — pin it (known recovery).
    # pyserini>=2.4 passes --add-modules=jdk.incubator.vector /
    # --enable-native-access, so the JVM must be Java 21+ (Java 11 refuses to
    # start: JNI_CreateJavaVM returns -1).
    if "JVM_PATH" not in os.environ:
        candidates = sorted(Path("/usr/lib/jvm").rglob("libjvm.so"))
        candidates.sort(key=lambda p: ("java-21" not in str(p), str(p)))
        if candidates:
            os.environ["JVM_PATH"] = str(candidates[0])
            print(f"JVM_PATH={candidates[0]}")
    se = root / "search_engine"
    for d in ("resources", "resources_100", "resources_1k", "resources_100k"):
        (se / d).mkdir(exist_ok=True)
    # official conversion script (imports the package via sys.path in ITS own
    # script file — package code itself never mutates sys.path)
    sh([sys.executable, "convert_product_file_format.py"], cwd=se)
    sh(
        [
            sys.executable, "-m", "pyserini.index.lucene",
            "--collection", "JsonCollection",
            "--input", resources,
            "--index", indexes,
            "--generator", "DefaultLuceneDocumentGenerator",
            "--threads", "1",
            "--storePositions", "--storeDocvectors", "--storeRaw",
        ],
        cwd=se,
    )


def freeze_goals(root: Path, src: Path, flavor: str) -> dict:
    """Materialize the official goal list under the pinned RNG seed."""
    items_rel = ITEMS_FOR_FLAVOR[flavor]
    items_path = root / items_rel
    num_products = 1000 if flavor == "small" else None
    # run inside the project tree so `import web_agent_site` resolves there
    code = f"""
import json, random, sys
sys.path.insert(0, {str(root)!r})
random.seed({FREEZE_SEED})
from web_agent_site.engine.engine import load_products
from web_agent_site.engine.goal import get_goals
products, product_item_dict, product_prices, _ = load_products(
    filepath={str(items_path)!r}, num_products={num_products!r}, human_goals=True)
goals = get_goals(products, product_prices, human_goals=True)
random.seed({OFFICIAL_SHUFFLE_SEED})
random.shuffle(goals)
out = {{
    "schema_version": "rsi_webshop_goals_v1",
    "freeze_seed": {FREEZE_SEED},
    "official_shuffle_seed": {OFFICIAL_SHUFFLE_SEED},
    "goals": goals,
    "product_prices": product_prices,
}}
with open({str(root / 'frozen_goals.json')!r}, "w") as f:
    json.dump(out, f)
print("N_GOALS", len(goals))
"""
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=7200)
    if proc.returncode != 0:
        raise RuntimeError(f"goal freeze failed:\n{(proc.stderr or proc.stdout)[-1200:]}")
    for line in proc.stdout.splitlines():
        if line.startswith("N_GOALS"):
            print(line)
    return json.loads((root / "frozen_goals.json").read_text("utf-8"))


def install_tree(root: Path, skip: bool) -> None:
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        pyproject.write_text(
            "[build-system]\n"
            'requires = ["setuptools>=68"]\n'
            'build-backend = "setuptools.build_meta"\n\n'
            "[project]\n"
            'name = "webshop-official-env"\n'
            'version = "0.0.0"\n'
            'description = "Frozen official WebShop env tree (installed by RSI prepare)"\n'
            "[tool.setuptools.packages.find]\n"
            'include = ["web_agent_site*\"]\n',
            "utf-8",
        )
    if skip:
        print("skip pip install")
        return
    sh([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(root)], timeout=1800)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--src", default=str(DEFAULT_SRC), help="frozen webshop source tree")
    ap.add_argument("--flavor", choices=["small", "all"], default="small")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-index", action="store_true")
    ap.add_argument("--skip-install", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    src = Path(args.src).resolve()
    root.mkdir(parents=True, exist_ok=True)

    copy_tree(src / "web_agent_site", root / "web_agent_site", skip_names=set())
    copy_tree(src / "search_engine", root / "search_engine",
              skip_names={"resources", "resources_100", "resources_1k", "resources_100k",
                          "indexes", "indexes_100", "indexes_1k", "indexes_100k"})
    rewrite_utils_for_flavor(root, args.flavor)
    download_data(root, args.flavor, args.skip_download)

    # spacy model needed by the official get_reward (goal.py)
    probe = subprocess.run(
        [sys.executable, "-c", "import spacy; spacy.load('en_core_web_sm')"],
        capture_output=True, text=True, timeout=300,
    )
    if probe.returncode == 0:
        print("have spacy model en_core_web_sm")
    else:
        staged = sorted((root / "_deps").glob("en_core_web_sm-*.tgz")) if (root / "_deps").is_dir() else []
        if staged:
            # staged site-packages tarball (cluster egress cannot reach the
            # spacy model release CDN; prepared on the dev box)
            import site

            sp_dir = site.getsitepackages()[0]
            sh(["tar", "xzf", str(staged[-1]), "-C", sp_dir])
            print(f"installed spacy model from {staged[-1]}")
        else:
            sh([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], timeout=3600)

    build_index(root, args.flavor, args.skip_index)
    goals_payload = freeze_goals(root, src, args.flavor)

    from rsi.contracts.common import digest_of, digest_of_file

    commit = "unknown"
    try:
        proc = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=60)
        if proc.returncode == 0:
            commit = proc.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    goals_path = root / "frozen_goals.json"
    manifest = {
        "schema_version": "rsi_webshop_frozen_v1",
        "flavor": args.flavor,
        "num_products": 1000 if args.flavor == "small" else None,
        "items_file": ITEMS_FOR_FLAVOR[args.flavor],
        "goals_file": "frozen_goals.json",
        "goals_sha256": digest_of_file(str(goals_path)),
        "n_goals": len(goals_payload["goals"]),
        "product_prices_sha256": digest_of(goals_payload["product_prices"]),
        "freeze_seed": FREEZE_SEED,
        "official_shuffle_seed": OFFICIAL_SHUFFLE_SEED,
        "panel": PANEL,
        "webshop_src_commit": commit,
        # Provenance: official Google Drive ids (setup.sh) were unreachable from
        # this network; the identical files were pulled from public ModelScope
        # mirrors of the official artifacts (digests below are the audit identity).
        "data_source": {
            "official_origin": "https://github.com/princeton-nlp/WebShop setup.sh Google Drive ids",
            "mirrors_used": [
                "modelscope:brihug/WebShop (items_shuffle.json, items_shuffle_1000.json, items_ins_v2_1000.json, items_human_ins.json)",
                "modelscope:Yimingastro/WebShop (items_ins_v2.json)",
            ],
        },
        "source_files": {
            name: digest_of_file(str(root / "data" / name))
            for name in sorted(set(GDOWN_IDS[args.flavor]) | {HUMAN_ATTR[0]})
            if (root / "data" / name).is_file()
        },
    }
    (root / "frozen_manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True, ensure_ascii=False), "utf-8"
    )
    install_tree(root, args.skip_install)
    print(json.dumps({"ok": True, "root": str(root), "n_goals": manifest["n_goals"],
                      "panel": PANEL}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
