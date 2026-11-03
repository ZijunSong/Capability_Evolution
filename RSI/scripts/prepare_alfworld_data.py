"""Prepare the ALFWorld dataset root + frozen manifest (resumable).

Produces the official ``$ALFWORLD_DATA``-style layout under ``RSI_ALFWORLD_ROOT``
(default ``/mnt/songzijun/datasets/alfworld``):

    json_2.1.1/<split>/<task>/trial_*/{traj_data.json, game.tw-pddl}
    logic/{alfred.pddl, alfred.twl2}
    frozen_manifest.json

The official ``AlfredTWEnv.collect_game_files`` requires ``traj_data.json`` and
``game.tw-pddl`` co-located in one directory; the release zips ship them in
separate trees (``json_2.1.1_json.zip`` / ``json_2.1.3_tw-pddl.zip``), so this
script co-locates them.  The ``alfworld`` package is installed editable from
the frozen source tree (``pip install -e --no-deps``).

Then writes ``frozen_manifest.json`` (per-split game counts, source commit, zip
digests) consumed by ``AlfWorldTaskProvider``.

Requires: network (GitHub releases).  Idempotent: existing downloads/copies are
reused; pass ``--force`` to redo co-location and the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

DEFAULT_ROOT = Path("/mnt/songzijun/datasets/alfworld")
DEFAULT_SRC = Path("/mnt/songzijun/datasets/alfworld_src")

ZIPS = {
    "json_2.1.1_json.zip": "https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_json.zip",
    "json_2.1.1_pddl.zip": "https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_pddl.zip",
    "json_2.1.3_tw-pddl.zip": "https://github.com/alfworld/alfworld/releases/download/0.4.2/json_2.1.3_tw-pddl.zip",
}

SPLITS = ("train", "valid_seen", "valid_unseen", "test_unseen", "out_of_distribution")


def sh(cmd: list[str], cwd: Path | None = None, timeout: int = 3600) -> None:
    print(f"$ {' '.join(cmd)[:120]}", flush=True)
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)[:120]}\n{(proc.stderr or proc.stdout)[-800:]}")


def download(url: str, dest: Path) -> None:
    def remote_size() -> int:
        req = urllib.request.Request(url, headers={"User-Agent": "rsi-prepare"}, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as resp:
            return int(resp.headers.get("Content-Length") or 0)

    expected = remote_size()
    if dest.is_file() and dest.stat().st_size > 0 and (not expected or dest.stat().st_size == expected):
        print(f"have {dest.name} ({dest.stat().st_size // (1 << 20)} MB)")
        return
    if dest.is_file():
        print(f"partial {dest.name}: {dest.stat().st_size} != {expected} — re-downloading")
        dest.unlink()
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url} ...", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "rsi-prepare"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as f:
        shutil.copyfileobj(resp, f, length=1 << 20)
    if expected and tmp.stat().st_size != expected:
        raise RuntimeError(f"truncated download for {dest.name}: {tmp.stat().st_size} != {expected}")
    tmp.rename(dest)
    print(f"done {dest.name} ({dest.stat().st_size // (1 << 20)} MB)")


def unzip(zip_path: Path, dst: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        skipped = 0
        for name in names:
            target = dst / name
            if target.exists():
                skipped += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if name.endswith("/"):
                continue
            with zf.open(name) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
        print(f"extracted {zip_path.name}: {len(names) - skipped} files ({skipped} skipped)")


def co_locate_games(root: Path) -> int:
    """Place each ``game.tw-pddl`` next to its ``traj_data.json``.

    Handles both observed release layouts: the tw-pddl zip either mirrors the
    ``json_2.1.1/<split>/...`` paths or extracts under ``json_2.1.3_tw-pddl/``.
    Skipped entirely when the extracted tree already carries games (JuiceFS
    metadata crawls are expensive — avoid a full walk we do not need).
    """
    extracted = root / "json_2.1.1"
    if extracted.is_dir():
        sample = next(extracted.rglob("game.tw-pddl"), None)
        if sample is not None and (sample.parent / "traj_data.json").is_file():
            print("games already co-located in extracted tree")
            return 0
    n = 0
    candidates = [root / "json_2.1.3_tw-pddl", root / "json_2.1.1"]
    for base in candidates:
        if not base.is_dir():
            continue
        for game in base.rglob("game.tw-pddl"):
            rel = game.relative_to(base)
            target = root / "json_2.1.1" / rel.parent
            if not (target / "traj_data.json").is_file():
                continue
            dst = target / "game.tw-pddl"
            if dst.is_file():
                continue
            shutil.copy2(game, dst)
            n += 1
    print(f"co-located {n} game.tw-pddl files")
    return n


def copy_logic(root: Path, src: Path) -> None:
    logic = root / "logic"
    logic.mkdir(parents=True, exist_ok=True)
    for fname in ("alfred.pddl", "alfred.twl2"):
        dst = logic / fname
        if dst.is_file():
            continue
        candidates = [src / "alfworld" / "data" / fname, root / "logic" / fname]
        for c in candidates:
            if c.is_file() and c != dst:
                shutil.copy2(c, dst)
                print(f"copied {c} -> {dst}")
                break
        else:
            raise RuntimeError(f"logic file {fname} not found under {src}/alfworld/data")


def install_package(src: Path, skip: bool) -> None:
    if skip:
        print("skip pip install")
        return
    sh([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(src)], timeout=1800)


def scan_tree(root: Path, workers: int = 48) -> tuple[dict[str, int], str]:
    """Parallel scan: per-split counts + tree digest.

    JuiceFS small-file latency dominates; a single-threaded rglob of the full
    tree takes tens of minutes.  Instead: shallow-list task dirs per split,
    then a thread pool over task dirs does the (stat + sha256) work.  Digest
    input is the sorted (relpath, sha256) stream of each game dir's
    ``traj_data.json`` + ``game.tw-pddl`` — the audit identity.
    """
    from concurrent.futures import ThreadPoolExecutor

    base = root / "json_2.1.1"

    def scan_task(task_dir: Path) -> list[tuple[str, str, str]]:
        out: list[tuple[str, str, str]] = []
        split = task_dir.parent.name
        try:
            trials = [p for p in task_dir.iterdir() if p.is_dir()]
        except OSError:
            return out
        for trial in trials:
            rel = trial.relative_to(base)
            if "movable" in str(rel) or "Sliced" in str(rel):
                continue
            traj, game = trial / "traj_data.json", trial / "game.tw-pddl"
            try:
                if not (traj.is_file() and game.is_file()):
                    continue
            except OSError:
                continue

            def sha(p: Path) -> str:
                fh = hashlib.sha256()
                with open(p, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        fh.update(chunk)
                return fh.hexdigest()

            out.append(
                (
                    split,
                    f"{rel}/traj_data.json:{sha(traj)}\n",
                    f"{rel}/game.tw-pddl:{sha(game)}\n",
                )
            )
        return out

    task_dirs: list[Path] = []
    for split_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        task_dirs.extend(sorted(p for p in split_dir.iterdir() if p.is_dir()))
    print(f"scanning {len(task_dirs)} task dirs with {workers} threads", flush=True)

    counts: dict[str, int] = {}
    lines: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for res in pool.map(scan_task, task_dirs):
            for split, l1, l2 in res:
                counts[split] = counts.get(split, 0) + 1
                lines.append(l1)
                lines.append(l2)
    lines.sort()
    h = hashlib.sha256()
    for line in lines:
        h.update(line.encode())
    return counts, h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--src", default=str(DEFAULT_SRC), help="frozen alfworld source tree")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-install", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    src = Path(args.src).resolve()
    root.mkdir(parents=True, exist_ok=True)
    dl = root / "_downloads"
    dl.mkdir(exist_ok=True)

    extracted = root / "json_2.1.1"
    have_extracted = extracted.is_dir() and any(extracted.rglob("traj_data.json"))
    if have_extracted and args.skip_download:
        print(f"using already-extracted data under {extracted} (skip zip stage)")
    else:
        for fname, url in ZIPS.items():
            zpath = dl / fname
            if args.skip_download and not zpath.is_file():
                raise RuntimeError(f"--skip-download set but {zpath} missing")
            if not args.skip_download:
                download(url, zpath)
            unzip(zpath, root)

    co_locate_games(root)
    copy_logic(root, src)

    counts, tree_sha = scan_tree(root)
    if not counts.get("train"):
        raise RuntimeError(f"prepared tree incomplete: split counts {counts}")
    print(f"split_counts: {counts}")

    commit = "unknown"
    try:
        proc = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=60)
        if proc.returncode == 0:
            commit = proc.stdout.strip()
    except Exception:  # noqa: BLE001
        pass

    from rsi.contracts.common import digest_of_file

    manifest = {
        "schema_version": "rsi_alfworld_frozen_v1",
        "alfworld_src_commit": commit,
        "split_counts": counts,
        "tree_digest": tree_sha,
        "data_source": {
            "official_origin": "https://github.com/alfworld/alfworld releases 0.2.2/0.4.2 (alfworld-download zips)",
            "mirror_used": (
                "modelscope:Ecollzhang/alfworld-json-2.1.1"
                if have_extracted and not (dl / "json_2.1.1_json.zip").is_file()
                else "official release zips"
            ),
        },
        "source_zips": {
            fname: digest_of_file(str(dl / fname)) for fname in ZIPS if (dl / fname).is_file()
        },
        "transfer_split_preference": ["out_of_distribution", "valid_seen", "valid_train"],
    }
    (root / "frozen_manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True, ensure_ascii=False), "utf-8"
    )
    install_package(src, args.skip_install)
    print(json.dumps({"ok": True, "root": str(root), "split_counts": counts}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
