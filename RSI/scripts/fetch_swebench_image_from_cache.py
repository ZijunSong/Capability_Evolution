#!/usr/bin/env python3
"""Fetch swebench eval images from the Artifactory dockerhub-remote cache.

Docker Hub is unreachable from this network and the Artifactory docker remote
does not expose a usable /v2/ endpoint for the docker CLI, but the *cache* of
previously pulled images is anonymously downloadable as artifacts
(``/artifactory/dockerhub-remote/<image>/...``).  This script reconstructs an
image from those artifacts into a docker-save tar and optionally ``docker load``s it.

Usage:
  python3 fetch_swebench_image_from_cache.py --image sweb.eval.x86_64.astropy_1776_astropy-12907 \
      --repo-tag swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest \
      --out-dir /mnt/songzijun/datasets/swebench/images [--load]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

BASE = "https://pkgs.d.xiaomi.net/artifactory/dockerhub-remote/swebench"


def fetch_to(url: str, dest: Path) -> str:
    """Stream-download to dest; return sha256 hex. Keeps memory flat."""
    h = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=300) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
            f.write(chunk)
    return h.hexdigest()


def verify_hex(got_hex: str, digest: str) -> None:
    algo, want = digest.split(":", 1)
    if algo != "sha256":
        raise ValueError(f"unsupported digest algo {algo}")
    if got_hex != want:
        raise ValueError(f"digest mismatch for {digest[:20]}…")


_SIBLINGS: list[str] | None = None


def sibling_images() -> list[str]:
    global _SIBLINGS
    if _SIBLINGS is None:
        listing = json.loads(urllib.request.urlopen(
            "https://pkgs.d.xiaomi.net/artifactory/api/storage/dockerhub-remote/swebench", timeout=60
        ).read())
        _SIBLINGS = [c["uri"].lstrip("/") for c in listing.get("children", []) if not c.get("folder") is False]
    return _SIBLINGS


def repack(image: str, repo_tag: str, out_dir: Path, load: bool) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{image}.tar"
    if out.is_file() and out.stat().st_size > 0:
        print(f"exists: {out}")
    else:
        manifest = json.loads(urllib.request.urlopen(f"{BASE}/{image}/latest/manifest.json", timeout=60).read())
        tmp = out.with_suffix(".partial")
        with tempfile.TemporaryDirectory(prefix="swebimg_") as td, tarfile.open(tmp, "w") as tf:
            tdir = Path(td)

            def fetch_blob(dig: str) -> Path:
                p = tdir / ("blob_" + dig.split(":")[1])
                h = dig.split(":")[1]
                urls = [f"{BASE}/{image}/latest/sha256__{h}"]
                # sibling fallback: layers are content-addressed and shared across
                # images; a partially-evicted cache entry can often be completed
                # from another image that shares the layer.
                for sib in sibling_images():
                    if sib != image:
                        urls.append(f"{BASE}/{sib}/latest/sha256__{h}")
                for u in urls:
                    try:
                        got = fetch_to(u, p)
                    except Exception:  # noqa: BLE001
                        continue
                    verify_hex(got, dig)
                    return p
                raise FileNotFoundError(f"blob {h[:16]}… not found for {image}")

            # config
            cfg_digest = manifest["config"]["digest"]
            cfg_path = fetch_blob(cfg_digest)
            cfg_name = cfg_digest.split(":")[1] + ".json"
            info = tarfile.TarInfo(cfg_name)
            info.size = cfg_path.stat().st_size
            with open(cfg_path, "rb") as f:
                tf.addfile(info, f)
            # layers (gunzip stream to layer.tar for docker-save compatibility)
            layers = []
            for i, layer in enumerate(manifest["layers"]):
                dig = layer["digest"]
                blob = fetch_blob(dig)
                layer_name = f"{i:05d}_{'_'.join(dig.split(':'))}"
                lpath = tdir / f"{layer_name}.tar"
                if layer.get("mediaType", "").endswith("gzip"):
                    with gzip.open(blob, "rb") as gz, open(lpath, "wb") as lf:
                        shutil.copyfileobj(gz, lf, 1 << 20)
                else:
                    lpath = blob
                info = tarfile.TarInfo(f"{layer_name}/layer.tar")
                info.size = lpath.stat().st_size
                with open(lpath, "rb") as lf:
                    tf.addfile(info, lf)
                layers.append(f"{layer_name}/layer.tar")
                lpath.unlink(missing_ok=True)
                blob.unlink(missing_ok=True)
            # docker-save manifest.json
            payload = json.dumps([{"Config": cfg_name, "RepoTags": [repo_tag], "Layers": layers}]).encode()
            info = tarfile.TarInfo("manifest.json")
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))
        tmp.rename(out)
        print(f"packed: {out} ({out.stat().st_size / 1e9:.2f} GB)")
    if load:
        subprocess.run(["docker", "load", "-i", str(out)], check=True)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="cache folder name, e.g. sweb.eval.x86_64.<spec>_<iid>")
    p.add_argument("--repo-tag", default=None, help="RepoTag to stamp (default swebench/<image>:latest)")
    p.add_argument("--out-dir", default="/mnt/songzijun/datasets/swebench/images")
    p.add_argument("--load", action="store_true")
    a = p.parse_args()
    tag = a.repo_tag or f"swebench/{a.image}:latest"
    repack(a.image, tag, Path(a.out_dir), a.load)
    return 0


if __name__ == "__main__":
    sys.exit(main())
