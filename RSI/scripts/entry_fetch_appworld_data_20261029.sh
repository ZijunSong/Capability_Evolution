#!/usr/bin/env bash
# AppWorld data fetch job: S3 从开发机不可达，借集群出口拉官方 data-0.1.0.bundle
set -euo pipefail
DEST=/mnt/songzijun/datasets/appworld
python -m pip install -q -i https://pkgs.d.xiaomi.net/artifactory/api/pypi/pypi-virtual/simple "appworld==0.1.3.post1"
python -m appworld.cli install
echo "== appworld download data -> $DEST =="
python -m appworld.cli download data --root "$DEST"
ls -la "$DEST/data" | head -8
echo "FETCH_APPWORLD_DONE"
