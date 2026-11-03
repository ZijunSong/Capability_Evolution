#!/usr/bin/env bash
# WebShop data fetch + prepare job: Google Drive 不可达(开发机 gdown 抓不到公开链接
# / drive.usercontent 直连 hang)，借集群出口跑官方 gdown 下载 + 索引构建 + 目标冻结。
# 落地 /mnt/songzijun/datasets/webshop (完整官方 project tree)。
set -euo pipefail
DEST=/mnt/songzijun/datasets/webshop
SRC=/mnt/songzijun/datasets/webshop_src
export REQUESTS_CA_BUNDLE=${REQUESTS_CA_BUNDLE:-/etc/ssl/certs/ca-certificates.crt}
export PYTHONUNBUFFERED=1
export PYTHONPATH=/mnt/songzijun/Capability_Evolution/RSI/src:${PYTHONPATH:-}

echo "== install prepare deps =="
# gdown 单独装且硬失败; 其余走 --ignore-installed(镜像 distutils blinker 1.4
# 冲突会让整批 install 中止 —— 已知坑)
python -m pip install -q -i https://pkgs.d.xiaomi.net/artifactory/api/pypi/pypi-virtual/simple gdown
python -m pip install -q --ignore-installed \
  -i https://pkgs.d.xiaomi.net/artifactory/api/pypi/pypi-virtual/simple \
  beautifulsoup4 thefuzz spacy flask "pyserini>=2.0,<2.4" || true
python -c "import gdown, bs4, thefuzz, spacy; print('deps ok')"
# 官方 get_reward 需要 en_core_web_sm; index 构建需要 Java
if ! command -v java >/dev/null 2>&1; then
  apt-get update -qq && apt-get install -y -qq default-jdk-headless || true
fi
java -version 2>&1 | head -1
# pyjnius 起 JVM 需要显式 libjvm 路径（已知坑）；pyserini>=2.4 需要 Java 21+
if [ -z "${JVM_PATH:-}" ]; then
  JVM_SO=$(find /usr/lib/jvm -path "*java-21*" -name libjvm.so 2>/dev/null | head -1 || true)
  if [ -z "$JVM_SO" ]; then JVM_SO=$(find /usr/lib/jvm -name libjvm.so 2>/dev/null | head -1 || true); fi
  if [ -n "$JVM_SO" ]; then export JVM_PATH="$JVM_SO"; echo "JVM_PATH=$JVM_PATH"; fi
fi

echo "== prepare_webshop_data --flavor small -> $DEST =="
python /mnt/songzijun/Capability_Evolution/RSI/scripts/prepare_webshop_data.py \
  --root "$DEST" --src "$SRC" --flavor small

echo "== artifacts =="
ls -la "$DEST" | head -12
ls -la "$DEST/data" | head -8
echo "FETCH_WEBSHOP_DONE"
