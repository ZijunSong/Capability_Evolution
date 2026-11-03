#!/usr/bin/env bash
# WebShop FULL flavor prepare + real smoke on a 224GB node.
# 开发机 user-container memcg ~16GB：items_shuffle.json(5.5GB) json.load 峰值 >16GB
# 会被 oom-kill（已实测）；全量 load_products/索引/目标冻结必须在大内存节点跑。
# 数据文件已在 JuiceFS（ModelScope 镜像下载，见 prepare_webshop_data.py 数据来源注记）。
set -euo pipefail
export REQUESTS_CA_BUNDLE=${REQUESTS_CA_BUNDLE:-/etc/ssl/certs/ca-certificates.crt}
export PYTHONUNBUFFERED=1
export PYTHONPATH=/mnt/songzijun/Capability_Evolution/RSI/src:${PYTHONPATH:-}
# 224GB 节点：放开 flavor=all 执行门（dev 容器 memcg 会 OOM，见 runbook）
export RSI_WEBSHOP_FULL_ALLOW=1

echo "== install prepare deps =="
# --ignore-installed: 镜像 distutils blinker 1.4 冲突会让整批 install 中止（已知坑）。
# cleantext 不在内网 PyPI → 从 _deps 站包 tgz 解到 site-packages（dev 打包）；
# emoji/unidecode 是 cleantext 的运行期依赖。gym 0.24 与 py3.12 不兼容故放宽 0.26。
python -m pip install -q -i https://pkgs.d.xiaomi.net/artifactory/api/pypi/pypi-virtual/simple gdown
python -m pip install -q --ignore-installed \
  -i https://pkgs.d.xiaomi.net/artifactory/api/pypi/pypi-virtual/simple \
  beautifulsoup4 thefuzz spacy flask "pyserini>=2.0,<2.4" \
  "rank_bm25==0.2.2" rich tqdm "gym==0.26.2" emoji unidecode nltk selenium || true
SP_DIR=$(python -c "import site; print(site.getsitepackages()[0])")
tar xzf /mnt/songzijun/datasets/webshop/_deps/cleantext-1.1.4-sitepkg.tgz -C "$SP_DIR"
# cleantext 模块级 import nltk(engine 只 import 不调用; nltk.corpus 懒加载,无需 corpus 下载)。
# selenium: web_agent_site.envs/__init__ 连带 import 浏览器版环境(仅 import 不驱动)。
python -c "import bs4, thefuzz, spacy, cleantext, rank_bm25, rich, gym, emoji, unidecode, nltk, selenium; print('deps ok')"

if [ -z "${JVM_PATH:-}" ]; then
  JVM_SO=$(find /usr/lib/jvm -path "*java-2[15]*" -name libjvm.so 2>/dev/null | head -1 || true)
  if [ -z "$JVM_SO" ]; then JVM_SO=$(find /usr/lib/jvm -name libjvm.so 2>/dev/null | head -1 || true); fi
  if [ -n "$JVM_SO" ]; then export JVM_PATH="$JVM_SO"; echo "JVM_PATH=$JVM_PATH"; fi
fi

echo "== prepare_webshop_data --flavor all --skip-download =="
python /mnt/songzijun/Capability_Evolution/RSI/scripts/prepare_webshop_data.py \
  --root /mnt/songzijun/datasets/webshop \
  --src /mnt/songzijun/datasets/webshop_src \
  --flavor all --skip-download

echo "== real smoke: provider -> adapter episode -> scorer =="
python - <<'PYEOF'
import json, sys
sys.path.insert(0, "/mnt/songzijun/Capability_Evolution/RSI/src")
from rsi.environments.registry import global_registry
from rsi.contracts.action import ToolCall, Finish

reg = global_registry()
ok, why = reg.probe_executable("webshop")
print("probe:", ok, why)
assert ok, why
adapter = reg.get("webshop").adapter_factory()
provider = reg.get("webshop").task_provider_factory()
scorer = reg.get("webshop").scorer_factory()
final = provider.public_tasks("final_test")
work = provider.public_tasks("work_train")
print("n_final:", len(final), "n_work_train:", len(work))
task = work[0]
obs = adapter.reset(task, seed=1)
print("reset_ok:", obs.content[:80].replace("\n", " "))
res = adapter.step(ToolCall(name="search", arguments={"query": task.instruction[:60]}))
print("search_ok:", res.observation.ok)
snap = adapter.snapshot()
fork = adapter.fork(snap)
print("fork_digest_match:", fork.state_digest() == adapter.state_digest())
fork.close()
art = adapter.final_artifact()
scorer.load_private(task.task_id, provider.private_record(task.task_id))
rec = scorer.evaluate(art, provider.private_ref(task.task_id))
print("score:", rec.metric_name, rec.value, "success:", rec.success, "synthetic:", rec.synthetic)
assert rec.metric_name == "task_reward" and 0.0 <= rec.value <= 1.0 and not rec.synthetic
print("WEBSHOP_SMOKE_OK")
PYEOF

echo "PREPARE_WEBSHOP_FULL_DONE"
