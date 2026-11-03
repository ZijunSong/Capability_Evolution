#!/usr/bin/env bash
# ALFWorld prepare + real smoke。dev 容器 JuiceFS 元数据遍历(1.4 万小文件 rglob+
# sha256)在高负载下过慢(25min 仅 2s CPU)——放集群节点跑。
# 数据已就位: /mnt/songzijun/datasets/alfworld/json_2.1.1(ModelScope 镜像
# Ecollzhang/alfworld-json-2.1.1, 官方 3-split: train/valid_seen/valid_unseen)。
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH=/mnt/songzijun/Capability_Evolution/RSI/src:${PYTHONPATH:-}

echo "== install smoke deps =="
python -m pip install -q --ignore-installed \
  -i https://pkgs.d.xiaomi.net/artifactory/api/pypi/pypi-virtual/simple \
  textworld termcolor tqdm || true
python -c "import textworld, textworld.gym; print('textworld ok')"

echo "== prepare_alfworld_data (skip zips; tree already extracted) =="
python /mnt/songzijun/Capability_Evolution/RSI/scripts/prepare_alfworld_data.py \
  --root /mnt/songzijun/datasets/alfworld \
  --src /mnt/songzijun/datasets/alfworld_src \
  --skip-download --skip-install

echo "== real smoke: provider -> adapter episode -> scorer =="
python - <<'PYEOF'
import sys
sys.path.insert(0, "/mnt/songzijun/Capability_Evolution/RSI/src")
from rsi.environments.registry import global_registry
from rsi.contracts.action import ToolCall

reg = global_registry()
ok, why = reg.probe_executable("alfworld")
print("probe:", ok, why)
assert ok, why
adapter = reg.get("alfworld").adapter_factory()
provider = reg.get("alfworld").task_provider_factory()
scorer = reg.get("alfworld").scorer_factory()
final = provider.public_tasks("final_test")
work = provider.public_tasks("work_train")
print("n_final:", len(final), "n_work_train:", len(work))
assert final and work
task = work[0]
obs = adapter.reset(task, seed=1)
print("reset_ok:", obs.content[:80].replace("\n", " "))
res = adapter.step(ToolCall(name="command", arguments={"text": "look"}))
print("step_ok:", res.observation.ok)
snap = adapter.snapshot()
fork = adapter.fork(snap)
print("fork_digest_match:", fork.state_digest() == adapter.state_digest())
fork.close()
art = adapter.final_artifact()
rec = scorer.evaluate(art, provider.private_ref(task.task_id))
print("score:", rec.metric_name, rec.value, "success:", rec.success, "synthetic:", rec.synthetic)
assert rec.metric_name == "task_success" and rec.value in (0.0, 1.0) and not rec.synthetic
print("ALFWORLD_SMOKE_OK")
PYEOF

echo "PREPARE_ALFWORLD_DONE"
