#!/usr/bin/env bash
# CloudML 多节点环境注入探针（CPU-only，快速，不烧 GPU）
# 目的：探明 framework=pytorch + nodeNumber=2 时 CloudML 是否注入 RANK/WORLD_SIZE/
# NODE_RANK/MASTER_ADDR/MASTER_PORT，或是否自带 torchrun 启动器，以及是否每节点各起一份进程。
set -euo pipefail
echo "=== [mnode-probe] host=$(hostname) pid=$$ ts=$(date -Iseconds) ==="
echo "--- distributed-relevant env ---"
env | grep -iE 'NODE|RANK|WORLD|MASTER|RDZV|CML_|NNODE|NCCL|GROUP|LOCAL_RANK|CUDA_VIS|PYTHONPATH' | sort || echo "(none matched)"
echo "--- host identity ---"
echo "hostname -I=$(hostname -I 2>/dev/null || true)"
echo "--- python dist env probe ---"
python3 - <<'PY' 2>/dev/null || python3 <<'PY2'
import os
keys=["NNODES","NODE_RANK","NODE_INDEX","NODE_NUMBER","RANK","LOCAL_RANK",
      "WORLD_SIZE","MASTER_ADDR","MASTER_PORT","CML_NODE_INDEX","CML_NODE_NUMBER",
      "CML_TASK_INDEX","CML_FRAMEWORK","GROUP_WORLD","RDZV_BACKEND","RDZV_ENDPOINT",
      "PMI_RANK","PMI_SIZE","OMPI_COMM_WORLD_RANK"]
pres={k:os.environ.get(k) for k in keys if os.environ.get(k) is not None}
print("PRESENT:", pres if pres else "{} (CloudML pytorch multi-node does NOT inject torchrun-style env)")
PY
exit 0
PY2
echo "(python probe skipped)"
exit 0
