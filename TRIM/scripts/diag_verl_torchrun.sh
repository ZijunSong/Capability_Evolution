set -euxo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH=/mnt/songzijun/Capability_Evolution/TRIM:/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "===== DIAG 1: distributed env injected by CloudML PyTorchJob? ====="
env | grep -iE "^(RANK|WORLD_SIZE|LOCAL_RANK|LOCAL_WORLD_SIZE|GROUP_RANK|NODE_RANK|MASTER_ADDR|MASTER_PORT|NPROC|NNODES|PET_|SLURM)" | sort || echo "(no distributed env present)"

echo "===== DIAG 2: CUDA_VISIBLE_DEVICES ====="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>/dev/null | head -3 || echo "(nvidia-smi unavailable on this 1-GPU diag)"

echo "===== DIAG 3: parse_dist_argv + needs_torchrun with real argv ====="
python3 - <<'PY'
import sys, os
sys.path.insert(0, "/mnt/songzijun/Capability_Evolution/TRIM")
from trim.training.dist_runtime import parse_dist_argv, needs_torchrun, under_torchrun, training_backend_from_argv, visible_cuda_count, torchrun_cmd
raw = ["--training-backend","verl","--nproc-per-node","8","--harness","Harness-1","--benchmark","bcplus_full","--train_method","rl","--component","all","--train-steps","100","--max-turns","40","--max-model-len","8192","--out","/tmp/diag_out"]
cfg, train_argv = parse_dist_argv(raw)
backend = training_backend_from_argv(train_argv)
print("backend:", backend, "in VERL_BACKENDS:", backend in {"verl","fsdp2","verl_fsdp2"})
print("cfg.nproc_per_node:", cfg.nproc_per_node)
print("cfg.nnodes:", cfg.nnodes)
print("cfg.already_worker:", cfg.already_worker)
print("under_torchrun():", under_torchrun())
print("needs_torchrun(cfg):", needs_torchrun(cfg))
print("visible_cuda_count():", visible_cuda_count())
cmd = torchrun_cmd(script="/mnt/songzijun/Capability_Evolution/TRIM/scripts/run_train.py", train_argv=train_argv, cfg=cfg, node_rank=0)
print("torchrun_cmd:", " ".join(cmd))
print("torchrun module importable:", __import__("importlib").util.find_spec("torch.distributed.run") is not None)
PY

echo "===== DIAG 4: actually exec torchrun --nproc_per_node 8 with a trivial script (proves execvpe + torchrun work) ====="
cat > /tmp/diag_worker.py <<'PY'
import os, sys
print(f"[diag worker] rank={os.environ.get('RANK')} world_size={os.environ.get('WORLD_SIZE')} local_rank={os.environ.get('LOCAL_RANK')} pid={os.getpid()}", flush=True)
PY
python3 -m torch.distributed.run --nnodes 1 --nproc_per_node 8 --max_restarts 0 --standalone /tmp/diag_worker.py 2>&1 | head -20 || echo "TORCHRUN EXITED NONZERO (this is the smoking gun)"
echo "===== DIAG DONE ====="
