# 三组 SFT 实验

三组实验各自有入口，不会注册成 `trim`、`only-opd` 或旧 `opd` 的别名。

| 实验 | 入口 | 数据 |
|---|---|---|
| Teacher SFT | `scripts/run_teacher_sft_multinode.py` | 固定 Full 教师轨迹上的原生动作 |
| Projected Teacher SFT | `scripts/run_projected_teacher_sft_multinode.py` | 同一教师轨迹在 Zero 环境里投影重放 |
| Student on-policy SFT | `scripts/run_student_onpolicy_sft_multinode.py` | 当前学生 Zero 状态上的 TRIM 合法动作 |

损失都是 action-only CE。没有 RL、advantage、teacher log-prob、gap 或 sigmoid gate。

## 32 卡布局

和 zero-rl / all-trim 一样：4 个节点，每节点 1 个协调进程，8 个 rollout replica，tensor parallel 1。一共 32 张卡。只有 rank 0 做梯度更新，另外 3 个协调进程只采样。

不传 `--nnodes` 时默认就是这个布局。本机试跑要显式写成 `--nnodes 1 --rollout-replicas 1`。

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=TRIM:SCAPE-EasyOPD
python TRIM/scripts/run_teacher_sft_multinode.py \
  --nnodes 4 --nproc-per-node 1 --node-rank "$NODE_RANK" \
  --master-addr "$MASTER_ADDR" --master-port 29611 \
  --rollout-replicas 8 --tensor-parallel-size 1 \
  --model_name /models/base-model \
  --init-adapter /checkpoints/common-init \
  --teacher-model /models/base-model \
  --teacher-adapter /checkpoints/common-init \
  --teacher-component all --student-component zero \
  --train-data sec --rl-data /data/sec/rl.jsonl \
  --sec-corpus-root /data/sec \
  --train-env upstream \
  --stage all --num-updates 100 --batch-size 128 \
  --learning-rate 5e-6 --lora-rank 32 \
  --max-turns 40 --max-model-len 8192 --max-new-tokens 2048 \
  --seed 42 --out /shared/trim_sft/teacher/seed42
```

Projected Teacher 把脚本换成 `run_projected_teacher_sft_multinode.py`，端口 `29612`。要和 Teacher SFT 共用同一份只读教师轨迹时，加上 `--teacher-traces /shared/trim_sft/teacher/seed42/teacher_archive`。

Student on-policy 使用 `run_student_onpolicy_sft_multinode.py`，端口 `29613`，不要传 `--teacher-model`、`--teacher-traces` 或 `--sft-data`。它只接受 `--stage all`。

`--dry-run` 在启动 torchrun、CUDA 和模型之前打印实验身份和 32 卡拓扑。

评测沿用 `scripts/run_eval.py`，三组都用 `--component zero` 和各自的 `adapters/final`。
