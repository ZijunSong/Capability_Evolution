# SCOPE ICLR Experiments

统一消融 / 对比实验框架（对应 `0804-todo3.md`）。

## 布局

```text
experiments/
├── registry.yaml              # 全部 experiment_id
├── EXPERIMENT_MATRIX.md       # 状态矩阵
├── IMPLEMENTATION_INVENTORY.md
├── common/                    # Spec / launcher / resume / validation
├── ablations/                 # A1–A15 configs / builders / runners
├── baselines/                 # B0–B6 + SEED/OPID/SDAR adapters
└── schemas/                   # JSON schemas
```

## 快速开始

```bash
# 环境
conda activate bishop
cd SCOPE

# 预检
bash scripts/iclr/preflight.sh

# 单元测试
bash scripts/iclr/run_unit_tests.sh

# 单实验 smoke
python -m experiments.common.launcher \
  --experiment-id a1_same_state_on_policy \
  --smoke-query-limit 4 --resume

# dry-run
python -m experiments.common.launcher --experiment-id b_seed_dryrun --dry-run

# 外部 baseline clone（需可用 GitHub 网络）
bash scripts/iclr/clone_external_baselines.sh
```

## 输出规范

```text
outputs/iclr_ablations/<group>/<variant>/seed_<seed>/
outputs/iclr_baselines/<group>/<variant>/seed_<seed>/
outputs/iclr_readiness/
```

`DONE` 仅在 schema 校验通过后写入。禁止覆盖 `outputs/scope_round2`–`round9`。

## 约束

- 禁止静默 fallback
- 外部仓库只放 `external/baselines/`，适配在 `experiments/baselines/adapters/`
- 本阶段不跑 100q/830q 全量训练
