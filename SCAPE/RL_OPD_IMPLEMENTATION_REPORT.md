# RL + SR-OPD implementation report

## 1. 旧 RL+OPD 实际做了什么

`SCOPE/training/opd/trainer.py` 对 **同一条 Student action** 做 token NLL / teacher-advantage KL，再单独 `train_step`。这是第二次 optimizer step，不是 CISPO 联合更新。标签：`legacy_rl_plus_tool_kl`，`protocol_complete_rl_opd=false`。

## 2. 被 deprecated 的 hook

- `compute_opd_loss` / `teacher_advantage_weights` 不当作 CISPO advantage
- 旧 `OPDTransition` 不再是 universal target
- `tool_token_kl` 不是 canonical auxiliary loss
- `run_opd_hf_save.py` 不是 RL+OPD

## 3. Canonical RL path

`SCOPE/training/train_rl.py` → Tinker `train.main`，`LOSS_FN=cispo`，`NUM_SUBSTEPS=4`，native `KL_PENALTY_COEF`。Hybrid 不改写 CISPO 公式。

## 4. Canonical SR-OPD path

与 PURE 共用 `StudentActionSpaceProjector` + `sr_opd_ce`。唯一差别：state 来自当前 Reduced-Harness Student on-policy rollout。

## 5. Joint optimizer contract

```text
forward_backward(rl, loss_fn=cispo)
forward_backward(opd, loss_fn=cross_entropy)
optim_step()          # 每个 hybrid substep 恰好一次
```

`lambda_opd` 只进入 OPD token weights，不二次相乘，也不写入 environment reward。

## 6. 四个角色

| 变量 | 角色 |
|---|---|
| `rollout_policy` | 生成本轮 H_min trajectory |
| `train_policy` | 当前反传的 π_θ，update 开始时等于 rollout |
| `harness_teacher_policy` | 同版本 + H_full 的 side branch |
| `kl_reference_policy` | 原 RL KL，不是 Teacher |

## 7. On-policy decision snapshot

`StudentDecisionPoint`：`pre_action_snapshot` 是 Student 在该 decision 前的 reduced state。Observer 只读，禁止改 WM / reward / observation。

## 8. Tinker OPD datum

`TinkerOPDDatum`：`model_input` = Student prefix；`target_tokens` 含 prompt 占位；prompt weight=0。禁止 Teacher verify / graph / importance 进入 `model_input`。

## 9. lambda normalization

`opd_weight_normalization=per_optimizer_substep_token_mean`  
`weight = λ * confidence / sum(confidence * |M|)`  
`sum(supervised weights) = λ`（v1 confidence=1）。

## 10. Constant-reward groups

Decision 提取发生在 `remove_constant_reward_groups` 之前。RL datums 可被丢弃；OPD 仍可更新，`update_type=opd_only_zero_rl_signal`。

## 11. Policy version

Sync only。`rollout == train == teacher`，否则 `PolicyVersionMismatch`。OPD 不跨 version replay。每轮 update 后 `HybridLoopState` 升 version，下一轮 rollout 必须用新 version。

## 12. No-leakage

Teacher `VERIFY_RESULT_SECRET` 不得出现在 Student RL prefix 或 OPD `model_input`。MACRO 第二步 prefix 来自 Student `review_docs` shadow。

## 13. Unit tests

`tests/test_rl_opd_{joint_step,policy_version,tinker_datum,dataflow,no_leakage}.py`

## 14. Real Harness-1 smoke

K=2 扩档已在 H20 GPU4 跑通：`SCAPE/outputs/rl_opd_four_cell_k2_0822/FOUR_CELL_SUMMARY.json`

- 6 query × group_size=2 × 2 turn；本地工具执行；四格同一 θ₀
- 组内 reward 有方差（4–6/6 groups variable）
- RL+OPD 正式 joint：`cispo FB(16) → CE FB(23) → 1 OPT`，`update_type=rl_opd_joint`
- projection_coverage=1.0，23 MACRO `curate`，overlap=0.35
- Teacher 未改 RL reward；leak=0

1-turn 旧 smoke 仍在 `outputs/rl_opd_four_cell_smoke_0822/`（constant-group skip）。未跑 Tinker `train.main`。

## 15. RL-only parity

`lambda_opd=0` 不调用 Teacher / projector / OPD FB。Native 训练仍走 `train_rl.py`。未做与 cookbook `train.main` 的逐 batch 数值 parity（需要 Tinker 集群）。

## 16. PURE parity

同一 `ProjectedTrainingStep` 在 HF `sr_opd_ce` 与 Tinker CE datum 上共享 tokenization / mask / 归一化语义。未要求浮点逐位一致。

## 17. 四格初步结果

K=2 Harmony + gpt-oss-20b（n=6，H_min，1 train step）：

| cell | legal | mean_reward | gold recall | RL FB | OPD FB | OPT |
|---|---:|---:|---:|---:|---:|---:|
| Before | 1.00 | 0.512 | 0.167 | 0 | 0 | 0 |
| RL | 1.00 | 0.711 | 0.500 | 1 | 0 | 1 |
| PURE | 0.83 | 0.604 | 0.333 | 0 | 1 | 1 |
| RL+OPD | 1.00 | 0.714 | 0.500 | 1 | 1 | 1 |

RL / RL+OPD 比 Before 更高的 curated gold recall。PURE 出现一次非法 `verify_corpus_item`。仍是 n=6 smoke，不能当论文数字。

## 18. 尚未支持

async off-policy、跨 version OPD replay、window-slicing 去重、PCGrad、现场 Tinker 累积 contract test（若本地 Tinker 不累积梯度，再走 custom CISPO+CE）。

## 19. 默认全量参数

启动器默认是全量四格 / hybrid，不再是 smoke。`--smoke` 才切回调试档。

| 脚本 | 默认 | `--smoke` |
|---|---|---|
| `scripts/run_true_scape_rl_opd.py` | batch 32 / group 8 / 4 substeps / 6 turns / 64 steps / λ=0.1 / 3 OPD states | 4 / 2 / 1 / 2 / 2 |
| `scripts/run_gptoss_four_cell_smoke.py` | 64 queries / group 8 / 6 turns / 8 train steps / 384 new tokens / GPU 0 | 6 / 2 / 2 / 1 / 256 |

对齐 `train_rl.py`（group 8、batch 32、CISPO、4 substeps）与既有 closed-loop 四格 n=64、`max_steps=6`。Harness-1 的 `MAX_TURNS=35` 对单卡 HF 四格过重，全量默认用 6 turn。64 query 由 seed query 扩展；论文级数字仍需 BrowseComp+ manifest。
