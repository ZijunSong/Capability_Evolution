# TRIM eval 记录

日期：2026-09-07

## 结论

**现有三次均值不能证明组件有效。** 这批 `run_eval.py` 产物一律降级为
`NOT_USABLE_FOR_FULL_VS_ZERO`：可以当作 H_min / `--component zero` 的过程日志，
但不能用来做 full-vs-zero 或「组件 X 有贡献」的结论。

原因不是分数波动，而是 **full mask 从未进入 Harness-1 运行时**：

| 本应生效的路径 | 修复前的实际行为 |
|---|---|
| `LAUNCH.json` / 8 路 `SHARD_CONFIG.json` 写入 `harness_mask` | 只写配置，env 不读 |
| `batched_env_rollout` / `one_episode` 创建 state | Harness-1 `new_state(query, store)` **不传 mask** |
| continuation prompt | `wm_text(..., auto_on=False)` 写死 |
| `execute_tool` | 不接收 mask；`verify` 在 zero 下仍合法 |
| `apply_auto_populate()` | **没有任何调用点** |
| sentence_compress / evidence_graph / content_dedup / token_budget / importance / subtractive | 只有 `dual_view.default_render` 离线占位；`snap_from_state` 也不带这些字段 |

因此即便某次 launch 把 full mask 写进主配置和全部 shard，rollout 仍等价于 zero。
对这种数据做三次均值，差值的期望是 0，不能解释为组件无效，只能解释为 **实验未接通**。

## 已降级 artifact

每个目录都有 `CLAIM.json`。`claim_usable_for_full_vs_zero=false`。

| 目录 | 实际 mask | 备注 |
|---|---|---|
| `outputs/eval_h1_bcplus_test166_zero_gpu4_v2` | zero | curated recall 0.1461 |
| `outputs/eval_h1_bcplus_test166_zero_gpu4_v4` | zero | curated recall 0.1751 |
| `outputs/eval_h1_bcplus_test166_zero_gpu4_v5` | zero | curated recall 0.1765 |
| `outputs/eval_h1_bcplus_test166_zero_gpu4` | zero | 未写出完整 FOUR_CELL |
| `outputs/eval_h1_bcplus_test166_zero_gpu4_v3` | zero | 未写出完整 FOUR_CELL |
| `outputs/eval_h1_bcplus_full_zero_gpu04` | zero（2 shard，不是 8） | 830-pool；recall 字段为 0，不能当 full 组件 run |
| `outputs/smoke_gpu04/eval` | smoke | 不可用于正式对比 |

三次完整 166-test zero 均值（v2/v4/v5）curated recall ≈ **0.166**。
这只描述「当时 H_min 运行时」的分数，**不能**和任何所谓 full 配置做减法。

## 修复（代码，需重跑才生效）

1. Harness-1 `new_state(..., harness_mask=)` 默认 zero；mask 写入 `state`。
2. Rollout 创建 state 时传入 launch/shard 的 mask。
3. `wm_text` 按 mask 渲染（不再写死 `auto_on=False`）。
4. `execute_tool` 读 mask：
   - `verify` 仅在 `verify_tool=ON` 时合法
   - 首次 search 在 `auto_populate_first_search=ON` 时调用 `apply_auto_populate()`
   - importance / subtractive / evidence graph / sentence compress / content dedup / token budget / chunk neighbors / rerank instruction 按位生效
5. Eval 启动先跑 **确定性 wiring probe**（不依赖模型）。任一 ON 位相对 zero 无 delta，或 zero 下仍触发，**直接失败**，不占 GPU。
6. 闭环结束后再做 live 门槛：若首次工具是 search 且 `auto_populate` 为 ON，则该 query 必须留下 `auto_seed`（阈值 1.0）；OFF 位 live fire 必须为 0。
7. 产物写 `RUNTIME_EFFECT_AUDIT.json` 与 `CLAIM.json`。单次 run 即使 wiring 通过，`claim_usable_for_full_vs_zero` 仍为 false，直到存在 **成对的、同样过审计的 full 与 zero**。

重跑命令（修复后）：

```bash
PYTHONPATH=TRIM:SCAPE-EasyOPD python TRIM/scripts/run_eval.py \
  --harness Harness-1 --benchmark bcplus_test_166 \
  --model_name /data/ppnm/models/harness-1 \
  --component all --tp 8 --eval-gpus 0,1,2,3,4,5,6,7

PYTHONPATH=TRIM:SCAPE-EasyOPD python TRIM/scripts/run_eval.py \
  --harness Harness-1 --benchmark bcplus_test_166 \
  --model_name /data/ppnm/models/harness-1 \
  --component zero --tp 8 --eval-gpus 0,1,2,3,4,5,6,7
```

启动时应立刻看到 `RUNTIME_EFFECT_AUDIT.json`。若 wiring probe 失败，进程必须退出，不得写出可用于对比的 FOUR_CELL。

`run_eval.py --benchmark` 现支持 `bcplus_test_50`（官方 BC+ test split 文件顺序的前 50 题，确定性，非随机）。别名：`test_50` / `bcplus_50`。

## 修复后实测：bcplus_test_50 all vs zero

日期：2026-09-07。模型 `/data/ppnm/models/harness-1`，单卡，`--max-turns 40`，`--temperature 1.0`，`--search-k 10`。不是 smoke。

| 边 | GPU | 目录 |
|---|---|---|
| `--component all` | 0 | `outputs/eval_h1_bcplus_test50_all_gpu0` |
| `--component zero` | 4 | `outputs/eval_h1_bcplus_test50_zero_gpu4` |

两边 wiring probe 与 live 审计均 `pass=true`。成对轨迹对比脚本 `scripts/compare_all_zero_traces.py` **通过**（产物：`outputs/eval_h1_bcplus_test50_all_vs_zero_compare.json`）。

| 量 | all | zero |
|---|---|---|
| auto_populate 生效题数 | **50 / 50**（每题 `auto_seed` 长度 8） | **0 / 50** |
| 工具序列完全相同 | **0 / 50** | — |
| `n_curated` 均值 | **20.06**（13 题顶到 cap 30） | **9.86**（1 题到 30） |
| curated recall | 0.075 | 0.190 |
| legal_action_rate | 0.992 | 0.996 |
| `curate` 次数 | 880 | 568 |
| `search_corpus` 次数 | 1043 | 1291 |
| live：importance / evidence_graph / sentence_compress / content_dedup / token_budget / chunk_neighbors / rerank | 50/50 有 fire | 全部 0 |
| live：auto_populate | 50/50 | 0 |
| live：zero 位泄漏 | — | 无 |

轨迹层面已经不是同一条路径：all 在首次 search 后立刻灌 curated，后续 WM 带 evidence graph / token budget 等标记；zero 没有 `auto_seed`，`runtime_effects` 全 0。

指标也有明确差，而且方向是 **all 的 curated recall 更低**。这能证明组件进入了运行时，**不能**证明组件提升了检索质量。subtractive live fire 为 0 是因为 eviction 只在 cap=30 且 incoming importance 严格优于最差条时触发；本批 curate 基本都是 `fair`，顶满后走 `[CAPACITY] not added`，不是 wiring 失效。模型两边都没调用 `verify`。

单次 run 的 `CLAIM.json` 仍写 `claim_usable_for_full_vs_zero=false`（脚本按「单次产物」落盘）。成对可用性以本对比文件为准：`pass=true`。166-test 的 full-vs-zero 仍需按同样门槛重跑，不能沿用旧三次均值。

## 训练侧 RL / TRIM（2026-09-08）

Eval 接通之后，训练 rollout 还有同类缺口：`harness_mask=None` 时 `new_state` 落到 **zero**，而 `snap_from_state` 用 `student_mask_for`（单组件是 minus，不是 zero）。HF 回退路径 `rollout_group` 根本不传 mask；batched 入口的测试也不传。`--train_method rl` / `trim` 的 `train_only` 又不走 `eval_closed_loop`，启动时没有 wiring probe，mask 没进 env 也会占 GPU。

已修：

1. `resolved_rollout_mask`：显式 mask 优先；否则 teacher cell 用 H_full，student / RL / TRIM 用 H_min。`rollout_queries_batched` / `one_episode` / `rollout_group` / `eval_closed_loop` 都先 resolve 再 `new_state`。
2. `collect_groups` / `eval_now` 按 `teacher_mode` 选 student vs teacher mask（不再一律 student）。
3. teacher cell 用 `teacher_for(component_id)`，不再写死 `sentence_compress_teacher`。
4. `run_four_cell` 在加载 query / vLLM **之前**跑 `audit_train_runtime_or_raise`：student + teacher 的 live wiring probe；`lambda_opd>0` 且非 sampled-gap 时还要求 teacher side-branch 能投影出 ≥1 个 student-legal step。失败直接退出。产物 `RUNTIME_EFFECT_AUDIT.json`。
5. `LAUNCH.json` / `RUN_MANIFEST.json` 写入真实 mask dict（不再只写 H_min 文字）。

语义保持不变：TRIM student 仍是 H_min（`--component all` 时 student = zero，teacher = full）；DualView teacher **不步进** env。`--train_method rl` 仍 `lambda_opd=0`、cell 丢掉 OPD datums；`trim`（`scape_seed`）是 CISPO + projected SEED gap。

无 GPU 单测：`tests/test_train_rl_trim_runtime.py`。
