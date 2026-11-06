# E1 中期发现（AppWorld/WebShop/ALFWorld 9B+27B 进化，2026-11-04）

> 状态：进行中（6 路迭代，部分 r3/r4/r5）。本文档记录跨域稳定模式，供修复方向定位。

## 发现 1（主导失败）：self-proposer 发不出合法 patch 封套

**证据**：
- alfworld-27b r3：0 个候选被评估，6 次提议全部 `CandidateInvalid('flat-file patch missing entrypoint / operation')`
- webshop-9b r2：0 评估，全废
- appworld-27b（待 r3 数据）

**已到位的防线**：proposer prompt 末尾的 OUTPUT ENVELOPE 重述（近因效应修复，9fhn3/3b1bb 复盘）+ repair 反馈循环 + PR-5 baseline 保留（没有加冕废候选）+ PROPOSAL_STALLED 命名。

**待修方向**：repair 反馈只在下次 prompt 里贴错误文本；模型仍无法从错误文本恢复 schema。需要把"上次被拒的原始输出 + 期望 schema 的最小 diff 对比"放进 retry notes，或在 proposer 侧做确定性 schema 修复（entrypoint 缺失时从 files[0] 推导——`_derive_entrypoint` 已存在，检查它为何没兜住）。

## 发现 2（webshop 打分链路）：合法候选全部 0.0 分

**证据**：webshop-9b r3（4 候选）与 r4（6 候选）全部 `status=legal、component_errors={}` 但 `macro=0.0`；gen0 起就 `NO_SUPPORTED_IMPROVEMENT`。

**初步定位**：webshop scorer 需要私有 goal（`private_record` 正常提供 instruction/asin/price_upper），但 panel 全零 + 基线分数不可见（evolve json 不存 parent per_task），且本地无法复现（webshop SimServer 需大内存，本机 OOM）。**待 job 内验证**：no-op episode 的 webshop value 是否也为 0——若 no-op=0 且任何合法动作也=0，则 scorer 读取链路断了（E0 链的"no private goal"同族问题）。

**附带缺口**：evolve 的 `parent_result`（含 per_task/steps）不落盘——父代分数不可追溯，修复方向是 panel_runner 结果持久化到 run_dir。

## 发现 3：改进信号可见

- PR-5 baseline 保留在两域生效：webshop r3/r4 的 gen1-2 都是 `NO_SUPPORTED_IMPROVEMENT: incumbent retained`——没有加冕废候选。
- 轨迹持久化（steps）已进入 candidate 评估记录，待第一个非零候选验证可用性。

## 当前 job 状态（12:40 快照）

| job | 状态 |
|---|---|
| appworld-9b r3（vb3b5 + 3gwab 双份） | running / enqueued |
| webshop-9b r4（d6iup） | running |
| webshop-27b r3（apdns） | running |
| alfworld-27b r3 | 完成：CANDIDATES_REPEATEDLY_INVALID（0 评估） |
| alfworld-9b | r3 完成 PLATEAU，待重提 |
| webshop-9b r3 | 完成 PLATEAU（4×0.0） |

## 下一步（按优先级）

1. **修 proposer schema 恢复**：retry 时给"原始输出 vs schema" diff（或确定性 fallback 推导 entrypoint/operation）。
2. **job 内验证 webshop scorer 链路**：no-op panel 一次。
3. **panel_runner 结果持久化**（parent per_task 落盘）。
4. 修完后统一重提失败的域。

---

## 补充（2026-11-04 晚：appworld-9b r3 收官）

- **appworld-9b r3**（3gwab，8.6h）：11 个合法候选 **全部 macro 0.0**，PLATEAU 无 champion。判定为**真零分**而非打分链路断裂：E0 同链路 G1-G4 全 PASS 且 0/16 状态变更；webshop 同管线能打出 0.45。
- **候选主题多样性坍塌**：11 个组件全是 format/action guard 家族（format_guard×3、action_sanitizer×2、safety_guard×2、action_guard、format_validator、completion_gate、action_formatter、action_validator）——与 webshop-27b 的"6/6 同名 error_tracker"同款。proposer 在 analysis 片段全是 parse/格式错误时只会提出"格式修复"组件，而 E0 已证明瓶颈在 API 使用（认证/读改链），格式守卫不改变 goal 完成度。
- **配置语义澄清**：`dev_panel_per_domain: 0` 在 evolve 路径（`if max_per_domain`）本就等于全量面板，v3 注释"0 = 全量"在 evolve 路径是对的；webshop r9 champion 0.45 与该值无冲突。已把 v3 appworld 配置改为 `null` 并注明语义差异（eval/campaign 路径的 int 0 才有空切片问题）。
- **修复方向升级**：仅修 proposer 格式不够——需要让 analysis 片段区分"格式错误"vs"API 使用失败"（认证错误/找不到实体/只读不写），否则进化器永远在格式守卫家族里打转。E0 的 0 状态变更 + env doc 的 api_docs 发现命令应成为 packet 的显式上下文。

---

## 补充二（2026-11-04 夜：alfworld-9b r4 收官 + webshop-27b env 修复）

- **alfworld-9b r4**（8hk1b succeed）：**第二个真实 champion** `3bb04695…`（单组件 `CompletionGuard`：veto 非法工具名 + 防提前 finish），gen0 当选 macro **0.25**（gen1-3 incumbent retained）。候选分数 0~0.25 分布——alfworld 打分链路正常（对比 r3 时代全 0）。
- **webshop-27b r4/r4b 全 0.0 根因修复**：27B entry 漏 `RSI_WEBSHOP_FULL_ALLOW=1`（9B entry 169 行有），probe 门把整个 panel domain-skip。已补 env 并提交 **r5=4gurh**（运行中）。
- **三域对照更新**：WebShop 9B champion 0.45 ✅ / ALFWorld 9B champion 0.25 ✅ / AppWorld 9B 真 0 分（瓶颈在 API 使用层，guard 家族不解决）。
- 两个 champion 均为**收尾/守卫类组件**（finish 门控），与 E0 发现一致——进化器从"termination 校验"起步是合理路径，但真正的能力增益要看 confirm 时 strict success 是否被这些守卫解锁。

---

## 补充三（2026-11-05 凌晨：webshop-27B r5 收官 = 27B 线打通）

- **webshop-27b r5**（spzne succeed，~2.5h）：**16 候选全 ACCEPTED 全非零**（0.16~0.617），**MAX_GENERATIONS 跑满 8 代**——27B 线端到端打通。
- **champion `88606c24`** macro **0.589** = **四组件组合 bank**（completion_gate + loop_detector + search_progress_tracker + search_stall_breaker），进化链 gen0(0.389)→gen3(0.500)→gen4(0.561)→gen6(0.589)——**bank 组合式生长首次完整呈现**（guide §7.2 的 within-run composition 设计意图）。
- 27B vs 9B：27B champion 0.589 > 9B 0.45；27B 组件更丰富（4 vs 2）；27B 的 gen1/2/5/7 保留 incumbent 也符合 baseline 保留语义（同 bank 复测方差仍在）。
- **未完成**：27B champion 与 9B champion 的独立 confirm 都未做；appworld-27b 的 proposer 层问题（rqdza evaluated=0）仍未解——堵点在 packet/proposer 层非 entry 层。

---

## 更正（2026-11-05：appworld "真零分"判定被推翻）

并发会话的 53 题逐题审计（`docs/badcase_appworld_9b_h0_final_test.md`）实锤：**AppWorld 的 Finish 路径从未把 `Finish.answer` 传给 `supervisor.respond()`**（`appworld_env.py` step() 的 Finish 分支只查 `task_completed()`）——41+12/53 题 answer=NOT_GIVEN，**是 harness 断链，不是模型不会**。

**影响更正**：
- 我此前"appworld-9b r3 11 候选真零分"的判定作废——候选们在断链的 harness 上跑，0.0 是测量伪影。
- E0 的"0/16 状态变更"部分保留（真实 DB 操作 49/53 有发生），但 strict=0 的主因是 answer 断链。
- 修复方向（并发会话 P0）：adapter 代 respond + 重跑 E0 对照 → appworld 基线和进化候选都要在新口径下重建。
- **教训（与 E0 审计同族）**：harness 断链类 bug 的症状（全零分数）与"能力为零"不可区分，除非逐题对 artifact 审计。这也是文档 §4.6 no-op/已知正确/错误状态三对照必须先跑的原因——no-op 对照只能证明初始状态，"已知正确动作"对照才能证明 answer 链路。

---

## 补充四（2026-11-05：H0_v2 口径下 alfworld-9B r5 收官）

- **alfworld-9b r5**（zhfrd succeed）：14 候选评估（0.33~0.75），**champion `622625ded4` macro 0.75**（gen3 当选）= `completion_gate + alfworld_checker` 双组件；进化链 gen0 0.583 → gen3 0.750。**vs r4（渲染修复前）champion 0.25 → r5 0.75**：structured 渲染修复（合法命令可见）把 alfworld 打分整体抬升约 +50pp——这是 harness 修复解锁能力的直接证据。
- 27B 两路（appworld r4 brurd / alfworld r5 gth6z）running；appworld-9b r4（v3hvt）running。
- webshop-27b r5 (0.589) 仍是当前全场景最高 champion。

---

## 补充五（2026-11-05：alfworld-27B r5 收官）

- **alfworld-27b r5**（gth6z）：8 候选 0.500~0.833，**champion `83ae9fdf`**（gen0 当选）PLATEAU。27B 把 alfworld 带推到 **0.833 最高单候选**（vs 9B r5 champion 0.75）。
- appworld-9b r4 / appworld-27b r4 仍在跑（gen0 已 ACCEPTED，panel 评估中）。

---

## 补充六（2026-11-05：webshop-9B r10 + 新口径三路启动）

- **webshop-9b r10**（afe1d succeed）：6 候选 0.089~0.562，champion `b0ad1c7d` 0.5625（gen1 当选）——H0_v2 口径下 webshop 线复现 champion（与 r9 的 0.45 同量级）。
- 并发会话夜间重启：appworld-9b r5（7fgip）、appworld-27b r5（zdbip）、webshop-9b r10——v3hvt r4 被 stop 取代。

---

## 补充七（2026-11-06：AppWorld 首个非零候选）

- **appworld-9b r5**（7fgip）gen0 首个候选 **macro 0.289 非零**——answer 断链修复 + H0_v2 口径下 AppWorld 进化线首次产生真实信号（断链时代 11 候选全 0.0）。gen0 两候选 ACCEPTED，进化推进中。
- appworld-27b r5（zdbip）仍在首代 panel。

---

## 补充八（2026-11-06：appworld-9B r5 收官——基线已非零，候选未超越）

- **appworld-9b r5**（7fgip）：6 候选 0.289~0.316，PLATEAU 无 champion。**关键读法变化**：gen0-2 全部 `incumbent retained` = **空 bank 基线本身已 ~0.29+**（H0_v2 口径，与 E0-v2 的 34.74% 一致）——候选与基线打平但未超越。断链时代"候选 0.0 vs 基线 0"的假持平已被真实信号取代：AppWorld 的 strict 口径下 9B 候选还有差距要跨。
- appworld-27B r5（zdbip）running（4 候选 0~0.261）；webshop-27B r11（cdyeb）running（2 候选 0.408~0.514）。

---

## 补充九（2026-11-06：appworld-27B r5 收官）

- **appworld-27b r5**（zdbip succeed ~20h）：8 候选（0~0.261），**champion `5ed3bbf1`**（gen0 当选 macro **0.261**）= 单组件 `ActionValidator`；gen1-3 incumbent retained（27B 基线同样 ~0.26+，与 9B r5 基线 0.29+ 同量级）。PLATEAU 收官。
- **AppWorld 双模型定案**：9B/27B 候选都与各自基线打平未超越（真信号，断链时代的假持平已消除）；两域 champion（webshop/alfworld）+ appworld honest-PLATEAU。
- 并发会话推进 trace 聚焦轮：alfworld-9B r6-trace（toqpp running）+ webshop-9B trace-r11（exm1p queued）。

---

## 补充十（2026-11-06：trace 聚焦轮首批）

- **alfworld-9b trace-r6**（toqpp succeed）：7 候选 0.25~0.417，champion `755b48b8`（gen0 0.417）PLATEAU。**分数带比 r5（0.33~0.75）低**——同 manifest 下 panel 方差的直接展示（跨 run 的 base/candidate 分数不可直接横比，方差警示再次坐实）。
- webshop-9b trace-r11（exm1p）running。

---

## 补充十一（2026-11-06：webshop-9B trace-r11 收官——8 代单调爬升）

- **webshop-9b trace-r11**（exm1p succeed）：16 候选 0.139~0.679，**MAX_GENERATIONS 跑满 8 代**，**champion `bd397b10` macro 0.679**（gen7）——逐代 0.49→0.48→0.57→0.46→0.59→0.63→0.68 **单调爬升**，trace 聚焦反馈（EvidenceStore 真实步骤）的进化轨迹质量显著优于此前各轮。**超越 27B r5 的 0.589，成为 webshop 全场景最高 champion**。
