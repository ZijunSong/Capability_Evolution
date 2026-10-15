# TRIM RSI 合约

固定基础 Harness `H0`（`--component zero`）不变。每一轮只进化一个可蒸馏的 bank `B_r`，再用 TRIM 在学生的 `H0` 轨迹上做 RL，教师在隔离旁路提供投影 OPD。下一轮仍从 `H0` 起步，不把上一轮组件写进 `zero`。

## 状态

`EVOLVING`、`STOP_NO_TEACHER_GAIN`、`STOP_UNDISTILLABLE`、`STOP_INSUFFICIENT_EVIDENCE`、`CONFIRM_TEACHER`、`TRAINING`、`STOP_CONFIRMATION_FAILED`、`STOP_WEIGHT_PROMOTION_FAILED`、`PROMOTED_PARTIAL_INTERNALIZATION`、`PROMOTED_FULL_RETIREMENT`、`DONE`。

搜索停止不等于可以训练。训练还要求开发门槛和一次独立的教师确认。确认失败后不能在同一面板上换第二个候选。

## 发布

`Delta_param = J(S2) - J(S1)` 达到预注册增益且下界为正，同时旧能力分层没有被零差值区间宣布为非劣，模型才进入下一轮。只提高 `S3` 不发布。`S2` 仍低于旧的 `S0` 时记为部分内化，不写成完整退休。

## 数据

`work`、`harness_select`、`weight_dev`、每轮的 `confirm_h` / `confirm_w`、以及最终 `test` 按问题簇互斥。`--n-queries` 不是隔离划分。

## 实验名

`seed_mode: empty` 是主实验。`seed_mode: legacy` 必须在实验名中标明 seeded。
