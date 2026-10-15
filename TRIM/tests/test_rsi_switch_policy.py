from trim.rsi.switch_policy import (
    CONFIRM_TEACHER,
    STOP_NO_TEACHER_GAIN,
    SearchMeasurement,
    decide,
    generations_since_record,
)


def _ready(**kwargs) -> SearchMeasurement:
    base = dict(
        generation=3,
        min_generations=3,
        quality_history=[0.2, 0.2, 0.2, 0.2],
        cost_history=[1, 1, 1, 1],
        g0_pass=True,
        g2_pass=True,
        g3_pass=True,
        retention_ok=True,
        has_champion=True,
        evidence_status="ok",
        delta_h=0.03,
    )
    base.update(kwargs)
    return SearchMeasurement(**base)


def test_plateau_without_gain_does_not_train():
    since = generations_since_record([0.20, 0.201, 0.202, 0.203], [1, 1, 1, 1], epsilon=0.005, epsilon_select=0.005, material_cost_reduction=0.05)
    assert since >= 3
    decision = decide(_ready(delta_h=0.0, has_champion=True, retention_ok=True))
    assert decision.status == STOP_NO_TEACHER_GAIN


def test_complexity_block_does_not_start_training():
    decision = decide(_ready(complexity_blocked=True, g3_pass=False, budget_exhausted=True))
    assert decision.status != "TRAINING"
    assert decision.stop_search is True


def test_small_cumulative_gain_resets_patience():
    stalled = generations_since_record(
        [0.10, 0.102, 0.103, 0.104], None, epsilon=0.005, epsilon_select=0.005, material_cost_reduction=0.05
    )
    moved = generations_since_record(
        [0.10, 0.103, 0.108], None, epsilon=0.005, epsilon_select=0.005, material_cost_reduction=0.05
    )
    assert stalled >= 3
    assert moved == 0
    decision = decide(
        _ready(
            generation=2,
            quality_history=[0.10, 0.103, 0.108],
            delta_h=0.008,
            recent_gains=[0.003, 0.008],
        )
    )
    assert decision.status != CONFIRM_TEACHER
    assert decision.status != "TRAINING"
