from __future__ import annotations

from trim.eval.harmony_runtime import fit_prompt_ids_to_context, recent_actions_obs


def test_recent_actions_obs_keeps_tail():
    pairs = [(i, i) for i in range(20)]
    got = recent_actions_obs(pairs, keep=12)
    assert [a for a, _ in got] == list(range(8, 20))


def test_fit_prompt_ids_noop_when_short():
    ids = list(range(100))
    assert fit_prompt_ids_to_context(ids, max_model_len=128, max_new_tokens=8) == ids


def test_fit_prompt_ids_keeps_prefix_and_recent_tail():
    ids = list(range(40000))
    fitted = fit_prompt_ids_to_context(
        ids, max_model_len=32768, max_new_tokens=2048, keep_prefix=4096
    )
    budget = 32768 - 2048
    assert len(fitted) == budget
    assert fitted[:4096] == ids[:4096]
    assert fitted[-100:] == ids[-100:]
    assert 8000 not in fitted


def test_fit_prompt_ids_covers_eval_overflow():
    ids = list(range(33095))
    fitted = fit_prompt_ids_to_context(ids, max_model_len=32768, max_new_tokens=2048)
    assert len(fitted) + 2048 <= 32768
