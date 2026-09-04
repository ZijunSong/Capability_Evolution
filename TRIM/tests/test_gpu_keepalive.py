from __future__ import annotations

import time

from trim.training.gpu_keepalive import GpuKeepAlive, keepalive_enabled
from trim.training.vllm_hybrid import wait_gpus_quiet


def test_keepalive_disabled_is_noop():
    ka = GpuKeepAlive(enabled=False)
    ka.start()
    assert ka._thread is None
    ka.pause()
    ka.resume()
    ka.stop()


def test_keepalive_env_off(monkeypatch):
    monkeypatch.setenv("TRIM_GPU_KEEPALIVE", "0")
    assert not keepalive_enabled()
    ka = GpuKeepAlive()
    ka.start()
    assert ka._thread is None
    ka.stop()


def test_acquire_release_is_refcount_safe():
    from trim.training import gpu_keepalive as mod

    mod._SHARED = None
    mod._REFS = 0
    a = mod.acquire_keepalive()
    b = mod.acquire_keepalive()
    assert a is b
    mod.release_keepalive()
    assert mod._REFS == 1
    mod.release_keepalive()
    assert mod._REFS == 0
    assert mod._SHARED is None


def test_wait_gpus_quiet_returns_quickly():
    t0 = time.perf_counter()
    wait_gpus_quiet(timeout_s=0.4)
    assert time.perf_counter() - t0 < 2.0
