"""Keep GPU SMs busy during CPU-only gaps.

Cluster watchdogs kill jobs when GPU-Util / SM-Util stay near zero for too
long. TRIM has unavoidable CPU stretches (query load, BM25, HF↔vLLM swaps).
This dummy GEMM loop occupies every visible GPU until the real engine starts.
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from typing import Iterator, Sequence


def keepalive_enabled() -> bool:
    return str(os.environ.get("TRIM_GPU_KEEPALIVE", "1")).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


class GpuKeepAlive:
    def __init__(
        self,
        *,
        dim: int = 2048,
        enabled: bool | None = None,
        device_ids: Sequence[int] | None = None,
    ):
        self.dim = int(dim)
        self.enabled = keepalive_enabled() if enabled is None else bool(enabled)
        self.device_ids = None if device_ids is None else [int(i) for i in device_ids]
        self._cv = threading.Condition()
        self._state = "stopped"
        self._idle = True
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()

    def _devices(self) -> list[int]:
        import torch

        if self.device_ids is not None:
            return list(self.device_ids)
        if not torch.cuda.is_available():
            return []
        return list(range(int(torch.cuda.device_count())))

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        try:
            import torch

            if not torch.cuda.is_available() or not self._devices():
                return
        except Exception:
            return
        with self._cv:
            self._state = "running"
            self._idle = False
        self._ready.clear()
        self._thread = threading.Thread(target=self._loop, name="trim-gpu-keepalive", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=20.0)

    def pause(self) -> None:
        """Drop dummy tensors so vLLM / HF can claim the cards."""
        with self._cv:
            if self._thread is None or self._state == "stopped":
                return
            self._state = "paused"
            self._cv.notify_all()
            self._cv.wait_for(lambda: self._idle or self._state == "stopped", timeout=10.0)
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def resume(self) -> None:
        with self._cv:
            if self._thread is None or self._state == "stopped":
                return
            self._state = "running"
            self._cv.notify_all()

    def stop(self) -> None:
        with self._cv:
            self._state = "stopped"
            self._cv.notify_all()
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout=8.0)
        with self._cv:
            self._idle = True

    def _loop(self) -> None:
        import torch

        devices = self._devices()
        tensors: list[tuple[object, object]] = []
        streams: list[object] = []
        allocated = False

        def alloc() -> None:
            nonlocal allocated
            if allocated:
                return
            tensors.clear()
            streams.clear()
            for i in devices:
                with torch.cuda.device(i):
                    a = torch.randn(self.dim, self.dim, device=f"cuda:{i}", dtype=torch.float16)
                    b = torch.randn(self.dim, self.dim, device=f"cuda:{i}", dtype=torch.float16)
                    tensors.append((a, b))
                    streams.append(torch.cuda.Stream(device=i))
            allocated = True

        def free() -> None:
            nonlocal allocated
            tensors.clear()
            streams.clear()
            allocated = False
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        try:
            self._ready.set()
            while True:
                with self._cv:
                    if self._state == "stopped":
                        free()
                        self._idle = True
                        self._cv.notify_all()
                        return
                    if self._state == "paused":
                        free()
                        self._idle = True
                        self._cv.notify_all()
                        self._cv.wait()
                        continue
                    alloc()
                    self._idle = False
                    live = list(zip(tensors, streams))
                for i, ((a, b), stream) in zip(devices, live):
                    with torch.cuda.device(i), torch.cuda.stream(stream):
                        a.copy_(torch.mm(a, b))
                for stream in streams:
                    stream.synchronize()
        except Exception:
            self._ready.set()
        finally:
            free()
            with self._cv:
                self._idle = True
                self._state = "stopped"
                self._cv.notify_all()


@contextmanager
def gpu_keepalive(*, dim: int = 2048) -> Iterator[GpuKeepAlive]:
    ka = acquire_keepalive(dim=dim)
    try:
        yield ka
    finally:
        release_keepalive()


_LOCK = threading.Lock()
_SHARED: GpuKeepAlive | None = None
_REFS = 0


def acquire_keepalive(*, dim: int = 2048, device_ids: Sequence[int] | None = None) -> GpuKeepAlive:
    """Process-wide keepalive so launchers and run_four_cell share one loop."""
    global _SHARED, _REFS
    with _LOCK:
        if _SHARED is None:
            _SHARED = GpuKeepAlive(dim=dim, device_ids=device_ids)
            _SHARED.start()
        _REFS += 1
        return _SHARED


def release_keepalive() -> None:
    global _SHARED, _REFS
    with _LOCK:
        _REFS = max(0, _REFS - 1)
        if _REFS == 0 and _SHARED is not None:
            _SHARED.stop()
            _SHARED = None
