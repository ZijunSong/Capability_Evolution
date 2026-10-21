"""Per-rank progress events. These diagnose a step; they are not a keepalive."""

from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path
from typing import Any


class RankProgress:
    def __init__(self, out: Path, *, rank: int, gpu_uuid: str, hostname: str | None = None) -> None:
        self.rank = int(rank)
        self.gpu_uuid = gpu_uuid
        self.hostname = hostname or socket.gethostname()
        self.path = Path(out) / "progress" / f"rank-{self.rank:05d}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._phase = "init"
        self._fields: dict[str, Any] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._beat, name=f"sft-progress-{self.rank}", daemon=True)
        self._thread.start()

    def event(self, phase: str, **fields: Any) -> None:
        self._phase = str(phase)
        self._fields = dict(fields)
        self._write(phase=self._phase, heartbeat=False, **fields)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1)

    def _beat(self) -> None:
        while not self._stop.wait(30):
            self._write(phase=self._phase, heartbeat=True, **self._fields)

    def _write(self, **fields: Any) -> None:
        payload = {
            "timestamp": time.time(),
            "hostname": self.hostname,
            "rank": self.rank,
            "gpu_uuid": self.gpu_uuid,
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()


def summarize_progress(out: Path) -> dict[str, Any]:
    """Aggregate events that actually exist. Missing measurements stay null."""
    root = Path(out) / "progress"
    events: list[dict[str, Any]] = []
    if root.is_dir():
        for path in sorted(root.glob("rank-*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    events.append(json.loads(line))
    updates = [
        float(item["update_wall_s"])
        for item in events
        if item.get("phase") == "optimizer_step" and not item.get("heartbeat") and "update_wall_s" in item
    ]
    steady = updates[2:] if len(updates) > 2 else list(updates)
    measured = bool(steady)
    median = _median(steady) if measured else None
    return {
        "gpu_measured": measured,
        "status": "measured" if measured else "not_measured",
        "n_progress_events": len(events),
        "n_recorded_updates": len(updates),
        "median_update_wall_s": median,
        "note": (
            "Times come from this run's progress events."
            if measured
            else "No completed GPU update was recorded. Do not treat this file as a throughput result."
        ),
    }


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])
