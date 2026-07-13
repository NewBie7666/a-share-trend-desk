"""In-process stage timing for one daily-signal run."""

from __future__ import annotations

from datetime import datetime
from time import perf_counter


class RuntimeProfiler:
    def __init__(self) -> None:
        self._active: dict[str, tuple[datetime, float]] = {}
        self._stages: dict[str, dict] = {}

    def start_stage(self, name: str) -> None:
        self._active[name] = (datetime.now(), perf_counter())

    def end_stage(self, name: str) -> None:
        started_at, started_clock = self._active.pop(name)
        finished_at = datetime.now()
        self._stages[name] = {
            "stage_name": name,
            "start_time": started_at.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": finished_at.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": round(perf_counter() - started_clock, 3),
        }

    def get_runtime_report(self) -> dict:
        stages = list(self._stages.values())
        total = round(sum(item["elapsed_seconds"] for item in stages), 3)
        return {"stages": stages, "total_seconds": total}

