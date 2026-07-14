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
        total_stage = self._stages.get("total")
        total = (
            float(total_stage["elapsed_seconds"])
            if total_stage else round(sum(item["elapsed_seconds"] for item in stages), 3)
        )
        return {"stages": stages, "total_seconds": total}


def add_runtime_diagnostics(runtime_report: dict, performance_budget: dict | None = None) -> dict:
    stages = {item["stage_name"]: float(item["elapsed_seconds"]) for item in runtime_report.get("stages", [])}
    total = float(runtime_report.get("total_seconds", 0) or 0)
    provider_seconds = stages.get("provider_fetch", 0.0)
    ratio = provider_seconds / total if total > 0 else 0.0
    budget = performance_budget or {}
    failed_stages = [name for name, seconds in stages.items() if name in budget and seconds > float(budget[name])]
    bottleneck = ratio >= 0.80
    reason = ""
    if bottleneck:
        reason = "个股日线Provider重试占总耗时超过80%"
    elif failed_stages:
        reason = f"阶段超过性能预算：{','.join(failed_stages)}"
    else:
        reason = "各阶段处于性能预算内"
    return {
        **runtime_report,
        "provider_fetch_ratio": round(ratio, 4),
        "provider_bottleneck": bottleneck,
        "performance_budget_status": "FAILED" if failed_stages else "PASSED",
        "performance_bottleneck_reason": reason,
    }
