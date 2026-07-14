"""Session-only adaptive provider concurrency and circuit-breaker state."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic


PROVIDER_FAILURE_CATEGORIES = {"network_error", "invalid_schema"}


def is_provider_failure(attempt: dict) -> bool:
    return bool(
        attempt.get("status") == "failed"
        and attempt.get("error_category") in PROVIDER_FAILURE_CATEGORIES
    )


@dataclass
class AdaptiveWorkerState:
    workers: int = 4
    freeze_windows: int = 0
    healthy_windows_below_four: int = 0
    history: list[dict] = field(default_factory=list)

    def evaluate(self, attempts: list[dict], freeze_after_change: int = 1) -> int:
        total = len(attempts)
        successes = sum(item.get("status") == "success" for item in attempts)
        timeout_count = sum("timeout" in str(item.get("error", "")).lower() for item in attempts)
        remote_disconnect = any("remotedisconnected" in str(item.get("error", "")).lower() for item in attempts)
        provider_failures = sum(is_provider_failure(item) for item in attempts)
        success_rate = successes / total if total else 0.0
        timeout_rate = timeout_count / total if total else 0.0
        healthy = bool(total and success_rate >= 0.95 and timeout_rate < 0.10 and not remote_disconnect)
        unhealthy = bool(provider_failures or timeout_count or remote_disconnect)
        before = self.workers
        reason = "保持"

        if unhealthy:
            self.healthy_windows_below_four = 0
            self.workers = {8: 4, 6: 4, 4: 2, 2: 1, 1: 1}[self.workers]
            reason = "Provider失败，降低并发"
        elif self.freeze_windows > 0:
            self.freeze_windows -= 1
            reason = "调整冻结窗口"
        elif healthy:
            if self.workers < 4:
                self.healthy_windows_below_four += 1
                if self.healthy_windows_below_four >= 2:
                    self.workers = {1: 2, 2: 4}[self.workers]
                    self.healthy_windows_below_four = 0
                    reason = "连续健康窗口，恢复并发"
            else:
                self.healthy_windows_below_four = 0
                self.workers = {4: 6, 6: 8, 8: 8}[self.workers]
                reason = "健康窗口，扩大并发"
        else:
            self.healthy_windows_below_four = 0

        if self.workers != before:
            self.freeze_windows = max(int(freeze_after_change), 0)
        self.history.append({
            "workers_before": before,
            "workers_after": self.workers,
            "request_count": total,
            "success_rate": round(success_rate, 4),
            "timeout_rate": round(timeout_rate, 4),
            "provider_failure_count": provider_failures,
            "reason": reason,
        })
        return self.workers


@dataclass
class ProviderCircuit:
    consecutive_failures: int = 0
    open_until: float = 0.0
    half_open_inflight: bool = False


class SessionProviderCircuits:
    def __init__(self, providers: list[str], failure_threshold: int = 5, cooldown_seconds: float = 120.0, clock=monotonic):
        self._clock = clock
        self.failure_threshold = max(int(failure_threshold), 1)
        self.cooldown_seconds = max(float(cooldown_seconds), 0.0)
        self.states = {provider: ProviderCircuit() for provider in providers}
        self.history: list[dict] = []

    def source_order_for_task(self, default_order: list[str]) -> list[str]:
        now = self._clock()
        output: list[str] = []
        for provider in default_order:
            state = self.states.setdefault(provider, ProviderCircuit())
            if state.open_until <= 0:
                output.append(provider)
            elif now >= state.open_until and not state.half_open_inflight:
                state.half_open_inflight = True
                output.append(provider)
        return output

    def record_symbol_result(self, source_attempts: list[dict]) -> None:
        by_provider: dict[str, list[dict]] = {}
        for attempt in source_attempts or []:
            provider = str(attempt.get("source_name", ""))
            if provider in self.states:
                by_provider.setdefault(provider, []).append(attempt)
        for provider, attempts in by_provider.items():
            state = self.states[provider]
            success = any(item.get("status") == "success" for item in attempts)
            provider_failed = bool(attempts) and all(is_provider_failure(item) for item in attempts)
            if success:
                state.consecutive_failures = 0
                state.open_until = 0.0
                state.half_open_inflight = False
            elif provider_failed:
                if state.open_until > self._clock() and not state.half_open_inflight:
                    # Ignore failures from requests that were already in flight
                    # when another result opened the circuit.
                    continue
                state.consecutive_failures += 1
                was_half_open = state.half_open_inflight
                state.half_open_inflight = False
                if state.consecutive_failures >= self.failure_threshold:
                    state.open_until = self._clock() + self.cooldown_seconds
                    self.history.append({
                        "provider": provider,
                        "event": "circuit_reopen" if was_half_open else "circuit_open",
                        "failure_count": state.consecutive_failures,
                        "cooldown_seconds": self.cooldown_seconds,
                    })
            elif state.half_open_inflight:
                # Non-provider failures do not count and close the probe slot.
                state.half_open_inflight = False

    def open_providers(self) -> list[str]:
        now = self._clock()
        return [provider for provider, state in self.states.items() if state.open_until > now]
