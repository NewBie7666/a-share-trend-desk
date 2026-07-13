"""Ephemeral, versioned cache used only while one CLI invocation is alive."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RuntimeCache:
    stock_daily_cache: dict[tuple, object] = field(default_factory=dict)
    stock_basic_cache: dict[tuple, object] = field(default_factory=dict)
    indicator_cache: dict[tuple, object] = field(default_factory=dict)
    timing_cache: dict[tuple, object] = field(default_factory=dict)
    cache_hit: int = 0
    cache_miss: int = 0
    duplicate_request_saved: int = 0

    def _get(self, store: dict, key: tuple):
        if key in store:
            self.cache_hit += 1
            self.duplicate_request_saved += 1
            return store[key]
        self.cache_miss += 1
        return None

    def get_daily(self, trade_date: str, symbol: str):
        return self._get(self.stock_daily_cache, (trade_date, str(symbol).zfill(6), "daily"))

    def set_daily(self, trade_date: str, symbol: str, value: object) -> None:
        self.stock_daily_cache[(trade_date, str(symbol).zfill(6), "daily")] = value

    def get_indicator(self, trade_date: str, symbol: str, timeframe: str = "daily", indicator_version: str = "indicator_v2"):
        return self._get(self.indicator_cache, (trade_date, str(symbol).zfill(6), timeframe, indicator_version))

    def set_indicator(self, trade_date: str, symbol: str, value: object, timeframe: str = "daily", indicator_version: str = "indicator_v2") -> None:
        self.indicator_cache[(trade_date, str(symbol).zfill(6), timeframe, indicator_version)] = value

    def get_timing(self, symbol: str, latest_trade_date: str, timing_version: str = "timing_v1"):
        return self._get(self.timing_cache, (str(symbol).zfill(6), latest_trade_date, timing_version))

    def set_timing(self, symbol: str, latest_trade_date: str, value: object, timing_version: str = "timing_v1") -> None:
        self.timing_cache[(str(symbol).zfill(6), latest_trade_date, timing_version)] = value

    def stats(self) -> dict:
        return {
            "cache_hit": self.cache_hit,
            "cache_miss": self.cache_miss,
            "duplicate_request_saved": self.duplicate_request_saved,
        }

