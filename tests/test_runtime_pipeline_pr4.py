from __future__ import annotations

import pandas as pd

from src.analysis_pool_selector import select_analysis_pool
from src.runtime_cache import RuntimeCache
from src.snapshots import StockSnapshot, compute_snapshot_timing


def test_runtime_cache_keys_are_isolated_by_date_timeframe_and_timing_version():
    cache = RuntimeCache()
    cache.set_daily("2026-07-10", "600001", "daily")
    cache.set_indicator("2026-07-10", "600001", "indicator", "daily", "indicator_v2")
    cache.set_timing("600001", "2026-07-10", "timing", "timing_v1")

    assert cache.get_daily("2026-07-10", "600001") == "daily"
    assert cache.get_daily("2026-07-11", "600001") is None
    assert cache.get_indicator("2026-07-10", "600001", "weekly", "indicator_v2") is None
    assert cache.get_timing("600001", "2026-07-10", "timing_v2") is None
    assert cache.stats()["duplicate_request_saved"] == 1


def test_analysis_selector_uses_basic_pool_fields_and_preserves_reasons():
    pool = [
        {"symbol": f"600{i:03d}", "name": str(i), "industry": "电子" if i % 2 else "医药", "amount": 1000 - i, "price": 10, "turnover": 2}
        for i in range(12)
    ]
    selected, reasons = select_analysis_pool(pool, 5)
    assert len(selected) == 5
    assert set(reasons) == {str(item["symbol"]).zfill(6) for item in pool}
    assert all(reasons[symbol]["analysis_pool_reason_codes"] for symbol in selected)
    assert all(reasons[symbol]["selected_for_analysis"] for symbol in selected)


def test_unknown_industry_disables_bonus_without_penalty():
    pool = [
        {"symbol": f"600{i:03d}", "name": str(i), "industry": "未知行业", "amount": 1000 - i, "price": 10, "turnover": 2}
        for i in range(6)
    ]
    selected, audits = select_analysis_pool(pool, 3)
    assert len(selected) == 3
    assert all(item["industry_coverage_bonus"] == 0 for item in audits.values())


def test_four_level_pipeline_limits_are_monotonic():
    scan, analysis, timing, candidate, buy = 500, 150, 30, 10, 1
    assert scan >= analysis >= timing >= candidate >= buy


def test_timing_cache_computes_only_once_for_selected_snapshot():
    frame = pd.DataFrame(
        [{"date": "2026-07-10", "open": 10, "high": 11, "low": 9, "close": 10.5, "volume": 1_000_000, "amount": 500_000_000,
          "ma5": 10, "ma20": 9.8, "ma60": 9.5, "ma20_slope": 0.01, "macd_dif": 1, "macd_dea": 0.5,
          "rsi14": 60, "kdj_k": 50, "kdj_d": 45, "kdj_j": 60, "avg_amount_20d": 400_000_000,
          "return_5d": 0.02, "return_10d": 0.03, "relative_strength_20d": 0.02}]
    )
    snapshot = StockSnapshot("600001", "测试", type("Result", (), {"effective_latest_date": "2026-07-10", "latest_date": "2026-07-10"})(), frame, frame)
    cache = RuntimeCache()
    assert compute_snapshot_timing({"600001": snapshot}, cache) == 1
    assert compute_snapshot_timing({"600001": snapshot}, cache) == 0
