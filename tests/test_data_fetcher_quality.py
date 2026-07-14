from datetime import datetime
import json

import pandas as pd

from src import data_fetcher
from src.data_fetcher import FetchResult, expected_trade_date, fetch_index_daily, fetch_stock_daily, fetch_stocks, summarize_stock_fetches, trade_day_diff
from src.data_quality import compute_data_quality_level
from src.main import _append_fetch_summary_rows, _cache_notes, _market_data_ready_report
from src.report import write_reports
from src.signals import categorize_forbidden


def daily_df(date="2026-07-01"):
    return pd.DataFrame(
        {
            "date": [date],
            "open": [10],
            "close": [10],
            "high": [11],
            "low": [9],
            "volume": [1000],
            "amount": [500_000_000],
        }
    )


def multi_daily_df(dates):
    return pd.DataFrame(
        {
            "date": dates,
            "open": [10 + idx for idx, _ in enumerate(dates)],
            "close": [10 + idx for idx, _ in enumerate(dates)],
            "high": [11 + idx for idx, _ in enumerate(dates)],
            "low": [9 + idx for idx, _ in enumerate(dates)],
            "volume": [1000 for _ in dates],
            "amount": [500_000_000 for _ in dates],
        }
    )


def test_missing_local_trade_calendar_falls_back_without_remote_call(tmp_path, monkeypatch):
    monkeypatch.setattr(data_fetcher, "CACHE_DIR", tmp_path)
    settings = {}
    dates, status = data_fetcher.get_trade_calendar_dates(settings)
    assert dates == []
    assert status == "fallback_weekday"
    assert settings["trade_calendar_source"] == "fallback_weekday"


def test_local_trade_calendar_cache_is_used(tmp_path, monkeypatch):
    monkeypatch.setattr(data_fetcher, "CACHE_DIR", tmp_path)
    payload = ["2026-07-09", "2026-07-10"]
    (tmp_path / "trade_calendar.json").write_text(json.dumps({"trade_dates": payload}), encoding="utf-8")
    settings = {}
    dates, status = data_fetcher.get_trade_calendar_dates(settings)
    assert dates == payload
    assert status == "fresh"
    assert settings["trade_calendar_source"] == "local_cache"
    assert settings["trade_calendar_error"] == ""


def test_akshare_preflight_timeout_uses_stock_cache_without_remote_fetch(monkeypatch):
    local_settings = {
        "akshare_runtime_unavailable": True,
        "data_fetch": {"max_retries": 2, "min_history_days": 1},
        "cache": {"max_cache_age_trade_days": 1},
    }
    called = []

    def forbidden_fetcher(*args, **kwargs):
        called.append(True)
        raise AssertionError("remote source must not run after AkShare preflight timeout")

    monkeypatch.setattr(data_fetcher, "_fetch_stock_source", forbidden_fetcher)

    result = fetch_stock_daily(
        "600001",
        "测试股份",
        settings=local_settings,
        expected_trade_date_value="2026-07-10",
        trade_dates=["2026-07-10"],
        cache_reader=lambda symbol: long_daily_df("2026-07-10"),
    )
    assert called == []
    assert result.final_source == "cache"
    assert result.effective_latest_date == "2026-07-10"


def test_requests_timeout_wrapper_adds_default_but_preserves_explicit_timeout(monkeypatch):
    import requests

    calls = []

    class FakeSession:
        def request(self, method, url, **kwargs):
            calls.append(kwargs.get("timeout"))
            return object()

    monkeypatch.setattr(requests.sessions, "Session", FakeSession)
    data_fetcher._install_requests_default_timeout(7)
    session = FakeSession()
    session.request("GET", "https://example.invalid")
    session.request("GET", "https://example.invalid", timeout=2)
    assert calls == [7, 2]


def long_daily_df(end_date="2026-07-01", periods=130):
    dates = pd.bdate_range(end=pd.to_datetime(end_date), periods=periods)
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "open": [10.0] * periods,
            "close": [10.0] * periods,
            "high": [11.0] * periods,
            "low": [9.0] * periods,
            "volume": [1000] * periods,
            "amount": [500_000_000] * periods,
        }
    )


SETTINGS = {
    "initial_cash": 30000,
    "data_fetch": {"max_retries": 2, "retry_backoff_seconds": [1, 3], "request_interval_seconds": 0, "min_history_days": 1},
    "cache": {"max_cache_age_trade_days": 1},
}

STABLE_FETCH_SETTINGS = {
    "initial_cash": 30000,
    "data_fetch": {
        "max_retries": 3,
        "retry_backoff_seconds": [1, 3, 6],
        "request_interval_seconds": 0,
        "jitter_seconds": 0,
        "batch_size": 0,
        "batch_pause_seconds": 0,
        "min_history_days": 1,
    },
    "cache": {"max_cache_age_trade_days": 1},
}


def test_expected_trade_date_before_close_uses_previous_trade_day():
    date, status, _ = expected_trade_date(
        datetime(2026, 7, 2, 0, 57),
        ["2026-07-01", "2026-07-02"],
    )
    assert status == "fresh"
    assert date == "2026-07-01"


def test_expected_trade_date_intraday_uses_previous_trade_day():
    date, status, _ = expected_trade_date(
        datetime(2026, 7, 2, 9, 49),
        ["2026-07-01", "2026-07-02"],
    )
    assert status == "fresh"
    assert date == "2026-07-01"


def test_expected_trade_date_after_close_uses_today():
    date, _, _ = expected_trade_date(
        datetime(2026, 7, 2, 15, 30),
        ["2026-07-01", "2026-07-02"],
    )
    assert date == "2026-07-02"


def test_expected_trade_date_weekend_uses_latest_trade_day():
    date, _, _ = expected_trade_date(
        datetime(2026, 7, 4, 12, 0),
        ["2026-07-01", "2026-07-02", "2026-07-03"],
    )
    assert date == "2026-07-03"


def test_trade_day_diff_uses_trade_calendar_not_calendar_days():
    assert trade_day_diff("2026-07-03", "2026-07-06", ["2026-07-03", "2026-07-06"]) == 1


def test_first_two_fail_third_success_is_fresh(monkeypatch):
    calls = {"count": 0}
    sleeps = []

    def fake_fetcher(**kwargs):
        calls["count"] += 1
        if calls["count"] < 3:
            raise ConnectionError("RemoteDisconnected")
        return daily_df("2026-07-01")

    monkeypatch.setattr(data_fetcher, "_write_cache", lambda symbol, df: None)
    result = fetch_stock_daily(
        "600000",
        "测试股份",
        settings=SETTINGS,
        expected_trade_date_value="2026-07-01",
        trade_dates=["2026-07-01"],
        sleep_fn=sleeps.append,
        fetcher=fake_fetcher,
    )
    assert result.final_source == "fresh"
    assert result.error_category == ""
    assert calls["count"] == 3
    assert sleeps == [1, 3]


def test_first_three_fail_fourth_success_is_fresh(monkeypatch):
    calls = {"count": 0}
    sleeps = []

    def fake_fetcher(**kwargs):
        calls["count"] += 1
        if calls["count"] < 4:
            raise ConnectionError("RemoteDisconnected")
        return daily_df("2026-07-01")

    monkeypatch.setattr(data_fetcher, "_write_cache", lambda symbol, df: None)
    result = fetch_stock_daily(
        "600000",
        "测试股份",
        settings=STABLE_FETCH_SETTINGS,
        expected_trade_date_value="2026-07-01",
        trade_dates=["2026-07-01"],
        sleep_fn=sleeps.append,
        fetcher=fake_fetcher,
    )
    assert result.final_source == "fresh"
    assert result.success_attempt == 4
    assert result.attempt_count == 4
    assert calls["count"] == 4
    assert sleeps == [1, 3, 6]


def test_three_failures_with_valid_cache_returns_cache():
    def fake_fetcher(**kwargs):
        raise ConnectionError("RemoteDisconnected")

    result = fetch_stock_daily(
        "600000",
        "测试股份",
        settings=SETTINGS,
        expected_trade_date_value="2026-07-01",
        trade_dates=["2026-07-01"],
        sleep_fn=lambda _: None,
        fetcher=fake_fetcher,
        cache_reader=lambda symbol: daily_df("2026-07-01"),
    )
    assert result.final_source == "cache"
    assert result.used_cache is True
    assert result.error_category == "network_error"


def test_four_failures_with_valid_cache_returns_cache_with_fallback_reason():
    def fake_fetcher(**kwargs):
        raise ConnectionError("RemoteDisconnected")

    result = fetch_stock_daily(
        "600000",
        "测试股份",
        settings=STABLE_FETCH_SETTINGS,
        expected_trade_date_value="2026-07-01",
        trade_dates=["2026-07-01"],
        sleep_fn=lambda _: None,
        fetcher=fake_fetcher,
        cache_reader=lambda symbol: daily_df("2026-07-01"),
    )
    assert result.final_source == "cache"
    assert result.attempt_count == 4
    assert result.fallback_reason == "all_retries_failed_network_error"


def test_primary_stock_source_failure_fallback_success_is_clean_fresh(monkeypatch):
    monkeypatch.setattr(data_fetcher, "_write_cache", lambda symbol, df: None)
    settings = {
        "data_fetch": {"max_retries": 0, "request_interval_seconds": 0, "min_history_days": 120},
        "cache": {"max_cache_age_trade_days": 1},
        "data_sources": {"stock_daily": {"primary": "primary", "fallback": ["fallback", "local_cache"]}},
    }

    def primary(**kwargs):
        raise ConnectionError("RemoteDisconnected")

    def fallback(**kwargs):
        return long_daily_df("2026-07-01", 130)

    result = fetch_stock_daily(
        "600000",
        "测试股份",
        settings=settings,
        expected_trade_date_value="2026-07-01",
        trade_dates=["2026-07-01"],
        sleep_fn=lambda _: None,
        source_fetchers={"primary": primary, "fallback": fallback},
    )
    assert result.final_source == "fresh"
    assert result.error_category == ""
    assert result.data_source_name == "fallback"
    assert result.fallback_source_used == "fallback"
    assert result.source_attempts[0]["error_category"] == "network_error"
    assert result.source_attempts[-1]["status"] == "success"


def test_primary_data_not_ready_fallback_success_is_fresh(monkeypatch):
    monkeypatch.setattr(data_fetcher, "_write_cache", lambda symbol, df: None)
    settings = {
        "data_fetch": {"max_retries": 0, "request_interval_seconds": 0, "min_history_days": 120},
        "cache": {"max_cache_age_trade_days": 1},
        "data_sources": {"stock_daily": {"primary": "primary", "fallback": ["fallback", "local_cache"]}},
    }
    result = fetch_stock_daily(
        "600000",
        "测试股份",
        settings=settings,
        expected_trade_date_value="2026-07-02",
        trade_dates=["2026-07-01", "2026-07-02"],
        sleep_fn=lambda _: None,
        source_fetchers={
            "primary": lambda **kwargs: long_daily_df("2026-07-01", 130),
            "fallback": lambda **kwargs: long_daily_df("2026-07-02", 130),
        },
    )
    assert result.final_source == "fresh"
    assert result.error_category == ""
    assert result.data_source_name == "fallback"
    assert result.source_attempts[0]["error_category"] == "data_not_ready"


def test_insufficient_history_after_trim_is_failed(monkeypatch):
    monkeypatch.setattr(data_fetcher, "_write_cache", lambda symbol, df: None)
    result = fetch_stock_daily(
        "600000",
        "测试股份",
        settings={"data_fetch": {"max_retries": 0, "min_history_days": 120}, "cache": {"max_cache_age_trade_days": 1}},
        expected_trade_date_value="2026-07-01",
        trade_dates=["2026-07-01"],
        sleep_fn=lambda _: None,
        fetcher=lambda **kwargs: long_daily_df("2026-07-01", 100),
        cache_reader=lambda symbol: pd.DataFrame(),
    )
    assert result.final_source == "failed"
    assert result.error_category == "insufficient_history"


def test_fallback_schema_is_standardized(monkeypatch):
    monkeypatch.setattr(data_fetcher, "_write_cache", lambda symbol, df: None)
    periods = 130
    dates = pd.bdate_range(end=pd.to_datetime("2026-07-01"), periods=periods)
    fallback_df = pd.DataFrame(
        {
            "日期": dates.strftime("%Y-%m-%d"),
            "开盘": [10] * periods,
            "最高": [11] * periods,
            "最低": [9] * periods,
            "收盘": [10] * periods,
            "成交量": [1000] * periods,
            "成交额": [500_000_000] * periods,
        }
    )
    result = fetch_stock_daily(
        "600000",
        "测试股份",
        settings={
            "data_fetch": {"max_retries": 0, "min_history_days": 120},
            "cache": {"max_cache_age_trade_days": 1},
            "data_sources": {"stock_daily": {"primary": "primary", "fallback": ["fallback", "local_cache"]}},
        },
        expected_trade_date_value="2026-07-01",
        trade_dates=["2026-07-01"],
        sleep_fn=lambda _: None,
        source_fetchers={"primary": lambda **kwargs: (_ for _ in ()).throw(ConnectionError("RemoteDisconnected")), "fallback": lambda **kwargs: fallback_df},
    )
    assert result.final_source == "fresh"
    assert list(result.data.columns[:7]) == ["date", "open", "high", "low", "close", "volume", "amount"]
    assert result.data["amount"].iloc[-1] == 500_000_000


def test_core_index_primary_failure_fallback_success_is_not_cache():
    settings = {
        "data_fetch": {"min_history_days": 120},
        "data_sources": {"index_daily": {"primary": "primary", "fallback": ["fallback", "local_cache"]}},
    }
    result = fetch_index_daily(
        "000300",
        "沪深300",
        "sh000300",
        expected_trade_date_value="2026-07-01",
        settings=settings,
        source_fetchers={
            "primary": lambda **kwargs: (_ for _ in ()).throw(ConnectionError("RemoteDisconnected")),
            "fallback": lambda **kwargs: long_daily_df("2026-07-01", 130),
        },
    )
    assert result.final_source == "fresh"
    assert result.data_source_name == "fallback"
    assert result.fallback_source_used == "fallback"
    assert result.source_attempts[0]["error_category"] == "network_error"


def test_three_failures_without_cache_returns_failed():
    def fake_fetcher(**kwargs):
        raise ConnectionError("RemoteDisconnected")

    result = fetch_stock_daily(
        "600000",
        "测试股份",
        settings=SETTINGS,
        expected_trade_date_value="2026-07-01",
        trade_dates=["2026-07-01"],
        sleep_fn=lambda _: None,
        fetcher=fake_fetcher,
        cache_reader=lambda symbol: pd.DataFrame(),
    )
    assert result.final_source == "failed"
    assert result.error_category == "cache_missing"


def test_stale_cache_is_failed_by_trade_day_age():
    def fake_fetcher(**kwargs):
        raise ConnectionError("RemoteDisconnected")

    result = fetch_stock_daily(
        "600000",
        "测试股份",
        settings=SETTINGS,
        expected_trade_date_value="2026-07-06",
        trade_dates=["2026-07-01", "2026-07-02", "2026-07-03", "2026-07-06"],
        sleep_fn=lambda _: None,
        fetcher=fake_fetcher,
        cache_reader=lambda symbol: daily_df("2026-07-02"),
    )
    assert result.final_source == "failed"
    assert result.error_category == "no_expected_trade_date_data"


def test_success_without_expected_trade_date_data_is_not_fresh(monkeypatch):
    monkeypatch.setattr(data_fetcher, "_write_cache", lambda symbol, df: None)
    result = fetch_stock_daily(
        "600000",
        "测试股份",
        settings=SETTINGS,
        expected_trade_date_value="2026-07-02",
        trade_dates=["2026-07-01", "2026-07-02"],
        sleep_fn=lambda _: None,
        fetcher=lambda **kwargs: daily_df("2026-07-01"),
    )
    assert result.final_source == "failed"
    assert result.error_category == "data_not_ready"
    assert result.is_expected_trade_date is False


def test_intraday_future_daily_bar_is_trimmed_and_kept_fresh(monkeypatch):
    monkeypatch.setattr(data_fetcher, "_write_cache", lambda symbol, df: None)
    result = fetch_stock_daily(
        "600000",
        "测试股份",
        settings=SETTINGS,
        expected_trade_date_value="2026-07-01",
        trade_dates=["2026-07-01", "2026-07-02"],
        sleep_fn=lambda _: None,
        fetcher=lambda **kwargs: multi_daily_df(["2026-07-01", "2026-07-02"]),
    )
    assert result.final_source == "fresh"
    assert result.error_category == ""
    assert result.is_expected_trade_date is True
    assert result.raw_latest_date == "2026-07-02"
    assert result.effective_latest_date == "2026-07-01"
    assert result.latest_date == "2026-07-01"
    assert result.trimmed_future_rows_count == 1
    assert list(result.data["date"].dt.strftime("%Y-%m-%d")) == ["2026-07-01"]


def test_intraday_future_only_daily_bar_is_no_expected_trade_date(monkeypatch):
    monkeypatch.setattr(data_fetcher, "_write_cache", lambda symbol, df: None)
    result = fetch_stock_daily(
        "600000",
        "测试股份",
        settings=SETTINGS,
        expected_trade_date_value="2026-07-01",
        trade_dates=["2026-07-01", "2026-07-02"],
        sleep_fn=lambda _: None,
        fetcher=lambda **kwargs: daily_df("2026-07-02"),
    )
    assert result.final_source == "failed"
    assert result.error_category == "no_expected_trade_date_data"
    assert result.raw_latest_date == "2026-07-02"
    assert result.effective_latest_date is None
    assert result.trimmed_future_rows_count == 1


def test_cache_with_future_row_trims_to_expected_and_remains_cache():
    def fake_fetcher(**kwargs):
        raise ConnectionError("RemoteDisconnected")

    result = fetch_stock_daily(
        "600000",
        "测试股份",
        settings=SETTINGS,
        expected_trade_date_value="2026-07-01",
        trade_dates=["2026-07-01", "2026-07-02"],
        sleep_fn=lambda _: None,
        fetcher=fake_fetcher,
        cache_reader=lambda symbol: multi_daily_df(["2026-07-01", "2026-07-02"]),
    )
    assert result.final_source == "cache"
    assert result.raw_latest_date == "2026-07-02"
    assert result.effective_latest_date == "2026-07-01"
    assert result.trimmed_future_rows_count == 1


def test_fetch_stocks_soft_pause_after_consecutive_network_failures():
    sleeps = []
    stock_pool = [{"symbol": f"60000{i}", "name": f"S{i}"} for i in range(6)]
    settings = {
        "data_fetch": {
            "request_interval_seconds": 0,
            "jitter_seconds": 0,
            "consecutive_failure_soft_limit": 5,
            "consecutive_failure_soft_pause_seconds": 10,
            "consecutive_failure_hard_limit": 0,
        }
    }

    def fake_stock_fetcher(symbol, name, **kwargs):
        return FetchResult(
            symbol,
            name,
            daily_df(),
            True,
            final_source="cache",
            error_category="network_error",
            attempt_logs=[{"error_category": "network_error"}],
        )

    fetch_stocks(stock_pool, settings=settings, sleep_fn=sleeps.append, stock_fetcher=fake_stock_fetcher)
    assert 10 in sleeps


def test_fetch_stocks_hard_abort_marks_remaining_symbols():
    stock_pool = [{"symbol": f"600{i:03d}", "name": f"S{i}"} for i in range(16)]
    settings = {
        "data_fetch": {
            "request_interval_seconds": 0,
            "jitter_seconds": 0,
            "consecutive_failure_soft_limit": 0,
            "consecutive_failure_hard_limit": 15,
        }
    }

    def fake_stock_fetcher(symbol, name, **kwargs):
        return FetchResult(
            symbol,
            name,
            daily_df(),
            True,
            final_source="cache",
            error_category="network_error",
            attempt_logs=[{"error_category": "network_error"}],
        )

    results = fetch_stocks(stock_pool, settings=settings, sleep_fn=lambda _: None, stock_fetcher=fake_stock_fetcher)
    assert len(results) == 16
    assert results[-1].final_source == "failed"
    assert results[-1].error_category == "data_fetch_aborted"
    assert results[-1].data_fetch_abort_reason == "too_many_consecutive_failures"


def test_fetch_stocks_batch_pause_every_full_non_final_batch():
    sleeps = []
    stock_pool = [{"symbol": f"600{i:03d}", "name": f"S{i}"} for i in range(40)]
    settings = {
        "data_fetch": {
            "request_interval_seconds": 0,
            "jitter_seconds": 0,
            "batch_size": 20,
            "batch_pause_seconds": 8,
        }
    }

    def fake_stock_fetcher(symbol, name, **kwargs):
        return FetchResult(symbol, name, daily_df(), False, final_source="fresh", is_expected_trade_date=True)

    fetch_stocks(stock_pool, settings=settings, sleep_fn=sleeps.append, stock_fetcher=fake_stock_fetcher)
    assert sleeps == [8]


def test_fetch_stats_and_quality_use_cache_ratio():
    results = [
        FetchResult("600001", "A", daily_df(), False, final_source="fresh", latest_date="2026-07-01", expected_trade_date="2026-07-01", is_expected_trade_date=True),
        FetchResult("600002", "B", daily_df(), True, final_source="cache", latest_date="2026-07-01", expected_trade_date="2026-07-01", is_expected_trade_date=True),
        FetchResult("600003", "C", daily_df(), True, final_source="cache", latest_date="2026-07-01", expected_trade_date="2026-07-01", is_expected_trade_date=True),
        FetchResult("600004", "D", daily_df(), True, final_source="cache", latest_date="2026-06-30", expected_trade_date="2026-07-01", is_expected_trade_date=False),
    ]
    stats = summarize_stock_fetches(results)
    assert stats["market_data_date"] == "2026-07-01"
    assert stats["unexpected_trade_date_count"] == 1
    assert compute_data_quality_level(stock_fetch_stats=stats) == "critical_cache"


def _index_result(code, date="2026-07-01", final_source="fresh"):
    return {"core": code in {"000300", "000001", "399001"}, "result": FetchResult(code, code, long_daily_df(date, 130), False, final_source=final_source, latest_date=date, effective_latest_date=date, expected_trade_date=date, is_expected_trade_date=True)}


def test_market_data_ready_after_close_lagging_is_not_ready():
    index_results = [_index_result("000300"), _index_result("000001"), _index_result("399001")]
    stats = {
        "total_symbols": 10,
        "ready_ratio": 0.0,
        "ready_ratio_text": "0/10 = 0.0%",
        "data_not_ready_count": 10,
        "network_error_count": 0,
        "data_fetch_abort_reason": "",
        "cache_fallback_ratio": 0,
    }
    report = _market_data_ready_report(index_results, stats, "2026-07-02", datetime(2026, 7, 2, 15, 31), {"daily_signal": {"earliest_reliable_run_time": "16:30"}})
    assert report["market_data_ready_status"] == "not_ready_after_close"
    assert report["recommended_rerun_time"] == "建议 16:30 后重跑"


def test_market_data_ready_requires_core_ready_ratio_and_no_abort():
    index_results = [_index_result("000300", "2026-07-02"), _index_result("000001", "2026-07-02"), _index_result("399001", "2026-07-02")]
    stats = {"total_symbols": 10, "ready_ratio": 0.8, "ready_ratio_text": "8/10 = 80.0%", "data_not_ready_count": 0, "network_error_count": 0, "data_fetch_abort_reason": "", "cache_fallback_ratio": 0}
    ready = _market_data_ready_report(index_results, stats, "2026-07-02", datetime(2026, 7, 2, 15, 31), {})
    assert ready["market_data_ready_status"] == "ready"
    stats["data_fetch_abort_reason"] = "too_many_consecutive_failures"
    aborted = _market_data_ready_report(index_results, stats, "2026-07-02", datetime(2026, 7, 2, 15, 31), {})
    assert aborted["market_data_ready_status"] == "source_unavailable"


def test_market_data_ready_zero_ratio_network_is_source_unavailable():
    index_results = [_index_result("000300", "2026-07-02"), _index_result("000001", "2026-07-02"), _index_result("399001", "2026-07-02")]
    stats = {"total_symbols": 10, "ready_ratio": 0.0, "ready_ratio_text": "0/10 = 0.0%", "data_not_ready_count": 0, "network_error_count": 10, "data_fetch_abort_reason": "", "cache_fallback_ratio": 0}
    report = _market_data_ready_report(index_results, stats, "2026-07-02", datetime(2026, 7, 2, 15, 31), {})
    assert report["market_data_ready_status"] == "source_unavailable"


def test_fetch_stats_attempt_distribution_and_rerun_suggestion_for_network_cache_ratio():
    results = [
        FetchResult("600001", "A", daily_df(), False, final_source="fresh", latest_date="2026-07-01", expected_trade_date="2026-07-01", is_expected_trade_date=True, success_attempt=1, data_source_name="primary", attempt_logs=[{"source_name":"primary","attempt":1,"final_source":"fresh"}], fetch_started_at="2026-07-02 09:00:00", fetch_finished_at="2026-07-02 09:00:01"),
        FetchResult("600002", "B", daily_df(), False, final_source="fresh", latest_date="2026-07-01", expected_trade_date="2026-07-01", is_expected_trade_date=True, success_attempt=2, data_source_name="primary", attempt_logs=[{"source_name":"primary","attempt":2,"final_source":"fresh"}], fetch_started_at="2026-07-02 09:00:01", fetch_finished_at="2026-07-02 09:00:03"),
        FetchResult("600003", "C", daily_df(), False, final_source="fresh", latest_date="2026-07-01", expected_trade_date="2026-07-01", is_expected_trade_date=True, success_attempt=3, data_source_name="primary", attempt_logs=[{"source_name":"primary","attempt":3,"final_source":"fresh"}], fetch_started_at="2026-07-02 09:00:03", fetch_finished_at="2026-07-02 09:00:06"),
        FetchResult("600004", "D", daily_df(), False, final_source="fresh", latest_date="2026-07-01", expected_trade_date="2026-07-01", is_expected_trade_date=True, success_attempt=4, data_source_name="fallback", attempt_logs=[{"source_name":"primary","attempt":3,"error_category":"network_error"},{"source_name":"fallback","attempt":1,"final_source":"fresh"}], fetch_started_at="2026-07-02 09:00:06", fetch_finished_at="2026-07-02 09:00:10"),
        FetchResult("600005", "E", daily_df(), True, final_source="cache", latest_date="2026-07-01", expected_trade_date="2026-07-01", is_expected_trade_date=True, error_category="network_error", attempt_logs=[{"error_category": "network_error"}], fetch_started_at="2026-07-02 09:00:10", fetch_finished_at="2026-07-02 09:00:12"),
    ]
    stats = summarize_stock_fetches(results)
    assert stats["attempt_success_distribution"] == {
        "primary_attempt_1_success": 1,
        "primary_attempt_2_success": 1,
        "primary_attempt_3_success": 1,
        "fallback_attempt_1_success": 1,
        "cache_fallback": 1,
    }
    assert stats["fetch_started_at"] == "2026-07-02 09:00:00"
    assert stats["fetch_finished_at"] == "2026-07-02 09:00:12"
    assert stats["fetch_duration_seconds"] == 12.0
    assert stats["network_error_count"] == 2
    assert stats["network_error_ratio"] == 0.4
    assert "\u91cd\u8dd1" in stats["rerun_suggestion"]
    assert compute_data_quality_level(stock_fetch_stats=stats) == "partial_cache"


def test_valid_history_cache_shortens_provider_range_but_provider_still_must_validate_fresh():
    cache = long_daily_df("2026-06-30", 130)
    requested = []

    def primary(**kwargs):
        requested.append(kwargs["start_date"])
        return daily_df("2026-07-01")

    local_settings = {
        "data_fetch": {"max_retries": 0, "min_history_days": 120},
        "cache": {"max_cache_age_trade_days": 1},
        "data_sources": {"stock_daily": {"primary": "primary", "fallback": ["local_cache"]}},
    }
    result = fetch_stock_daily(
        "600001", "A", settings=local_settings,
        expected_trade_date_value="2026-07-01",
        trade_dates=["2026-06-30", "2026-07-01"],
        cache_reader=lambda _: cache,
        source_fetchers={"primary": primary},
    )

    assert requested == ["20260630"]
    assert result.final_source == "fresh"
    assert result.provider_success is True
    assert result.daily_history_cache_hit is True
    assert len(result.data) >= 120


def test_provider_health_demotion_is_session_only(monkeypatch):
    orders = []

    def fake_fetch(symbol, name, **kwargs):
        order = list(kwargs.get("source_order") or [])
        orders.append(order)
        return FetchResult(
            symbol, name, daily_df(), False, final_source="fresh",
            effective_latest_date="2026-07-01", expected_trade_date="2026-07-01",
            is_expected_trade_date=True, data_source_name="fallback",
            attempt_logs=[
                {"source_name": "primary", "attempt": 1, "error_category": "network_error"},
                {"source_name": "fallback", "attempt": 1, "final_source": "fresh"},
            ],
        )

    monkeypatch.setattr(data_fetcher, "fetch_stock_daily", fake_fetch)
    local_settings = {
        "data_fetch": {"request_interval_seconds": 0, "max_retries": 2, "provider_circuit_breaker_failure_symbols": 3},
        "data_sources": {"stock_daily": {"primary": "primary", "fallback": ["fallback", "local_cache"]}},
    }
    pool = [{"symbol": f"60000{i}", "name": str(i)} for i in range(4)]
    first = fetch_stocks(pool, settings=local_settings, expected_trade_date_value="2026-07-01")
    assert orders[:3] == [["primary", "fallback"]] * 3
    assert orders[3] == ["fallback", "primary"]
    assert first[3].provider_health_scope == "session_only"

    orders.clear()
    fetch_stocks(pool[:1], settings=local_settings, expected_trade_date_value="2026-07-01")
    assert orders[0] == ["primary", "fallback"]


def test_cache_ratio_above_threshold_keeps_critical_and_rerun_suggestion():
    results = [
        FetchResult("600001", "A", daily_df(), False, final_source="fresh", latest_date="2026-07-01", expected_trade_date="2026-07-01", is_expected_trade_date=True),
        FetchResult("600002", "B", daily_df(), True, final_source="cache", latest_date="2026-07-01", expected_trade_date="2026-07-01", is_expected_trade_date=True, error_category="network_error", attempt_logs=[{"error_category": "network_error"}]),
        FetchResult("600003", "C", daily_df(), True, final_source="cache", latest_date="2026-07-01", expected_trade_date="2026-07-01", is_expected_trade_date=True, error_category="network_error", attempt_logs=[{"error_category": "network_error"}]),
    ]
    stats = summarize_stock_fetches(results)
    assert stats["cache_fallback_ratio"] > 0.20
    assert "\u91cd\u8dd1" in stats["rerun_suggestion"]
    assert compute_data_quality_level(stock_fetch_stats=stats) == "critical_cache"


def test_fetch_summary_rows_and_cache_notes_include_diagnostics():
    stock_stats = {
        "fetch_started_at": "2026-07-02 09:00:00",
        "fetch_finished_at": "2026-07-02 09:00:12",
        "fetch_duration_seconds": 12.0,
        "attempt_success_distribution": {"attempt_1_success": 1, "cache_fallback": 1},
        "network_error_count": 1,
        "network_error_ratio": 0.5,
        "data_fetch_abort_reason": "",
        "rerun_suggestion": "\u7f13\u5b58\u56de\u9000\u6bd4\u4f8b\u8fc7\u9ad8\uff0c\u5efa\u8bae\u7a0d\u540e\u91cd\u8dd1\u3002",
        "future_rows_trimmed_symbols_count": 0,
        "cache_fallback_symbols": [
            {
                "symbol": "600000",
                "name": "\u6d4b\u8bd5\u80a1\u4efd",
                "fallback_reason": "all_retries_failed_network_error",
                "hit_temp_candidate": True,
                "hit_style_sample": True,
                "hit_current_holding": True,
            }
        ],
        "failed_symbols": [],
    }
    rows = _append_fetch_summary_rows([], stock_stats)
    row_names = [row.get("\u6570\u636e\u6a21\u5757") or row.get("??????") for row in rows]
    assert "\u6570\u636e\u5c31\u7eea\u72b6\u6001" in row_names
    assert any("attempt_1_success=1" in (row.get("\u8bf4\u660e") or row.get("???") or "") for row in rows)
    notes = _cache_notes(stock_stats, None, "critical_cache", "fresh")
    joined = "\n".join(notes)
    assert "\u547d\u4e2d\u4e34\u65f6\u5019\u9009" in joined
    assert "\u547d\u4e2d\u98ce\u683c\u6837\u672c" in joined
    assert "\u547d\u4e2d\u5f53\u524d\u6301\u4ed3" in joined
    assert "\u91cd\u8dd1\u5efa\u8bae" in joined


def test_trimmed_future_rows_do_not_degrade_quality_or_market_date():
    results = [
        FetchResult(
            "600001",
            "A",
            daily_df("2026-07-01"),
            False,
            final_source="fresh",
            latest_date="2026-07-01",
            raw_latest_date="2026-07-02",
            effective_latest_date="2026-07-01",
            trimmed_future_rows_count=1,
            expected_trade_date="2026-07-01",
            is_expected_trade_date=True,
        )
    ]
    stats = summarize_stock_fetches(results)
    assert stats["market_data_date"] == "2026-07-01"
    assert stats["latest_date_distribution"] == {"2026-07-01": 1}
    assert stats["future_rows_trimmed_symbols_count"] == 1
    assert stats["future_rows_trimmed_total_count"] == 1
    assert stats["unexpected_trade_date_count"] == 0
    assert stats["failed_count"] == 0
    assert compute_data_quality_level(stock_fetch_stats=stats) == "fresh"


def test_trade_calendar_fallback_caps_quality_at_partial_cache():
    assert compute_data_quality_level(trade_calendar_status="fallback_weekday") == "partial_cache"
    assert compute_data_quality_level(trade_calendar_status="failed") == "critical_cache"


def test_report_displays_real_fetch_counts_and_not_old_success_summary(tmp_path):
    settings = {
        "report_date": "2026-07-02",
        "generated_at": "2026-07-02 00:57:00",
        "expected_trade_date": "2026-07-01",
        "market_data_date": "2026-07-01",
        "data_quality_level": "critical_cache",
        "scan_type": "candidate_pool",
        "scan_count": 3,
        "initial_cash": 30000,
        "data_quality_summary": [
            {"数据模块": "个股日线", "状态": "critical_cache", "说明": "实时获取成功：1/3；缓存回退：2/3；彻底失败：0/3"},
            {"数据模块": "非预期交易日数据数量", "状态": "0", "说明": "latest_date 不等于 expected_trade_date 的有效日线数量"},
            {"数据模块": "过期缓存数量", "状态": "0", "说明": "缓存交易日差超过配置的数量"},
        ],
    }
    market = {
        "label": "绿色：可交易",
        "reasons": [],
        "market_index_table": [],
        "style_state_table": [],
        "portfolio_mode": {"portfolio_mode": "cash", "strongest_styles": "无", "potential_strong_styles": "无", "allow_new_position": False},
    }
    path, _ = write_reports(market, {"candidates": [], "watchlist": [], "forbidden": {}}, [], True, ["缓存回退股票数量：2"], settings)
    text = path.read_text(encoding="utf-8")
    assert "实时获取成功：1/3" in text
    assert "缓存回退：2/3" in text
    assert "成功数量/扫描数量" not in text
    assert "行情交易日：2026-07-01" in text


def test_data_unavailable_reason_does_not_enter_forbidden_categories():
    forbidden = categorize_forbidden(
        {
            "600000": ["行情数据不可用于交易信号"],
            "600001": ["no_expected_trade_date_data"],
        },
        {"600000": "A", "600001": "B"},
    )
    assert forbidden["overheated"]["rows"] == []
    assert forbidden["trend_broken"]["rows"] == []
    assert forbidden["abnormal_risk"]["rows"] == []
