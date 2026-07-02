import pandas as pd

from src.data_fetcher import FetchResult
from src.data_quality import compute_data_quality_report
from src.market_state import evaluate_market_state
from src.report import write_reports
from src.signals import generate_buy_signals


RULES = {
    "market_state": {
        "green_hs300_above_ma": 20,
        "green_pool_above_ma20_ratio": 0.60,
        "green_pool_rs_strong_ratio": 0.50,
        "yellow_hs300_above_ma": 60,
        "yellow_pool_above_ma20_ratio": 0.35,
        "red_pool_above_ma20_below_ratio": 0.30,
        "red_pool_below_ma60_ratio": 0.50,
    },
    "condition_order": {
        "trigger_close_multiple": 1.01,
        "trigger_ma20_multiple": 1.005,
        "take_profit_reduce_multiple": 1.25,
        "take_profit_strong_multiple": 1.40,
    },
}


def frame(close=20, volume=1000, amount=800_000_000):
    return pd.DataFrame(
        [
            {
                "date": "2026-07-01",
                "open": close,
                "close": close,
                "high": close + 1,
                "low": close - 1,
                "volume": volume,
                "amount": amount,
                "ma20": close - 1,
                "ma60": close - 0.5,
                "ma20_slope": 0.01,
                "macd_dif": 2,
                "macd_dea": 1,
                "rsi14": 60,
                "avg_amount_20d": 500_000_000,
                "relative_strength_20d": 0.06,
            }
        ]
    )


def settings(extra=None):
    base = {
        "initial_cash": 30000,
        "lot_size": 100,
        "account_risk_limit": 0.05,
        "max_same_style_candidates": 1,
        "minimum_trade_guarantee": {"min_candidates": 1},
        "position_rules": {
            "attack": {"max_total_position": 0.80, "max_single_position": 0.30, "max_new_candidates": 3, "min_score": 70, "max_account_risk_pct": 0.030},
            "balanced": {"max_total_position": 0.50, "max_single_position": 0.25, "max_new_candidates": 2, "min_score": 75, "max_account_risk_pct": 0.025},
            "defensive": {"max_total_position": 0.30, "max_single_position": 0.20, "max_new_candidates": 1, "min_score": 78, "max_account_risk_pct": 0.015},
            "cash": {"max_total_position": 0.0, "max_single_position": 0.0, "max_new_candidates": 0, "min_score": 999, "max_account_risk_pct": 0.000},
        },
    }
    if extra:
        base.update(extra)
    return base


def fetch(symbol, source="fresh", expected=True, empty=False):
    return FetchResult(
        symbol,
        "中信证券",
        pd.DataFrame() if empty else frame(),
        source == "cache",
        final_source=source,
        latest_date="2026-07-01" if expected else "2026-06-30",
        expected_trade_date="2026-07-01",
        is_expected_trade_date=expected,
        error="fetch failed" if source == "failed" else None,
        error_category="network_error" if source == "failed" else "",
    )


def test_small_cache_ratio_and_noncritical_stock_basic_is_not_critical():
    stats = {"total_symbols": 138, "cache_fallback_count": 8, "failed_count": 0, "stale_cache_count": 0, "ready_ratio": 0.94}
    report = compute_data_quality_report(
        stock_fetch_stats=stats,
        any_stock_used_cache=True,
        stock_basic_status="cache",
        stock_basic_is_trading_critical=False,
    )
    assert report["level"] == "partial_cache"
    assert report["data_integrity_score"] > 0.5
    assert report["partial_cache_requires_mode_degrade"] is False


def test_stock_basic_cache_without_risk_fallback_is_diagnostic_not_gate():
    report = compute_data_quality_report(stock_basic_status="cache", stock_basic_is_trading_critical=True)
    assert report["level"] in {"partial_cache", "critical_cache"}
    assert report["partial_cache_requires_mode_degrade"] is False


def test_cache_and_failed_ratio_boundaries_remain_diagnostic():
    partial = compute_data_quality_report(stock_fetch_stats={"total_symbols": 100, "cache_fallback_count": 20, "failed_count": 0})
    critical_cache = compute_data_quality_report(stock_fetch_stats={"total_symbols": 100, "cache_fallback_count": 21, "failed_count": 0})
    critical_failed = compute_data_quality_report(stock_fetch_stats={"total_symbols": 100, "cache_fallback_count": 0, "failed_count": 6})
    assert partial["level"] == "partial_cache"
    assert critical_cache["level"] == "critical_cache"
    assert critical_failed["level"] == "critical_cache"
    assert critical_cache["partial_cache_requires_mode_degrade"] is False


def test_partial_cache_does_not_degrade_portfolio_mode_in_v2():
    hs = frame(110)
    stocks = {f"60003{i}": frame(20) for i in range(5)}
    names = {symbol: "中信证券" for symbol in stocks}
    result = evaluate_market_state(
        hs,
        stocks,
        RULES,
        names,
        set(stocks),
        data_quality_level="partial_cache",
        settings=settings({"partial_cache_requires_mode_degrade": True}),
    )
    portfolio = result["portfolio_mode"]
    assert portfolio["raw_portfolio_mode"] == portfolio["final_portfolio_mode"]
    assert portfolio["portfolio_mode_adjustment_reason"] == "无"


def test_formal_candidate_has_data_source_fields():
    market = {"state": "green", "market_state": "bull", "portfolio_mode": {"portfolio_mode": "attack", "strongest_styles": "finance_brokerage"}, "style_state_table": [{"style": "finance_brokerage", "state": "strong", "sample_size": 5}]}
    signals = generate_buy_signals(market, {"600030": frame()}, {"600030": "中信证券"}, {"600030": []}, settings(), RULES, fetch_results_by_symbol={"600030": fetch("600030")})
    candidate = signals["candidates"][0]
    assert candidate["candidate_data_source"] == "fresh"
    assert candidate["candidate_latest_date"] == "2026-07-01"
    assert candidate["is_expected_trade_date"] is True


def test_candidate_cache_is_allowed_but_unexpected_date_is_downgraded():
    market = {"state": "green", "market_state": "bull", "portfolio_mode": {"portfolio_mode": "attack", "strongest_styles": "finance_brokerage"}, "style_state_table": [{"style": "finance_brokerage", "state": "strong", "sample_size": 5}]}
    signals = generate_buy_signals(market, {"600030": frame()}, {"600030": "中信证券"}, {"600030": []}, settings(), RULES, fetch_results_by_symbol={"600030": fetch("600030", source="cache")})
    assert len(signals["candidates"]) == 1
    assert signals["candidates"][0]["candidate_data_source"] == "cache"

    signals = generate_buy_signals(market, {"600030": frame()}, {"600030": "中信证券"}, {"600030": []}, settings(), RULES, fetch_results_by_symbol={"600030": fetch("600030", expected=False)})
    assert signals["candidates"] == []
    assert signals["watchlist"][0]["is_expected_trade_date"] is False


def test_zero_volume_or_amount_downgrades_candidate():
    market = {"state": "green", "market_state": "bull", "portfolio_mode": {"portfolio_mode": "attack", "strongest_styles": "finance_brokerage"}, "style_state_table": [{"style": "finance_brokerage", "state": "strong", "sample_size": 5}]}
    signals = generate_buy_signals(market, {"600030": frame(volume=0)}, {"600030": "中信证券"}, {"600030": []}, settings(), RULES, fetch_results_by_symbol={"600030": fetch("600030")})
    assert signals["candidates"] == []
    assert "成交量或成交额为0" in signals["watchlist"][0]["downgrade_reason"]


def test_failed_source_enters_data_issue_list_and_not_candidate():
    market = {"state": "green", "market_state": "bull", "portfolio_mode": {"portfolio_mode": "attack", "strongest_styles": "finance_brokerage"}, "style_state_table": [{"style": "finance_brokerage", "state": "strong", "sample_size": 5}]}
    signals = generate_buy_signals(market, {"600030": pd.DataFrame()}, {"600030": "中信证券"}, {"600030": ["行情数据不可用于交易信号"]}, settings(), RULES, fetch_results_by_symbol={"600030": fetch("600030", source="failed", empty=True)})
    assert signals["candidates"] == []
    assert signals["watchlist"] == []
    assert signals["data_issue_list"][0]["symbol"] == "600030"


def test_report_shows_v2_integrity_and_data_issue_list():
    market = {
        "label": "绿色：市场偏多",
        "market_state": "bull",
        "market_strength": 85,
        "reasons": [],
        "market_index_table": [],
        "style_state_table": [],
        "portfolio_mode": {
            "portfolio_mode": "balanced",
            "raw_portfolio_mode": "balanced",
            "final_portfolio_mode": "balanced",
            "portfolio_mode_adjustment_reason": "无",
            "strongest_styles": "finance_brokerage",
            "potential_strong_styles": "无",
            "allow_new_position": True,
        },
    }
    candidate = {
        "candidate_rank": 1,
        "symbol": "600030",
        "name": "中信证券",
        "best_style": "finance_brokerage",
        "style_state": "strong",
        "final_score": 90,
        "signal_score": 90,
        "market_strength": 85,
        "data_integrity_score": 0.7,
        "confidence_position_multiplier": 0.8,
        "candidate_data_source": "fresh",
        "candidate_latest_date": "2026-07-01",
        "is_expected_trade_date": True,
        "close_price": 20,
        "trigger_price": 20.2,
        "suggested_lots": 3,
        "suggested_shares": 300,
        "estimated_amount": 6060,
        "account_risk_pct": 0.01,
        "half_stop_price": 18.58,
        "final_exit_price": 18,
        "take_profit_reduce_price": 25.25,
        "take_profit_strong_price": 28.28,
        "rank": 1,
        "score": 90,
        "half_stop_action": "跌破该价，减半",
        "final_exit_action": "跌破该价，清仓",
        "trend_stop_price": 17.82,
        "trend_exit_action": "连续2日跌破MA60，清仓",
        "account_risk_note": "以 final_exit_price 估算账户风险",
        "reason": "券商风格较强",
        "risk_note": "注意指数风险偏好变化",
    }
    report_settings = settings(
        {
            "report_date": "2026-07-02",
            "generated_at": "2026-07-02 16:00:00",
            "expected_trade_date": "2026-07-02",
            "market_data_date": "2026-07-02",
            "data_quality_level": "critical_cache",
            "data_integrity_report": {
                "data_integrity_score": 0.7,
                "source_health": "usable",
                "confidence_level": "medium",
                "confidence_position_multiplier": 0.8,
            },
            "data_integrity_score": 0.7,
            "confidence_position_multiplier": 0.8,
            "scan_type": "candidate_pool",
            "scan_count": 1,
            "data_quality_summary": [],
            "trading_critical_reasons": ["无"],
            "non_critical_warnings": ["缓存比例偏高，只降低置信度"],
        }
    )
    path, _ = write_reports(
        market,
        {"candidates": [candidate], "watchlist": [], "forbidden": {}, "data_issue_list": [{"symbol": "600001", "name": "问题股", "failed_reason": "失败", "error_category": "network_error"}]},
        [],
        True,
        [],
        report_settings,
    )
    text = path.read_text(encoding="utf-8")
    assert "data_integrity_score" in text
    assert "confidence_reduced_mode" in text
    assert "数据完整性只降低置信度和建议金额" in text
    assert "数据问题名单 data_issue_list" in text
