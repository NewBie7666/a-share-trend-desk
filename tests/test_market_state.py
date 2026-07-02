import pandas as pd

from src.data_quality import compute_data_quality_report
from src.market_state import evaluate_market_state
from src.signals import evaluate_holdings, generate_buy_signals
from src.styles import identify_best_style
from src.position_rules import operation_summary


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


def frame(close=20, ma20=None, ma60=None, slope=0.01, rs=0.06, amount=800_000_000, volume=1000):
    ma20 = close - 1 if ma20 is None else ma20
    ma60 = close - 0.5 if ma60 is None else ma60
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
                "ma5": close - 0.2,
                "ma20": ma20,
                "ma60": ma60,
                "ma20_slope": slope,
                "macd_dif": 2,
                "macd_dea": 1,
                "rsi14": 60,
                "avg_amount_20d": 500_000_000,
                "relative_strength_20d": rs,
            }
        ]
    )


def index_row(code, state="strong"):
    values = {
        "strong": (100, 95, 90),
        "neutral": (100, 105, 90),
        "weak": (80, 90, 95),
    }[state]
    close, ma20, ma60 = values
    return {
        "index_code": code,
        "index_name": code,
        "close": close,
        "MA20": ma20,
        "MA60": ma60,
        "close_above_MA20": close > ma20,
        "close_above_MA60": close > ma60,
        "state": state,
    }


def test_finance_style_detection_is_split():
    assert identify_best_style("600030", "中信证券") == "finance_brokerage"
    assert identify_best_style("601211", "国泰海通") == "finance_brokerage"
    assert identify_best_style("600036", "招商银行") == "finance_bank"
    assert identify_best_style("601398", "工商银行") == "finance_bank"
    assert identify_best_style("601318", "中国平安") == "finance_insurance"
    assert identify_best_style("601336", "新华保险") == "finance_insurance"


def test_market_regime_bull_neutral_bear_and_no_active_mode():
    stocks = {"600030": frame(), "601211": frame(), "601688": frame(), "600999": frame(), "601066": frame()}
    names = {symbol: "中信证券" for symbol in stocks}
    hs = frame(100, 95, 90)

    bull = evaluate_market_state(hs, stocks, RULES, names, set(stocks), market_index_table=[index_row("000300"), index_row("000001"), index_row("399001")])
    assert bull["market_state"] == "bull"
    assert bull["portfolio_mode"]["portfolio_mode"] != "active"
    assert bull["portfolio_mode"]["allow_new_position"] is True

    neutral = evaluate_market_state(hs, stocks, RULES, names, set(stocks), market_index_table=[index_row("000300", "neutral"), index_row("000001", "strong"), index_row("399001", "neutral")])
    assert neutral["market_state"] == "neutral"

    bear = evaluate_market_state(hs, stocks, RULES, names, set(stocks), market_index_table=[index_row("000300", "weak"), index_row("000001", "weak"), index_row("399001", "neutral")])
    assert bear["market_state"] == "bear"
    assert bear["portfolio_mode"]["portfolio_mode"] == "cash"


def test_partial_or_critical_cache_does_not_change_portfolio_mode():
    stocks = {"600030": frame(), "601211": frame(), "601688": frame(), "600999": frame(), "601066": frame()}
    names = {symbol: "中信证券" for symbol in stocks}
    market_index_table = [index_row("000300"), index_row("000001"), index_row("399001")]
    partial = evaluate_market_state(frame(100, 95, 90), stocks, RULES, names, set(stocks), market_index_table=market_index_table, data_quality_level="partial_cache")
    critical = evaluate_market_state(frame(100, 95, 90), stocks, RULES, names, set(stocks), market_index_table=market_index_table, data_quality_level="critical_cache")
    assert partial["portfolio_mode"]["portfolio_mode"] == critical["portfolio_mode"]["portfolio_mode"]
    assert partial["portfolio_mode"]["portfolio_mode_adjustment_reason"] == "无"


def test_same_style_allows_only_one_candidate():
    market = {"state": "green", "market_state": "bull", "market_strength": 85, "portfolio_mode": {"portfolio_mode": "attack", "max_total_position": 0.80, "strongest_styles": "finance_brokerage"}, "style_state_table": [{"style": "finance_brokerage", "state": "strong", "sample_size": 5}]}
    signals = generate_buy_signals(
        market,
        {"600030": frame(20), "601211": frame(18)},
        {"600030": "中信证券", "601211": "国泰海通"},
        {"600030": [], "601211": []},
        settings(),
        RULES,
    )
    assert len(signals["candidates"]) == 1
    assert signals["candidates"][0]["best_style"] == "finance_brokerage"
    assert signals["watchlist"]


def test_other_style_never_enters_candidate():
    market = {"state": "green", "market_state": "bull", "portfolio_mode": {"portfolio_mode": "attack", "strongest_styles": "无"}}
    signals = generate_buy_signals(
        market,
        {"600001": frame()},
        {"600001": "未知股份"},
        {"600001": []},
        settings(),
        RULES,
    )
    assert signals["candidates"] == []
    assert "风格未知" in signals["watchlist"][0]["reason"]


def test_minimum_trade_guarantee_allows_one_low_score_neutral_candidate():
    market = {
        "state": "green",
        "market_state": "bull",
        "market_strength": 60,
        "portfolio_mode": {"portfolio_mode": "balanced", "max_total_position": 0.50, "strongest_styles": "无"},
        "style_state_table": [{"style": "technology_hardware", "state": "neutral", "sample_size": 8}],
    }
    df = frame(20, slope=-0.01, rs=0.0, amount=400_000_000)
    df.loc[0, "macd_dif"] = 1
    df.loc[0, "macd_dea"] = 2
    df.loc[0, "rsi14"] = 40
    signals = generate_buy_signals(market, {"600522": df}, {"600522": "中天科技"}, {"600522": []}, settings(), RULES)
    assert len(signals["candidates"]) == 1
    assert "最小候选保障" in signals["candidates"][0]["reason"]


def test_bear_market_generates_no_new_candidate():
    market = {"state": "red", "market_state": "bear", "portfolio_mode": {"portfolio_mode": "cash", "strongest_styles": "finance_brokerage"}}
    signals = generate_buy_signals(market, {"600030": frame()}, {"600030": "中信证券"}, {"600030": []}, settings(), RULES)
    assert signals["candidates"] == []


def test_data_integrity_score_changes_final_score_and_position_amount():
    market = {"state": "green", "market_state": "bull", "market_strength": 80, "portfolio_mode": {"portfolio_mode": "attack", "strongest_styles": "finance_brokerage"}, "style_state_table": [{"style": "finance_brokerage", "state": "strong", "sample_size": 5}]}
    low_integrity_settings = settings({"data_integrity_report": {"data_integrity_score": 0.5, "confidence_position_multiplier": 0.4}})
    signals = generate_buy_signals(market, {"600030": frame()}, {"600030": "中信证券"}, {"600030": []}, low_integrity_settings, RULES)
    candidate = signals["candidates"][0]
    assert candidate["data_integrity_score"] == 0.5
    assert candidate["confidence_position_multiplier"] == 0.4
    assert candidate["estimated_amount"] <= 30000 * 0.30 * 0.4


def test_critical_cache_no_longer_globally_downgrades_candidate():
    market = {"state": "green", "market_state": "bull", "market_strength": 85, "portfolio_mode": {"portfolio_mode": "balanced", "max_total_position": 0.50, "strongest_styles": "finance_brokerage"}, "style_state_table": [{"style": "finance_brokerage", "state": "strong", "sample_size": 5}]}
    signals = generate_buy_signals(market, {"600030": frame()}, {"600030": "中信证券"}, {"600030": []}, settings(), RULES, data_quality_level="critical_cache")
    assert len(signals["candidates"]) == 1
    assert signals["candidates"][0]["symbol"] == "600030"


def test_quality_level_can_be_critical_but_is_diagnostic_only():
    report = compute_data_quality_report(stock_fetch_stats={"total_symbols": 100, "cache_fallback_count": 30, "failed_count": 0})
    assert report["level"] == "critical_cache"
    assert report["partial_cache_requires_mode_degrade"] is False
    assert "不直接决定" in report["final_reason"]


def test_holdings_loss_rules_still_work():
    holdings = pd.DataFrame([{"symbol": "600030", "name": "中信证券", "quantity": 100, "cost_price": 20}])
    df = frame(17.5)
    market = {"state": "green", "market_state": "bull"}
    actions = evaluate_holdings(holdings, {"600030": df}, market, {"stop_loss_half": 0.08, "stop_loss_full": 0.12, "take_profit_watch": 0.25, "take_profit_strong": 0.40}, RULES)
    assert actions[0]["action"] == "exit"


def test_operation_summary_mentions_potential_strong_sample_issue():
    text = operation_summary("balanced", "无", "semiconductor")
    assert "潜在强风格样本不足" in text
