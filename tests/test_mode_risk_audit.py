import pandas as pd

from src.data_fetcher import FetchResult
from src.signals import generate_buy_signals
from src.styles import build_style_state_table


RULES = {
    "condition_order": {
        "trigger_close_multiple": 1.01,
        "trigger_ma20_multiple": 1.005,
        "take_profit_reduce_multiple": 1.25,
        "take_profit_strong_multiple": 1.40,
    }
}


def frame(close=20, ma60=19.5, volume=1000, amount=800_000_000):
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
                "ma60": ma60,
                "ma20_slope": 0.01,
                "macd_dif": 2,
                "macd_dea": 1,
                "rsi14": 60,
                "avg_amount_20d": 500_000_000,
                "relative_strength_20d": 0.06,
            }
        ]
    )


def settings():
    return {
        "initial_cash": 30000,
        "lot_size": 100,
        "account_risk_limit": 0.05,
        "max_same_style_candidates": 1,
        "position_rules": {
            "attack": {"max_total_position": 0.80, "max_single_position": 0.30, "max_new_candidates": 3, "min_score": 70, "max_account_risk_pct": 0.030},
            "balanced": {"max_total_position": 0.50, "max_single_position": 0.25, "max_new_candidates": 2, "min_score": 75, "max_account_risk_pct": 0.025},
            "defensive": {"max_total_position": 0.30, "max_single_position": 0.20, "max_new_candidates": 1, "min_score": 78, "max_account_risk_pct": 0.015},
            "cash": {"max_total_position": 0.0, "max_single_position": 0.0, "max_new_candidates": 0, "min_score": 999, "max_account_risk_pct": 0.000},
        },
    }


def market(mode, style="finance_brokerage"):
    return {
        "state": "green",
        "portfolio_mode": {"portfolio_mode": mode, "strongest_styles": style},
        "market_state": "bull",
        "market_strength": 85,
        "style_state_table": [{"style": style, "state": "strong", "sample_size": 5}],
    }


def fetch(symbol, source="fresh", expected=True):
    return FetchResult(
        symbol,
        "test",
        frame(),
        source == "cache",
        final_source=source,
        latest_date="2026-07-01" if expected else "2026-06-30",
        expected_trade_date="2026-07-01",
        is_expected_trade_date=expected,
    )


def test_mode_account_risk_limits_are_applied_by_mode():
    defensive = generate_buy_signals(market("defensive"), {"600030": frame(ma60=18)}, {"600030": "broker"}, {"600030": []}, settings(), RULES)
    assert defensive["candidates"] == []
    risk_row = defensive["watchlist"][0]
    assert risk_row["review_scope"] == "candidate_risk_review"
    assert risk_row["mode_account_risk_limit"] == 0.015
    assert risk_row["account_risk_pass"] is False
    assert risk_row["original_account_risk_pct"] > 0.015
    assert risk_row["downgrade_reason"]

    balanced = generate_buy_signals(market("balanced"), {"600030": frame(ma60=18)}, {"600030": "broker"}, {"600030": []}, settings(), RULES)
    assert balanced["candidates"][0]["mode_account_risk_limit"] == 0.025
    assert balanced["candidates"][0]["account_risk_pass"] is True

    attack = generate_buy_signals(market("attack"), {"600030": frame(ma60=18.5)}, {"600030": "broker"}, {"600030": []}, settings(), RULES)
    assert attack["candidates"][0]["mode_account_risk_limit"] == 0.030
    assert attack["candidates"][0]["account_risk_pass"] is True


def test_cash_mode_has_no_formal_candidate():
    signals = generate_buy_signals(market("cash"), {"600030": frame()}, {"600030": "broker"}, {"600030": []}, settings(), RULES)
    assert signals["candidates"] == []


def test_observation_rows_have_nonblank_source_audit_fields():
    signals = generate_buy_signals(
        market("attack"),
        {"600030": frame(), "601211": frame(19, ma60=18.7)},
        {"600030": "broker", "601211": "broker"},
        {"600030": [], "601211": []},
        settings(),
        RULES,
    )
    ordinary = signals["watchlist"][0]
    assert ordinary["candidate_data_source"] == "not_reviewed"
    assert ordinary["candidate_latest_date"] == "-"
    assert ordinary["is_expected_trade_date"] == "-"
    assert ordinary["review_scope"] == "observation_only"


def test_cache_candidate_is_downgraded_with_source_scope_and_fields():
    signals = generate_buy_signals(
        market("attack"),
        {"600030": frame()},
        {"600030": "broker"},
        {"600030": []},
        settings(),
        RULES,
        fetch_results_by_symbol={"600030": fetch("600030", source="cache")},
    )
    assert signals["candidates"] == []
    row = signals["watchlist"][0]
    assert row["review_scope"] == "candidate_data_review"
    assert row["candidate_data_source"] == "cache"
    assert row["candidate_latest_date"] == "2026-07-01"
    assert row["is_expected_trade_date"] is True


def test_style_quality_and_strongest_eligibility_are_separate():
    symbols = ["600030", "601211", "601688", "600999", "601066"]
    strong_rows = build_style_state_table({symbol: frame() for symbol in symbols}, {symbol: "broker" for symbol in symbols}, set(symbols))
    strong = next(row for row in strong_rows if row["style"] == "finance_brokerage")
    assert strong["sample_quality_eligible"] is True
    assert strong["strongest_eligible"] is True

    weak_frames = {symbol: frame() for symbol in symbols}
    for df in weak_frames.values():
        df.loc[0, "close"] = 18
        df.loc[0, "ma20"] = 19
        df.loc[0, "ma60"] = 19.5
        df.loc[0, "relative_strength_20d"] = -0.2
        df.loc[0, "macd_dif"] = 0
        df.loc[0, "macd_dea"] = 1
    weak_rows = build_style_state_table(weak_frames, {symbol: "broker" for symbol in symbols}, set(symbols))
    weak = next(row for row in weak_rows if row["style"] == "finance_brokerage")
    assert weak["sample_quality_eligible"] is True
    assert weak["state"] == "weak"
    assert weak["strongest_eligible"] is False


def test_defensive_trend_candidate_has_small_probe_note(monkeypatch):
    import src.scoring as scoring

    monkeypatch.setattr(scoring, "identify_best_style", lambda symbol, name: "technology_hardware")
    signals = generate_buy_signals(
        market("defensive", "technology_hardware"),
        {"600522": frame(close=21, ma60=19.9)},
        {"600522": "trend"},
        {"600522": []},
        settings(),
        RULES,
        fetch_results_by_symbol={"600522": fetch("600522")},
    )
    assert signals["candidates"]
    assert "小仓位试探，不代表进攻仓" in signals["candidates"][0]["risk_note"]
    assert signals["candidates"][0]["account_risk_pass"] is True
