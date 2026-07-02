import pandas as pd

from src.filters import evaluate_hard_filters, is_long_upper_shadow, is_risk_name
from src.indicators import add_indicators


SETTINGS = {"min_avg_amount_20d": 300_000_000}
RULES = {
    "min_history_days": 120,
    "max_5d_return": 0.15,
    "max_10d_return": 0.25,
    "max_ma20_deviation": 0.12,
    "limit_up_pct": 0.095,
    "long_upper_shadow_ratio": 1.5,
    "long_upper_shadow_amount_multiple": 1.5,
    "long_upper_shadow_close_below_high": 0.03,
    "huge_amount_multiple": 2.0,
    "huge_amount_days": 3,
}


def make_df(amount=500_000_000):
    close = [100 + i * 0.1 for i in range(130)]
    return add_indicators(
        pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=130),
                "open": close,
                "close": close,
                "high": [x + 1 for x in close],
                "low": [x - 1 for x in close],
                "volume": [1000] * 130,
                "amount": [amount] * 130,
            }
        )
    )


def test_st_name_filter():
    assert is_risk_name("*ST测试")
    df = make_df()
    reasons = evaluate_hard_filters("600000", "*ST测试", df, SETTINGS, RULES)
    assert any("ST" in reason for reason in reasons)


def test_low_amount_filter():
    df = make_df(amount=100_000_000)
    reasons = evaluate_hard_filters("600000", "测试股份", df, SETTINGS, RULES)
    assert "20日平均成交额低于3亿元" in reasons


def test_five_day_return_too_high_filter():
    df = make_df()
    df.loc[df.index[-6], "close"] = 100
    df.loc[df.index[-1], "close"] = 120
    df = add_indicators(df[["date", "open", "close", "high", "low", "volume", "amount"]])
    reasons = evaluate_hard_filters("600000", "测试股份", df, SETTINGS, RULES)
    assert "近5日涨幅大于15%" in reasons


def test_long_upper_shadow_detection():
    row = pd.Series({"open": 10, "close": 10.2, "high": 11.2, "amount": 800, "avg_amount_20d": 500})
    assert is_long_upper_shadow(row, RULES)
