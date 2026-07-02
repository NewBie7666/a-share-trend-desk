import pandas as pd

from src.signals import evaluate_holdings


SETTINGS = {"stop_loss_half": 0.08, "stop_loss_full": 0.12, "take_profit_watch": 0.25, "take_profit_strong": 0.40}
RULES = {"filters": {"long_upper_shadow_ratio": 1.5, "long_upper_shadow_amount_multiple": 1.5, "long_upper_shadow_close_below_high": 0.03}}


def holding(cost=100):
    return pd.DataFrame(
        [{"symbol": "600000", "name": "测试股份", "quantity": 100, "cost_price": cost, "buy_date": "2024-01-01", "stop_loss_price": 0, "take_profit_price": 0, "note": ""}]
    )


def stock_frame(closes):
    rows = []
    for i, close in enumerate(closes):
        rows.append(
            {
                "date": f"2024-01-{i + 1:02d}",
                "open": close,
                "close": close,
                "high": close + 1,
                "low": close - 1,
                "amount": 500_000_000,
                "avg_amount_20d": 500_000_000,
                "ma5": 100,
                "ma20": 95,
                "ma60": 90,
            }
        )
    return pd.DataFrame(rows)


def test_half_cut_at_8_percent_loss():
    actions = evaluate_holdings(holding(100), {"600000": stock_frame([92])}, {"state": "green"}, SETTINGS, RULES)
    assert actions[0]["action"] == "reduce_half"


def test_full_exit_at_12_percent_loss():
    actions = evaluate_holdings(holding(100), {"600000": stock_frame([88])}, {"state": "green"}, SETTINGS, RULES)
    assert actions[0]["action"] == "exit"


def test_full_exit_two_days_below_ma60():
    df = stock_frame([89, 88])
    actions = evaluate_holdings(holding(100), {"600000": df}, {"state": "green"}, SETTINGS, RULES)
    assert actions[0]["action"] == "exit"


def test_reduce_after_profit_and_below_ma5():
    df = stock_frame([124])
    df.loc[0, "close"] = 124
    df.loc[0, "ma5"] = 125
    actions = evaluate_holdings(holding(95), {"600000": df}, {"state": "green"}, SETTINGS, RULES)
    assert actions[0]["action"] == "reduce_half"
