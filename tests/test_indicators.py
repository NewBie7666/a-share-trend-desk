import pandas as pd

from src.indicators import add_indicators, macd, moving_average, rsi


def test_moving_average_basic():
    series = pd.Series([1, 2, 3, 4, 5])
    result = moving_average(series, 3)
    assert pd.isna(result.iloc[1])
    assert result.iloc[-1] == 4


def test_macd_columns_exist():
    close = pd.Series(range(1, 80), dtype=float)
    result = macd(close)
    assert {"macd_dif", "macd_dea", "macd_hist"} <= set(result.columns)
    assert result["macd_dif"].iloc[-1] > 0


def test_rsi_bounds():
    close = pd.Series([1, 2, 3, 2, 4, 5, 4, 6, 7, 8, 7, 9, 10, 11, 12], dtype=float)
    result = rsi(close)
    assert result.min() >= 0
    assert result.max() <= 100


def test_add_indicators_kdj_and_returns():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=80),
            "open": range(80, 160),
            "close": range(81, 161),
            "high": range(82, 162),
            "low": range(79, 159),
            "volume": [1000] * 80,
            "amount": [500_000_000] * 80,
        }
    )
    result = add_indicators(df)
    assert "kdj_k" in result.columns
    assert "return_5d" in result.columns
    assert result["ma60"].notna().sum() > 0
