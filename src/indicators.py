from __future__ import annotations

import numpy as np
import pandas as pd


def moving_average(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window=window, min_periods=window).mean()


def ma_slope(ma: pd.Series, periods: int = 5) -> pd.Series:
    return ma.pct_change(periods=periods)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = (dif - dea) * 2
    return pd.DataFrame({"macd_dif": dif, "macd_dea": dea, "macd_hist": hist})


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    return result.fillna(100).clip(0, 100)


def kdj(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 9) -> pd.DataFrame:
    low_min = low.rolling(window=window, min_periods=window).min()
    high_max = high.rolling(window=window, min_periods=window).max()
    rsv = (close - low_min) / (high_max - low_min).replace(0, np.nan) * 100
    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()
    j = 3 * k - 2 * d
    return pd.DataFrame({"kdj_k": k.fillna(50), "kdj_d": d.fillna(50), "kdj_j": j.fillna(50)})


def add_indicators(df: pd.DataFrame, hs300_df: pd.DataFrame | None = None) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").reset_index(drop=True)
    out["ma5"] = moving_average(out["close"], 5)
    out["ma20"] = moving_average(out["close"], 20)
    out["ma60"] = moving_average(out["close"], 60)
    out["ma20_slope"] = ma_slope(out["ma20"])
    out = pd.concat([out, macd(out["close"])], axis=1)
    out["rsi14"] = rsi(out["close"])
    out = pd.concat([out, kdj(out["high"], out["low"], out["close"])], axis=1)
    out["avg_amount_20d"] = out["amount"].rolling(window=20, min_periods=20).mean()
    out["volatility_20d"] = out["close"].pct_change().rolling(window=20, min_periods=20).std()
    out["return_5d"] = out["close"].pct_change(5)
    out["return_10d"] = out["close"].pct_change(10)
    out["return_20d"] = out["close"].pct_change(20)
    out["ma20_deviation"] = (out["close"] - out["ma20"]) / out["ma20"]

    if hs300_df is not None and not hs300_df.empty:
        index = hs300_df.copy()
        index["date"] = pd.to_datetime(index["date"])
        index = index.sort_values("date")
        index["hs300_return_20d"] = index["close"].pct_change(20)
        out = out.merge(index[["date", "hs300_return_20d"]], on="date", how="left")
        out["hs300_return_20d"] = out["hs300_return_20d"].ffill()
        out["relative_strength_20d"] = out["return_20d"] - out["hs300_return_20d"]
    else:
        out["hs300_return_20d"] = 0.0
        out["relative_strength_20d"] = 0.0
    return out
