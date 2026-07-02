from __future__ import annotations

import numpy as np
import pandas as pd

from .styles import identify_best_style, style_reason, style_risk_note, style_state_from_score


def clamp_score(value: float) -> float:
    return float(max(0, min(100, value)))


def trend_score(row: pd.Series) -> float:
    score = 0.0
    if row["close"] > row.get("ma20", float("inf")) > row.get("ma60", float("inf")):
        score += 35
    if row.get("ma20_slope", 0) > 0:
        score += 20
    if row.get("macd_dif", 0) > row.get("macd_dea", 0):
        score += 25
    rsi = row.get("rsi14", 0)
    if 50 <= rsi <= 75:
        score += 20
    elif 45 <= rsi < 50 or 75 < rsi <= 85:
        score += 10
    return clamp_score(score)


def activity_score(row: pd.Series) -> float:
    avg = row.get("avg_amount_20d", 0)
    if avg <= 0 or pd.isna(avg):
        return 0.0
    ratio = row.get("amount", 0) / avg
    return clamp_score(40 + min(ratio, 2.0) / 2.0 * 60)


def relative_strength_score(row: pd.Series) -> float:
    rs = row.get("relative_strength_20d", 0)
    return clamp_score(50 + rs * 300)


def score_stock(symbol: str, name: str, df: pd.DataFrame) -> dict:
    row = df.iloc[-1]
    trend = trend_score(row)
    activity = activity_score(row)
    rs = relative_strength_score(row)
    financial = 100.0
    total = trend * 0.40 + activity * 0.25 + rs * 0.20 + financial * 0.15
    best_style = identify_best_style(symbol, name)
    style_scores = {
        "trend": round(trend, 2),
        "activity": round(activity, 2),
        "relative_strength": round(rs, 2),
        "financial_safety": financial,
    }
    return {
        "symbol": symbol,
        "name": name,
        "score": round(float(total), 2),
        "final_score": round(float(total), 2),
        "best_style": best_style,
        "style_state": style_state_from_score(total),
        "style_scores": style_scores,
        "trend_score": round(trend, 2),
        "activity_score": round(activity, 2),
        "relative_strength_score": round(rs, 2),
        "financial_safety_score": financial,
        "close_price": round(float(row["close"]), 2),
        "ma20": round(float(row["ma20"]), 2),
        "ma60": round(float(row["ma60"]), 2),
        "return_5d": float(row.get("return_5d", np.nan)),
        "return_10d": float(row.get("return_10d", np.nan)),
        "relative_strength_20d": float(row.get("relative_strength_20d", 0)),
        "reason": style_reason(best_style),
        "risk_note": style_risk_note(best_style),
    }
