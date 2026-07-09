from __future__ import annotations

import pandas as pd


STRUCTURES = {"breakout", "acceleration", "pullback", "distribution", "breakdown"}
BUY_ALLOWED_STRUCTURES = {"breakout", "acceleration", "pullback"}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _latest(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        raise ValueError("timing requires non-empty indicator dataframe")
    return df.iloc[-1]


def _safe_float(row: pd.Series, key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if pd.isna(value):
        return default
    return float(value)


def is_trend_strong(df: pd.DataFrame) -> bool:
    row = _latest(df)
    close = _safe_float(row, "close")
    ma20 = _safe_float(row, "ma20")
    ma60 = _safe_float(row, "ma60")
    amount = _safe_float(row, "amount")
    return bool(
        close > ma20 > ma60
        and _safe_float(row, "ma20_slope") > 0
        and _safe_float(row, "relative_strength_20d") > 0
        and amount > 0
    )


def _long_upper_shadow(row: pd.Series) -> bool:
    open_price = _safe_float(row, "open")
    close = _safe_float(row, "close")
    high = _safe_float(row, "high")
    amount = _safe_float(row, "amount")
    avg_amount = _safe_float(row, "avg_amount_20d")
    body = max(abs(close - open_price), 0.01)
    upper = high - max(open_price, close)
    close_below_high = (high - close) / high if high > 0 else 0.0
    return bool(upper / body > 1.5 and close_below_high > 0.03 and avg_amount > 0 and amount > avg_amount * 1.2)


def detect_structure_state(df: pd.DataFrame) -> dict:
    row = _latest(df)
    close = _safe_float(row, "close")
    ma5 = _safe_float(row, "ma5")
    ma20 = _safe_float(row, "ma20")
    ma60 = _safe_float(row, "ma60")
    ma20_slope = _safe_float(row, "ma20_slope")
    amount = _safe_float(row, "amount")
    avg_amount = _safe_float(row, "avg_amount_20d")
    return_5d = _safe_float(row, "return_5d")
    return_10d = _safe_float(row, "return_10d")
    rs20 = _safe_float(row, "relative_strength_20d")
    k = _safe_float(row, "kdj_k", 50)
    d = _safe_float(row, "kdj_d", 50)
    macd_dif = _safe_float(row, "macd_dif")
    macd_dea = _safe_float(row, "macd_dea")
    volume_ratio = amount / avg_amount if avg_amount > 0 else 1.0
    recent_high = float(df["high"].tail(60).max()) if "high" in df.columns and len(df) else close
    near_recent_high = close >= recent_high * 0.97 if recent_high > 0 else False

    if close < ma60 or close < ma20 or (ma20_slope < 0 and rs20 < 0):
        return {"structure_state": "breakdown", "structure_reason": "跌破关键均线或趋势斜率转弱"}

    distribution = (
        _long_upper_shadow(row)
        or (volume_ratio >= 1.5 and close < ma5)
        or (near_recent_high and volume_ratio >= 1.3 and (k < d or macd_dif < macd_dea))
        or (close < ma5 and (k < d or macd_dif < macd_dea))
    )
    if distribution:
        return {"structure_state": "distribution", "structure_reason": "出现放量分歧、长上影或短线动能转弱"}

    if ma5 > ma20 > ma60 and return_5d >= 0.06 and return_10d >= 0.08 and volume_ratio >= 1.0:
        return {"structure_state": "acceleration", "structure_reason": "均线多头且短线涨幅和成交活跃"}

    if close > ma20 > ma60 and ma20_slope > 0 and near_recent_high and volume_ratio >= 0.8:
        return {"structure_state": "breakout", "structure_reason": "站上中长期均线并接近阶段高点"}

    pullback = close > ma60 and ma20_slope >= 0 and (abs(close - ma20) / ma20 <= 0.06 if ma20 > 0 else False)
    if pullback:
        return {"structure_state": "pullback", "structure_reason": "趋势未破并回踩均线区域"}

    return {"structure_state": "breakout", "structure_reason": "趋势保持在关键均线上方"}


def calculate_entry_quality_score(df: pd.DataFrame, structure_state: str) -> dict:
    row = _latest(df)
    close = _safe_float(row, "close")
    ma5 = _safe_float(row, "ma5")
    ma20 = _safe_float(row, "ma20")
    ma60 = _safe_float(row, "ma60")
    ma20_slope = _safe_float(row, "ma20_slope")
    amount = _safe_float(row, "amount")
    avg_amount = _safe_float(row, "avg_amount_20d")
    k = _safe_float(row, "kdj_k", 50)
    d = _safe_float(row, "kdj_d", 50)
    j = _safe_float(row, "kdj_j", 50)
    volume_ratio = amount / avg_amount if avg_amount > 0 else 1.0

    ma20_deviation = abs(close - ma20) / ma20 if ma20 > 0 else 1.0
    ma5_deviation = abs(close - ma5) / ma5 if ma5 > 0 else 1.0
    if structure_state == "pullback":
        pullback_score = 30 if ma20_deviation <= 0.03 else 22 if ma20_deviation <= 0.06 else 12
    else:
        pullback_score = 30 if ma5_deviation <= 0.03 or ma20_deviation <= 0.04 else 20 if ma20_deviation <= 0.08 else 8

    if 35 <= k <= 75 and k >= d:
        kdj_score = 25
    elif 25 <= k <= 85:
        kdj_score = 18
    elif j > 100 or k < d:
        kdj_score = 8
    else:
        kdj_score = 12

    ma_score = 0
    if close > ma20 > ma60:
        ma_score += 14
    elif close > ma60:
        ma_score += 8
    if ma5 >= ma20:
        ma_score += 6
    if ma20_slope > 0:
        ma_score += 5
    ma_score = min(ma_score, 25)

    if structure_state == "pullback" and 0.6 <= volume_ratio <= 1.2:
        volume_score = 20
    elif 0.8 <= volume_ratio <= 1.5:
        volume_score = 18
    elif 1.5 < volume_ratio <= 2.0:
        volume_score = 12
    elif volume_ratio > 2.0:
        volume_score = 6
    else:
        volume_score = 10

    if structure_state == "breakdown":
        pullback_score = min(pullback_score, 8)
        ma_score = min(ma_score, 8)
    if structure_state == "distribution":
        kdj_score = min(kdj_score, 12)
        volume_score = min(volume_score, 10)

    total = pullback_score + kdj_score + ma_score + volume_score
    return {
        "entry_quality_score": round(_clamp(total, 0, 100), 2),
        "entry_quality_parts": {
            "pullback_position": round(pullback_score, 2),
            "kdj": round(kdj_score, 2),
            "ma_structure": round(ma_score, 2),
            "volume_quality": round(volume_score, 2),
        },
    }


def calculate_decision_confidence(structure_state: str, entry_quality_score: float, trend_strong: bool) -> dict:
    base = {
        "breakout": 0.70,
        "acceleration": 0.68,
        "pullback": 0.58,
        "distribution": 0.35,
        "breakdown": 0.05,
    }.get(structure_state, 0.05)
    if entry_quality_score >= 75:
        base += 0.15
    elif entry_quality_score >= 60:
        base += 0.07
    if trend_strong:
        base += 0.05
    return {"decision_confidence": round(_clamp(base, 0, 1), 4)}


def final_decision(
    structure_state: str,
    entry_quality_score: float,
    decision_confidence: float,
    trend_strong: bool,
) -> dict:
    if structure_state == "breakdown":
        return {
            "timing_decision": "AVOID",
            "timing_reason": "结构为 breakdown，破位阶段禁止买入",
            "timing_risk_tag": "",
        }
    if structure_state == "distribution" and trend_strong:
        return {
            "timing_decision": "WAIT",
            "timing_reason": "强趋势中出现 distribution，按保护规则等待确认",
            "timing_risk_tag": "high_risk_distribution_in_strong_trend",
        }
    if structure_state == "distribution":
        return {
            "timing_decision": "AVOID",
            "timing_reason": "结构为 distribution 且趋势保护不成立，规避买入",
            "timing_risk_tag": "",
        }
    if structure_state not in BUY_ALLOWED_STRUCTURES:
        return {
            "timing_decision": "AVOID",
            "timing_reason": f"结构 {structure_state} 不允许买入",
            "timing_risk_tag": "",
        }
    if entry_quality_score < 60:
        return {
            "timing_decision": "AVOID",
            "timing_reason": f"买点质量分 {entry_quality_score:.2f} 低于60，规避买入",
            "timing_risk_tag": "",
        }
    if entry_quality_score < 75:
        return {
            "timing_decision": "WAIT",
            "timing_reason": f"买点质量分 {entry_quality_score:.2f} 处于60~75，等待更好买点",
            "timing_risk_tag": "",
        }
    if decision_confidence >= 0.75:
        return {
            "timing_decision": "BUY",
            "timing_reason": f"结构允许且决策置信度 {decision_confidence:.2f} 达到BUY阈值",
            "timing_risk_tag": "",
        }
    if decision_confidence >= 0.60:
        return {
            "timing_decision": "WAIT",
            "timing_reason": f"决策置信度 {decision_confidence:.2f} 处于WAIT区间",
            "timing_risk_tag": "",
        }
    return {
        "timing_decision": "AVOID",
        "timing_reason": f"决策置信度 {decision_confidence:.2f} 低于0.60，规避买入",
        "timing_risk_tag": "",
    }


def evaluate_timing(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {
            "structure_state": "breakdown",
            "entry_quality_score": 0.0,
            "decision_confidence": 0.05,
            "timing_decision": "AVOID",
            "timing_reason": "缺少行情指标数据，无法评估交易时机",
            "trend_strong": False,
            "timing_risk_tag": "",
        }
    structure = detect_structure_state(df)
    structure_state = structure["structure_state"]
    entry = calculate_entry_quality_score(df, structure_state)
    trend_strong = is_trend_strong(df)
    confidence = calculate_decision_confidence(structure_state, entry["entry_quality_score"], trend_strong)
    decision = final_decision(
        structure_state,
        entry["entry_quality_score"],
        confidence["decision_confidence"],
        trend_strong,
    )
    return {
        "structure_state": structure_state,
        "entry_quality_score": entry["entry_quality_score"],
        "decision_confidence": confidence["decision_confidence"],
        "timing_decision": decision["timing_decision"],
        "timing_reason": decision["timing_reason"],
        "trend_strong": trend_strong,
        "timing_risk_tag": decision["timing_risk_tag"],
        "structure_reason": structure.get("structure_reason", ""),
        "entry_quality_parts": entry.get("entry_quality_parts", {}),
    }
