from __future__ import annotations

import pandas as pd


CORE_INDEX_CODES = {"000300", "000001", "399001"}


def _latest(snapshot) -> pd.Series | None:
    indicators = getattr(snapshot, "indicators", None)
    if indicators is None or indicators.empty:
        return None
    return indicators.iloc[-1]


def _previous_close(snapshot) -> float | None:
    indicators = getattr(snapshot, "indicators", None)
    if indicators is None or len(indicators) < 2:
        return None
    return float(indicators.iloc[-2].get("close", 0) or 0)


def _index_condition_count(row: dict) -> int:
    close = float(row.get("close", 0) or 0)
    ma20 = float(row.get("MA20", row.get("ma20", 0)) or 0)
    ma60 = float(row.get("MA60", row.get("ma60", 0)) or 0)
    above_ma20 = bool(row.get("close_above_MA20", close > ma20 if ma20 else False))
    above_ma60 = bool(row.get("close_above_MA60", close > ma60 if ma60 else False))
    slope_positive = row.get("ma20_slope", None)
    if slope_positive is None:
        slope_ok = row.get("state") == "strong" or (above_ma20 and above_ma60)
    else:
        slope_ok = float(slope_positive or 0) > 0
    return sum([above_ma20, above_ma60, slope_ok])


def _index_trend_score(index_rows: list[dict]) -> tuple[float, list[str]]:
    core_rows = [
        row for row in index_rows
        if str(row.get("index_code", "")) in CORE_INDEX_CODES and row.get("state") != "failed"
    ]
    if not core_rows:
        return 0.0, ["核心指数样本不足，市场趋势得分为0"]
    points = {3: 40.0, 2: 30.0, 1: 15.0, 0: 0.0}
    scores = [points[_index_condition_count(row)] for row in core_rows]
    avg = round(sum(scores) / len(scores), 2)
    return avg, [f"核心指数趋势得分 {avg:.2f}/40，样本数 {len(core_rows)}"]


def _breadth_score(snapshots_by_symbol: dict) -> tuple[float, list[str], dict]:
    rows = []
    rising = 0
    for snapshot in snapshots_by_symbol.values():
        if not getattr(snapshot, "usable_for_signal", False):
            continue
        latest = _latest(snapshot)
        if latest is None:
            continue
        rows.append(latest)
        prev_close = _previous_close(snapshot)
        if prev_close and float(latest.get("close", 0) or 0) > prev_close:
            rising += 1
    total = len(rows)
    if not total:
        return 0.0, ["股票池有效样本不足，市场宽度得分为0"], {
            "above_ma20_ratio": 0.0,
            "above_ma60_ratio": 0.0,
            "rising_ratio": 0.0,
        }
    above_ma20 = sum(float(row.get("close", 0) or 0) > float(row.get("ma20", 0) or 0) for row in rows)
    above_ma60 = sum(float(row.get("close", 0) or 0) > float(row.get("ma60", 0) or 0) for row in rows)
    above_ma20_ratio = above_ma20 / total
    above_ma60_ratio = above_ma60 / total
    rising_ratio = rising / total
    strong_flags = [
        above_ma20_ratio >= 0.60,
        above_ma60_ratio >= 0.50,
        rising_ratio >= 0.50,
    ]
    medium_flags = [
        above_ma20_ratio >= 0.35,
        above_ma60_ratio >= 0.35,
        rising_ratio >= 0.35,
    ]
    if sum(strong_flags) >= 2:
        score = 30.0
    elif sum(medium_flags) >= 2:
        score = 15.0
    else:
        score = 0.0
    metrics = {
        "above_ma20_ratio": round(above_ma20_ratio, 4),
        "above_ma60_ratio": round(above_ma60_ratio, 4),
        "rising_ratio": round(rising_ratio, 4),
    }
    return score, [
        f"市场宽度得分 {score:.2f}/30，站上MA20比例 {above_ma20_ratio:.1%}，站上MA60比例 {above_ma60_ratio:.1%}，上涨比例 {rising_ratio:.1%}"
    ], metrics


def _activity_score(snapshots_by_symbol: dict) -> tuple[float, list[str], dict]:
    current_amount = 0.0
    average_amount = 0.0
    for snapshot in snapshots_by_symbol.values():
        if not getattr(snapshot, "usable_for_signal", False):
            continue
        latest = _latest(snapshot)
        if latest is None:
            continue
        current_amount += float(latest.get("amount", 0) or 0)
        average_amount += float(latest.get("avg_amount_20d", 0) or 0)
    ratio = current_amount / average_amount if average_amount > 0 else 0.0
    if ratio >= 1.0:
        score = 20.0
    elif ratio >= 0.80:
        score = 10.0
    else:
        score = 0.0
    return score, [f"成交活跃得分 {score:.2f}/20，当前成交额/20日均额={ratio:.2f}"], {
        "amount_ratio": round(ratio, 4),
        "current_amount": round(current_amount, 2),
        "average_amount_20d": round(average_amount, 2),
    }


def _risk_score(snapshots_by_symbol: dict) -> tuple[float, list[str], dict]:
    rows = []
    limit_down_like = 0
    abnormal_volatility = 0
    shrinking = 0
    for snapshot in snapshots_by_symbol.values():
        if not getattr(snapshot, "usable_for_signal", False):
            continue
        latest = _latest(snapshot)
        if latest is None:
            continue
        rows.append(latest)
        open_price = float(latest.get("open", 0) or 0)
        close = float(latest.get("close", 0) or 0)
        high = float(latest.get("high", close) or close)
        low = float(latest.get("low", close) or close)
        amount = float(latest.get("amount", 0) or 0)
        avg_amount = float(latest.get("avg_amount_20d", 0) or 0)
        if open_price > 0 and close / open_price - 1 <= -0.095:
            limit_down_like += 1
        if close > 0 and (high - low) / close >= 0.09:
            abnormal_volatility += 1
        if avg_amount > 0 and amount / avg_amount < 0.60:
            shrinking += 1
    total = len(rows)
    if not total:
        return 0.0, ["风险指标样本不足，风险得分为0"], {
            "limit_down_ratio": 0.0,
            "abnormal_volatility_ratio": 0.0,
            "shrinking_ratio": 0.0,
        }
    limit_ratio = limit_down_like / total
    volatility_ratio = abnormal_volatility / total
    shrinking_ratio = shrinking / total
    score = 10.0
    if limit_ratio >= 0.03:
        score -= 4.0
    if volatility_ratio >= 0.20:
        score -= 3.0
    if shrinking_ratio >= 0.50:
        score -= 3.0
    score = max(0.0, score)
    return score, [
        f"风险指标得分 {score:.2f}/10，疑似大跌比例 {limit_ratio:.1%}，异常波动比例 {volatility_ratio:.1%}，缩量比例 {shrinking_ratio:.1%}"
    ], {
        "limit_down_ratio": round(limit_ratio, 4),
        "abnormal_volatility_ratio": round(volatility_ratio, 4),
        "shrinking_ratio": round(shrinking_ratio, 4),
    }


def _regime_from_score(score: float) -> str:
    if score >= 80:
        return "attack"
    if score >= 60:
        return "normal"
    if score >= 40:
        return "defensive"
    return "cash"


def _permission(regime: str) -> str:
    return {
        "attack": "允许主动交易",
        "normal": "允许交易，但降低仓位",
        "defensive": "防守环境，仅允许低风险低金额候选",
        "cash": "禁止新开仓",
    }.get(regime, "市场环境不可判定")


def evaluate_market_regime(index_rows: list[dict], snapshots_by_symbol: dict) -> dict:
    index_score, index_reasons = _index_trend_score(index_rows)
    breadth_score, breadth_reasons, breadth_metrics = _breadth_score(snapshots_by_symbol)
    activity_score, activity_reasons, activity_metrics = _activity_score(snapshots_by_symbol)
    risk_score, risk_reasons, risk_metrics = _risk_score(snapshots_by_symbol)
    market_score = round(index_score + breadth_score + activity_score + risk_score, 2)
    market_regime = _regime_from_score(market_score)
    return {
        "market_score": market_score,
        "market_regime": market_regime,
        "trade_permission": _permission(market_regime),
        "market_reason": index_reasons + breadth_reasons + activity_reasons + risk_reasons,
        "market_regime_metrics": {
            "index_trend_score": index_score,
            "breadth_score": breadth_score,
            "activity_score": activity_score,
            "risk_score": risk_score,
            **breadth_metrics,
            **activity_metrics,
            **risk_metrics,
        },
    }
