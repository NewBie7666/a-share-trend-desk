"""V3 scoring primitives. Pure calculations only; no IO and no V2 gates."""

from __future__ import annotations

from .filters import has_three_day_huge_amount, is_long_upper_shadow


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, float(value)))


def market_position_multiplier(market_score: float) -> float:
    score = float(market_score or 0)
    if score < 20:
        return 0.0
    if score < 40:
        return 0.25
    if score < 60:
        return 0.50
    if score < 80:
        return 0.80
    return 1.0


def calculate_timing_score(entry_quality_score: float, structure_state: str, config: dict) -> dict:
    adjustments = config.get("structure_adjustment", {}) or {}
    adjustment = float(adjustments.get(structure_state, 0) or 0)
    return {
        "entry_quality_score": round(float(entry_quality_score or 0), 2),
        "structure_adjustment": round(adjustment, 2),
        "timing_score": round(clamp(float(entry_quality_score or 0) + adjustment), 2),
    }


def _consecutive_near_limit(df) -> bool:
    if df is None or len(df) < 3:
        return False
    closes = df["close"].astype(float)
    daily_returns = closes.pct_change().tail(2)
    return bool(len(daily_returns) == 2 and (daily_returns >= 0.095).all())


def evaluate_v3_risk(df, structure_state: str, filter_rules: dict) -> dict:
    if df is None or df.empty:
        return {"risk_adjustment": 25.0, "extreme_risk": False, "risk_flags": ["data_unavailable"], "risk_reason": "行情数据不可用"}

    latest = df.iloc[-1]
    avg_amount = float(latest.get("avg_amount_20d", 0) or 0)
    amount = float(latest.get("amount", 0) or 0)
    deviation = float(latest.get("ma20_deviation", 0) or 0)
    return_10d = float(latest.get("return_10d", 0) or 0)
    volume_abnormal = avg_amount > 0 and amount >= avg_amount * 1.5
    long_upper = is_long_upper_shadow(latest, filter_rules)
    continuous_huge = has_three_day_huge_amount(df, 2.0, 3)
    consecutive_limit = _consecutive_near_limit(df)

    flags: list[str] = []
    adjustment = 0.0
    if deviation >= 0.06:
        flags.append("ma20_deviation")
        adjustment = max(adjustment, 5.0)
    if deviation >= 0.12 or return_10d > 0.25:
        flags.append("short_term_overheated")
        adjustment = max(adjustment, 10.0)
    if volume_abnormal:
        flags.append("volume_abnormal")
        adjustment = max(adjustment, 10.0)
    if long_upper:
        flags.append("long_upper_shadow")
        adjustment = max(adjustment, 15.0)
    if continuous_huge:
        flags.append("continuous_huge_volume")
        adjustment = max(adjustment, 15.0)
    if structure_state == "distribution":
        flags.append("high_distribution")
        adjustment = max(adjustment, 20.0)
    if return_10d > 0.35:
        flags.append("return_10d_over_35pct")
        adjustment = max(adjustment, 20.0)
    if deviation > 0.20:
        flags.append("ma20_deviation_over_20pct")
        adjustment = max(adjustment, 20.0)
    if consecutive_limit:
        flags.append("consecutive_near_limit_up")
        adjustment = max(adjustment, 25.0)

    high_distribution = structure_state == "distribution" and (volume_abnormal or long_upper or deviation >= 0.12)
    overheat_factors = [return_10d > 0.35, deviation > 0.20, continuous_huge, consecutive_limit]
    extreme = high_distribution or sum(bool(item) for item in overheat_factors) >= 2
    if extreme:
        adjustment = 25.0
    reason = "；".join(flags) if flags else "无明显新增开仓风险"
    return {
        "risk_adjustment": round(clamp(adjustment, 0, 25), 2),
        "extreme_risk": bool(extreme),
        "risk_flags": flags,
        "risk_reason": reason,
    }


def calculate_v3_score(stock_factor_score: float, market_score: float, timing_score: float, risk_adjustment: float, config: dict) -> dict:
    weights = config.get("weights", {}) or {}
    total = (
        float(stock_factor_score or 0) * float(weights.get("stock_factor", 0.50))
        + float(timing_score or 0) * float(weights.get("timing", 0.30))
        + float(market_score or 0) * float(weights.get("market", 0.20))
        - float(risk_adjustment or 0)
    )
    total = round(clamp(total), 2)
    minimum_stock = float(config.get("minimum_stock_factor_for_buy", 60))
    buy_threshold = float(config.get("buy_threshold", 75))
    watch_threshold = float(config.get("watch_threshold", 60))
    if stock_factor_score < minimum_stock:
        action = "WATCH"
        reason = f"个股因子分 {stock_factor_score:.2f} 低于最低要求 {minimum_stock:.2f}"
    elif total >= buy_threshold:
        action = "BUY"
        reason = f"V3综合分 {total:.2f} 达到买入资格线 {buy_threshold:.2f}"
    elif total >= watch_threshold:
        action = "WATCH"
        reason = f"V3综合分 {total:.2f} 处于观察区间"
    else:
        action = "IGNORE"
        reason = f"V3综合分 {total:.2f} 低于观察线 {watch_threshold:.2f}"
    return {
        "total_score": total,
        "stock_factor_score": round(float(stock_factor_score or 0), 2),
        "market_score": round(float(market_score or 0), 2),
        "timing_score": round(float(timing_score or 0), 2),
        "risk_adjustment": round(float(risk_adjustment or 0), 2),
        "v3_action": action,
        "v3_reason": reason,
    }
