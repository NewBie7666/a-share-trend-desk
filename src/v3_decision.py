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


def _risk_component(
    code: str,
    name: str,
    penalty: float,
    triggered: bool,
    observed_value,
    threshold: str,
    condition_expression: str,
) -> dict:
    return {
        "risk_code": code,
        "risk_name": name,
        "raw_penalty": float(penalty),
        "triggered": bool(triggered),
        "observed_value": observed_value,
        "threshold": threshold,
        "condition_expression": condition_expression,
    }


def build_v3_risk_audit(existing_risk_result: dict) -> list[dict]:
    """Render existing risk facts without re-evaluating any risk condition."""
    output = []
    for component in existing_risk_result.get("risk_components", []) or []:
        item = dict(component)
        item["evidence"] = "命中触发条件" if bool(item.get("triggered")) else "未达到触发条件"
        output.append(item)
    return output


def _risk_level(actual_penalty: float, extreme: bool) -> str:
    if extreme:
        return "EXTREME"
    if actual_penalty <= 0:
        return "NONE"
    if actual_penalty <= 5:
        return "LOW"
    if actual_penalty <= 15:
        return "MEDIUM"
    return "HIGH"


def evaluate_v3_risk(df, structure_state: str, filter_rules: dict) -> dict:
    """Evaluate existing new-position risk and attach non-decision audit fields."""
    if df is None or df.empty:
        result = {
            "risk_adjustment": 25.0,
            "extreme_risk": False,
            "risk_flags": ["data_unavailable"],
            "risk_reason": "行情数据不可用",
            "risk_components": [_risk_component(
                "data_unavailable", "行情数据不可用", 25, True,
                "无可用日线", "必须存在有效日线", "daily_data_available == true",
            )],
            "risk_adjustment_source": "data_unavailable",
            "risk_adjustment_audit": {
                "raw_max_penalty": 25.0,
                "actual_penalty": 25.0,
                "risk_control_mode": "max_penalty",
            },
            "risk_level": "HIGH",
        }
        result["risk_components"] = build_v3_risk_audit(result)
        return result

    latest = df.iloc[-1]
    avg_amount = float(latest.get("avg_amount_20d", 0) or 0)
    amount = float(latest.get("amount", 0) or 0)
    deviation = float(latest.get("ma20_deviation", 0) or 0)
    return_10d = float(latest.get("return_10d", 0) or 0)
    volume_abnormal = avg_amount > 0 and amount >= avg_amount * 1.5
    long_upper = is_long_upper_shadow(latest, filter_rules)
    continuous_huge = has_three_day_huge_amount(df, 2.0, 3)
    consecutive_limit = _consecutive_near_limit(df)

    amount_ratio = amount / avg_amount if avg_amount > 0 else 0
    facts = [
        ("ma20_deviation", "MA20偏离", 5.0, deviation >= 0.06, f"{deviation:.2%}", ">= 6%", "ma20_deviation >= 0.06"),
        ("short_term_overheated", "短期过热", 10.0, deviation >= 0.12 or return_10d > 0.25, f"MA20偏离={deviation:.2%}; 10日涨幅={return_10d:.2%}", "MA20偏离>=12% 或 10日涨幅>25%", "ma20_deviation >= 0.12 OR return_10d > 0.25"),
        ("volume_abnormal", "异常成交量", 10.0, volume_abnormal, f"当日成交额/20日均额={amount_ratio:.2f}", ">= 1.5", "amount / avg_amount_20d >= 1.5"),
        ("long_upper_shadow", "放量长上影", 15.0, long_upper, bool(long_upper), "现有放量长上影定义", "is_long_upper_shadow == true"),
        ("continuous_huge_volume", "连续异常放量", 15.0, continuous_huge, bool(continuous_huge), "连续3日成交额均高于20日均额2倍", "has_three_day_huge_amount(2.0, 3) == true"),
        ("return_10d_over_35pct", "10日涨幅超过35%", 20.0, return_10d > 0.35, f"{return_10d:.2%}", "> 35%", "return_10d > 0.35"),
        ("ma20_deviation_over_20pct", "MA20偏离超过20%", 20.0, deviation > 0.20, f"{deviation:.2%}", "> 20%", "ma20_deviation > 0.20"),
        ("consecutive_near_limit_up", "连续接近涨停", 25.0, consecutive_limit, bool(consecutive_limit), "连续2日涨幅均不低于9.5%", "two_consecutive_returns >= 9.5%"),
    ]
    components = [_risk_component(*item) for item in facts]
    triggered = [item for item in facts if item[3]]
    flags = [item[0] for item in triggered]
    raw_max = max((item[2] for item in triggered), default=0.0)
    controlling = next((item[0] for item in reversed(triggered) if item[2] == raw_max), "none")

    independent_risks = [volume_abnormal, long_upper, deviation >= 0.12, return_10d > 0.35, continuous_huge, consecutive_limit]
    extreme = sum(bool(item) for item in independent_risks) >= 2
    actual = 25.0 if extreme else raw_max
    source = "extreme_risk_override" if extreme else controlling
    reason = "；".join(item[1] for item in triggered) if triggered else "无明显新增开仓风险"
    result = {
        "risk_adjustment": round(clamp(actual, 0, 25), 2),
        "extreme_risk": bool(extreme),
        "risk_flags": flags,
        "risk_reason": reason,
        "risk_components": components,
        "risk_adjustment_source": source,
        "risk_adjustment_audit": {
            "raw_max_penalty": round(raw_max, 2),
            "actual_penalty": round(actual, 2),
            "risk_control_mode": "extreme_override" if extreme else "max_penalty",
        },
        "risk_level": _risk_level(actual, extreme),
    }
    result["risk_components"] = build_v3_risk_audit(result)
    return result


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
        reason = f"个股因子分{stock_factor_score:.2f}低于最低要求{minimum_stock:.2f}"
    elif total >= buy_threshold:
        action = "BUY"
        reason = f"V3综合分{total:.2f}达到买入资格线{buy_threshold:.2f}"
    elif total >= watch_threshold:
        action = "WATCH"
        reason = f"V3综合分{total:.2f}处于观察区间"
    else:
        action = "IGNORE"
        reason = f"V3综合分{total:.2f}低于观察线{watch_threshold:.2f}"
    return {
        "total_score": total,
        "stock_factor_score": round(float(stock_factor_score or 0), 2),
        "market_score": round(float(market_score or 0), 2),
        "timing_score": round(float(timing_score or 0), 2),
        "risk_adjustment": round(float(risk_adjustment or 0), 2),
        "v3_action": action,
        "v3_reason": reason,
    }
