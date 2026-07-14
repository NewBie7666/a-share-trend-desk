"""Pure watchlist and holding diagnostics built from an existing snapshot."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .v3_decision import build_v3_risk_audit, evaluate_v3_risk
from .watchlist_io import is_valid_stock_name


TREND_SCORES = {"strong": 90.0, "healthy": 75.0, "neutral": 60.0, "weak": 40.0, "broken": 20.0}
POSITION_SCORES = {"good_position": 85.0, "normal_position": 65.0, "chasing_risk": 35.0}

REASON_TEXTS = {
    "TREND_STRONG": "均线结构、斜率和相对强弱共同保持强势",
    "TREND_HEALTHY": "中期趋势仍然健康",
    "TREND_NEUTRAL": "趋势证据相互混合",
    "TREND_WEAK": "趋势转弱但尚未形成持续破位",
    "TREND_BROKEN": "价格、均线斜率和相对强弱共同转弱",
    "POSITION_GOOD": "当前位置接近MA20且结构适合继续观察",
    "POSITION_NORMAL": "当前位置没有明显追高风险",
    "POSITION_CHASING": "价格偏离MA20或位于高位加速分歧区",
    "RISK_PENALTY_APPLIED": "沿用V3新增开仓风险扣分",
    "HOLDING_RISK_NAME": "名称命中ST、*ST或退市风险",
    "HOLDING_CONFIRMED_BREAKDOWN": "连续5个交易日低于MA60且趋势已破坏",
    "HOLDING_REDUCE_RISK": "出现单次破位、分歧或极端技术风险",
    "DIAGNOSIS_ADD": "持仓趋势和位置满足增加关注条件",
    "DIAGNOSIS_HOLD": "持仓趋势未破坏，可继续持有观察",
    "DIAGNOSIS_CONSIDER": "无持仓股票达到重点关注条件",
    "DIAGNOSIS_WATCH": "当前条件不足，继续观察",
    "DATA_UNAVAILABLE": "缺少可用行情快照，无法完成诊断",
}


def _number(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_risk_name(name: str) -> bool:
    text = str(name or "").upper()
    return "ST" in text or "*ST" in text or "退" in str(name or "")


def _trend_health(latest: pd.Series, structure: str) -> tuple[str, float, str]:
    close = _number(latest.get("close"))
    ma20 = _number(latest.get("ma20"))
    ma60 = _number(latest.get("ma60"))
    slope = _number(latest.get("ma20_slope"))
    relative_strength = _number(latest.get("relative_strength_20d"))

    if close < ma60 and slope < 0 and relative_strength < 0:
        state = "broken"
    elif structure == "breakdown" or close < ma60 or (slope < 0 and relative_strength < 0):
        state = "weak"
    elif (
        close > ma20 > ma60
        and slope > 0
        and relative_strength > 0
        and structure in {"breakout", "acceleration"}
    ):
        state = "strong"
    elif close > ma60 and slope >= 0 and structure in {"breakout", "pullback", "acceleration"}:
        state = "healthy"
    else:
        state = "neutral"
    code = f"TREND_{state.upper()}"
    return state, TREND_SCORES[state], code


def _position_quality(df: pd.DataFrame, structure: str) -> tuple[str, float, str, float, bool]:
    latest = df.iloc[-1]
    close = _number(latest.get("close"))
    ma20 = _number(latest.get("ma20"))
    deviation = (close - ma20) / ma20 if ma20 > 0 else 0.0
    recent_high = _number(df["high"].tail(60).max(), close) if "high" in df.columns else close
    near_high = bool(recent_high > 0 and close >= recent_high * 0.97)

    if deviation > 0.12 or (structure in {"acceleration", "distribution"} and near_high):
        state = "chasing_risk"
        code = "POSITION_CHASING"
    elif structure in {"pullback", "breakout"} and -0.03 <= deviation <= 0.06:
        state = "good_position"
        code = "POSITION_GOOD"
    else:
        state = "normal_position"
        code = "POSITION_NORMAL"
    return state, POSITION_SCORES[state], code, deviation, near_high


def _confirmed_long_term_breakdown(df: pd.DataFrame, trend_state: str) -> bool:
    if trend_state != "broken" or len(df) < 5 or "ma60" not in df.columns:
        return False
    tail = df.tail(5)
    return bool((tail["close"].astype(float) < tail["ma60"].astype(float)).all())


def _holding_analysis(current_price: float | None, holding_info: dict) -> dict:
    current = _number(current_price) if current_price is not None else None
    cost_value = holding_info.get("cost_price", holding_info.get("cost"))
    shares_value = holding_info.get("shares", holding_info.get("quantity"))
    cost = _number(cost_value, 0.0)
    shares = _number(shares_value, 0.0)
    has_cost = cost > 0
    return {
        "current_price": round(current, 2) if current is not None else None,
        "cost_price": round(cost, 2) if has_cost else None,
        "shares": round(shares, 2) if shares > 0 else None,
        "profit_rate": round((current - cost) / cost, 4) if has_cost and current is not None else None,
        "distance_to_cost": round(current - cost, 2) if has_cost and current is not None else None,
        "holding_market_value": round(current * shares, 2) if shares > 0 and current is not None else None,
        "unrealized_pnl": round((current - cost) * shares, 2) if has_cost and shares > 0 and current is not None else None,
    }


def _date_text(value) -> str:
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        return ""


def _resolve_name(snapshot, holding: dict, name_cache: dict[str, str]) -> tuple[str, str]:
    symbol = str(getattr(snapshot, "symbol", holding.get("symbol", ""))).zfill(6)
    supplied = holding.get("name")
    supplied_source = str(holding.get("name_source") or "")
    if is_valid_stock_name(supplied, symbol):
        if supplied_source in {"holding", "watchlist"}:
            return str(supplied).strip(), supplied_source
        position_source = str(holding.get("position_source") or "")
        return str(supplied).strip(), "holding" if position_source in {"holding", "both"} else "watchlist"
    snapshot_name = getattr(snapshot, "name", "")
    if is_valid_stock_name(snapshot_name, symbol):
        return str(snapshot_name).strip(), "snapshot"
    fetch_name = getattr(getattr(snapshot, "fetch_result", None), "name", "")
    if is_valid_stock_name(fetch_name, symbol):
        return str(fetch_name).strip(), "fetch_result"
    cached_name = name_cache.get(symbol, "")
    if is_valid_stock_name(cached_name, symbol):
        return str(cached_name).strip(), "stock_basic_cache"
    return "", "unavailable"


def _validate_price(snapshot, market_state: dict) -> dict:
    df = getattr(snapshot, "indicators", None)
    result = getattr(snapshot, "fetch_result", None)
    effective = _date_text(getattr(result, "effective_latest_date", None))
    if not effective and result is None and df is not None and not df.empty and "date" in df.columns:
        effective = _date_text(pd.to_datetime(df["date"], errors="coerce").max())
    expected = _date_text(getattr(result, "expected_trade_date", None)) or _date_text(
        market_state.get("market_context_trade_date")
    )
    final_source = str(getattr(result, "final_source", "fresh" if result is None else "") or "")
    provider_success = getattr(result, "provider_success", None)
    used_cache = bool(getattr(result, "used_cache", False))
    if final_source == "fresh" and provider_success is not False:
        source_type = "fresh"
    elif final_source == "cache" or used_cache:
        source_type = "cache"
    else:
        source_type = "unavailable"
    provider = str(getattr(result, "data_source_name", "") or "")

    matched = pd.DataFrame()
    if df is not None and not df.empty and "date" in df.columns and effective:
        dates = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        matched = df.loc[dates == effective]
    last_available_price = None
    last_available_date = ""
    if df is not None and not df.empty and "date" in df.columns:
        valid_dates = pd.to_datetime(df["date"], errors="coerce")
        if valid_dates.notna().any():
            last_index = valid_dates.idxmax()
            last_available_date = _date_text(df.loc[last_index, "date"])
            last_available_price = round(_number(df.loc[last_index, "close"]), 4)

    stale = bool(expected and effective != expected)
    if not effective or matched.empty or source_type == "unavailable":
        status = "STALE" if effective or stale else "UNAVAILABLE"
        return {
            "current_price": None,
            "diagnosis_trade_date": effective,
            "price_trade_date": effective,
            "price_source_type": source_type,
            "price_provider": provider,
            "price_validation_status": status,
            "diagnosis_data_status": status,
            "last_available_price": last_available_price,
            "last_available_trade_date": last_available_date,
            "validated_frame": df,
        }
    row = matched.iloc[-1]
    if stale:
        return {
            "current_price": None,
            "diagnosis_trade_date": effective,
            "price_trade_date": effective,
            "price_source_type": source_type,
            "price_provider": provider,
            "price_validation_status": "STALE",
            "diagnosis_data_status": "STALE",
            "last_available_price": round(_number(row.get("close")), 4),
            "last_available_trade_date": effective,
            "validated_frame": df.loc[pd.to_datetime(df["date"], errors="coerce") <= pd.Timestamp(effective)],
        }
    return {
        "current_price": round(_number(row.get("close")), 4),
        "diagnosis_trade_date": effective,
        "price_trade_date": effective,
        "price_source_type": source_type,
        "price_provider": provider,
        "price_validation_status": "VALID",
        "diagnosis_data_status": "READY",
        "last_available_price": None,
        "last_available_trade_date": "",
        "validated_frame": df.loc[pd.to_datetime(df["date"], errors="coerce") <= pd.Timestamp(effective)],
    }


def _action(
    *,
    holding_status: str,
    name: str,
    score: float,
    trend_state: str,
    structure: str,
    risk_level: str,
    extreme_risk: bool,
    close: float,
    ma60: float,
    confirmed_breakdown: bool,
) -> tuple[str, str]:
    high_quality = (
        score >= 80
        and trend_state in {"strong", "healthy"}
        and structure in {"pullback", "breakout"}
        and risk_level in {"NONE", "LOW"}
    )
    if holding_status != "holding":
        return ("CONSIDER", "DIAGNOSIS_CONSIDER") if high_quality else ("WATCH", "DIAGNOSIS_WATCH")
    if _is_risk_name(name):
        return "EXIT", "HOLDING_RISK_NAME"
    if confirmed_breakdown:
        return "EXIT", "HOLDING_CONFIRMED_BREAKDOWN"
    if structure in {"breakdown", "distribution"} or close < ma60 or extreme_risk:
        return "REDUCE", "HOLDING_REDUCE_RISK"
    if high_quality:
        return "ADD", "DIAGNOSIS_ADD"
    if score >= 70 and trend_state not in {"weak", "broken"}:
        return "HOLD", "DIAGNOSIS_HOLD"
    return "WATCH", "DIAGNOSIS_WATCH"


def diagnose_watchlist_stock(
    snapshot,
    market_state: dict,
    holding_info: dict | None = None,
    *,
    risk_rules: dict | None = None,
    name_cache: dict[str, str] | None = None,
) -> dict:
    """Return an advisory diagnosis without invoking candidate or holding actions."""
    holding = dict(holding_info or {})
    holding_status = str(holding.get("holding_status") or "watching")
    resolved_name, name_source = _resolve_name(snapshot, holding, name_cache or {})
    price = _validate_price(snapshot, market_state)
    base = {
        "symbol": str(getattr(snapshot, "symbol", holding.get("symbol", ""))).zfill(6),
        "name": resolved_name,
        "name_source": name_source,
        "position_source": holding.get("position_source", "watchlist"),
        "holding_status": holding_status,
        "market_context_source": market_state.get("market_context_source", "unavailable"),
        "market_score": market_state.get("market_score"),
        "market_condition": market_state.get("market_condition", "UNKNOWN"),
        "market_regime": market_state.get("market_regime", "UNKNOWN"),
        "portfolio_mode": market_state.get("portfolio_mode"),
        "v3_market_permission": market_state.get("v3_market_permission", "UNKNOWN"),
        "market_context_compatibility_mode": market_state.get("market_context_compatibility_mode", "none"),
        "timing_source": "v3_snapshot",
        "risk_source": "v3_risk_adjustment",
        **{key: value for key, value in price.items() if key != "validated_frame"},
    }
    if not getattr(snapshot, "usable_for_signal", False) or snapshot.indicators is None or snapshot.indicators.empty:
        return {
            **base,
            "stock_factor_score": None,
            "trend_state": "unknown",
            "trend_health_score": None,
            "position_state": "unknown",
            "position_quality_score": None,
            "structure_state": "",
            "raw_score_before_risk": None,
            "risk_penalty": None,
            "final_diagnosis_score": None,
            "risk_level": "UNKNOWN",
            "risk_components": [],
            "diagnosis_action": "WATCH",
            "reason_codes": ["DATA_UNAVAILABLE"],
            "reason_texts": [REASON_TEXTS["DATA_UNAVAILABLE"]],
            "holding_analysis": {},
            "diagnostic_facts": {},
        }

    df = price.get("validated_frame")
    if df is None or df.empty:
        df = snapshot.indicators
    latest = df.iloc[-1]
    timing = snapshot.timing or {}
    structure = str(timing.get("structure_state", ""))
    stock_score = _number((snapshot.score or {}).get("score"))
    trend_state, trend_score, trend_code = _trend_health(latest, structure)
    position_state, position_score, position_code, deviation, near_high = _position_quality(df, structure)
    risk = evaluate_v3_risk(df, structure, risk_rules or {})
    risk_components = build_v3_risk_audit(risk)
    raw_score = stock_score * 0.50 + trend_score * 0.30 + position_score * 0.20
    risk_penalty = _number(risk.get("risk_adjustment"))
    final_score = max(0.0, min(100.0, raw_score - risk_penalty))
    close = _number(latest.get("close"))
    ma60 = _number(latest.get("ma60"))
    confirmed = _confirmed_long_term_breakdown(df, trend_state)
    action, action_code = _action(
        holding_status=holding_status,
        name=base["name"],
        score=final_score,
        trend_state=trend_state,
        structure=structure,
        risk_level=str(risk.get("risk_level", "NONE")),
        extreme_risk=bool(risk.get("extreme_risk", False)),
        close=close,
        ma60=ma60,
        confirmed_breakdown=confirmed,
    )
    if price.get("diagnosis_data_status") != "READY":
        action, action_code = "WATCH", "DATA_UNAVAILABLE"
    reason_codes = [trend_code, position_code]
    if risk_penalty > 0:
        reason_codes.append("RISK_PENALTY_APPLIED")
    reason_codes.append(action_code)
    amount = _number(latest.get("amount"))
    avg_amount = _number(latest.get("avg_amount_20d"))
    return {
        **base,
        "stock_factor_score": round(stock_score, 2),
        "trend_state": trend_state,
        "trend_health_score": trend_score,
        "position_state": position_state,
        "position_quality_score": position_score,
        "structure_state": structure,
        "timing_decision": timing.get("timing_decision", ""),
        "timing_reason_codes": list(timing.get("timing_reason_codes", [])),
        "raw_score_before_risk": round(raw_score, 2),
        "risk_penalty": round(risk_penalty, 2),
        "final_diagnosis_score": round(final_score, 2),
        "risk_level": risk.get("risk_level", "NONE"),
        "risk_components": risk_components,
        "extreme_risk": bool(risk.get("extreme_risk", False)),
        "diagnosis_action": action,
        "reason_codes": reason_codes,
        "reason_texts": [REASON_TEXTS.get(code, code) for code in reason_codes],
        "holding_analysis": _holding_analysis(price.get("current_price"), holding) if holding_status == "holding" else {},
        "diagnostic_facts": {
            "close": round(close, 4),
            "ma20": round(_number(latest.get("ma20")), 4),
            "ma60": round(ma60, 4),
            "above_ma20": bool(close > _number(latest.get("ma20"))),
            "above_ma60": bool(close > ma60),
            "ma20_slope": round(_number(latest.get("ma20_slope")), 6),
            "relative_strength_20d": round(_number(latest.get("relative_strength_20d")), 6),
            "volume_ratio": round(amount / avg_amount, 4) if avg_amount > 0 else None,
            "ma20_deviation": round(deviation, 6),
            "near_60d_high": near_high,
            "confirmed_long_term_breakdown": confirmed,
        },
    }
