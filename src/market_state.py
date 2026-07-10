from __future__ import annotations

import pandas as pd

from .position_rules import get_position_rule
from .styles import build_style_state_table
from .snapshots import snapshot_fetch_results, snapshot_names, snapshots_to_frames


def _index_state(close: float, ma20: float, ma60: float) -> str:
    if close > ma20 > ma60:
        return "strong"
    if close > ma60:
        return "neutral"
    return "weak"


def _state_from_regime(regime: str) -> str:
    return {"bull": "green", "neutral": "yellow", "bear": "red", "unknown": "unknown"}.get(regime, regime)


def _regime_from_state(state: str) -> str:
    return {"green": "bull", "yellow": "neutral", "red": "bear", "unknown": "unknown"}.get(state, state)


def _market_regime_from_indices(index_rows: list[dict], fallback_hs300_state: str) -> str:
    core_codes = {"000300", "000001", "399001"}
    valid = [
        row for row in index_rows
        if str(row.get("index_code", "")) in core_codes and row.get("state") in {"strong", "neutral", "weak"}
    ]
    if len(valid) < 2:
        return _regime_from_state(fallback_hs300_state) if fallback_hs300_state in {"green", "yellow", "red"} else "unknown"
    strong_count = sum(row["state"] == "strong" for row in valid)
    weak_count = sum(row["state"] == "weak" for row in valid)
    above_ma60_count = sum(bool(row.get("close_above_MA60")) for row in valid)
    if weak_count >= 2 or above_ma60_count < 2:
        return "bear"
    if strong_count >= 2:
        return "bull"
    return "neutral"


def _market_strength(regime: str, index_rows: list[dict]) -> float:
    base = {"bull": 85.0, "neutral": 60.0, "bear": 20.0, "unknown": 0.0}.get(regime, 0.0)
    valid = [row for row in index_rows if row.get("state") in {"strong", "neutral", "weak"}]
    if not valid:
        return base
    strong_count = sum(row["state"] == "strong" for row in valid)
    weak_count = sum(row["state"] == "weak" for row in valid)
    adjustment = (strong_count - weak_count) * 5.0 / max(len(valid), 1)
    return round(max(0.0, min(100.0, base + adjustment)), 2)


def _breadth_persistence(snapshots_by_symbol: dict, threshold: float = 0.55) -> tuple[int, list[dict]]:
    """Count consecutive complete trading days with broad upward participation."""
    rows_by_offset: list[dict] = []
    for offset in range(1, 4):
        rows = []
        rising = 0
        for snapshot in snapshots_by_symbol.values():
            indicators = getattr(snapshot, "indicators", None)
            if not getattr(snapshot, "usable_for_signal", False) or indicators is None or len(indicators) < offset:
                continue
            row = indicators.iloc[-offset]
            rows.append(row)
            if len(indicators) > offset and float(row.get("close", 0) or 0) > float(indicators.iloc[-offset - 1].get("close", 0) or 0):
                rising += 1
        total = len(rows)
        if not total:
            break
        above_ma60 = sum(float(row.get("close", 0) or 0) > float(row.get("ma60", 0) or 0) for row in rows) / total
        rising_ratio = rising / total
        rows_by_offset.append(
            {
                "date": str(rows[0].get("date", ""))[:10],
                "above_ma60_ratio": round(above_ma60, 4),
                "rising_ratio": round(rising_ratio, 4),
                "persistent": above_ma60 >= threshold and rising_ratio >= threshold,
            }
        )

    consecutive = 0
    for item in rows_by_offset:
        if not item["persistent"]:
            break
        consecutive += 1
    return consecutive, rows_by_offset


def apply_market_opportunity(market: dict, snapshots_by_symbol: dict, settings: dict | None = None) -> dict:
    """Add the V2.1 opportunity layer without changing index state or regime scoring."""
    settings = settings or {}
    metrics = market.get("market_regime_metrics", {}) or {}
    rising_ratio = float(metrics.get("rising_ratio", 0) or 0)
    above_ma60_ratio = float(metrics.get("above_ma60_ratio", 0) or 0)
    risk_score = float(metrics.get("risk_score", 0) or 0)
    style_rows = market.get("style_state_table", []) or []
    strong_style_count = sum(
        row.get("state") == "strong" and bool(row.get("strongest_eligible", False))
        for row in style_rows
    )
    breadth_persistence_days, persistence_rows = _breadth_persistence(snapshots_by_symbol)

    opportunity_score = (
        min(rising_ratio / 0.55, 1.0) * 25
        + min(above_ma60_ratio / 0.55, 1.0) * 25
        + min(strong_style_count / 3.0, 1.0) * 20
        + min(risk_score / 10.0, 1.0) * 15
        + min(breadth_persistence_days / 3.0, 1.0) * 15
    )
    structural_market = (
        rising_ratio >= 0.55
        and above_ma60_ratio >= 0.55
        and strong_style_count >= 1
        and risk_score >= 7.0
        and breadth_persistence_days >= 3
    )
    reasons = [
        f"上涨比例 {rising_ratio:.1%}",
        f"站上 MA60 比例 {above_ma60_ratio:.1%}",
        f"高置信度强势风格数量 {strong_style_count}",
        f"市场风险评分 {risk_score:.2f}/10",
        f"市场广度持续 {breadth_persistence_days} 日",
    ]

    market["base_market_state"] = market.get("market_state", "unknown")
    market["base_market_regime"] = market.get("market_regime", "")
    market["structural_market"] = structural_market
    market["market_opportunity_score"] = round(opportunity_score, 2)
    market["market_opportunity_reason"] = reasons
    market["strong_style_count"] = strong_style_count
    market["breadth_persistence_days"] = breadth_persistence_days
    market["breadth_persistence_table"] = persistence_rows

    portfolio = dict(market.get("portfolio_mode", {}) or {})
    if structural_market and portfolio.get("portfolio_mode") == "cash":
        rule = get_position_rule(settings, "structural_market")
        portfolio.update(
            {
                "final_portfolio_mode": "structural_market",
                "portfolio_mode": "structural_market",
                "max_total_position": rule["max_total_position"],
                "allow_new_position": True,
                "portfolio_mode_adjustment_reason": "指数环境偏弱，但市场广度、强势风格和风险指标满足结构性行情条件；组合模式调整为 structural_market。",
            }
        )
    market["portfolio_mode"] = portfolio
    return market


def _portfolio_mode(state: str, style_rows: list[dict], data_quality_level: str = "fresh", settings: dict | None = None) -> dict:
    settings = settings or {}
    regime = _regime_from_state(state)
    strongest = [
        row["style"]
        for row in style_rows
        if row["state"] == "strong" and row.get("sample_size", 0) >= 5 and row.get("strongest_eligible", True)
    ]
    potential = [
        row["style"]
        for row in style_rows
        if row["state"] == "strong" and (row.get("sample_size", 0) < 5 or not row.get("strongest_eligible", True))
    ]
    neutral_or_strong = [row["style"] for row in style_rows if row["state"] in {"strong", "neutral"}]
    attack_styles = {"semiconductor", "technology_hardware", "resource", "manufacturing"}
    defensive_styles = {"high_dividend", "finance_bank", "finance_insurance"}

    if regime in {"bear", "unknown"}:
        mode = "cash"
    elif any(style in attack_styles for style in strongest):
        mode = "attack"
    elif "finance_brokerage" in strongest and regime == "bull":
        mode = "attack"
    elif len(neutral_or_strong) >= 2:
        mode = "balanced"
    elif any(style in defensive_styles for style in strongest):
        mode = "defensive"
    else:
        mode = "balanced" if regime in {"bull", "neutral"} else "cash"

    rule = get_position_rule(settings, mode)
    return {
        "market_risk_state": state,
        "market_state": regime,
        "strongest_styles": ", ".join(strongest[:5]) if strongest else "无",
        "potential_strong_styles": ", ".join(potential[:5]) if potential else "无",
        "neutral_or_strong_styles": ", ".join(neutral_or_strong[:8]) if neutral_or_strong else "无",
        "raw_portfolio_mode": mode,
        "final_portfolio_mode": mode,
        "portfolio_mode_adjustment_reason": "无",
        "portfolio_mode": mode,
        "max_total_position": rule["max_total_position"],
        "allow_new_position": regime in {"bull", "neutral"} and mode != "cash",
    }


def _no_market_result(reason: str, data_quality_level: str, settings: dict | None) -> dict:
    return {
        "state": "unknown",
        "market_state": "unknown",
        "market_strength": 0.0,
        "label": "灰色：市场状态不可判定",
        "max_position": 0.0,
        "reasons": [reason],
        "metrics": {},
        "market_index_table": [],
        "style_state_table": [],
        "data_quality_level": data_quality_level,
        "portfolio_mode": _portfolio_mode("unknown", [], data_quality_level, settings),
    }


def evaluate_market_state(
    hs300_df: pd.DataFrame,
    stock_frames: dict[str, pd.DataFrame],
    rules: dict,
    names: dict[str, str] | None = None,
    confirmed_style_symbols: set[str] | None = None,
    market_index_table: list[dict] | None = None,
    data_quality_level: str = "fresh",
    settings: dict | None = None,
    fetch_results_by_symbol: dict | None = None,
) -> dict:
    if hs300_df.empty:
        return _no_market_result("沪深300数据不足，市场状态不可判定。", data_quality_level, settings)

    hs = hs300_df.copy().sort_values("date")
    hs_latest = hs.iloc[-1]
    hs_close = float(hs_latest["close"])
    hs_ma20 = float(hs_latest.get("ma20", 0) or 0)
    hs_ma60 = float(hs_latest.get("ma60", 0) or 0)
    hs_state = "green" if hs_close > hs_ma20 > hs_ma60 else "yellow" if hs_close > hs_ma60 else "red"

    latest_rows = [df.iloc[-1] for df in stock_frames.values() if not df.empty]
    total = len(latest_rows)
    above_ma20_ratio = sum(row["close"] > row.get("ma20", float("inf")) for row in latest_rows) / total if total else 0.0
    below_ma60_ratio = sum(row["close"] < row.get("ma60", 0) for row in latest_rows) / total if total else 0.0
    rs_strong_ratio = sum(row.get("relative_strength_20d", 0) > 0 for row in latest_rows) / total if total else 0.0
    metrics = {
        "hs300_close": hs_close,
        "hs300_ma20": hs_ma20,
        "hs300_ma60": hs_ma60,
        "pool_above_ma20_ratio": above_ma20_ratio,
        "pool_below_ma60_ratio": below_ma60_ratio,
        "pool_rs_strong_ratio": rs_strong_ratio,
    }
    market_index_table = market_index_table or [
        {
            "index_code": "000300",
            "index_name": "沪深300",
            "close": round(hs_close, 2),
            "MA20": round(hs_ma20, 2),
            "MA60": round(hs_ma60, 2),
            "close_above_MA20": hs_close > hs_ma20,
            "close_above_MA60": hs_close > hs_ma60,
            "state": _index_state(hs_close, hs_ma20, hs_ma60),
        }
    ]
    style_state_table = build_style_state_table(stock_frames, names or {}, confirmed_style_symbols, fetch_results_by_symbol)
    regime = _market_regime_from_indices(market_index_table, hs_state)
    state = _state_from_regime(regime)
    strength = _market_strength(regime, market_index_table)

    if regime == "bull":
        label = "绿色：市场偏多"
        reasons = ["核心指数多数处于强势结构，市场状态为 bull。"]
        max_position = 0.80
    elif regime == "neutral":
        label = "黄色：市场中性"
        reasons = ["核心指数未形成一致强势或弱势，市场状态为 neutral。"]
        max_position = 0.50
    elif regime == "bear":
        label = "红色：市场偏弱"
        reasons = ["核心指数多数处于弱势结构，市场状态为 bear，不新开仓。"]
        max_position = 0.0
    else:
        label = "灰色：市场状态不可判定"
        reasons = ["核心指数有效样本不足，市场状态不可判定。"]
        max_position = 0.0

    return {
        "state": state,
        "market_state": regime,
        "base_market_state": regime,
        "base_market_regime": regime,
        "market_strength": strength,
        "label": label,
        "max_position": max_position,
        "reasons": reasons,
        "metrics": metrics,
        "market_index_table": market_index_table,
        "style_state_table": style_state_table,
        "data_quality_level": data_quality_level,
        "portfolio_mode": _portfolio_mode(state, style_state_table, data_quality_level, settings),
    }


def evaluate_market_state_from_snapshots(
    hs300_df: pd.DataFrame,
    snapshots_by_symbol: dict,
    rules: dict,
    confirmed_style_symbols: set[str] | None = None,
    market_index_table: list[dict] | None = None,
    data_quality_level: str = "fresh",
    settings: dict | None = None,
) -> dict:
    return evaluate_market_state(
        hs300_df,
        snapshots_to_frames(snapshots_by_symbol),
        rules,
        snapshot_names(snapshots_by_symbol),
        confirmed_style_symbols,
        market_index_table,
        data_quality_level,
        settings,
        snapshot_fetch_results(snapshots_by_symbol),
    )
