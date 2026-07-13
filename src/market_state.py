from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .position_rules import get_position_rule
from .styles import build_style_state_table
from .snapshots import snapshot_fetch_results, snapshot_names, snapshots_to_frames
from .utils import CACHE_DIR, ensure_directories


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


def _opportunity_state_path() -> Path:
    return CACHE_DIR / "market_opportunity_state.json"


def _load_opportunity_state() -> dict:
    path = _opportunity_state_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_opportunity_state(state: dict) -> None:
    ensure_directories()
    _opportunity_state_path().write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _opportunity_level(score: float) -> str:
    if score < 30:
        return "poor"
    if score < 60:
        return "normal"
    if score < 80:
        return "favorable"
    return "strong"


def _update_opportunity_state(
    *,
    current_date: str,
    conditions: dict[str, bool],
    confirm_days: int,
) -> dict:
    previous = _load_opportunity_state()
    all_passed = all(conditions.values())
    previous_date = str(previous.get("last_updated", ""))[:10]
    previous_count = int(previous.get("confirm_days", 0) or 0)
    if all_passed:
        confirm_count = previous_count if previous_date == current_date else previous_count + 1
        first_confirmed = previous.get("first_confirmed_date")
        if confirm_count >= confirm_days and not first_confirmed:
            first_confirmed = current_date
    else:
        confirm_count = 0
        first_confirmed = None
    state = {
        "confirm_days": confirm_count,
        "first_confirmed_date": first_confirmed,
        "last_updated": current_date,
        "conditions": conditions,
        "confirmed": all_passed and confirm_count >= confirm_days,
    }
    _save_opportunity_state(state)
    return state


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
    conditions = {
        "rising_ratio": rising_ratio >= 0.55,
        "above_ma60_ratio": above_ma60_ratio >= 0.55,
        "strong_style_count": strong_style_count >= 1,
        "risk_score": risk_score >= 7.0,
        "breadth_persistence": breadth_persistence_days >= 3,
    }
    reasons = [
        f"上涨比例 {rising_ratio:.1%}",
        f"站上 MA60 比例 {above_ma60_ratio:.1%}",
        f"高置信度强势风格数量 {strong_style_count}",
        f"市场风险评分 {risk_score:.2f}/10",
        f"市场广度持续 {breadth_persistence_days} 日",
    ]

    fail_reasons = [reason for key, reason in zip(conditions, reasons) if not conditions[key]]
    confirm_days = int(settings.get("market_condition_confirm_days", 3))
    current_date = str(settings.get("expected_trade_date", "")) or "unknown"
    opportunity_state = _update_opportunity_state(
        current_date=current_date,
        conditions=conditions,
        confirm_days=confirm_days,
    )
    structural_opportunity = bool(opportunity_state["confirmed"])
    market["raw_market_state"] = market.get("market_state", "unknown")
    market["base_market_state"] = market.get("market_state", "unknown")
    market["base_market_regime"] = market.get("market_regime", "")
    market["structural_market"] = structural_opportunity
    market["structural_opportunity"] = structural_opportunity
    market["market_condition"] = "structural_opportunity" if structural_opportunity else "normal"
    market["market_opportunity_score"] = round(opportunity_score, 2)
    market["market_opportunity_level"] = _opportunity_level(opportunity_score)
    market["market_opportunity_reason"] = reasons
    market["structural_trigger_conditions"] = conditions
    market["structural_fail_reasons"] = fail_reasons
    market["strong_style_count"] = strong_style_count
    market["breadth_persistence_days"] = breadth_persistence_days
    market["breadth_persistence_table"] = persistence_rows
    market["market_condition_state"] = opportunity_state

    portfolio = dict(market.get("portfolio_mode", {}) or {})
    raw_mode = portfolio.get("raw_portfolio_mode", portfolio.get("portfolio_mode", "cash"))
    data_block = bool(settings.get("trading_critical_blocker_codes", []))
    if data_block:
        cash_reason = "data_block"
    elif raw_mode == "cash":
        cash_reason = "market_risk"
    else:
        cash_reason = "unknown"
    market_regime = market.get("market_regime", "")
    adjustment_reason = "无"
    if structural_opportunity and raw_mode == "cash" and cash_reason == "market_risk" and market_regime != "cash":
        rule = get_position_rule(settings, "defensive")
        portfolio.update(
            {
                "final_portfolio_mode": "defensive",
                "portfolio_mode": "defensive",
                "max_total_position": rule["max_total_position"],
                "allow_new_position": True,
                "portfolio_mode_adjustment_reason": "结构性机会已连续确认，且不存在数据阻断；原始 cash 按防守仓位调整为 defensive。",
            }
        )
        adjustment_reason = portfolio["portfolio_mode_adjustment_reason"]
    elif structural_opportunity and raw_mode == "cash":
        adjustment_reason = "结构性机会已确认，但 market_regime=cash 或数据阻断存在，组合模式维持 cash。"
        portfolio["portfolio_mode_adjustment_reason"] = adjustment_reason
    market["portfolio_cash_reason"] = cash_reason
    market["portfolio_decision_chain"] = {
        "data_block": data_block,
        "market_regime": market_regime,
        "raw_portfolio_mode": raw_mode,
        "market_condition": market["market_condition"],
        "final_portfolio_mode": portfolio.get("portfolio_mode", raw_mode),
        "adjustment_reason": adjustment_reason,
    }
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
