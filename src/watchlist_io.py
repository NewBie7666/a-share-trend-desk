"""Input, market-context and history persistence for watchlist diagnosis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from .utils import CACHE_DIR, DATA_DIR, LOG_DIR, load_yaml


WATCHLIST_PATH = DATA_DIR / "watchlist.yaml"
HISTORY_PATH = LOG_DIR / "watchlist_history.json"
STOCK_NAME_CACHE_PATH = CACHE_DIR / "main_board_pool_cache.json"
WATCHLIST_HISTORY_SCHEMA_VERSION = "v3.1.1"
WATCHLIST_REPORT_VERSION = "v3.1.1"

CHANGE_REASON_TEXTS = {
    "FIRST_DIAGNOSIS": "首次诊断，没有上一交易日记录",
    "SCORE_UP": "诊断分上升",
    "SCORE_DOWN": "诊断分下降",
    "SCORE_STABLE": "诊断分基本不变",
    "TREND_CHANGED": "趋势状态发生变化",
    "POSITION_CHANGED": "位置状态发生变化",
    "ACTION_CHANGED": "诊断建议发生变化",
    "CROSSED_ABOVE_MA20": "收盘价重新站上MA20",
    "CROSSED_BELOW_MA20": "收盘价跌破MA20",
    "CROSSED_ABOVE_MA60": "收盘价重新站上MA60",
    "CROSSED_BELOW_MA60": "收盘价跌破MA60",
    "MA20_SLOPE_POSITIVE": "MA20斜率转正",
    "MA20_SLOPE_NEGATIVE": "MA20斜率转负",
    "RELATIVE_STRENGTH_POSITIVE": "相对强弱转正",
    "RELATIVE_STRENGTH_NEGATIVE": "相对强弱转负",
    "VOLUME_EXPANDED": "相对20日均量明显放大",
    "VOLUME_CONTRACTED": "相对20日均量明显收缩",
}


def normalize_symbol(value) -> str:
    text = str(value or "").strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text.zfill(6) if text.isdigit() and len(text) <= 6 else ""


def is_valid_stock_name(value, symbol: str = "") -> bool:
    text = str(value or "").strip()
    normalized_symbol = normalize_symbol(symbol)
    if not text or text.lower() in {"none", "null", "nan", "unknown", "n/a"}:
        return False
    if text in {"-", "--", "未知", "未知股票", "未命名"}:
        return False
    return normalize_symbol(text) != normalized_symbol if normalized_symbol else True


def load_stock_name_cache(path: Path = STOCK_NAME_CACHE_PATH) -> dict[str, str]:
    """Load the local basic-name cache once for a watchlist run."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    records = payload.get("records", []) if isinstance(payload, dict) else []
    output: dict[str, str] = {}
    for row in records:
        if not isinstance(row, dict):
            continue
        symbol = normalize_symbol(row.get("symbol"))
        name = str(row.get("name") or "").strip()
        if symbol and is_valid_stock_name(name, symbol):
            output[symbol] = name
    return output


def load_watchlist(path: Path = WATCHLIST_PATH) -> list[dict]:
    if not path.exists():
        return []
    payload = load_yaml(path)
    rows = payload.get("watchlist", []) if isinstance(payload, dict) else []
    return [dict(item) for item in rows if isinstance(item, dict)]


def merge_watchlist_inputs(
    holdings: pd.DataFrame | None,
    watchlist_rows: Iterable[dict],
    manual_symbols: Iterable[str] = (),
) -> list[dict]:
    """Merge manual, watchlist and holdings inputs while preserving source audit."""
    merged: dict[str, dict] = {}
    source_flags: dict[str, set[str]] = {}

    for value in manual_symbols:
        symbol = normalize_symbol(value)
        if not symbol:
            continue
        merged[symbol] = {"symbol": symbol, "name": ""}
        source_flags.setdefault(symbol, set()).add("manual")

    for row in watchlist_rows:
        symbol = normalize_symbol(row.get("symbol"))
        if not symbol:
            continue
        current = merged.setdefault(symbol, {"symbol": symbol, "name": ""})
        watchlist_name = row.get("name")
        current.update({
            "symbol": symbol,
            "name": str(watchlist_name if is_valid_stock_name(watchlist_name, symbol) else current.get("name") or ""),
            "cost": row.get("cost"),
            "shares": row.get("shares"),
        })
        if is_valid_stock_name(watchlist_name, symbol):
            current["name_source"] = "watchlist"
        source_flags.setdefault(symbol, set()).add("watchlist")

    if holdings is not None and not holdings.empty:
        for _, row in holdings.iterrows():
            symbol = normalize_symbol(row.get("symbol"))
            if not symbol:
                continue
            current = merged.setdefault(symbol, {"symbol": symbol, "name": ""})
            holding_name = row.get("name")
            current.update({
                "symbol": symbol,
                "name": str(holding_name if is_valid_stock_name(holding_name, symbol) else current.get("name") or ""),
                "cost": row.get("cost_price"),
                "cost_price": row.get("cost_price"),
                "shares": row.get("quantity"),
                "quantity": row.get("quantity"),
            })
            if is_valid_stock_name(holding_name, symbol):
                current["name_source"] = "holding"
            source_flags.setdefault(symbol, set()).add("holding")

    output = []
    for symbol, item in merged.items():
        flags = source_flags.get(symbol, set())
        if "holding" in flags and "watchlist" in flags:
            source = "both"
        elif "holding" in flags:
            source = "holding"
        elif "watchlist" in flags:
            source = "watchlist"
        else:
            source = "manual"
        output.append({
            **item,
            "position_source": source,
            "holding_status": "holding" if "holding" in flags else "watching",
        })
    return output


def load_market_context(expected_trade_date: str, directory: Path | None = None) -> dict:
    root = directory or (LOG_DIR / "daily_decision")
    if not root.exists():
        return _unavailable_market_context()
    for path in sorted(root.glob("*.json"), reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        context = payload.get("market_context", {}) or {}
        context_trade_date = context.get("market_context_trade_date") or context.get("expected_trade_date")
        if str(context_trade_date or "") != str(expected_trade_date):
            continue
        decision_engine = str(context.get("decision_engine") or "")
        compatibility_mode = "current"
        if not context.get("market_context_version") or not decision_engine:
            compatibility_mode = "legacy"
        elif decision_engine.lower() != "v3":
            continue
        portfolio_mode = context.get("portfolio_mode")
        if isinstance(portfolio_mode, dict):
            portfolio_mode = portfolio_mode.get("final_portfolio_mode") or portfolio_mode.get("portfolio_mode")
        return {
            "market_context_source": "daily_signal",
            "market_context_version": context.get("market_context_version", "UNKNOWN"),
            "market_context_trade_date": context_trade_date,
            "decision_engine": decision_engine or "UNKNOWN",
            "market_context_compatibility_mode": compatibility_mode,
            "market_score": context.get("market_score"),
            "market_condition": context.get("market_condition", context.get("market_environment", "UNKNOWN")),
            "market_regime": context.get("market_regime", "UNKNOWN"),
            "portfolio_mode": portfolio_mode,
            "v3_market_permission": context.get("v3_market_permission", "UNKNOWN"),
            "market_context_generated_at": context.get("generated_at", payload.get("generated_at", "")),
        }
    return _unavailable_market_context()


def _unavailable_market_context() -> dict:
    return {
        "market_context_source": "unavailable",
        "market_context_version": "UNKNOWN",
        "market_context_trade_date": None,
        "decision_engine": "UNKNOWN",
        "market_context_compatibility_mode": "none",
        "market_score": None,
        "market_condition": "UNKNOWN",
        "market_regime": "UNKNOWN",
        "portfolio_mode": None,
        "v3_market_permission": "UNKNOWN",
        "market_context_generated_at": None,
    }


def _sign(value) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0
    return 1 if number > 0 else -1 if number < 0 else 0


def compare_history(previous: dict | None, current: dict) -> dict:
    if not previous:
        codes = ["FIRST_DIAGNOSIS"]
        return {
            "diagnosis_score_change": None,
            "trend_change": None,
            "position_change": None,
            "action_change": None,
            "change_reason_codes": codes,
            "change_reason_texts": [CHANGE_REASON_TEXTS[code] for code in codes],
        }
    previous_score = previous.get("final_diagnosis_score")
    current_score = current.get("final_diagnosis_score")
    delta = None if previous_score is None or current_score is None else round(float(current_score) - float(previous_score), 2)
    codes = []
    if delta is None or abs(delta) < 0.01:
        codes.append("SCORE_STABLE")
    else:
        codes.append("SCORE_UP" if delta > 0 else "SCORE_DOWN")

    trend_change = None
    if previous.get("trend_state") != current.get("trend_state"):
        trend_change = f"{previous.get('trend_state', '')}->{current.get('trend_state', '')}"
        codes.append("TREND_CHANGED")
    position_change = None
    if previous.get("position_state") != current.get("position_state"):
        position_change = f"{previous.get('position_state', '')}->{current.get('position_state', '')}"
        codes.append("POSITION_CHANGED")
    action_change = None
    if previous.get("diagnosis_action") != current.get("diagnosis_action"):
        action_change = f"{previous.get('diagnosis_action', '')}->{current.get('diagnosis_action', '')}"
        codes.append("ACTION_CHANGED")

    old = previous.get("diagnostic_facts", {}) or {}
    new = current.get("diagnostic_facts", {}) or {}
    for key, up_code, down_code in (
        ("above_ma20", "CROSSED_ABOVE_MA20", "CROSSED_BELOW_MA20"),
        ("above_ma60", "CROSSED_ABOVE_MA60", "CROSSED_BELOW_MA60"),
    ):
        if old.get(key) is not None and new.get(key) is not None and bool(old.get(key)) != bool(new.get(key)):
            codes.append(up_code if bool(new.get(key)) else down_code)
    for key, positive_code, negative_code in (
        ("ma20_slope", "MA20_SLOPE_POSITIVE", "MA20_SLOPE_NEGATIVE"),
        ("relative_strength_20d", "RELATIVE_STRENGTH_POSITIVE", "RELATIVE_STRENGTH_NEGATIVE"),
    ):
        if _sign(old.get(key)) != _sign(new.get(key)):
            codes.append(positive_code if _sign(new.get(key)) > 0 else negative_code)
    old_volume, new_volume = old.get("volume_ratio"), new.get("volume_ratio")
    if old_volume is not None and new_volume is not None:
        volume_delta = float(new_volume) - float(old_volume)
        if abs(volume_delta) >= 0.30:
            codes.append("VOLUME_EXPANDED" if volume_delta > 0 else "VOLUME_CONTRACTED")
    codes = list(dict.fromkeys(codes))
    return {
        "diagnosis_score_change": delta,
        "trend_change": trend_change,
        "position_change": position_change,
        "action_change": action_change,
        "change_reason_codes": codes,
        "change_reason_texts": [CHANGE_REASON_TEXTS.get(code, code) for code in codes],
    }


def update_watchlist_history(
    rows: list[dict],
    trade_date: str,
    *,
    path: Path = HISTORY_PATH,
    rule_version: str = "watchlist_v1",
    indicator_version: str = "indicator_v2",
    report_version: str = WATCHLIST_REPORT_VERSION,
) -> list[dict]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, ValueError, json.JSONDecodeError):
        payload = {}
    symbols = payload.get("symbols", {}) if isinstance(payload.get("symbols", {}), dict) else {}
    output = []
    for row in rows:
        symbol = normalize_symbol(row.get("symbol"))
        diagnosis_trade_date = str(row.get("diagnosis_trade_date") or "")
        if not symbol or not diagnosis_trade_date:
            output.append({**row, "history_change": compare_history(None, row)})
            continue
        records = list(symbols.get(symbol, []))
        previous = next((item for item in reversed(records) if str(item.get("diagnosis_trade_date") or item.get("trade_date")) != diagnosis_trade_date), None)
        enriched = {**row, "history_change": compare_history(previous, row)}
        history_row = {
            "diagnosis_trade_date": diagnosis_trade_date,
            "trade_date": diagnosis_trade_date,
            "name": row.get("name"),
            "name_source": row.get("name_source"),
            "current_price": row.get("current_price"),
            "price_source_type": row.get("price_source_type"),
            "price_provider": row.get("price_provider"),
            "price_validation_status": row.get("price_validation_status"),
            "diagnosis_data_status": row.get("diagnosis_data_status"),
            "stock_factor_score": row.get("stock_factor_score"),
            "trend_state": row.get("trend_state"),
            "trend_health_score": row.get("trend_health_score"),
            "position_state": row.get("position_state"),
            "position_quality_score": row.get("position_quality_score"),
            "structure_state": row.get("structure_state"),
            "risk_level": row.get("risk_level"),
            "risk_penalty": row.get("risk_penalty"),
            "final_diagnosis_score": row.get("final_diagnosis_score"),
            "diagnosis_action": row.get("diagnosis_action"),
            "market_context_source": row.get("market_context_source"),
            "diagnostic_facts": row.get("diagnostic_facts", {}),
            "diagnosis_rule_version": rule_version,
            "watchlist_report_version": report_version,
            "indicator_version": indicator_version,
        }
        records = [item for item in records if str(item.get("diagnosis_trade_date") or item.get("trade_date")) != diagnosis_trade_date]
        records.append(history_row)
        records.sort(key=lambda item: str(item.get("diagnosis_trade_date") or item.get("trade_date", "")))
        symbols[symbol] = records
        output.append(enriched)
    payload = {
        "watchlist_history_schema_version": WATCHLIST_HISTORY_SCHEMA_VERSION,
        "diagnosis_rule_version": rule_version,
        "watchlist_report_version": report_version,
        "symbols": symbols,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temp.replace(path)
    return output
