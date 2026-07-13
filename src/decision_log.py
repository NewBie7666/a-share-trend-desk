"""Persist one auditable V2/V3 comparison record per trade date."""

from __future__ import annotations

import json
from pathlib import Path

from .utils import LOG_DIR


def _signal_rows(signals: dict) -> list[dict]:
    rows = list(signals.get("ranked") or [])
    if not rows:
        rows.extend(signals.get("candidates", []))
        rows.extend(signals.get("watchlist", []))
        rows.extend(signals.get("timing_avoid_list", []))
        rows.extend(signals.get("blocked_list", []))
        for category in (signals.get("forbidden", {}) or {}).values():
            rows.extend((category or {}).get("rows", []))
    output = []
    for row in rows:
        output.append({
            "symbol": row.get("symbol", ""), "name": row.get("name", ""),
            "rank": row.get("candidate_rank", row.get("v3_rank", "")),
            "action": row.get("new_position_action", row.get("final_action", row.get("v3_action", ""))),
            "reason": row.get("v3_reason", row.get("downgrade_reason", row.get("final_reason", ""))),
            "signal_price": row.get("signal_price", row.get("close_price", "")),
            "stock_factor_score": row.get("stock_factor_score", row.get("base_signal_score", row.get("score", ""))),
            "timing_score": row.get("timing_score", row.get("entry_quality_score", "")),
            "market_score": row.get("market_score", ""), "risk_adjustment": row.get("risk_adjustment", ""),
            "total_score": row.get("total_score", row.get("final_score", "")),
            "decision_path": row.get("decision_path", []), "future_check": {},
        })
    return output


def _differences(v2_rows: list[dict], v3_rows: list[dict]) -> list[dict]:
    v2_by_symbol = {row["symbol"]: row for row in v2_rows if row.get("symbol")}
    v3_by_symbol = {row["symbol"]: row for row in v3_rows if row.get("symbol")}
    output = []
    for symbol in sorted(set(v2_by_symbol) | set(v3_by_symbol)):
        v2 = v2_by_symbol.get(symbol, {})
        v3 = v3_by_symbol.get(symbol, {})
        v2_action = v2.get("action", "NOT_LISTED")
        v3_action = v3.get("action", "NOT_LISTED")
        if v2_action != v3_action:
            output.append({
                "symbol": symbol,
                "name": v3.get("name") or v2.get("name", ""),
                "v2_action": v2_action,
                "v3_action": v3_action,
                "v2_reason": v2.get("reason", ""),
                "v3_reason": v3.get("reason", ""),
            })
    return output


def write_daily_decision_log(settings: dict, v2_signals: dict, v3_signals: dict) -> Path:
    report_date = str(settings.get("report_date", ""))
    directory = LOG_DIR / "daily_decision"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{report_date}.json"
    temp = path.with_suffix(".json.tmp")
    v2_rows = _signal_rows(v2_signals)
    v3_rows = _signal_rows(v3_signals)
    payload = {
        "report_date": report_date,
        "market_data_date": settings.get("market_data_date", ""),
        "generated_at": settings.get("generated_at", ""),
        "pool_version": settings.get("pool_version", ""),
        "v2": {"candidates": v2_rows},
        "v3": {"rule_version": (settings.get("v3", {}) or {}).get("rule_version", "v3.0.0"), "candidates": v3_rows},
        "differences": _differences(v2_rows, v3_rows),
    }
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temp.replace(path)
    return path
