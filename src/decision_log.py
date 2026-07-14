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
            "risk_level": row.get("risk_level", ""),
            "risk_adjustment_source": row.get("risk_adjustment_source", ""),
            "risk_components": row.get("risk_components", []),
            "risk_adjustment_audit": row.get("risk_adjustment_audit", {}),
            "total_score": row.get("total_score", row.get("final_score", "")),
            "score_percentile": row.get("score_percentile", ""),
            "timing_score_percentile": row.get("timing_score_percentile", ""),
            "total_score_percentile": row.get("total_score_percentile", ""),
            "timing_reason_codes": row.get("timing_reason_codes", []),
            "timing_reason_texts": row.get("timing_reason_texts", []),
            "ranking_action": row.get("ranking_action", ""),
            "eligibility_status": row.get("eligibility_status", ""),
            "new_position_action": row.get("new_position_action", ""),
            "fresh_validation": row.get("fresh_validation", {}),
            "decision_path": row.get("decision_path", []),
            "future_check": {
                "return_5d": None,
                "return_10d": None,
                "return_20d": None,
                "max_drawdown_20d": None,
            },
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


def _comparison_rows(v2_rows: list[dict], v3_rows: list[dict]) -> list[dict]:
    v2_by_symbol = {row["symbol"]: row for row in v2_rows if row.get("symbol")}
    output = []
    for v3 in v3_rows[:30]:
        v2 = v2_by_symbol.get(v3.get("symbol"), {})
        output.append({
            "symbol": v3.get("symbol", ""),
            "name": v3.get("name", ""),
            "signal_price": v3.get("signal_price", ""),
            "v2_action": v2.get("action", "NOT_LISTED"),
            "v2_reason": v2.get("reason", ""),
            "v3_action": v3.get("action", ""),
            "v3_reason": v3.get("reason", ""),
            "v3_rank": v3.get("rank", ""),
            "stock_factor_score": v3.get("stock_factor_score", ""),
            "timing_score": v3.get("timing_score", ""),
            "market_score": v3.get("market_score", ""),
            "risk_adjustment": v3.get("risk_adjustment", ""),
            "total_score": v3.get("total_score", ""),
            "decision_path": v3.get("decision_path", []),
            "future_check": v3.get("future_check", {}),
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
        "decision_version": settings.get("decision_version", "v3_pr2.1"),
        "timing_version": settings.get("timing_version", "timing_v1"),
        "indicator_version": settings.get("indicator_version", "indicator_v2"),
        "market_context": settings.get("market_context", {}),
        "v2": {"candidates": v2_rows},
        "v3": {
            "rule_version": (settings.get("v3", {}) or {}).get("rule_version", "v3.1.0"),
            "decision_version": settings.get("decision_version", "v3_pr2.1"),
            "timing_version": settings.get("timing_version", "timing_v1"),
            "indicator_version": settings.get("indicator_version", "indicator_v2"),
            "rule_freeze_status": (settings.get("v3", {}) or {}).get("rule_freeze_status", ""),
            "rule_freeze_trade_days": (settings.get("v3", {}) or {}).get("rule_freeze_trade_days", ""),
            "rule_freeze_start_date": (settings.get("v3", {}) or {}).get("rule_freeze_start_date", ""),
            "score_diagnostics": v3_signals.get("score_diagnostics", {}),
            "candidates": v3_rows,
        },
        "comparisons": _comparison_rows(v2_rows, v3_rows),
        "differences": _differences(v2_rows, v3_rows),
    }
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temp.replace(path)
    return path
