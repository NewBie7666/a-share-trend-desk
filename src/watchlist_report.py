"""Markdown output for the independent watchlist diagnosis command."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .utils import REPORT_DIR


def _table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "无"
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = ""
    return frame[columns].fillna("").to_markdown(index=False)


def write_watchlist_report(
    rows: list[dict],
    market_context: dict,
    report_date: str,
    expected_trade_date: str,
    *,
    path: Path | None = None,
) -> Path:
    output_path = path or (REPORT_DIR / "watchlist" / f"{report_date}_watchlist.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = []
    details = []
    for row in rows:
        history = row.get("history_change", {}) or {}
        holding = row.get("holding_analysis", {}) or {}
        summary.append({
            "symbol": row.get("symbol"),
            "name": row.get("name"),
            "position_source": row.get("position_source"),
            "holding_status": row.get("holding_status"),
            "diagnosis_score": row.get("final_diagnosis_score"),
            "trend": row.get("trend_state"),
            "position": row.get("position_state"),
            "structure": row.get("structure_state"),
            "risk": row.get("risk_level"),
            "diagnosis_action": row.get("diagnosis_action"),
            "data_status": row.get("diagnosis_data_status"),
            "complete_trade_date_close": row.get("current_price"),
            "price_trade_date": row.get("price_trade_date"),
            "score_change": history.get("diagnosis_score_change"),
        })
        details.append({
            "symbol": row.get("symbol"),
            "name": row.get("name"),
            "name_source": row.get("name_source"),
            "stock_factor": row.get("stock_factor_score"),
            "trend_score": row.get("trend_health_score"),
            "position_score": row.get("position_quality_score"),
            "raw_score": row.get("raw_score_before_risk"),
            "risk_penalty": row.get("risk_penalty"),
            "final_score": row.get("final_diagnosis_score"),
            "timing_source": row.get("timing_source"),
            "timing_decision": row.get("timing_decision"),
            "cost_price": holding.get("cost_price"),
            "complete_trade_date_close": row.get("current_price"),
            "price_trade_date": row.get("price_trade_date"),
            "price_source_type": row.get("price_source_type"),
            "price_provider": row.get("price_provider"),
            "price_validation_status": row.get("price_validation_status"),
            "last_available_price": row.get("last_available_price"),
            "last_available_trade_date": row.get("last_available_trade_date"),
            "profit_rate": holding.get("profit_rate"),
            "shares": holding.get("shares"),
            "reason": "；".join(row.get("reason_texts", [])),
            "change_reason": "；".join(history.get("change_reason_texts", [])),
            "risk_components_json": json.dumps(row.get("risk_components", []), ensure_ascii=False),
        })
    text = "\n".join([
        "# V3.1 自选股诊断报告",
        "",
        f"- 报告日期：{report_date}",
        f"- 预期交易日：{expected_trade_date}",
        "- 报告版本：v3.1.1",
        f"- 市场上下文来源：{market_context.get('market_context_source', 'unavailable')}",
        f"- 市场上下文兼容模式：{market_context.get('market_context_compatibility_mode', 'none')}",
        f"- 市场上下文交易日：{market_context.get('market_context_trade_date')}",
        f"- 决策引擎：{market_context.get('decision_engine', 'UNKNOWN')}",
        f"- 结构性机会条件：{market_context.get('market_condition', 'UNKNOWN')}",
        f"- 市场评分：{market_context.get('market_score')}",
        f"- 市场环境层：{market_context.get('market_regime', 'UNKNOWN')}",
        f"- 最终组合模式：{market_context.get('portfolio_mode')}",
        f"- V3新增开仓权限：{market_context.get('v3_market_permission', 'UNKNOWN')}",
        "- 价格口径：current_price列表示通过校验的完整交易日收盘价，不表示盘中实时价格；过期价格仅列为历史参考。",
        "- 说明：diagnosis_action仅供人工研究参考，不会修改holding_action、条件单或真实持仓。",
        "",
        "## 诊断总览",
        "",
        _table(summary, ["symbol", "name", "position_source", "holding_status", "diagnosis_score", "trend", "position", "structure", "risk", "diagnosis_action", "data_status", "complete_trade_date_close", "price_trade_date", "score_change"]),
        "",
        "## 评分、Timing、风险与持仓审计",
        "",
        _table(details, ["symbol", "name", "name_source", "stock_factor", "trend_score", "position_score", "raw_score", "risk_penalty", "final_score", "timing_source", "timing_decision", "cost_price", "complete_trade_date_close", "price_trade_date", "price_source_type", "price_provider", "price_validation_status", "last_available_price", "last_available_trade_date", "profit_rate", "shares", "reason", "change_reason", "risk_components_json"]),
        "",
        "本系统不会自动下单；诊断结果不构成投资建议或收益承诺。",
    ])
    temp = output_path.with_suffix(output_path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(output_path)
    return output_path
