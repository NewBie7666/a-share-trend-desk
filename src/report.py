from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data_quality import data_quality_message
from .position_rules import get_position_rule, operation_summary, trade_permission_text
from .utils import REPORT_DIR, pct, today_str


TEXT_REPLACEMENTS = {
    "fresh": "实时",
    "cache": "缓存",
    "failed": "失败",
    "not_reviewed": "未复核",
    "observation_only": "仅观察",
    "candidate_data_review": "候选数据复核",
    "candidate_risk_review": "候选风险复核",
    "data_not_ready": "数据未更新",
    "network_error": "网络错误",
    "data_fetch_aborted": "抓取中止",
    "insufficient_history": "历史数据不足",
    "cache_missing": "缓存缺失",
    "stale_cache": "缓存过期",
    "no_expected_trade_date_data": "无预期交易日数据",
}


def _cn_text(value):
    if not isinstance(value, str):
        return value
    out = value
    for old, new in TEXT_REPLACEMENTS.items():
        out = out.replace(old, new)
    return out


def _sanitize_rows(rows):
    if isinstance(rows, list):
        return [_sanitize_rows(item) for item in rows]
    if isinstance(rows, dict):
        return {key: _sanitize_rows(value) for key, value in rows.items()}
    return _cn_text(rows)


def _markdown_table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "无\n"
    rows = _sanitize_rows(rows)
    df = pd.DataFrame(rows)
    for col in columns:
        if col not in df.columns:
            df[col] = ""
    return df[columns].fillna("").to_markdown(index=False)


def _notes(notes: list[str]) -> str:
    return "\n".join(f"- {_cn_text(note)}" for note in notes) if notes else "- 无异常数据提示"


def _reason_list(title: str, rows: list[str]) -> str:
    body = "\n".join(f"- {_cn_text(item)}" for item in rows) if rows else "- 无"
    return f"### {title}\n\n{body}"


def _portfolio_summary(settings: dict, market: dict, portfolio: dict, portfolio_mode: str) -> str:
    summary = operation_summary(
        portfolio_mode,
        portfolio.get("strongest_styles", "无"),
        portfolio.get("potential_strong_styles", "无"),
    )
    multiplier = float(settings.get("confidence_position_multiplier", 1.0) or 1.0)
    if multiplier < 1.0:
        summary += f" 数据完整性降低，建议金额已按 {multiplier:.0%} 置信度乘数缩小。"
    return summary


def write_reports(
    market: dict,
    signals: dict,
    holding_actions: list[dict],
    used_cache: bool,
    cache_notes: list[str],
    settings: dict,
) -> tuple[Path, Path]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    date = settings.get("report_date") or today_str()
    md_path = REPORT_DIR / f"{date}_daily_signal.md"
    csv_path = REPORT_DIR / f"{date}_daily_signal.csv"

    candidates = signals["candidates"]
    watchlist = signals["watchlist"]
    forbidden = signals["forbidden"]
    data_issue_list = signals.get("data_issue_list") or settings.get("data_issue_list", [])
    portfolio = market.get("portfolio_mode", {})
    portfolio_mode = portfolio.get("portfolio_mode", "cash")
    max_position = get_position_rule(settings, portfolio_mode)["max_total_position"]
    data_quality_level = settings.get("data_quality_level", market.get("data_quality_level", "fresh"))
    integrity = settings.get("data_integrity_report", {}) or {}
    data_integrity_score = integrity.get("data_integrity_score", settings.get("data_integrity_score", 1.0))
    confidence_multiplier = integrity.get("confidence_position_multiplier", settings.get("confidence_position_multiplier", 1.0))
    confidence_reduced_mode = bool(float(confidence_multiplier or 1.0) < 1.0)
    summary = _portfolio_summary(settings, market, portfolio, portfolio_mode)
    potential = portfolio.get("potential_strong_styles", "无")
    potential_note = []
    if potential and potential != "无":
        potential_note.append(f"{potential} 状态为 strong，但样本数或置信度不足，暂不作为进攻组合主依据。")

    candidate_columns = [
        "candidate_rank", "symbol", "name", "best_style", "style_state", "final_score",
        "base_signal_score", "signal_score", "style_momentum_score", "breadth_score",
        "style_rotation_bonus", "style_retreat_penalty", "retreat_risk",
        "market_strength", "data_integrity_score", "confidence_position_multiplier",
        "candidate_data_source", "candidate_latest_date", "is_expected_trade_date",
        "data_source_name", "fallback_source_used", "source_attempts_summary",
        "close_price", "trigger_price", "suggested_lots", "suggested_shares", "estimated_amount",
        "account_risk_pct", "mode_account_risk_limit", "account_risk_pass",
        "half_stop_price", "final_exit_price", "take_profit_reduce_price", "take_profit_strong_price",
    ]
    condition_columns = [
        "symbol", "name", "best_style", "candidate_data_source", "candidate_latest_date",
        "is_expected_trade_date", "data_source_name", "fallback_source_used", "source_attempts_summary",
        "trigger_price", "suggested_lots", "suggested_shares", "estimated_amount",
        "half_stop_price", "half_stop_action", "final_exit_price", "final_exit_action",
        "trend_stop_price", "trend_exit_action", "account_risk_pct", "mode_account_risk_limit",
        "account_risk_pass", "account_risk_note", "take_profit_reduce_price",
        "take_profit_strong_price", "reason", "risk_note",
    ]

    md = [
        f"# A股主板多风格每日信号 - {date}",
        "",
        f"- 报告生成日期：{date}",
        f"- 行情交易日：{settings.get('market_data_date') or '无'}",
        f"- 预期交易日：{settings.get('expected_trade_date') or '无'}",
        f"- 数据更新时间：{settings.get('generated_at') or ''}",
        f"- 是否使用缓存数据：{'是' if used_cache else '否'}",
        f"- market_state：{market.get('market_state', 'unknown')}",
        f"- market_strength：{market.get('market_strength', 0)}",
        f"- portfolio_mode：{portfolio_mode}",
        f"- 交易权限：{trade_permission_text(portfolio_mode)}",
        f"- 今日总仓位建议：最高 {pct(max_position)}",
        f"- data_quality_level（兼容诊断项）：{data_quality_level}",
        f"- data_integrity_score：{data_integrity_score}",
        f"- source_health：{integrity.get('source_health', settings.get('source_health', 'unknown'))}",
        f"- confidence_level：{integrity.get('confidence_level', settings.get('confidence_level', 'unknown'))}",
        f"- confidence_position_multiplier：{confidence_multiplier}",
        f"- confidence_reduced_mode：{confidence_reduced_mode}",
        "- V2说明：数据完整性只降低置信度和建议金额，不直接决定交易开关；市场状态才是新开仓开关。",
        f"- market_data_ready_status：{settings.get('market_data_ready_status', 'unknown')}",
        f"- recommended_rerun_time：{settings.get('recommended_rerun_time', '')}",
        f"- ready_ratio：{settings.get('ready_ratio_text', '0/0 = 0.0%')}",
        f"- rerun_suggestion：{settings.get('rerun_suggestion', '')}",
        f"- 数据完整性说明：{data_quality_message(data_quality_level)}",
        f"- 最终数据完整性判定原因：{settings.get('data_quality_final_reason', '')}",
        f"- 本次扫描类型：{settings.get('scan_type', 'candidate_pool')}",
        f"- 本次扫描股票数量：{settings.get('scan_count', '见数据提示')}",
        f"- 今日操作摘要：{summary}",
        "",
        "## 数据完整性摘要",
        "",
        _markdown_table(settings.get("data_quality_summary", []), ["数据模块", "状态", "说明"]),
        "",
        "## 性能摘要",
        "",
        _markdown_table(
            [settings.get("performance_summary", {})],
            ["fetch_count", "indicator_compute_count", "index_indicator_compute_count", "snapshot_count", "valid_snapshot_count", "data_issue_count", "no_duplicate_io"],
        ),
        "",
        _reason_list("trading_critical_reasons", settings.get("trading_critical_reasons", [])),
        "",
        _reason_list("non_critical_warnings", settings.get("non_critical_warnings", [])),
        "",
        "## 市场状态证据链",
        "",
        "\n".join(f"- {_cn_text(reason)}" for reason in market.get("reasons", [])) or "- 无",
        "",
        "### market_index_table",
        "",
        _markdown_table(
            market.get("market_index_table", []),
            ["index_code", "index_name", "close", "MA20", "MA60", "close_above_MA20", "close_above_MA60", "state", "latest_date", "raw_latest_date", "effective_latest_date", "final_source", "data_source_name", "fallback_source_used", "source_attempts_summary", "error"],
        ),
        "",
        "### style_state_table",
        "",
        _markdown_table(
            market.get("style_state_table", []),
            ["style", "display_name", "state", "sample_size", "state_confidence", "cache_count", "cache_ratio", "failed_count", "failed_ratio", "sample_quality_eligible", "strongest_eligible", "above_ma20_ratio", "above_ma60_ratio", "rs_positive_ratio", "macd_bull_ratio", "breadth_score", "style_momentum_score", "style_rotation_bonus", "style_retreat_penalty", "retreat_risk", "retreat_ratio"],
        ),
        "",
        "### portfolio_mode",
        "",
        _markdown_table(
            [portfolio],
            ["market_risk_state", "market_state", "strongest_styles", "potential_strong_styles", "raw_portfolio_mode", "final_portfolio_mode", "portfolio_mode_adjustment_reason", "portfolio_mode", "max_total_position", "allow_new_position"],
        ),
        "",
        "### strongest_styles说明",
        "",
        "\n".join(f"- {_cn_text(item)}" for item in (signals.get("style_explanations", []) + potential_note)) or "- 无",
        "",
        "## 买入候选排名",
        "",
        _markdown_table(candidates, candidate_columns),
        "",
        "## 条件单参数",
        "",
        _markdown_table(candidates, condition_columns),
        "",
        "## 观察名单",
        "",
        _markdown_table(
            watchlist,
            ["symbol", "name", "best_style", "style_state", "final_score", "base_signal_score", "signal_score", "style_momentum_score", "breadth_score", "style_rotation_bonus", "style_retreat_penalty", "retreat_risk", "market_strength", "data_integrity_score", "close_price", "candidate_data_source", "candidate_latest_date", "is_expected_trade_date", "data_source_name", "fallback_source_used", "source_attempts_summary", "review_scope", "reason", "risk_note", "original_candidate_rank", "original_reason", "downgrade_reason", "original_account_risk_pct", "mode_account_risk_limit", "account_risk_pass", "original_estimated_amount"],
        ),
        "",
        "## 数据问题名单 data_issue_list",
        "",
        _markdown_table(data_issue_list, ["symbol", "name", "failed_reason", "error_category", "raw_latest_date", "effective_latest_date", "data_source_name", "fallback_source_used", "source_attempts_summary"]),
        "",
        "## 禁止买入名单及原因",
        "",
        "### 1. 过热禁止",
        "",
        _markdown_table(forbidden.get("overheated", {}).get("rows", []), ["symbol", "name", "primary_forbidden_category", "all_reasons"]),
        f"\n其余未展示数量：{forbidden.get('overheated', {}).get('extra_count', 0)}",
        "",
        "### 2. 趋势破坏",
        "",
        _markdown_table(forbidden.get("trend_broken", {}).get("rows", []), ["symbol", "name", "primary_forbidden_category", "all_reasons"]),
        f"\n其余未展示数量：{forbidden.get('trend_broken', {}).get('extra_count', 0)}",
        "",
        "### 3. 异常风险",
        "",
        _markdown_table(forbidden.get("abnormal_risk", {}).get("rows", []), ["symbol", "name", "primary_forbidden_category", "all_reasons"]),
        f"\n其余未展示数量：{forbidden.get('abnormal_risk', {}).get('extra_count', 0)}",
        "",
        "## 持仓处理建议",
        "",
        _markdown_table(holding_actions, ["symbol", "name", "action", "close_price", "cost_price", "pnl", "reason"]),
        "",
        "## 数据与缓存提示",
        "",
        _notes(cache_notes),
        "",
        "## 风险提示",
        "",
        "- 本报告仅为量化研究信号，不构成投资建议或收益承诺。",
        "- 系统不连接证券账户，不自动下单，最终交易由用户本人决定。",
        "- 数据完整性会影响置信度和建议金额，但不能消除市场风险。",
        "- 报告不代表股票必然上涨，极端行情下可能出现滑点、跳空或无法成交。",
        "",
    ]
    md_path.write_text("\n".join(md), encoding="utf-8")

    csv_rows = []
    for candidate in candidates:
        csv_rows.append(
            {
                "date": date,
                "action": "buy_candidate",
                "symbol": candidate["symbol"],
                "name": candidate["name"],
                "rank": candidate.get("rank", candidate.get("candidate_rank", "")),
                "score": candidate["final_score"],
                "signal_score": candidate.get("signal_score", ""),
                "market_strength": candidate.get("market_strength", ""),
                "data_integrity_score": candidate.get("data_integrity_score", ""),
                "confidence_position_multiplier": candidate.get("confidence_position_multiplier", ""),
                "best_style": candidate["best_style"],
                "style_state": candidate["style_state"],
                "candidate_data_source": candidate.get("candidate_data_source", ""),
                "candidate_latest_date": candidate.get("candidate_latest_date", ""),
                "is_expected_trade_date": candidate.get("is_expected_trade_date", ""),
                "data_source_name": candidate.get("data_source_name", ""),
                "fallback_source_used": candidate.get("fallback_source_used", ""),
                "source_attempts_summary": candidate.get("source_attempts_summary", ""),
                "close_price": candidate["close_price"],
                "trigger_price": candidate["trigger_price"],
                "suggested_lots": candidate["suggested_lots"],
                "suggested_shares": candidate["suggested_shares"],
                "estimated_amount": candidate["estimated_amount"],
                "half_stop_price": candidate["half_stop_price"],
                "full_stop_price": candidate.get("full_stop_price", ""),
                "trend_stop_price": candidate["trend_stop_price"],
                "final_exit_price": candidate["final_exit_price"],
                "half_stop_action": candidate["half_stop_action"],
                "final_exit_action": candidate["final_exit_action"],
                "trend_exit_action": candidate["trend_exit_action"],
                "account_risk_pct": candidate["account_risk_pct"],
                "mode_account_risk_limit": candidate.get("mode_account_risk_limit", ""),
                "account_risk_pass": candidate.get("account_risk_pass", ""),
                "take_profit_reduce_price": candidate["take_profit_reduce_price"],
                "take_profit_strong_price": candidate["take_profit_strong_price"],
                "reason": _cn_text(candidate["reason"]),
                "risk_note": _cn_text(candidate["risk_note"]),
            }
        )
    for action in holding_actions:
        csv_rows.append(
            {
                "date": date,
                "action": action["action"],
                "symbol": action["symbol"],
                "name": action["name"],
                "rank": "",
                "score": "",
                "signal_score": "",
                "market_strength": "",
                "data_integrity_score": "",
                "confidence_position_multiplier": "",
                "best_style": "",
                "style_state": "",
                "candidate_data_source": "",
                "candidate_latest_date": "",
                "is_expected_trade_date": "",
                "data_source_name": "",
                "fallback_source_used": "",
                "source_attempts_summary": "",
                "close_price": action.get("close_price", ""),
                "trigger_price": "",
                "suggested_lots": "",
                "suggested_shares": "",
                "estimated_amount": "",
                "half_stop_price": "",
                "full_stop_price": "",
                "trend_stop_price": "",
                "final_exit_price": "",
                "half_stop_action": "",
                "final_exit_action": "",
                "trend_exit_action": "",
                "account_risk_pct": "",
                "mode_account_risk_limit": "",
                "account_risk_pass": "",
                "take_profit_reduce_price": "",
                "take_profit_strong_price": "",
                "reason": _cn_text(action.get("reason", "")),
                "risk_note": "已有持仓处理建议，需用户本人确认",
            }
        )
    columns = [
        "date", "action", "symbol", "name", "rank", "score", "signal_score", "market_strength",
        "data_integrity_score", "confidence_position_multiplier", "best_style", "style_state",
        "candidate_data_source", "candidate_latest_date", "is_expected_trade_date", "data_source_name",
        "fallback_source_used", "source_attempts_summary", "close_price", "trigger_price",
        "suggested_lots", "suggested_shares", "estimated_amount", "half_stop_price", "full_stop_price",
        "trend_stop_price", "final_exit_price", "half_stop_action", "final_exit_action",
        "trend_exit_action", "account_risk_pct", "mode_account_risk_limit", "account_risk_pass",
        "take_profit_reduce_price", "take_profit_strong_price", "reason", "risk_note",
    ]
    pd.DataFrame(csv_rows, columns=columns).to_csv(csv_path, index=False, encoding="utf-8-sig")
    return md_path, csv_path
