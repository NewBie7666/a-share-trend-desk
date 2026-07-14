from __future__ import annotations

import json
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


def _serialize_gate_audit_rows(rows):
    """Convert gate audit containers into stable Markdown/CSV-safe values."""
    serialized = []
    for row in rows:
        item = row.copy()
        failed_reasons = item.get("gate_failed_reasons", [])
        if isinstance(failed_reasons, (list, tuple)):
            item["gate_failed_reasons"] = "；".join(str(reason) for reason in failed_reasons if reason)
        elif failed_reasons is None:
            item["gate_failed_reasons"] = ""
        checked_fields = item.get("gate_checked_fields", {})
        if isinstance(checked_fields, dict):
            item["gate_checked_fields_json"] = json.dumps(checked_fields, ensure_ascii=False, sort_keys=True)
        else:
            item["gate_checked_fields_json"] = item.get("gate_checked_fields_json", "")
        serialized.append(item)
    return serialized


def _markdown_table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "无\n"
    rows = _sanitize_rows(_serialize_gate_audit_rows(rows))
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

    candidates = _serialize_gate_audit_rows(signals["candidates"])
    watchlist = _serialize_gate_audit_rows(signals["watchlist"])
    forbidden = signals["forbidden"]
    timing_avoid_list = _serialize_gate_audit_rows(signals.get("timing_avoid_list", []))
    data_issue_list = signals.get("data_issue_list") or settings.get("data_issue_list", [])
    portfolio = market.get("portfolio_mode", {})
    portfolio_mode = portfolio.get("portfolio_mode", "cash")
    active_engine = str(settings.get("active_decision_engine", "v2"))
    display_portfolio_mode = signals.get("v3_position_mode", portfolio_mode) if active_engine == "v3" else portfolio_mode
    max_position = get_position_rule(settings, display_portfolio_mode)["max_total_position"]
    if active_engine == "v3":
        max_position *= float(signals.get("v3_position_multiplier", 0) or 0)
    data_quality_level = settings.get("data_quality_level", market.get("data_quality_level", "fresh"))
    integrity = settings.get("data_integrity_report", {}) or {}
    data_integrity_score = integrity.get("data_integrity_score", settings.get("data_integrity_score", 1.0))
    confidence_multiplier = integrity.get("confidence_position_multiplier", settings.get("confidence_position_multiplier", 1.0))
    confidence_reduced_mode = bool(float(confidence_multiplier or 1.0) < 1.0)
    summary = _portfolio_summary(settings, market, portfolio, portfolio_mode)
    if active_engine == "v3":
        summary = (
            f"V3市场权限为{signals.get('v3_market_permission', '')}，"
            f"按{display_portfolio_mode}基础规则并应用{signals.get('v3_position_multiplier', 0)}倍市场乘数。"
        )
    potential = portfolio.get("potential_strong_styles", "无")
    potential_note = []
    if potential and potential != "无":
        potential_note.append(f"{potential} 状态为 strong，但样本数或置信度不足，暂不作为进攻组合主依据。")
    v3_permission_lines = []
    if active_engine == "v3":
        v3_permission_lines = [
            f"- V2 portfolio_mode：{portfolio_mode}",
            f"- V2 trade_permission：{settings.get('v2_trade_permission', '')}",
            f"- V3 raw market_permission：{signals.get('raw_v3_market_permission', '')}",
            f"- V3 market_permission：{signals.get('v3_market_permission', '')}",
            f"- V3 permission_raw_target：{signals.get('permission_raw_target', '')}",
            f"- V3 permission_confirmation_status：{signals.get('permission_confirmation_status', '')}",
            f"- V3 permission_final：{signals.get('permission_final', '')}",
            f"- V3 permission_adjustment_reason：{signals.get('permission_adjustment_reason', '') or '无'}",
            f"- V3 position_mode：{signals.get('v3_position_mode', '')}",
            f"- V3 position_multiplier：{signals.get('v3_position_multiplier', '')}",
            f"- V3 max_candidates：{signals.get('v3_max_candidates', '')}",
            f"- V3 permission reason：{'；'.join(signals.get('v3_market_reason', []))}",
            "- V2与V3使用不同决策逻辑：V2 portfolio_mode仅代表V2复杂门控结果；V3新开仓由V3 market_permission独立决定。",
        ]

    score_diagnostics = signals.get("score_diagnostics", {}) or {}
    score_diagnostic_rows = [
        {"scope": scope, **values, "distribution_json": json.dumps(values.get("distribution", {}), ensure_ascii=False)}
        for scope, values in score_diagnostics.items()
        if isinstance(values, dict)
    ]
    provider_performance = settings.get("stock_fetch_stats", {}) or {}

    candidate_columns = [
        "candidate_rank", "symbol", "name", "best_style", "style_state", "final_score",
        "stock_factor_score", "minimum_stock_factor_for_buy", "structure_adjustment", "timing_score",
        "risk_adjustment", "risk_level", "risk_adjustment_source", "risk_components_json", "risk_adjustment_audit_json", "extreme_risk", "risk_flags", "risk_reason", "total_score", "v3_action",
        "ranking_action", "eligibility_status", "new_position_action", "signal_price", "decision_path_json", "fresh_validation_json", "provider_final_source", "style_is_advisory", "style_not_gate", "v3_style_role",
        "raw_v3_market_permission", "v3_market_permission", "v3_position_mode", "v3_position_multiplier", "v3_max_candidates",
        "base_position_amount", "market_multiplier", "risk_limited_amount", "final_order_amount",
        "base_signal_score", "signal_score", "style_momentum_score", "breadth_score",
        "style_rotation_bonus", "style_retreat_penalty", "retreat_risk",
        "structure_state", "entry_quality_score", "decision_confidence", "score_percentile", "timing_score_percentile", "total_score_percentile", "timing_reason_codes_json", "timing_reason_text",
        "timing_decision", "timing_reason", "trend_strong", "timing_risk_tag",
        "final_action", "decision_source", "gate_pass", "gate_failed_reasons", "gate_checked_fields_json", "position_multiplier", "final_position_multiplier", "final_reason", "market_regime", "market_score",
        "market_strength", "data_integrity_score", "confidence_position_multiplier",
        "candidate_data_source", "candidate_latest_date", "candidate_raw_latest_date", "is_expected_trade_date",
        "volume", "amount", "data_source_name", "fallback_source_used", "source_attempts_summary",
        "close_price", "trigger_price", "suggested_lots", "suggested_shares", "estimated_amount",
        "account_risk_pct", "mode_account_risk_limit", "account_risk_pass",
        "half_stop_price", "final_exit_price", "take_profit_reduce_price", "take_profit_strong_price",
    ]
    condition_columns = [
        "symbol", "name", "best_style", "candidate_data_source", "candidate_latest_date",
        "candidate_raw_latest_date", "is_expected_trade_date", "volume", "amount", "data_source_name", "fallback_source_used", "source_attempts_summary",
        "trigger_price", "suggested_lots", "suggested_shares", "estimated_amount",
        "half_stop_price", "half_stop_action", "final_exit_price", "final_exit_action",
        "trend_stop_price", "trend_exit_action", "account_risk_pct", "mode_account_risk_limit",
        "account_risk_pass", "structure_state", "entry_quality_score", "decision_confidence",
        "timing_decision", "timing_reason", "trend_strong", "timing_risk_tag",
        "final_action", "decision_source", "gate_pass", "gate_failed_reasons", "gate_checked_fields_json", "position_multiplier", "final_position_multiplier", "final_reason", "market_regime", "market_score",
        "account_risk_note", "take_profit_reduce_price",
        "take_profit_strong_price", "base_position_amount", "market_multiplier", "risk_limited_amount", "final_order_amount", "reason", "risk_note",
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
        f"- 基础市场状态：{market.get('base_market_state', market.get('market_state', 'unknown'))}",
        f"- 基础市场环境：{market.get('base_market_regime', market.get('market_regime', 'unknown'))}",
        f"- 结构性机会状态：{'是' if market.get('structural_market') else '否'}",
        f"- market_condition：{market.get('market_condition', 'normal')}",
        f"- market_opportunity_level：{market.get('market_opportunity_level', '')}",
        f"- portfolio_cash_reason：{market.get('portfolio_cash_reason', '')}",
        f"- 市场机会评分：{market.get('market_opportunity_score', '')}",
        f"- 市场风险评分：{(market.get('market_regime_metrics', {}) or {}).get('risk_score', '')}",
        f"- market_regime：{market.get('market_regime', 'unknown')}",
        f"- market_score：{market.get('market_score', '')}",
        (f"- v3_market_permission：{signals.get('v3_market_permission', '')}" if active_engine == "v3" else f"- v2_market_permission：{market.get('trade_permission', '')}"),
        f"- market_strength：{market.get('market_strength', 0)}",
        f"- 最终组合模式：{display_portfolio_mode}",
        f"- 交易权限：{('禁止正式新开仓' if signals.get('v3_market_permission') == 'BLOCKED' else '按V3市场权限控制新增仓位') if active_engine == 'v3' else trade_permission_text(portfolio_mode)}",
        f"- 今日总仓位建议：最高 {pct(max_position)}",
        f"- data_quality_level（兼容诊断项）：{data_quality_level}",
        f"- data_integrity_score：{data_integrity_score}",
        f"- source_health：{integrity.get('source_health', settings.get('source_health', 'unknown'))}",
        f"- confidence_level：{integrity.get('confidence_level', settings.get('confidence_level', 'unknown'))}",
        f"- confidence_position_multiplier：{confidence_multiplier}",
        f"- confidence_reduced_mode：{confidence_reduced_mode}",
        f"- system_health：{settings.get('system_health', '')}",
        f"- health_reasons：{'；'.join(settings.get('health_reasons', []))}",
        f"- final_position_multiplier：{settings.get('final_position_multiplier', 1.0)}",
        f"- final_trade_permission：{signals.get('final_trade_permission', '')}",
        f"- final_trade_permission_reason：{'；'.join(signals.get('final_trade_permission_reason', []))}",
        *v3_permission_lines,
        "- V2说明：数据完整性只降低置信度和建议金额；若触发核心指数、交易日历、基础风控或大面积行情异常等关键阻断项，仍禁止正式新开仓。结构性机会层只调整组合模式，不覆盖基础指数状态或最终市场环境门控。",
        f"- market_data_ready_status：{settings.get('market_data_ready_status', 'unknown')}",
        f"- recommended_rerun_time：{settings.get('recommended_rerun_time', '')}",
        f"- ready_ratio：{settings.get('ready_ratio_text', '0/0 = 0.0%')}",
        f"- rerun_suggestion：{settings.get('rerun_suggestion', '')}",
        f"- 数据完整性说明：{data_quality_message(data_quality_level)}",
        f"- 最终数据完整性判定原因：{settings.get('data_quality_final_reason', '')}",
        f"- 本次扫描类型：{settings.get('scan_type', 'candidate_pool')}",
        f"- 本次扫描股票数量：{settings.get('scan_count', '见数据提示')}",
        f"- 策略池定义：{(settings.get('pool_definition', {}) or {}).get('universe_type', '')}",
        f"- 理论扫描容量：{settings.get('pool_capacity', '')}",
        f"- 目标数量：{settings.get('pool_target', '')}",
        f"- 当前可用数量：{settings.get('pool_available', '')}",
        f"- 覆盖率：{settings.get('coverage_ratio', '')}",
        f"- 覆盖池质量评分：{settings.get('universe_quality_score', '')}",
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
            ["fetch_count", "indicator_compute_count", "timing_compute_count", "index_indicator_compute_count", "snapshot_count", "valid_snapshot_count", "data_issue_count", "cache_hit", "cache_miss", "duplicate_request_saved", "no_duplicate_io"],
        ),
        "",
        "## Decision Engine",
        "",
        _markdown_table([{"decision_engine": settings.get("active_decision_engine", "v2"), "daily_decision_log": settings.get("daily_decision_log_path", "")}], ["decision_engine", "daily_decision_log"]),
        "",
        "### V2 Decision",
        "",
        _markdown_table([{
            "v2_market_permission": settings.get("v2_trade_permission", ""),
            "v2_portfolio_mode": portfolio_mode,
            "v2_reason": "；".join(settings.get("v2_trade_permission_reason", [])) if isinstance(settings.get("v2_trade_permission_reason"), list) else settings.get("v2_trade_permission_reason", ""),
        }], ["v2_market_permission", "v2_portfolio_mode", "v2_reason"]),
        "",
        "### V3 Decision",
        "",
        _markdown_table([{
            "v3_market_permission": signals.get("v3_market_permission", ""),
            "v3_position_mode": signals.get("v3_position_mode", ""),
            "v3_reason": "；".join(signals.get("v3_market_reason", [])),
        }], ["v3_market_permission", "v3_position_mode", "v3_reason"]),
        "",
        "## Daily Signal Runtime",
        "",
        "## V3 Decision Funnel",
        "",
        _markdown_table(
            [dict(stage=stage, **values) for stage, values in (signals.get("signal_funnel", {}) or {}).items()],
            ["stage", "input_count", "output_count", "reject_count", "reject_reason"],
        ),
        "",
        "## Market Permission Chain",
        "",
        _markdown_table([{
            "market_score": market.get("market_score", ""),
            "permission_raw_target": signals.get("permission_raw_target", ""),
            "permission_confirmation_status": signals.get("permission_confirmation_status", ""),
            "permission_final": signals.get("permission_final", ""),
            "permission_adjustment_reason": signals.get("permission_adjustment_reason", "") or "无",
        }], ["market_score", "permission_raw_target", "permission_confirmation_status", "permission_final", "permission_adjustment_reason"]),
        "",
        "## V3 Stock Qualification Table",
        "",
        _markdown_table(
            signals.get("ranked", []),
            ["v3_rank", "symbol", "name", "stock_factor_score", "score_percentile", "timing_score", "timing_score_percentile", "market_score", "risk_adjustment", "risk_level", "risk_adjustment_source", "risk_components_json", "risk_adjustment_audit_json", "total_score", "total_score_percentile", "structure_state", "timing_reason_codes_json", "timing_reason_text", "ranking_action", "eligibility_status", "new_position_action", "candidate_data_source", "fresh_validation_json", "market_block_reason", "v3_reason"],
        ),
        "",
        "## Score Distribution",
        "",
        _markdown_table(score_diagnostic_rows, ["scope", "count", "min", "max", "mean", "median", "p10", "p50", "p90", "score_spread", "p90_minus_p10", "score_concentration_warning", "score_dispersion_warning", "distribution_json"]),
        "",
        "## Performance Optimization",
        "",
        _markdown_table([provider_performance], ["provider_parallel_enabled", "provider_parallel_initial_workers", "provider_parallel_peak_workers", "provider_parallel_final_workers", "latency_p50", "latency_p90", "latency_max", "fresh_rate", "serial_estimated_seconds", "actual_parallel_seconds", "provider_parallel_efficiency", "provider_seconds_saved"]),
        "",
        "### Provider Worker Adjustments",
        "",
        _markdown_table(provider_performance.get("provider_worker_history", []), ["workers_before", "workers_after", "request_count", "success_rate", "timeout_rate", "provider_failure_count", "reason"]),
        "",
        "### Provider Circuit Breaker",
        "",
        _markdown_table(provider_performance.get("provider_circuit_history", []), ["provider", "event", "failure_count", "cooldown_seconds"]),
        "",
        _markdown_table(
            (settings.get("runtime_report", {}) or {}).get("stages", []),
            ["stage_name", "start_time", "end_time", "elapsed_seconds"],
        ),
        "",
        _markdown_table(
            [{
                "total_seconds": (settings.get("runtime_report", {}) or {}).get("total_seconds", 0),
                "provider_bottleneck": (settings.get("runtime_report", {}) or {}).get("provider_bottleneck", False),
                "provider_fetch_ratio": (settings.get("runtime_report", {}) or {}).get("provider_fetch_ratio", 0),
                "performance_budget_status": (settings.get("runtime_report", {}) or {}).get("performance_budget_status", ""),
                "performance_bottleneck_reason": (settings.get("runtime_report", {}) or {}).get("performance_bottleneck_reason", ""),
                **(settings.get("runtime_cache_stats", {}) or {}),
            }],
            ["total_seconds", "provider_bottleneck", "provider_fetch_ratio", "performance_budget_status", "performance_bottleneck_reason", "cache_hit", "cache_miss", "duplicate_request_saved"],
        ),
        "",
        _markdown_table(settings.get("pool_providers", []), ["source", "status", "latency", "symbol_count", "latest_trade_date", "data_role", "error"]),
        "",
        _markdown_table([settings.get("pool_funnel", {})], ["raw_count", "main_board_count", "after_ST_filter", "after_liquidity_filter", "after_data_filter", "final_scan_count"]),
        "",
        "## 股票池摘要",
        "",
        _markdown_table(
            settings.get("pool_layers", []),
            ["pool", "size", "source"],
        ),
        "",
        _markdown_table(settings.get("analysis_pool_selection", []), ["symbol", "name", "selected_for_analysis", "analysis_pool_rank", "analysis_pool_score", "amount_score", "liquidity_score", "basic_trend_score", "industry_coverage_bonus", "analysis_pool_reason_codes", "analysis_pool_reason"]),
        "",
        _markdown_table(settings.get("timing_pool_selection", []), ["symbol", "name", "selected_for_timing", "timing_selection_rank", "timing_selection_score", "timing_selection_reason_codes", "timing_selection_reason"]),
        "",
        "### Universe Quality",
        "",
        _markdown_table([settings.get("universe_quality", {})], ["pool_size", "pool_target", "coverage_ratio", "pool_source", "industry_known_ratio", "industry_unknown_count", "industry_source", "industry_coverage_enabled", "industry_coverage_reason"]),
        "",
        _markdown_table(
            [{
                "universe_type": (settings.get("pool_definition", {}) or {}).get("universe_type", ""),
                "pool_capacity": settings.get("pool_capacity", ""),
                "pool_target": settings.get("pool_target", ""),
                "pool_available": settings.get("pool_available", ""),
                "coverage_ratio": settings.get("coverage_ratio", ""),
                "universe_quality_score": settings.get("universe_quality_score", ""),
                "pool_version": settings.get("pool_version", ""),
                "pool_size": settings.get("pool_size", settings.get("scan_count", "")),
                "pool_source": settings.get("pool_source", ""),
                "industry_distribution": "；".join(
                    f"{industry}:{count}" for industry, count in (settings.get("industry_distribution", {}) or {}).items()
                ),
            }],
            ["universe_type", "pool_capacity", "pool_target", "pool_available", "coverage_ratio", "universe_quality_score", "pool_version", "pool_size", "pool_source", "industry_distribution"],
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
        "### market_regime",
        "",
        _markdown_table(
            [{
                "market_score": market.get("market_score", ""),
                "market_regime": market.get("market_regime", ""),
                "trade_permission": market.get("trade_permission", ""),
                "market_reason": "；".join(market.get("market_reason", [])),
            }],
            ["market_score", "market_regime", "trade_permission", "market_reason"],
        ),
        "",
        "### market_regime_metrics",
        "",
        _markdown_table(
            [market.get("market_regime_metrics", {})],
            ["index_trend_score", "breadth_score", "activity_score", "risk_score", "above_ma20_ratio", "above_ma60_ratio", "rising_ratio", "amount_ratio", "limit_down_ratio", "abnormal_volatility_ratio", "shrinking_ratio"],
        ),
        "",
        "### market_opportunity",
        "",
        _markdown_table(
            [{
                "base_market_state": market.get("base_market_state", ""),
                "base_market_regime": market.get("base_market_regime", ""),
                "structural_market": market.get("structural_market", False),
                "market_opportunity_score": market.get("market_opportunity_score", ""),
                "strong_style_count": market.get("strong_style_count", ""),
                "breadth_persistence_days": market.get("breadth_persistence_days", ""),
                "risk_score": (market.get("market_regime_metrics", {}) or {}).get("risk_score", ""),
                "market_opportunity_reason": "；".join(market.get("market_opportunity_reason", [])),
            }],
            ["base_market_state", "base_market_regime", "structural_market", "market_opportunity_score", "strong_style_count", "breadth_persistence_days", "risk_score", "market_opportunity_reason"],
        ),
        "",
        _markdown_table([market.get("portfolio_decision_chain", {})], ["data_block", "market_regime", "raw_portfolio_mode", "market_condition", "final_portfolio_mode", "adjustment_reason"]),
        "",
        _markdown_table([market.get("market_condition_state", {})], ["confirm_days", "first_confirmed_date", "last_updated", "confirmed", "conditions"]),
        "",
        _markdown_table(
            market.get("breadth_persistence_table", []),
            ["date", "above_ma60_ratio", "rising_ratio", "persistent"],
        ),
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
        "## 交易时机说明",
        "",
        "- BUY：允许进入正式候选，并继续经过仓位、账户风险、同风格限制和数据源复核。",
        "- WAIT：股票可能仍有价值，但当前买点未满足正式买入条件，只进入观察名单。",
        "- AVOID：当前交易时机不适合开仓，不进入正式候选。",
        "",
        "## 买入候选排名",
        "",
        _markdown_table(candidates, candidate_columns),
        "",
        "## 候选淘汰链",
        "",
        _markdown_table(
            [dict(stage=name, **values) for name, values in signals.get("signal_funnel", {}).items()],
            ["stage", "input_count", "output_count", "reject_count", "reject_reason"],
        ),
        "",
        _markdown_table(signals.get("funnel_trace", []), ["symbol", "entered_stage", "exited_stage", "reject_code", "reject_reason", "stage_history"]),
        "",
        "## 为什么没有BUY",
        "",
        "\n".join(f"- {item}" for item in signals.get("why_no_buy", [])) or "- 存在正式 BUY 候选。",
        "",
        "## 条件单参数",
        "",
        _markdown_table(candidates, condition_columns),
        "",
        "## 观察名单",
        "",
        _markdown_table(
            watchlist,
            ["symbol", "name", "stock_factor_score", "timing_score", "market_score", "risk_adjustment", "extreme_risk", "risk_flags", "risk_reason", "total_score", "v3_action", "new_position_action", "signal_price", "decision_path_json", "raw_v3_market_permission", "v3_market_permission", "v3_position_mode", "v3_position_multiplier", "v3_max_candidates", "style_is_advisory", "style_not_gate", "v3_style_role", "best_style", "style_state", "final_score", "minimum_score", "base_signal_score", "signal_score", "style_momentum_score", "breadth_score", "style_rotation_bonus", "style_retreat_penalty", "retreat_risk", "structure_state", "entry_quality_score", "decision_confidence", "timing_decision", "final_action", "decision_source", "gate_pass", "gate_failed_reasons", "gate_checked_fields_json", "position_multiplier", "final_reason", "timing_reason", "trend_strong", "timing_risk_tag", "market_regime", "market_strength", "portfolio_mode", "final_portfolio_mode", "raw_portfolio_mode", "data_integrity_score", "close_price", "candidate_data_source", "candidate_latest_date", "candidate_raw_latest_date", "is_expected_trade_date", "volume", "amount", "data_source_name", "fallback_source_used", "source_attempts_summary", "review_scope", "reason", "risk_note", "original_candidate_rank", "original_reason", "downgrade_reason", "original_account_risk_pct", "mode_account_risk_limit", "account_risk_pass", "original_estimated_amount", "final_trade_permission", "final_trade_permission_reason"],
        ),
        "",
        "## 交易时机规避名单",
        "",
        _markdown_table(
            timing_avoid_list,
            ["symbol", "name", "best_style", "style_state", "final_score", "minimum_score", "structure_state", "entry_quality_score", "decision_confidence", "timing_decision", "final_action", "decision_source", "gate_pass", "gate_failed_reasons", "gate_checked_fields_json", "position_multiplier", "final_reason", "timing_reason", "trend_strong", "timing_risk_tag", "market_regime", "market_score", "portfolio_mode", "final_portfolio_mode", "raw_portfolio_mode", "candidate_data_source", "candidate_latest_date", "candidate_raw_latest_date", "is_expected_trade_date", "volume", "amount", "review_scope", "reason", "downgrade_reason", "final_trade_permission", "final_trade_permission_reason"],
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
                "stock_factor_score": candidate.get("stock_factor_score", ""),
                "minimum_stock_factor_for_buy": candidate.get("minimum_stock_factor_for_buy", (settings.get("v3", {}) or {}).get("minimum_stock_factor_for_buy", "")),
                "structure_adjustment": candidate.get("structure_adjustment", ""),
                "timing_score": candidate.get("timing_score", ""),
                "score_percentile": candidate.get("score_percentile", ""),
                "timing_score_percentile": candidate.get("timing_score_percentile", ""),
                "total_score_percentile": candidate.get("total_score_percentile", ""),
                "timing_reason_codes_json": candidate.get("timing_reason_codes_json", ""),
                "timing_reason_text": candidate.get("timing_reason_text", ""),
                "risk_adjustment": candidate.get("risk_adjustment", ""),
                "risk_level": candidate.get("risk_level", ""),
                "risk_adjustment_source": candidate.get("risk_adjustment_source", ""),
                "risk_components_json": candidate.get("risk_components_json", ""),
                "risk_adjustment_audit_json": candidate.get("risk_adjustment_audit_json", ""),
                "extreme_risk": candidate.get("extreme_risk", ""),
                "risk_flags": "；".join(candidate.get("risk_flags", [])) if isinstance(candidate.get("risk_flags"), list) else candidate.get("risk_flags", ""),
                "risk_reason": candidate.get("risk_reason", ""),
                "total_score": candidate.get("total_score", ""),
                "v3_action": candidate.get("v3_action", ""),
                "ranking_action": candidate.get("ranking_action", ""),
                "eligibility_status": candidate.get("eligibility_status", ""),
                "new_position_action": candidate.get("new_position_action", ""),
                "signal_price": candidate.get("signal_price", candidate.get("close_price", "")),
                "decision_path": candidate.get("decision_path_json", ""),
                "fresh_validation_json": candidate.get("fresh_validation_json", ""),
                "provider_final_source": candidate.get("provider_final_source", ""),
                "style_is_advisory": candidate.get("style_is_advisory", ""),
                "style_not_gate": candidate.get("style_not_gate", ""),
                "v3_style_role": candidate.get("v3_style_role", ""),
                "raw_v3_market_permission": candidate.get("raw_v3_market_permission", ""),
                "v3_market_permission": candidate.get("v3_market_permission", ""),
                "v3_position_mode": candidate.get("v3_position_mode", ""),
                "v3_position_multiplier": candidate.get("v3_position_multiplier", ""),
                "v3_max_candidates": candidate.get("v3_max_candidates", ""),
                "base_position_amount": candidate.get("base_position_amount", ""),
                "market_multiplier": candidate.get("market_multiplier", ""),
                "risk_limited_amount": candidate.get("risk_limited_amount", ""),
                "final_order_amount": candidate.get("final_order_amount", ""),
                "signal_score": candidate.get("signal_score", ""),
                "market_strength": candidate.get("market_strength", ""),
                "data_integrity_score": candidate.get("data_integrity_score", ""),
                "confidence_position_multiplier": candidate.get("confidence_position_multiplier", ""),
                "best_style": candidate["best_style"],
                "style_state": candidate["style_state"],
                "structure_state": candidate.get("structure_state", ""),
                "entry_quality_score": candidate.get("entry_quality_score", ""),
                "decision_confidence": candidate.get("decision_confidence", ""),
                "timing_decision": candidate.get("timing_decision", ""),
                "timing_reason": _cn_text(candidate.get("timing_reason", "")),
                "trend_strong": candidate.get("trend_strong", ""),
                "timing_risk_tag": candidate.get("timing_risk_tag", ""),
                "final_action": candidate.get("final_action", ""),
                "decision_source": candidate.get("decision_source", ""),
                "gate_pass": candidate.get("gate_pass", ""),
                "gate_failed_reasons": candidate.get("gate_failed_reasons", ""),
                "gate_checked_fields_json": candidate.get("gate_checked_fields_json", ""),
                "position_multiplier": candidate.get("position_multiplier", ""),
                "final_reason": _cn_text(candidate.get("final_reason", "")),
                "market_regime": candidate.get("market_regime", ""),
                "market_score": candidate.get("market_score", ""),
                "candidate_data_source": candidate.get("candidate_data_source", ""),
                "candidate_latest_date": candidate.get("candidate_latest_date", ""),
                "candidate_raw_latest_date": candidate.get("candidate_raw_latest_date", ""),
                "is_expected_trade_date": candidate.get("is_expected_trade_date", ""),
                "volume": candidate.get("volume", ""),
                "amount": candidate.get("amount", ""),
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
    candidate_symbols = {str(row.get("symbol", "")) for row in candidates}
    for row in signals.get("ranked", []):
        if str(row.get("symbol", "")) in candidate_symbols:
            continue
        csv_rows.append({
            "date": date,
            "action": "v3_ranking",
            "symbol": row.get("symbol", ""),
            "name": row.get("name", ""),
            "rank": row.get("v3_rank", ""),
            "stock_factor_score": row.get("stock_factor_score", ""),
            "timing_score": row.get("timing_score", ""),
            "score_percentile": row.get("score_percentile", ""),
            "timing_score_percentile": row.get("timing_score_percentile", ""),
            "total_score_percentile": row.get("total_score_percentile", ""),
            "timing_reason_codes_json": row.get("timing_reason_codes_json", ""),
            "timing_reason_text": row.get("timing_reason_text", ""),
            "market_score": row.get("market_score", ""),
            "risk_adjustment": row.get("risk_adjustment", ""),
            "risk_level": row.get("risk_level", ""),
            "risk_adjustment_source": row.get("risk_adjustment_source", ""),
            "risk_components_json": row.get("risk_components_json", ""),
            "risk_adjustment_audit_json": row.get("risk_adjustment_audit_json", ""),
            "total_score": row.get("total_score", ""),
            "v3_action": row.get("v3_action", ""),
            "ranking_action": row.get("ranking_action", ""),
            "eligibility_status": row.get("eligibility_status", ""),
            "new_position_action": row.get("new_position_action", ""),
            "signal_price": row.get("signal_price", ""),
            "decision_path": row.get("decision_path_json", ""),
            "fresh_validation_json": row.get("fresh_validation_json", ""),
            "provider_final_source": row.get("provider_final_source", ""),
            "structure_state": row.get("structure_state", ""),
            "candidate_data_source": row.get("candidate_data_source", ""),
            "candidate_latest_date": row.get("candidate_latest_date", ""),
            "volume": row.get("volume", ""),
            "amount": row.get("amount", ""),
            "reason": _cn_text(row.get("v3_reason", "")),
            "risk_note": _cn_text(row.get("risk_reason", "")),
        })
    for action in holding_actions:
        csv_rows.append(
            {
                "date": date,
                "action": action["action"],
                "symbol": action["symbol"],
                "name": action["name"],
                "rank": "",
                "score": "",
                "stock_factor_score": "",
                "minimum_stock_factor_for_buy": "",
                "structure_adjustment": "",
                "timing_score": "",
                "risk_adjustment": "",
                "extreme_risk": "",
                "risk_flags": "",
                "risk_reason": "",
                "total_score": "",
                "v3_action": "",
                "new_position_action": action.get("new_position_action", ""),
                "signal_price": action.get("close_price", ""),
                "decision_path": "",
                "signal_score": "",
                "market_strength": "",
                "data_integrity_score": "",
                "confidence_position_multiplier": "",
                "best_style": "",
                "style_state": "",
                "structure_state": "",
                "entry_quality_score": "",
                "decision_confidence": "",
                "timing_decision": "",
                "timing_reason": "",
                "trend_strong": "",
                "timing_risk_tag": "",
                "final_action": "",
                "decision_source": "",
                "gate_pass": "",
                "gate_failed_reasons": "",
                "gate_checked_fields_json": "",
                "position_multiplier": "",
                "final_reason": "",
                "market_regime": "",
                "market_score": "",
                "candidate_data_source": "",
                "candidate_latest_date": "",
                "candidate_raw_latest_date": "",
                "is_expected_trade_date": "",
                "volume": "",
                "amount": "",
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
        "stock_factor_score", "minimum_stock_factor_for_buy", "structure_adjustment", "timing_score",
        "score_percentile", "timing_score_percentile", "total_score_percentile", "timing_reason_codes_json", "timing_reason_text",
        "risk_adjustment", "risk_level", "risk_adjustment_source", "risk_components_json", "risk_adjustment_audit_json", "extreme_risk", "risk_flags", "risk_reason", "total_score", "v3_action",
        "ranking_action", "eligibility_status", "new_position_action", "signal_price", "decision_path", "fresh_validation_json", "provider_final_source", "style_is_advisory", "style_not_gate", "v3_style_role",
        "raw_v3_market_permission", "v3_market_permission", "v3_position_mode", "v3_position_multiplier", "v3_max_candidates",
        "base_position_amount", "market_multiplier", "risk_limited_amount", "final_order_amount",
        "data_integrity_score", "confidence_position_multiplier", "best_style", "style_state",
        "structure_state", "entry_quality_score", "decision_confidence", "timing_decision",
        "timing_reason", "trend_strong", "timing_risk_tag", "final_action", "decision_source", "gate_pass", "gate_failed_reasons", "gate_checked_fields_json", "position_multiplier",
        "final_reason", "market_regime", "market_score",
        "candidate_data_source", "candidate_latest_date", "candidate_raw_latest_date", "is_expected_trade_date", "volume", "amount", "data_source_name",
        "fallback_source_used", "source_attempts_summary", "close_price", "trigger_price",
        "suggested_lots", "suggested_shares", "estimated_amount", "half_stop_price", "full_stop_price",
        "trend_stop_price", "final_exit_price", "half_stop_action", "final_exit_action",
        "trend_exit_action", "account_risk_pct", "mode_account_risk_limit", "account_risk_pass",
        "take_profit_reduce_price", "take_profit_strong_price", "reason", "risk_note",
    ]
    pd.DataFrame(csv_rows, columns=columns).to_csv(csv_path, index=False, encoding="utf-8-sig")
    return md_path, csv_path
