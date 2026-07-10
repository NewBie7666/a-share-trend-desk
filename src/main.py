from __future__ import annotations

import argparse
from datetime import datetime
import sys

try:
    from rich.console import Console
except ImportError:  # pragma: no cover
    class Console:
        def print(self, *args, **kwargs):
            print(*args)

from .data_fetcher import (
    expected_trade_date,
    fetch_market_indices,
    fetch_stocks,
    is_main_board_symbol,
    is_risk_stock_name,
    load_main_board_stock_pool,
    summarize_stock_fetches,
    write_fetch_failure_log,
)
from .data_quality import compute_data_quality_report
from .indicators import add_indicators
from .market_regime import evaluate_market_regime
from .market_state import apply_market_opportunity, evaluate_market_state_from_snapshots
from .pool_layers import pool_metadata, select_analysis_pool, select_candidate_pool, subset_snapshots
from .report import write_reports
from .signals import evaluate_holdings_from_snapshots, generate_buy_signals_from_snapshots
from .snapshots import build_stock_snapshots, snapshot_data_issue_list
from .utils import ensure_directories, load_config, load_holdings


console = Console()


def _distribution_text(distribution: dict[str, int]) -> str:
    if not distribution:
        return "no valid effective trade date"
    return "; ".join(f"{date}: {count}" for date, count in sorted(distribution.items(), reverse=True))


def _attempt_distribution_text(distribution: dict[str, int]) -> str:
    if not distribution:
        return "no attempt distribution"
    return "; ".join(f"{key}={value}" for key, value in distribution.items())


def _source_distribution_text(distribution: dict[str, int]) -> str:
    if not distribution:
        return "无备用源成功分布"
    return "; ".join(f"{key}={value}" for key, value in distribution.items())


def _source_attempts_summary(attempts: list[dict] | None) -> str:
    if not attempts:
        return "无来源尝试记录"
    parts = []
    for item in attempts:
        name = item.get("source_name", "")
        status = "成功" if item.get("status") == "success" else "失败"
        category = item.get("error_category") or "无错误"
        effective = item.get("effective_latest_date") or "-"
        parts.append(f"{name}:{status},有效日={effective},类别={category}")
    return "；".join(parts)


def _is_after_close_window(generated_at: datetime, expected_date: str) -> bool:
    return generated_at.strftime("%Y-%m-%d") == expected_date and generated_at.strftime("%H:%M") >= "15:10"


def _market_data_ready_report(index_results: list[dict], stock_stats: dict, expected_date: str, generated_at: datetime, settings: dict) -> dict:
    core_results = [item["result"] for item in index_results if item.get("core")]
    core_ready = bool(core_results) and all(
        result.effective_latest_date == expected_date and result.final_source == "fresh" and not result.data.empty
        for result in core_results
    )
    core_lagging = any(result.effective_latest_date and result.effective_latest_date < expected_date for result in core_results)
    ready_ratio = float(stock_stats.get("ready_ratio", 0) or 0)
    abort_reason = stock_stats.get("data_fetch_abort_reason", "")
    network_count = int(stock_stats.get("network_error_count", 0) or 0)
    data_not_ready_count = int(stock_stats.get("data_not_ready_count", 0) or 0)
    status = "unknown"
    if core_ready and ready_ratio >= 0.80 and not abort_reason:
        status = "ready"
    elif abort_reason or (ready_ratio == 0 and network_count > 0):
        status = "source_unavailable"
    elif _is_after_close_window(generated_at, expected_date) and (core_lagging or data_not_ready_count > 0):
        status = "not_ready_after_close"
    elif data_not_ready_count > 0 or ready_ratio < 0.80:
        status = "data_not_ready"
    earliest = ((settings.get("daily_signal", {}) or {}).get("earliest_reliable_run_time") or "16:30")
    if status == "not_ready_after_close":
        suggestion = f"\u6570\u636e\u6e90\u5c1a\u672a\u66f4\u65b0\u5230\u9884\u671f\u4ea4\u6613\u65e5\uff0c\u5efa\u8bae {earliest} \u540e\u91cd\u8dd1\u3002"
        recommended = f"\u5efa\u8bae {earliest} \u540e\u91cd\u8dd1"
    elif abort_reason:
        suggestion = "\u672c\u6b21\u6293\u53d6\u56e0\u8fde\u7eed\u7f51\u7edc\u5931\u8d25\u4e2d\u6b62\uff0c\u5efa\u8bae\u7a0d\u540e\u91cd\u8dd1\uff1b\u82e5\u591a\u6b21\u51fa\u73b0\uff0c\u8bf7\u5207\u6362\u5907\u7528\u6570\u636e\u6e90\u3002"
        recommended = "\u5efa\u8bae\u7a0d\u540e\u91cd\u8dd1"
    elif stock_stats.get("cache_fallback_ratio", 0) > 0.20:
        suggestion = "\u7f13\u5b58\u56de\u9000\u6bd4\u4f8b\u8fc7\u9ad8\uff0c\u5efa\u8bae\u7a0d\u540e\u91cd\u8dd1\u3002"
        recommended = "\u5efa\u8bae\u7a0d\u540e\u91cd\u8dd1"
    else:
        suggestion = "\u65e0\u9700\u56e0\u6570\u636e\u83b7\u53d6\u91cd\u8dd1\u3002"
        recommended = "\u65e0\u9700\u56e0\u6570\u636e\u83b7\u53d6\u91cd\u8dd1"
    return {
        "market_data_ready_status": status,
        "core_indices_ready": core_ready,
        "ready_ratio": ready_ratio,
        "ready_ratio_text": stock_stats.get("ready_ratio_text", "0/0 = 0.0%"),
        "recommended_rerun_time": recommended,
        "rerun_suggestion": suggestion,
    }


def _append_fetch_summary_rows(summary: list[dict], stock_stats: dict) -> list[dict]:
    summary.extend(
        [
            {"\u6570\u636e\u6a21\u5757": "\u6570\u636e\u5c31\u7eea\u72b6\u6001", "\u72b6\u6001": stock_stats.get("market_data_ready_status", "unknown"), "\u8bf4\u660e": f"\u6570\u636e\u5c31\u7eea\u6bd4\u4f8b\uff1a{stock_stats.get('ready_ratio_text', '0/0 = 0.0%')}"},
            {"\u6570\u636e\u6a21\u5757": "\u6570\u636e\u6e90\u6210\u529f\u5206\u5e03", "\u72b6\u6001": "distribution", "\u8bf4\u660e": _source_distribution_text(stock_stats.get("source_success_distribution", {}))},
            {"\u6570\u636e\u6a21\u5757": "\u6570\u636e\u672a\u66f4\u65b0\u6570\u91cf", "\u72b6\u6001": str(stock_stats.get("data_not_ready_count", 0)), "\u8bf4\u660e": "\u63a5\u53e3\u53ef\u8fde\u63a5\u4f46 effective_latest_date \u65e9\u4e8e expected_trade_date"},
            {"\u6570\u636e\u6a21\u5757": "\u5386\u53f2\u4e0d\u8db3\u6570\u91cf", "\u72b6\u6001": str(stock_stats.get("insufficient_history_count", 0)), "\u8bf4\u660e": "\u6807\u51c6\u5316\u5e76\u622a\u65ad\u540e\u5c11\u4e8e min_history_days\uff0c\u5df2\u7981\u6b62\u53c2\u4e0e\u4fe1\u53f7"},
            {"数据模块": "抓取开始时间", "状态": stock_stats.get("fetch_started_at", ""), "说明": "个股日线本轮抓取开始时间"},
            {"数据模块": "抓取结束时间", "状态": stock_stats.get("fetch_finished_at", ""), "说明": "个股日线本轮抓取结束时间"},
            {"数据模块": "抓取耗时秒数", "状态": str(stock_stats.get("fetch_duration_seconds", 0)), "说明": "fetch_finished_at - fetch_started_at"},
            {"数据模块": "重试成功分布", "状态": "distribution", "说明": _attempt_distribution_text(stock_stats.get("attempt_success_distribution", {}))},
            {
                "数据模块": "网络错误比例",
                "状态": f"{stock_stats.get('network_error_ratio', 0):.2%}",
                "说明": f"network_error_count={stock_stats.get('network_error_count', 0)}",
            },
            {"数据模块": "抓取中止原因", "状态": stock_stats.get("data_fetch_abort_reason", "") or "无", "说明": "触发连续网络失败硬熔断时显示"},
            {"数据模块": "重跑建议", "状态": "suggestion", "说明": stock_stats.get("rerun_suggestion", "无需因数据获取重跑。")},
        ]
    )
    return summary


def _annotate_cache_fallback_hits(
    stock_stats: dict,
    *,
    holding_symbols: set[str] | None = None,
    style_symbols: set[str] | None = None,
    temp_candidate_symbols: set[str] | None = None,
) -> None:
    holding_symbols = holding_symbols or set()
    style_symbols = style_symbols or set()
    temp_candidate_symbols = temp_candidate_symbols or set()
    for item in stock_stats.get("cache_fallback_symbols", []):
        symbol = str(item.get("symbol", "")).zfill(6)
        item["hit_current_holding"] = symbol in holding_symbols
        item["hit_style_sample"] = symbol in style_symbols
        item["hit_temp_candidate"] = symbol in temp_candidate_symbols


def _build_data_quality_summary(
    *,
    core_index_used_cache: bool,
    optional_index_failed: bool,
    pool_used_cache: bool,
    stock_stats: dict,
    expected_date: str,
    market_data_date: str,
    trade_calendar_status: str,
) -> list[dict]:
    total = stock_stats.get("total_symbols", 0)
    cache_ratio = stock_stats.get("cache_fallback_count", 0) / total if total else 0.0
    trimmed_symbols = stock_stats.get("future_rows_trimmed_symbols_count", 0)
    trimmed_rows = stock_stats.get("future_rows_trimmed_total_count", 0)
    return [
        {"数据模块": "核心指数", "状态": "critical" if core_index_used_cache else "fresh", "说明": "000300/000001/399001 " + ("存在缓存、失败或非预期交易日" if core_index_used_cache else "全部为预期交易日实时获取")},
        {"数据模块": "可选指数", "状态": "failed" if optional_index_failed else "fresh", "说明": "000510 " + ("获取失败、缓存或非预期交易日" if optional_index_failed else "预期交易日实时获取")},
        {"数据模块": "股票池配置", "状态": "local_config", "说明": "config/stock_pool.yaml 正常加载；本地配置不视为缓存或 backup"},
        {"数据模块": "股票基础信息", "状态": "cache" if pool_used_cache else "fresh", "说明": "名称/ST/停牌/预筛基础信息 " + ("使用缓存或本地备用池" if pool_used_cache else "来自实时基础行情")},
        {
            "数据模块": "个股日线",
            "状态": "critical_cache" if cache_ratio > 0.20 else "partial_cache" if stock_stats.get("cache_fallback_count", 0) else "fresh",
            "说明": (
                f"实时获取成功：{stock_stats.get('fresh_success_count', 0)}/{total}；"
                f"缓存回退：{stock_stats.get('cache_fallback_count', 0)}/{total}；"
                f"请求失败：{stock_stats.get('request_failed_count', 0)}/{total}；"
                f"无预期交易日数据：{stock_stats.get('no_expected_trade_date_data_count', 0)}/{total}；"
                f"不可用于信号：{stock_stats.get('unusable_for_signal_count', 0)}/{total}"
            ),
        },
        {
            "数据模块": "盘中未完成日线截断",
            "状态": str(trimmed_symbols),
            "说明": f"盘中未完成日线截断：{trimmed_symbols}只股票，共{trimmed_rows}行；已使用 {expected_date} 完整交易日数据。该截断是 daily-signal 完整交易日模式的正常行为，不影响 data_quality_level。",
        },
        {"数据模块": "交易日历状态", "状态": trade_calendar_status, "说明": "fresh 为交易日历接口可用；fallback_weekday 表示使用工作日规则推断"},
        {"数据模块": "预期交易日", "状态": expected_date, "说明": "daily-signal 收盘后模式应使用的完整交易日"},
        {"数据模块": "实际行情交易日", "状态": market_data_date or "无", "说明": "按个股 effective_latest_date 分布中数量最多的日期确定；不得使用 raw_latest_date"},
        {"数据模块": "行情日期分布", "状态": "distribution", "说明": _distribution_text(stock_stats.get("latest_date_distribution", {}))},
        {"数据模块": "非预期交易日数据数量", "状态": str(stock_stats.get("unexpected_trade_date_count", 0)), "说明": "effective_latest_date 不等于 expected_trade_date 的有效日线数量"},
        {"数据模块": "过期缓存数量", "状态": str(stock_stats.get("stale_cache_count", 0)), "说明": "缓存交易日差超过 cache.max_cache_age_trade_days 的数量"},
    ]


def _stock_basic_has_local_fallback(stock_pool: list[dict[str, str]]) -> bool:
    if not stock_pool:
        return False
    for item in stock_pool:
        symbol = str(item.get("symbol", "")).zfill(6)
        name = str(item.get("name", ""))
        if not symbol or not name or not is_main_board_symbol(symbol) or is_risk_stock_name(name):
            return False
    return True


def _fallback_basis_rows(stock_basic_status: str, stock_basic_is_trading_critical: bool) -> list[dict]:
    if stock_basic_status != "cache" or stock_basic_is_trading_critical:
        return []
    return [
        {"依据项": "ST/退市过滤依据", "说明": "symbol/name 存在，name 不含 ST、*ST、退；可按本地名称字段执行风险过滤"},
        {"依据项": "主板范围依据", "说明": "候选池为主板白名单或主板预筛结果，symbol 前缀符合沪深主板范围"},
        {"依据项": "风格识别依据", "说明": "使用本地 symbol/name 风格映射作为 style fallback"},
        {"依据项": "停牌风险兜底依据", "说明": "正式候选必须 effective_latest_date = expected_trade_date，且 volume > 0、amount > 0"},
    ]


def _data_issue_list(stock_results) -> list[dict]:
    rows = []
    for result in stock_results:
        if result.final_source == "failed" or result.data.empty:
            rows.append(
                {
                    "symbol": result.symbol,
                    "name": result.name,
                    "failed_reason": result.error or result.data_staleness_reason or "行情数据不可用",
                    "error_category": result.error_category or "unknown",
                    "raw_latest_date": result.raw_latest_date or "",
                    "effective_latest_date": result.effective_latest_date or result.latest_date or "",
                    "data_source_name": result.data_source_name or "",
                    "fallback_source_used": result.fallback_source_used or "",
                    "source_attempts_summary": _source_attempts_summary(result.source_attempts),
                }
            )
    return rows


def _cache_notes(stock_stats: dict, fetch_log_path, data_quality_level: str, trade_calendar_status: str) -> list[str]:
    notes: list[str] = []
    if data_quality_level == "critical_cache":
        notes.append("本次数据完整性明显降级；V2 仅降低信号置信度和建议金额，不再单独冻结全部新开仓。")
    if data_quality_level == "partial_cache":
        notes.append("数据存在少量缓存或失败诊断；V2 仅降低信号置信度和建议金额。")
    if data_quality_level == "stale":
        notes.append("\u6570\u636e\u9648\u65e7\uff0c\u4e0d\u9002\u5408\u4ea4\u6613\u51b3\u7b56\u3002")
    if trade_calendar_status == "fallback_weekday":
        notes.append("\u4ea4\u6613\u65e5\u5386\u4e0d\u53ef\u7528\uff0cexpected_trade_date \u4f7f\u7528\u5de5\u4f5c\u65e5\u89c4\u5219\u63a8\u65ad\uff0c\u6570\u636e\u8d28\u91cf\u7f6e\u4fe1\u5ea6\u4e0b\u964d\u3002")
    trimmed_symbols = stock_stats.get("future_rows_trimmed_symbols_count", 0)
    if trimmed_symbols:
        notes.append(
            f"\u76d8\u4e2d\u672a\u5b8c\u6210\u65e5\u7ebf\u622a\u65ad\uff1a{trimmed_symbols}\u53ea\u80a1\u7968\uff0c\u5171{stock_stats.get('future_rows_trimmed_total_count', 0)}\u884c\uff1b"
            "\u8be5\u622a\u65ad\u662f daily-signal \u5b8c\u6574\u4ea4\u6613\u65e5\u6a21\u5f0f\u7684\u6b63\u5e38\u884c\u4e3a\uff0c\u4e0d\u5f71\u54cd data_quality_level\u3002"
        )
    cache_symbols = stock_stats.get("cache_fallback_symbols", [])
    if cache_symbols:
        notes.append(f"\u7f13\u5b58\u56de\u9000\u80a1\u7968\u6570\u91cf\uff1a{len(cache_symbols)}")
        notes.append("\u5c55\u793a\u524d20\u53ea\uff1a")
        for item in cache_symbols[:20]:
            hit_flags = []
            if item.get("hit_temp_candidate"):
                hit_flags.append("\u547d\u4e2d\u4e34\u65f6\u5019\u9009")
            if item.get("hit_style_sample"):
                hit_flags.append("\u547d\u4e2d\u98ce\u683c\u6837\u672c")
            if item.get("hit_current_holding"):
                hit_flags.append("\u547d\u4e2d\u5f53\u524d\u6301\u4ed3")
            flag_text = f"\uff1b{','.join(hit_flags)}" if hit_flags else ""
            fallback_reason = item.get("fallback_reason") or item.get("error", "")
            notes.append(f"{item.get('symbol', '')} {item.get('name', '')}\uff1a{fallback_reason}{flag_text}")
        if len(cache_symbols) > 20:
            notes.append(f"\u5176\u4f59{len(cache_symbols) - 20}\u53ea\u7565\u3002")
    failed_symbols = stock_stats.get("failed_symbols", [])
    if failed_symbols:
        notes.append(f"\u4e0d\u53ef\u7528\u4e8e\u4fe1\u53f7\u80a1\u7968\u6570\u91cf\uff1a{len(failed_symbols)}")
    if fetch_log_path:
        notes.append(f"\u5b8c\u6574\u6293\u53d6\u5931\u8d25\u660e\u7ec6\uff1a{fetch_log_path}")
    if stock_stats.get("rerun_suggestion"):
        notes.append(f"\u91cd\u8dd1\u5efa\u8bae\uff1a{stock_stats.get('rerun_suggestion')}")
    return notes

def run_daily_signal() -> int:
    ensure_directories()
    generated_at = datetime.now()
    report_date = generated_at.strftime("%Y-%m-%d")
    expected_date, trade_calendar_status, trade_dates = expected_trade_date(generated_at)

    config = load_config()
    settings = config["settings"]
    risk_rules = config["risk_rules"]
    settings["generated_at"] = generated_at.strftime("%Y-%m-%d %H:%M:%S")
    settings["report_date"] = report_date
    settings["expected_trade_date"] = expected_date
    settings["trade_calendar_status"] = trade_calendar_status

    console.print("[bold]Step 1/5: loading stock pool...[/bold]")
    stock_pool, pool_used_cache, pool_error, scan_pool_meta = load_main_board_stock_pool(settings, config["stock_pool"])
    holdings = load_holdings()
    merged = {item["symbol"]: item for item in stock_pool}
    for item in config["stock_pool"]:
        merged.setdefault(item["symbol"], item)
    stock_pool = list(merged.values())[: int(settings.get("max_scan_symbols", 500))]
    scan_pool_meta = pool_metadata(
        stock_pool,
        version=str(settings.get("pool_version", "v2.1")),
        source=scan_pool_meta.get("pool_source", ""),
    )
    settings["scan_count"] = len(stock_pool)
    settings.update(scan_pool_meta)
    settings["pool_layers"] = [{"pool": "scan_pool", "size": len(stock_pool), "source": scan_pool_meta.get("pool_source", "")}]
    names = {item["symbol"]: item["name"] for item in stock_pool}

    console.print(f"[bold]Step 2/5: fetching indices and daily bars, scan_count={len(stock_pool)}...[/bold]")
    index_results = fetch_market_indices(expected_date, settings=settings)
    hs300_result = next(item["result"] for item in index_results if item["index_code"] == "000300")
    stock_results = fetch_stocks(stock_pool, progress=True, settings=settings, expected_trade_date_value=expected_date, trade_dates=trade_dates)
    stock_results_by_symbol = {result.symbol: result for result in stock_results}
    stock_stats = summarize_stock_fetches(stock_results)
    holding_symbols = set(holdings["symbol"].astype(str).str.zfill(6)) if not holdings.empty else set()
    confirmed_style_symbols = {item["symbol"] for item in config["stock_pool"]}
    _annotate_cache_fallback_hits(stock_stats, holding_symbols=holding_symbols, style_symbols=confirmed_style_symbols)
    market_data_date = stock_stats.get("market_data_date", "")
    ready_report = _market_data_ready_report(index_results, stock_stats, expected_date, generated_at, settings)
    stock_stats["rerun_suggestion"] = ready_report["rerun_suggestion"]
    stock_stats["market_data_ready_status"] = ready_report["market_data_ready_status"]
    settings["market_data_date"] = market_data_date
    settings["market_data_ready_status"] = ready_report["market_data_ready_status"]
    settings["recommended_rerun_time"] = ready_report["recommended_rerun_time"]
    settings["ready_ratio"] = ready_report["ready_ratio"]
    settings["ready_ratio_text"] = ready_report["ready_ratio_text"]
    settings["rerun_suggestion"] = ready_report["rerun_suggestion"]
    settings["stock_fetch_stats"] = stock_stats
    fetch_log_path = write_fetch_failure_log(stock_results, report_date)

    core_index_used_cache = any(item["core"] and (item["result"].final_source != "fresh" or not item["result"].is_expected_trade_date or item["result"].data.empty) for item in index_results)
    optional_index_failed = any((not item["core"]) and (item["result"].final_source != "fresh" or not item["result"].is_expected_trade_date or item["result"].data.empty) for item in index_results)
    stock_basic_status = "cache" if pool_used_cache else "fresh"
    stock_basic_is_trading_critical = pool_used_cache and not _stock_basic_has_local_fallback(stock_pool)
    data_quality_report = compute_data_quality_report(
        core_index_used_cache=core_index_used_cache,
        optional_index_failed=optional_index_failed,
        any_stock_used_cache=stock_stats.get("cache_fallback_count", 0) > 0,
        stock_fetch_stats=stock_stats,
        trade_calendar_status=trade_calendar_status,
        stock_basic_status=stock_basic_status,
        stock_basic_is_trading_critical=stock_basic_is_trading_critical,
        market_data_ready_status=ready_report["market_data_ready_status"],
    )
    data_quality_level = data_quality_report["level"]
    settings["data_quality_level"] = data_quality_level
    settings["data_integrity_report"] = data_quality_report
    settings["data_integrity_score"] = data_quality_report["data_integrity_score"]
    settings["confidence_level"] = data_quality_report["confidence_level"]
    settings["confidence_position_multiplier"] = data_quality_report["confidence_position_multiplier"]
    settings["source_health"] = data_quality_report["source_health"]
    settings["data_quality_report"] = data_quality_report
    settings["trading_critical_reasons"] = data_quality_report["trading_critical_reasons"]
    settings["non_critical_warnings"] = data_quality_report["non_critical_warnings"]
    settings["data_quality_final_reason"] = data_quality_report["final_reason"]
    settings["partial_cache_requires_mode_degrade"] = data_quality_report["partial_cache_requires_mode_degrade"]
    settings["stock_basic_status"] = stock_basic_status
    settings["stock_basic_is_trading_critical"] = stock_basic_is_trading_critical
    settings["stock_basic_fallback_basis"] = _fallback_basis_rows(stock_basic_status, stock_basic_is_trading_critical)
    settings["data_issue_list"] = _data_issue_list(stock_results)
    settings["data_quality_summary"] = _build_data_quality_summary(
        core_index_used_cache=core_index_used_cache,
        optional_index_failed=optional_index_failed,
        pool_used_cache=pool_used_cache,
        stock_stats=stock_stats,
        expected_date=expected_date,
        market_data_date=market_data_date,
        trade_calendar_status=trade_calendar_status,
    )
    settings["data_quality_summary"] = _append_fetch_summary_rows(settings["data_quality_summary"], stock_stats)
    used_cache = data_quality_level != "fresh"

    if hs300_result.data.empty:
        console.print("[red]沪深300数据不可用，无法生成有效报告。[/red]")
        return 1

    console.print("[bold]Step 3/5: calculating indicators and filters...[/bold]")
    hs300 = add_indicators(hs300_result.data)
    market_index_table = []
    for item in index_results:
        result = item["result"]
        if result.data.empty:
            market_index_table.append({"index_code": item["index_code"], "index_name": item["index_name"], "state": "failed", "error": result.error or "???????", "data_source_name": result.data_source_name, "fallback_source_used": result.fallback_source_used, "source_attempts_summary": _source_attempts_summary(result.source_attempts)})
            continue
        indexed = add_indicators(result.data)
        latest = indexed.iloc[-1]
        market_index_table.append(
            {
                "index_code": item["index_code"],
                "index_name": item["index_name"],
                "close": round(float(latest["close"]), 2),
                "MA20": round(float(latest["ma20"]), 2),
                "MA60": round(float(latest["ma60"]), 2),
                "close_above_MA20": bool(latest["close"] > latest["ma20"]),
                "close_above_MA60": bool(latest["close"] > latest["ma60"]),
                "state": "strong" if latest["close"] > latest["ma20"] > latest["ma60"] else "neutral" if latest["close"] > latest["ma60"] else "weak",
                "error": result.error or result.data_staleness_reason or "",
                "latest_date": result.latest_date or "",
                "raw_latest_date": result.raw_latest_date or "",
                "effective_latest_date": result.effective_latest_date or result.latest_date or "",
                "final_source": result.final_source,
                "data_source_name": result.data_source_name,
                "fallback_source_used": result.fallback_source_used,
                "source_attempts_summary": _source_attempts_summary(result.source_attempts),
            }
        )

    snapshots_by_symbol, performance_summary = build_stock_snapshots(
        stock_results,
        names,
        settings,
        risk_rules,
        hs300,
    )
    performance_summary["index_indicator_compute_count"] = sum(1 for item in index_results if not item["result"].data.empty)
    settings["performance_summary"] = performance_summary
    settings["data_issue_list"] = snapshot_data_issue_list(snapshots_by_symbol)

    analysis_symbols = select_analysis_pool(
        stock_pool,
        snapshots_by_symbol,
        int(settings.get("analysis_pool_size", 150)),
    )
    analysis_snapshots = subset_snapshots(snapshots_by_symbol, analysis_symbols)
    candidate_symbols = select_candidate_pool(
        analysis_snapshots,
        int(settings.get("candidate_pool_size", 30)),
    )
    candidate_snapshots = subset_snapshots(analysis_snapshots, candidate_symbols)
    analysis_pool = [item for item in stock_pool if str(item.get("symbol", "")).zfill(6) in set(analysis_symbols)]
    candidate_pool = [item for item in analysis_pool if str(item.get("symbol", "")).zfill(6) in set(candidate_symbols)]
    settings["pool_layers"] = [
        {"pool": "scan_pool", "size": len(stock_pool), "source": scan_pool_meta.get("pool_source", "")},
        {"pool": "analysis_pool", "size": len(analysis_pool), "source": "流动性排序并保留行业覆盖"},
        {"pool": "candidate_pool", "size": len(candidate_pool), "source": "沿用现有评分排序"},
    ]
    settings["analysis_pool_size"] = len(analysis_pool)
    settings["candidate_pool_size"] = len(candidate_pool)
    settings["industry_distribution"] = pool_metadata(
        stock_pool,
        version=str(settings.get("pool_version", "v2.1")),
        source=scan_pool_meta.get("pool_source", ""),
    )["industry_distribution"]

    console.print("[bold]Step 4/5: evaluating market state and signals...[/bold]")
    market = evaluate_market_state_from_snapshots(
        hs300,
        analysis_snapshots,
        risk_rules,
        set(analysis_symbols),
        market_index_table,
        data_quality_level,
        settings,
    )
    market.update(evaluate_market_regime(market_index_table, analysis_snapshots))
    market = apply_market_opportunity(market, analysis_snapshots, settings)
    signals = generate_buy_signals_from_snapshots(market, candidate_snapshots, settings, risk_rules, holdings, data_quality_level)
    temp_candidate_symbols = {
        str(item.get("symbol", "")).zfill(6)
        for item in signals.get("candidates", [])
    }
    temp_candidate_symbols.update(
        str(item.get("symbol", "")).zfill(6)
        for item in signals.get("watchlist", [])
        if item.get("review_scope") in {"candidate_data_review", "candidate_risk_review"} or item.get("original_candidate_rank")
    )
    _annotate_cache_fallback_hits(
        stock_stats,
        holding_symbols=holding_symbols,
        style_symbols=confirmed_style_symbols,
        temp_candidate_symbols=temp_candidate_symbols,
    )
    cache_notes = _cache_notes(stock_stats, fetch_log_path, data_quality_level, trade_calendar_status)
    if pool_used_cache:
        cache_notes.append(f"股票基础信息使用缓存或备用池：{pool_error or '实时股票池不可用'}")
    cache_notes.append(f"本次扫描股票数量：{len(stock_pool)}")
    holding_actions = evaluate_holdings_from_snapshots(holdings, snapshots_by_symbol, market, settings, risk_rules)

    console.print("[bold]Step 5/5: writing reports...[/bold]")
    md_path, csv_path = write_reports(market, signals, holding_actions, used_cache, cache_notes, settings)

    console.print(f"[bold]market_state:[/bold]{market['label']}")
    console.print(f"[bold]scan_count:[/bold]{len(stock_pool)}")
    console.print(f"[bold]buy_candidates:[/bold]{len(signals['candidates'])}")
    console.print(f"[bold]Markdown:[/bold]{md_path}")
    console.print(f"[bold]CSV:[/bold]{csv_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A-share main-board multi-style semi-auto signal system")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("daily-signal", help="Generate daily complete-trade-date signal report")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "daily-signal":
        return run_daily_signal()
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
