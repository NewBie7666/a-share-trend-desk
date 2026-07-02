from __future__ import annotations


QUALITY_ORDER = ["fresh", "partial_cache", "critical_cache", "stale"]


def degrade_portfolio_mode(mode: str) -> str:
    # Kept for old imports/tests. V2 no longer calls this from trading logic.
    return {
        "attack": "balanced",
        "balanced": "defensive",
        "defensive": "cash",
        "cash": "cash",
    }.get(mode, "cash")


def data_quality_message(level: str) -> str:
    return {
        "fresh": "数据完整性良好。该字段仅为诊断项，不直接决定交易开关。",
        "partial_cache": "数据完整性存在轻度降级。V2 中只降低信号置信度和建议金额，不直接切换为 cash。",
        "critical_cache": "数据完整性明显降级。V2 中作为强风险提示，不再单独冻结全部新开仓。",
        "stale": "数据存在陈旧风险。需重点复核，但全局交易开关仍由市场状态决定。",
    }.get(level, "数据完整性状态未知，仅作为诊断项展示。")


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _confidence_from_score(score: float) -> tuple[str, float]:
    if score >= 0.90:
        return "high", 1.0
    if score >= 0.70:
        return "medium", 0.8
    if score >= 0.50:
        return "low", 0.6
    return "very_low", 0.4


def _diagnostic_level(score: float, cache_ratio: float, failed_ratio: float, stale: bool) -> str:
    if stale:
        return "stale"
    if cache_ratio > 0.20 or failed_ratio > 0.05:
        return "critical_cache"
    if score >= 0.90 and cache_ratio == 0 and failed_ratio == 0:
        return "fresh"
    if score >= 0.50:
        return "partial_cache"
    return "critical_cache"


def compute_data_integrity_report(
    *,
    core_index_used_cache: bool = False,
    optional_index_failed: bool = False,
    any_stock_used_cache: bool = False,
    stale: bool = False,
    stock_fetch_stats: dict | None = None,
    trade_calendar_status: str = "fresh",
    stock_basic_status: str = "fresh",
    stock_basic_is_trading_critical: bool = False,
    candidate_symbols_with_cache: set[str] | None = None,
    candidate_symbols_failed_or_unexpected: set[str] | None = None,
    style_sample_cache_ratio_by_style: dict[str, float] | None = None,
    style_sample_failed_ratio_by_style: dict[str, float] | None = None,
    market_data_ready_status: str = "unknown",
) -> dict:
    """V2 data integrity report.

    This is intentionally diagnostic only. It must not decide portfolio mode,
    cash mode, or whether the system may generate new candidates.
    """
    stats = stock_fetch_stats or {}
    total = int(stats.get("total_symbols", 0) or 0)
    cache_count = int(stats.get("cache_fallback_count", 0) or 0)
    failed_count = int(stats.get("failed_count", 0) or 0)
    stale_cache_count = int(stats.get("stale_cache_count", 0) or 0)
    ready_ratio = float(stats.get("ready_ratio", 1.0) if stats.get("ready_ratio", None) is not None else 1.0)
    cache_ratio = cache_count / total if total else 0.0
    failed_ratio = failed_count / total if total else 0.0

    warnings: list[str] = []
    notes: list[str] = []
    score = 1.0
    score -= cache_ratio * 0.35
    score -= failed_ratio * 0.70
    score -= max(0.0, 1.0 - ready_ratio) * 0.35
    score -= (min(stale_cache_count, total) / total * 0.60) if total else 0.0

    if optional_index_failed:
        score -= 0.03
        warnings.append("可选指数获取失败，只降低数据完整性置信度。")
    if any_stock_used_cache:
        warnings.append(f"个股日线有 {cache_count}/{total} 使用缓存，缓存比例 {cache_ratio:.1%}。")
    if failed_count:
        warnings.append(f"个股日线有 {failed_count}/{total} 不可用于信号，失败比例 {failed_ratio:.1%}。")
    if stale:
        score -= 0.25
        warnings.append("存在陈旧数据诊断标记，已降低完整性评分。")
    if trade_calendar_status != "fresh":
        score -= 0.08 if trade_calendar_status == "fallback_weekday" else 0.20
        warnings.append(f"交易日历状态为 {trade_calendar_status}，只影响数据完整性评分。")
    if stock_basic_is_trading_critical:
        score -= 0.20
        warnings.append("股票基础信息无法可靠判断风险，已降低完整性评分。")
    elif stock_basic_status == "cache":
        score -= 0.03
        warnings.append("股票基础信息使用缓存，但本地字段可执行基础风险过滤。")
    if core_index_used_cache:
        score -= 0.10
        warnings.append("核心指数存在缓存、失败或非预期交易日诊断；市场状态会基于可用指数单独判断。")
    if market_data_ready_status in {"not_ready_after_close", "data_not_ready", "source_unavailable"}:
        score -= {"not_ready_after_close": 0.12, "data_not_ready": 0.10, "source_unavailable": 0.20}.get(
            market_data_ready_status, 0.0
        )
        warnings.append(f"数据就绪状态为 {market_data_ready_status}，只降低置信度，不直接禁止交易。")

    for style, ratio in sorted((style_sample_cache_ratio_by_style or {}).items()):
        if ratio > 0:
            notes.append(f"{style} 风格样本缓存比例 {ratio:.1%}。")
    for style, ratio in sorted((style_sample_failed_ratio_by_style or {}).items()):
        if ratio > 0:
            notes.append(f"{style} 风格样本失败比例 {ratio:.1%}。")

    score = round(_clamp01(score), 4)
    confidence_level, multiplier = _confidence_from_score(score)
    level = _diagnostic_level(score, cache_ratio, failed_ratio, stale)
    if trade_calendar_status == "failed":
        level = "critical_cache"
    if level == "fresh" and trade_calendar_status != "fresh":
        level = "partial_cache"
    source_health = "good" if score >= 0.90 else "usable" if score >= 0.70 else "degraded" if score >= 0.50 else "poor"
    final_reason = (
        f"数据完整性评分 {score:.2f}，只用于修正信号置信度和建议金额，"
        "不直接决定 portfolio_mode 或是否新开仓。"
    )

    return {
        "level": level,
        "data_integrity_score": score,
        "ready_ratio": ready_ratio,
        "cache_ratio": cache_ratio,
        "failed_ratio": failed_ratio,
        "source_health": source_health,
        "confidence_level": confidence_level,
        "confidence_position_multiplier": multiplier,
        "trading_critical_reasons": ["无"],
        "non_critical_warnings": warnings or ["无"],
        "final_reason": final_reason,
        "partial_cache_requires_mode_degrade": False,
        "candidate_symbols_with_cache": sorted(candidate_symbols_with_cache or []),
        "candidate_symbols_failed_or_unexpected": sorted(candidate_symbols_failed_or_unexpected or []),
        "style_integrity_notes": notes,
    }


def compute_data_quality_report(**kwargs) -> dict:
    # Backward-compatible name; V2 returns a data-integrity diagnostic report.
    return compute_data_integrity_report(**kwargs)


def compute_data_quality_level(**kwargs) -> str:
    if "pool_used_cache" in kwargs:
        pool_used_cache = bool(kwargs.pop("pool_used_cache"))
        kwargs.setdefault("stock_basic_status", "cache" if pool_used_cache else "fresh")
        kwargs.setdefault("stock_basic_is_trading_critical", False)
    if "style_sample_cache_ratio" in kwargs:
        ratio = float(kwargs.pop("style_sample_cache_ratio") or 0)
        if ratio:
            kwargs.setdefault("style_sample_cache_ratio_by_style", {"unknown": ratio})
    return compute_data_integrity_report(**kwargs)["level"]
