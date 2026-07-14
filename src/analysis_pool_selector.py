"""Auditable basic-data-only scan to analysis pool selector."""

from __future__ import annotations

from collections import defaultdict

from .industry_resolver import UNKNOWN_INDUSTRY, industry_coverage_audit, normalize_industry


def _number(item: dict, key: str) -> float:
    try:
        return float(item.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def select_analysis_pool(scan_pool: list[dict], limit: int = 150) -> tuple[list[str], dict[str, dict]]:
    """Select without final signal features and return per-symbol score audit."""
    coverage_audit = industry_coverage_audit(scan_pool)
    coverage_enabled = bool(coverage_audit["industry_coverage_enabled"])
    if limit <= 0:
        return [], {
            str(item.get("symbol", "")).zfill(6): {
                "selected_for_analysis": False,
                "analysis_pool_reason_codes": ["ANALYSIS_POOL_LIMIT"],
                "analysis_pool_reason": "分析池上限为0",
            }
            for item in scan_pool
        }
    amounts = [_number(item, "amount") for item in scan_pool]
    max_amount = max(amounts, default=1.0)
    sorted_amounts = sorted(amounts, reverse=True)
    top_amount_threshold = sorted_amounts[max(min(len(sorted_amounts) // 3, len(sorted_amounts) - 1), 0)] if sorted_amounts else 0
    industry_counts: dict[str, int] = defaultdict(int)
    for item in scan_pool:
        industry_counts[normalize_industry(item.get("industry"))] += 1

    ranked: list[tuple[float, str, str]] = []
    audits: dict[str, dict] = {}
    for item in scan_pool:
        symbol = str(item.get("symbol", "")).zfill(6)
        industry = normalize_industry(item.get("industry"))
        amount = _number(item, "amount")
        turnover = _number(item, "turnover")
        price = _number(item, "price")
        previous_price = _number(item, "previous_price")
        amount_score = min(amount / max(max_amount, 1.0), 1.0) * 40
        industry_bonus = (
            min(1.0 / max(industry_counts[industry], 1) * 8, 1.0) * 30
            if coverage_enabled and industry != UNKNOWN_INDUSTRY else 0.0
        )
        trend_score = 20.0 if price > 0 and (previous_price <= 0 or price >= previous_price) else 8.0
        liquidity_score = min(max(turnover, 0.0) / 5.0, 1.0) * 10 if turnover else min(amount / max(max_amount, 1.0), 1.0) * 10
        score = amount_score + industry_bonus + trend_score + liquidity_score
        codes = []
        if amount >= top_amount_threshold:
            codes.append("TOP_AMOUNT")
        if amount_score >= 20 or liquidity_score >= 5:
            codes.append("HIGH_LIQUIDITY")
        if trend_score >= 20:
            codes.append("BASIC_TREND_STABLE")
        if industry_bonus > 0:
            codes.append("INDUSTRY_COVERAGE")
        audits[symbol] = {
            "analysis_pool_score": round(score, 2),
            "amount_score": round(amount_score, 2),
            "liquidity_score": round(liquidity_score, 2),
            "basic_trend_score": round(trend_score, 2),
            "industry_coverage_bonus": round(industry_bonus, 2),
            "analysis_pool_reason_codes": codes,
            "analysis_pool_reason": "；".join({
                "TOP_AMOUNT": "成交额位于前列",
                "HIGH_LIQUIDITY": "流动性支持",
                "BASIC_TREND_STABLE": "基础价格趋势稳定",
                "INDUSTRY_COVERAGE": "补充真实行业覆盖",
            }[code] for code in codes) or "基础字段有效",
            "selected_for_analysis": False,
        }
        ranked.append((score, symbol, industry))

    ranked.sort(key=lambda row: (-row[0], row[1]))
    selected: list[str] = []
    seen_industries: set[str] = set()
    if coverage_enabled:
        for _, symbol, industry in ranked:
            if industry != UNKNOWN_INDUSTRY and industry not in seen_industries:
                selected.append(symbol)
                seen_industries.add(industry)
                if len(selected) >= limit:
                    break
    for _, symbol, _ in ranked:
        if len(selected) >= limit:
            break
        if symbol not in selected:
            selected.append(symbol)
    selected_set = set(selected)
    for rank, (_, symbol, _) in enumerate(ranked, start=1):
        audits[symbol]["analysis_pool_rank"] = rank
        audits[symbol]["selected_for_analysis"] = symbol in selected_set
        if symbol not in selected_set:
            audits[symbol]["analysis_pool_reason_codes"].append("RANK_BELOW_ANALYSIS_LIMIT")
            audits[symbol]["analysis_pool_reason"] = "综合基础排序低于分析池上限"
    return selected, audits
