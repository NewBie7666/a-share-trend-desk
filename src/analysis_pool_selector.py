"""Basic-data-only scan pool selector. It deliberately knows nothing about signals."""

from __future__ import annotations

from collections import defaultdict


def _number(item: dict, key: str) -> float:
    try:
        return float(item.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def select_analysis_pool(scan_pool: list[dict], limit: int = 150) -> tuple[list[str], dict[str, str]]:
    """Select by liquidity and industry coverage without final signal features."""
    if limit <= 0:
        return [], {str(item.get("symbol", "")).zfill(6): "analysis_pool_limit" for item in scan_pool}
    amounts = sorted((_number(item, "amount") for item in scan_pool), reverse=True)
    max_amount = amounts[0] if amounts else 1.0
    industry_counts: dict[str, int] = defaultdict(int)
    for item in scan_pool:
        industry_counts[str(item.get("industry") or "未知行业")] += 1

    ranked: list[tuple[float, str, str, str]] = []
    for item in scan_pool:
        symbol = str(item.get("symbol", "")).zfill(6)
        industry = str(item.get("industry") or "未知行业")
        amount = _number(item, "amount")
        turnover = _number(item, "turnover")
        price = _number(item, "price")
        previous_price = _number(item, "previous_price")
        liquidity = min(amount / max(max_amount, 1.0), 1.0) * 40
        coverage = min(1.0 / max(industry_counts[industry], 1) * 8, 1.0) * 30
        trend = (20 if price > 0 and (previous_price <= 0 or price >= previous_price) else 8)
        stability = min(max(turnover, 0.0) / 5.0, 1.0) * 10 if turnover else min(amount / max(max_amount, 1.0), 1.0) * 10
        score = liquidity + coverage + trend + stability
        reason = "成交额与流动性较好；行业覆盖补充；基础价格趋势稳定"
        ranked.append((score, symbol, industry, reason))

    ranked.sort(key=lambda row: (-row[0], row[1]))
    selected: list[str] = []
    reasons: dict[str, str] = {}
    selected_industries: set[str] = set()
    for _, symbol, industry, reason in ranked:
        if industry != "未知行业" and industry not in selected_industries:
            selected.append(symbol)
            selected_industries.add(industry)
            reasons[symbol] = reason
            if len(selected) >= limit:
                break
    for _, symbol, _, reason in ranked:
        if len(selected) >= limit:
            break
        if symbol not in selected:
            selected.append(symbol)
            reasons[symbol] = reason
    for _, symbol, _, _ in ranked:
        reasons.setdefault(symbol, "成交额排名靠后或行业覆盖已满足，未进入分析池")
    return selected, reasons
