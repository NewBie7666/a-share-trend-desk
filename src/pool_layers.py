from __future__ import annotations

from collections import Counter


def _latest_amount(snapshot) -> float:
    latest = getattr(snapshot, "latest", None)
    if latest is None:
        return 0.0
    return float(latest.get("amount", 0) or 0)


def _industry(item: dict) -> str:
    return str(item.get("industry") or "未知行业")


def pool_metadata(pool: list[dict], *, version: str, source: str) -> dict:
    distribution = Counter(_industry(item) for item in pool)
    return {
        "pool_version": version,
        "pool_size": len(pool),
        "pool_source": source,
        "industry_distribution": dict(sorted(distribution.items())),
    }


def select_analysis_pool(
    scan_pool: list[dict],
    snapshots_by_symbol: dict,
    limit: int = 150,
) -> list[str]:
    """Keep one liquid representative per available industry, then fill by liquidity."""
    records = []
    for item in scan_pool:
        symbol = str(item.get("symbol", "")).zfill(6)
        snapshot = snapshots_by_symbol.get(symbol)
        records.append((symbol, _industry(item), _latest_amount(snapshot)))
    records.sort(key=lambda item: (-item[2], item[0]))

    selected: list[str] = []
    seen_industries: set[str] = set()
    for symbol, industry, _ in records:
        if industry == "未知行业" or industry in seen_industries:
            continue
        selected.append(symbol)
        seen_industries.add(industry)
        if len(selected) >= limit:
            return selected
    for symbol, _, _ in records:
        if symbol not in selected:
            selected.append(symbol)
        if len(selected) >= limit:
            break
    return selected


def select_timing_pool(snapshots_by_symbol: dict, limit: int = 30) -> list[str]:
    """Rank only existing snapshot scores; hard filters remain owned by signals.py."""
    scored = []
    for symbol, snapshot in snapshots_by_symbol.items():
        score = getattr(snapshot, "score", None) or {}
        if not getattr(snapshot, "usable_for_signal", False) or not score:
            continue
        scored.append((symbol, float(score.get("final_score", score.get("score", 0)) or 0)))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return [symbol for symbol, _ in scored[:limit]]


def timing_pool_audit(snapshots_by_symbol: dict, selected_symbols: list[str]) -> dict[str, dict]:
    """Explain timing-pool selection using only fields available before Timing runs."""
    selected = set(selected_symbols)
    scored = []
    for symbol, snapshot in snapshots_by_symbol.items():
        score = getattr(snapshot, "score", None) or {}
        value = float(score.get("final_score", score.get("score", 0)) or 0)
        scored.append((symbol, value, snapshot))
    scored.sort(key=lambda item: (-item[1], item[0]))
    output: dict[str, dict] = {}
    for rank, (symbol, value, snapshot) in enumerate(scored, start=1):
        codes = ["FACTOR_SCORE_TOP"] if symbol in selected else ["RANK_BELOW_TIMING_LIMIT"]
        score_data = getattr(snapshot, "score", None) or {}
        if str(score_data.get("style_state", "")) == "strong":
            codes.append("STRONG_STYLE_CANDIDATE")
        if _latest_amount(snapshot) > 0:
            codes.append("LIQUIDITY_SUPPORT")
        reason_map = {
            "FACTOR_SCORE_TOP": "个股因子分排名进入Timing池",
            "STRONG_STYLE_CANDIDATE": "强风格候选",
            "LIQUIDITY_SUPPORT": "成交活跃度支持",
            "RANK_BELOW_TIMING_LIMIT": "因子分排名低于Timing池上限",
        }
        output[symbol] = {
            "selected_for_timing": symbol in selected,
            "timing_selection_rank": rank if symbol in selected else None,
            "timing_selection_score": round(value, 2),
            "timing_selection_reason_codes": codes,
            "timing_selection_reason": "；".join(reason_map[code] for code in codes),
        }
    return output


def select_candidate_pool(snapshots_by_symbol: dict, limit: int = 10) -> list[str]:
    return select_timing_pool(snapshots_by_symbol, limit)


def subset_snapshots(snapshots_by_symbol: dict, symbols: list[str]) -> dict:
    return {symbol: snapshots_by_symbol[symbol] for symbol in symbols if symbol in snapshots_by_symbol}
