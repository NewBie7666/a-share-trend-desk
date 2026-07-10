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


def select_candidate_pool(snapshots_by_symbol: dict, limit: int = 30) -> list[str]:
    """Rank only existing snapshot scores; hard filters remain owned by signals.py."""
    scored = []
    for symbol, snapshot in snapshots_by_symbol.items():
        score = getattr(snapshot, "score", None) or {}
        if not getattr(snapshot, "usable_for_signal", False) or not score:
            continue
        scored.append((symbol, float(score.get("final_score", score.get("score", 0)) or 0)))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return [symbol for symbol, _ in scored[:limit]]


def subset_snapshots(snapshots_by_symbol: dict, symbols: list[str]) -> dict:
    return {symbol: snapshots_by_symbol[symbol] for symbol in symbols if symbol in snapshots_by_symbol}
