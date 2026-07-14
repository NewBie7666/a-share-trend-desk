"""Resolve real industry metadata from already available local/provider data."""

from __future__ import annotations

import json
from pathlib import Path

from .utils import CACHE_DIR, ensure_directories


UNKNOWN_INDUSTRY = "未知行业"


def normalize_industry(value) -> str:
    text = str(value or "").strip()
    return text if text and text.lower() not in {"nan", "none", "null", "-"} else UNKNOWN_INDUSTRY


class IndustryResolver:
    """No-network industry resolver with explicit source priority."""

    def __init__(self, local_basic: dict[str, str] | None = None, historical_pool: dict[str, str] | None = None) -> None:
        self.local_basic = local_basic or {}
        self.historical_pool = historical_pool or {}

    def resolve(self, symbol: str, provider_industry=None) -> tuple[str, str]:
        provider = normalize_industry(provider_industry)
        if provider != UNKNOWN_INDUSTRY:
            return provider, "realtime_provider"
        symbol = str(symbol).zfill(6)
        local = normalize_industry(self.local_basic.get(symbol))
        if local != UNKNOWN_INDUSTRY:
            return local, "local_stock_basic_cache"
        historical = normalize_industry(self.historical_pool.get(symbol))
        if historical != UNKNOWN_INDUSTRY:
            return historical, "historical_pool_cache"
        return UNKNOWN_INDUSTRY, "unknown"


def _mapping_from_records(records: list[dict]) -> dict[str, str]:
    return {
        str(item.get("symbol", "")).zfill(6): normalize_industry(item.get("industry"))
        for item in records
        if item.get("symbol") and normalize_industry(item.get("industry")) != UNKNOWN_INDUSTRY
    }


def load_industry_resolver(pool_cache_path: Path | None = None) -> IndustryResolver:
    ensure_directories()
    local_path = CACHE_DIR / "stock_basic_industry.json"
    pool_path = pool_cache_path or (CACHE_DIR / "main_board_pool_cache.json")
    local_basic: dict[str, str] = {}
    historical: dict[str, str] = {}
    try:
        payload = json.loads(local_path.read_text(encoding="utf-8")) if local_path.exists() else {}
        local_basic = _mapping_from_records(list(payload.get("records", [])))
    except (OSError, ValueError, json.JSONDecodeError):
        local_basic = {}
    try:
        payload = json.loads(pool_path.read_text(encoding="utf-8")) if pool_path.exists() else {}
        historical = _mapping_from_records(list(payload.get("records", [])))
    except (OSError, ValueError, json.JSONDecodeError):
        historical = {}
    return IndustryResolver(local_basic, historical)


def persist_known_industries(records: list[dict]) -> None:
    known = {
        str(item.get("symbol", "")).zfill(6): normalize_industry(item.get("industry"))
        for item in records
        if item.get("symbol") and normalize_industry(item.get("industry")) != UNKNOWN_INDUSTRY
    }
    if not known:
        return
    ensure_directories()
    path = CACHE_DIR / "stock_basic_industry.json"
    try:
        existing_payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        existing = _mapping_from_records(list(existing_payload.get("records", [])))
    except (OSError, ValueError, json.JSONDecodeError):
        existing = {}
    existing.update(known)
    rows = [{"symbol": symbol, "industry": industry} for symbol, industry in sorted(existing.items())]
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps({"records": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def industry_coverage_audit(records: list[dict]) -> dict:
    total = len(records)
    unknown = sum(normalize_industry(item.get("industry")) == UNKNOWN_INDUSTRY for item in records)
    known_ratio = (total - unknown) / total if total else 0.0
    enabled = total > 0 and unknown / total <= 0.50
    sources: dict[str, int] = {}
    for item in records:
        source = str(item.get("industry_source") or "unknown")
        sources[source] = sources.get(source, 0) + 1
    return {
        "industry_known_ratio": round(known_ratio, 4),
        "industry_unknown_count": unknown,
        "industry_source": sources,
        "industry_coverage_enabled": enabled,
        "industry_coverage_reason": (
            "行业数据可用，启用分析池行业覆盖加分"
            if enabled else "行业数据缺失比例超过50%，关闭行业覆盖加分；仅影响排序解释，不影响交易资格"
        ),
    }
