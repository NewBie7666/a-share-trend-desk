from __future__ import annotations


def evaluate_system_health(settings: dict, provider_results: list[dict] | None = None) -> dict:
    providers = provider_results or []
    blocker_codes = list(settings.get("trading_critical_blocker_codes", []) or [])
    reasons: list[str] = []
    if blocker_codes:
        reasons.append(f"交易关键阻断：{', '.join(blocker_codes)}")
    if providers and not any(item.get("status") in {"success", "fallback"} for item in providers):
        reasons.append("所有关键 provider 均不可用")
    if reasons:
        return {"system_health": "BLOCKED", "health_reasons": reasons, "health_position_multiplier": 0.0}

    level = settings.get("data_quality_level", "fresh")
    cache_meta = settings.get("pool_cache_metadata", {}) or {}
    max_age = int((settings.get("pool_cache", {}) or {}).get("max_age_days", 20))
    cache_expired = cache_meta.get("cache_age_days", 0) > max_age
    fallback_used = any(item.get("source") in {"local_cache", "local_whitelist"} or item.get("status") == "fallback" for item in providers)
    if level != "fresh" or fallback_used or cache_expired:
        details = []
        if level != "fresh":
            details.append(f"数据完整性诊断为 {level}")
        if fallback_used:
            details.append("存在 provider 回退或本地缓存")
        if cache_expired:
            details.append(f"股票池缓存年龄超过 {max_age} 天")
        return {
            "system_health": "LIMITED",
            "health_reasons": details,
            "health_position_multiplier": float(settings.get("confidence_position_multiplier", 0.8) or 0.8),
        }
    return {"system_health": "READY", "health_reasons": ["关键数据与 provider 状态正常"], "health_position_multiplier": 1.0}
