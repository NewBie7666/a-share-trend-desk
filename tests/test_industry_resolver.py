from src.industry_resolver import IndustryResolver, industry_coverage_audit


def test_industry_source_priority_and_coverage_audit():
    resolver = IndustryResolver({"600001": "电子"}, {"600002": "医药"})
    assert resolver.resolve("600001", "银行") == ("银行", "realtime_provider")
    assert resolver.resolve("600001", "") == ("电子", "local_stock_basic_cache")
    assert resolver.resolve("600002", None) == ("医药", "historical_pool_cache")
    assert resolver.resolve("600003", None) == ("未知行业", "unknown")
    audit = industry_coverage_audit([
        {"industry": "电子", "industry_source": "realtime_provider"},
        {"industry": "未知行业", "industry_source": "unknown"},
        {"industry": "未知行业", "industry_source": "unknown"},
    ])
    assert audit["industry_coverage_enabled"] is False
    assert audit["industry_unknown_count"] == 2
    assert "不影响交易资格" in audit["industry_coverage_reason"]
