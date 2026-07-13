from datetime import datetime, timedelta
import json

import src.data_fetcher as data_fetcher
from src.system_health import evaluate_system_health


def _settings():
    return {"pool_cache": {"cache_version": 1, "filter_version": "v2.1-pr3", "max_age_days": 20}}


def test_json_pool_cache_requires_matching_versions(tmp_path, monkeypatch):
    monkeypatch.setattr(data_fetcher, "CACHE_DIR", tmp_path)
    records = [{"symbol": "600030", "name": "中信证券", "industry": "金融", "pool_source": "测试"}]
    data_fetcher._write_stock_pool_cache(records, _settings(), "测试provider")
    loaded, meta = data_fetcher._read_stock_pool_cache(_settings())
    assert loaded == records
    assert meta["cache_compatible"] is True
    incompatible = {"pool_cache": {"cache_version": 1, "filter_version": "changed", "max_age_days": 20}}
    loaded, meta = data_fetcher._read_stock_pool_cache(incompatible)
    assert loaded == []
    assert meta["cache_compatible"] is False


def test_expired_cache_marks_health_limited():
    health = evaluate_system_health(
        {"data_quality_level": "fresh", "pool_cache": {"max_age_days": 20}, "pool_cache_metadata": {"cache_age_days": 21}},
        [{"source": "local_cache", "status": "fallback"}],
    )
    assert health["system_health"] == "LIMITED"
    assert any("缓存年龄" in reason for reason in health["health_reasons"])
