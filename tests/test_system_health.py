from src.system_health import evaluate_system_health


def test_system_health_blocked_for_critical_blocker():
    health = evaluate_system_health(
        {"trading_critical_blocker_codes": ["core_index_unavailable"], "data_quality_level": "critical_cache"},
        [{"source": "daily_price", "status": "fallback"}],
    )
    assert health["system_health"] == "BLOCKED"
    assert health["health_position_multiplier"] == 0.0
    assert health["health_reasons"]


def test_system_health_limited_uses_one_multiplier():
    health = evaluate_system_health(
        {"data_quality_level": "partial_cache", "confidence_position_multiplier": 0.8},
        [{"source": "akshare_eastmoney_spot", "status": "success"}],
    )
    assert health["system_health"] == "LIMITED"
    assert health["health_position_multiplier"] == 0.8


def test_system_health_ready_when_primary_data_is_healthy():
    health = evaluate_system_health(
        {"data_quality_level": "fresh"},
        [{"source": "akshare_eastmoney_spot", "status": "success"}],
    )
    assert health["system_health"] == "READY"
    assert health["health_position_multiplier"] == 1.0
