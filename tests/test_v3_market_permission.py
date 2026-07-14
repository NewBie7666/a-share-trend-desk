from src.v3_market_permission import confirm_v3_market_permission, resolve_v3_market_permission


def test_permission_score_bands_and_blockers():
    assert resolve_v3_market_permission(10, False, True)["v3_market_permission"] == "BLOCKED"
    assert resolve_v3_market_permission(30, False, True)["v3_max_candidates"] == 1
    assert resolve_v3_market_permission(42, False, True)["v3_position_multiplier"] == 0.5
    assert resolve_v3_market_permission(70, False, True)["v3_market_permission"] == "BALANCED"
    assert resolve_v3_market_permission(90, False, True)["v3_market_permission"] == "ATTACK"
    assert resolve_v3_market_permission(90, True, True)["v3_market_permission"] == "BLOCKED"
    assert resolve_v3_market_permission(90, False, False)["v3_market_permission"] == "BLOCKED"


def test_upgrade_needs_two_distinct_days_and_downgrade_is_immediate(tmp_path):
    path = tmp_path / "state.json"
    defensive = resolve_v3_market_permission(42, False, True)
    first = confirm_v3_market_permission(defensive, "2026-07-10", 2, path)
    assert first["v3_market_permission"] == "DEFENSIVE"
    attack = resolve_v3_market_permission(90, False, True)
    same_day = confirm_v3_market_permission(attack, "2026-07-10", 2, path)
    assert same_day["v3_market_permission"] == "BALANCED"
    assert same_day["market_permission_confirmed_days"] == 1
    upgraded = confirm_v3_market_permission(attack, "2026-07-11", 2, path)
    assert upgraded["v3_market_permission"] == "ATTACK"
    downgraded = confirm_v3_market_permission(defensive, "2026-07-12", 2, path)
    assert downgraded["v3_market_permission"] == "DEFENSIVE"


def test_corrupt_state_restarts_conservatively(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("not-json", encoding="utf-8")
    result = confirm_v3_market_permission(resolve_v3_market_permission(90, False, True), "2026-07-10", 2, path)
    assert result["v3_market_permission"] == "BALANCED"
    assert any("保守重建" in reason for reason in result["v3_market_reason"])


def test_defensive_is_never_weakened_by_historical_blocked_state(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(
        '{"confirmed_permission":"BLOCKED","pending_permission":"DEFENSIVE",'
        '"pending_confirmation_days":1,"last_trade_date":"2026-07-10"}',
        encoding="utf-8",
    )

    result = confirm_v3_market_permission(
        resolve_v3_market_permission(42, False, True), "2026-07-10", 2, path
    )

    assert result["v3_market_permission"] == "DEFENSIVE"
    assert result["v3_position_multiplier"] == 0.5
    assert result["v3_max_candidates"] == 2
    assert result["permission_confirmation_required"] is False
    assert result["permission_confirmation_status"] == "not_required"


def test_balanced_upgrade_uses_defensive_floor_until_second_day(tmp_path):
    path = tmp_path / "state.json"
    first = confirm_v3_market_permission(
        resolve_v3_market_permission(75, False, True), "2026-07-10", 2, path
    )
    second = confirm_v3_market_permission(
        resolve_v3_market_permission(75, False, True), "2026-07-11", 2, path
    )

    assert first["v3_market_permission"] == "DEFENSIVE"
    assert first["permission_confirmation_status"] == "pending"
    assert second["v3_market_permission"] == "BALANCED"
    assert second["permission_confirmation_status"] == "confirmed"
