from __future__ import annotations

import json

import pandas as pd

from src.data_fetcher import FetchResult
from src.decision_log import write_daily_decision_log
from src.report import write_reports
from src.snapshots import StockSnapshot
from src.v3_decision import calculate_timing_score, calculate_v3_score, evaluate_v3_risk, market_position_multiplier
from src.v3_signals import annotate_v3_holding_boundary, generate_v3_signals_from_snapshots


V3 = {
    "rule_version": "v3.0.0", "buy_threshold": 75, "watch_threshold": 60,
    "minimum_stock_factor_for_buy": 60, "top_pool_size": 30, "max_candidates": 3,
    "weights": {"stock_factor": 0.5, "timing": 0.3, "market": 0.2},
    "structure_adjustment": {"breakout": 10, "pullback": 5, "acceleration": 0, "distribution": -20},
}


RULES = {
    "filters": {"long_upper_shadow_ratio": 1.5, "long_upper_shadow_amount_multiple": 1.5, "long_upper_shadow_close_below_high": 0.03},
    "condition_order": {"trigger_close_multiple": 1.01, "trigger_ma20_multiple": 1.005, "take_profit_reduce_multiple": 1.25, "take_profit_strong_multiple": 1.40},
}


def frame(*, return_10d=0.05, deviation=0.03, amount=500_000_000, structure="breakout"):
    rows = 130
    close = pd.Series([10.0] * rows)
    if structure == "breakdown":
        close.iloc[-1] = 8.0
    return pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=rows, freq="B").strftime("%Y-%m-%d"),
        "open": 9.9, "high": 10.1, "low": 9.8, "close": close,
        "volume": 1_000_000, "amount": amount, "avg_amount_20d": 500_000_000,
        "ma5": 9.9, "ma20": 9.5, "ma60": 9.0, "ma20_slope": 0.01,
        "macd_dif": 1.0, "macd_dea": 0.5, "rsi14": 60,
        "kdj_k": 55, "kdj_d": 50, "kdj_j": 65,
        "return_5d": 0.02, "return_10d": return_10d,
        "ma20_deviation": deviation, "relative_strength_20d": 0.05,
    })


def snapshot(symbol="600001", *, source="fresh", stock_score=90, structure="breakout", entry=80, df=None):
    df = frame(structure=structure) if df is None else df
    result = FetchResult(
        symbol=symbol, name="测试股份", data=df, used_cache=source == "cache", final_source=source,
        latest_date="2026-07-10", effective_latest_date="2026-07-10", expected_trade_date="2026-07-10",
        is_expected_trade_date=True,
    )
    score = {
        "symbol": symbol, "name": "测试股份", "score": stock_score, "final_score": stock_score,
        "best_style": "manufacturing", "style_state": "strong", "style_scores": {},
        "close_price": float(df.iloc[-1]["close"]), "ma20": 9.5, "ma60": 9.0,
        "reason": "个股趋势与活跃度评分", "risk_note": "注意新增开仓风险",
    }
    return StockSnapshot(
        symbol, "测试股份", result, df, df, [], score, "manufacturing", "strong", {},
        {"structure_state": structure, "entry_quality_score": entry, "timing_decision": "WAIT"}, None,
    )


def settings():
    return {
        "initial_cash": 30000, "lot_size": 100, "account_risk_limit": 0.025,
        "expected_trade_date": "2026-07-10", "report_date": "2026-07-10", "generated_at": "2026-07-10 16:30:00",
        "pool_version": "test", "v3": V3,
        "position_rules": {"attack": {"max_total_position": 0.8, "max_single_position": 0.3, "max_new_candidates": 3, "min_score": 70, "max_account_risk_pct": 0.03}},
    }


def test_v3_score_uses_50_30_20_and_stock_minimum_blocks_buy():
    high = calculate_v3_score(90, 80, 80, 0, V3)
    assert high["total_score"] == 85
    assert high["v3_action"] == "BUY"
    low_stock = calculate_v3_score(59, 100, 100, 0, V3)
    assert low_stock["total_score"] >= 75
    assert low_stock["v3_action"] == "WATCH"


def test_timing_structure_adjustment_and_market_multiplier():
    assert calculate_timing_score(70, "breakout", V3)["timing_score"] == 80
    assert calculate_timing_score(70, "distribution", V3)["timing_score"] == 50
    assert [market_position_multiplier(x) for x in (10, 30, 50, 70, 90)] == [0, 0.25, 0.5, 0.8, 1.0]


def test_single_risk_only_penalizes_but_compound_risk_is_extreme():
    single = evaluate_v3_risk(frame(return_10d=0.36, deviation=0.03), "breakout", RULES["filters"])
    assert single["risk_adjustment"] == 20
    assert single["extreme_risk"] is False
    compound = evaluate_v3_risk(frame(return_10d=0.36, deviation=0.21), "breakout", RULES["filters"])
    assert compound["extreme_risk"] is True
    assert compound["risk_adjustment"] == 25


def test_distribution_plus_abnormal_volume_is_extreme():
    risk = evaluate_v3_risk(frame(amount=800_000_000), "distribution", RULES["filters"])
    assert risk["extreme_risk"] is True
    assert "high_distribution" in risk["risk_flags"]


def test_v3_buy_requires_fresh_and_breakdown_is_blocked():
    market = {"market_score": 90}
    fresh = generate_v3_signals_from_snapshots(market, {"600001": snapshot()}, settings(), RULES)
    assert len(fresh["candidates"]) == 1
    assert fresh["candidates"][0]["new_position_action"] == "BUY"
    cached = generate_v3_signals_from_snapshots(market, {"600001": snapshot(source="cache")}, settings(), RULES)
    assert cached["candidates"] == []
    assert cached["watchlist"][0]["v3_action"] == "WATCH"
    broken = generate_v3_signals_from_snapshots(market, {"600001": snapshot(structure="breakdown")}, settings(), RULES)
    assert broken["candidates"] == []
    assert broken["blocked_list"][0]["new_position_action"] == "BLOCKED"


def test_extreme_risk_never_changes_existing_holding_action():
    risky = frame(return_10d=0.36, deviation=0.21)
    snap = snapshot(df=risky)
    actions = [{"symbol": "600001", "name": "测试股份", "action": "hold", "reason": "原持仓规则继续持有"}]
    annotated = annotate_v3_holding_boundary(actions, {"600001": snap}, settings(), RULES)
    assert annotated[0]["action"] == "hold"
    assert annotated[0]["existing_holding_action"] == "hold"
    assert annotated[0]["holding_action"] == "UNCHANGED"
    assert annotated[0]["new_position_action"] == "BLOCKED"


def test_daily_decision_log_contains_audit_fields(tmp_path, monkeypatch):
    monkeypatch.setattr("src.decision_log.LOG_DIR", tmp_path)
    v3 = generate_v3_signals_from_snapshots({"market_score": 90}, {"600001": snapshot()}, settings(), RULES)
    path = write_daily_decision_log(settings(), {"candidates": []}, v3)
    payload = json.loads(path.read_text(encoding="utf-8"))
    row = payload["v3"]["candidates"][0]
    assert "decision_path" in row
    assert row["signal_price"] == 10.0
    assert row["future_check"] == {}
    assert payload["differences"]


def test_v3_report_and_csv_expose_scoring_fields(tmp_path, monkeypatch):
    monkeypatch.setattr("src.report.REPORT_DIR", tmp_path)
    local_settings = settings()
    local_settings.update({
        "active_decision_engine": "v3", "market_data_date": "2026-07-10",
        "data_quality_summary": [], "performance_summary": {}, "pool_layers": [],
    })
    market = {
        "market_score": 90, "market_regime": "attack", "market_state": "bull", "label": "市场偏强",
        "portfolio_mode": {"portfolio_mode": "attack", "strongest_styles": "", "potential_strong_styles": ""},
    }
    signals = generate_v3_signals_from_snapshots(market, {"600001": snapshot()}, local_settings, RULES)
    md_path, csv_path = write_reports(market, signals, [], False, [], local_settings)
    markdown = md_path.read_text(encoding="utf-8")
    csv_text = csv_path.read_text(encoding="utf-8-sig")
    assert "V3 Decision Funnel" in markdown
    assert "stock_factor_score" in markdown
    assert "risk_adjustment" in csv_text
    assert "new_position_action" in csv_text
