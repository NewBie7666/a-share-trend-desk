from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src import main
from src.watchlist_diagnosis import diagnose_watchlist_stock
from src.watchlist_io import (
    is_valid_stock_name,
    load_market_context,
    load_stock_name_cache,
    merge_watchlist_inputs,
    update_watchlist_history,
)


def frame(*, closes=None, ma20=100.0, ma60=90.0, slope=0.01, rs=0.05, deviation=0.02):
    closes = closes or [102.0] * 6
    rows = []
    for index, close in enumerate(closes):
        rows.append({
            "date": pd.Timestamp("2026-07-01") + pd.Timedelta(days=index),
            "open": close - 0.2,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1_000_000,
            "amount": 100_000_000,
            "avg_amount_20d": 100_000_000,
            "ma5": close - 0.5,
            "ma20": ma20,
            "ma60": ma60,
            "ma20_slope": slope,
            "relative_strength_20d": rs,
            "ma20_deviation": deviation,
            "return_10d": 0.02,
        })
    return pd.DataFrame(rows)


def snapshot(df, *, structure="breakout", score=80.0, name="测试股份", fetch_result=None):
    return SimpleNamespace(
        symbol="002241",
        name=name,
        usable_for_signal=True,
        indicators=df,
        score={"score": score},
        timing={"structure_state": structure, "timing_decision": "BUY", "timing_reason_codes": ["PRICE_NEAR_HIGH"]},
        fetch_result=fetch_result,
    )


def holding(**overrides):
    return {
        "symbol": "002241",
        "name": "测试股份",
        "position_source": "holding",
        "holding_status": "holding",
        "cost_price": 100,
        "quantity": 1000,
        **overrides,
    }


def test_diagnosis_formula_and_market_is_display_only():
    item = snapshot(frame())
    low_market = diagnose_watchlist_stock(item, {"market_score": 10, "market_environment": "弱"}, holding())
    high_market = diagnose_watchlist_stock(item, {"market_score": 90, "market_environment": "强"}, holding())
    assert low_market["raw_score_before_risk"] == 84.0
    assert low_market["risk_penalty"] == 0.0
    assert low_market["final_diagnosis_score"] == 84.0
    assert low_market["diagnosis_action"] == "ADD"
    assert high_market["final_diagnosis_score"] == low_market["final_diagnosis_score"]
    assert high_market["diagnosis_action"] == low_market["diagnosis_action"]


def test_watching_stock_can_only_consider_or_watch():
    good = diagnose_watchlist_stock(snapshot(frame()), {}, {"holding_status": "watching", "position_source": "watchlist"})
    weak = diagnose_watchlist_stock(
        snapshot(frame(closes=[80, 80, 80, 80, 80, 80], ma20=100, ma60=90, slope=-0.01, rs=-0.05), structure="breakdown", score=95),
        {},
        {"holding_status": "watching", "position_source": "watchlist"},
    )
    assert good["diagnosis_action"] == "CONSIDER"
    assert weak["diagnosis_action"] == "WATCH"
    assert {good["diagnosis_action"], weak["diagnosis_action"]} <= {"CONSIDER", "WATCH"}


def test_single_breakdown_reduces_but_confirmed_breakdown_exits():
    single = frame(closes=[105, 105, 105, 105, 105, 80], ma20=100, ma60=90, slope=-0.01, rs=-0.05)
    confirmed = frame(closes=[105, 80, 80, 80, 80, 80], ma20=100, ma60=90, slope=-0.01, rs=-0.05)
    single_result = diagnose_watchlist_stock(snapshot(single, structure="breakdown"), {}, holding())
    confirmed_result = diagnose_watchlist_stock(snapshot(confirmed, structure="breakdown"), {}, holding())
    assert single_result["diagnosis_action"] == "REDUCE"
    assert single_result["diagnostic_facts"]["confirmed_long_term_breakdown"] is False
    assert confirmed_result["diagnosis_action"] == "EXIT"
    assert confirmed_result["diagnostic_facts"]["confirmed_long_term_breakdown"] is True


def test_risk_name_exit_and_holding_cost_audit():
    result = diagnose_watchlist_stock(snapshot(frame(), name="*ST测试"), {}, holding(name="*ST测试"))
    assert result["diagnosis_action"] == "EXIT"
    assert result["holding_analysis"]["profit_rate"] == 0.02
    assert result["holding_analysis"]["distance_to_cost"] == 2.0
    assert result["holding_analysis"]["holding_market_value"] == 102000.0
    assert result["holding_analysis"]["unrealized_pnl"] == 2000.0


def test_input_merge_priority_and_source():
    holdings = pd.DataFrame([{"symbol": "002241", "name": "持仓名称", "quantity": 1000, "cost_price": 8.35}])
    watchlist = [
        {"symbol": "002241", "name": "自选名称", "cost": 9.0, "shares": 200},
        {"symbol": "600036", "name": "招商银行"},
    ]
    rows = merge_watchlist_inputs(holdings, watchlist, ["002241", "000001"])
    by_symbol = {row["symbol"]: row for row in rows}
    assert by_symbol["002241"]["position_source"] == "both"
    assert by_symbol["002241"]["holding_status"] == "holding"
    assert by_symbol["002241"]["name"] == "持仓名称"
    assert by_symbol["002241"]["cost"] == 8.35
    assert by_symbol["600036"]["position_source"] == "watchlist"
    assert by_symbol["000001"]["position_source"] == "manual"


def test_market_context_requires_matching_expected_date(tmp_path):
    directory = tmp_path / "daily_decision"
    directory.mkdir()
    payload = {
        "market_context": {
            "expected_trade_date": "2026-07-13",
            "market_score": 42,
            "market_environment": "normal",
            "portfolio_mode": "defensive",
        }
    }
    (directory / "2026-07-14.json").write_text(json.dumps(payload), encoding="utf-8")
    matched = load_market_context("2026-07-13", directory)
    stale = load_market_context("2026-07-14", directory)
    assert matched["market_context_source"] == "daily_signal"
    assert matched["market_score"] == 42
    assert matched["portfolio_mode"] == "defensive"
    assert stale["market_context_source"] == "unavailable"
    assert stale["market_score"] is None


def test_market_context_normalizes_legacy_portfolio_object(tmp_path):
    directory = tmp_path / "daily_decision"
    directory.mkdir()
    payload = {"market_context": {"expected_trade_date": "2026-07-13", "portfolio_mode": {"final_portfolio_mode": "cash"}}}
    (directory / "2026-07-14.json").write_text(json.dumps(payload), encoding="utf-8")
    assert load_market_context("2026-07-13", directory)["portfolio_mode"] == "cash"


def test_market_context_requires_v3_engine_and_keeps_fields_separate(tmp_path):
    directory = tmp_path / "daily_decision"
    directory.mkdir()
    context = {
        "market_context_version": "v3.1.1",
        "market_context_trade_date": "2026-07-13",
        "decision_engine": "v3",
        "market_condition": "normal",
        "market_score": 28,
        "market_regime": "cash",
        "portfolio_mode": "cash",
        "v3_market_permission": "LIMITED",
    }
    (directory / "2026-07-13.json").write_text(json.dumps({"market_context": context}), encoding="utf-8")
    loaded = load_market_context("2026-07-13", directory)
    assert loaded["market_context_compatibility_mode"] == "current"
    assert loaded["market_condition"] == "normal"
    assert loaded["market_regime"] == "cash"
    assert loaded["v3_market_permission"] == "LIMITED"
    context["decision_engine"] = "v2"
    (directory / "2026-07-13.json").write_text(json.dumps({"market_context": context}), encoding="utf-8")
    assert load_market_context("2026-07-13", directory)["market_context_source"] == "unavailable"


def test_history_same_day_overwrites_and_cross_day_compares(tmp_path):
    path = tmp_path / "history.json"
    base = diagnose_watchlist_stock(snapshot(frame()), {}, holding())
    first = update_watchlist_history([{**base, "diagnosis_trade_date": "2026-07-10"}], "2026-07-10", path=path)
    same_day = update_watchlist_history([{**base, "diagnosis_trade_date": "2026-07-10", "final_diagnosis_score": 83}], "2026-07-10", path=path)
    next_day = update_watchlist_history([{**base, "diagnosis_trade_date": "2026-07-11", "final_diagnosis_score": 80, "trend_state": "neutral"}], "2026-07-11", path=path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert first[0]["history_change"]["change_reason_codes"] == ["FIRST_DIAGNOSIS"]
    assert len(payload["symbols"]["002241"]) == 2
    assert same_day[0]["history_change"]["change_reason_codes"] == ["FIRST_DIAGNOSIS"]
    assert next_day[0]["history_change"]["diagnosis_score_change"] == -3.0
    assert "TREND_CHANGED" in next_day[0]["history_change"]["change_reason_codes"]
    assert payload["watchlist_history_schema_version"] == "v3.1.1"
    assert payload["diagnosis_rule_version"] == "watchlist_v1"
    assert payload["watchlist_report_version"] == "v3.1.1"


def test_price_uses_effective_trade_date_row_not_last_row():
    df = frame(closes=[100, 101, 102, 103, 104, 105])
    effective = pd.Timestamp(df.iloc[-2]["date"]).strftime("%Y-%m-%d")
    fetch = SimpleNamespace(
        effective_latest_date=effective,
        expected_trade_date=effective,
        final_source="fresh",
        provider_success=True,
        used_cache=False,
        data_source_name="akshare_eastmoney",
        name="测试股份",
    )
    result = diagnose_watchlist_stock(snapshot(df, fetch_result=fetch), {"market_context_trade_date": effective}, holding())
    assert result["current_price"] == 104.0
    assert result["price_trade_date"] == effective
    assert result["price_source_type"] == "fresh"
    assert result["price_provider"] == "akshare_eastmoney"
    assert result["price_validation_status"] == "VALID"


def test_stale_cache_price_is_reference_only_and_forces_watch():
    df = frame()
    effective = pd.Timestamp(df.iloc[-1]["date"]).strftime("%Y-%m-%d")
    fetch = SimpleNamespace(
        effective_latest_date=effective,
        expected_trade_date="2026-07-13",
        final_source="cache",
        provider_success=False,
        used_cache=True,
        data_source_name="local_cache",
        name="测试股份",
    )
    result = diagnose_watchlist_stock(snapshot(df, fetch_result=fetch), {}, holding())
    assert result["current_price"] is None
    assert result["last_available_price"] == 102.0
    assert result["price_source_type"] == "cache"
    assert result["diagnosis_data_status"] == "STALE"
    assert result["diagnosis_action"] == "WATCH"


def test_name_validation_and_single_cache_resolution(tmp_path):
    path = tmp_path / "pool.json"
    path.write_text(json.dumps({"records": [{"symbol": "600036", "name": "招商银行"}]}), encoding="utf-8")
    cache = load_stock_name_cache(path)
    item = snapshot(frame(), name="600036")
    item.symbol = "600036"
    item.fetch_result = SimpleNamespace(
        effective_latest_date="2026-07-06", expected_trade_date="2026-07-06",
        final_source="fresh", provider_success=True, used_cache=False,
        data_source_name="provider", name="未知股票",
    )
    result = diagnose_watchlist_stock(item, {}, {"symbol": "600036", "name": "-", "holding_status": "watching"}, name_cache=cache)
    assert is_valid_stock_name("600036", "600036") is False
    assert result["name"] == "招商银行"
    assert result["name_source"] == "stock_basic_cache"


def test_history_preserves_other_symbols_and_dates(tmp_path):
    path = tmp_path / "history.json"
    first = diagnose_watchlist_stock(snapshot(frame()), {}, holding())
    second = {**first, "symbol": "600036", "name": "招商银行"}
    update_watchlist_history([{**first, "diagnosis_trade_date": "2026-07-10"}, {**second, "diagnosis_trade_date": "2026-07-10"}], "ignored", path=path)
    update_watchlist_history([{**first, "diagnosis_trade_date": "2026-07-11"}], "ignored", path=path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(payload["symbols"]["002241"]) == 2
    assert len(payload["symbols"]["600036"]) == 1


def test_data_unavailable_is_watch_only():
    item = SimpleNamespace(symbol="600036", name="招商银行", usable_for_signal=False, indicators=pd.DataFrame(), score=None, timing={})
    result = diagnose_watchlist_stock(item, {}, holding(symbol="600036"))
    assert result["diagnosis_action"] == "WATCH"
    assert result["final_diagnosis_score"] is None
    assert result["reason_codes"] == ["DATA_UNAVAILABLE"]


def test_watchlist_command_is_isolated_from_v3_and_holding_actions(tmp_path, monkeypatch):
    item = snapshot(frame())
    monkeypatch.setattr(main, "load_config", lambda: {
        "settings": {"indicator_version": "indicator_v2", "timing_version": "timing_v1", "watchlist_diagnosis": {"version": "watchlist_v1"}},
        "risk_rules": {"filters": {}},
        "stock_pool": [],
    })
    monkeypatch.setattr(main, "expected_trade_date", lambda *args, **kwargs: ("2026-07-13", "fresh", ["2026-07-13"]))
    monkeypatch.setattr(main, "load_holdings", lambda: pd.DataFrame())
    monkeypatch.setattr(main, "load_watchlist", lambda: [])
    monkeypatch.setattr(main, "load_stock_name_cache", lambda: {})
    monkeypatch.setattr(main, "load_market_context", lambda date: {"market_context_source": "unavailable", "market_score": None, "market_environment": "UNKNOWN", "portfolio_mode": None})
    monkeypatch.setattr(main, "fetch_index_daily", lambda *args, **kwargs: SimpleNamespace(data=pd.DataFrame()))
    monkeypatch.setattr(main, "fetch_stocks", lambda *args, **kwargs: [])
    monkeypatch.setattr(main, "build_stock_snapshots", lambda *args, **kwargs: ({"002241": item}, {"snapshot_count": 1}))
    monkeypatch.setattr(main, "update_watchlist_history", lambda rows, *args, **kwargs: rows)
    report_path = tmp_path / "watchlist.md"
    monkeypatch.setattr(main, "write_watchlist_report", lambda *args, **kwargs: report_path)
    monkeypatch.setattr(main, "generate_v3_signals_from_snapshots", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("V3 must not run")))
    monkeypatch.setattr(main, "evaluate_holdings_from_snapshots", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("holdings must not run")))
    assert main.run_watchlist_diagnosis(manual_symbols=["002241"]) == 0
