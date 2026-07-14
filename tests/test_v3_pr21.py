from __future__ import annotations

import pandas as pd
import threading
import time

from src import data_fetcher
from src.data_fetcher import FetchResult, fetch_stocks, summarize_stock_fetches
from src.provider_parallel import AdaptiveWorkerState, SessionProviderCircuits
from src.timing import evaluate_timing
from src.v3_decision import calculate_v3_score, evaluate_v3_risk
from src.v3_score_diagnostics import analyze_score_distribution, build_v3_score_diagnostics, score_percentile
from src.utils import load_config


def indicator_frame() -> pd.DataFrame:
    rows = []
    for index in range(130):
        close = 10 + index * 0.02
        rows.append({
            "date": f"2026-01-{(index % 28) + 1:02d}",
            "open": close - 0.05, "high": close + 0.10, "low": close - 0.10,
            "close": close, "volume": 1000, "amount": 100000,
            "ma5": close - 0.02, "ma20": close - 0.10, "ma60": close - 0.30,
            "ma20_slope": 0.02, "relative_strength_20d": 0.05,
            "return_5d": 0.02, "return_10d": 0.04,
            "avg_amount_20d": 100000, "ma20_deviation": 0.01,
            "kdj_k": 60, "kdj_d": 55, "kdj_j": 70,
            "macd_dif": 0.2, "macd_dea": 0.1,
        })
    return pd.DataFrame(rows)


def test_score_diagnostics_percentiles_spread_and_warnings():
    scores = [60, 70, 80, 90, 100]
    report = analyze_score_distribution(scores)
    assert report["p10"] == 64
    assert report["p50"] == 80
    assert report["p90"] == 96
    assert report["score_spread"] == 40
    assert report["p90_minus_p10"] == 32
    assert score_percentile(80, scores) == 50

    concentrated = build_v3_score_diagnostics(
        [{"stock_factor_score": 95}] * 20,
        [{"stock_factor_score": 95, "timing_score": 70, "total_score": 75}] * 10,
    )
    assert concentrated["score_concentration_warning"] is True
    assert concentrated["score_dispersion_warning"] is True


def test_risk_audit_does_not_change_score_calculation():
    frame = indicator_frame()
    frame.loc[frame.index[-1], "ma20_deviation"] = 0.13
    risk = evaluate_v3_risk(frame, "distribution", {})
    assert risk["risk_adjustment"] == 10
    assert risk["risk_adjustment_audit"]["actual_penalty"] == 10
    assert risk["risk_adjustment_source"] == "short_term_overheated"
    assert risk["risk_level"] == "MEDIUM"
    assert all(item["risk_name"] != "distribution" for item in risk["risk_components"])
    assert all({"observed_value", "threshold", "condition_expression", "evidence"} <= set(item) for item in risk["risk_components"])
    assert all(item["evidence"] == ("命中触发条件" if item["triggered"] else "未达到触发条件") for item in risk["risk_components"])
    before = calculate_v3_score(90, 60, 70, 10, {"weights": {"stock_factor": 0.5, "timing": 0.3, "market": 0.2}})
    after = calculate_v3_score(90, 60, 70, risk["risk_adjustment"], {"weights": {"stock_factor": 0.5, "timing": 0.3, "market": 0.2}})
    assert before == after


def test_timing_reason_codes_are_audit_only():
    frame = indicator_frame()
    first = evaluate_timing(frame)
    assert first["timing_reason_codes"]
    original_decision = first["timing_decision"]
    first["timing_reason_texts"] = ["展示文本已修改"]
    assert evaluate_timing(frame)["timing_decision"] == original_decision


def healthy_attempts(count=20):
    return [{"status": "success", "error": "", "error_category": ""} for _ in range(count)]


def failed_attempts(count=1):
    return [{"status": "failed", "error": "RemoteDisconnected", "error_category": "network_error"} for _ in range(count)]


def test_adaptive_worker_state_uses_fixed_nodes_and_freeze_window():
    state = AdaptiveWorkerState(workers=4)
    assert state.evaluate(healthy_attempts()) == 6
    assert state.evaluate(healthy_attempts()) == 6  # frozen
    assert state.evaluate(healthy_attempts()) == 8
    assert state.evaluate(healthy_attempts()) == 8  # frozen
    assert state.evaluate(failed_attempts()) == 4
    assert state.evaluate(healthy_attempts()) == 4  # frozen
    assert state.evaluate(failed_attempts()) == 2
    assert state.evaluate(healthy_attempts()) == 2  # frozen
    assert state.evaluate(healthy_attempts()) == 2  # first recovery window
    assert state.evaluate(healthy_attempts()) == 4
    assert all(item["workers_after"] in {1, 2, 4, 6, 8} for item in state.history)

    immediate = AdaptiveWorkerState(workers=4)
    assert immediate.evaluate(healthy_attempts()) == 6
    assert immediate.evaluate(failed_attempts()) == 4  # failures bypass upgrade freeze


def test_provider_circuit_counts_only_provider_failures():
    now = [0.0]
    circuits = SessionProviderCircuits(["primary"], failure_threshold=5, cooldown_seconds=120, clock=lambda: now[0])
    ignored = [
        {"source_name": "primary", "status": "failed", "error_category": "empty_data"},
        {"source_name": "primary", "status": "failed", "error_category": "insufficient_history"},
        {"source_name": "primary", "status": "failed", "error_category": "data_not_ready"},
    ]
    for attempt in ignored:
        circuits.record_symbol_result([attempt])
    assert circuits.open_providers() == []
    for _ in range(5):
        circuits.record_symbol_result([{"source_name": "primary", "status": "failed", "error_category": "network_error"}])
    assert circuits.open_providers() == ["primary"]
    circuits.record_symbol_result([{"source_name": "primary", "status": "failed", "error_category": "network_error"}])
    assert len(circuits.history) == 1
    assert circuits.source_order_for_task(["primary"]) == []
    now[0] = 121
    assert circuits.source_order_for_task(["primary"]) == ["primary"]
    assert circuits.source_order_for_task(["primary"]) == []
    circuits.record_symbol_result([{"source_name": "primary", "status": "success", "error_category": ""}])
    assert circuits.open_providers() == []


def test_parallel_fetch_preserves_result_order_and_fresh_semantics(monkeypatch):
    active = 0
    peak = 0
    lock = threading.Lock()

    def fake_fetch(symbol, name, **kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        return FetchResult(
            symbol, name, indicator_frame(), False,
            final_source="fresh", effective_latest_date="2026-07-13",
            expected_trade_date="2026-07-13", is_expected_trade_date=True,
            provider_success=True, data_source_name="primary", provider_request_count=1,
            source_attempts=[{"source_name": "primary", "status": "success", "error_category": "", "latency_seconds": 0.01}],
        )

    monkeypatch.setattr(data_fetcher, "fetch_stock_daily", fake_fetch)
    pool = [{"symbol": f"600{i:03d}", "name": str(i)} for i in range(8)]
    settings = {
        "provider_parallel": {
            "enabled": True, "initial_workers": 4, "evaluation_window_requests": 20,
            "request_jitter_seconds": [0, 0], "adjustment_freeze_windows": 1,
            "circuit_breaker": {"enabled": True, "failure_threshold": 5, "cooldown_seconds": 120},
        },
        "data_fetch": {"network_timeout_seconds": 1, "consecutive_failure_hard_limit": 0},
        "data_sources": {"stock_daily": {"primary": "primary", "fallback": ["local_cache"]}},
    }
    results = fetch_stocks(pool, settings=settings, expected_trade_date_value="2026-07-13", sleep_fn=lambda _: None)
    assert [item.symbol for item in results] == [item["symbol"] for item in pool]
    assert peak > 1
    assert all(item.final_source == "fresh" for item in results)
    stats = summarize_stock_fetches(results)
    assert stats["fresh_rate"] == 1.0
    assert stats["provider_parallel_enabled"] is True
    assert stats["provider_parallel_efficiency"] > 1


def test_v3_rule_freeze_contract_is_explicit():
    settings = load_config()["settings"]
    v3 = settings["v3"]
    assert v3["rule_freeze_status"] == "frozen"
    assert v3["rule_freeze_trade_days"] == 30
    assert v3["weights"] == {"stock_factor": 0.50, "timing": 0.30, "market": 0.20}
    assert v3["buy_threshold"] == 75
    assert v3["watch_threshold"] == 60
    assert v3["structure_adjustment"] == {"breakout": 5, "pullback": 3, "acceleration": 0, "distribution": -10}
