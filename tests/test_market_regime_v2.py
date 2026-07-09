from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from src.market_regime import evaluate_market_regime
from src.signals import generate_buy_signals
from src.trading_decision import make_final_trade_decision


RULES = {
    "condition_order": {
        "trigger_close_multiple": 1.01,
        "trigger_ma20_multiple": 1.005,
        "take_profit_reduce_multiple": 1.25,
        "take_profit_strong_multiple": 1.40,
    },
}


def settings() -> dict:
    return {
        "initial_cash": 30000,
        "lot_size": 100,
        "max_same_style_candidates": 1,
        "minimum_trade_guarantee": {"min_candidates": 1},
        "data_integrity_report": {"data_integrity_score": 1.0, "confidence_position_multiplier": 1.0},
        "data_integrity": {"signal_score_weight": 0.70, "market_strength_weight": 0.20, "score_weight": 0.10},
        "position_rules": {
            "balanced": {"max_total_position": 0.50, "max_single_position": 0.25, "max_new_candidates": 2, "min_score": 70, "max_account_risk_pct": 0.05},
            "cash": {"max_total_position": 0.0, "max_single_position": 0.0, "max_new_candidates": 0, "min_score": 999, "max_account_risk_pct": 0.0},
        },
    }


def index_row(code: str, *, close: float = 100, ma20: float = 95, ma60: float = 90, state: str = "strong") -> dict:
    return {
        "index_code": code,
        "index_name": code,
        "close": close,
        "MA20": ma20,
        "MA60": ma60,
        "close_above_MA20": close > ma20,
        "close_above_MA60": close > ma60,
        "state": state,
    }


def frame(close: float = 20, *, ma20: float = 19, ma60: float = 18, amount: float = 600_000_000, avg_amount: float = 500_000_000) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2026-06-30",
                "open": close - 0.5,
                "high": close + 0.5,
                "low": close - 1,
                "close": close - 0.5,
                "volume": 1_000_000,
                "amount": amount,
                "avg_amount_20d": avg_amount,
                "ma5": close - 0.2,
                "ma20": ma20,
                "ma60": ma60,
                "ma20_slope": 0.01,
                "macd_dif": 2,
                "macd_dea": 1,
                "rsi14": 60,
                "relative_strength_20d": 0.05,
            },
            {
                "date": "2026-07-01",
                "open": close - 0.2,
                "high": close + 0.8,
                "low": close - 0.6,
                "close": close,
                "volume": 1_100_000,
                "amount": amount,
                "avg_amount_20d": avg_amount,
                "ma5": close - 0.2,
                "ma20": ma20,
                "ma60": ma60,
                "ma20_slope": 0.01,
                "macd_dif": 2,
                "macd_dea": 1,
                "rsi14": 60,
                "relative_strength_20d": 0.05,
            },
        ]
    )


def snapshot(close: float = 20, *, ma20: float = 19, ma60: float = 18, amount: float = 600_000_000):
    return SimpleNamespace(indicators=frame(close, ma20=ma20, ma60=ma60, amount=amount), usable_for_signal=True)


def market(regime: str = "normal") -> dict:
    return {
        "state": "yellow",
        "market_state": "neutral",
        "market_regime": regime,
        "market_score": 65,
        "market_strength": 60,
        "style_state_table": [{"style": "finance_brokerage", "state": "strong", "sample_size": 5}],
        "portfolio_mode": {"portfolio_mode": "balanced", "strongest_styles": "finance_brokerage", "potential_strong_styles": "无"},
    }


def test_market_regime_attack_and_cash():
    strong_indices = [index_row("000300"), index_row("000001"), index_row("399001")]
    strong_pool = {str(i): snapshot() for i in range(10)}
    attack = evaluate_market_regime(strong_indices, strong_pool)
    assert attack["market_regime"] == "attack"
    assert attack["market_score"] >= 80

    weak_indices = [
        index_row("000300", close=80, ma20=90, ma60=95, state="weak"),
        index_row("000001", close=80, ma20=90, ma60=95, state="weak"),
        index_row("399001", close=80, ma20=90, ma60=95, state="weak"),
    ]
    weak_pool = {str(i): snapshot(17, ma20=19, ma60=18, amount=200_000_000) for i in range(10)}
    cash = evaluate_market_regime(weak_indices, weak_pool)
    assert cash["market_regime"] == "cash"


def test_market_regime_has_no_io(monkeypatch):
    monkeypatch.setattr("src.data_fetcher.fetch_stocks", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("market regime must not fetch")))
    result = evaluate_market_regime([index_row("000300"), index_row("000001"), index_row("399001")], {"600030": snapshot()})
    assert result["market_regime"] in {"attack", "normal", "defensive", "cash"}


def test_market_regime_normal_and_defensive():
    normal_indices = [
        index_row("000300", close=100, ma20=105, ma60=90, state="neutral"),
        index_row("000001", close=100, ma20=95, ma60=90, state="strong"),
        index_row("399001", close=100, ma20=105, ma60=90, state="neutral"),
    ]
    normal = evaluate_market_regime(normal_indices, {str(i): snapshot() for i in range(10)})
    assert normal["market_regime"] in {"normal", "attack"}

    defensive_pool = {str(i): snapshot(20 if i < 4 else 17, ma20=19, ma60=18, amount=350_000_000) for i in range(10)}
    defensive = evaluate_market_regime(normal_indices, defensive_pool)
    assert defensive["market_regime"] in {"defensive", "normal"}


def test_final_trade_decision_priority():
    assert make_final_trade_decision("cash", {"timing_decision": "BUY"})["final_action"] == "AVOID"
    attack = make_final_trade_decision("attack", {"timing_decision": "BUY", "timing_reason": "可以买"})
    assert attack["final_action"] == "BUY"
    assert attack["position_multiplier"] == 1.0
    defensive = make_final_trade_decision("defensive", {"timing_decision": "BUY"})
    assert defensive["final_action"] == "BUY"
    assert defensive["position_multiplier"] == 0.4
    wait = make_final_trade_decision("normal", {"timing_decision": "WAIT", "timing_reason": "等待确认"})
    assert wait["final_action"] == "WAIT"
    avoid = make_final_trade_decision("attack", {"timing_decision": "AVOID", "timing_reason": "破位"})
    assert avoid["final_action"] == "AVOID"


def test_generate_signals_requires_final_action_buy():
    timing = {"600030": {"timing_decision": "BUY", "structure_state": "breakout", "entry_quality_score": 82, "decision_confidence": 0.85, "timing_reason": "可以买"}}
    blocked = generate_buy_signals(market("cash"), {"600030": frame()}, {"600030": "中信证券"}, {"600030": []}, settings(), RULES, timing_by_symbol=timing)
    assert blocked["candidates"] == []
    assert blocked["timing_avoid_list"][0]["final_action"] == "AVOID"

    normal = generate_buy_signals(market("normal"), {"600030": frame()}, {"600030": "中信证券"}, {"600030": []}, settings(), RULES, timing_by_symbol=timing)
    assert normal["candidates"][0]["final_action"] == "BUY"
    assert normal["candidates"][0]["position_multiplier"] == 0.7
    assert normal["candidates"][0]["estimated_amount"] <= 30000 * 0.25 * 0.7

    wait_timing = {"600030": {"timing_decision": "WAIT", "structure_state": "pullback", "entry_quality_score": 68, "decision_confidence": 0.65, "timing_reason": "等待确认"}}
    waiting = generate_buy_signals(market("normal"), {"600030": frame()}, {"600030": "中信证券"}, {"600030": []}, settings(), RULES, timing_by_symbol=wait_timing)
    assert waiting["candidates"] == []
    assert waiting["watchlist"][0]["final_action"] == "WAIT"
