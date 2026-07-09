from __future__ import annotations

import pandas as pd

from src.data_fetcher import FetchResult
from src.signals import generate_buy_signals_from_snapshots
from src.snapshots import StockSnapshot, build_stock_snapshots
from src.timing import (
    calculate_decision_confidence,
    detect_structure_state,
    evaluate_timing,
    final_decision,
)


def frame(
    *,
    close: float = 20,
    ma5: float = 19.8,
    ma20: float = 19,
    ma60: float = 18,
    ma20_slope: float = 0.01,
    high: float | None = None,
    open_price: float | None = None,
    amount: float = 600_000_000,
    avg_amount: float = 500_000_000,
    return_5d: float = 0.03,
    return_10d: float = 0.06,
    rs: float = 0.04,
    k: float = 60,
    d: float = 50,
    j: float = 70,
    macd_bull: bool = True,
    rows: int = 130,
) -> pd.DataFrame:
    high = high if high is not None else close * 1.01
    open_price = open_price if open_price is not None else close * 0.99
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=rows, freq="B").strftime("%Y-%m-%d"),
            "open": open_price,
            "high": high,
            "low": close * 0.97,
            "close": close,
            "volume": 1_000_000,
            "amount": amount,
            "ma5": ma5,
            "ma20": ma20,
            "ma60": ma60,
            "ma20_slope": ma20_slope,
            "macd_dif": 1.0 if macd_bull else 0.2,
            "macd_dea": 0.5,
            "rsi14": 60,
            "kdj_k": k,
            "kdj_d": d,
            "kdj_j": j,
            "avg_amount_20d": avg_amount,
            "return_5d": return_5d,
            "return_10d": return_10d,
            "return_20d": 0.08,
            "ma20_deviation": (close - ma20) / ma20 if ma20 else 0,
            "relative_strength_20d": rs,
        }
    )


def fetch_result(symbol: str, df: pd.DataFrame) -> FetchResult:
    return FetchResult(
        symbol=symbol,
        name="测试股",
        data=df,
        used_cache=False,
        final_source="fresh",
        latest_date="2026-07-01",
        effective_latest_date="2026-07-01",
        expected_trade_date="2026-07-01",
        is_expected_trade_date=True,
    )


def settings() -> dict:
    return {
        "initial_cash": 30000,
        "lot_size": 100,
        "min_avg_amount_20d": 300000000,
        "max_same_style_candidates": 1,
        "stop_loss_half": 0.08,
        "stop_loss_full": 0.12,
        "take_profit_watch": 0.25,
        "take_profit_strong": 0.40,
        "data_integrity_report": {"data_integrity_score": 1.0, "confidence_position_multiplier": 1.0},
        "data_integrity": {"signal_score_weight": 0.70, "market_strength_weight": 0.20, "score_weight": 0.10},
        "minimum_trade_guarantee": {"min_candidates": 1},
        "position_rules": {
            "balanced": {"max_total_position": 0.50, "max_single_position": 0.25, "max_new_candidates": 2, "min_score": 70, "max_account_risk_pct": 0.05}
        },
    }


RULES = {
    "filters": {"min_history_days": 120},
    "condition_order": {
        "trigger_close_multiple": 1.01,
        "trigger_ma20_multiple": 1.005,
        "take_profit_reduce_multiple": 1.25,
        "take_profit_strong_multiple": 1.40,
    },
}


def market() -> dict:
    return {
        "state": "yellow",
        "market_state": "neutral",
        "market_regime": "normal",
        "market_strength": 60,
        "style_state_table": [{"style": "finance_brokerage", "state": "strong"}],
        "portfolio_mode": {"portfolio_mode": "balanced", "strongest_styles": "finance_brokerage", "potential_strong_styles": "无"},
    }


def snapshot(symbol: str, timing: dict) -> StockSnapshot:
    score = {
        "symbol": symbol,
        "name": "中信证券",
        "score": 92,
        "final_score": 92,
        "best_style": "finance_brokerage",
        "style_state": "strong",
        "style_scores": {},
        "close_price": 20,
        "ma20": 19,
        "ma60": 18,
        "reason": "券商风格较强",
        "risk_note": "注意指数风险偏好变化",
    }
    return StockSnapshot(
        symbol=symbol,
        name="中信证券",
        fetch_result=fetch_result(symbol, frame()),
        price_data=frame(),
        indicators=frame(),
        filter_reasons=[],
        score=score,
        style="finance_brokerage",
        style_state="strong",
        timing=timing,
    )


def test_detect_structure_states():
    assert detect_structure_state(frame(close=17, ma20=19, ma60=18))["structure_state"] == "breakdown"
    assert detect_structure_state(frame(close=20, ma5=19.5, ma20=19, ma60=18, high=22, open_price=20.1, k=40, d=55, macd_bull=False, amount=900_000_000))["structure_state"] == "distribution"
    assert detect_structure_state(frame(close=22, ma5=21, ma20=19, ma60=18, return_5d=0.08, return_10d=0.12))["structure_state"] == "acceleration"
    assert detect_structure_state(frame(close=20, ma5=20.1, ma20=19.6, ma60=18, high=22, return_5d=0.02))["structure_state"] == "pullback"
    assert detect_structure_state(frame(close=20.5, ma5=20, ma20=19, ma60=18))["structure_state"] == "breakout"


def test_decision_confidence_examples():
    assert calculate_decision_confidence("breakout", 80, True)["decision_confidence"] == 0.9
    assert calculate_decision_confidence("pullback", 65, False)["decision_confidence"] == 0.65
    assert calculate_decision_confidence("distribution", 80, True)["decision_confidence"] == 0.55
    assert calculate_decision_confidence("breakdown", 90, False)["decision_confidence"] == 0.2


def test_final_decision_priority_rules():
    assert final_decision("breakdown", 95, 0.95, True)["timing_decision"] == "AVOID"
    protected = final_decision("distribution", 80, 0.55, True)
    assert protected["timing_decision"] == "WAIT"
    assert protected["timing_risk_tag"] == "high_risk_distribution_in_strong_trend"
    assert final_decision("distribution", 80, 0.50, False)["timing_decision"] == "AVOID"
    assert final_decision("breakout", 59, 0.85, True)["timing_decision"] == "AVOID"
    assert final_decision("breakout", 70, 0.72, True)["timing_decision"] == "WAIT"
    assert final_decision("breakout", 80, 0.85, True)["timing_decision"] == "BUY"


def test_snapshot_build_computes_timing_once():
    snapshots, _ = build_stock_snapshots(
        [fetch_result("600030", frame())],
        {"600030": "中信证券"},
        settings(),
        RULES,
        frame(),
        indicator_fn=lambda df, hs300=None: frame(),
        scorer_fn=lambda symbol, name, df: {
            "symbol": symbol,
            "name": name,
            "score": 90,
            "final_score": 90,
            "best_style": "finance_brokerage",
            "style_state": "strong",
            "style_scores": {},
            "close_price": 20,
            "ma20": 19,
            "ma60": 18,
            "reason": "券商风格较强",
            "risk_note": "注意指数风险偏好变化",
        },
    )
    timing = snapshots["600030"].timing
    assert timing["structure_state"] in {"breakout", "acceleration", "pullback"}
    assert timing["timing_decision"] in {"BUY", "WAIT", "AVOID"}
    assert "entry_quality_score" in timing


def test_signal_integration_buy_wait_avoid_from_snapshot_timing(monkeypatch):
    monkeypatch.setattr("src.signals.score_stock", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must reuse snapshot score")))
    monkeypatch.setattr("src.timing.evaluate_timing", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not recalculate timing in signal stage")))

    buy = snapshot("600030", {"timing_decision": "BUY", "structure_state": "breakout", "entry_quality_score": 82, "decision_confidence": 0.85, "timing_reason": "可以买", "trend_strong": True, "timing_risk_tag": ""})
    wait = snapshot("601211", {"timing_decision": "WAIT", "structure_state": "pullback", "entry_quality_score": 68, "decision_confidence": 0.65, "timing_reason": "等待确认", "trend_strong": True, "timing_risk_tag": ""})
    avoid = snapshot("601688", {"timing_decision": "AVOID", "structure_state": "breakdown", "entry_quality_score": 40, "decision_confidence": 0.05, "timing_reason": "破位规避", "trend_strong": False, "timing_risk_tag": ""})

    signals = generate_buy_signals_from_snapshots(market(), {"600030": buy, "601211": wait, "601688": avoid}, settings(), RULES)

    assert [row["symbol"] for row in signals["candidates"]] == ["600030"]
    assert signals["candidates"][0]["timing_decision"] == "BUY"
    assert any(row["symbol"] == "601211" and row["timing_decision"] == "WAIT" for row in signals["watchlist"])
    assert all(row["symbol"] != "601688" for row in signals["candidates"])
    assert all(row["symbol"] != "601688" for row in signals["watchlist"])
    assert signals["timing_avoid_list"][0]["symbol"] == "601688"
