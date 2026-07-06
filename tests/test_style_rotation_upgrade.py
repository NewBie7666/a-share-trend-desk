from __future__ import annotations

import pandas as pd

from src.signals import generate_buy_signals
from src.styles import build_style_state_table, identify_best_style


def frame(
    *,
    close: float = 10,
    ma5: float = 9.8,
    ma20: float = 9.5,
    ma60: float = 9,
    return_5d: float = 0.04,
    return_10d: float = 0.08,
    rs: float = 0.05,
    macd_bull: bool = True,
) -> pd.DataFrame:
    rows = 130
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=rows, freq="B").strftime("%Y-%m-%d"),
            "open": close * 0.99,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": 1000000,
            "amount": 500000000,
            "ma5": ma5,
            "ma20": ma20,
            "ma60": ma60,
            "ma20_slope": 0.01,
            "macd_dif": 1.0 if macd_bull else 0.2,
            "macd_dea": 0.5,
            "rsi14": 60,
            "avg_amount_20d": 500000000,
            "return_5d": return_5d,
            "return_10d": return_10d,
            "ma20_deviation": 0.04,
            "relative_strength_20d": rs,
        }
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
    "condition_order": {
        "trigger_close_multiple": 1.01,
        "trigger_ma20_multiple": 1.005,
        "take_profit_reduce_multiple": 1.25,
        "take_profit_strong_multiple": 1.40,
    }
}


def market(style_row: dict, strongest: str = "medicine_innovation_drug") -> dict:
    return {
        "state": "yellow",
        "market_state": "neutral",
        "market_strength": 60,
        "style_state_table": [style_row],
        "portfolio_mode": {"portfolio_mode": "balanced", "strongest_styles": strongest, "potential_strong_styles": "无"},
    }


def test_hengrui_maps_to_innovation_drug_style():
    assert identify_best_style("600276", "恒瑞医药") == "medicine_innovation_drug"


def test_style_table_marks_rotation_bonus_for_broad_innovation_drug_strength():
    symbols = ["600276", "688235", "688180", "300558", "688331"]
    names = {symbol: "恒瑞医药" if symbol == "600276" else "创新药样本" for symbol in symbols}
    frames = {symbol: frame() for symbol in symbols}

    rows = build_style_state_table(frames, names, set(symbols))
    row = rows[0]

    assert row["style"] == "medicine_innovation_drug"
    assert row["style_rotation_bonus"] > 0
    assert row["breadth_score"] >= 90
    assert row["retreat_risk"] is False


def test_semiconductor_retreat_penalty_reduces_candidate_signal_score():
    stock_frames = {"600703": frame(close=10, ma5=10.8, ma20=9.5, ma60=9, return_5d=-0.06, macd_bull=False)}
    names = {"600703": "三安光电"}
    style_row = {
        "style": "semiconductor",
        "state": "strong",
        "style_momentum_score": 45,
        "breadth_score": 40,
        "style_rotation_bonus": 0,
        "style_retreat_penalty": 15,
        "retreat_risk": True,
    }

    signals = generate_buy_signals(market(style_row, "semiconductor"), stock_frames, names, {"600703": []}, settings(), RULES)
    row = (signals["candidates"] or signals["watchlist"])[0]

    assert row["best_style"] == "semiconductor"
    assert row["style_retreat_penalty"] == 15
    assert row["signal_score"] < row["base_signal_score"]
    assert "退潮风险" in row["reason"]


def test_innovation_drug_rotation_bonus_increases_candidate_signal_score():
    stock_frames = {"600276": frame()}
    names = {"600276": "恒瑞医药"}
    style_row = {
        "style": "medicine_innovation_drug",
        "state": "strong",
        "style_momentum_score": 90,
        "breadth_score": 95,
        "style_rotation_bonus": 8,
        "style_retreat_penalty": 0,
        "retreat_risk": False,
    }

    signals = generate_buy_signals(market(style_row), stock_frames, names, {"600276": []}, settings(), RULES)
    row = (signals["candidates"] or signals["watchlist"])[0]

    assert row["best_style"] == "medicine_innovation_drug"
    assert row["style_rotation_bonus"] == 8
    assert row["signal_score"] > row["base_signal_score"]
    assert "扩散转强" in row["reason"]
