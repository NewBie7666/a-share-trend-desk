from __future__ import annotations

import pandas as pd

from src.data_fetcher import FetchResult
from src.market_state import evaluate_market_state_from_snapshots
from src.signals import generate_buy_signals_from_snapshots
from src.snapshots import StockSnapshot, build_stock_snapshots, snapshot_data_issue_list
from src.styles import build_style_state_table_from_snapshots


def frame(rows: int = 130, close: float = 10.0) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=rows, freq="B")
    data = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "open": close * 0.99,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": 1000000,
            "amount": 500000000,
            "ma5": close * 0.99,
            "ma20": close * 0.95,
            "ma60": close * 0.90,
            "ma20_slope": 0.01,
            "macd_dif": 1.0,
            "macd_dea": 0.5,
            "rsi14": 60,
            "avg_amount_20d": 500000000,
            "return_5d": 0.03,
            "return_10d": 0.05,
            "ma20_deviation": 0.04,
            "relative_strength_20d": 0.05,
        }
    )
    return data


def fetch(symbol: str = "600030", source: str = "fresh", data: pd.DataFrame | None = None) -> FetchResult:
    df = frame() if data is None and source != "failed" else data if data is not None else pd.DataFrame()
    return FetchResult(
        symbol=symbol,
        name="中信证券" if symbol == "600030" else "测试股",
        data=df,
        used_cache=source == "cache",
        final_source=source,
        latest_date="2026-07-01" if source != "failed" else None,
        effective_latest_date="2026-07-01" if source != "failed" else None,
        expected_trade_date="2026-07-01",
        is_expected_trade_date=source != "failed",
        error="网络失败" if source == "failed" else None,
        error_category="network_error" if source == "failed" else "",
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
            "balanced": {"max_total_position": 0.50, "max_single_position": 0.25, "max_new_candidates": 2, "min_score": 75, "max_account_risk_pct": 0.025}
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
        "market_strength": 60,
        "style_state_table": [{"style": "finance_brokerage", "state": "strong"}],
        "portfolio_mode": {
            "portfolio_mode": "balanced",
            "strongest_styles": "finance_brokerage",
            "potential_strong_styles": "无",
        },
    }


def test_snapshot_builder_computes_indicators_once_per_usable_stock():
    calls = []

    def fake_indicator(df, hs300=None):
        calls.append(len(df))
        return df.copy()

    def fake_filter(symbol, name, df, settings_arg, rules_arg):
        return []

    def fake_score(symbol, name, df):
        return {"symbol": symbol, "name": name, "score": 90, "final_score": 90, "best_style": "finance_brokerage", "style_state": "strong"}

    snapshots, stats = build_stock_snapshots(
        [fetch("600030"), fetch("600001", "failed")],
        {"600030": "中信证券", "600001": "问题股"},
        settings(),
        RULES,
        frame(),
        indicator_fn=fake_indicator,
        filter_fn=fake_filter,
        scorer_fn=fake_score,
    )

    assert calls == [130]
    assert stats["fetch_count"] == 2
    assert stats["indicator_compute_count"] == 1
    assert stats["snapshot_count"] == 2
    assert stats["data_issue_count"] == 1
    assert snapshots["600030"].fetch_result.final_source == "fresh"
    assert snapshots["600030"].score["best_style"] == "finance_brokerage"
    assert snapshots["600001"].data_issue["error_category"] == "network_error"


def test_style_and_signal_stages_do_not_fetch(monkeypatch):
    def fail_fetch(*args, **kwargs):
        raise AssertionError("snapshot pipeline stage must not fetch")

    monkeypatch.setattr("src.data_fetcher.fetch_stock_daily", fail_fetch)
    monkeypatch.setattr("src.data_fetcher.fetch_stocks", fail_fetch)
    monkeypatch.setattr("src.signals.score_stock", fail_fetch)
    snapshot = StockSnapshot(
        symbol="600030",
        name="中信证券",
        fetch_result=fetch("600030"),
        price_data=frame(),
        indicators=frame(),
        filter_reasons=[],
        score={
            "symbol": "600030",
            "name": "中信证券",
            "score": 92,
            "final_score": 92,
            "best_style": "finance_brokerage",
            "style_state": "strong",
            "style_scores": {},
            "close_price": 10,
            "ma20": 9.5,
            "ma60": 9.0,
            "reason": "券商风格强势",
            "risk_note": "券商股需关注指数风险偏好",
        },
        style="finance_brokerage",
        style_state="strong",
    )
    snapshots = {"600030": snapshot}

    style_rows = build_style_state_table_from_snapshots(snapshots, {"600030"})
    local_settings = settings()
    local_settings["position_rules"]["balanced"]["max_account_risk_pct"] = 0.03
    signals = generate_buy_signals_from_snapshots(market(), snapshots, local_settings, RULES)

    assert style_rows[0]["style"] == "finance_brokerage"
    assert signals["candidates"]
    candidate = signals["candidates"][0]
    assert candidate["candidate_data_source"] == "fresh"
    assert candidate["candidate_latest_date"] == "2026-07-01"
    assert candidate["is_expected_trade_date"] is True


def test_market_state_from_snapshots_reuses_snapshot_data(monkeypatch):
    monkeypatch.setattr("src.data_fetcher.fetch_stock_daily", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no io")))
    snapshot = StockSnapshot(
        symbol="600030",
        name="中信证券",
        fetch_result=fetch("600030"),
        price_data=frame(),
        indicators=frame(),
        filter_reasons=[],
        score={"best_style": "finance_brokerage", "style_state": "strong"},
        style="finance_brokerage",
        style_state="strong",
    )
    result = evaluate_market_state_from_snapshots(
        frame(),
        {"600030": snapshot},
        RULES,
        {"600030"},
        market_index_table=[
            {"index_code": "000300", "state": "neutral", "close_above_MA60": True},
            {"index_code": "000001", "state": "neutral", "close_above_MA60": True},
            {"index_code": "399001", "state": "neutral", "close_above_MA60": True},
        ],
        settings=settings(),
    )

    assert result["market_state"] == "neutral"
    assert result["style_state_table"][0]["style"] == "finance_brokerage"


def test_failed_snapshot_enters_data_issue_list_without_extra_fetch():
    snapshots, _ = build_stock_snapshots(
        [fetch("600001", "failed")],
        {"600001": "问题股"},
        settings(),
        RULES,
        frame(),
    )

    issues = snapshot_data_issue_list(snapshots)
    assert issues == [
        {
            "symbol": "600001",
            "name": "测试股",
            "failed_reason": "网络失败",
            "error_category": "network_error",
            "raw_latest_date": "",
            "effective_latest_date": "",
            "data_source_name": "",
            "fallback_source_used": "",
        }
    ]
