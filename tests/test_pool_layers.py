from types import SimpleNamespace

import pandas as pd

from src.pool_layers import pool_metadata, select_analysis_pool, select_candidate_pool, subset_snapshots


def _snapshot(symbol: str, amount: float, score: float):
    frame = pd.DataFrame([{"date": "2026-07-01", "close": 10, "amount": amount}])
    return SimpleNamespace(symbol=symbol, latest=frame.iloc[-1], usable_for_signal=True, score={"final_score": score})


def test_three_tier_pool_selection_preserves_industry_then_liquidity_and_score():
    pool = [
        {"symbol": "600001", "name": "A", "industry": "电子"},
        {"symbol": "600002", "name": "B", "industry": "医药"},
        {"symbol": "600003", "name": "C", "industry": "电子"},
        {"symbol": "600004", "name": "D", "industry": "未知行业"},
    ]
    snapshots = {
        "600001": _snapshot("600001", 100, 80),
        "600002": _snapshot("600002", 90, 90),
        "600003": _snapshot("600003", 80, 95),
        "600004": _snapshot("600004", 70, 70),
    }
    analysis = select_analysis_pool(pool, snapshots, limit=3)
    assert {"600001", "600002"}.issubset(analysis)
    candidates = select_candidate_pool(subset_snapshots(snapshots, analysis), limit=2)
    assert candidates == ["600003", "600002"]
    metadata = pool_metadata(pool, version="v2.1", source="测试来源")
    assert metadata["pool_size"] == 4
    assert metadata["industry_distribution"]["电子"] == 2
