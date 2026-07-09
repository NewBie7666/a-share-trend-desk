from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from .data_fetcher import FetchResult
from .filters import evaluate_hard_filters
from .indicators import add_indicators
from .scoring import score_stock
from .timing import evaluate_timing


@dataclass
class StockSnapshot:
    symbol: str
    name: str
    fetch_result: FetchResult
    price_data: pd.DataFrame
    indicators: pd.DataFrame
    filter_reasons: list[str] = field(default_factory=list)
    score: dict | None = None
    style: str = "other"
    style_state: str = ""
    risk_state: dict = field(default_factory=dict)
    timing: dict = field(default_factory=dict)
    data_issue: dict | None = None

    @property
    def latest(self) -> pd.Series | None:
        if self.indicators is None or self.indicators.empty:
            return None
        return self.indicators.iloc[-1]

    @property
    def usable_for_signal(self) -> bool:
        return self.data_issue is None and self.indicators is not None and not self.indicators.empty


def fetch_result_is_usable(result: FetchResult) -> bool:
    blocked_categories = {
        "stale_cache",
        "no_expected_trade_date_data",
        "data_not_ready",
        "insufficient_history",
    }
    return (
        result.final_source in {"fresh", "cache"}
        and result.data is not None
        and not result.data.empty
        and result.error_category not in blocked_categories
    )


def data_issue_from_fetch_result(result: FetchResult) -> dict:
    return {
        "symbol": result.symbol,
        "name": result.name,
        "failed_reason": result.error or result.data_staleness_reason or "日线行情数据不可用于交易信号",
        "error_category": result.error_category or "unknown",
        "raw_latest_date": result.raw_latest_date or "",
        "effective_latest_date": result.effective_latest_date or result.latest_date or "",
        "data_source_name": result.data_source_name or "",
        "fallback_source_used": result.fallback_source_used or "",
    }


def build_stock_snapshots(
    stock_results: list[FetchResult],
    names: dict[str, str],
    settings: dict,
    risk_rules: dict,
    hs300_indicators: pd.DataFrame | None = None,
    *,
    indicator_fn: Callable[[pd.DataFrame, pd.DataFrame | None], pd.DataFrame] = add_indicators,
    filter_fn: Callable[[str, str, pd.DataFrame, dict, dict], list[str]] = evaluate_hard_filters,
    scorer_fn: Callable[[str, str, pd.DataFrame], dict] = score_stock,
) -> tuple[dict[str, StockSnapshot], dict]:
    snapshots: dict[str, StockSnapshot] = {}
    indicator_compute_count = 0
    valid_snapshot_count = 0
    data_issue_count = 0
    filter_rules = risk_rules.get("filters", risk_rules)

    for result in stock_results:
        symbol = str(result.symbol).zfill(6)
        name = result.name or names.get(symbol, "")
        price_data = result.data

        if not fetch_result_is_usable(result):
            data_issue = data_issue_from_fetch_result(result)
            snapshots[symbol] = StockSnapshot(
                symbol=symbol,
                name=name,
                fetch_result=result,
                price_data=price_data,
                indicators=pd.DataFrame(),
                filter_reasons=[],
                data_issue=data_issue,
            )
            data_issue_count += 1
            continue

        indicators = indicator_fn(price_data, hs300_indicators)
        indicator_compute_count += 1
        filter_reasons = filter_fn(symbol, name, indicators, settings, filter_rules)
        score = scorer_fn(symbol, name, indicators) if not indicators.empty else None
        timing = evaluate_timing(indicators) if not indicators.empty else {}
        style = (score or {}).get("best_style", "other")
        style_state = (score or {}).get("style_state", "")
        snapshots[symbol] = StockSnapshot(
            symbol=symbol,
            name=name,
            fetch_result=result,
            price_data=price_data,
            indicators=indicators,
            filter_reasons=filter_reasons,
            score=score,
            style=style,
            style_state=style_state,
            risk_state={"hard_filter_pass": not bool(filter_reasons)},
            timing=timing,
            data_issue=None,
        )
        valid_snapshot_count += 1

    stats = {
        "fetch_count": len(stock_results),
        "indicator_compute_count": indicator_compute_count,
        "snapshot_count": len(snapshots),
        "valid_snapshot_count": valid_snapshot_count,
        "data_issue_count": data_issue_count,
        "no_duplicate_io": True,
    }
    return snapshots, stats


def snapshots_to_frames(snapshots: dict[str, StockSnapshot], *, usable_only: bool = True) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for symbol, snapshot in snapshots.items():
        if usable_only and not snapshot.usable_for_signal:
            continue
        if snapshot.indicators is not None and not snapshot.indicators.empty:
            frames[symbol] = snapshot.indicators
    return frames


def snapshot_names(snapshots: dict[str, StockSnapshot]) -> dict[str, str]:
    return {symbol: snapshot.name for symbol, snapshot in snapshots.items()}


def snapshot_filter_reasons(snapshots: dict[str, StockSnapshot]) -> dict[str, list[str]]:
    return {
        symbol: list(snapshot.filter_reasons)
        for symbol, snapshot in snapshots.items()
        if snapshot.usable_for_signal
    }


def snapshot_fetch_results(snapshots: dict[str, StockSnapshot]) -> dict[str, FetchResult]:
    return {symbol: snapshot.fetch_result for symbol, snapshot in snapshots.items()}


def snapshot_data_issue_list(snapshots: dict[str, StockSnapshot]) -> list[dict]:
    return [
        snapshot.data_issue
        for snapshot in snapshots.values()
        if snapshot.data_issue is not None
    ]
